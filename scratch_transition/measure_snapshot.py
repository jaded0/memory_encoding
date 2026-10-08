"""Per-snapshot measurements for the key-recall plateau/onset study.

Run from the code dir on Deckard:
    CUDA_VISIBLE_DEVICES=0 python scratch_transition/measure_snapshot.py --checkpoint CKPT --init INIT_CKPT --out OUT.json [--batches 64]

Held-out episodes (validation split) run step by step as heldout.py's "observed" protocol
(wipe, slow frozen, fast entries written with the model's own dfa_layer_step). At every row's
answer ('!') step we additionally rebuild the trunk by hand from the live per-sample weights
(verified equal to the model's logits), and by autograd get, per trunk layer l in {L0,L1,L2}:
  J_l  [B,V,d]  Jacobian of the logits w.r.t. layer l's pre-activation z_l
so the true backprop signal for any output error e is g_l = e J_l, to be compared with the DFA
signal p_l = e B_l. Reported:
  (a) alignment: Frobenius cos theta_l = <J_l,B_l>/(|J||B|); cos(g_q,p_q); cos(g_q,p_s*);
      first-order loss change of the store-step write read at the answer step, g_q . term_s*;
      fraction of episodes where it is negative; toy prediction Phi(sqrt(d) cot theta);
      min eigenvalue of sym(J B^T) on 1-perp; fraction of e with e^T J B^T e > 0.
  (b) key geometry, (c) causal ablations (recall), (d) F norms and fast fraction of pre-activation,
  (e) slow weight change relative to the init checkpoint.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from ephemeral_model import dfa_output_error  # noqa: E402
from heldout import load_heldout_batches  # noqa: E402
from utils import initialize_charset, load_checkpoint, model_input, read_checkpoint, upgrade_legacy_config  # noqa: E402

LN = ["L0", "L1", "L2"]


def ms(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "se": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std()), "se": float(v.std() / math.sqrt(v.size)), "n": int(v.size)}


def pr(eigs):
    e = np.asarray(eigs, dtype=np.float64)
    return float(e.sum() ** 2 / (e ** 2).sum())


def load_model(path, device):
    from train import build_model, build_parser, positional_encoding
    checkpoint = read_checkpoint(path)
    defaults = {k: v for k, v in vars(build_parser().parse_args([])).items() if not k.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(checkpoint.get("config", {}))}
    charset, _, _, n_characters = initialize_charset(config["dataset"])
    model = build_model(config, charset, n_characters)
    model, _, next_iter, _, _ = load_checkpoint(path, model, config, device=device, checkpoint=checkpoint)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], device)
    model.to(device).eval()
    return model, config, charset, next_iter - 1


def layers_of(model):
    return [*model.linear_layers, model.i2h]


def slow_part(layer):
    W = layer.per_sample_weights.data[0]
    return torch.where(layer.ephemeral_mask, torch.zeros_like(W), W).double()


def ablation_forward(model, lay, rec, i, store, onehot, config):
    """Answer logits at step i under modified fast weights (all rows; use rows with query==i)."""
    B = onehot.shape[0]
    c = float(config["learning_rate"] * lay[0].fused_plasticity())
    rho = 1 - model.forget_rate
    s = torch.as_tensor(store, device=onehot.device).clamp(max=i - 1)
    ar = torch.arange(B, device=onehot.device)

    def sig_matrix(k):
        layer = lay[k]
        x = torch.stack(rec[k]["x"])[s, ar]
        p = torch.stack(rec[k]["p"])[s, ar]
        coef = -c * rho ** (i - s).to(x.dtype)
        T = (coef[:, None] * p).unsqueeze(2) * x.unsqueeze(1)
        return torch.where(layer.ephemeral_mask.unsqueeze(0), T, torch.zeros_like(T))

    x0 = model_input(onehot, i, config["input_mode"], config["pe_matrix"])
    full = {}
    for k in range(3):
        layer = lay[k]
        m = layer.ephemeral_mask.unsqueeze(0)
        W = layer.per_sample_weights.data
        slow = torch.where(m, torch.zeros_like(W), W)
        sig = sig_matrix(k)
        full[k] = {"full": W, "drop_signal": W - sig, "signal_only": slow + sig, "no_fast": slow}

    def logits(choice):
        h = torch.cat((x0, torch.zeros(B, model.hidden_size, device=x0.device)), 1)
        for k in range(3):
            h = torch.einsum("boi,bi->bo", full[k][choice[k]], h) + lay[k].bias
            h = torch.nn.functional.gelu(h)
        return torch.einsum("boi,bi->bo", model.i2o.per_sample_weights, torch.tanh(h) if model.output_tanh else h) + model.i2o.bias

    out = {"full": logits(["full"] * 3)}
    for mode in ("drop_signal", "signal_only", "no_fast"):
        for k in range(3):
            ch = ["full"] * 3
            ch[k] = mode
            out[f"{mode}@L{k}"] = logits(ch)
        out[f"{mode}@all"] = logits([mode] * 3)
    return out


def jacobians(model, lay, x0, B):
    """Hand-built trunk with grad: returns logits and J_l [B,V,out] (d logits / d z_l) for l=0..2,
    the pre-activations z_l (detached) and gelu'(z) not needed (it is inside J)."""
    with torch.enable_grad():
        h = torch.cat((x0, torch.zeros(B, model.hidden_size, device=x0.device)), 1)
        zs = []
        for k in range(3):
            z = torch.einsum("boi,bi->bo", lay[k].per_sample_weights.data, h) + lay[k].bias.data
            zero = torch.zeros_like(z, requires_grad=True)  # d/d(zero) = d/dz
            z = z + zero
            zs.append(zero)
            h = torch.nn.functional.gelu(z)
        out = torch.einsum("boi,bi->bo", model.i2o.per_sample_weights.data, torch.tanh(h) if model.output_tanh else h) + model.i2o.bias.data
        V = out.shape[1]
        J = [[] for _ in range(3)]
        for v in range(V):
            gs = torch.autograd.grad(out[:, v].sum(), zs, retain_graph=True)
            for k in range(3):
                J[k].append(gs[k])
    return out.detach(), [torch.stack(j, 1).detach() for j in J], None


@torch.no_grad()
def run_batch(model, config, onehot, texts, device, probe):
    Bn, Tn = onehot.shape[0], onehot.shape[1]
    steps = Tn - 1
    lay = layers_of(model)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    cap = {}
    hooks = []
    for k, layer in enumerate(lay[:3]):
        def hook(mod, inp, out, k=k):
            x = inp[0]
            m = mod.ephemeral_mask
            fast = torch.bmm(torch.where(m.unsqueeze(0), mod.per_sample_weights, torch.zeros_like(mod.per_sample_weights)),
                             x.unsqueeze(2)).squeeze(2)
            cap[k] = (x.detach().clone(), out.detach().clone(), fast)
        hooks.append(layer.register_forward_hook(hook))
    store = np.array([t.index("?") for t in texts])
    query = np.array([t.index("!") for t in texts])
    model.start_sequence_wipe()
    hidden = model.initHidden(Bn)
    rec = {k: {"x": [], "p": [], "pre": [], "fast": [], "Fnorm": []} for k in range(3)}
    errs = []
    valid = []
    abl, grads = {}, {}
    c = float(config["learning_rate"] * lay[0].fused_plasticity())
    rho = 1 - model.forget_rate
    for i in range(steps):
        inp = model_input(onehot, i, config["input_mode"], config["pe_matrix"])
        if (query == i).any():
            # grads need the weights *before* this forward's hooks overwrite nothing; weights are unchanged by forward
            logits_h, J, zs = jacobians(model, lay, inp, Bn)
        output, hidden = model(inp, hidden)
        if (query == i).any():
            assert torch.allclose(logits_h, output, atol=1e-3, rtol=1e-3), float((logits_h - output).abs().max())
        loss, err = dfa_output_error(output, onehot[:, i + 1], criterion)
        errs.append(err)
        valid.append(onehot[:, i + 1].sum(1) > 0)
        for k in range(3):
            x, pre, fast = cap[k]
            rec[k]["x"].append(x)
            rec[k]["pre"].append(pre)
            rec[k]["fast"].append(fast)
            layer = lay[k]
            F = torch.where(layer.ephemeral_mask.unsqueeze(0), layer.per_sample_weights, torch.zeros_like(layer.per_sample_weights))
            rec[k]["Fnorm"].append(torch.linalg.vector_norm(F, dim=(1, 2)))
        if (query == i).any():
            abl[i] = ablation_forward(model, lay, rec, i, store, onehot, config)
            grads[i] = (J, err, loss)
        projected, _ = model.dfa_step_errors(err, 0)
        for k, (layer, e) in enumerate(zip(model.trained_layers(), projected)):
            if layer.is_last_layer or k >= 3:
                continue
            rec[k]["p"].append(e.clone())
            model._fast_entry_step(layer, e, config["learning_rate"], config["ephemeral_update_clamp"])
    for h in hooks:
        h.remove()
    return rec, errs, torch.stack(valid), store, query, abl, grads, c, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=64)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, charset, it = load_model(args.checkpoint, device)
    lay = layers_of(model)
    # (e) slow weight change relative to init
    init_model, _, _, init_it = load_model(args.init, device)
    ilay = layers_of(init_model)
    res = {"iteration": it, "init_iteration": init_it}
    slow = {}
    for k in range(3):
        W, W0 = slow_part(lay[k]), slow_part(ilay[k])
        D = W - W0
        sv = torch.linalg.svdvals(D)
        slow[LN[k]] = {"rel_change": float(D.norm() / W0.norm()), "norm": float(W.norm()), "norm_init": float(W0.norm()),
                       "dW_PR_sv2": pr((sv ** 2).cpu().numpy()), "dW_top1_energy": float(sv[0] ** 2 / (sv ** 2).sum()),
                       "bias_change": float((lay[k].bias.data - ilay[k].bias.data).double().norm())}
    Wo, Wo0 = model.i2o.per_sample_weights.data[0].double(), init_model.i2o.per_sample_weights.data[0].double()
    slow["i2o"] = {"rel_change": float((Wo - Wo0).norm() / Wo0.norm()), "norm": float(Wo.norm()), "norm_init": float(Wo0.norm())}
    res["slow"] = slow
    del init_model

    batches = load_heldout_batches(config["dataset"], config["batch_size"], args.batches, device, "validation")
    V = len(charset)
    res["episodes"] = len(batches) * config["batch_size"]
    Bfb = [lay[k].feedback_weights.double() for k in range(3)]  # [V,out]
    from scipy.stats import norm as _norm

    A = {k: {n: [] for n in ["cosF", "cos_gq_pq", "cos_gq_ps", "dL_store", "dL_store_unmasked", "dL_fast_total",
                             "cos_gq_negterm", "pred_phi", "min_eig", "self_desc", "cross_desc", "J_norm", "g_norm", "p_norm",
                             "ps_norm", "term_norm", "Fnorm", "Fq_over_pre", "cos_xs_xq"]}
         for k in range(3)}
    G = {k: {"X": [], "off": [], "csq": []} for k in range(3)}
    abl_logit, abl_true = {}, []
    full_loss, full_margin, nofast_loss = [], [], []
    for texts, onehot in batches:
        rec, errs, valid, store, query, abl, grads, c, rho = run_batch(model, config, onehot, texts, device, None)
        Bn = onehot.shape[0]
        for i in sorted(abl):
            rows = np.where(query == i)[0]
            ridx = torch.as_tensor(rows, device=device)
            tgt = onehot[ridx, i + 1].argmax(1)
            for name, lg in abl[i].items():
                abl_logit.setdefault(name, []).append(lg[ridx].cpu())
            abl_true.append(tgt.cpu())
            J, err, loss = grads[i]
            lg = abl[i]["full"][ridx]
            ce = torch.nn.functional.cross_entropy(lg, tgt, reduction="none")
            full_loss.append(ce.cpu())
            nofast_loss.append(torch.nn.functional.cross_entropy(abl[i]["no_fast@all"][ridx], tgt, reduction="none").cpu())
            full_margin.append(((lg.gather(1, tgt[:, None])[:, 0]) - lg.masked_fill(torch.nn.functional.one_hot(tgt, V).bool(), -1e9).max(1).values).cpu())
            for k in range(3):
                layer = lay[k]
                Jk = J[k][ridx].double()  # [b,V,out]
                Bk = Bfb[k]
                eq = err[ridx].double()  # [b,V]
                gq = torch.einsum("bv,bvo->bo", eq, Jk)
                pq = eq @ Bk
                s_idx = torch.as_tensor(store[rows], device=device)
                # store-step error and projected error
                es = torch.stack(errs)[s_idx, ridx].double()
                ps = torch.stack(rec[k]["p"])[s_idx, ridx].double()
                xs = torch.stack(rec[k]["x"])[s_idx, ridx].double()
                xq = torch.stack(rec[k]["x"])[i, ridx].double()
                m = layer.ephemeral_mask.double()
                coef = -c * rho ** (i - s_idx).double()
                mx = (xs * xq) @ m.T  # [b,out]
                term = coef[:, None] * ps * mx
                term_un = coef[:, None] * ps * (xs * xq).sum(1, keepdim=True)
                fast_q = torch.stack(rec[k]["fast"])[i, ridx].double()
                pre_q = torch.stack(rec[k]["pre"])[i, ridx].double()
                cosf = lambda a, b: ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-30))
                Jn, Bn_ = Jk.flatten(1).norm(dim=1), Bk.norm()
                cth = (Jk.flatten(1) * Bk.flatten()[None]).sum(1) / (Jn * Bn_).clamp_min(1e-30)
                M = torch.einsum("bvo,wo->bvw", Jk, Bk)  # J B^T [b,V,V]
                Hc = torch.eye(V, device=device, dtype=torch.float64) - 1.0 / V
                Q = torch.linalg.eigh(Hc)[1][:, 1:]  # orthonormal basis of 1-perp
                evp = torch.linalg.eigvalsh(Q.T @ (0.5 * (M + M.transpose(1, 2))) @ Q)
                d_out = Bk.shape[1]
                cot = cth / torch.sqrt((1 - cth ** 2).clamp_min(1e-12))
                phi = torch.as_tensor(_norm.cdf((math.sqrt(d_out) * cot).cpu().numpy()))
                a = A[k]
                a["cosF"].append(cth.cpu()); a["cos_gq_pq"].append(cosf(gq, pq).cpu()); a["cos_gq_ps"].append(cosf(gq, ps).cpu())
                a["dL_store"].append((gq * term).sum(1).cpu()); a["dL_store_unmasked"].append((gq * term_un).sum(1).cpu())
                a["dL_fast_total"].append((gq * fast_q).sum(1).cpu())
                a["cos_gq_negterm"].append(cosf(gq, -term).cpu()); a["pred_phi"].append(phi)
                a["min_eig"].append(evp[:, 0].cpu()); a["J_norm"].append(Jn.cpu())
                a["self_desc"].append(torch.einsum("bv,bvw,bw->b", eq, M, eq).cpu())
                a["cross_desc"].append(torch.einsum("bv,bvw,bw->b", eq, M, es).cpu())
                a["g_norm"].append(gq.norm(dim=1).cpu()); a["p_norm"].append(pq.norm(dim=1).cpu()); a["ps_norm"].append(ps.norm(dim=1).cpu())
                a["term_norm"].append(term.norm(dim=1).cpu())
                a["Fnorm"].append(torch.stack(rec[k]["Fnorm"])[i, ridx].double().cpu())
                a["Fq_over_pre"].append((fast_q.norm(dim=1) / pre_q.norm(dim=1).clamp_min(1e-30)).cpu())
                a["cos_xs_xq"].append(cosf(xs, xq).cpu())
        # key geometry (valid write steps), per sequence
        for k in range(3):
            Xs = torch.stack(rec[k]["x"]).double()
            for b in range(Bn):
                idx = np.where(valid[:, b].cpu().numpy())[0]
                Xv = Xs[idx, b]
                U = Xv / Xv.norm(dim=1, keepdim=True).clamp_min(1e-12)
                Gm = U @ U.T
                n = len(idx)
                G[k]["off"].append(float((Gm.sum() - Gm.diag().sum()) / (n * (n - 1))))
                G[k]["csq"].append(float(U[list(idx).index(store[b])] @ U[list(idx).index(query[b])]) if (store[b] in idx and query[b] in idx) else float("nan"))
                G[k]["X"].append(U.cpu())
    # ------------- aggregate
    true = torch.cat(abl_true)
    res["ablation"] = {n: {"recall": float((torch.cat(l).argmax(1) == true).float().mean())} for n, l in abl_logit.items()}
    extra = {}
    for n in ("full", "no_fast@all", "signal_only@all", "no_fast@L2", "no_fast@L1", "no_fast@L0"):
        lg = torch.cat(abl_logit[n]).double()
        oh = torch.nn.functional.one_hot(true, V).bool()
        marg = lg.gather(1, true[:, None])[:, 0] - lg.masked_fill(oh, -1e9).max(1).values
        pr_ = torch.softmax(lg, 1)
        ent = -(pr_ * pr_.clamp_min(1e-12).log()).sum(1)
        pm = pr_.mean(0)  # average predicted distribution
        extra[n] = {"margin": ms(marg.numpy()), "frac_margin_pos": float((marg > 0).float().mean()), "pmax": ms(pr_.max(1).values.numpy()),
                    "entropy": ms(ent.numpy()), "mean_pred_entropy": float(-(pm * pm.clamp_min(1e-12).log()).sum()),
                    "modal_pred_share": float(torch.bincount(lg.argmax(1), minlength=V).max() / lg.shape[0]),
                    "ce": ms(torch.nn.functional.cross_entropy(lg, true, reduction="none").numpy())}
    res["answer_extra"] = extra
    res["target_hist"] = torch.bincount(true, minlength=V).tolist()
    res["answer_loss"] = {"full": ms(torch.cat(full_loss).numpy()), "no_fast": ms(torch.cat(nofast_loss).numpy())}
    res["answer_margin"] = ms(torch.cat(full_margin).numpy())
    res["recall_n"] = int(true.numel())
    res["align"], res["geom"] = {}, {}
    for k in range(3):
        a = {n: torch.cat(v).numpy() for n, v in A[k].items()}
        out = {n: ms(v) for n, v in a.items()}
        out["frac_store_write_descends"] = float((a["dL_store"] < 0).mean())
        out["frac_store_write_descends_unmasked"] = float((a["dL_store_unmasked"] < 0).mean())
        out["frac_all_fast_descends"] = float((a["dL_fast_total"] < 0).mean())
        out["frac_self_e_descent"] = float((a["self_desc"] > 0).mean())
        out["frac_cross_e_descent"] = float((a["cross_desc"] > 0).mean())
        out["frac_minEig_positive"] = float((a["min_eig"] > 0).mean())
        out["frac_cos_gq_pq_positive"] = float((a["cos_gq_pq"] > 0).mean())
        d = Bfb[k].shape[1]
        kk = math.sqrt(2 * (V - 1) / d)
        out["toy_threshold_cos"] = kk / math.sqrt(1 + kk * kk)
        out["V"], out["d"] = V, d
        res["align"][LN[k]] = out
        X = torch.cat(G[k]["X"]).double()
        ev = torch.linalg.eigvalsh(X.T @ X).clamp_min(0).cpu().numpy()
        res["geom"][LN[k]] = {"off_diag_cos": ms(G[k]["off"]), "cos_store_query": ms(G[k]["csq"]), "PR_dim": pr(ev),
                              "top1_eig_share": float(ev.max() / ev.sum())}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("done", it, "recall", res["ablation"]["full"], "align cos", {l: res["align"][l]["cosF"]["mean"] for l in LN})


if __name__ == "__main__":
    main()
