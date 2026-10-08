"""Measures the low-rank / fast-weight-programmer claims on a trained key-recall EphemeralRNN.

Run from the repo root (code dir) on a checkpoint:
    CUDA_VISIBLE_DEVICES=0 python scratch_lowrank/measure.py --checkpoint CKPT --out OUTDIR [--batches 128]

Held-out sequences (validation split) are run step by step exactly as heldout.py's "observed"
protocol (start_sequence_wipe, slow weights frozen, every target writes the fast entries through
the model's own dfa_layer_step), with forward hooks recording per layer (3 trunk layers + i2h):
x_t (the layer input), p_t (the DFA projected error of the step), and the fast read F^(t) x_t.
Everything else (key geometry, value subspace, mask rank, read decomposition, signal vs
crosstalk, causal ablations, F norms) is computed from those records, in float64 where noted.
Writes OUTDIR/results.json and OUTDIR/arrays.npz (data for plot.py).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from ephemeral_model import EphemeralRNN, dfa_output_error, dfa_layer_step  # noqa: E402
from heldout import load_heldout_batches  # noqa: E402
from utils import initialize_charset, load_checkpoint, model_input, read_checkpoint, upgrade_legacy_config  # noqa: E402

LAYERS = ["L0", "L1", "L2", "i2h"]


def pr(eigs):
    eigs = np.asarray(eigs, dtype=np.float64)
    return float(eigs.sum() ** 2 / (eigs ** 2).sum())


def ms(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "std": float(v.std()), "n": int(v.size)}


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


@torch.no_grad()
def run_batch(model, config, texts, onehot, device):
    """One batch of episodes, 'observed' protocol. Returns per-step records (lists over steps)."""
    B, Tn = onehot.shape[0], onehot.shape[1]
    steps = Tn - 1
    lay = layers_of(model)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    cap = {}
    hooks = []
    for k, layer in enumerate(lay):
        def hook(mod, inp, out, k=k):
            x = inp[0]
            m = mod.ephemeral_mask
            fast = torch.bmm(torch.where(m.unsqueeze(0), mod.per_sample_weights, torch.zeros_like(mod.per_sample_weights)),
                             x.unsqueeze(2)).squeeze(2)
            cap[k] = (x.detach().clone(), out.detach().clone(), fast)
        hooks.append(layer.register_forward_hook(hook))

    store = np.array([t.index("?") for t in texts])
    query = np.array([t.index("!") for t in texts])
    lengths = np.array([len(t) for t in texts])
    model.start_sequence_wipe()
    hidden = model.initHidden(B)
    rec = {k: {"x": [], "p": [], "pre": [], "fast": [], "Fnorm": [], "slow_norm": []} for k in range(4)}
    valid = []
    preds, logits_q = [], []
    abl_rows = {}
    # weight snapshots at each row's query step, for ablations (taken at that step)
    for i in range(steps):
        inp = model_input(onehot, i, config["input_mode"], config["pe_matrix"])
        output, hidden = model(inp, hidden)
        preds.append(output.argmax(1))
        loss, err = dfa_output_error(output, onehot[:, i + 1], criterion)
        valid.append(onehot[:, i + 1].sum(1) > 0)
        # ablation forwards at rows whose query step is i (needs records of earlier steps)
        for k in range(4):
            x, pre, fast = cap[k]
            rec[k]["x"].append(x)
            rec[k]["pre"].append(pre)
            rec[k]["fast"].append(fast)
            layer = lay[k]
            F = torch.where(layer.ephemeral_mask.unsqueeze(0), layer.per_sample_weights, torch.zeros_like(layer.per_sample_weights))
            rec[k]["Fnorm"].append(torch.linalg.vector_norm(F, dim=(1, 2)))
        if (query == i).any():
            abl_rows[i] = ablation_forward(model, lay, rec, i, store, onehot, config)
        # the model's own DFA step, restricted to fast entries (as heldout.py), recording p_t
        projected, _ = model.dfa_step_errors(err, 0)
        for k, (layer, e) in enumerate(zip(model.trained_layers(), projected)):
            if layer.is_last_layer:
                continue
            rec[k]["p"].append(e.clone())
            model._fast_entry_step(layer, e, config["learning_rate"], config["ephemeral_update_clamp"])
    for h in hooks:
        h.remove()
    return rec, torch.stack(valid), torch.stack(preds), store, query, lengths, abl_rows


def ablation_forward(model, lay, rec, i, store, onehot, config):
    """At step i (the query step of some rows), the answer logits under modified fast weights, for
    all rows (use only rows whose query is i). Modifications are per layer l in {L0,L1,L2}:
    drop_signal (remove the step-s* term), signal_only (slow + step-s* term only), no_fast (slow
    only); plus all layers at once. Terms are rebuilt from recorded x_s, p_s: exact closed form."""
    B = onehot.shape[0]
    c = float(config["learning_rate"] * lay[0].fused_plasticity())
    rho = 1 - model.forget_rate
    s = torch.as_tensor(store, device=onehot.device).clamp(max=i - 1)
    ar = torch.arange(B, device=onehot.device)

    def sig_matrix(k):
        layer = lay[k]
        x = torch.stack(rec[k]["x"])[s, ar]  # [B,in] x_{s*}
        p = torch.stack(rec[k]["p"])[s, ar]  # [B,out]
        coef = -c * rho ** (i - s).to(x.dtype)
        T = (coef[:, None] * p).unsqueeze(2) * x.unsqueeze(1)
        return torch.where(layer.ephemeral_mask.unsqueeze(0), T, torch.zeros_like(T))

    x0 = model_input(onehot, i, config["input_mode"], config["pe_matrix"])
    # state at this moment: the real per_sample_weights (already includes writes up to step i-1)
    full = {}
    for k in range(3):
        layer = lay[k]
        m = layer.ephemeral_mask.unsqueeze(0)
        W = layer.per_sample_weights.data
        slow = torch.where(m, torch.zeros_like(W), W)
        sig = sig_matrix(k)
        full[k] = {"full": W, "drop_signal": W - sig, "signal_only": slow + sig, "no_fast": slow}

    def logits(choice):  # choice[k] in full[k] keys
        h = torch.cat((x0, torch.zeros(B, model.hidden_size, device=x0.device)), 1)
        for k in range(3):
            h = torch.einsum("boi,bi->bo", full[k][choice[k]], h) + lay[k].bias
            h = torch.nn.functional.gelu(h)
        return torch.einsum("boi,bi->bo", model.i2o.per_sample_weights, h) + model.i2o.bias

    out = {"full": logits(["full"] * 3)}
    for mode in ("drop_signal", "signal_only", "no_fast"):
        for k in range(3):
            ch = ["full"] * 3
            ch[k] = mode
            out[f"{mode}@L{k}"] = logits(ch)
        out[f"{mode}@all"] = logits([mode] * 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=128)
    ap.add_argument("--svd_seqs", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    model, config, charset, it = load_model(args.checkpoint, device)
    dataset = config["dataset"]
    batches = load_heldout_batches(dataset, config["batch_size"], args.batches, device, "validation")
    lay = layers_of(model)
    c = float(config["learning_rate"] * lay[0].fused_plasticity())
    rho = 1 - model.forget_rate
    chars = list(charset)
    res = {"iteration": it, "episodes": len(batches) * config["batch_size"], "c_lr_alpha": c, "rho": rho,
           "charset": "".join(map(str, chars))}
    d_in = lay[0].in_features
    print("checkpoint iter", it, "c", c, "d_in", d_in, "episodes", res["episodes"])

    # ----------------------------------------------------------- (c) mask rank (actual masks)
    mask_rank = {}
    for k, layer in enumerate(lay):
        m = layer.ephemeral_mask.data.double().cpu()
        sv = torch.linalg.svdvals(m)
        tol = sv[0] * max(m.shape) * torch.finfo(torch.float64).eps
        mask_rank[LAYERS[k]] = {"shape": list(m.shape), "frac": float(m.mean()), "rank": int((sv > tol).sum()),
                                "sigma_max": float(sv[0]), "sigma_min": float(sv[-1]),
                                "sigma_2": float(sv[1]), "nnz_per_row_mean": float(m.sum(1).mean()),
                                "n_sv_gt_1e-6_max": int((sv > 1e-6 * sv[0]).sum())}
        print("mask", LAYERS[k], mask_rank[LAYERS[k]])
    res["mask_rank"] = mask_rank

    # ------------------------------------------------------------------ accumulate over batches
    acc = {k: {"X": [], "P": [], "seqrow": [], "step": []} for k in range(4)}
    per_seq = []  # dict per sequence of scalar stats
    abl_logit = {}  # name -> list of [n,vocab] logits at query rows
    abl_true = []
    closed_err = {k: [] for k in range(4)}
    fn_traj = {k: [] for k in range(4)}  # [n_seq, steps]
    fx_traj = {k: [] for k in range(4)}  # ||F x_t|| per step
    pre_traj = {k: [] for k in range(4)}
    svd_done = 0
    svd_res = {k: [] for k in range(4)}  # list of per-sequence lists of (rank_actual, rank_closed, smin/smax)
    recall_hits = []
    cos_sq = {k: [] for k in range(4)}  # cos(x_s*, q) per sequence
    reads = {k: [] for k in range(4)}  # per-sequence read decomposition stats
    cos_pairs_within = {k: [] for k in range(4)}
    cos_pairs_centered = {k: [] for k in range(4)}
    cos_s_other = {k: [] for k in range(4)}
    prof = {k: [] for k in range(4)}  # per step-before-query share profiles (relative to query)
    maxlag = 12
    for bi, (texts, onehot) in enumerate(batches):
        rec, valid, preds, store, query, lengths, abl = run_batch(model, config, texts, onehot, device)
        steps = valid.shape[0]
        B = onehot.shape[0]
        for k in range(4):
            Xs = torch.stack(rec[k]["x"]).double()  # [T,B,in]
            Ps = torch.stack(rec[k]["p"]).double()  # [T,B,out]
            fast_actual = torch.stack(rec[k]["fast"]).double()  # [T,B,out]
            pre = torch.stack(rec[k]["pre"]).double()
            Fn = torch.stack(rec[k]["Fnorm"]).double()  # [T,B]
            layer = lay[k]
            M = layer.ephemeral_mask.data.double()
            # closed-form fast read at every step t for every row: F_t x_t = sum_{s<t} -c rho^{t-s} p_s (m (x_s*x_t))
            for b in range(B):
                q_i = int(query[b])
                s_i = int(store[b])
                v = valid[:, b].cpu().numpy()
                Xb, Pb = Xs[:, b], Ps[:, b]
                # (a) geometry within sequence over writing steps
                idx = np.where(v)[0]
                Xv = Xb[idx]
                nrm = Xv.norm(dim=1, keepdim=True).clamp_min(1e-12)
                U = Xv / nrm
                G = U @ U.T
                n = len(idx)
                off = (G.sum() - G.diag().sum()) / (n * (n - 1))
                cos_pairs_within[k].append(float(off))
                Xc = Xv - Xv.mean(0, keepdim=True)
                Uc = Xc / Xc.norm(dim=1, keepdim=True).clamp_min(1e-12)
                Gc = Uc @ Uc.T
                cos_pairs_centered[k].append(float((Gc.sum() - Gc.diag().sum()) / (n * (n - 1))))
                cos_sq[k].append(float(Xb[s_i] @ Xb[q_i] / (Xb[s_i].norm() * Xb[q_i].norm()).clamp_min(1e-12)))
                others = [j for j in idx if j not in (s_i, q_i)]
                if others:
                    Uo = Xb[others] / Xb[others].norm(dim=1, keepdim=True).clamp_min(1e-12)
                    cos_s_other[k].append(float((Uo @ (Xb[q_i] / Xb[q_i].norm().clamp_min(1e-12))).abs().mean()))
                acc[k]["X"].append(Xv.cpu().float())
                acc[k]["P"].append(Pb[idx].cpu().float())
                # closed-form check of the fast read at all steps
                T = steps
                read_cf = torch.zeros(T, M.shape[0], dtype=torch.float64, device=Xs.device)
                terms = {}
                for t in range(T):
                    if t == 0:
                        continue
                    ss = torch.arange(0, t, device=Xs.device)
                    prod = Xb[ss] * Xb[t].unsqueeze(0)  # [t,in]
                    g = (M @ prod.T).T  # [t,out]: m (x_s*q)
                    coef = -c * rho ** (t - ss).double()
                    tm = coef[:, None] * Pb[ss] * g  # [t,out] terms
                    read_cf[t] = tm.sum(0)
                    terms[t] = tm
                ca = fast_actual[:, b]
                rel = (read_cf - ca).norm(dim=1) / ca.norm(dim=1).clamp_min(1e-12)
                ok = ca.norm(dim=1) > 1e-8
                if ok.any():
                    closed_err[k].append(float(rel[ok].max()))
                # (f) trajectories (valid lengths)
                L = int(lengths[b]) - 1
                fn_traj[k].append(np.pad(Fn[:, b].cpu().numpy()[:L], (0, maxlag - L), constant_values=np.nan))
                fx_traj[k].append(np.pad(ca.norm(dim=1).cpu().numpy()[:L], (0, maxlag - L), constant_values=np.nan))
                pre_traj[k].append(np.pad(pre[:, b].norm(dim=1).cpu().numpy()[:L], (0, maxlag - L), constant_values=np.nan))
                # (d)/(e) read decomposition at the answer step
                tm = terms[q_i]  # [q_i, out]
                ss = np.arange(q_i)
                tn = tm.norm(dim=1)
                total = tm.sum(0)  # masked F q (closed form)
                Pq = Pb[:q_i]
                pn = Pq.norm(dim=1).clamp_min(1e-12)
                cos_to_p = (tm * Pq).sum(1) / (tn.clamp_min(1e-30) * pn)
                # unmasked-formula read: -c sum rho^{q-s} p_s (x_s . q)
                dots = (Xb[:q_i] * Xb[q_i].unsqueeze(0)).sum(1)
                un = (-c * rho ** (q_i - torch.arange(q_i, device=Xs.device)).double() * dots)[:, None] * Pq
                U_read = un.sum(0)
                un_norm = U_read.norm()
                tot_norm = total.norm()
                sig = tm[s_i] if s_i < q_i else torch.zeros_like(total)
                cross = total - sig
                dom = int(tn.argmax())
                # share via squared norms of terms and via projection on the true p_s direction
                e_tot = (tn ** 2).sum().clamp_min(1e-30)
                psd = Pb[s_i] / Pb[s_i].norm().clamp_min(1e-12)
                reads[k].append({
                    "tot_norm": float(tot_norm), "unmasked_norm": float(un_norm),
                    "ratio_to_phi": float(tot_norm / (layer.ephemeral_mask.float().mean() * un_norm).clamp_min(1e-30)),
                    "ratio_to_unmasked": float(tot_norm / un_norm.clamp_min(1e-30)),
                    "rel_err_unmasked": float((total - U_read).norm() / tot_norm.clamp_min(1e-30)),
                    "rel_err_phi_unmasked": float((total - layer.ephemeral_mask.float().mean().double() * U_read).norm() / tot_norm.clamp_min(1e-30)),
                    "dom_is_store": float(dom == s_i), "dom_lag": int(q_i - dom),
                    "sig_norm": float(sig.norm()), "cross_norm": float(cross.norm()),
                    "sig_share_sq": float(tn[s_i] ** 2 / e_tot) if s_i < q_i else 0.0,
                    "sig_over_cross": float(sig.norm() / cross.norm().clamp_min(1e-30)),
                    "sig_proj_share": float((total @ psd) / total.norm().clamp_min(1e-30)),
                    "cos_total_psstar": float((total @ psd) / total.norm().clamp_min(1e-30)),
                    "cos_sig_psstar": float((sig @ psd) / sig.norm().clamp_min(1e-30)),
                    "sig_proj_on_total_over_total": float((sig @ total) / (total @ total).clamp_min(1e-30)),
                    "cos_term_p_store": float(cos_to_p[s_i]) if s_i < q_i else float("nan"),
                    "mean_abs_cos_term_p_other": float(cos_to_p[[j for j in range(q_i) if j != s_i]].abs().mean()) if q_i > 1 else float("nan"),
                    "pre_norm_at_q": float(pre[q_i, b].norm()),
                    "fast_norm_at_q": float(ca[q_i].norm()),
                    "slow_part_norm_at_q": float((pre[q_i, b] - ca[q_i]).norm()),
                    "fast_over_pre": float(ca[q_i].norm() / pre[q_i, b].norm()),
                    "cos_fast_slowpart": float(ca[q_i] @ (pre[q_i, b] - ca[q_i]) / (ca[q_i].norm() * (pre[q_i, b] - ca[q_i]).norm()).clamp_min(1e-30)),
                    "term_norms_by_lag": [float(tn[q_i - lag]) if q_i - lag >= 0 else float("nan") for lag in range(1, maxlag)],
                    "lag_store": int(q_i - s_i),
                })
                # (c) rank of masked F^(t) on a few sequences
                if k >= 0 and svd_done < args.svd_seqs and b == 0:
                    pass
        # SVD of F^(t) from actual fast weights: replay for first sequences of first batches
        if bi < args.svd_seqs:
            sv_rows = svd_replay(model, config, onehot, texts, device)
            for k in range(4):
                svd_res[k].append(sv_rows[k])
        # ablation logits at query steps
        q_rows = {}
        for b in range(onehot.shape[0]):
            q_rows.setdefault(int(query[b]), []).append(b)
        tgt = onehot[:, :, :].argmax(-1)
        for qi_, bs in q_rows.items():
            for name, lg in abl[qi_].items():
                abl_logit.setdefault(name, []).append(lg[bs].cpu())
            abl_true.append(tgt[bs, qi_ + 1].cpu())
        if bi % 16 == 0:
            print("batch", bi, flush=True)

    # --------------------------------------------------------------- summaries
    # ablation accuracies
    true = torch.cat(abl_true)
    abl = {}
    for name, lst in abl_logit.items():
        lg = torch.cat(lst)
        correct = lg.gather(1, true[:, None])[:, 0]
        other = lg.clone()
        other.scatter_(1, true[:, None], -1e30)
        margin = correct - other.max(1).values
        abl[name] = {"recall_acc": float((lg.argmax(1) == true).float().mean()),
                     "margin_mean": float(margin.mean()), "margin_std": float(margin.std()), "n": int(len(true))}
    res["ablation"] = abl
    res["closed_form_max_rel_err_of_fast_read"] = {LAYERS[k]: ms(closed_err[k]) for k in range(4)}

    res["geometry"] = {}
    for k in range(4):
        X = torch.cat(acc[k]["X"]).double()
        N, d = X.shape
        Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        ev = torch.linalg.eigvalsh(Xn.T @ Xn).clamp_min(0).cpu().numpy()  # unit-norm keys, uncentered second moment
        evc = torch.linalg.eigvalsh(((Xn - Xn.mean(0)).T @ (Xn - Xn.mean(0)))).clamp_min(0).cpu().numpy()
        # isotropic finite-sample baseline with the same N
        G = torch.randn(N, d, dtype=torch.float64, device=X.device)
        G = G / G.norm(dim=1, keepdim=True)
        evg = torch.linalg.eigvalsh(G.T @ G).clamp_min(0).cpu().numpy()
        # distinct-key spectrum: group by nearest-of-9 chars is implicit; also PR of char-type means
        # participation in top components
        srt = np.sort(ev)[::-1]
        mean_dir = Xn.mean(0)
        res["geometry"][LAYERS[k]] = {
            "d": d, "N_keys": N, "iso_cos_rms_1_over_sqrt_d": 1 / math.sqrt(d),
            "iso_mean_abs_cos": math.sqrt(2 / (math.pi * d)),
            "within_seq_mean_offdiag_cos": ms(cos_pairs_within[k]),
            "within_seq_mean_offdiag_cos_centered": ms(cos_pairs_centered[k]),
            "cos_store_key_vs_query": ms(cos_sq[k]),
            "mean_abs_cos_query_vs_other_keys": ms(cos_s_other[k]),
            "PR_uncentered_unitnorm": pr(ev), "PR_centered_unitnorm": pr(evc), "PR_isotropic_same_N": pr(evg),
            "PR_isotropic_limit_d": d,
            "top1_eig_share": float(srt[0] / srt.sum()), "top9_eig_share": float(srt[:9].sum() / srt.sum()),
            "top20_eig_share": float(srt[:20].sum() / srt.sum()),
            "n_eig_for_99pct": int(np.searchsorted(np.cumsum(srt) / srt.sum(), 0.99) + 1),
            "mean_direction_norm": float(mean_dir.norm()),
            "frac_exact_zero_entries": float((X == 0).double().mean()),
            "mean_nonzero_entries_per_key": float((X != 0).double().sum(1).mean()),
        }
    res["value_subspace"] = {}
    for k in range(4):
        P = torch.cat(acc[k]["P"]).double()
        B_ = lay[k].feedback_weights.data.double().to(P.device)  # [vocab,out]
        Q, _ = torch.linalg.qr(B_.T)  # [out, vocab]
        proj = P @ Q
        frac = (proj.norm(dim=1) ** 2 / (P.norm(dim=1) ** 2).clamp_min(1e-30))
        ev = torch.linalg.eigvalsh(P.T @ P).clamp_min(0).cpu().numpy()
        srt = np.sort(ev)[::-1]
        w = (P.norm(dim=1) ** 2)
        res["value_subspace"][LAYERS[k]] = {
            "N": int(P.shape[0]), "out": int(P.shape[1]), "PR_p": pr(ev), "top9_eig_share": float(srt[:9].sum() / srt.sum()),
            "n_eig_for_99pct": int(np.searchsorted(np.cumsum(srt) / srt.sum(), 0.99) + 1),
            "n_eig_for_999pct": int(np.searchsorted(np.cumsum(srt) / srt.sum(), 0.999) + 1),
            "energy_frac_in_rowspace_of_B_mean_per_p": float(frac.mean()), "energy_frac_in_rowspace_of_B_min": float(frac.min()),
            "energy_frac_in_rowspace_of_B_energy_weighted": float((frac * w).sum() / w.sum()),
            "vocab": int(B_.shape[0]), "p_norm_mean": float(P.norm(dim=1).mean()),
        }
    res["read"] = {}
    for k in range(4):
        keys = [kk for kk in reads[k][0] if kk != "term_norms_by_lag"]
        out = {kk: ms([r[kk] for r in reads[k]]) for kk in keys}
        lags = np.array([r["term_norms_by_lag"] for r in reads[k]], dtype=np.float64)
        out["term_norm_by_lag_mean"] = [float(np.nanmean(lags[:, j])) for j in range(lags.shape[1])]
        out["store_lag_counts"] = {int(a): int(b) for a, b in zip(*np.unique([r["lag_store"] for r in reads[k]], return_counts=True))}
        res["read"][LAYERS[k]] = out
    res["F_traj"] = {LAYERS[k]: {"Fnorm_mean_by_step": np.nanmean(np.array(fn_traj[k]), 0).tolist(),
                                 "Fnorm_std_by_step": np.nanstd(np.array(fn_traj[k]), 0).tolist(),
                                 "Fx_norm_mean_by_step": np.nanmean(np.array(fx_traj[k]), 0).tolist(),
                                 "pre_norm_mean_by_step": np.nanmean(np.array(pre_traj[k]), 0).tolist()} for k in range(4)}
    res["slow_norm_Fro_masked_out"] = {}
    for k in range(4):
        layer = lay[k]
        W = layer.per_sample_weights.data[0]
        m = layer.ephemeral_mask.data
        res["slow_norm_Fro_masked_out"][LAYERS[k]] = float(torch.linalg.vector_norm(torch.where(m, torch.zeros_like(W), W)))
    res["svd_F"] = {LAYERS[k]: svd_res[k] for k in range(4)}
    np.savez(os.path.join(args.out, "arrays.npz"),
             **{f"fn_{LAYERS[k]}": np.array(fn_traj[k]) for k in range(4)},
             **{f"fx_{LAYERS[k]}": np.array(fx_traj[k]) for k in range(4)},
             **{f"pre_{LAYERS[k]}": np.array(pre_traj[k]) for k in range(4)},
             **{f"lagterms_{LAYERS[k]}": np.array([r["term_norms_by_lag"] for r in reads[k]]) for k in range(4)},
             **{f"geom_eig_{LAYERS[k]}": np.sort(torch.linalg.eigvalsh((lambda Xn: Xn.T @ Xn)(
                 (lambda X: X / X.norm(dim=1, keepdim=True).clamp_min(1e-12))(torch.cat(acc[k]["X"]).double()))).cpu().numpy())[::-1] for k in range(4)},
             **{f"p_eig_{LAYERS[k]}": np.sort(torch.linalg.eigvalsh((lambda P: P.T @ P)(torch.cat(acc[k]["P"]).double())).clamp_min(0).cpu().numpy())[::-1] for k in range(4)},
             **{f"sigsh_{LAYERS[k]}": np.array([r["sig_share_sq"] for r in reads[k]]) for k in range(4)},
             **{f"ratio_phi_{LAYERS[k]}": np.array([r["ratio_to_phi"] for r in reads[k]]) for k in range(4)})
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("done")


@torch.no_grad()
def svd_replay(model, config, onehot, texts, device):
    """Replays the batch's first sequence (row 0) and returns, per layer, a list over steps t of the
    float64 singular-value summary of the masked F^(t) from the actual fast weights, plus the exact
    closed form F^(t) rebuilt from x_s, p_s in float64."""
    lay = layers_of(model)
    crit = torch.nn.CrossEntropyLoss(reduction="none")
    model.start_sequence_wipe()
    hidden = model.initHidden(onehot.shape[0])
    steps = onehot.shape[1] - 1
    L = len(texts[0]) - 1
    xs = {k: [] for k in range(4)}
    ps = {k: [] for k in range(4)}
    out = {k: [] for k in range(4)}
    c = float(config["learning_rate"] * lay[0].fused_plasticity())
    rho = 1 - model.forget_rate
    for i in range(L):
        output, hidden = model(model_input(onehot, i, config["input_mode"], config["pe_matrix"]), hidden)
        _, err = dfa_output_error(output, onehot[:, i + 1], crit)
        projected, _ = model.dfa_step_errors(err, 0)
        for k, (layer, e) in enumerate(zip(model.trained_layers(), projected)):
            if layer.is_last_layer:
                continue
            xs[k].append(layer.in_traces.data[0].double().clone())
            ps[k].append(e[0].double().clone())
            model._fast_entry_step(layer, e, config["learning_rate"], config["ephemeral_update_clamp"])
            F = torch.where(layer.ephemeral_mask, layer.per_sample_weights.data[0], torch.zeros_like(layer.per_sample_weights.data[0])).double()
            sv = torch.linalg.svdvals(F.cpu())
            tol = sv[0] * max(F.shape) * torch.finfo(torch.float64).eps
            # closed form in float64
            Mm = layer.ephemeral_mask.double()
            Fc = torch.zeros_like(F)
            for s in range(len(xs[k])):
                Fc += -c * rho ** (len(xs[k]) - s) * (ps[k][s][:, None] * xs[k][s][None, :]) * Mm
            svc = torch.linalg.svdvals(Fc.cpu())
            tolc = svc[0] * max(F.shape) * torch.finfo(torch.float64).eps
            out[k].append({"t": len(xs[k]), "rank_actual_f32weights": int((sv > tol).sum()),
                           "rank_closed_f64": int((svc > tolc).sum()),
                           "rank_closed_f64_tol1e-6": int((svc > 1e-6 * svc[0]).sum()),
                           "smin_over_smax_closed": float(svc[-1] / svc[0]), "sigma_max": float(svc[0]),
                           "rel_diff_actual_vs_closed": float((F - Fc).norm() / Fc.norm().clamp_min(1e-30)),
                           "n_nonzero_cols": int((F != 0).any(0).sum()),
                           "PR_sv2": float((svc ** 2).sum() ** 2 / (svc ** 4).sum())})
    return out


if __name__ == "__main__":
    main()
