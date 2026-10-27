"""Order-two experiment (video experiment 5, PREREG_fastlayers.md "Order-two fast read (2026-10-22)"):
key-match ratio and selectivity of every fast read at the answer step, with a kernel swap.

Run from the repo root:  python scratch/order2_2026-10-22/o2_keys.py CKPT [CKPT ...] --out DIR [--batches N]

Replays held-out episodes (key recall or MQAR) under heldout.py's strict protocol: slow weights frozen,
the model's own fast-only DFA step on every step before the first recall target, then forgetting only.
For every recall target (answer predicted at step q, stored at step s* = source - 1, whose input is the
key and whose target is the value) and every trunk layer, with the written steps t < q:

  linear masked read (the fast entries), as in cheap checks section 1:
      t_s = -c rho^e p_s * (M (x_s * x_q)),  key factor g_s = |t_s| / (c rho^e |p_s|)
  order-two matrix G (layers with --order2_layers):
      u_s = -c rho^e p_s (psi_s . psi_q),     key factor |psi_s . psi_q|
  kernel swap on the same inputs (any checkpoint, any layer): with the projection this run's seed would
  draw (ephemeral_model.order2_projection) at --order2_dim 16, the scalar kernels
      lin  = fraction * x_s . x_q      (the masked read's expectation over the mask)
      o2   = psi2_s . psi2_q           (order2 map, temperature T)
      o1   = psi1_s . psi1_q           (order1 map, same projection and T)
  for T in --temps (default 4).

Per kernel or read: key_ratio = key factor of s* / median key factor of the other written steps (the
section-1 "key factor store / median other"); selectivity = signed kernel of s* / rms of the signed
kernels of the others (section 2's measure); for reads also the signal share |t_s*|^2 / sum |t_s|^2.
Also: raw and projected cosines (store-query, mean other-query), the exponent check of the closed form,
and the answer accuracy with fast reads removed at the answer step (all, no G, no linear fast in the
order-two layers, only G, no fast at all).
Medians over targets per checkpoint and layer go to DIR/<run>__<checkpoint>.json."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from ephemeral_model import dfa_output_error, order2_features, order2_projection  # noqa: E402
from heldout import load_heldout_batches, update_mask  # noqa: E402
from metrics import recall_targets  # noqa: E402
from train import build_model, build_parser, positional_encoding  # noqa: E402
from utils import initialize_charset, load_checkpoint, model_input, read_checkpoint, upgrade_legacy_config  # noqa: E402


def load(path, device):
    ck = read_checkpoint(path)
    defaults = {k: v for k, v in vars(build_parser().parse_args([])).items() if not k.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(ck.get("config", {}))}
    charset, _, _, n = initialize_charset(config["dataset"])
    model = build_model(config, charset, n)
    model, _, it, _, _ = load_checkpoint(path, model, config, device=device, checkpoint=ck)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], device)
    return model.to(device).eval(), config, it - 1


def ratio(store, others):
    return float(store / max(float(np.median(others)), 1e-300))


def selectivity(store, others):
    return float(store / max(float(np.sqrt(np.mean(np.square(others)))), 1e-300))


@torch.no_grad()
def run(path, nb, device, temps, kernel_observer=None):
    model, config, it = load(path, device)
    layers = list(model.linear_layers)
    L = len(layers)
    lr, rho = config["learning_rate"], 1.0 - model.forget_rate
    seed = config.get("seed") or 0
    swap_proj = {k: order2_projection(l.in_features, 16, seed, k).to(device).double() for k, l in enumerate(layers)}
    frac = {k: float(l.ephemeral_mask.float().mean()) for k, l in enumerate(layers)}
    cap = {}
    hooks = [l.register_forward_hook(lambda m, inp, out, k=k: cap.__setitem__(k, inp[0].detach().double()))
             for k, l in enumerate(layers)]
    crit = torch.nn.CrossEntropyLoss(reduction="none")
    rec = {k: [] for k in range(L)}
    chk = {k: {"lin": {0: [], 1: []}, "G": {0: [], 1: []}} for k in range(L)}
    abl_names = ("all", "noG", "nolin_o2layers", "onlyG", "none")
    acc = {n: [] for n in abl_names}
    o2 = [k for k, l in enumerate(layers) if l.order2 is not None]
    dataset = config["dataset"]
    for texts, onehot in load_heldout_batches(dataset, config["batch_size"], nb, device, "validation"):
        B, T = onehot.shape[0], onehot.shape[1]
        mask = update_mask("strict", texts, dataset, onehot)
        targets = []
        for b, text in enumerate(texts):
            for target, lag in recall_targets(text, dataset)[0].items():
                if target - 1 < T - 1:
                    targets.append((b, target - 1, target - lag - 2))  # (row, q, s*)
        by_step = {}
        for b, q, s in targets:
            by_step.setdefault(q, []).append((b, s))
        model.start_sequence_wipe()
        hidden = model.initHidden(B)
        X = {k: [] for k in range(L)}
        PSI = {k: [] for k in range(L)}
        P = {k: [] for k in range(L)}
        PG = {k: [] for k in range(L)}
        W = []  # [T] bool per row: was step t written
        for i in range(T - 1):
            inp = model_input(onehot, i, config["input_mode"], config["pe_matrix"])
            if i in by_step:
                saved = [l.per_sample_weights.clone() for l in layers]
                savedG = {k: layers[k].order2_fast.clone() for k in o2}
                for name in abl_names:
                    for k, l in enumerate(layers):
                        drop_lin = name in ("none", "onlyG") or (name == "nolin_o2layers" and k in o2)
                        l.per_sample_weights.copy_(torch.where(l.ephemeral_mask.unsqueeze(0), torch.zeros_like(saved[k]), saved[k])
                                                   if drop_lin else saved[k])
                        if k in o2:
                            l.order2_fast.copy_(torch.zeros_like(savedG[k]) if name in ("noG", "none") else savedG[k])
                    o, _ = model(inp, hidden)
                    for b, s in by_step[i]:
                        acc[name].append(int(o[b].argmax()) == int(onehot[b, i + 1].argmax()))
                for k, l in enumerate(layers):
                    l.per_sample_weights.copy_(saved[k])
                    if k in o2:
                        l.order2_fast.copy_(savedG[k])
            out, hidden = model(inp, hidden)
            for k, l in enumerate(layers):
                X[k].append(cap[k])
                PSI[k].append(l.order2_trace.double() if k in o2 else None)
            if i in by_step:
                for b, s in by_step[i]:
                    q = i
                    written = [t for t in range(q) if bool(W[t][b])]
                    if s not in written:
                        continue
                    others = [t for t in written if t != s]
                    if not others:
                        continue
                    for k, l in enumerate(layers):
                        r = {"q": q, "s": s, "n_others": len(others)}
                        xq = X[k][q][b]
                        Xs = torch.stack([X[k][t][b] for t in written])  # [n, in]
                        cos = torch.nn.functional.cosine_similarity(Xs, xq.unsqueeze(0), dim=1).cpu().numpy()
                        si = written.index(s)
                        oi = [j for j in range(len(written)) if j != si]
                        if kernel_observer is not None:
                            kernel_observer(k, Xs, xq, si, oi, swap_proj[k], frac[k], seed)
                        r["cos_sq"], r["cos_oth"] = float(cos[si]), float(np.mean(cos[oi]))
                        # Uncentered unit-key second-moment PR (same definition as B2).
                        unit = torch.nn.functional.normalize(Xs, dim=1)
                        gram = unit @ unit.t()
                        r["key_pr"] = float(gram.trace().square() / gram.square().sum().clamp_min(1e-30))
                        r["cos_oth_abs"] = float(np.mean(np.abs(cos[oi])))
                        r["cos_oth_p95"] = float(np.quantile(cos[oi], 0.95))
                        dots = (Xs @ xq).cpu().numpy()
                        r["lin_sel"] = selectivity(dots[si], dots[oi])
                        r["lin_ratio"] = ratio(abs(dots[si]), np.abs(dots[oi]))
                        u = Xs @ swap_proj[k].t(); uq = swap_proj[k] @ xq
                        pc = torch.nn.functional.cosine_similarity(u, uq.unsqueeze(0), dim=1).cpu().numpy()
                        r["pcos_sq"], r["pcos_oth"] = float(pc[si]), float(np.mean(pc[oi]))
                        for tau in temps:
                            for order in ("order2", "order1"):
                                fs = order2_features(Xs, swap_proj[k], order, tau, frac[k] or 0.1)
                                fq = order2_features(xq.unsqueeze(0), swap_proj[k], order, tau, frac[k] or 0.1)
                                kv = (fs @ fq[0]).cpu().numpy()
                                tag = f"{'o2' if order == 'order2' else 'o1'}_t{tau:g}"
                                r[f"{tag}_ratio"] = ratio(abs(kv[si]), np.abs(kv[oi]))
                                r[f"{tag}_sel"] = selectivity(kv[si], kv[oi])
                        if bool(l.ephemeral_mask.any()):
                            M = l.ephemeral_mask.double()
                            c = lr * float(l.order2_alpha())
                            Ps = torch.stack([P[k][t][b] for t in written])
                            raw = -(Ps * ((Xs * xq.unsqueeze(0)) @ M.t()))  # [n, out]
                            actual = torch.where(l.ephemeral_mask, l.per_sample_weights[b], 0).double() @ xq
                            ex = np.array([q - t for t in written], dtype=float)
                            for off in (0, 1):
                                w = torch.tensor(c * rho ** (ex + off), dtype=torch.float64, device=device)
                                chk[k]["lin"][off].append(float(((raw * w[:, None]).sum(0) - actual).norm()
                                                                / actual.norm().clamp_min(1e-30)))
                            pn = Ps.norm(dim=1).cpu().numpy()
                            g = (raw.norm(dim=1).cpu().numpy()) / np.maximum(pn, 1e-300)
                            r["read_lin_key_ratio"] = ratio(g[si], g[oi])
                            r["p_ratio"] = ratio(pn[si], pn[oi])
                            w0 = c * rho ** ex
                            tn = w0 * raw.norm(dim=1).cpu().numpy()
                            r["read_lin_share"] = float(tn[si] ** 2 / np.sum(tn ** 2))
                            r["read_lin_largest"] = int(np.argmax(tn) == si)
                        if k in o2:
                            c = lr * float(l.order2_alpha())
                            Pg = torch.stack([PG[k][t][b] for t in written])
                            Fs = torch.stack([PSI[k][t][b] for t in written])
                            kg = (Fs @ PSI[k][q][b]).cpu().numpy()
                            actual = l.order2_fast[b].double() @ PSI[k][q][b]
                            ex = np.array([q - t for t in written], dtype=float)
                            for off in (0, 1):
                                w = torch.tensor(c * rho ** (ex + off), dtype=torch.float64, device=device)
                                pred = -(Pg * (torch.tensor(kg, device=device) * w)[:, None]).sum(0)
                                chk[k]["G"][off].append(float((pred - actual).norm() / actual.norm().clamp_min(1e-30)))
                            r["read_G_key_ratio"] = ratio(abs(kg[si]), np.abs(kg[oi]))
                            r["read_G_sel"] = selectivity(kg[si], kg[oi])
                            tn = (c * rho ** ex) * np.abs(kg) * Pg.norm(dim=1).cpu().numpy()
                            r["read_G_share"] = float(tn[si] ** 2 / np.sum(tn ** 2))
                            r["read_G_largest"] = int(np.argmax(tn) == si)
                        rec[k].append(r)
            _, err = dfa_output_error(out, onehot[:, i + 1], crit)
            err = err * mask[:, i, None]
            W.append(mask[:, i].clone())
            projected, _ = model.dfa_step_errors(err, 0)
            fast_lin = model.fast_write_projected(err, 0)
            overrides = model._fast_error_overrides(err, projected)
            by_id = {id(l): (e, f if f is not None else e2) for l, e, e2, f in
                     zip(model.trained_layers(), fast_lin, projected, overrides)}
            for k, l in enumerate(layers):
                P[k].append(by_id[id(l)][0].double())
                PG[k].append(by_id[id(l)][1].double())
            model.fast_only_dfa_step(err, lr, config["ephemeral_update_clamp"], 0)
    for h in hooks:
        h.remove()
    res = {"checkpoint": path, "iteration": it, "targets": len(acc["all"]),
           "order2_layers": o2, "config": {k: config.get(k) for k in
                                           ("dataset", "order2_layers", "order2_map", "order2_mode",
                                            "order2_temperature", "seed", "fast_layers")},
           "acc": {k: float(np.mean(v)) for k, v in acc.items()}, "layers": {}}
    for k in range(L):
        rows = rec[k]
        if not rows:
            continue
        keys = [key for key in rows[0] if key not in ("q", "s")]
        summ = {key: float(np.nanmedian([r[key] for r in rows if key in r])) for key in keys}
        summ["n"] = len(rows)
        for kind in ("lin", "G"):
            if chk[k][kind][0]:
                summ[f"closed_form_{kind}_err_e0"] = float(np.median(chk[k][kind][0]))
                summ[f"closed_form_{kind}_err_e1"] = float(np.median(chk[k][kind][1]))
        res["layers"][f"L{k}"] = summ
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temps", default="4")
    a = ap.parse_args()
    temps = [float(t) for t in a.temps.split(",")]
    os.makedirs(a.out, exist_ok=True)
    for path in a.checkpoints:
        res = run(path, a.batches, a.device, temps)
        name = os.path.basename(os.path.dirname(os.path.abspath(path))) + "__" + os.path.basename(path).replace(".pth", "")
        with open(os.path.join(a.out, name + ".json"), "w") as f:
            json.dump(res, f, indent=1)
        print(name, json.dumps({"acc": res["acc"], **{n: {k: round(v, 3) for k, v in d.items()} for n, d in res["layers"].items()}}),
              flush=True)


if __name__ == "__main__":
    main()
