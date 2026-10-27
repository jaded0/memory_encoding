"""P3 (expansive write gain, 2026-10-27): the 10-16 salience decomposition (check SG), extended for
--write_gain_power checkpoints. A copy of scratch/cheap_checks_2026-10-16/surprise_gating.py; every output
key of that script is computed the same way (on a checkpoint without the flag it gives the same numbers),
and these are added per layer:
  ratio_p_raw    the store/median-other ratio of the RAW (ungained) write error norms |e_s|
  ratio_gain     the same for the write gain g_s = |p_s| / |e_s| (1 everywhere when the flag is off)
  log_gain_part  the gain's part of the log lead (the log value part = raw part + gain part)
  store_largest_no_gain  fixed-trajectory counterfactual (terms divided by gain; NOT an ungained replay)
  gain_by_role   median gain by the role of the step's input character
and per checkpoint the flag's settings and the frozen running medians. With the gain on, |p_s| (the
write the read sums, as before) is the gained error, so the closed form still matches the actual read.

Original description:
Check SG (PREREG_fastlayers.md, "Cheap checks (2026-10-16)"): does the store-step term dominate the
answer-step fast read because its write (value) is big, or because its key matches the query?

Run from the repo root:  python scratch/p3_2026-10-27/p3_sg.py CKPT [CKPT ...] --out DIR
Key-recall episodes look like '0000?v0!v': the store step s* is the '?' step (its write carries the error of
predicting v), the query step q is the '!' step, the answer is predicted at q. Held-out episodes are replayed
under heldout.py's "observed" protocol (slow weights frozen, the model's own fast-only DFA step after every
target). Per layer with fast entries, and per earlier write step s < q, the answer-step read decomposes as
    F_q x_q = sum_s  t_s,   t_s = -c rho^e p_s * (M (x_s * x_q))        (exclusive mask M, c = lr * alpha)
with e = q - s (checked against the actual masked read; the script tries e = q - s and q - s + 1 and keeps the
one that matches). Writes DIR/<name>.json with, per layer:
  ratio_p   median over episodes of |p_s*| / median_{s != s*} |p_s|          (SG1)
  ratio_g   same for the key factor g_s = |t_s| / (c rho^e |p_s|)            (SG2)
  log-lead decomposition of log|t_s*| - mean_{s != s*} log|t_s| into value, key and decay parts  (SG3)
  store_largest, and the same after equalizing |p| or g across s           (SG4)
  signal share |t_s*|^2 / sum |t_s|^2, mean |p| by the role of the step's input character.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from ephemeral_model import dfa_output_error  # noqa: E402
from heldout import load_heldout_batches  # noqa: E402
from train import build_model, build_parser, positional_encoding  # noqa: E402
from utils import initialize_charset, load_checkpoint, model_input, read_checkpoint, upgrade_legacy_config  # noqa: E402

torch.set_num_threads(int(os.environ.get("SG_THREADS", "16")))


def load(path, device):
    ck = read_checkpoint(path)
    defaults = {k: v for k, v in vars(build_parser().parse_args([])).items() if not k.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(ck.get("config", {}))}
    charset, _, _, n = initialize_charset(config["dataset"])
    model = build_model(config, charset, n)
    model, _, it, _, _ = load_checkpoint(path, model, config, device=device, checkpoint=ck)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], device)
    return model.to(device).eval(), config, it - 1


def role(text, s, store, query):
    if s == store:
        return "store '?'"
    if s == store + 1:
        return "value char"
    if s < store:
        return "prefix"
    return "between value and '!'"


@torch.no_grad()
def run(path, nb, device):
    model, config, it = load(path, device)
    named = [(f"L{k}", l) for k, l in enumerate(model.linear_layers)] + [("i2h", model.i2h)]
    named = [(n, l) for n, l in named if bool(l.ephemeral_mask.any())]
    order = {id(l): k for k, l in enumerate(model.trained_layers())}
    lr, rho = config["learning_rate"], 1.0 - model.forget_rate
    cap = {}
    hooks = [l.register_forward_hook(lambda m, inp, out, n=n: cap.__setitem__(n, inp[0].detach().double()))
             for n, l in named]
    rows = {n: [] for n, _ in named}
    chk = {n: {0: [], 1: []} for n, _ in named}
    crit = torch.nn.CrossEntropyLoss(reduction="none")
    acc = []
    for texts, onehot in load_heldout_batches(config["dataset"], config["batch_size"], nb, device, "validation"):
        B, T = onehot.shape[0], onehot.shape[1]
        sq = [(t.index("?"), t.index("!")) for t in texts]
        model.start_sequence_wipe()
        hidden = model.initHidden(B)
        X = {n: [] for n, _ in named}
        P = {n: [] for n, _ in named}
        R = {n: [] for n, _ in named}  # raw (ungained) write error norms
        for i in range(T - 1):
            out, hidden = model(model_input(onehot, i, config["input_mode"], config["pe_matrix"]), hidden)
            for b, (s, q) in enumerate(sq):
                if q == i:
                    acc.append(int(out[b].argmax()) == int(onehot[b, i + 1].argmax()))
            for n, l in named:
                X[n].append(cap[n])
            for b, (s, q) in enumerate(sq):
                if q != i:
                    continue
                for n, l in named:
                    M = l.ephemeral_mask.double()
                    c = lr * float(l.fused_plasticity())
                    xq = X[n][q][b]
                    actual = (torch.where(l.ephemeral_mask, l.per_sample_weights[b], 0).double() @ xq)
                    raw = torch.stack([-(P[n][t][b] * (M @ (X[n][t][b] * xq))) for t in range(q)])  # [q, out]
                    exps = np.arange(q, 0, -1)  # q - s for s = 0..q-1
                    terms = {}
                    for off in (0, 1):
                        w = torch.tensor(c * rho ** (exps + off), dtype=torch.float64, device=raw.device)
                        terms[off] = raw * w[:, None]
                        chk[n][off].append(float((terms[off].sum(0) - actual).norm() / actual.norm().clamp_min(1e-30)))
                    pn = torch.stack([P[n][t][b].norm() for t in range(q)]).cpu().numpy()
                    rn = torch.stack([R[n][t][b] for t in range(q)]).cpu().numpy()
                    xn = torch.stack([X[n][t][b].norm() for t in range(q)]).cpu().numpy()
                    cs = torch.stack([torch.nn.functional.cosine_similarity(X[n][t][b], xq, dim=0)
                                      for t in range(q)]).cpu().numpy()
                    rows[n].append({"q": q, "s": s, "pn": pn, "rn": rn, "xn": xn, "cos": cs, "raw": raw.cpu(),
                                    "exps": exps, "c": c, "text": texts[b]})
            _, err = dfa_output_error(out, onehot[:, i + 1], crit)
            fast = model.fast_write_projected(err, config.get("grad_norm_clip", 0))
            projected, _ = model.dfa_step_errors(err, config.get("grad_norm_clip", 0))
            raw = [e if o is None else o for e, o in zip(projected, model._fast_error_overrides(err, projected))]
            by_layer = {id(l): e for l, e in zip(model.trained_layers(), fast)}
            raw_by_layer = {id(l): e for l, e in zip(model.trained_layers(), raw)}
            for n, l in named:
                P[n].append(by_layer[id(l)].double())
                R[n].append(raw_by_layer[id(l)].double().norm(dim=1))
            model.fast_only_dfa_step(err, lr, config["ephemeral_update_clamp"], config.get("grad_norm_clip", 0))
    for h in hooks:
        h.remove()
    gained = bool(getattr(model, "write_gain_power", 0))
    res = {"checkpoint": path, "iteration": it, "episodes": len(acc), "recall": float(np.mean(acc)), "layers": {},
           "write_gain": {"power": float(config.get("write_gain_power", 0.0)),
                          "window": int(config.get("write_gain_window", 256)),
                          "max": float(config.get("write_gain_max", 0.0)),
                          "medians": ({str(k): v for k, v in zip(model.write_gain_layers,
                                                                 model.write_gain_medians().tolist())}
                                      if gained else {})}}
    for n, _ in named:
        off = min((0, 1), key=lambda o: float(np.median(chk[n][o])))
        reconstruction_error = float(np.median(chk[n][off]))
        if not np.isfinite(reconstruction_error) or reconstruction_error > 1e-4:
            raise ValueError(f"{path}: {n} invalid closed-form reconstruction: {reconstruction_error}")
        res["layers"][n] = summarize(rows[n], rho, off, float(np.median(chk[n][off])), float(np.median(chk[n][1 - off])))
    return res


def summarize(rows, rho, off, err_used, err_other):
    rp, rg, rd, rt, lead, vpart, kpart, dpart = [], [], [], [], [], [], [], []
    largest, largest_p, largest_g, share, margin = [], [], [], [], []
    by_role, cos_store, cos_other, xn_ratio = {}, [], [], []
    rp_raw, rgain, gpart, largest_ng, gain_role = [], [], [], [], {}
    for r in rows:
        s, q, pn = r["s"], r["q"], r["pn"]
        w = r["c"] * rho ** (r["exps"] + off)
        tn = (r["raw"] * torch.tensor(w)[:, None]).norm(dim=1).numpy()
        g = tn / np.maximum(w * pn, 1e-300)
        oth = [t for t in range(q) if t != s]
        rp.append(pn[s] / np.median(pn[oth]))
        rg.append(g[s] / np.median(g[oth]))
        rd.append(w[s] / np.median(w[oth]))
        rt.append(tn[s] / np.median(tn[oth]))
        lv = np.log(pn[s]) - np.mean(np.log(pn[oth]))
        lk = np.log(g[s]) - np.mean(np.log(g[oth]))
        ld = np.log(w[s]) - np.mean(np.log(w[oth]))
        lead.append(lv + lk + ld); vpart.append(lv); kpart.append(lk); dpart.append(ld)
        largest.append(int(np.argmax(tn)) == s)
        # counterfactuals: equalize |p| (directions kept) or the key factor g across the episode's writes
        largest_p.append(int(np.argmax(tn * pn.mean() / np.maximum(pn, 1e-300))) == s)
        largest_g.append(int(np.argmax(tn * g.mean() / np.maximum(g, 1e-300))) == s)
        share.append(tn[s] ** 2 / np.sum(tn ** 2))
        margin.append(tn[s] / np.max(tn[oth]))
        cos_store.append(r["cos"][s]); cos_other.append(np.mean(r["cos"][oth]))
        xn_ratio.append(r["xn"][s] / np.median(r["xn"][oth]))
        for t in range(q):
            by_role.setdefault(role(r["text"], t, s, q), []).append(pn[t])
        rn = r["rn"]
        gain = np.where(rn > 0, pn / np.maximum(rn, 1e-300), 1.0)
        rp_raw.append(rn[s] / np.median(rn[oth]))
        rgain.append(gain[s] / np.median(gain[oth]))
        gpart.append(np.log(gain[s]) - np.mean(np.log(gain[oth])))
        largest_ng.append(int(np.argmax(tn / np.maximum(gain, 1e-300))) == s)
        for t in range(q):
            gain_role.setdefault(role(r["text"], t, s, q), []).append(gain[t])
    med = lambda v: float(np.median(v))  # noqa: E731
    lead_m, v_m = float(np.mean(lead)), float(np.mean(vpart))
    return {
        "exponent": "q-s" if off == 0 else "q-s+1", "closed_form_rel_err": err_used, "other_exponent_rel_err": err_other,
        "ratio_p": med(rp), "ratio_g": med(rg), "ratio_decay": med(rd), "ratio_term": med(rt),
        "log_lead_mean": lead_m, "log_value_part": v_m, "log_key_part": float(np.mean(kpart)),
        "log_decay_part": float(np.mean(dpart)), "value_share_of_lead": v_m / lead_m if lead_m else None,
        "value_share_median_ratios": float(np.log(med(rp)) / (np.log(med(rp)) + np.log(med(rg)) + np.log(med(rd)))),
        "store_largest": float(np.mean(largest)), "store_largest_equal_p": float(np.mean(largest_p)),
        "store_largest_equal_key": float(np.mean(largest_g)), "signal_share": med(share),
        "margin_over_best_other": med(margin), "cos_store_query": med(cos_store), "cos_other_query": med(cos_other),
        "xnorm_ratio": med(xn_ratio),
        "p_norm_by_role": {k: {"median": med(v), "n": len(v)} for k, v in sorted(by_role.items())},
        "ratio_p_raw": med(rp_raw), "ratio_gain": med(rgain), "log_gain_part": float(np.mean(gpart)),
        "gain_share_of_lead": float(np.mean(gpart)) / lead_m if lead_m else None,
        "store_largest_no_gain": float(np.mean(largest_ng)),
        "gain_by_role": {k: {"median": med(v), "n": len(v)} for k, v in sorted(gain_role.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for path in a.checkpoints:
        res = run(path, a.batches, a.device)
        name = os.path.basename(path).replace(".pth", "")
        with open(os.path.join(a.out, name + ".json"), "w") as f:
            json.dump(res, f, indent=1)
        print(name, json.dumps({n: {k: v for k, v in d.items() if k not in ("p_norm_by_role", "gain_by_role")}
                                for n, d in res["layers"].items()}))


if __name__ == "__main__":
    main()
