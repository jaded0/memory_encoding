"""Batch BFR: fast/slow read ratio of each trunk layer on held-out replays (agent A, 2026-10-22).

Run from a code directory of main >= b26e88d (it imports train/heldout from the cwd and reads synth_datasets/ there):
    cd <code>; python <this dir>/bfr_ratio.py --checkpoint RUN/checkpoint_00030000.pth --batches 16 --json out.json

Replays held-out episodes with heldout.evaluate_held_out (slow weights frozen, fast entries written by the training DFA step) under
one protocol (default observed). A forward hook on every trunk layer with fast entries splits its pre-activation (bias excluded):
    slow = (W * ~m) x,  raw = (W * m) x  (the stored fast read F x),  eff = output - bias - slow  (what the forward actually added:
    F x for off, scale * F x for running/absrun/rms, LN(F x) for ln).
Per (row, step) with a real target (padding excluded) it keeps |slow|, |raw|, |eff|; it reports medians and p90 of eff/slow and raw/slow,
the same at the step that predicts the first recall target, and the median absolute sizes. Prints one JSON line and optionally writes it."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

from heldout import evaluate_held_out, first_recall_steps, load_heldout_batches, update_mask  # noqa: E402
from train import build_model, build_parser, positional_encoding  # noqa: E402
from utils import initialize_charset, load_checkpoint, read_checkpoint, upgrade_legacy_config  # noqa: E402


def q(t, p):
    return float(torch.quantile(t.float(), p)) if t.numel() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--protocol", default="observed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    checkpoint = read_checkpoint(args.checkpoint)
    defaults = {k: v for k, v in vars(build_parser().parse_args([])).items() if not k.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(checkpoint.get("config", {}))}
    charset, _, _, n_characters = initialize_charset(config["dataset"])
    model = build_model(config, charset, n_characters)
    model, _, next_iter, _, _ = load_checkpoint(args.checkpoint, model, config, device=args.device, checkpoint=checkpoint)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], args.device)
    dataset = config["dataset"]
    batches = load_heldout_batches(dataset, config["batch_size"], args.batches, args.device)
    layers = [(i, l) for i, l in enumerate(model.linear_layers) if bool(l.ephemeral_mask.any())]
    rec = {}

    def hook(layer, inputs, output):
        x = inputs[0].detach()
        w, m = layer.per_sample_weights.detach(), layer.ephemeral_mask
        slow = torch.bmm(w.masked_fill(m.unsqueeze(0), 0), x.unsqueeze(2)).squeeze(2)
        raw = torch.bmm(w.masked_fill(~m.unsqueeze(0), 0), x.unsqueeze(2)).squeeze(2)
        eff = output.detach() - (layer.bias.detach() if layer.bias is not None else 0) - slow
        rec[id(layer)].append(torch.stack([slow.norm(dim=1), raw.norm(dim=1), eff.norm(dim=1)], 1))  # [B, 3]

    handles = [l.register_forward_hook(hook) for _, l in layers]
    acc = {i: [] for i, _ in layers}  # rows of [slow, raw, eff, is_recall_step]
    model.eval()
    with torch.no_grad():
        for texts, onehot in batches:
            for _, l in layers:
                rec[id(l)] = []
            steps = onehot.shape[1] - 1
            evaluate_held_out(model, onehot, update_mask(args.protocol, texts, dataset, onehot), config)
            valid = onehot[:, 1:].sum(-1) > 0  # [B, T]
            first = first_recall_steps(texts, dataset, steps).to(onehot.device)
            is_recall = torch.arange(steps, device=onehot.device)[None, :] == first[:, None]
            for i, l in layers:
                r = torch.stack(rec[id(l)], 1)  # [B, T, 3]
                acc[i].append(torch.cat([r, is_recall[..., None].float()], -1)[valid])
    for h in handles:
        h.remove()
    out = {"checkpoint": args.checkpoint, "iteration": next_iter - 1, "protocol": args.protocol,
           "fast_read_norm": config.get("fast_read_norm", "off"), "episodes": len(batches) * config["batch_size"], "layers": {}}
    for i, _ in layers:
        a = torch.cat(acc[i]).cpu().double()
        s, raw, eff, rc = a[:, 0], a[:, 1], a[:, 2], a[:, 3] > 0
        er, rr = eff / s.clamp_min(1e-30), raw / s.clamp_min(1e-30)
        out["layers"][str(i)] = {
            "n": int(a.shape[0]), "eff_over_slow_med": q(er, .5), "eff_over_slow_p90": q(er, .9),
            "raw_over_slow_med": q(rr, .5), "raw_over_slow_p90": q(rr, .9),
            "eff_over_slow_recall_med": q(er[rc], .5), "raw_over_slow_recall_med": q(rr[rc], .5),
            "slow_med": q(s, .5), "eff_med": q(eff, .5), "raw_med": q(raw, .5),
            "slow_recall_med": q(s[rc], .5), "eff_recall_med": q(eff[rc], .5)}
    print(json.dumps(out))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
