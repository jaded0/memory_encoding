"""Mask load test evaluation: one checkpoint, every load K of the wide-key MQAR family.

For each K in --loads, the first --batches validation batches of mqarw_<K>_q<min(K,8)> are scored with
heldout.py's protocols (strict, no_fast, observed: frozen slow weights, prequential) and with
read-time ablations of the strict protocol: from the step that predicts the first answer on (the
same step for every row of a fixed-K batch), the fast part of the listed trunk layers is zeroed, so
the answers are read only through the other layers' fast weights. Writes happen exactly as in
strict (none from that step on; every fast entry still forgets), so 'all' is strict itself and is
checked against heldout.py's number.

Usage (from the code directory):
  python scratch/fast_structure_2026-10-22/load_eval.py --checkpoint RUN/latest_checkpoint.pth --json out.json
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, ".")
import bench_tasks  # noqa: E402
from ephemeral_model import dfa_output_error  # noqa: E402
from heldout import evaluate_protocols, first_recall_steps, load_heldout_batches  # noqa: E402
from metrics import IntervalMetrics  # noqa: E402
from utils import initialize_charset, load_checkpoint, read_checkpoint, upgrade_legacy_config, model_input  # noqa: E402

LOADS = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 52)
ABLATIONS = {"all": (), "only0": (1, 2), "only1": (0, 2), "only2": (0, 1), "drop0": (0,), "drop2": (2,),
             "none": (0, 1, 2)}


def zero_fast(layer):
    if layer.is_additive:
        layer.fast_state.data.zero_()
    else:
        layer.per_sample_weights.data.masked_fill_(layer.ephemeral_mask.unsqueeze(0), 0)


@torch.no_grad()
def strict_with_ablation(model, texts, onehot, config, dataset, drop):
    """heldout.evaluate_held_out's strict protocol, with the fast part of trunk layers `drop` zeroed
    from the first answer step on. Returns preds, losses [T-1, B]."""
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    batch, steps = onehot.shape[0], onehot.shape[1] - 1
    first = first_recall_steps(texts, dataset, steps)
    assert (first == first[0]).all(), "fixed-K batches share the first answer step"
    first = int(first[0])
    model.start_sequence_wipe()
    hidden = model.initHidden(batch)
    preds, losses = [], []
    for i in range(steps):
        if i == first:
            for index in drop:
                zero_fast(model.linear_layers[index])
        output, hidden = model(model_input(onehot, i, config['input_mode'], config['pe_matrix']), hidden)
        loss, error = dfa_output_error(output, onehot[:, i + 1], criterion)
        preds.append(output.argmax(dim=1))
        losses.append(loss.detach())
        write = 1.0 if i < first else 0.0  # strict: from the first answer on, the fast entries only forget
        model.fast_only_dfa_step(error * write, config['learning_rate'], config['ephemeral_update_clamp'],
                                 config.get('grad_norm_clip', 0))
    return torch.stack(preds), torch.stack(losses)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--loads", type=int, nargs="+", default=list(LOADS))
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--protocols", nargs="+", default=["strict", "no_fast", "observed"])
    parser.add_argument("--ablations", nargs="*", default=list(ABLATIONS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)

    from train import build_model, build_parser, positional_encoding
    checkpoint = read_checkpoint(args.checkpoint)
    defaults = {k: v for k, v in vars(build_parser().parse_args([])).items() if not k.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(checkpoint.get("config", {}))}
    charset, _, _, n_characters = initialize_charset(config["dataset"])
    model = build_model(config, charset, n_characters)
    model, _, next_iter, _, _ = load_checkpoint(args.checkpoint, model, config, device=args.device,
                                                checkpoint=checkpoint)
    if config.get("fused_update") and args.device.startswith("cuda"):
        model.enable_fused_update(compile=True)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], args.device)
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    model.eval()
    out = {"checkpoint": args.checkpoint, "iteration": next_iter - 1, "train_dataset": config["dataset"],
           "fast_structure": config.get("fast_structure", "exclusive"), "plasticity": config["plasticity"],
           "learning_rate": config["learning_rate"], "ephemeral_fraction": config["ephemeral_fraction"],
           "seed": config.get("seed"), "batches": args.batches, "loads": {}}
    for k in args.loads:
        start = time.time()
        dataset = f"mqarw_{k}_q{min(k, 8)}"
        batches = load_heldout_batches(dataset, config["batch_size"], args.batches, args.device)
        result = evaluate_protocols(model, batches, config, dataset, args.protocols)
        for name in args.ablations:
            interval = IntervalMetrics(dataset)
            by_lag, by_query = {}, {}
            for texts, onehot in batches:
                preds, losses = strict_with_ablation(model, texts, onehot, config, dataset, ABLATIONS[name])
                interval.update(texts, onehot, preds, losses)
                if name == "all":  # exact (uncapped) lags and query positions, strict protocol
                    for row, text in enumerate(texts):
                        for j, (answer, source) in enumerate(sorted(bench_tasks.episode(text, dataset)["answers"].items())):
                            hit = int(preds[answer - 1, row].item() == onehot[row, answer].argmax().item())
                            for table, key in ((by_lag, answer - source - 1), (by_query, j)):
                                n, c = table.get(key, (0, 0))
                                table[key] = (n + 1, c + hit)
            model.load_state_dict(saved)
            if name == "all":
                result["strict_by_lag"] = {lag: [n, c] for lag, (n, c) in sorted(by_lag.items())}
                result["strict_by_query"] = {j: [n, c] for j, (n, c) in sorted(by_query.items())}
            summary = interval.summary()
            result[f"read_{name}/recall_acc"] = summary["recall_acc"]
            result[f"read_{name}/recall_loss"] = summary.get("recall_loss")
        if "strict" in args.protocols and "all" in args.ablations:
            gap = abs(result["read_all/recall_acc"] - result["heldout_strict/recall_acc"])
            result["read_all_matches_strict"] = gap < 1e-9
        out["loads"][k] = result
        print(json.dumps({"K": k, "seconds": round(time.time() - start, 1),
                          **{key: round(v, 4) for key, v in result.items()
                             if isinstance(v, float) and key.endswith("recall_acc")}}), flush=True)
    with open(args.json, "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
