"""Replays saved checkpoints through the training step and reports the within-sequence
feedback-loop traces (loop_trace.py), so a blowup can be inspected after the fact.

For each checkpoint the training step itself (train.train_batch, with the checkpoint's own flags:
--fused_update, --slow_update_every, --fast_backward_per_forward, --output_tanh, --layer_norm,
clamps, clip) runs over the same fixed batches (the dataset's validation split in stored order, as
heldout.py), with a LoopTracer attached. The model's state is restored before every batch, so each
batch starts from the checkpoint rather than from the previous replay batch, and nothing is
written back to the checkpoint. A replay uses no random numbers.

    python trace_replay.py --checkpoints runs/control/checkpoint_*.pth [--batches 2] [--out replay.pt]
    python trace_replay.py --checkpoints runs/control        # every checkpoint_*.pth in the directory

--out holds {"meta": ..., "checkpoints": [{"path", "iter", "texts", "batches": [traces, ...]}]};
traces are loop_trace.LoopTracer.finish()'s arrays, one dict per batch (batches can differ in
length). plots/loop_figures.py draws from it.
"""
import argparse
import glob
import os

import torch

from heldout import load_heldout_batches
from loop_trace import LoopTracer, summarize
from utils import initialize_charset, load_checkpoint, read_checkpoint, upgrade_legacy_config


def expand_checkpoints(patterns):
    """Paths, globs and directories (every checkpoint_*.pth in them) as a sorted list, in order
    of the iteration stored in the file name where there is one."""
    paths = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            paths += sorted(glob.glob(os.path.join(pattern, "checkpoint_*.pth")))
        else:
            paths += sorted(glob.glob(pattern)) or [pattern]
    return paths


def load_model(path, device, fused_update=None):
    """(model, config, state, iteration) of a checkpoint, built as heldout.py builds it.
    fused_update True/False overrides the checkpoint's --fused_update (None keeps it)."""
    from train import build_model, build_parser, positional_encoding  # train.py imports heldout
    checkpoint = read_checkpoint(path)
    defaults = {key: value for key, value in vars(build_parser().parse_args([])).items()
                if not key.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(checkpoint.get("config", {}))}
    charset, _, _, n_characters = initialize_charset(config["dataset"])
    model = build_model(config, charset, n_characters)
    model, _, next_iter, main_state, _ = load_checkpoint(path, model, config, device=device, checkpoint=checkpoint)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], device)
    config["criterion"] = torch.nn.CrossEntropyLoss(reduction="none")
    use_fused = config.get("fused_update", False) if fused_update is None else fused_update
    if use_fused:
        compile_ok = torch.device(device).type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7
        model.enable_fused_update(compile=compile_ok)
    return model, config, dict(main_state), next_iter - 1


def replay_checkpoint(model, config, state, batches):
    """The traces of the training step on each of batches [(texts, onehot)], each from the model's
    current state; the state is restored afterwards. Returns a list of trace dicts."""
    import train as train_module
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    tracer = LoopTracer(model, config["learning_rate"], config["ephemeral_update_clamp"],
                        config.get("grad_norm_clip", 0))
    results = []
    try:
        for _, onehot in batches:
            model.load_state_dict(saved)
            model.pending_slow_steps = 0
            train_module.train_batch(None, onehot, model, config, {**state, "log_norms_now": False}, tracer=tracer)
            results.append(tracer.finish())
    finally:
        model.load_state_dict(saved)
    return results


def mean_summary(per_batch):
    """summarize() of each batch, averaged (the max-type entries take the max)."""
    summaries = [summarize(traces) for traces in per_batch]
    merged = {}
    for key in summaries[0]:
        values = torch.tensor([summary[key] for summary in summaries])
        finite = values[torch.isfinite(values)]
        if not finite.numel():
            merged[key] = float("nan")
        else:
            merged[key] = float(finite.max() if key.endswith("_max") or "_max_" in key else finite.mean())
    merged["loss"] = float(torch.stack([traces["loss"].mean() for traces in per_batch]).mean())
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay checkpoints through the training step and report loop traces.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="paths, globs or directories")
    parser.add_argument("--dataset", default=None, help="default: the first checkpoint's")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batches", type=int, default=2, help="fixed batches per checkpoint (0 = the whole split)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fused_update", default=None, choices=["true", "false"],
                        help="override the checkpoint's --fused_update (default: keep it)")
    parser.add_argument("--out", default="trace_replay.pt")
    args = parser.parse_args(argv)

    paths = expand_checkpoints(args.checkpoints)
    if not paths:
        parser.error("no checkpoints found")
    fused = None if args.fused_update is None else args.fused_update == "true"
    records, batches = [], None
    print(f"{'iter':>9} {'loss':>7} {'gain_med':>9} {'gain_p90':>10} {'frac>1':>7} {'run>1':>6} "
          f"{'fast|F|':>10} {'trunk|x|':>10} {'max_logit':>10} {'F/S drive':>10}")
    for path in paths:
        model, config, state, iteration = load_model(path, args.device, fused)
        if batches is None:
            dataset = args.dataset or config["dataset"]
            batches = load_heldout_batches(dataset, config["batch_size"], args.batches, args.device, args.split)
        per_batch = replay_checkpoint(model, config, state, batches)
        s = mean_summary(per_batch)
        print(f"{iteration:>9} {s['loss']:>7.3f} {s['trace/loop_gain_median']:>9.3f} {s['trace/loop_gain_p90']:>10.2f} "
              f"{s['trace/frac_gain_gt1']:>7.3f} {s['trace/longest_run_gt1']:>6.1f} {s['trace/fast_norm_last']:>10.3g} "
              f"{s['trace/trunk_act_norm_max']:>10.3g} {s['trace/max_logit_max']:>10.3g} "
              f"{s['trace/fast_over_slow_drive_last']:>10.3g}")
        records.append({"path": path, "iter": iteration, "texts": [t for t, _ in batches],
                        "batches": per_batch, "summary": s})
    meta = {"dataset": dataset, "split": args.split, "batches": len(batches), "fused_update": args.fused_update,
            "config": {key: value for key, value in config.items() if key not in ("criterion", "pe_matrix")}}
    torch.save({"meta": meta, "checkpoints": records}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
