"""Standalone SP1 online training and paired frozen-slow evaluation.

No dataset is cached for training. Iterations count complete *rows*, not streams
or characters; --train_steps is an absolute target on resume. Query interventions
occur before the entire query row (including its filler prefix), not at its answer.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from ephemeral_model import dfa_output_error
from kv_tasks import CHARSET
from reproducibility import capture_rng_state, restore_rng_state, seed_everything
from spacing_task import (SpacingBatchStream, classify_answer, generate_split,
                          minimum_length)
from utils import model_input


DEFAULT_GRID = (1, 2, 4, 8, 16, 32, 64)


def integer_list(text):
    values = tuple(int(x) for x in text.split(","))
    if not values or any(x < 1 for x in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected distinct positive comma-separated integers")
    return values


def band_indices(text):
    try:
        indices = tuple(int(x) for x in text.split(","))
        if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
            raise ValueError()
        return indices
    except ValueError:
        raise argparse.ArgumentTypeError("expected distinct nonnegative comma-separated band indices") from None


def forget_overrides(text):
    try:
        pairs = [item.split(":") for item in text.split(",")]
        rates = {int(band): float(rate) for band, rate in pairs}
        if len(rates) != len(pairs) or any(b < 0 or not math.isfinite(r) or not 0 <= r <= 1
                                         for b, r in rates.items()):
            raise ValueError()
        return rates
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("expected distinct BAND:RATE entries; bands >= 0, rates in [0,1]") from None


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("shared", "separate", "flat", "oneband"), default="shared")
    p.add_argument("--bands", "--plasticity_bands", dest="bands",
                   default="0.05:30000:0.03,0.05:3000:0.001",
                   help="fraction:alpha:forget per band (same syntax as train.py)")
    p.add_argument("--hidden_size", "--n_hidden", dest="hidden_size", type=int, default=64)
    p.add_argument("--layers", "--n_layers", dest="layers", type=int, default=2)
    p.add_argument("--fast_layers", default="", help="existing trunk-only fast layer selector")
    p.add_argument("--input_mode", choices=("last_one", "last_two"), default="last_one")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--plasticity", type=float, default=30000)
    p.add_argument("--forget_rate", type=float, default=0.03)
    p.add_argument("--weight_clamp", type=float, default=0)
    p.add_argument("--ephemeral_update_clamp", type=float, default=0)
    p.add_argument("--fast_weight_clamp", type=float, default=0)
    p.add_argument("--fused_update", action="store_true", help="optional CUDA-only torch.compile DFA updates")
    p.add_argument("--train_steps", type=int, default=0, help="absolute row-batch target, including resumed rows")
    p.add_argument("--train_intervals", type=integer_list, default=DEFAULT_GRID)
    p.add_argument("--train_delays", default="0,1,4", help="nonnegative comma-separated sequence counts")
    p.add_argument("--eval_grid", type=integer_list, default=DEFAULT_GRID)
    p.add_argument("--eval_delay", type=int, default=4)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--n_keys", type=int, default=4)
    p.add_argument("--horizon", type=int, default=None, help="training row count (default: shortest fitting train schedules)")
    p.add_argument("--eval_horizon", type=int, default=None, help="common evaluation row count (default: shortest fitting eval grid)")
    p.add_argument("--eval_streams", type=int, default=64)
    p.add_argument("--eval_protocols", nargs="+", default=None,
                   choices=("no_wipe", "wipe_band0", "wipe_band1", "wipe_all"))
    p.add_argument("--eval_wipe_bands", type=band_indices, default=None,
                   help="append a custom protocol wiping these band indices before the query row")
    p.add_argument("--forget_rate_override", type=forget_overrides, default=None,
                   help="inference-only BAND:RATE,...; requires --eval_only and --forget_multipliers 1")
    p.add_argument("--forget_multipliers", default="0.3333333333333333,1,3",
                   help="inference-only band-0 forgetting-rate multipliers")
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", help="save/resume destination (atomic replacement)")
    p.add_argument("--load", help="load SP1 checkpoint, including exact stream and RNG state")
    p.add_argument("--json", default="spacing_results.json")
    p.add_argument("--seed", type=int, default=91105)
    p.add_argument("--eval_seed", type=int, default=None, help="default: training seed; heldout split is disjoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def make_config(args):
    from train import build_parser as train_parser
    config = vars(train_parser().parse_args([]))
    delays = tuple(int(x) for x in args.train_delays.split(","))
    if not delays or min(delays) < 0 or args.eval_delay < 0:
        raise ValueError("delays must be nonnegative")
    if args.train_steps < 0 or args.eval_streams < 1 or args.batch_size < 1:
        raise ValueError("train_steps >= 0, eval_streams and batch_size >= 1 required")
    if min(args.eval_every, args.save_every, args.log_every) < 0:
        raise ValueError("progress/evaluation/checkpoint periods must be nonnegative")
    horizon = max(minimum_length(i, d, args.repetitions)
                  for i in args.train_intervals for d in delays)
    if args.horizon is not None:
        if args.horizon < horizon:
            raise ValueError(f"horizon must be >= {horizon} for all schedules")
        horizon = args.horizon
    config.update(model_type="ephemeral", updater="dfa", input_mode=args.input_mode,
                  enable_recurrence=False, fast_layers=args.fast_layers,
                  positional_encoding_dim=0, n_hidden=args.hidden_size, n_layers=args.layers,
                  batch_size=args.batch_size, learning_rate=args.learning_rate,
                  plasticity=args.plasticity, forget_rate=args.forget_rate,
                  ephemeral_fraction=0.1, plasticity_bands=args.bands,
                  band_feedback="separate" if args.arm == "separate" else "shared",
                  write_gain_power=-1.0 if args.arm == "flat" else 0.0,
                  weight_clamp=args.weight_clamp, ephemeral_update_clamp=args.ephemeral_update_clamp,
                  fast_weight_clamp=args.fast_weight_clamp,
                  wipe_every=horizon, seed=args.seed, spacing_arm=args.arm,
                  spacing_horizon=horizon, spacing_train_intervals=tuple(args.train_intervals),
                  spacing_train_delays=delays, spacing_repetitions=args.repetitions,
                  spacing_n_keys=args.n_keys)
    if args.arm == "oneband":
        config["plasticity_bands"] = f"0.1:{args.plasticity}:{args.forget_rate}"
    config["criterion"] = torch.nn.CrossEntropyLoss(reduction="none")
    config["pe_matrix"] = None
    return config


def collate(rows, device):
    lookup = {char: i for i, char in enumerate(CHARSET)}
    ids = torch.tensor([[lookup[c] for c in row["text"]] for row in rows], device=device)
    return ids, F.one_hot(ids, len(CHARSET)).float()


def make_stream(config):
    return SpacingBatchStream(config["batch_size"], seed=config["seed"],
                              intervals=config["spacing_train_intervals"],
                              delays=config["spacing_train_delays"], length=config["spacing_horizon"],
                              repetitions=config["spacing_repetitions"], n_keys=config["spacing_n_keys"])


def _band_count(config):
    bands = config.get("plasticity_bands", "")
    return len(bands.split(",")) if isinstance(bands, str) and bands else len(bands) if bands else 1


def _set_forgetting(model, config, multiplier, overrides=None):
    """Model API adapter: inference-only band zero, never scalar training rate."""
    bands = config["plasticity_bands"]
    rate = float(bands.split(",")[0].split(":")[2]) if isinstance(bands, str) else bands[0][2]
    rates = {0: rate * multiplier}
    rates.update(overrides or {})
    return model.band_forget_override(rates)


@torch.no_grad()
def _eval_row(model, onehot, config):
    # Deliberately do NOT start_sequence_wipe here: that averages slow copies.
    # Slow consolidation happens once on the private evaluation model, below.
    hidden = model.initHidden(onehot.shape[0])
    for step in range(onehot.shape[1] - 1):
        output, hidden = model(model_input(onehot, step, config["input_mode"], config["pe_matrix"]), hidden)
        loss, error = dfa_output_error(output, onehot[:, step + 1], config["criterion"])
        pred = output.argmax(1)  # score before the target writes
        model.fast_only_dfa_step(error, config["learning_rate"], config["ephemeral_update_clamp"],
                                 config.get("grad_norm_clip", 0))
    return pred, loss


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def _model_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def configure_runtime(model, args):
    if args.fused_update:
        if next(model.parameters()).device.type != "cuda":
            raise ValueError("--fused_update requires a CUDA model")
        model.enable_fused_update(compile=True)


def evaluate(model, config, args):
    """Paired cells, all target writes enabled, frozen slow weights.

    Uses a private model: training weights, mode, fast state, write-gain history
    and RNG are untouched. All cells share stream IDs/bindings and query position.
    Tail batches are padded, but only requested streams contribute to metrics.
    """
    started = time.monotonic()
    bands = _band_count(config)
    protocols = args.eval_protocols or (["no_wipe", "wipe_all"] if bands == 1 else
                                       ["no_wipe", "wipe_band0", "wipe_band1"])
    if bands == 1 and any(p not in ("no_wipe", "wipe_all") for p in protocols):
        raise ValueError("one-band control supports only no_wipe and wipe_all")
    if bands < 2 and "wipe_band1" in protocols:
        raise ValueError("wipe_band1 requires at least two bands")
    custom_protocol = None
    if args.eval_wipe_bands is not None:
        if any(index >= bands for index in args.eval_wipe_bands):
            raise ValueError("--eval_wipe_bands index outside configured bands")
        custom_protocol = "wipe_bands_" + "_".join(map(str, args.eval_wipe_bands))
        protocols = [*protocols, custom_protocol]
    multipliers = tuple(float(x) for x in args.forget_multipliers.split(","))
    if not multipliers or any(not 0 < x < float("inf") for x in multipliers):
        raise ValueError("forget multipliers must be finite and positive")
    if args.forget_rate_override:
        if multipliers != (1.0,):
            raise ValueError("--forget_rate_override requires --forget_multipliers 1")
        if any(index >= bands for index in args.forget_rate_override):
            raise ValueError("--forget_rate_override index outside configured bands")
    working = copy.deepcopy(model)
    working.eval()
    working.start_sequence_wipe()
    baseline = _model_state(working)
    device = next(working.parameters()).device
    needed = max(minimum_length(i, args.eval_delay, args.repetitions) for i in args.eval_grid)
    length = args.eval_horizon if args.eval_horizon is not None else needed
    if length < needed:
        raise ValueError(f"eval_horizon must be >= {needed}")
    batch_size = config["batch_size"]
    padded_streams = ((args.eval_streams + batch_size - 1) // batch_size) * batch_size
    cost = {"prefix_reuse": True,
            "sequence_rows": padded_streams * len(args.eval_grid) * len(multipliers) *
                             (length - 1 + len(protocols)),
            "sequence_rows_without_reuse": padded_streams * len(args.eval_grid) * len(multipliers) *
                                           length * len(protocols)}
    cost["character_predictions"] = cost["sequence_rows"] * (2 * args.n_keys)
    print(f"eval estimate: {cost['sequence_rows']} stream-rows, "
          f"{cost['character_predictions']} character predictions; "
          f"prefix reuse vs {cost['sequence_rows_without_reuse']} naive stream-rows", flush=True)
    results = []
    for interval in args.eval_grid:
        if minimum_length(interval, args.eval_delay, args.repetitions) > length:
            raise ValueError("evaluation schedule does not fit checkpoint horizon")
        for multiplier in multipliers:
            outcomes_by_protocol = {protocol: [] for protocol in protocols}
            cell_start = time.monotonic()
            for start in range(0, args.eval_streams, batch_size):
                # Each stream group starts with the training checkpoint's history.
                # No query branch can influence a later group's write-gain history.
                working.load_state_dict(baseline)
                count = min(batch_size, args.eval_streams - start)
                rows = generate_split("heldout", count,
                                      seed=args.eval_seed if args.eval_seed is not None else config["seed"],
                                      intervals=(interval,), delays=(args.eval_delay,), length=length,
                                      repetitions=args.repetitions, n_keys=args.n_keys, start_stream=start)
                streams = [rows[b * length:(b + 1) * length] for b in range(count)]
                streams += [streams[-1]] * (batch_size - count)
                # One device transfer, rather than rebuilding character tensors for every row.
                _, hot = collate([row for stream in streams for row in stream], device)
                hot = hot.reshape(batch_size, length, hot.shape[1], hot.shape[2])
                working.wipe_fast_state()
                with _set_forgetting(working, config, multiplier, args.forget_rate_override):
                    for position in range(length - 1):
                        _eval_row(working, hot[:, position], config)
                    prequery = _model_state(working)
                    for protocol in protocols:
                        working.load_state_dict(prequery)
                        if protocol != "no_wipe":
                            if protocol == custom_protocol:
                                working.wipe_bands(args.eval_wipe_bands)
                            elif protocol == "wipe_all":
                                working.wipe_fast_state()
                            else:
                                working.wipe_bands([int(protocol[-1])])
                        pred, losses = _eval_row(working, hot[:, -1], config)
                        for b in range(count):
                            row = streams[b][-1]
                            predicted = CHARSET[int(pred[b])]
                            outcomes_by_protocol[protocol].append({
                                "stream": row["stream"], "context": row["context"],
                                "query": row["query"], "answer": row["answer"],
                                "predicted": predicted, "class": classify_answer(row, predicted),
                                "loss": float(losses[b]), "query_lag_steps": row["query_lag_steps"]})
            for protocol, outcomes in outcomes_by_protocol.items():
                counts = {kind: sum(o["class"] == kind for o in outcomes)
                          for kind in ("correct", "wrong_key", "other")}
                cell = {"interval": interval, "delay": args.eval_delay, "protocol": protocol,
                        "forget_multiplier_band0": multiplier, "streams": len(outcomes),
                        "counts": counts, "accuracy": counts["correct"] / len(outcomes),
                        "query_loss": sum(o["loss"] for o in outcomes) / len(outcomes),
                        "shared_prefix_group_seconds": time.monotonic() - cell_start, "outcomes": outcomes}
                results.append(cell)
                print(f"eval I={interval} D={args.eval_delay} {protocol} f0x={multiplier:g} "
                      f"acc={cell['accuracy']:.4f} group_seconds={cell['shared_prefix_group_seconds']:.2f}", flush=True)
    return {"version": 1, "arm": config["spacing_arm"], "seed": config["seed"],
            "config": json.loads(json.dumps({k: v for k, v in config.items()
                                               if k not in ("criterion", "pe_matrix")})),
            "eval_seed": args.eval_seed if args.eval_seed is not None else config["seed"],
            "eval_wipe_bands": list(args.eval_wipe_bands) if args.eval_wipe_bands is not None else None,
            "forget_rate_override": {str(k): v for k, v in (args.forget_rate_override or {}).items()},
            "horizon": length, "repetitions": args.repetitions, "n_keys": args.n_keys,
            "slow_frozen": True, "fast_writes": True, "wipe_boundary": "before_query_sequence",
            "cost": cost, "seconds": time.monotonic() - started, "cells": results}


def save_checkpoint(path, model, config, state, stream):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    saved_config = {k: v for k, v in config.items() if k not in ("criterion", "pe_matrix")}
    torch.save({"spacing_version": 1, "config": saved_config, "model": _model_state(model),
                "state": state, "stream": stream.state_dict(), "rng": capture_rng_state()}, temp)
    temp.replace(path)
    print(f"saved {path} row={state['sequence_count']}", flush=True)


def load_checkpoint(path, device="cpu"):
    from train import build_model
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("spacing_version") != 1:
        raise ValueError("not an SP1 runner checkpoint")
    config = saved["config"]
    config.update(criterion=torch.nn.CrossEntropyLoss(reduction="none"), pe_matrix=None)
    model = build_model(config, CHARSET, len(CHARSET)).to(device)
    model.load_state_dict(saved["model"])
    stream = make_stream(config)
    stream.load_state_dict(saved["stream"])
    restore_rng_state(saved["rng"])
    return model, config, saved["state"], stream


def main(argv=None):
    from train import build_model, train_batch
    args = build_parser().parse_args(argv)
    if args.forget_rate_override and not args.eval_only:
        raise ValueError("--forget_rate_override requires --eval_only (never changes training)")
    seed_everything(args.seed)
    if args.eval_only and not args.load:
        raise ValueError("--eval_only requires --load")
    if args.load:
        model, config, state, stream = load_checkpoint(args.load, args.device)
        # The checkpoint owns task/model/training settings; eval knobs remain CLI-owned.
        args.repetitions, args.n_keys = config["spacing_repetitions"], config["spacing_n_keys"]
    else:
        config = make_config(args)
        model = build_model(config, CHARSET, len(CHARSET)).to(args.device)
        state = {"sequence_count": 0, "training_instance": 0}
        stream = make_stream(config)
    configure_runtime(model, args)
    config["fused_update"] = args.fused_update
    started = time.monotonic()
    last_eval_step = None
    result = None
    while not args.eval_only and state["sequence_count"] < args.train_steps:
        rows = next(stream)
        ids, hot = collate(rows, args.device)
        _, loss, *_ = train_batch(ids, hot, model, config, state)
        step = state["sequence_count"]
        if args.log_every and step % args.log_every == 0:
            print(f"train row={step}/{args.train_steps} char_steps={state['training_instance']} "
                  f"loss={loss:.5f} elapsed={time.monotonic() - started:.2f}s", flush=True)
        if args.save_every and step % args.save_every == 0 and args.checkpoint:
            save_checkpoint(args.checkpoint, model, config, state, stream)
        if args.eval_every and step % args.eval_every == 0 and not args.skip_eval:
            result = evaluate(model, config, args)
            result["training_rows"] = step
            _atomic_json(args.json, result)
            last_eval_step = step
    if args.checkpoint and not args.eval_only:
        save_checkpoint(args.checkpoint, model, config, state, stream)
    if not args.skip_eval and last_eval_step != state["sequence_count"]:
        result = evaluate(model, config, args)
        result["training_rows"] = state["sequence_count"]
        _atomic_json(args.json, result)
    return result


if __name__ == "__main__":
    main()
