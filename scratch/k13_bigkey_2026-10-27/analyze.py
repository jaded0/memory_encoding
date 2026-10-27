"""Frozen pilot diagnostics for two-letter MQAR; run from the repository root.

Example: python scratch/k13_bigkey_2026-10-27/analyze.py CKPT --out result.json
Uses stored validation batches by default; --generated draws hash-held-out cores.
Suppression and paired gap controls score ONLY the first answer: no teacher answer
has been seen. Post-store retains all writes through the desired store (inclusive),
store-only retains that write alone. Both retain inputs and the usual forgetting.
The paired gap retains only the first queried table item, not all Q queried items;
this is distinct from the standalone mqar2gap_K_qQ dataset.

PR and item wrong-cosines use table value-store steps, not character writes. The
gate pools unit-key second moments over episodes, each table counted ONCE (not Q
times). Per-episode PR remains diagnostic and is bounded by K; pooled PR is not.
All-written-step
diagnostics and the reused hypothetical sparse-kernel screen are also reported.
No hypothetical kernel changes a replay or authorizes training.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

import bench_tasks
from ephemeral_model import order2_projection
from heldout import evaluate_held_out, load_heldout_batches, update_mask
from utils import initialize_charset


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


o2 = module("bigkey_o2", "scratch/order2_2026-10-22/o2_keys.py")
sparse = module("bigkey_sparse", "scratch/k13_crosstalk_2026-10-27/sparse_screen.py")


def metadata(text, dataset):
    """Use exact source indices, never capped recall-lag buckets or key search."""
    parsed = bench_tasks.parse_name(dataset)
    if parsed is None or parsed[0] not in ("mqar2", "mqar2gap"):
        raise ValueError(f"not a two-letter MQAR dataset: {dataset}")
    answers = bench_tasks.episode(text, dataset)["answers"]
    if not answers:
        raise ValueError("episode has no answers")
    return sorted(answers.items())


def paired_gap(text, dataset):
    _, source = metadata(text, dataset)[0]
    boundary = text.index("?")
    table = ["."] * boundary
    table[source - 2:source + 1] = text[source - 2:source + 1]
    return "".join(table) + text[boundary:]


def suppression_mask(texts, dataset, onehot, mode):
    mask = update_mask("strict", texts, dataset, onehot)
    steps = torch.arange(mask.shape[1], device=mask.device)[None]
    stores = torch.tensor([metadata(t, dataset)[0][1] - 1 for t in texts],
                          device=mask.device)[:, None]
    if mode == "post_store":
        return mask & (steps <= stores)
    if mode == "store_only":
        return mask & (steps == stores)
    raise ValueError(mode)


def encode(texts, charset, device):
    # Zero padding, not a space-character write.
    result = torch.zeros(len(texts), max(map(len, texts)), len(charset), device=device)
    for row, text in enumerate(texts):
        ids = torch.tensor([charset.index(c) for c in text], device=device)
        result[row, :len(text)] = F.one_hot(ids, len(charset)).float()
    return result


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0, "mean": None, "abs_median": None, "quantiles": {}, "abs_quantiles": {}}
    return {"count": int(values.size), "mean": float(values.mean()),
            "abs_median": float(np.median(abs(values))),
            "quantiles": dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                                  np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist())),
            "abs_quantiles": dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                                      np.quantile(abs(values), [0, .05, .25, .5, .75, .95, 1]).tolist()))}


def geometry(xs, xq, si):
    unit = F.normalize(xs.double(), dim=1)
    gram = unit @ unit.T
    cos = F.cosine_similarity(xs.double(), xq[None].double(), dim=1).cpu().tolist()
    return {"key_pr": float(gram.trace().square() / gram.square().sum().clamp_min(1e-30)),
            "store_cosine": cos[si], "wrong_cosines": cos[:si] + cos[si + 1:]}


def accumulate_pooled(pooled, layer, xs):
    """Sum uncentered unit-key xx^T; call once per episode, never per query."""
    unit = F.normalize(xs.double(), dim=1)
    covariance = unit.T @ unit
    gram = unit @ unit.T
    pr = float(gram.trace().square() / gram.square().sum().clamp_min(1e-30))
    if layer not in pooled:
        pooled[layer] = {"second_moment_sum": torch.zeros_like(covariance),
                         "stores": 0, "per_episode_pr": []}
    pooled[layer]["second_moment_sum"].add_(covariance)
    pooled[layer]["stores"] += len(xs)
    pooled[layer]["per_episode_pr"].append(pr)


def pooled_pr(moment):
    return float(moment.trace().square() / moment.square().sum().clamp_min(1e-30))


def summarize_geometry(records, pooled):
    result = {}
    for layer, rows in records.items():
        result[layer] = {}
        for population in ("items", "written"):
            rs = [r[population] for r in rows]
            result[layer][population] = {
                "targets": len(rs),
                "key_pr": float(np.median([r["key_pr"] for r in rs])),
                "key_pr_distribution": distribution([r["key_pr"] for r in rs]),
                "store_cosines": distribution([r["store_cosine"] for r in rs]),
                "wrong_cosines": distribution([v for r in rs for v in r["wrong_cosines"]])}
        p = pooled[layer]
        item = result[layer]["items"]
        item["key_pr_pooled"] = pooled_pr(p["second_moment_sum"])
        item["key_pr"] = item["key_pr_pooled"]  # Explicit compatibility alias: gate population.
        item["key_pr_per_episode_median"] = float(np.median(p["per_episode_pr"]))
        item["key_pr_per_episode_distribution"] = distribution(p["per_episode_pr"])
        item["key_pr_distribution"] = item["key_pr_per_episode_distribution"]
        item["pooled_episodes"] = len(p["per_episode_pr"])
        item["pooled_stores"] = p["stores"]
        item["key_pr_definition"] = "tr(S)^2 / tr(S^2), S=sum unit(x)unit(x)^T over table stores once per episode"
    return result


def score(preds, onehot, texts, dataset):
    all_hits, first_hits = [], []
    for b, text in enumerate(texts):
        hits = [int(preds[target - 1, b]) == int(onehot[b, target].argmax())
                for target, _ in metadata(text, dataset)]
        all_hits.extend(hits)
        first_hits.append(hits[0])
    return all_hits, first_hits


def exact_score(preds, onehot, texts, dataset):
    return [all(int(preds[t - 1, b]) == int(onehot[b, t].argmax())
                for t, _ in metadata(text, dataset)) for b, text in enumerate(texts)]


@torch.no_grad()
def strict_replay(model, config, texts, onehot, dataset, observer, records, pooled=None, row_offset=0):
    """Delegate writes/forgetting/extra passes to the canonical heldout evaluator."""
    layers = list(model.linear_layers)
    captures = {k: [] for k in range(min(3, len(layers)))}
    calls = dict.fromkeys(captures, 0)
    repeats = model.fast_iterations

    def hook(k, inp):
        if calls[k] % repeats == 0:
            captures[k].append(inp[0].detach().double().clone())
        calls[k] += 1

    hooks = [layers[k].register_forward_pre_hook(lambda m, inp, k=k: hook(k, inp)) for k in captures]
    mask = update_mask("strict", texts, dataset, onehot)
    mask &= onehot[:, 1:].sum(-1) > 0
    try:
        preds, _ = evaluate_held_out(model, onehot, mask, config)
    finally:
        for handle in hooks:
            handle.remove()
    seed = config.get("seed") or 0
    for b, text in enumerate(texts):
        boundary = text.index("?")
        # Slots containing blanks are not memory items (mqar2gap).
        item_stores = [s - 1 for s in range(2, boundary, 3) if text[s] != "."]
        if pooled is not None:
            for k, history in captures.items():
                accumulate_pooled(pooled, f"L{k}", torch.stack([history[t][b] for t in item_stores]))
        for answer, source in metadata(text, dataset):
            q, store = answer - 1, source - 1
            written = [t for t in range(q) if bool(mask[b, t])]
            if store not in written or store not in item_stores:
                raise ValueError("desired value-store was not written")
            for k, history in captures.items():
                xq = history[q][b]
                xs = torch.stack([history[t][b] for t in written])
                si = written.index(store)
                item_xs = torch.stack([history[t][b] for t in item_stores])
                records.setdefault(f"L{k}", []).append({
                    "row": row_offset + b, "answer": answer, "source": source,
                    "items": geometry(item_xs, xq, item_stores.index(store)),
                    "written": geometry(xs, xq, si)})
                if observer is not None:
                    proj = order2_projection(layers[k].in_features, 16, seed, k).to(xs)
                    fraction = float(layers[k].ephemeral_mask.float().mean())
                    observer(k, xs, xq, si, [j for j in range(len(written)) if j != si],
                             proj, fraction, seed)
    return preds


def gate(strict, strict_first, post_store, layers, episodes=128):
    """Preregistered pooled-PR gate; evidence only, never launch approval."""
    improvement = post_store - strict_first
    recovered = improvement / (1 - strict_first) if strict_first < 1 else 0.
    checks = {"strict_range": .2 <= strict <= .85,
              "strict_first_range": .2 <= strict_first <= .85,
              "at_least_128_episodes": episodes >= 128,
              "post_store_absolute_improvement": improvement >= .1 - 1e-12,
              "post_store_recovers_80pct_error": recovered >= .8 - 1e-12,
              "L0_item_key_pr_pooled": layers.get("L0", {}).get("items", {}).get("key_pr_pooled", 0) >= 4 - 1e-10,
              "L1_item_key_pr_pooled": layers.get("L1", {}).get("items", {}).get("key_pr_pooled", 0) >= 2 - 1e-10}
    return {"regime_verified": all(checks.values()), "checks": checks,
            "post_store_improvement": improvement, "recoverable_error_fraction": recovered,
            "launch_approved": False}


def summarize_kernels(rows):
    return {layer: {name: {
        "targets": len(rs), "key_ratio": float(np.median([r["ratio"] for r in rs])),
        "selectivity": float(np.median([r["selectivity"] for r in rs])),
        "store_cosines": distribution([r["store_cosine"] for r in rs]),
        "wrong_cosines": distribution([v for r in rs for v in r["wrong_cosines"]])}
        for name, rs in maps.items()} for layer, maps in rows.items()}


@torch.no_grad()
def analyze_dataset(model, config, batches, dataset, kernel_screen=True, raw_records=False):
    observer, kernels = sparse.make_observer(o2, layers=(0, 1, 2))
    records, pooled, episode_texts = {}, {}, []
    hits = {name: [] for name in ("strict", "strict_first", "strict_exact", "observed", "observed_first",
                                  "observed_exact", "no_fast", "no_fast_first", "no_fast_exact",
                                  "post_store", "store_only", "paired_gap")}
    charset = initialize_charset(dataset)[0]
    for texts, onehot in batches:
        preds = strict_replay(model, config, texts, onehot, dataset,
                              observer if kernel_screen else None, records, pooled, len(episode_texts))
        episode_texts.extend(texts)
        all_hits, first_hits = score(preds, onehot, texts, dataset)
        hits["strict"].extend(all_hits)
        hits["strict_first"].extend(first_hits)
        hits["strict_exact"].extend(exact_score(preds, onehot, texts, dataset))
        for protocol in ("observed", "no_fast"):
            pred, _ = evaluate_held_out(model, onehot, update_mask(protocol, texts, dataset, onehot), config)
            all_hits, first_hits = score(pred, onehot, texts, dataset)
            hits[protocol].extend(all_hits)
            hits[protocol + "_first"].extend(first_hits)
            hits[protocol + "_exact"].extend(exact_score(pred, onehot, texts, dataset))
        for mode in ("post_store", "store_only"):
            mask = suppression_mask(texts, dataset, onehot, mode)
            pred, _ = evaluate_held_out(model, onehot, mask, config)
            hits[mode].extend(score(pred, onehot, texts, dataset)[1])
        gaps = encode([paired_gap(t, dataset) for t in texts], charset, onehot.device)
        # Deliberately score with ORIGINAL metadata: later query keys are absent.
        pred, _ = evaluate_held_out(model, gaps, update_mask("strict", texts, dataset, gaps), config)
        hits["paired_gap"].extend(score(pred, onehot, texts, dataset)[1])
    if not hits["strict"]:
        raise ValueError("no evaluation targets")
    acc = {k: float(np.mean(v)) for k, v in hits.items()}
    layers = summarize_geometry(records, pooled)
    result = {"dataset": dataset, "episodes": len(hits["strict_first"]),
              "targets": len(hits["strict"]), "acc": acc, "layers": layers,
              "kernels": summarize_kernels(kernels),
              "gate": gate(acc["strict"], acc["strict_first"], acc["post_store"], layers,
                           len(hits["strict_first"])),
              "key_pr_population": "nonblank table value-store steps (source-1), pooled once per episode",
              "kernel_population": "all written character steps; observer only"}
    if raw_records:
        result.update(records=records, kernel_records=kernels, episode_texts=episode_texts)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--datasets", "--dataset", nargs="+", default=[f"mqar2_{k}_q4" for k in (4, 8, 16, 32)])
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--generated", action="store_true", help="generate split-filtered held-out cores, not disk rows")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--memory-fraction", type=float, default=.30)
    ap.add_argument("--no-kernel-screen", action="store_true")
    ap.add_argument("--raw-records", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.batches < 1:
        ap.error("--batches must be positive (bounded pilot evaluation)")
    torch.set_num_threads(4)
    if str(args.device).startswith("cuda"):
        cuda_index = torch.device(args.device).index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    # Avoid fused kernels on older pilot GPUs; do not modify the checkpoint.
    read = o2.read_checkpoint
    def unfused(path):
        ck = read(path)
        ck["config"]["fused_update"] = False
        return ck
    o2.read_checkpoint = unfused
    start = time.monotonic()
    try:
        model, config, iteration = o2.load(args.checkpoint, args.device)
    finally:
        o2.read_checkpoint = read
    model.check_fast_only_step()
    results = []
    for dataset in args.datasets:
        bench_tasks.parse_name(dataset)
        if initialize_charset(dataset)[0] != initialize_charset(config["dataset"])[0]:
            raise ValueError("checkpoint and evaluation charset/index order differ")
        if args.generated:
            texts = bench_tasks.generate_split(dataset, args.split, args.batches * config["batch_size"], args.seed)
            batches = [(texts[i:i + config["batch_size"]],
                        encode(texts[i:i + config["batch_size"]], initialize_charset(dataset)[0], args.device))
                       for i in range(0, len(texts), config["batch_size"])]
        else:
            batches = load_heldout_batches(dataset, config["batch_size"], args.batches, args.device, args.split)
        results.append(analyze_dataset(model, config, batches, dataset,
                                       not args.no_kernel_screen, args.raw_records))
    output = {"checkpoint": str(Path(args.checkpoint).resolve()), "iteration": iteration,
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "split": args.split, "generated": args.generated, "episode_seed": args.seed,
              "elapsed_seconds": time.monotonic() - start, "results": results,
              "peak_allocated_mb": torch.cuda.max_memory_allocated(args.device) / 2**20
              if str(args.device).startswith("cuda") else None,
              "launch_approved": False}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out), "datasets": [r["dataset"] for r in results],
                      "gates": [r["gate"] for r in results]}), flush=True)


if __name__ == "__main__":
    main()
