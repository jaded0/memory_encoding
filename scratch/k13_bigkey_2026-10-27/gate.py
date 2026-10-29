"""Fail-closed K13bigkey gate. No submission, and no approval without --approve.

Build: gate.py build --pilot-root DIR --runtime-estimate JSON --out manifest.json
Then, only after reviewing evidence, repeat with --approve --approval-reference REF.
Runtime JSON: {"arms": {ARM: {"seconds_per_iter": N, "scoring_seconds": N,
"evidence": "/path/to/pilot/train.log", "basis": "measured",
"explanation": "pilot throughput and scoring"}}} for all five arms. For feature
arms, basis="conservative_estimate" may document pilot throughput times a safe
multiplier (e.g. 3) and a scoring reserve; no unapproved feature-arm training is
required. The core estimate charges all 25 runs at the slowest arm's cost.
Missing evidence or an unexplained estimate blocks confirmation.
--profile deckard3 is the 2026-10-29 addendum (K13bigkey_addendum_2026-10-29.md):
lin/par2/spar only, on deckard, budgeted per arm in lane-hours.
Pilot directories contain analysis.json, learning_curve.json and pilot_run.json.
Curve JSON is {"split":"validation", "points":[{"iteration":5000,
"strict_recall":0.5}, ...]}. pilot_run.json is written by train.sh.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

KS = (4, 8, 16, 32)
ARMS = ("lin", "par2", "par1", "rep2", "spar")
SEEDS = (2718, 3141, 1618, 1414, 1732)
APPROVAL = ('Jaden chat 2026-10-27: try the higher-dimensional-key task; '
            'evaluate order two vs expand-and-sparsify')
APPROVAL_DECKARD3 = ('Jaden chat 2026-10-29: option 1, lin/par2/spar x 5 seeds x 195k '
                     'on deckard after B2fu-A')
# The deckard pilots ran the full 200k under explicit authorization (K13B_PILOT_TIMEOUT).
PROFILES = dict(
    a100=dict(arms=ARMS, pilot_wall=4510, approval=APPROVAL),
    deckard3=dict(arms=("lin", "par2", "spar"), pilot_wall=86400, approval=APPROVAL_DECKARD3,
                  lane_hour_cap=504, run_wall_cap=72 * 3600),
)
RECIPE = dict(model="KRC1", hidden_size=1024, num_layers=3, batch_size=16,
              input_mode="last_two", updater="dfa", optimizer="sgd",
              learning_rate=1e-4, plasticity=3e4, ephemeral_fraction=.1,
              forget_rate=.01, recurrence=False, residual=False,
              positional_encoding_dim=0, clamps=False, layers=[0, 1, 2],
              temperature=4, projection_dim=16, sparse_m=273, sparse_k=17,
              fixed_projection=True)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def number(value, name, low=0, high=float("inf")):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and low <= value <= high, f"invalid {name}")
    return value


def load(path):
    return json.loads(Path(path).read_text())


def evidence(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def stable_budget(curve, final_iteration):
    require(curve.get("split") == "validation", "curve must use validation")
    points = curve["points"]
    require(len(points) >= 2, "need two curve points")
    previous = 0
    for point in points:
        it = number(point["iteration"], "curve iteration", 5000, 200000)
        require(it == int(it) and it % 5000 == 0 and it > previous,
                "curve must contain ordered distinct 5000-step checkpoints")
        number(point["strict_recall"], "curve recall", 0, 1)
        previous = it
    require(previous <= final_iteration <= previous + 5000,
            "curve does not reach final checkpoint")
    final = points[-1]["strict_recall"]
    require(abs(final - points[-2]["strict_recall"]) <= .03 + 1e-12,
            "unstable final two curve points")
    return next(p["iteration"] for p in points
                if abs(p["strict_recall"] - final) <= .02 + 1e-12)


def pooled_pr(layer):
    # Never substitute the median per-episode PR for the pooled distinct-key PR.
    if "pooled_items" in layer:
        return number(layer["pooled_items"]["key_pr"], "pooled key PR")
    return number(layer["items"]["key_pr_pooled"], "pooled key PR")


def candidate(analysis, curve, run, k, pilot_wall=4510):
    dataset = f"mqar2_{k}_q4"
    require(analysis.get("split") == "validation", "pilot cannot select on test")
    require(run.get("dataset") == dataset and run.get("arm") == "lin"
            and run.get("seed") == 2718 and run.get("requested_iterations") == 200000,
            "not an approved lin pilot")
    require(run.get("recipe") == RECIPE, "pilot recipe differs from preregistration")
    require(run.get("status") in ("completed", "timeout"), "failed/unfinished pilot")
    require(run.get("train_exit") == (0 if run["status"] == "completed" else 124),
            "pilot has no valid TRAIN EXIT status")
    number(run["training_seconds"], "training walltime", 0, pilot_wall)
    require(analysis.get("code_commit") == run.get("code_commit") and bool(run.get("code_commit")),
            "analysis/pilot commit mismatch")
    results = [r for r in analysis["results"] if r["dataset"] == dataset]
    require(len(results) == 1, "missing/ambiguous dataset analysis")
    result = results[0]
    number(result["episodes"], "heldout episode count", 128)
    acc = result["acc"]
    strict = number(acc["strict"], "strict recall", 0, 1)
    first = number(acc["strict_first"], "first-answer recall", 0, 1)
    post = number(acc["post_store"], "post-store recall", 0, 1)
    number(acc["store_only"], "store-only diagnostic", 0, 1)
    number(acc["paired_gap"], "paired-gap diagnostic", 0, 1)
    for protocol in ("observed", "no_fast"):
        number(acc[protocol], f"{protocol} diagnostic", 0, 1)
    for layer in ("L0", "L1"):
        wrong = result["layers"][layer]["items"]["wrong_cosines"]
        number(wrong["count"], "wrong-key cosine count", 1)
        require(bool(wrong["quantiles"]), "missing signed wrong-key quantiles")
        for value in wrong["quantiles"].values():
            number(value, "signed wrong-key cosine quantile", -1, 1)
        number(wrong["abs_median"], "absolute wrong-key cosine", 0, 1)
        # Analyzer may expose absolute quantiles beside the signed distribution.
        absolute = wrong.get("abs_quantiles") or result["layers"][layer]["items"].get("wrong_abs_cosines", {}).get("quantiles")
        require(bool(absolute), "missing absolute wrong-key quantiles")
        for value in absolute.values():
            number(value, "absolute wrong-key cosine quantile", 0, 1)
    checks = dict(strict_range=.2 <= strict <= .85, first_range=.2 <= first <= .85,
                  post_improvement=post - first >= .1 - 1e-12,
                  error_recovered=post - first >= .8 * (1 - first) - 1e-12,
                  L0_PR=pooled_pr(result["layers"]["L0"]) >= 4 - 1e-10,
                  L1_PR=pooled_pr(result["layers"]["L1"]) >= 2 - 1e-10)
    try:
        budget = stable_budget(curve, analysis["iteration"])
    except ValueError as exc:
        budget = None
        checks["stable_budget"] = False
        checks["stability_reason"] = str(exc)
    else:
        checks["stable_budget"] = True
    return dict(k=k, checks=checks, qualifies=all(v is True for v in checks.values()),
                n_iters=budget, strict_recall=strict, first_answer_recall=first,
                post_store_recall=post, actual_iterations=analysis["iteration"],
                status=run["status"])


def estimate_runtime(runtime, iterations, profile="a100"):
    arms = PROFILES[profile]["arms"]
    if profile == "a100":
        require(set(runtime["arms"]) == set(ARMS), "need documented runtime for all five arms")
    require(set(arms) <= set(runtime["arms"]), "need documented runtime for every authorized arm")
    costs, files = [], []
    for arm in arms:
        item = runtime["arms"][arm]
        require(item.get("basis") in ("measured", "conservative_estimate")
                and isinstance(item.get("explanation"), str) and bool(item["explanation"].strip()),
                f"{arm} runtime estimate must document its basis")
        rate = number(item["seconds_per_iter"], f"{arm} seconds/iter", 1e-12)
        score = number(item["scoring_seconds"], f"{arm} scoring seconds", 1e-12)
        files.append(evidence(item["evidence"]))
        costs.append(iterations * rate + score)
    if profile == "deckard3":
        # Arm-specific lane-hours (two lanes per A6000), not the worst-arm envelope.
        hours = len(SEEDS) * sum(costs) / 3600
        require(hours <= PROFILES[profile]["lane_hour_cap"],
                "core 15-run estimate exceeds 504 deckard lane-hours")
        return hours, files
    hours = 25 * max(costs) / 3600
    require(hours <= 40, "core 25-run training + scoring estimate exceeds 40 A100-hours")
    return hours, files


def allocation_budget(runtime, iterations, profile="a100"):
    # Include a 20% margin plus a full bounded diagnostic window and setup time.
    arms = [runtime["arms"][a] for a in PROFILES[profile]["arms"]]
    train_seconds = max(iterations * r["seconds_per_iter"] for r in arms)
    scoring_seconds = max(r["scoring_seconds"] for r in arms)
    reserve = max(600, math.ceil(1.2 * scoring_seconds))
    wall = math.ceil((1.2 * train_seconds + reserve + 60) / 60) * 60
    if profile == "deckard3":
        # A per-run timeout only; the lane-hour estimate is the budget.
        require(wall <= PROFILES[profile]["run_wall_cap"], "per-run deckard wall exceeds 72 hours")
        return dict(per_run_wall_seconds=wall, training_timeout_seconds=wall - reserve - 60,
                    scoring_timeout_seconds=reserve)
    require(25 * wall <= 40 * 3600, "core allocated walltime exceeds 40 A100-hours")
    return dict(per_run_wall_seconds=wall, training_timeout_seconds=wall - reserve - 60,
                scoring_timeout_seconds=reserve, allocated_a100_hours=25 * wall / 3600)


def build(root, runtime_path, approve=False, approval_reference=None, profile="a100"):
    require(profile in PROFILES, "unknown gate profile")
    P = PROFILES[profile]
    records, files = [], []
    for k in KS:
        d = Path(root) / f"mqar2_{k}_q4-lin-s2718"
        paths = [d / n for n in ("analysis.json", "learning_curve.json", "pilot_run.json")]
        files.extend(evidence(p) for p in paths)
        analysis, curve, run = (load(p) for p in paths)
        log = d / "train.log"
        files.append(evidence(log))
        markers = [line for line in log.read_text().splitlines() if line.startswith("TRAIN EXIT ")]
        require(markers == [f"TRAIN EXIT {run['train_exit']}"], "missing/ambiguous TRAIN EXIT marker")
        records.append(candidate(analysis, curve, run, k, P["pilot_wall"]))
    selected = next((r for r in records if r["qualifies"]), None)
    require(selected is not None, "no K passes the frozen regime and stability gates")
    files.append(evidence(runtime_path))
    hours, runtime_files = estimate_runtime(load(runtime_path), selected["n_iters"], profile)
    allocation = allocation_budget(load(runtime_path), selected["n_iters"], profile)
    files.extend(runtime_files)
    require(not approve or approval_reference == P["approval"], "explicit frozen approval reference required")
    if profile != "a100":
        return dict(**allocation, schema_version=1, profile=profile, regime_verified=True,
                    launch_approved=bool(approve),
                    approval_reference=approval_reference if approve else None,
                    dataset=f"mqar2_{selected['k']}_q4", n_iters=selected["n_iters"],
                    arms=list(P["arms"]), seeds=list(SEEDS), core_runs=len(SEEDS) * len(P["arms"]),
                    recipe=RECIPE, estimated_lane_hours=hours, includes_scoring=True,
                    pilot_evidence=files, pilot_results=records,
                    pilot_root=str(Path(root).resolve()),
                    runtime_estimate=str(Path(runtime_path).resolve()))
    return dict(**allocation, schema_version=1, regime_verified=True, launch_approved=bool(approve),
                approval_reference=approval_reference if approve else None,
                dataset=f"mqar2_{selected['k']}_q4", n_iters=selected["n_iters"],
                arms=list(ARMS), seeds=list(SEEDS), core_runs=25, recipe=RECIPE,
                estimated_a100_hours=hours, includes_scoring=True,
                most_expensive_arm_included=True, pilot_evidence=files,
                pilot_results=records, pilot_root=str(Path(root).resolve()),
                runtime_estimate=str(Path(runtime_path).resolve()))


def validate_manifest(manifest, dataset=None, iterations=None, arm=None, seed=None):
    require(manifest.get("regime_verified") is True and manifest.get("launch_approved") is True,
            "confirmation is not explicitly approved")
    profile = manifest.get("profile", "a100")
    require(profile in PROFILES, "unknown gate profile")
    approval = PROFILES[profile]["approval"]
    require(manifest.get("approval_reference") == approval, "invalid approval reference")
    for item in manifest["pilot_evidence"]:
        require(evidence(item["path"]) == item, "evidence changed since gate evaluation")
    rebuilt = build(manifest["pilot_root"], manifest["runtime_estimate"], True, approval, profile)
    require(manifest == rebuilt, "manifest does not match frozen gate/evidence")
    require(dataset is None or dataset == manifest["dataset"], "dataset not authorized")
    require(iterations is None or iterations == manifest["n_iters"], "iterations not authorized")
    require(arm is None or arm in PROFILES[profile]["arms"], "arm not authorized")
    require(seed is None or seed in SEEDS, "seed not authorized")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pilot-root", required=True)
    b.add_argument("--runtime-estimate", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--approve", action="store_true")
    b.add_argument("--approval-reference")
    b.add_argument("--profile", choices=sorted(PROFILES), default="a100")
    v = sub.add_parser("validate")
    v.add_argument("manifest")
    v.add_argument("--dataset")
    v.add_argument("--iterations", type=int)
    v.add_argument("--arm")
    v.add_argument("--seed", type=int)
    args = p.parse_args(argv)
    try:
        if args.command == "validate":
            validate_manifest(load(args.manifest), args.dataset, args.iterations, args.arm, args.seed)
        else:
            result = build(args.pilot_root, args.runtime_estimate, args.approve, args.approval_reference,
                           args.profile)
            Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        # Replace stale approval on a failed rebuild; never leave an old launchable file.
        if args.command == "build":
            Path(args.out).write_text(json.dumps(dict(regime_verified=False, launch_approved=False,
                                                     blocked_reason=str(exc)), indent=2) + "\n")
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
