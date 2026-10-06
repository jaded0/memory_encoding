#!/usr/bin/env python3
"""Part (a): stability-margin proxy analysis.

Reads the supplied replay/feature/health JSON files and writes:
  * results_a.json
  * results_a_table.md
  * fig_a_proxy.png

Missing measurements are represented as None/null and shown as em dashes in the
Markdown table; the script never interpolates checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stab_margin_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ARMS = ("ST1", "ST2", "ST3", "ST4")
DISPLAY_NAMES = {
    "ST1": "ST1 (LayerNorm)",
    "ST2": "ST2 (clamp 0.3)",
    "ST3": "ST3 (grad clip 30)",
    "ST4": "ST4 (output tanh)",
    "B": "B (plain, seed 3141)",
    "4241": "plain 4241",
    "2718": "plain 2718",
}
TABLE_STAGES = tuple(range(50_000, 600_001, 50_000))
PREDICTION_STAGES = (150_000, 300_000, 450_000, 600_000)


def load_json(name: str) -> Any:
    with (DATA / name).open() as f:
        return json.load(f)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def find_nonfinite(obj: Any, path: str = "$") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(find_nonfinite(value, f"{path}.{key}"))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(find_nonfinite(value, f"{path}[{index}]"))
    elif isinstance(obj, float) and not math.isfinite(obj):
        found.append({"path": path, "value": str(obj)})
    return found


def replay_point(record: dict[str, Any]) -> dict[str, Any]:
    """Extract only requested values from one replay record."""
    svals = record.get("svals")
    s0 = record.get("s0") or {}
    s1 = record.get("s1") or {}
    return {
        "sv2": svals[2][0] if isinstance(svals, list) and len(svals) > 2 and svals[2] else None,
        "sv1": svals[1][0] if isinstance(svals, list) and len(svals) > 1 and svals[1] else None,
        "s0_act_med": s0.get("act_med"),
        "s0_gact_med": s0.get("gact_med"),
        "s0_loss": s0.get("loss"),
        "s1_loss": s1.get("loss"),
    }


def checkpoint_stage(path_key: str, record: dict[str, Any]) -> int:
    """Return checkpoint stage, undoing the feature extractor's +1 step convention."""
    match = re.search(r"_(\d+)\.pth$", path_key)
    if match:
        return int(match.group(1))
    # Fallback retained for schema robustness; current feature rows all match.
    return int(record["iter"]) - 1


def feature_series(raw: dict[str, Any], lineage: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    marker = f"L{lineage}_train" if lineage in {"4241", "2718"} else f"/{lineage}_"
    for key, record in raw.items():
        if marker not in key:
            continue
        out[checkpoint_stage(key, record)] = {
            "sv2": record.get("sv_t2"),
            "sv1": record.get("sv_t1"),
            "s0_act_med": None,
            "s0_gact_med": None,
            "s0_loss": None,
            "s1_loss": None,
        }
    return out


def ratio(series: dict[int, dict[str, Any]], field: str, numerator: int, denominator: int) -> float | None:
    num = series.get(numerator, {}).get(field)
    den = series.get(denominator, {}).get(field)
    if not (finite_number(num) and finite_number(den)) or den == 0:
        return None
    return float(num / den)


def rank_trend(series: dict[int, dict[str, Any]], field: str, start: int, end: int) -> dict[str, Any]:
    pairs = [
        (stage, values.get(field))
        for stage, values in sorted(series.items())
        if start <= stage <= end and finite_number(values.get(field))
    ]
    if len(pairs) < 2:
        return {"rho": None, "pvalue": None, "n": len(pairs), "range": [start, end]}
    stages, values = zip(*pairs)
    result = spearmanr(stages, values)
    return {
        "rho": float(result.statistic) if finite_number(float(result.statistic)) else None,
        "pvalue": float(result.pvalue) if finite_number(float(result.pvalue)) else None,
        "n": len(pairs),
        "range": [start, end],
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or not finite_number(value):
        return "—"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 10_000 or abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.{digits}g}"


def stage_label(stage: int) -> str:
    return f"{stage // 1000}k"


def markdown_table(headers: list[str], rows: Iterable[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def main() -> None:
    raw_inputs = {f"replay_{arm}.json": load_json(f"replay_{arm}.json") for arm in (*ARMS, "B")}
    raw_inputs["feat_local.json"] = load_json("feat_local.json")
    raw_inputs["feat_L.json"] = load_json("feat_L.json")
    raw_inputs["health_rows.json"] = load_json("health_rows.json")

    nonfinite_by_file: dict[str, list[dict[str, Any]]] = {}
    for name, raw in raw_inputs.items():
        findings = find_nonfinite(raw)
        nonfinite_by_file[name] = findings
        if findings:
            print(f"NONFINITE {name}: {len(findings)} value(s)")
            for finding in findings:
                print(f"  {finding['path']} = {finding['value']}")
        else:
            print(f"NONFINITE {name}: none")

    series: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in (*ARMS, "B"):
        series[arm] = {
            int(stage): replay_point(record)
            for stage, record in raw_inputs[f"replay_{arm}.json"].items()
        }
    series["4241"] = feature_series(raw_inputs["feat_L.json"], "4241")
    series["2718"] = feature_series(raw_inputs["feat_L.json"], "2718")

    # Requested stage table: include only checkpoints that actually exist.
    stage_rows: list[dict[str, Any]] = []
    for arm in (*ARMS, "B", "4241", "2718"):
        for stage in TABLE_STAGES:
            if stage in series[arm]:
                stage_rows.append({"arm": arm, "stage": stage, **series[arm][stage]})

    # Ratios and trends. ST2 deliberately stops at 290k for pre-collapse summary.
    endpoints = {
        "ST1": (600_000, 150_000),
        "ST2": (290_000, 150_000),
        "ST3": (600_000, 150_000),
        "ST4": (600_000, 150_000),
        "B": (205_000, 150_000),
        "4241": (150_000, 50_000),
        "2718": (150_000, 50_000),
    }
    summaries: dict[str, Any] = {}
    for arm, (last_stage, base_stage) in endpoints.items():
        summaries[arm] = {
            "ratio_endpoints": {"numerator_stage": last_stage, "denominator_stage": base_stage},
            "sv2_ratio": ratio(series[arm], "sv2", last_stage, base_stage),
            "s0_gact_med_ratio": ratio(series[arm], "s0_gact_med", last_stage, base_stage),
            "spearman_iteration_sv2": rank_trend(series[arm], "sv2", 100_000, last_stage),
            "spearman_iteration_s0_gact_med": rank_trend(series[arm], "s0_gact_med", 100_000, last_stage),
        }

    # Naive power-law/log-log fit: log(edge) = intercept + slope * log(SV2).
    fit_rows = [
        row
        for row in raw_inputs["health_rows.json"]
        if row.get("stage", -1) >= 40_000
        and finite_number(row.get("sv_t2"))
        and finite_number(row.get("edge"))
        and row["sv_t2"] > 0
        and row["edge"] > 0
    ]
    x = np.log(np.asarray([row["sv_t2"] for row in fit_rows], dtype=float))
    y = np.log(np.asarray([row["edge"] for row in fit_rows], dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    r_squared = 1.0 - float(np.sum((y - fitted) ** 2) / np.sum((y - np.mean(y)) ** 2))
    fit_min, fit_max = float(np.exp(x.min())), float(np.exp(x.max()))
    edge_model = {
        "description": "Naive power-law fit: log(edge) = intercept + slope * log(SV2)",
        "selection": "health_rows with stage >= 40k and finite positive edge/SV2",
        "n": len(fit_rows),
        "intercept": float(intercept),
        "slope": float(slope),
        "r_squared_log_space": r_squared,
        "sv2_training_range": [fit_min, fit_max],
        "monotone_decreasing": bool(slope < 0),
    }
    edge_predictions: list[dict[str, Any]] = []
    for arm in ARMS:
        for stage in PREDICTION_STAGES:
            sv2 = series[arm].get(stage, {}).get("sv2")
            if finite_number(sv2) and sv2 > 0:
                prediction = float(math.exp(intercept + slope * math.log(sv2)))
                extrapolation = bool(sv2 < fit_min or sv2 > fit_max)
            else:
                prediction = None
                extrapolation = None
            edge_predictions.append(
                {
                    "arm": arm,
                    "stage": stage,
                    "sv2": sv2,
                    "predicted_edge": prediction,
                    "extrapolation": extrapolation,
                }
            )

    # Same-iteration sanity comparison. "Strong" is declared here as >=2x or <=0.5x B.
    b100 = series["B"][100_000]["sv2"]
    comparisons_100k: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_sv2 = series[arm].get(100_000, {}).get("sv2")
        fold = float(arm_sv2 / b100) if finite_number(arm_sv2) and finite_number(b100) else None
        strong = bool(fold is not None and (fold >= 2.0 or fold <= 0.5))
        comparisons_100k.append(
            {"arm": arm, "arm_sv2": arm_sv2, "b_sv2": b100, "fold_vs_B": fold, "strong_difference": strong}
        )
        print(
            f"100k SV2 sanity: {arm}={fmt(arm_sv2)} vs B={fmt(b100)}, "
            f"fold={fmt(fold)}; strong (>=2x or <=0.5x): {'YES' if strong else 'no'}"
        )

    # Figure: every measured checkpoint, without interpolation beyond connecting lines.
    colors = {
        "ST1": "#0072B2", "ST2": "#D55E00", "ST3": "#009E73", "ST4": "#CC79A7",
        "B": "#222222", "4241": "#E69F00", "2718": "#56B4E9",
    }
    markers = {"ST1": "o", "ST2": "s", "ST3": "^", "ST4": "D", "B": "x", "4241": "v", "2718": "P"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for arm in (*ARMS, "B", "4241", "2718"):
        pts = [(stage, values["sv2"]) for stage, values in sorted(series[arm].items()) if finite_number(values["sv2"])]
        if pts:
            sx, sy = zip(*pts)
            plain = arm in {"B", "4241", "2718"}
            axes[0].plot(
                np.asarray(sx) / 1000, sy, label=DISPLAY_NAMES[arm], color=colors[arm], marker=markers[arm],
                markevery=max(1, len(sx) // 8), markersize=4.5, linewidth=1.8 if not plain else 1.4,
                linestyle="--" if plain else "-", alpha=0.95,
            )
    for arm in (*ARMS, "B"):
        pts = [
            (stage, values["s0_gact_med"])
            for stage, values in sorted(series[arm].items())
            if finite_number(values["s0_gact_med"]) and values["s0_gact_med"] > 0
        ]
        if pts:
            sx, sy = zip(*pts)
            axes[1].plot(
                np.asarray(sx) / 1000, sy, label=DISPLAY_NAMES[arm], color=colors[arm], marker=markers[arm],
                markevery=max(1, len(sx) // 8), markersize=4.5, linewidth=1.8 if arm in ARMS else 1.4,
                linestyle="-" if arm in ARMS else "--", alpha=0.95,
            )
    axes[0].set(title="Slow trunk-2 top singular value", xlabel="Training iteration (k)", ylabel="SV2 (log scale)", yscale="log")
    axes[1].set(title="Slow-only pre-LN/pre-tanh activation", xlabel="Training iteration (k)", ylabel="s0.gact_med (log scale)", yscale="log")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, ncol=2, frameon=True)
    fig.suptitle("Part (a): spectral-growth proxy and slow-only activation", fontsize=13)
    fig.savefig(ROOT / "fig_a_proxy.png", dpi=150)
    plt.close(fig)

    results = {
        "metadata": {
            "table_stages": list(TABLE_STAGES),
            "missing_policy": "No interpolation; absent fields/checkpoints are null.",
            "activation_definition": "s0.gact_med is the slow-only pre-LN/pre-tanh GELU norm; s0.act_med is final trunk activation median.",
            "st2_summary_cutoff": "290k (pre-collapse); its available 300k row is retained in the stage table and edge prediction.",
            "feature_checkpoint_stage_normalization": "Feature records use iter=checkpoint+1; stage is taken from the checkpoint filename.",
        },
        "stage_rows": stage_rows,
        "summary_statistics": summaries,
        "edge_model": edge_model,
        "edge_predictions": edge_predictions,
        "sanity": {
            "nonfinite_input_counts": {name: len(rows) for name, rows in nonfinite_by_file.items()},
            "nonfinite_input_values": nonfinite_by_file,
            "sv2_comparisons_at_100k": comparisons_100k,
            "strong_difference_definition": "fold vs B >= 2.0 or <= 0.5",
        },
    }
    with (ROOT / "results_a.json").open("w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
        f.write("\n")

    md: list[str] = [
        "# Part (a): stability-margin proxy analysis",
        "",
        "## Short README",
        "",
        "Values are read directly from the supplied JSON files; no checkpoints are interpolated. "
        "SV2/SV1 are the leading slow singular values for trunk 2/trunk 1. `s0.act_med` is the final "
        "slow-only trunk activation median and `s0.gact_med` is its pre-LN/pre-tanh GELU norm. "
        "The latter is the comparable growth measure for ST1 (LayerNorm hides growth in `act_med`) "
        "and ST4 (output tanh can hide saturation). Lineage feature files contain no replay activation "
        "or loss fields, so those cells are shown as —. ST2's trend summary ends at 290k, before the "
        "reported 295.5k collapse; its measured 300k row is still displayed.",
        "",
        "## Summary statistics",
        "",
        "Spearman trends use every available measured checkpoint from 100k through the stated endpoint. "
        "Ratios use the exact endpoint pair shown.",
        "",
    ]
    summary_rows = []
    for arm in (*ARMS, "B", "4241", "2718"):
        s = summaries[arm]
        ep = s["ratio_endpoints"]
        summary_rows.append([
            DISPLAY_NAMES[arm],
            f"{stage_label(ep['numerator_stage'])}/{stage_label(ep['denominator_stage'])}",
            fmt(s["sv2_ratio"]),
            fmt(s["s0_gact_med_ratio"]),
            f"{fmt(s['spearman_iteration_sv2']['rho'])} (n={s['spearman_iteration_sv2']['n']})",
            f"{fmt(s['spearman_iteration_s0_gact_med']['rho'])} (n={s['spearman_iteration_s0_gact_med']['n']})",
        ])
    md.extend(markdown_table(
        ["Arm", "Ratio endpoints", "SV2 ratio", "s0.gact ratio", "Spearman(iter, SV2)", "Spearman(iter, gact)"],
        summary_rows,
    ))
    md.extend([
        "",
        "## Naive plain-recipe edge prediction",
        "",
        f"Fit on {edge_model['n']} plain-recipe health rows at stage ≥40k: "
        f"`log(edge) = {intercept:.4f} + ({slope:.4f}) log(SV2)` (log-space R²={r_squared:.3f}). "
        f"The observed fitting range is SV2 {fit_min:.3f}–{fit_max:.3f}. This is only a naive proxy "
        "transfer: predictions outside that interval are explicitly marked **EXTRAPOLATION** and should "
        "not be interpreted as calibrated stability margins.",
        "",
    ])
    pred_rows = []
    for row in edge_predictions:
        if row["predicted_edge"] is None:
            flag = "missing checkpoint"
        elif row["extrapolation"]:
            flag = "**EXTRAPOLATION**"
        else:
            flag = "within fit SV2 range"
        pred_rows.append([
            DISPLAY_NAMES[row["arm"]], stage_label(row["stage"]), fmt(row["sv2"]), fmt(row["predicted_edge"]), flag
        ])
    md.extend(markdown_table(["Arm", "Stage", "SV2", "Naive predicted edge", "Range flag"], pred_rows))
    md.extend([
        "",
        "## SV2 sanity check at 100k",
        "",
        "A strong same-iteration difference is defined here as at least 2× B or at most 0.5× B.",
        "",
    ])
    sanity_rows = [
        [DISPLAY_NAMES[row["arm"]], fmt(row["arm_sv2"]), fmt(row["b_sv2"]), fmt(row["fold_vs_B"]), "yes" if row["strong_difference"] else "no"]
        for row in comparisons_100k
    ]
    md.extend(markdown_table(["Arm", "Arm SV2", "B SV2", "Fold vs B", "Strong?"], sanity_rows))
    md.extend([
        "",
        "## Requested stage values",
        "",
        "Only available requested-stage checkpoints are listed. For B, SV2 is specifically read from "
        "`replay_B.json` as `svals[2][0]`, as requested.",
        "",
    ])
    value_rows = [
        [
            DISPLAY_NAMES[row["arm"]], stage_label(row["stage"]), fmt(row["sv2"]), fmt(row["sv1"]),
            fmt(row["s0_act_med"]), fmt(row["s0_gact_med"]), fmt(row["s0_loss"]), fmt(row["s1_loss"]),
        ]
        for row in stage_rows
    ]
    md.extend(markdown_table(
        ["Arm", "Stage", "SV2", "SV1", "s0.act_med", "s0.gact_med", "s0 loss", "s1 loss"], value_rows
    ))
    md.extend([
        "",
        "## Data sanity",
        "",
        "The script recursively checks all supplied inputs and prints every NaN/inf path. Nonfinite "
        "values occur in replay diagnostic fields such as gain/frac values when fast writes are off; "
        "requested extracted measurements and generated JSON are finite or null. Counts by input:",
        "",
    ])
    md.extend(markdown_table(
        ["Input", "NaN/inf count"],
        [[name, str(len(rows))] for name, rows in nonfinite_by_file.items()],
    ))
    md.extend(["", "Figure: [fig_a_proxy.png](fig_a_proxy.png). Machine-readable output: [results_a.json](results_a.json).", ""])
    (ROOT / "results_a_table.md").write_text("\n".join(md))

    print(f"Wrote {ROOT / 'results_a.json'}")
    print(f"Wrote {ROOT / 'results_a_table.md'}")
    print(f"Wrote {ROOT / 'fig_a_proxy.png'}")


if __name__ == "__main__":
    main()
