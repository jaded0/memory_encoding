"""Prints results.json compactly.  python summarize.py DIR"""
import json
import sys

r = json.load(open(sys.argv[1] + "/results.json"))


def f(x):
    return f"{x['mean']:.4g}±{x['std']:.2g}"


def show(x):
    return {k: (f(v) if isinstance(v, dict) and "mean" in v else (round(v, 4) if isinstance(v, float) else v)) for k, v in x.items()
            if not isinstance(v, (list, dict)) or (isinstance(v, dict) and "mean" in v)}


print(r["iteration"], r["episodes"], r["c_lr_alpha"])
print("closed", {l: f(v) for l, v in r["closed_form_max_rel_err_of_fast_read"].items()})
for sec in ("geometry", "value_subspace", "read", "mask_rank"):
    for l in ["L0", "L1", "L2", "i2h"]:
        print(sec, l, show(r[sec][l]))
for l in ["L0", "L1", "L2", "i2h"]:
    print("SVD", l, [(x["t"], x["rank_actual_f32weights"], x["rank_closed_f64"], x["rank_closed_f64_tol1e-6"], x["n_nonzero_cols"], round(x["PR_sv2"], 1)) for x in r["svd_F"][l][0]])
    print("SVDmin over seqs", l, [min(x["rank_closed_f64"] for x in s) for s in r["svd_F"][l]], [max(x["rank_closed_f64"] for x in s) for s in r["svd_F"][l]], r["svd_F"][l][0][0]["rel_diff_actual_vs_closed"])
print("ABL")
for k, v in r["ablation"].items():
    print(k, round(v["recall_acc"], 4), round(v["margin_mean"], 3), round(v["margin_std"], 3), v["n"])
for l in ["L0", "L1", "L2", "i2h"]:
    print("TRAJ", l, [round(x, 3) for x in r["F_traj"][l]["Fnorm_mean_by_step"]][:11])
    print("FX", l, [round(x, 3) for x in r["F_traj"][l]["Fx_norm_mean_by_step"]][:11])
    print("PRE", l, [round(x, 3) for x in r["F_traj"][l]["pre_norm_mean_by_step"]][:11])
print("slow Fro", r["slow_norm_Fro_masked_out"])
for l in ["L0", "L1", "L2", "i2h"]:
    print("lagterms", l, [round(x, 4) for x in r["read"][l]["term_norm_by_lag_mean"]], r["read"][l]["store_lag_counts"])
