"""Print one line per ratio JSON (bfr_ratio.py output). Usage: python ratio_summary.py DIR"""
import glob, json, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f))
    print(os.path.basename(f)[6:-5], d["fast_read_norm"], " | ".join(
        "L%s eff/s %.2f p90 %.2f rec %.2f raw/s %.2f slow %.3g eff %.3g" % (
            k, v["eff_over_slow_med"], v["eff_over_slow_p90"], v["eff_over_slow_recall_med"], v["raw_over_slow_med"],
            v["slow_med"], v["eff_med"]) for k, v in d["layers"].items()))
