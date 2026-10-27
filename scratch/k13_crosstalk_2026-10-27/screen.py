"""Print screening table and a fail-closed gate (never authorizes a real launch)."""
import json
import pathlib
import sys


def assess(root):
    root = pathlib.Path(root)
    baseline = json.loads((root / "load0.json").read_text())["acc"]["all"]
    selected = None
    print("D strict gap selective L1_PR L2_PR L1_abs_cos L2_abs_cos eligible")
    for d in (4, 8, 16, 32):
        x = json.loads((root / f"load{d}.json").read_text())
        gap = json.loads((root / f"gap{d}.json").read_text())["acc"]["all"]
        selective = json.loads((root / f"selective{d}.json").read_text())["acc"]["all"]
        acc = x["acc"]["all"]
        layers = [x["layers"][k] for k in ("L1", "L2")]
        valid = (.30 <= acc <= .85 and baseline - acc >= .10 and gap - acc >= .10
                 and selective - acc >= .10
                 and all(l["key_pr"] >= 1.5 and l["cos_oth_abs"] <= .70 for l in layers))
        print(d, *(f"{v:.4f}" for v in (acc, gap, selective, layers[0]["key_pr"],
              layers[1]["key_pr"], layers[0]["cos_oth_abs"], layers[1]["cos_oth_abs"])), valid)
        if valid and selected is None:
            selected = d
    return {"regime_verified": selected is not None, "selected_load": selected,
            "launch_approved": False, "n_iters": None,
            "reason": "Frozen screening only; main session must fix budget and re-read Jaden's log."}


if __name__ == "__main__":
    print(json.dumps(assess(sys.argv[1]), indent=2))
