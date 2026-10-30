#!/usr/bin/env bash
# Part B RNN lr sweep selection (10-29 addendum): CE on validation stories 4096..8191 of
# val8192.json, which no B2/B2fu test evaluation reads (final uses 0..4095, horizon 0..1023,
# curves 0..255). Sweep runs live in $ROOT/lrsweep/lr$LR (run.sh with B2FU_ROOT there).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
ROOT=${B2FU_ROOT:-$HOME/b2_followups_2026-10-27}; PY=${PYTHON:-python}
LRS=(${RNN_LRS:-3e-5 1e-4 3e-4 1e-3})
(cd "$ROOT/inputs" && sha256sum --strict -c SHA256SUMS)
sel=$ROOT/inputs/val_select.json
"$PY" -c 'import json,sys; v=json.load(open(sys.argv[1])); assert len(v)==8192; json.dump(v[4096:], open(sys.argv[2],"w"))' \
  "$ROOT/inputs/val8192.json" "$sel"
mkdir -p "$ROOT/lrsweep/select"
for lr in "${LRS[@]}"; do
  ck=$ROOT/lrsweep/lr$lr/runs/RNN-BPTT-s7201/latest_checkpoint.pth
  "$PY" -u scratch/b2_tinystories_2026-10-22/b2_eval.py curve --checkpoint "$ck" --val_file "$sel" \
    --length 512 --stories 4096 --out "$ROOT/lrsweep/select/lr$lr"
done
"$PY" - "$ROOT/lrsweep/select" "${LRS[@]}" <<'PY'
import json, sys
d, lrs = sys.argv[1], sys.argv[2:]
rows = []
for lr in lrs:
    r = json.load(open(f"{d}/lr{lr}.json"))
    assert r["iteration"] >= 20000 and r["stories"] == 4096, (lr, r["iteration"], r["stories"])
    assert abs(r["arm_config"]["learning_rate"] - float(lr)) < 1e-12, r["arm_config"]
    rows.append((r["ce"]["all"]["on"], lr))
for ce, lr in sorted(rows, key=lambda t: float(t[1])):
    print(f"lr {lr}: val CE {ce:.5f} nats/char")
pick = min(rows)[1]
print("PICK", pick)
json.dump({"val_ce": {lr: ce for ce, lr in rows}, "pick": pick}, open(f"{d}/pick.json", "w"), indent=1)
PY
