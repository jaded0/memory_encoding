#!/bin/bash
# P3, the expansive write-gain test (prereg draft queue_2026-10-27/prereg_drafts/P3.md). One run + its measurements.
# usage (from the repo root, hebby env active): p3_train.sh ARM SEED OUTROOT [N_ITERS]
#   Recipe = KRC1 (the 10-16 salience decomposition's key-recall checkpoints; same flags as Batch O2's KR "lin"):
#   lr 1e-4, alpha 3e4, frac .1, forget .01, no clamps, 3x1024, B16, last_one, recurrence off, fused, deterministic, 100k.
#   ARM: ctl (flag off) | g1 | g2 | g4 (--write_gain_power 1/2/4, window 256, no cap)
#        | g2w64 (p 2, window 64: median-window sensitivity, optional) | g4cap10 (p 4, --write_gain_max 10, optional).
#   Env (smoke tests only): P3_SAVE_FREQ / P3_KEEP_EVERY (default 5000 / 25000).
#   Env: P3_SG_DEVICE (default cuda) and P3_SG_BATCHES (default 128) for the decomposition; P3_SG_CKPTS (default "50000 N").
# Afterwards: heldout.py (observed/strict/no_fast, 128 batches) on the last checkpoint, then p3_sg.py (the 10-16
# decomposition, 128 batches = 2048 episodes) on the 50k and final checkpoints (the last one if the run stopped early).
set -euo pipefail
arm=$1; seed=$2; root=$3; n=${4:-100000}
case $arm in
  ctl) af="";;
  g1) af="--write_gain_power 1";;
  g2) af="--write_gain_power 2";;
  g4) af="--write_gain_power 4";;
  g2w64) af="--write_gain_power 2 --write_gain_window 64";;
  g4cap10) af="--write_gain_power 4 --write_gain_max 10";;
  *) echo "unknown arm $arm"; exit 2;;
esac
export WANDB_MODE=offline CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ -z "${TORCHINDUCTOR_CACHE_DIR:-}" ]; then
  mkdir -p $HOME/.cache/inductor_p3
  export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_p3/XXXX)
fi
d=$root/KR-$arm-s$seed; mkdir -p $d
git rev-parse HEAD > $d/code_commit; git status --porcelain --untracked-files=no >> $d/code_commit
echo "$(date -Is) host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} arm $arm seed $seed n $n" >> $d/launch.log
if [ ! -f $d/final_eval.json ]; then
train_rc=0
python -u train.py --dataset long_range_memory_dataset --input_mode last_one --forget_rate 0.01 --ephemeral_update_clamp 0 \
  --weight_clamp 0 --slow_weight_decay 0 --hidden_size 1024 --num_layers 3 \
  --residual_connection false --positional_encoding_dim 0 --batch_size 16 --track false --deterministic true \
  --resume true --model_type ephemeral --updater dfa --enable_recurrence false --fused_update true \
  --learning_rate 1e-4 --plasticity 3e4 --ephemeral_fraction 0.1 --optimizer sgd \
  --n_iters $n --print_freq 500 --early_stop_window 0 --heldout_eval_every 25000 --heldout_batches 16 \
  --checkpoint_save_freq ${P3_SAVE_FREQ:-5000} --checkpoint_keep_every ${P3_KEEP_EVERY:-25000} --seed $seed --checkpoint_dir $d $af >> $d/train.log 2>&1 || train_rc=$?
echo "TRAIN EXIT $train_rc" >> $d/train.log
if (( train_rc )); then
  # A scientific blowup is an outcome; infrastructure errors must fail the job.
  grep -q '^Non-finite loss' $d/train.log || exit "$train_rc"
  echo 'Scientific outcome: nonfinite; evaluating last saved pre-failure checkpoint.' >> $d/train.log
  if [ ! -f $d/latest_checkpoint.pth ]; then touch $d/DONE; exit 0; fi
fi
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches ${P3_EVAL_BATCHES:-128} --json $d/final_eval.json > $d/final_eval.log 2>&1
echo "EVAL EXIT $?" >> $d/train.log
fi
ck=""; for it in ${P3_SG_CKPTS:-50000 $n}; do f=$d/checkpoint_$(printf %08d $it).pth; [ -f $f ] && ck="$ck $f"; done
[ -f $d/checkpoint_$(printf %08d $n).pth ] || ck="$ck $d/latest_checkpoint.pth"
python -u scratch/p3_2026-10-27/p3_sg.py $ck --out $d/sg --batches ${P3_SG_BATCHES:-128} --device ${P3_SG_DEVICE:-cuda} > $d/sg.log 2>&1
echo "SG EXIT $?" >> $d/train.log
touch $d/DONE
