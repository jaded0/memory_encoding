# B2 main batch (pre-registered in PREREG_fastlayers.md "Batch B2" + addendum). Sourced by main.sbatch and main_deckard.sh.
# Base recipe (smoke-selected): R-break (lr 1e-3, alpha 1e4, frac .2, forget .01, clamp 1) + --layer_norm True, TinyStories cropped to 512 chars.
COMMON="--dataset roneneldan/tinystories --model_type ephemeral --updater dfa --hidden_size 1024 --num_layers 3 --batch_size 16
  --input_mode last_one --learning_rate 0.001 --plasticity 10000 --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1
  --layer_norm True --enable_recurrence False --residual_connection False --positional_encoding_dim 0 --track false
  --deterministic true --fused_update True --optimizer sgd --print_freq 100 --early_stop_window 0
  --checkpoint_save_freq 1000 --checkpoint_keep_every 2500 --resume true --max_seq_len 512 --n_iters 20000"
declare -A ARMFLAGS=([NOFAST]="--plasticity 0" [L0]="--fast_layers 0" [DEEP]="" [DEEPS]="--fast_read_norm rms")
# run_arm ARM SEED PY: train, then the pre-registered held-out evaluation (4096 stories) and curve points (256 stories).
run_arm() {
  local arm=$1 seed=$2 py=$3 d=../runs/$1-s$2
  mkdir -p $d ../eval_out/final ../eval_out/curves
  $py -u train.py $COMMON ${ARMFLAGS[$arm]} --seed $seed --checkpoint_dir $d
  echo "TRAIN EXIT $?"
  $py -u ../b2/b2_eval.py tiny --checkpoint $d/latest_checkpoint.pth --names ../b2/names.json --val_file ../b2/val8192.json \
     --stories 4096 --out ../eval_out/final/$arm-s$seed
  echo "EVAL EXIT $?"
  for ck in $d/checkpoint_*.pth; do
    it=$(basename $ck .pth | sed 's/checkpoint_0*//')
    $py -u ../b2/b2_eval.py curve --checkpoint $ck --val_file ../b2/val8192.json --stories 256 \
       --out ../eval_out/curves/$arm-s$seed-it$it > /dev/null 2>&1
  done
  echo "CURVES DONE"
}
