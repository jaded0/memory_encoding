# Shared flags for the causal-cap arms: the INT_R0 recipe (B_late-s3141 resumed from 150k).
# usage: source flags.sh; FLAGS=("${CTRL[@]}" ...) ; arm flags appended by arm_flags NAME
CTRL=(--model_type ephemeral --updater dfa --enable_recurrence false --fused_update true
  --dataset 3_palindrome_dataset_vary_length --input_mode last_one --learning_rate 1e-3
  --plasticity 1e4 --forget_rate 0.01 --ephemeral_fraction 0.2 --weight_clamp 1
  --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0
  --batch_size 16 --track false --deterministic true --resume true)
COMMON=(--n_iters 210000 --print_freq 500 --trace_loop_every 500 --checkpoint_save_freq 5000
  --checkpoint_keep_every 5000 --checkpoint_keep_max 100 --early_stop_window 0)
# arm_flags NAME SPECDIR: sets ARM_FLAGS (array)
arm_flags() {
  ARM_FLAGS=()
  case $1 in
    CAP8)   ARM_FLAGS=(--sv_cap_file $2/cap8.pt) ;;
    CAP16)  ARM_FLAGS=(--sv_cap_file $2/cap16.pt) ;;
    SHAM)   ARM_FLAGS=(--sv_cap_file $2/sham8.pt) ;;
    SHAM2)  ARM_FLAGS=(--sv_cap_file $2/sham2_8.pt) ;;
    SHAM3)  ARM_FLAGS=(--sv_cap_file $2/sham3_8.pt) ;;
    CTRL)   ;;
    CAP8_D165) ARM_FLAGS=(--sv_cap_file $2/cap8.pt --sv_cap_start 165000) ;;
    CAP8_D170) ARM_FLAGS=(--sv_cap_file $2/cap8.pt --sv_cap_start 170000) ;;
    *) return 1 ;;
  esac
}
