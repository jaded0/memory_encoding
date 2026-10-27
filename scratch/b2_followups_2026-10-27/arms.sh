# Source from bash; command arrays avoid word-splitting/path ambiguities.
SEEDS=(7201 7302 7403 7504 7605)
ARMS_A=(DEEP-W16 DEEP-F03 DEEP-F10)
ARMS_B=(REC-NOFAST REC-DEEP RNN-BPTT)
arm_args() {
  FLAGS=(--dataset roneneldan/tinystories --model_type ephemeral --updater dfa
    --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one
    --learning_rate .001 --plasticity 10000 --ephemeral_fraction .2 --forget_rate .01
    --weight_clamp 1 --layer_norm true --enable_recurrence false --residual_connection false
    --positional_encoding_dim 0 --track false --deterministic true --fused_update true
    --optimizer sgd --grad_norm_clip 0 --print_freq 100 --early_stop_window 0
    --checkpoint_save_freq 1000 --checkpoint_keep_every 2500 --resume true
    --max_seq_len 512 --n_iters 20000)
  case "$1" in
    DEEP-W16) FLAGS+=(--train_wipe_chars 16);;
    DEEP-F03) FLAGS+=(--forget_rate .03);;
    DEEP-F10) FLAGS+=(--forget_rate .1);;
    REC-NOFAST) FLAGS+=(--enable_recurrence true --plasticity 0);;
    REC-DEEP) FLAGS+=(--enable_recurrence true);;
    RNN-BPTT) FLAGS+=(--model_type rnn --updater bptt --enable_recurrence true
      --optimizer adam --learning_rate .0001 --grad_norm_clip 1 --residual_connection true
      --weight_clamp 0 --fused_update false);;
    *) echo "Unknown arm $1" >&2; return 2;;
  esac
}
