# C2 (2026-10-27; PREREG_fastlayers.md "Batch C2"): the arm table, sourced by c2.sbatch (ORC) and
# run_local.sh (smoke tests, or a local GPU). c2_arm TASK sets NAME and ARGV (train.py's arguments,
# without --n_iters, --seed-independent output paths or --checkpoint_dir).
#
# Recipe: the stabilizer pilot's output_tanh arm (benchmarks/stabilizer_pilot.md, sweeps/orc_stabilizer_pilot.sbatch),
# 3-char reversal, lr 1e-3, alpha 1e4, forget .01, 20% fast, weight clamp 1, 3 x 1024, batch 16, recurrence
# off, 400k, print every 5000, the default early stop (loss > 5 for 10 intervals = collapse). Seeds 3141 2718 4241
# are the pilot's, so its DFA runs at this recipe are the reference panel. Per-step backprop has no --fused_update.
#   tasks 0-5   Step 1: S1-off / S1-on (--backprop_step_parity) x 3 seeds
#   tasks 6-14  Step 2: S2-a1e4 / a1e3 / a1e2 = --backprop_step_parity --decouple_fast_feedback 0,1,2 at alpha 1e4, 1e3, 1e2
C2_SEEDS=(3141 2718 4241)
C2_N_TASKS=15
c2_arm() {
    local t=$1 seed arm alpha=1e4 extra=()
    seed=${C2_SEEDS[$((t % 3))]}
    case $((t / 3)) in
        0) arm=S1-off ;;
        1) arm=S1-on; extra=(--backprop_step_parity true) ;;
        2) arm=S2-a1e4; extra=(--backprop_step_parity true --decouple_fast_feedback 0,1,2) ;;
        3) arm=S2-a1e3; alpha=1e3; extra=(--backprop_step_parity true --decouple_fast_feedback 0,1,2) ;;
        4) arm=S2-a1e2; alpha=1e2; extra=(--backprop_step_parity true --decouple_fast_feedback 0,1,2) ;;
        *) echo "no C2 task $t" >&2; return 1 ;;
    esac
    NAME=$arm-s$seed
    ARGV=(--model_type ephemeral --updater backprop --enable_recurrence false --fused_update false
          --dataset 3_palindrome_dataset_vary_length --input_mode last_one
          --learning_rate 1e-3 --plasticity "$alpha" --forget_rate 0.01 --ephemeral_fraction 0.2
          --ephemeral_update_clamp 0 --grad_norm_clip 0 --weight_clamp 1 --output_tanh true
          --hidden_size 1024 --num_layers 3 --residual_connection false --layer_norm false
          --positional_encoding_dim 0 --batch_size 16 --optimizer sgd --print_freq 5000
          --checkpoint_save_freq 25000 --checkpoint_keep_every 50000 --resume true
          --track false --deterministic true --seed "$seed" "${extra[@]}")
}
