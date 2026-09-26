#!/bin/bash --login
# GPU throughput at the 3-palindrome benchmark size: the production DFA paths against the
# scratch harness (benchmarks/benchmark_dfa_throughput.py, with a profiler table), then
# end-to-end train.py rates for each arm of the current-code benchmark panel.
#
# Submit from the code checkout, choosing the GPU on the command line, e.g.
#   sbatch -p m9g --qos test sweeps/orc_speed_benchmark.sbatch     # P100
#   sbatch -p cs --qos cs sweeps/orc_speed_benchmark.sbatch        # A100
#   sbatch -p cs2 --qos cs sweeps/orc_speed_benchmark.sbatch       # H100
# The checkout needs synth_datasets/3_palindrome_dataset_vary_length.

#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --job-name=me_speed
#SBATCH --output=%x_%j.out

set -eo pipefail
conda activate hebby
set -u
export WANDB_MODE=disabled
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$SLURM_SUBMIT_DIR"
out=speed_results/${SLURM_JOB_PARTITION}_$(hostname -s)_$SLURM_JOB_ID
mkdir -p "$out"
{
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    lscpu | grep 'Model name'
    git rev-parse --short HEAD 2>/dev/null || cat CODE_REVISION 2>/dev/null || true
} | tee "$out/host.txt"

paths=(--task palindrome3 --symbols 7 --hidden-size 1024 --num-layers 3 --batch-size 16
       --learning-rate 1e-3 --plasticity 1e3 --forget-rate 0.01 --weight-clamp 1
       --device cuda --threads "$SLURM_CPUS_PER_TASK"
       --warmup-batches 20 --timed-batches 200 --repeats 3)
# The --fused_update paths need Triton, which needs compute capability 7.0 (not the P100).
fused=() fused_arm=()
if python -c 'import sys, torch; sys.exit(torch.cuda.get_device_capability()[0] < 7)'; then
    fused=(fused_core fused_native --check-batches 50)
    fused_arm=(ephemeral:dfa:false:true)
fi
python -u benchmarks/benchmark_dfa_throughput.py "${paths[@]}" --profile 10 \
    --paths main_native main_core scratch_core "${fused[@]}" \
    --output "$out/paths_deterministic.json" > "$out/paths_deterministic.log" 2>&1
python -u benchmarks/benchmark_dfa_throughput.py "${paths[@]}" --no-deterministic \
    --paths main_native scratch_core \
    --output "$out/paths_nondeterministic.json" > "$out/paths_nondeterministic.log" 2>&1

# The panel's shared settings; the arms differ only in model, updater and recurrence.
train=(--input_mode last_one --learning_rate 1e-3 --plasticity 1e3 --forget_rate 0.01
       --ephemeral_fraction 0.2 --ephemeral_update_clamp 0 --grad_norm_clip 0
       --unit_norm_weights false --weight_clamp 1 --hidden_size 1024 --num_layers 3
       --residual_connection false --positional_encoding_dim 0
       --dataset 3_palindrome_dataset_vary_length --batch_size 16
       --n_iters 3000 --print_freq 500 --checkpoint_save_freq 100000 --resume false
       --track false --seed 2718 --deterministic true)
for arm in ephemeral:dfa:false:false rnn:dfa:true:false rnn:bptt:true:false "${fused_arm[@]}"; do
    IFS=: read -r model updater recurrence fuse <<< "$arm"
    name=${model}_${updater}$([[ $fuse == true ]] && echo _fused)
    python -u train.py "${train[@]}" --model_type "$model" --updater "$updater" \
        --enable_recurrence "$recurrence" --fused_update "$fuse" --checkpoint_dir "$out/checkpoints_$name" \
        > "$out/train_$name.log" 2>&1
    echo "$name: $(grep iters_per_sec "$out/train_$name.log" | tail -4 | awk '{print $2}' | tr '\n' ' ')"
done | tee "$out/train_rates.txt"
