# Memory Encoding with EphemeralRNN

This repository implements and compares different weight update mechanisms for recurrent neural networks, with a focus on the EphemeralRNN architecture that features high-plasticity weights for short-term memory.

## Overview

The project explores three different training approaches:
1. **DFA (Direct Feedback Alignment)** - Uses fixed random feedback weights for error propagation
2. **Backprop** - Standard backpropagation through time for each step
3. **BPTT (Backpropagation Through Time)** - Accumulates gradients across the entire sequence

## Key Features

### EphemeralRNN Architecture
- **High-Plasticity Weights**: A subset of weights that can adapt rapidly for short-term memory
- **Per-Batch Adaptation**: Each sequence in a batch has independent weight adaptations
- **Forgetting Mechanism**: Controlled decay of high-plasticity weights between updates
- **Unified Update Approach**: DFA and backprop now share the same update mechanism (see [UNIFIED_UPDATES.md](UNIFIED_UPDATES.md))

### Training Methods

#### DFA (Direct Feedback Alignment)
- Computes error signals using fixed random feedback weights
- Updates weights immediately at each time step
- Independent gradient computation for each sequence in batch
- No temporal gradient flow (hidden state detached)

#### Backprop
- Standard gradient computation through the network
- Updates weights immediately at each time step
- Independent gradient computation for each sequence in batch
- No temporal gradient flow (hidden state detached)

#### BPTT (Backpropagation Through Time)
- Accumulates loss across entire sequence
- Updates weights only after processing the complete sequence
- Temporal gradient flow through hidden states
- Sequence-level optimization

## Unified Update Approach

Both DFA and backprop now use a unified update mechanism that ensures:
- Identical forgetting, scaling, and clipping behavior
- Consistent logging and normalization
- Per-batch gradient preservation for short-term memory
- Reduced code duplication and improved maintainability

See [UNIFIED_UPDATES.md](UNIFIED_UPDATES.md) for detailed implementation information.

## Installation

```bash
# Clone the repository
git clone https://github.com/jaded0/memory_encoding.git
cd memory_encoding

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate hebby

# Or install dependencies manually
pip install torch wandb matplotlib numpy psutil
```

## Usage

### Basic Training

The defaults are the configuration the run scripts use: ephemeral model, DFA, no recurrence,
`last_one` input, no normalization or clipping, batch 16, and the hyperparameters
(lr 1e-4, plasticity 1e5, forget rate 0.01, hidden 1024, 10% ephemeral weights) that solve
3-char palindromes. `python train.py --help` lists every default.

```bash
# Train with DFA updates (the defaults)
python train.py

# Train with backprop updates
python train.py --updater backprop --model_type ephemeral

# Train with BPTT updates
python train.py --updater bptt --model_type ephemeral
```

### Key Parameters

- `--updater`: Choose between `dfa`, `backprop`, or `bptt`
- `--model_type`: Choose between `rnn` or `ephemeral`
- `--learning_rate`: Learning rate for weight updates
- `--plast_clip`: Plasticity (learning-rate multiplier, alpha) of the ephemeral weights
- `--plast_proportion`: Proportion of weights that are high-plasticity
- `--forget_rate`: Fraction of each ephemeral weight removed per step
- `--resume` / `--resume_checkpoint PATH`: Resume from `latest_checkpoint.pth`, or from an explicit checkpoint
- `--batch_size`: Number of sequences processed together
- `--seed`: Seed Python, NumPy, Torch, dataset shuffling, and DataLoader sampling
- `--deterministic`: Require deterministic Torch operations; requires `--seed`

### SLURM time limits

The sbatch scripts request `#SBATCH --signal=B:USR1@600` and launch training through
`forward_signals` (from `forward_signals.sh`). Ten minutes before the wall-time limit,
`train.py` stops at the next iteration, saves `latest_checkpoint.pth` (if `--checkpoint_save_freq > 0`),
records `end_reason: time_limit` in W&B, and exits with code 124; resume with `--resume`. A SIGTERM
stops the same way with `end_reason: terminated` and exit code 143.

### Advanced Features

- **Positional Encoding**: Add positional information with `--positional_encoding_dim N`
- **Residual Connections**: Enable/disable with `--residual_connection True/False`
- **Weight Normalization**: Enable/disable with `--normalize True/False`
- **Input Modes**: Choose between `--input_mode last_one` or `--input_mode last_two`

## Testing

Run the network-free assertion suite from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s . -t . -v
```

The suite includes fixed-input golden traces through the real `train.train()`
path for DFA, backprop, and BPTT. See
[BASELINE_CHARACTERIZATION.md](BASELINE_CHARACTERIZATION.md) for the trace
schema, regeneration command, and known limitations.

## Project Structure

- `train.py`: Main training script with unified training loop
- `ephemeral_model.py`: Implementation of EphemeralRNN and EphemeralLinear layers
- `preprocess.py`: Data loading and preprocessing utilities
- `reproducibility.py`: Opt-in seeding and checkpoint RNG state helpers
- `utils.py`: Helper functions and utilities
- `test_unified_updates.py`: Test script for unified update approach
- `tests/fixtures/training_traces.json`: CPU golden traces for all update paths
- `BASELINE_CHARACTERIZATION.md`: Reproducibility and baseline trace contract
- `UNIFIED_UPDATES.md`: Detailed documentation of unified approach

## Results and Monitoring

The training process logs metrics to Weights & Biases (WandB) including:
- Loss values
- Accuracy metrics
- Weight and gradient norms
- Plasticity parameter statistics

To disable WandB tracking, use `--track False`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure nothing is broken
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Direct Feedback Alignment papers
- Hebbian learning principles
- Recurrent neural network training methods
