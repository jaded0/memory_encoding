# Memory Encoding with EtherealRNN

This repository implements and compares different weight update mechanisms for recurrent neural networks, with a focus on the EtherealRNN architecture that features high-plasticity weights for short-term memory.

## Overview

The project explores three different training approaches:
1. **DFA (Direct Feedback Alignment)** - Uses fixed random feedback weights for error propagation
2. **Backprop** - Standard backpropagation through time for each step
3. **BPTT (Backpropagation Through Time)** - Accumulates gradients across the entire sequence

## Key Features

### EtherealRNN Architecture
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

```bash
# Train with DFA updates
python hebby.py --updater dfa --model_type ethereal

# Train with backprop updates
python hebby.py --updater backprop --model_type ethereal

# Train with BPTT updates
python hebby.py --updater bptt --model_type ethereal
```

### Key Parameters

- `--updater`: Choose between `dfa`, `backprop`, or `bptt`
- `--model_type`: Choose between `rnn` or `ethereal`
- `--learning_rate`: Learning rate for weight updates
- `--plast_learning_rate`: Learning rate for plasticity parameters
- `--plast_clip`: Maximum value for high-plasticity weights
- `--plast_proportion`: Proportion of weights that are high-plasticity
- `--forget_rate`: Forgetting factor for high-plasticity weights
- `--batch_size`: Number of sequences processed together

### Advanced Features

- **Positional Encoding**: Add positional information with `--positional_encoding_dim N`
- **Residual Connections**: Enable/disable with `--residual_connection True/False`
- **Weight Normalization**: Enable/disable with `--normalize True/False`
- **Input Modes**: Choose between `--input_mode last_one` or `--input_mode last_two`

## Testing

Run the unified updates test to verify the implementation:

```bash
python test_unified_updates.py
```

## Project Structure

- `hebby.py`: Main training script with unified training loop
- `hebbian_model.py`: Implementation of EtherealRNN and HebbianLinear layers
- `preprocess.py`: Data loading and preprocessing utilities
- `utils.py`: Helper functions and utilities
- `test_unified_updates.py`: Test script for unified update approach
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
