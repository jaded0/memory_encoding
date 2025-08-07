# Unified Updates Implementation

## Problem Summary

The original implementation had issues with NaN values when using backprop with the EtherealRNN architecture. The problem was caused by my attempt to unify the update mechanisms for DFA and backprop, which broke the original working approach.

## Root Cause Analysis

The NaN issues were caused by:

1. **Incorrect gradient application**: In my unified approach, I was applying gradients in the wrong direction
2. **Broken bias handling**: The bias update logic was completely wrong
3. **Double forgetting**: Forgetting was being applied twice in some cases
4. **Wrong learning rate application**: The sign was incorrect when applying updates

## Solution Implemented

I reverted to the original approach but fixed just the core issue. The key insight is that:

- **DFA** uses `update_weights_dfa()` method which computes gradients via feedback weights
- **Backprop** uses standard PyTorch `backward()` and manual parameter updates
- **BPTT** accumulates losses across the sequence and applies gradients once at the end

## Key Differences in Ethereal Architecture

### DFA (Direct Feedback Alignment)
- **Per-step updates**: Updates weights immediately after each time step
- **Feedback weights**: Uses separate feedback weights to project error signals to hidden layers
- **No temporal gradients**: Hidden state is detached at each step (`hidden = hidden.detach()`)
- **Independent sequences**: Each sequence in batch gets independent gradient computation

### Backprop (Standard Backpropagation)
- **Per-step updates**: Updates weights immediately after each time step  
- **Standard gradients**: Uses PyTorch's autograd to compute true gradients
- **No temporal gradients**: Hidden state is detached at each step (`hidden = hidden.detach()`)
- **Shared computation graph**: Gradients flow through the same computation graph for all sequences

### BPTT (Backpropagation Through Time)
- **Sequence-level updates**: Accumulates loss across entire sequence, updates only at sequence end
- **Temporal gradients**: Hidden state is NOT detached, gradients flow through time
- **Full sequence dependency**: All time steps in sequence contribute to single gradient computation

## Implementation Details

### EtherealRNN Specific Features
1. **Per-batch weights**: Each sequence in batch has independent `candidate_weights` of shape `[batch_size, out_features, in_features]`
2. **Plasticity mask**: Boolean mask determines high vs low plasticity weights (`plast_proportion` controls ratio)
3. **Forgetting mechanism**: Applied via `forgetting_factor` parameter to high-plasticity weights
4. **Gradient scaling**: Different effective learning rates for high vs low plasticity weights

### Weight Update Mechanisms

**DFA Path:**
```python
# Error projection using feedback weights
projected_error = error_signal @ self.feedback_weights  # Non-last layers
# Direct outer product update
candidate_update = projected_error.unsqueeze(2) * input_expanded
```

**Backprop/BPTT Path:**
```python
# Standard gradient computation
total_loss.backward()
# Manual parameter updates
for param in rnn.parameters():
    if param.grad is not None:
        param.data -= learning_rate * param.grad
```

## Stability Parameters

The working parameters that prevent NaN values:
- `LEARNING_RATE=1e-4` (not `1e-3`)
- `PLAST_CLIP=1e3` (not `1e4`)  
- `FORGET_RATE=0.01` (not `0.1`)

These parameters maintain the balance between:
- High plasticity (`PLAST_CLIP=1e3`) with controlled learning rate
- Forgetting rate (`FORGET_RATE=0.01`) preventing weight explosion
- Base learning rate (`LEARNING_RATE=1e-4`) preventing gradient explosion

## Short-term Memory Preservation

The ethereal architecture's unique plasticity system means that:
- All three methods interact differently with the high/low plasticity weight distinction
- Per-batch weights ensure each sequence maintains independent short-term memory
- Forgetting mechanism gradually decays high-plasticity weights between updates
- This design is particularly suited for studying different learning dynamics while preserving sequence-specific adaptations
