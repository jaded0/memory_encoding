# Dead Code Cleanup Summary

This document summarizes the unused/dead code that was removed from `hebbian_model.py` and `hebby.py`.

## Removed from `hebbian_model.py`:

### 1. **PlasticityNorm class** (lines ~12-18)
- **Status**: REMOVED
- **Reason**: Entire class was defined but never instantiated or used anywhere in the codebase
- **Impact**: No functional impact, reduces code complexity

### 2. **self.imprints parameter** (HebbianLinear.__init__)
- **Status**: REMOVED
- **Reason**: Created but never actually used; `update_imprints` method only copied data to traces
- **Impact**: No functional impact, reduces memory usage

### 3. **self.plasticity_candidate_weights** (HebbianLinear.__init__)
- **Status**: REMOVED
- **Reason**: Created with `torch.zeros_like(self.weight)` but never referenced
- **Impact**: No functional impact, reduces memory usage

### 4. **self.frequency parameter** (HebbianLinear.__init__)
- **Status**: REMOVED
- **Reason**: Had complex initialization logic with uniform distributions and wave patterns but never used in computations
- **Impact**: No functional impact, reduces memory usage and initialization complexity

### 5. **self.phase_shift parameter** (HebbianLinear.__init__)
- **Status**: REMOVED
- **Reason**: Had initialization logic but was never used in any computations
- **Impact**: No functional impact, reduces memory usage

### 6. **self.learning_rate = 1** (HebbianLinear.__init__)
- **Status**: REMOVED
- **Reason**: Set but never referenced anywhere in the code
- **Impact**: No functional impact

### 7. **update_weights method** (HebbianLinear class, lines ~183-262)
- **Status**: REMOVED
- **Reason**: Legacy update method superseded by the unified approach (`populate_dfa_gradients` + `apply_unified_updates`)
- **Impact**: No functional impact, the new unified approach is used instead

### 8. **update_weights_dfa method** (EtherealRNN class)
- **Status**: REMOVED
- **Reason**: Called the now-removed `update_weights` method on individual layers; superseded by unified approach
- **Impact**: No functional impact, unified approach handles this functionality

## Removed from `hebby.py`:

### 9. **Fallback to update_weights_dfa** (lines ~155-157)
- **Status**: REMOVED
- **Reason**: Fallback case for "non-Ethereal models" that would never execute in practice since the codebase only uses EtherealRNN (unified approach) or SimpleRNN (no Hebbian methods)
- **Impact**: No functional impact, removes unreachable code

## Code Architecture Changes:

The cleanup reveals a clear transition from an **old update system** to a **new unified system**:

### Old System (removed):
- `update_weights_dfa()` → `update_weights()` on each layer
- Complex parameter initialization for unused features (frequency, phase_shift, etc.)

### New System (retained):
- `populate_dfa_gradients()` to set up gradients using DFA feedback weights
- `apply_unified_updates()` to apply updates consistently for both DFA and backprop

## Verification:

- ✅ Both `hebbian_model.py` and `hebby.py` import successfully after cleanup
- ✅ No syntax errors introduced
- ✅ All functional code paths preserved
- ✅ Memory usage reduced by removing unused parameters
- ✅ Code complexity reduced by removing dead methods and classes

## Files Modified:

1. `hebbian_model.py` - Removed unused class, parameters, and methods
2. `hebby.py` - Removed unreachable fallback code

The cleanup maintains full backward compatibility while reducing memory footprint and code complexity.
