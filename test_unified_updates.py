#!/usr/bin/env python3
"""
Test script to verify that all three update methods (DFA, backprop, BPTT) 
work correctly with the EtherealRNN architecture.
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hebbian_model import EtherealRNN
from utils import initialize_charset

def test_method(model_type, updater, device):
    """Test a specific combination of model type and updater."""
    print(f"\n=== Testing {model_type} with {updater} ===")
    
    # Initialize charset (using a simple one for testing)
    charset = ['a', 'b', 'c', 'd']
    char_to_idx = {char: idx for idx, char in enumerate(charset)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    n_characters = len(charset)
    
    # Model parameters
    input_size = n_characters
    hidden_size = 32  # Small for testing
    output_size = n_characters
    num_layers = 1   # Single layer for simplicity
    
    try:
        # Initialize model
        model = EtherealRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            charset=charset,
            updater=updater,
            plast_clip=1e3,
            batch_size=2,
            forget_rate=0.01,
            plast_proportion=0.2
        ).to(device)
        
        # Create test data
        batch_size = 2
        seq_length = 5
        input_tensor = torch.randn(batch_size, seq_length, input_size).to(device)
        hidden = model.initHidden(batch_size)
        
        print(f"Model initialized successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test forward pass
        for i in range(seq_length - 1):
            output, hidden, self_grad = model(input_tensor[:, i, :], hidden)
            
            # Check for NaN values
            if torch.isnan(output).any():
                print(f"ERROR: NaN detected in output at step {i}")
                return False
                
            if torch.isnan(hidden).any():
                print(f"ERROR: NaN detected in hidden at step {i}")
                return False
                
        print("Forward pass completed without NaN values")
        
        # Test that model can be trained without errors
        model.train()
        hidden = model.initHidden(batch_size)
        target = torch.randint(0, n_characters, (batch_size,)).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Run one training step
        for i in range(seq_length - 1):
            # For DFA, we need to ensure output requires gradients
            if updater == 'dfa':
                input_tensor[:, i, :].requires_grad_(True)
            
            output, hidden, self_grad = model(input_tensor[:, i, :], hidden)
            
            # For DFA, ensure output requires gradients
            if updater == 'dfa':
                output.requires_grad_(True)
            
            loss = criterion(output, target)
            
            if updater == 'dfa':
                # DFA training step using unified approach
                model.zero_grad()
                global_error = torch.autograd.grad(loss, output, grad_outputs=torch.ones_like(loss), retain_graph=False)[0]
                reward_update = -global_error
                
                # Populate gradients using DFA feedback weights
                for layer in model.linear_layers:
                    layer.populate_dfa_gradients(reward_update)
                model.i2o.populate_dfa_gradients(reward_update)
                model.self_grad.populate_dfa_gradients(reward_update)
                
                # Apply forgetting step before weight updates
                model.apply_forget_step()
                
                # Apply unified updates using the DFA-populated gradients
                state = {"log_norms_now": False}
                for layer in model.linear_layers:
                    layer.apply_unified_updates(1e-4, 0, state)
                model.i2o.apply_unified_updates(1e-4, 0, state)
                model.self_grad.apply_unified_updates(1e-4, 0, state)
                
                # Clear gradients after unified updates
                model.zero_grad()
                
            elif updater == 'backprop':
                # Backprop training step using unified approach
                # For backprop, we only do one step per sequence (not per timestep)
                if i == 0:  # Only on first step to avoid multiple backwards
                    model.zero_grad()
                    loss.backward()
                    
                    # Apply forgetting step before weight updates
                    model.apply_forget_step()
                    
                    # Scale gradients for high-plasticity weights
                    model.scale_gradients(1e3)
                    
                    # Apply unified updates using the gradients computed by backprop
                    state = {"log_norms_now": False}
                    for layer in model.linear_layers:
                        layer.apply_unified_updates(1e-4, 0, state)
                    model.i2h.apply_unified_updates(1e-4, 0, state)
                    model.i2o.apply_unified_updates(1e-4, 0, state)
                    
                    # Clear gradients after unified updates
                    model.zero_grad()
                
            elif updater == 'bptt':
                # BPTT training step (simplified)
                if i == seq_length - 2:  # Only backward on last step
                    model.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        for param in model.parameters():
                            if param.grad is not None:
                                param.data -= 1e-4 * param.grad
                                param.grad.zero_()
            
            # Check for NaN after training step
            for name, param in model.named_parameters():
                if param.requires_grad and torch.isnan(param).any():
                    print(f"ERROR: NaN detected in parameter {name} after training step")
                    return False
                    
        print("Training step completed without NaN values")
        return True
        
    except Exception as e:
        print(f"ERROR: Exception occurred - {str(e)}")
        return False

def main():
    """Main test function."""
    print("Testing unified updates implementation...")
    
    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test all combinations
    model_types = ['ethereal']
    updaters = ['dfa', 'backprop', 'bptt']
    
    results = {}
    
    for model_type in model_types:
        for updater in updaters:
            key = f"{model_type}_{updater}"
            try:
                results[key] = test_method(model_type, updater, device)
            except Exception as e:
                print(f"Test failed for {key}: {str(e)}")
                results[key] = False
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    all_passed = True
    for key, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{key}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\nAll tests PASSED! The unified updates implementation is working correctly.")
        return 0
    else:
        print("\nSome tests FAILED! Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
