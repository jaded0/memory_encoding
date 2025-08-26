#!/usr/bin/env python3
"""
Test script to verify that plast_clip updates work correctly when resuming from checkpoints.
"""

import torch
import os
import tempfile
import numpy as np
from hebbian_model import EtherealRNN
from utils import save_checkpoint, load_checkpoint, initialize_charset

def test_plast_clip_update():
    """Test that plast_clip changes take effect when resuming from checkpoints."""
    
    print("Testing plast_clip update functionality...")
    
    # Initialize test parameters
    charset, char_to_idx, idx_to_char, n_characters = initialize_charset("palindrome_dataset")
    input_size = n_characters
    hidden_size = 64
    output_size = n_characters
    num_layers = 2
    batch_size = 4
    initial_plast_clip = 10.0
    new_plast_clip = 50.0
    
    # Create temporary directory for test checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pth")
        
        print(f"Step 1: Creating model with initial plast_clip = {initial_plast_clip}")
        
        # Create initial model
        rnn1 = EtherealRNN(
            input_size, hidden_size, output_size, num_layers, charset,
            updater='dfa', plast_clip=initial_plast_clip, batch_size=batch_size,
            plast_proportion=0.3  # Use higher proportion for easier testing
        )
        
        # Check initial plasticity values
        initial_plasticity_values = []
        for layer in rnn1.linear_layers:
            high_plast_mask = layer.mask
            high_plast_values = layer.plasticity.data[high_plast_mask]
            initial_plasticity_values.extend(high_plast_values.tolist())
        
        print(f"Initial high-plasticity values (sample): {initial_plasticity_values[:5]}")
        assert all(abs(val - initial_plast_clip) < 1e-6 for val in initial_plasticity_values), \
            f"Initial plasticity values should be {initial_plast_clip}"
        
        # Create config for saving
        config = {
            'plast_clip': initial_plast_clip,
            'n_hidden': hidden_size,
            'n_layers': num_layers,
            'charset_size': n_characters,
            'updater': 'dfa'
        }
        
        # Save checkpoint
        print("Step 2: Saving checkpoint...")
        checkpoint_state = {
            'iter': 1000,
            'model_state_dict': rnn1.state_dict(),
            'optimizer_state_dict': None,
            'main_program_state': {'training_instance': 500},
            'config': config,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
        }
        save_checkpoint(checkpoint_state, temp_dir, "test_checkpoint.pth")
        
        print(f"Step 3: Creating new model with updated plast_clip = {new_plast_clip}")
        
        # Create new model with different plast_clip
        rnn2 = EtherealRNN(
            input_size, hidden_size, output_size, num_layers, charset,
            updater='dfa', plast_clip=new_plast_clip, batch_size=batch_size,
            plast_proportion=0.3
        )
        
        # Update config with new plast_clip
        new_config = config.copy()
        new_config['plast_clip'] = new_plast_clip
        
        print("Step 4: Loading checkpoint...")
        
        # Load checkpoint (this should trigger plasticity update)
        rnn2, _, start_iter, loaded_state, loaded_config = load_checkpoint(
            checkpoint_path, rnn2, new_config, device='cpu'
        )
        
        # Simulate the plasticity update logic from hebby.py
        if isinstance(rnn2, EtherealRNN):
            loaded_plast_clip = loaded_config.get('plast_clip', 1.0)
            current_plast_clip = new_config.get('plast_clip', 1.0)
            
            if loaded_plast_clip != current_plast_clip:
                print(f"Plasticity clip changed from {loaded_plast_clip} to {current_plast_clip}")
                print("Updating plasticity parameters in all layers...")
                rnn2.update_plasticity_clip(current_plast_clip)
                print("Plasticity parameters updated successfully!")
        
        print("Step 5: Verifying plasticity values were updated...")
        
        # Check that plasticity values were updated
        updated_plasticity_values = []
        for layer in rnn2.linear_layers:
            high_plast_mask = layer.mask
            high_plast_values = layer.plasticity.data[high_plast_mask]
            updated_plasticity_values.extend(high_plast_values.tolist())
        
        print(f"Updated high-plasticity values (sample): {updated_plasticity_values[:5]}")
        
        # Verify all high-plasticity weights now have the new value
        assert all(abs(val - new_plast_clip) < 1e-6 for val in updated_plasticity_values), \
            f"Updated plasticity values should be {new_plast_clip}, but got {updated_plasticity_values[:5]}"
        
        # Verify that low-plasticity weights remain unchanged (should be 1.0)
        for layer in rnn2.linear_layers:
            low_plast_mask = ~layer.mask
            low_plast_values = layer.plasticity.data[low_plast_mask]
            assert all(abs(val - 1.0) < 1e-6 for val in low_plast_values.tolist()), \
                "Low-plasticity values should remain 1.0"
        
        print("✅ Test passed! Plasticity clip updates work correctly when resuming from checkpoints.")
        print(f"   - Initial plast_clip: {initial_plast_clip}")
        print(f"   - Updated plast_clip: {new_plast_clip}")
        print(f"   - High-plasticity weights updated: {len(updated_plasticity_values)}")
        
        return True

if __name__ == "__main__":
    try:
        test_plast_clip_update()
        print("\n🎉 All tests passed! The plast_clip update functionality is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
