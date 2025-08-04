#!/usr/bin/env python3
"""
Comparison script to test DFA vs Backprop with the unified update mechanism.
This will help identify if the unification resolves the performance discrepancy.
"""

import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hebbian_model import HebbyRNN
from utils import initialize_charset

def create_simple_dataset(batch_size=4, seq_len=10, vocab_size=5, num_batches=100):
    """Create a simple repeating pattern dataset for easier learning."""
    datasets = []
    
    # Create a few simple patterns that repeat
    patterns = [
        [0, 1, 2, 0, 1, 2],  # Simple repeating pattern
        [1, 3, 1, 3, 1, 3],  # Another pattern
        [2, 4, 0, 2, 4, 0],  # Third pattern
        [0, 0, 1, 0, 0, 1],  # Pattern with repetition
        [3, 2, 1, 3, 2, 1],  # Descending pattern
    ]
    
    for _ in range(num_batches):
        batch_sequences = []
        batch_targets = []
        
        for b in range(batch_size):
            # Choose a random pattern and extend it to seq_len
            pattern = patterns[torch.randint(0, len(patterns), (1,)).item()]
            sequence = []
            for i in range(seq_len):
                sequence.append(pattern[i % len(pattern)])
            
            # Target is the next character in the pattern
            target = []
            for i in range(seq_len):
                target.append(pattern[(i + 1) % len(pattern)])
            
            batch_sequences.append(sequence)
            batch_targets.append(target)
        
        sequences = torch.tensor(batch_sequences)
        targets = torch.tensor(batch_targets)
        
        # Convert to one-hot
        onehot_sequences = torch.zeros(batch_size, seq_len, vocab_size)
        onehot_sequences.scatter_(2, sequences.unsqueeze(2), 1.0)
        
        datasets.append((sequences, onehot_sequences, targets))
    
    return datasets

def train_model(model_type='dfa', num_epochs=50, learning_rate=0.01):
    """Train a model with specified type and return loss history."""
    
    # Set up parameters
    batch_size = 4
    seq_len = 10
    vocab_size = 5
    hidden_size = 16
    num_layers = 2
    
    charset = [str(i) for i in range(vocab_size)]
    
    print(f"\nTraining {model_type.upper()} model...")
    print("=" * 40)
    
    # Create model
    rnn = HebbyRNN(
        input_size=vocab_size,
        hidden_size=hidden_size,
        output_size=vocab_size,
        num_layers=num_layers,
        charset=charset,
        updater=model_type,
        batch_size=batch_size,
        plast_proportion=0.3,
        forget_rate=0.1,
        normalize=False,  # Disable normalization for cleaner comparison
        clip_weights=0    # Disable weight clipping for cleaner comparison
    )
    
    # Create dataset
    dataset = create_simple_dataset(batch_size, seq_len, vocab_size, num_batches=100)
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    loss_history = []
    accuracy_history = []
    
    state = {"log_norms_now": False}
    
    print(f"Dataset size: {len(dataset)} batches")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"Vocabulary size: {vocab_size}, Hidden size: {hidden_size}")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_accuracies = []
        
        for batch_idx, (sequences, onehot_sequences, targets) in enumerate(dataset):
            # Reset model state
            rnn.wipe()
            hidden = rnn.initHidden(batch_size)
            
            batch_loss = 0.0
            batch_correct = 0
            batch_total = 0
            
            # Process sequence step by step
            for t in range(seq_len - 1):
                # Get input and target
                input_char = onehot_sequences[:, t, :]
                target_char = targets[:, t]
                
                # Forward pass
                output, hidden, _ = rnn(input_char, hidden)
                hidden = hidden.detach()  # Prevent BPTT
                
                # Compute loss
                step_loss = criterion(output, target_char)
                batch_loss += step_loss.item()
                
                # Compute accuracy
                predicted = torch.argmax(output, dim=1)
                batch_correct += (predicted == target_char).sum().item()
                batch_total += target_char.size(0)
                
                # Compute error signal and update weights
                if model_type == 'dfa':
                    # For DFA, use the same error signal for all layers
                    output_for_grad = output.clone().detach().requires_grad_(True)
                    step_loss_for_grad = criterion(output_for_grad, target_char)
                    error_signal = torch.autograd.grad(step_loss_for_grad, output_for_grad, 
                                                     grad_outputs=torch.ones_like(step_loss_for_grad), 
                                                     retain_graph=False)[0]
                    reward = -error_signal
                    
                    # Apply updates using unified mechanism
                    rnn.update_weights_dfa(reward, learning_rate, learning_rate, 1.0, 0.0, 0.0, state)
                else:
                    # For backprop, compute gradients properly for each layer
                    step_loss.backward(retain_graph=False)
                    
                    # Apply forgetting step before optimizer step
                    rnn.apply_forget_step()
                    
                    # Scale gradients for high-plasticity weights
                    rnn.scale_gradients(1.0)
                    
                    # Store gradient norms for logging
                    rnn.store_all_grad_norms()
                    
                    # Apply optimizer step (manual since we don't have an optimizer)
                    with torch.no_grad():
                        for param in rnn.parameters():
                            if param.grad is not None:
                                param.data -= learning_rate * param.grad
                                param.grad.zero_()
            
            # Record metrics
            avg_loss = batch_loss / (seq_len - 1)
            accuracy = batch_correct / batch_total
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)
        
        # Calculate epoch averages
        epoch_avg_loss = np.mean(epoch_losses)
        epoch_avg_accuracy = np.mean(epoch_accuracies)
        loss_history.append(epoch_avg_loss)
        accuracy_history.append(epoch_avg_accuracy)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {epoch_avg_loss:.4f}, Accuracy = {epoch_avg_accuracy:.4f}")
    
    return loss_history, accuracy_history

def main():
    """Compare DFA and Backprop performance."""
    
    print("Comparing DFA vs Backprop with Unified Update Mechanism")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train both models
    dfa_losses, dfa_accuracies = train_model('dfa', num_epochs=50, learning_rate=0.01)
    
    # Reset seed for fair comparison
    torch.manual_seed(42)
    np.random.seed(42)
    
    backprop_losses, backprop_accuracies = train_model('backprop', num_epochs=50, learning_rate=0.01)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(dfa_losses, label='DFA', linewidth=2)
    ax1.plot(backprop_losses, label='Backprop', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(dfa_accuracies, label='DFA', linewidth=2)
    ax2.plot(backprop_accuracies, label='Backprop', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dfa_vs_backprop_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'dfa_vs_backprop_comparison.png'")
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    print(f"DFA     - Final Loss: {dfa_losses[-1]:.4f}, Final Accuracy: {dfa_accuracies[-1]:.4f}")
    print(f"Backprop - Final Loss: {backprop_losses[-1]:.4f}, Final Accuracy: {backprop_accuracies[-1]:.4f}")
    
    # Check if backprop is still underperforming
    if backprop_losses[-1] > dfa_losses[-1]:
        print(f"\n⚠️  Backprop still underperforming by {backprop_losses[-1] - dfa_losses[-1]:.4f} loss")
        print("   Further investigation needed...")
    else:
        print(f"\n✅ Performance gap resolved! Backprop performing as expected.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
