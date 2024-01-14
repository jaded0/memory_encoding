import os
import torch
import torch.nn as nn

# Create a directory for saving data
save_dir = 'model_data'
os.makedirs(save_dir, exist_ok=True)

def save_tensor(tensor, path):
    torch.save(tensor, path)

def register_hooks(model):
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):  # Replace with your layer types if different
            layer.register_forward_hook(get_activation(name))

    return activations

def save_model_data(model, activations, epoch, batch_idx):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):  # Replace with your layer types if different
            weight_path = os.path.join(save_dir, f'weights_{name}_epoch{epoch}_batch{batch_idx}.pt')
            activation_path = os.path.join(save_dir, f'activations_{name}_epoch{epoch}_batch{batch_idx}.pt')
            save_tensor(layer.weight, weight_path)
            if name in activations:
                save_tensor(activations[name], activation_path)

# Example usage
# rnn = SimpleRNN(input_size, n_hidden, output_size, 3)
# activations = register_hooks(rnn)
# ...
# Inside your training loop, after each forward pass
# save_model_data(rnn, activations, epoch, batch_idx)
