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

def save_model_data(model, activations, epoch):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):  # Replace with your layer types if different
            weight_path = os.path.join(save_dir, f'weights_{name}_epoch{epoch}.pt')
            activation_path = os.path.join(save_dir, f'activations_{name}_epoch{epoch}.pt')
            save_tensor(layer.weight, weight_path)
            if name in activations:
                save_tensor(activations[name], activation_path)

# Example usage
# rnn = SimpleRNN(input_size, n_hidden, output_size, 3)
# activations = register_hooks(rnn)
# ...
# Inside your training loop, after each forward pass
# save_model_data(rnn, activations, epoch, batch_idx)
import os
import torch
import matplotlib.pyplot as plt

def load_tensor(path):
    if os.path.exists(path):
        return torch.load(path)
    else:
        return None

def plot_tensors(weight_tensor, activation_tensor, layer_name, instance):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plotting weights
    if weight_tensor is not None:
        ax = axes[0]
        im = ax.imshow(weight_tensor.detach().numpy(), cmap='viridis')
        ax.set_title(f'Weights of {layer_name}')
        fig.colorbar(im, ax=ax)

    print(activation_tensor.shape)
    # Plotting activations
    if activation_tensor is not None:
        ax = axes[1]
        # Remove singleton dimensions
        activation_tensor = activation_tensor.squeeze()
        if activation_tensor.ndim == 1:
            # Convert tensor to numpy array and plot as a bar chart
            ax.bar(range(len(activation_tensor)), activation_tensor.detach().numpy())
        else:
            # For multi-dimensional activations, retain the heatmap representation
            im = ax.imshow(activation_tensor.detach().numpy(), cmap='viridis')
            fig.colorbar(im, ax=ax)
        ax.set_title(f'Activations of {layer_name}')

    plt.suptitle(f'Layer {layer_name} at Instance {instance}')
    plt.show()

def visualize_model_data(layer_name, instance):
    weight_path = os.path.join(save_dir, f'weights_{layer_name}_epoch{instance}.pt')
    activation_path = os.path.join(save_dir, f'activations_{layer_name}_epoch{instance}.pt')

    weight_tensor = load_tensor(weight_path)
    activation_tensor = load_tensor(activation_path)
    # print(weight_tensor)
    # print(activation_tensor)
    plot_tensors(weight_tensor, activation_tensor, layer_name, instance)

# Example usage: visualize_model_data('layer1', 10)



import matplotlib.pyplot as plt

def visualize_all_layers(model, instance):
    # Assuming model layers are stored in a list or accessible by named_modules()
    layer_names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]

    # Create a large figure
    num_layers = len(layer_names)
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, num_layers * 4))  # Adjust the size as needed

    for i, layer_name in enumerate(layer_names):
        # Load tensors for each layer
        weight_path = os.path.join(save_dir, f'weights_{layer_name}_epoch{instance}.pt')
        activation_path = os.path.join(save_dir, f'activations_{layer_name}_epoch{instance}.pt')
        weight_tensor = load_tensor(weight_path)
        activation_tensor = load_tensor(activation_path).squeeze()

        # Plot weights
        if weight_tensor is not None:
            ax = axes[i][0]
            im = ax.imshow(weight_tensor.detach().numpy(), cmap='viridis')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Weights of {layer_name}')

        # Plot activations as bar chart
        if activation_tensor is not None and activation_tensor.ndim == 1:
            ax = axes[i][1]
            ax.bar(range(len(activation_tensor)), activation_tensor.detach().numpy())
            ax.set_title(f'Activations of {layer_name}')

    plt.tight_layout()
    plt.show()

# Example usage: visualize_all_layers(rnn, 1080)
