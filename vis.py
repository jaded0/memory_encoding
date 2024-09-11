import os
import torch
import torch.nn as nn
import wandb

# Create a directory for saving data
save_dir = '/tmp/model_data'
os.makedirs(save_dir, exist_ok=True)

def save_tensor(tensor, path, scale=1.0, precision=torch.float16):
    # Scale and reduce precision
    scaled_tensor = (tensor * scale).to(precision)
    torch.save(scaled_tensor, path)

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

def save_model_data(model, activations, epoch, track):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):  # Replace with your layer types if different
            weight_path = os.path.join(save_dir, f'weights_{name}_epoch{epoch}.pt')
            activation_path = os.path.join(save_dir, f'activations_{name}_epoch{epoch}.pt')
            plasticity_path = os.path.join(save_dir, f'plasticity_{name}_epoch{epoch}.pt')

            save_tensor(layer.weight, weight_path)
            if name in activations:
                save_tensor(activations[name][0], activation_path) # 0 bc I want the activations from only one of the batch of data
            
            if hasattr(layer, 'plasticity'):
                save_tensor(layer.plasticity, plasticity_path)
    
    # plot to image and log to wandb
    visualize_all_layers_and_save(model, epoch, os.path.join(save_dir, f'visualization_{epoch}.png'))
    if track: 
        wandb.log({"model_evolution": wandb.Image(os.path.join(save_dir, f'visualization_{epoch}.png'))},commit=False)


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

def plot_tensors(weight_tensor, activation_tensor, plasticity_tensor, layer_name, instance):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Update to 1x3 layout

    # Plotting weights
    if weight_tensor is not None:
        ax = axes[0]
        im = ax.imshow(weight_tensor.detach().numpy(), cmap='viridis')
        ax.set_title(f'Weights of {layer_name}')
        fig.colorbar(im, ax=ax)

    # Plotting activations
    if activation_tensor is not None:
        ax = axes[1]
        activation_tensor = activation_tensor.squeeze()
        if activation_tensor.ndim == 1:
            ax.bar(range(len(activation_tensor)), activation_tensor.detach().numpy())
        else:
            im = ax.imshow(activation_tensor.detach().numpy(), cmap='viridis')
            fig.colorbar(im, ax=ax)
        ax.set_title(f'Activations of {layer_name}')

    # Plotting plasticity
    if plasticity_tensor is not None:
        ax = axes[2]
        im = ax.imshow(plasticity_tensor.detach().numpy(), cmap='viridis')
        ax.set_title(f'Plasticity of {layer_name}')
        fig.colorbar(im, ax=ax)

    plt.suptitle(f'Layer {layer_name} at Instance {instance}')
    plt.show()


def visualize_model_data(layer_name, instance):
    weight_path = os.path.join(save_dir, f'weights_{layer_name}_epoch{instance}.pt')
    activation_path = os.path.join(save_dir, f'activations_{layer_name}_epoch{instance}.pt')
    plasticity_path = os.path.join(save_dir, f'plasticity_{layer_name}_epoch{instance}.pt')

    weight_tensor = load_tensor(weight_path)
    activation_tensor = load_tensor(activation_path)
    plasticity_tensor = load_tensor(plasticity_path)
    
    plot_tensors(weight_tensor, activation_tensor, plasticity_tensor, layer_name, instance)


# Example usage: visualize_model_data('layer1', 10)



import matplotlib.pyplot as plt

def visualize_all_layers_and_save(model, instance, save_path):
    layer_names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]
    num_layers = len(layer_names)
    fig, axes = plt.subplots(num_layers, 3, figsize=(18, num_layers * 4))  # Update to 3 columns

    for i, layer_name in enumerate(layer_names):
        weight_path = os.path.join(save_dir, f'weights_{layer_name}_epoch{instance}.pt')
        activation_path = os.path.join(save_dir, f'activations_{layer_name}_epoch{instance}.pt')
        plasticity_path = os.path.join(save_dir, f'plasticity_{layer_name}_epoch{instance}.pt')

        weight_tensor = load_tensor(weight_path)
        activation_tensor = load_tensor(activation_path).squeeze()
        plasticity_tensor = load_tensor(plasticity_path)

        if weight_tensor is not None:
            ax = axes[i][0]
            im = ax.imshow(weight_tensor.cpu().detach().numpy(), cmap='viridis')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Weights of {layer_name}')

        if activation_tensor is not None and activation_tensor.ndim == 1:
            ax = axes[i][1]
            ax.bar(range(len(activation_tensor)), activation_tensor.cpu().detach().numpy())
            ax.set_title(f'Activations of {layer_name}')

        if plasticity_tensor is not None:
            ax = axes[i][2]
            im = ax.imshow(plasticity_tensor.cpu().detach().numpy(), cmap='viridis')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Plasticity of {layer_name}')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


    # Optionally, show the plot in Jupyter Notebook
    # plt.show()

# Example usage: visualize_all_layers(rnn, 1080)

import os
import imageio


def create_animation_from_visualizations(model, save_dir, output_path, frame_duration=0.3, format='gif'):
    # Find all unique instances
    instances = set()
    for filename in os.listdir(save_dir):
        parts = filename.split('_')
        if len(parts) > 3 and parts[0] in ['weights', 'activations']:
            # Extract the part with the instance number and remove the file extension
            instance_part = parts[-1].split('.')[0]
            # Further split if needed and extract the number
            instance_number = instance_part.split('epoch')[-1]
            instances.add(int(instance_number))

    # Generate and save visualizations for each instance
    image_paths = []
    for instance in sorted(instances):
        save_path = os.path.join(save_dir, f'visualization_{instance}.png')
        visualize_all_layers_and_save(model, instance, save_path)
        image_paths.append(save_path)

    # Compile images into an animation (GIF or MP4)
    if format == 'gif':
        with imageio.get_writer(output_path, mode='I', duration=frame_duration, loop=0) as writer:
            for image_path in image_paths:
                image = imageio.imread(image_path)
                writer.append_data(image)
    elif format == 'mp4':
        with imageio.get_writer(output_path, mode='I', fps=1/frame_duration) as writer:
            for image_path in image_paths:
                image = imageio.imread(image_path)
                writer.append_data(image)

    # Clean up saved images if desired
    # for image_path in image_paths:
    #     os.remove(image_path)

# Example usage
# create_animation_from_visualizations(rnn, 'model_data', 'model_evolution.mp4', format='mp4')
