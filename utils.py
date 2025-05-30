import torch
import time
import math
import argparse
import os
import numpy as np

dataset_keys = {
    "roneneldan/tinystories": "text",
    "jbrazzy/baby_names": "Names",
    "brucewlee1/htest-palindrome": "centerpiece",
    "long_range_memory_dataset": "text",
    "1_resequence": "text",
    "2_resequence": "text",
    "3_resequence": "text",
    "4_resequence": "text",
    "palindrome_dataset": "text",
    "palindrome_dataset_vary_length": "text",
    "1_palindrome_dataset_vary_length": "text", 
    "2_palindrome_dataset_vary_length": "text", 
    "3_palindrome_dataset_vary_length": "text", 
    "4_palindrome_dataset_vary_length": "text", }

def get_charset(dataset_name):
    if dataset_name == "long_range_memory_dataset" or any(tag in dataset_name for tag in ("palindrome_dataset", "resequence")):
    # if (dataset_name == "long_range_memory_dataset") or (dataset_name == "palindrome_dataset") or (dataset_name == "palindrome_dataset_vary_length") or "resequence" in dataset_name:
        set = "0?!123,. "
        print(f"Using a custom charset for long_range_memory_dataset or palindrome_dataset, of length {len(set)}")
        return set
    else:
        return " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;:'\"?!\n-"

def initialize_charset(dataset_name):
    charset = get_charset(dataset_name)
    # print(f"length of the charset is {len(charset)}")

    char_to_idx = {char: idx for idx, char in enumerate(charset)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    n_characters = len(charset)
    return charset, char_to_idx, idx_to_char, n_characters

charset, char_to_idx, idx_to_char, n_characters = initialize_charset("roneneldan/tinystories")  # default dataset

def filter_text(examples, dataset_name):
    """Filter out characters not in the charset and pad sequences to ensure a minimum length of 3 characters."""
    key = dataset_keys.get(dataset_name)
    filtered_and_padded_texts = []
    charset, _, _, _ = initialize_charset(dataset_name)

    for text in examples[key]:
        # Filter out characters not in the charset
        filtered_text = ''.join([char for char in text if char in charset])
        if 'á' in filtered_text:
            print(f"problem with {filtered_text}")
        # Pad the text with spaces if it's shorter than 3 characters
        while len(filtered_text) < 3:
            filtered_text += ' '

        filtered_and_padded_texts.append(filtered_text)

    return {'text': filtered_and_padded_texts}


def text_to_indices(examples, dataset_name):
    key = dataset_keys.get(dataset_name)
    _, char_to_idx, _, _ = initialize_charset(dataset_name)
    tensors = [torch.tensor([char_to_idx[char] for char in text], dtype=torch.long) for text in examples['text']]
    return {'tensor': tensors}

def text_to_indices_and_one_hot(examples, dataset_name):
    key = dataset_keys.get(dataset_name)
    _, char_to_idx, _, n_characters = initialize_charset(dataset_name)
    one_hot_tensors = []
    for text in examples['text']:
        indices = [char_to_idx[char] for char in text]
        one_hot = torch.nn.functional.one_hot(torch.tensor(indices, dtype=torch.long), num_classes=n_characters).type(torch.float)
        one_hot_tensors.append(one_hot)
    return {'onehot_tensor': one_hot_tensors}

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """ Collate function for DataLoader """
    texts = [item['text'] for item in batch]
    tensors = pad_sequence([torch.tensor(item['tensor']) for item in batch], batch_first=True)
    onehot_tensors = pad_sequence([torch.tensor(item['onehot_tensor']) for item in batch], batch_first=True)
    return texts, tensors, onehot_tensors

def randomTrainingExample(dataloader):
    """ Get a random training example """
    for text, tensor, onehot_line_tensor in dataloader:
        return text, tensor, onehot_line_tensor

def timeSince(since):
    """ Calculate elapsed time """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Place these functions near the top of hebby.py or in utils.py

def save_checkpoint(state_dict, checkpoint_dir, filename="checkpoint.pth"):
    """Saves checkpoint to disk"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state_dict, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(checkpoint_path, model, config, optimizer=None, device='cpu'):
    """Loads checkpoint from disk"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"=> Loading checkpoint '{checkpoint_path}'")
    # Load checkpoint to CPU first to avoid GPU OOM issues with mismatched models/devices
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    loaded_config = checkpoint.get('config', {}) # The config used for this checkpoint

    if loaded_config.get('n_hidden') != config['n_hidden'] or \
        loaded_config.get('n_layers') != config['n_layers'] or \
        loaded_config.get('update_rule') != config['update_rule'] or \
        loaded_config.get('charset_size') != config['charset_size']:
        print("--------------------------------------------------------------------")
        print("ERROR: Checkpoint loaded with potentially incompatible configuration!")
        print(f"  Current n_hidden: {config['n_hidden']}, Loaded: {loaded_config.get('n_hidden')}")
        print(f"  Current n_layers: {config['n_layers']}, Loaded: {loaded_config.get('n_layers')}")
        print(f"  Current update_rule: {config['update_rule']}, Loaded: {loaded_config.get('update_rule')}")
        print(f"  Current charset_size: {config['charset_size']}, Loaded: {loaded_config.get('charset_size')}")
        print("  Please verify settings or delete checkpoint if starting a new experiment.")
        print("--------------------------------------------------------------------")
        # Decide whether to proceed or exit, e.g., sys.exit("Config mismatch with checkpoint.")
        raise RuntimeError("Checkpoint configuration mismatch – aborting run.")
    
    # saved_vocab = checkpoint['config']['charset_size']
    # current_vocab = len(get_charset(args.dataset))
    # assert saved_vocab == current_vocab, (
    #     f"Vocabulary changed {saved_vocab} → {current_vocab}; "
    #     "old checkpoints will not load."
    # )

    # allow new logging-only parameters to remain at their default values
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing:
        print(f"✔  missing keys initialised fresh: {missing}")
    if unexpected:
        print(f"⚠️  unexpected keys in checkpoint: {unexpected}")
    model.to(device) # Move model to target device after loading

    if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


    start_iter = checkpoint.get('iter', 1)
    main_state = checkpoint.get('main_program_state', {}) # Your custom state dict from main
    # loaded_config = checkpoint.get('config', {}) # The config used for this checkpoint

    # RNG states
    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu()) # RNG state must be on CPU
    if 'numpy_rng_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_rng_state'])
    # import random # if you use it
    # if 'python_rng_state' in checkpoint:
    #     random.setstate(checkpoint['python_rng_state'])

    print(f"=> Loaded checkpoint '{checkpoint_path}' (iteration {start_iter})")


    return model, optimizer, start_iter, main_state, loaded_config