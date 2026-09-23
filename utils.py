import torch
import time
import math
import argparse
import os

from reproducibility import restore_rng_state

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
    "4_palindrome_dataset_vary_length": "text",
    "1_small_palindrome_dataset_vary_length": "text", 
    "2_small_palindrome_dataset_vary_length": "text", 
    "3_small_palindrome_dataset_vary_length": "text", 
    "4_small_palindrome_dataset_vary_length": "text", }

def get_charset(dataset_name):

    if "small" in dataset_name:
        set = "23. "
        return set
    elif dataset_name == "long_range_memory_dataset" or any(tag in dataset_name for tag in ("palindrome_dataset", "resequence")):
    # if (dataset_name == "long_range_memory_dataset") or (dataset_name == "palindrome_dataset") or (dataset_name == "palindrome_dataset_vary_length") or "resequence" in dataset_name:
        set = "0?!123,. "
        # print(f"Using a custom charset for long_range_memory_dataset or palindrome_dataset, of length {len(set)}")
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


# Place these functions near the top of train.py or in utils.py

def save_checkpoint(state_dict, checkpoint_dir, filename="checkpoint.pth"):
    """Saves checkpoint to disk"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state_dict, filepath)
    print(f"Checkpoint saved to {filepath}")

def read_checkpoint(checkpoint_path):
    """Reads a checkpoint dict from disk onto the CPU."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    # Load checkpoint to CPU first to avoid GPU OOM issues with mismatched models/devices
    return torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Config entries that are objects rather than settings, so they are not diffed on resume.
_UNDIFFED_CONFIG_KEYS = {'criterion', 'pe_matrix'}

def print_config_diff(config, loaded_config):
    """Prints every config field that differs between the checkpoint and this run."""
    missing = object()
    keys = sorted((set(config) | set(loaded_config)) - _UNDIFFED_CONFIG_KEYS)
    diffs = [
        (key, loaded_config.get(key, missing), config.get(key, missing))
        for key in keys
        if loaded_config.get(key, missing) != config.get(key, missing)
    ]
    if not diffs:
        print("Config matches the checkpoint.")
        return
    print("Config differences (checkpoint -> this run):")
    for key, loaded_value, current_value in diffs:
        loaded_text = "(not in checkpoint)" if loaded_value is missing else repr(loaded_value)
        current_text = "(not in this run)" if current_value is missing else repr(current_value)
        print(f"  {key}: {loaded_text} -> {current_text}")

def load_checkpoint(checkpoint_path, model, config, optimizer=None, device='cpu', checkpoint=None):
    """Loads checkpoint from disk (or from `checkpoint`, if it was already read)"""
    print(f"=> Loading checkpoint '{checkpoint_path}'")
    if checkpoint is None:
        checkpoint = read_checkpoint(checkpoint_path)
    loaded_config = checkpoint.get('config', {}) # The config used for this checkpoint
    print_config_diff(config, loaded_config)

    compatibility_defaults = {
        'n_hidden': None,
        'n_layers': None,
        'updater': None,
        'charset_size': None,
        'seed': None,
        'deterministic': False,
    }
    mismatches = [
        (key, config.get(key, default), loaded_config.get(key, default))
        for key, default in compatibility_defaults.items()
        if config.get(key, default) != loaded_config.get(key, default)
    ]
    # Checkpoints written before the naming cleanup store model_type 'ethereal'.
    loaded_model_type = {'ethereal': 'ephemeral'}.get(loaded_config.get('model_type'), loaded_config.get('model_type'))
    if loaded_model_type is not None and loaded_model_type != config.get('model_type'):
        mismatches.append(('model_type', config.get('model_type'), loaded_config.get('model_type')))
    # forgetting_factor is restored from the state dict, so a new --forget_rate would be
    # silently ignored (while W&B records it). Checked only if the checkpoint recorded it.
    if 'forget_rate' in loaded_config and config.get('forget_rate') != loaded_config['forget_rate']:
        mismatches.append(('forget_rate', config.get('forget_rate'), loaded_config['forget_rate']))
    # slurm_run.sh keys checkpoints by job name and always resumes, so a reused job name for a
    # new experiment would silently continue an old checkpoint. A different dataset or
    # learning rate means a different experiment. Checked only if the checkpoint recorded them.
    for key in ('dataset', 'learning_rate'):
        if key in loaded_config and config.get(key) != loaded_config[key]:
            mismatches.append((key, config.get(key), loaded_config[key]))
    if mismatches:
        print("--------------------------------------------------------------------")
        print("ERROR: Checkpoint configuration mismatch!")
        for key, current_value, loaded_value in mismatches:
            print(f"  Current {key}: {current_value}, Loaded: {loaded_value}")
        print("  Please verify settings or delete checkpoint if starting a new experiment.")
        print("--------------------------------------------------------------------")
        raise RuntimeError("Checkpoint configuration mismatch - aborting run.")
    
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

    restore_rng_state(checkpoint)

    print(f"=> Loaded checkpoint '{checkpoint_path}' (iteration {start_iter})")


    return model, optimizer, start_iter, main_state, loaded_config
