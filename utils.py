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


# Version of the training mechanics, saved in every checkpoint as 'code_version'. A resume
# refuses a checkpoint whose version differs from this one (or that has none), so the run
# must start fresh.
#
# BUMP THIS whenever training mechanics change in a way that makes an in-flight run's
# continuation meaningless: the forward pass or layer layout, what an updater changes or by
# how much, forgetting, wiping, normalization or clamping, the loss, or the meaning of a
# saved tensor. Pure renames that load_checkpoint maps, logging and refactors that keep the
# golden trace (tests/fixtures/training_traces.json) identical do not need a bump. As a rule
# of thumb, a commit that regenerates the golden trace with changed values bumps it.
#
# History: 1 was never written (checkpoints before versioning have no code_version and are
# refused like any other mismatch). 2: introduced, 2026-09. 3: --unit_norm_weights
# normalises each sequence's [out, in] slice separately. 4: Elman layout, y_t = i2o(h_t)
# (i2o and self_grad take hidden_size inputs; i2h learns under DFA and per-step backprop).
# 5: DFA for the SimpleRNN baseline (--model_type rnn --updater dfa now trains every layer,
# and its state dict holds the DFA feedback matrices).
CHECKPOINT_CODE_VERSION = 5


def check_checkpoint_code_version(checkpoint, checkpoint_path="<checkpoint>"):
    """Raises unless the checkpoint was written by code with this CHECKPOINT_CODE_VERSION."""
    saved = checkpoint.get('code_version')
    if saved == CHECKPOINT_CODE_VERSION:
        return
    found = "has no code_version (written before checkpoints were versioned)" if saved is None \
        else f"has code_version {saved}"
    raise RuntimeError(
        f"Checkpoint {checkpoint_path} {found}, but this code is CHECKPOINT_CODE_VERSION "
        f"{CHECKPOINT_CODE_VERSION}. The training mechanics changed in between, so continuing that "
        "run would not be meaningful: start the run fresh (use a new --checkpoint_dir or delete the "
        "old checkpoint, and do not pass --resume/--resume_checkpoint for it).")


def save_checkpoint(state_dict, checkpoint_dir, filename="checkpoint.pth"):
    """Saves checkpoint to disk, stamped with code_version = CHECKPOINT_CODE_VERSION
    unless the caller already set it."""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save({**state_dict, 'code_version': state_dict.get('code_version', CHECKPOINT_CODE_VERSION)}, filepath)
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

# Config (CLI) keys before the naming cleanup (2026-09), and the keys that replaced them.
LEGACY_CONFIG_KEYS = {
    'plast_clip': 'plasticity',
    'plast_proportion': 'ephemeral_fraction',
    'clip_weights': 'weight_clamp',
    'normalize': 'unit_norm_weights',
}
# Removed flags that never did anything; dropped from old configs so they do not show in the diff.
REMOVED_CONFIG_KEYS = ('plast_learning_rate', 'imprint_rate')

def upgrade_legacy_config(loaded_config):
    """Returns a checkpoint config with old CLI keys renamed, so the resume checks and the
    config diff compare like with like. grad_clip becomes ephemeral_update_clamp for the
    ephemeral model and grad_norm_clip for the rnn baseline (the only use each model made of
    it), and the other one gets its default, 0, which is what the old flags give today."""
    upgraded = {}
    for key, value in loaded_config.items():
        if key in REMOVED_CONFIG_KEYS:
            continue
        new_key = LEGACY_CONFIG_KEYS.get(key, key)
        if new_key != key and new_key in loaded_config:
            raise RuntimeError(f"Checkpoint config has both {key} and {new_key}.")
        upgraded[new_key] = value
    if 'grad_clip' in upgraded:
        model_type = {'ethereal': 'ephemeral'}.get(upgraded.get('model_type'), upgraded.get('model_type'))
        targets = {'ephemeral': 'ephemeral_update_clamp', 'rnn': 'grad_norm_clip'}
        if model_type in targets:  # otherwise left as grad_clip, and the diff shows it
            if any(key in upgraded for key in targets.values()):
                raise RuntimeError("Checkpoint config has both grad_clip and a key that replaced it.")
            grad_clip = upgraded.pop('grad_clip')
            for name, target in targets.items():
                upgraded[target] = grad_clip if name == model_type else 0
    return upgraded

# State-dict names of EphemeralLinear tensors before the naming cleanup (2026-09).
LEGACY_STATE_DICT_NAMES = {
    'candidate_weights': 'per_sample_weights',
    'mask': 'ephemeral_mask',
    'last_high_plast_update_norm': 'last_ephemeral_step_norm',
    'last_low_plast_update_norm': 'last_slow_step_norm',
}
# Stored by old checkpoints, now computed as forget_rate * ephemeral_mask.
LEGACY_FORGETTING_FACTOR = 'forgetting_factor'

def upgrade_legacy_state_dict(state_dict, model):
    """Maps a checkpoint's old EphemeralLinear tensor names to the current ones.

    Only keys whose module is an EphemeralLinear in `model` are renamed. An old
    forgetting_factor tensor must equal forget_rate * mask for that layer (anything else is
    an unexpected state and raises), and is then dropped. Returns the new state dict and the
    dropped forgetting_factor keys, in checkpoint order.
    """
    from ephemeral_model import EphemeralLinear  # utils is imported by ephemeral_model

    modules = dict(model.named_modules())
    upgraded, dropped = {}, []
    for key, value in state_dict.items():
        prefix, _, name = key.rpartition('.')
        if not isinstance(modules.get(prefix), EphemeralLinear):
            upgraded[key] = value
        elif name in LEGACY_STATE_DICT_NAMES:
            new_key = f"{prefix}.{LEGACY_STATE_DICT_NAMES[name]}"
            if new_key in state_dict:
                raise RuntimeError(f"Checkpoint has both {key} and {new_key}.")
            upgraded[new_key] = value
        elif name == LEGACY_FORGETTING_FACTOR:
            mask = state_dict.get(f"{prefix}.mask", state_dict.get(f"{prefix}.ephemeral_mask"))
            if mask is None:
                raise RuntimeError(f"Checkpoint has {key} but no mask for {prefix}.")
            forget_rate = modules[prefix].forget_rate
            expected = forget_rate * mask.to(value.device)
            if value.shape != expected.shape or not torch.equal(value, expected.to(value.dtype)):
                raise RuntimeError(
                    f"Checkpoint {key} is not forget_rate ({forget_rate}) on the mask, so it cannot be "
                    "dropped safely. It may come from a run with --normalize before 2026-09 (which "
                    "rescaled it), from plast_proportion < 0.01 before mask_tier_two was removed, or "
                    "from a different --forget_rate.")
            dropped.append(key)
        else:
            upgraded[key] = value
    return upgraded, dropped

def _drop_optimizer_params(optimizer_state, saved_state_dict, dropped_keys, model):
    """Removes the dropped parameters from a saved optimizer state dict.

    The optimizer was built from model.parameters(), whose order is the state dict's
    parameter order (buffers excluded), so each dropped key's position there is its index in
    the optimizer's flattened param groups.
    """
    if not dropped_keys:
        return optimizer_state
    buffer_names = {name for name, _ in model.named_buffers()}
    parameter_keys = [key for key in saved_state_dict if key not in buffer_names]
    drop_positions = {parameter_keys.index(key) for key in dropped_keys}
    position, groups, state = 0, [], dict(optimizer_state['state'])
    for group in optimizer_state['param_groups']:
        kept = []
        for param_id in group['params']:
            if position in drop_positions:
                if state.pop(param_id, None) is not None:
                    raise RuntimeError(f"Optimizer holds state for a dropped parameter at position {position}.")
            else:
                kept.append(param_id)
            position += 1
        groups.append({**group, 'params': kept})
    if position != len(parameter_keys):
        raise RuntimeError(
            f"Optimizer state has {position} parameters but the checkpoint's model has {len(parameter_keys)}.")
    return {'state': state, 'param_groups': groups}

def load_checkpoint(checkpoint_path, model, config, optimizer=None, device='cpu', checkpoint=None):
    """Loads checkpoint from disk (or from `checkpoint`, if it was already read)"""
    print(f"=> Loading checkpoint '{checkpoint_path}'")
    if checkpoint is None:
        checkpoint = read_checkpoint(checkpoint_path)
    # Before anything else: a checkpoint from other training mechanics is never continued.
    check_checkpoint_code_version(checkpoint, checkpoint_path)
    # The config used for this checkpoint, with pre-2026-09 CLI names mapped to today's
    loaded_config = upgrade_legacy_config(checkpoint.get('config', {}))
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
    # A new --forget_rate would change a resumed run's decay mid-run (checkpoints from before
    # the rename stored the per-entry rate as forgetting_factor, which was restored and so
    # silently ignored the new value). Checked only if the checkpoint recorded it.
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

    # Map old tensor names first. Then every key must match: a missing key would leave a
    # freshly initialised tensor in a "resumed" model, and an unexpected one would be ignored.
    saved_state_dict = checkpoint["model_state_dict"]
    state_dict, dropped_keys = upgrade_legacy_state_dict(saved_state_dict, model)
    if dropped_keys:
        print(f"Dropped stored forgetting_factor (checked equal to forget_rate * mask): {dropped_keys}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint state dict does not match the model after mapping old names: "
            f"missing keys {missing}, unexpected keys {unexpected}.")
    model.to(device) # Move model to target device after loading

    if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(_drop_optimizer_params(
            checkpoint['optimizer_state_dict'], saved_state_dict, dropped_keys, model))
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
