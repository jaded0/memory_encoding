import torch
import time
import math
import argparse

dataset_keys = {
    "roneneldan/tinystories": "text",
    "jbrazzy/baby_names": "Names",
    "brucewlee1/htest-palindrome": "centerpiece",
    "long_range_memory_dataset": "text",
    "1_resequence": "text",
    "2_resequence": "text",
    "3_resequence": "text",
    "4_resequence": "text",
}

def get_charset(dataset_name):
    if dataset_name == "long_range_memory_dataset" or "resequence" in dataset_name:
        return "0?!123,."
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
