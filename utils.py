import torch
import time
import math
import argparse

dataset_keys = {
    "roneneldan/tinystories": "text",
    "jbrazzy/baby_names": "Names"
}

# Your charset
charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;:'\"?!\n- "
char_to_idx = {char: idx for idx, char in enumerate(charset)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
n_characters = len(charset)

def filter_text(examples, dataset_name):
    """Filter out characters not in the charset for a batch of texts."""
    key = dataset_keys.get(dataset_name)
    return {'text': [''.join([char for char in text if char in charset]) for text in examples[key]]}

def text_to_indices(examples, dataset_name):
    key = dataset_keys.get(dataset_name)
    tensors = [torch.tensor([char_to_idx[char] for char in text], dtype=torch.long) for text in examples[key]]
    return {'tensor': tensors}

def text_to_indices_and_one_hot(examples, dataset_name):
    key = dataset_keys.get(dataset_name)
    one_hot_tensors = []
    for text in examples[key]:
        indices = [char_to_idx[char] for char in text]
        one_hot = torch.nn.functional.one_hot(torch.tensor(indices, dtype=torch.long), num_classes=n_characters).type(torch.float)
        one_hot_tensors.append(one_hot)
    return {'onehot_tensor': one_hot_tensors}

# def collate_fn(batch):
#     """ Collate function for DataLoader """
#     item = batch[0]
#     text = item['text']
#     tensor = item['tensor']
#     tensor = torch.tensor(tensor)
#     return text, tensor
def collate_fn(batch):
    """ Collate function for DataLoader """
    item = batch[0]
    text = item['text']
    tensor = item['tensor']
    tensor = torch.tensor(tensor)
    onehot_tensor = item['onehot_tensor']
    onehot_tensor = torch.tensor(onehot_tensor)
    return text, tensor, onehot_tensor


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
