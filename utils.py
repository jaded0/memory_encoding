import torch
import time
import math

# Your charset
charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;:'\"?!\n- "
char_to_idx = {char: idx for idx, char in enumerate(charset)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
n_characters = len(charset)

def filter_text(examples):
    """Filter out characters not in the charset for a batch of texts."""
    return {'text': [''.join([char for char in text if char in charset]) for text in examples['text']]}

def text_to_indices(examples):
    tensors = [torch.tensor([char_to_idx[char] for char in text], dtype=torch.long) for text in examples['text']]
    return {'tensor': tensors}

def collate_fn(batch):
    """ Collate function for DataLoader """
    item = batch[0]
    text = item['text']
    tensor = item['tensor']
    tensor = torch.tensor(tensor)
    return text, tensor

def randomTrainingExample(dataloader):
    """ Get a random training example """
    for text, tensor in dataloader:
        return text, tensor

def timeSince(since):
    """ Calculate elapsed time """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
