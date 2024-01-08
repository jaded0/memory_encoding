import torch
from torch.utils.data import Dataset
import os
import random

class TextDataset(Dataset):
    def __init__(self, directory, sequence_length):
        self.directory = directory
        self.sequence_length = sequence_length
        self.charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;'\"?! "
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.charset)}
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith("_tokens.txt")]
        self.file_sizes = [os.path.getsize(f) - self.sequence_length for f in self.file_paths]

    def __len__(self):
        return sum(self.file_sizes)

    def __getitem__(self, idx):
        file_idx, char_idx = self.map_index(idx)
        file_path = self.file_paths[file_idx]
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            file.seek(char_idx)
            sequence = file.read(self.sequence_length + 1)
            input_seq = sequence[:-1].replace('\n', ' ')

        tensor = torch.tensor([self.char_to_idx.get(char, 0) for char in input_seq], dtype=torch.long)
        return input_seq, tensor

    def map_index(self, idx):
        cumulative_size = 0
        for file_idx, size in enumerate(self.file_sizes):
            if idx < cumulative_size + size:
                char_idx = random.randint(0, size)  # Generate a random starting point
                return file_idx, char_idx
            cumulative_size += size
        raise IndexError("Index out of range")
