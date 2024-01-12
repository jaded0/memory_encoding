import torch
from torch.utils.data import Dataset
import os
import random
from torch.utils.data import DataLoader


class TextDataset(Dataset):
    def __init__(self, directory, sequence_length):
        self.directory = directory
        self.sequence_length = sequence_length
        self.charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;'\"?! "
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.charset)}
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith("_tokens.txt")]
        self.file_sizes = [os.path.getsize(f) - self.sequence_length for f in self.file_paths]
        self.n_characters = len(self.charset)

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


    # Find letter index from all_letters, e.g. "a" = 0
    def letterToIndex(letter):
        return text_dataset.char_to_idx[letter]

    # Just for demonstration, turn a letter into a <1 x n_characters> Tensor
    def letterToTensor(self, letter):
        tensor = torch.zeros(1, self.n_characters)
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x self.n_characters>,
    # or an array of one-hot letter vectors
    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_characters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor


if __name__ == "__main__":
    # Instantiate the dataset
    text_dataset = TextDataset(directory='data/SPGC-tokens-2018-07-18/', sequence_length=100)
    print(f"Dataset created with {len(text_dataset)} sequences.")
    # Create a DataLoader without a sampler
    dataloader = DataLoader(text_dataset, batch_size=1)

    # Iterate over a few batches and print their contents
    for i, (sequences, inputs) in enumerate(dataloader):
        if i >= 2:  # Adjust this value to see more/less batches
            break

        print(f"\nBatch {i+1}")
        print(f"Inputs shape: {inputs.shape}")

        # Optionally print the actual sequences (comment out if too verbose)
        sequence = ''.join([text_dataset.idx_to_char[int(idx)] for idx in inputs[0]])
        # target = text_dataset.idx_to_char[int(targets[0])]
        print(f"Sequence: {sequence}")
    
    print(text_dataset.letterToTensor('J'))

    print(text_dataset.lineToTensor('Jones').size())
