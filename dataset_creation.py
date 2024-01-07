# dataset_creation.py
import os
import torch
import pandas as pd

def load_and_preprocess_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Preprocess text
    text = text.replace('\n', ' ')
    return text

def create_dataset(directory, sequence_length=100):
    all_text = ''
    for filename in os.listdir(directory):
        if filename.endswith("_tokens.txt"):
            file_path = os.path.join(directory, filename)
            # print(f"Reading file: {file_path}")  # Debug print
            all_text += load_and_preprocess_text(file_path)
    
    characters = list(set(all_text))
    char_to_idx = {char: idx for idx, char in enumerate(characters)}
    idx_to_char = {idx: char for idx, char in enumerate(characters)}

    inputs = []
    targets = []
    for i in range(len(all_text) - sequence_length):
        input_seq = all_text[i:i + sequence_length]
        target_char = all_text[i + sequence_length]
        inputs.append([char_to_idx[char] for char in input_seq])
        targets.append(char_to_idx[target_char])

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long), idx_to_char, char_to_idx

