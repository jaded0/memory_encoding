from datasets import load_dataset
import torch.utils.data
from utils import filter_text, text_to_indices, collate_fn

# Load dataset
def load_and_preprocess_data():
    dataset = load_dataset("roneneldan/TinyStories")
    dataset = dataset['train'].select(range(1000))

    print("mapping the filter")
    dataset = dataset.map(filter_text, batched=True)
    print("mapping text to indices")
    dataset = dataset.map(text_to_indices, batched=True)
    print('preprocessed') 

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    return dataloader
