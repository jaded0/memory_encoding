from datasets import load_dataset
import torch.utils.data
from utils import filter_text, text_to_indices, text_to_indices_and_one_hot, collate_fn

# Load dataset
def load_and_preprocess_data(dataset_name):
    dataset = load_dataset(dataset_name)
    # dataset.cleanup_cache_files()
    dataset = dataset['train'].select(range(10))
    
    print("mapping the filter")
    dataset = dataset.map(filter_text, batched=True, fn_kwargs={"dataset_name": dataset_name})
    print("mapping text to indices")
    dataset = dataset.map(text_to_indices, batched=True, fn_kwargs={"dataset_name": dataset_name})
    print("mapping text to indices, onehotted")
    dataset = dataset.map(text_to_indices_and_one_hot, batched=True, fn_kwargs={"dataset_name": dataset_name})
    print('preprocessed') 

    # # Inspect a few examples from the dataset
    # for i in range(3):
    #     print(dataset[i]['onehot_tensor'])

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    return dataloader
