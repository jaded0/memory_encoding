from datasets import load_dataset, load_from_disk
import torch.utils.data
from utils import filter_text, text_to_indices, text_to_indices_and_one_hot, collate_fn

dataset_keys = {
    "roneneldan/tinystories": "train",
    "jbrazzy/baby_names": "train",
    "brucewlee1/htest-palindrome": "test",
    "long_range_memory_dataset": "train",
    "1_resequence": "train",
    "2_resequence": "train",
    "3_resequence": "train",
    "4_resequence": "train",
    "palindrome_dataset": "train",
    "palindrome_dataset_vary_length": "train",  # Added new dataset
}

# Load dataset
def load_and_preprocess_data(dataset_name, batch_size=4, drop_last=True):
    if ("palindrome_dataset" in dataset_name) or ("long_range_memory_dataset" in dataset_name) or ("resequence" in dataset_name):
        dataset = load_from_disk(dataset_name)
        print(f"loaded dataset {dataset_name}")
    else:
        dataset = load_dataset(dataset_name)
    # print(f"cleaned up from cache: {dataset.cleanup_cache_files()}")
    # print(f"first dataset is {dataset}")
    # print(f"dataset is {dataset[dataset_keys[dataset_name]]}")
    # Inspect each split
    for split in dataset:
        print(f"Loaded {dataset_name} {split} columns: {dataset[split].column_names}")
        print(f"Loaded sample {split}: {dataset[split][0]}")
        
    if not "roneneldan/tinystories" in dataset_name:
        dataset = dataset[dataset_keys[dataset_name]]
    else:
        dataset = dataset[dataset_keys[dataset_name]].select(range(1000000))

    print(f"{dataset_name} columns:", dataset.column_names)  # Debugging line
    print("Sample data:", dataset[0])

    if dataset_name == "brucewlee1/htest-palindrome":
        dataset = dataset.filter(lambda example: example["correct_options_idx"][0] == 0)

    print("mapping the filter")
    dataset = dataset.map(filter_text, batched=True, fn_kwargs={"dataset_name": dataset_name})
    print("mapping text to indices")
    dataset = dataset.map(text_to_indices, batched=True, fn_kwargs={"dataset_name": dataset_name})
    print("mapping text to indices, onehotted")
    dataset = dataset.map(text_to_indices_and_one_hot, batched=True, fn_kwargs={"dataset_name": dataset_name})
    print('preprocessed') 

    # Shuffle the dataset
    dataset = dataset.shuffle()#seed=42)

    # # Inspect a few examples from the dataset
    # for i in range(3):
    #     print(dataset[i]['onehot_tensor'])

    # Create a DataLoader
    if not "roneneldan/tinystories" in dataset_name:
        dataset = list(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        drop_last=drop_last,
        num_workers=10,
        pin_memory=True
    )


    return dataloader
