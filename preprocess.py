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
}

# Load dataset
def load_and_preprocess_data(dataset_name, batch_size=4):
    if dataset_name == "long_range_memory_dataset" or "resequence" in dataset_name:
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name)
    # print(f"cleaned up from cache: {dataset.cleanup_cache_files()}")
    # print(f"first dataset is {dataset}")
    # print(f"dataset is {dataset[dataset_keys[dataset_name]]}")
    dataset = dataset[dataset_keys[dataset_name]].select(range(200))
    if dataset_name == "brucewlee1/htest-palindrome":
        dataset = dataset.filter(lambda example: example["correct_options_idx"][0] == 0)

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
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    return dataloader
