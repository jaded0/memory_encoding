import random
from datasets import Dataset, DatasetDict
from utils import charset

def generate_sample():
    random_char = random.choice(charset)
    
    # Generate random number of zeroes before '?'
    zeroes_before_question = "0" * random.randint(1, 5)
    
    # Generate random number of zeroes between '?' and '!'
    zeroes_between_question_exclamation = "0" * random.randint(1, 5)
    
    # Generate random number of zeroes after '!'
    zeroes_after_exclamation = "0" * random.randint(1, 5)
    
    sample = (
        zeroes_before_question
        + "?"
        + random_char
        + zeroes_between_question_exclamation
        + "!"
        + random_char
        + zeroes_after_exclamation
    )
    
    return sample

def generate_dataset(num_samples, split):
    samples = [generate_sample() for _ in range(num_samples)]
    return Dataset.from_dict({"text": samples}, split=split)

# Generate train, validation, and test datasets
train_dataset = generate_dataset(num_samples=1000, split="train")
validation_dataset = generate_dataset(num_samples=50, split="validation")
test_dataset = generate_dataset(num_samples=200, split="test")

# Create a DatasetDict with the train, validation, and test datasets
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

# Save the dataset to disk
dataset_dict.save_to_disk("long_range_memory_dataset")