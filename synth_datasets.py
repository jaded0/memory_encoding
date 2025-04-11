import random
from datasets import Dataset, DatasetDict
from utils import initialize_charset

# def generate_long_range_sample():
#     charset, char_to_idx, idx_to_char, n_characters = initialize_charset("long_range_memory_dataset")  # default dataset

#     random_char = random.choice(charset[3:])
    
#     # Generate random number of zeroes before '?'
#     zeroes_before_question = "0" * random.randint(1, 5)
    
#     # Generate random number of zeroes between '?' and '!'
#     zeroes_between_question_exclamation = "0" * random.randint(1, 2)
    
#     # Generate random number of zeroes after '!'
#     # zeroes_after_exclamation = "0" * random.randint(1, 5)
    
#     sample = (
#         zeroes_before_question
#         + "?"
#         + random_char
#         + zeroes_between_question_exclamation
#         + "!"
#         + random_char
#         # + zeroes_after_exclamation
#     )
    
#     return sample

# def generate_resequence_sample(length=30, n=1):
#     charset, char_to_idx, idx_to_char, n_characters = initialize_charset("long_range_memory_dataset")  # default dataset
#     random_n = random.randint(1, n)  # Select a random number between 1 and n
#     if random_n > len(charset):
#         raise ValueError("random_n should be less than or equal to the length of the charset")

#     # Select random_n characters from the charset
#     selected_chars = random.sample(charset, random_n)

#     # Generate the alternating sequence
#     sequence = ''.join(selected_chars[i % random_n] for i in range(length))

#     return sequence


# def generate_dataset(num_samples, split, sample_type="long_range"):
#     if sample_type == "long_range":
#         samples = [generate_long_range_sample() for _ in range(num_samples)]
#     else:
#         n = int(sample_type.split('_')[0])  # Extract the number of characters from sample_type
#         samples = [generate_resequence_sample(n=n) for _ in range(num_samples)]
    
#     return Dataset.from_dict({"text": samples}, split=split)

# # Generate train, validation, and test datasets for long range memory dataset
# train_dataset = generate_dataset(num_samples=10000000, split="train")
# validation_dataset = generate_dataset(num_samples=50000, split="validation")
# test_dataset = generate_dataset(num_samples=200000, split="test")

# # Create a DatasetDict with the train, validation, and test datasets
# dataset_dict = DatasetDict({
#     "train": train_dataset,
#     "validation": validation_dataset,
#     "test": test_dataset
# })

# # Save the long range memory dataset to disk
# dataset_dict.save_to_disk("long_range_memory_dataset")

# # Generate resequence datasets
# for n in range(1, 5):
#     sample_type = f"{n}_resequence"
#     train_dataset = generate_dataset(num_samples=1000000, split="train", sample_type=sample_type)
#     validation_dataset = generate_dataset(num_samples=5000, split="validation", sample_type=sample_type)
#     test_dataset = generate_dataset(num_samples=20000, split="test", sample_type=sample_type)
    
#     # Create a DatasetDict with the train, validation, and test datasets
#     resequence_dataset_dict = DatasetDict({
#         "train": train_dataset,
#         "validation": validation_dataset,
#         "test": test_dataset
#     })
    
#     # Save the resequence dataset to disk
#     resequence_dataset_dict.save_to_disk(f"{n}_resequence")


# palindromes
def generate_palindrome_sample(fixed=True):
    """
    Generates a palindrome sample.

    Args:
        fixed (bool): If True, generates a fixed-length palindrome (half_length=4)
                      using the original method.
                      If False, generates a variable-length palindrome with:
                        - half_length between 1 and 3.
                        - middle character fixed to '.'.
                        - allowed characters for halves exclude '.' and ' '.
                        - a final EOS character ' '.

    Returns:
        str: The generated palindrome sample.
    """
    # Assume initialize_charset returns: charset_string, char_to_idx, idx_to_char, n_characters
    # We only need the charset_string here.
    charset, _, _, _ = initialize_charset("palindrome_dataset")
    start = "" # Seems unused in the original, kept for consistency
    half_length = 4

    if fixed:
        # --- Original Fixed-Length Logic ---
        # Build the left half using the full original charset
        left_half = ''.join(random.choice(charset) for _ in range(half_length))
        # Original middle character logic (random)
        middle = random.choice(charset)
        # Return the fixed palindrome without EOS
        return start + left_half + middle + left_half[::-1]

    else:
        # --- New Variable-Length Logic ---
        middle_char = '.'
        eos_char = ' '

        # Define characters allowed for the palindrome halves (exclude middle and EOS)
        # Convert charset string to a list for easy filtering
        allowed_chars_for_half = [ch for ch in charset if ch != middle_char and ch != eos_char]

        if not allowed_chars_for_half:
            raise ValueError(f"Charset '{charset}' becomes empty after removing middle ('{middle_char}') and EOS ('{eos_char}') characters.")

        # Choose a random length for the first half
        r_half_length = random.randint(1, half_length)

        # Build the left half using only the allowed characters
        left_half = ''.join(random.choice(allowed_chars_for_half) for _ in range(r_half_length))

        # Middle character is fixed
        middle = middle_char

        # Return the variable-length palindrome with the fixed middle and EOS char
        return start + left_half + middle + left_half[::-1] + (eos_char * ((half_length - r_half_length) * 2))

def generate_palindrome_dataset(num_samples, split, fixed=True):
    samples = [generate_palindrome_sample(fixed=fixed) for _ in range(num_samples)]
    print(f"First 5 samples for {split}: {samples[:5]}")
    return Dataset.from_dict({"text": samples}, split=split)

# Generate and save palindrome datasets
train_palindrome = generate_palindrome_dataset(num_samples=1000000, split="train")
validation_palindrome = generate_palindrome_dataset(num_samples=5000, split="validation")
test_palindrome = generate_palindrome_dataset(num_samples=20000, split="test")

palindrome_dataset_dict = DatasetDict({
    "train": train_palindrome,
    "validation": validation_palindrome,
    "test": test_palindrome
})
for split in palindrome_dataset_dict:
    print(f"palindrome_dataset {split} columns: {palindrome_dataset_dict[split].column_names}")
    print(f"Sample {split}: {palindrome_dataset_dict[split][0]}")
palindrome_dataset_dict.save_to_disk("palindrome_dataset")

# Generate and save palindrome datasets with varying lengths
train_palindrome_vary_length = generate_palindrome_dataset(num_samples=1000000, split="train", fixed=False)
validation_palindrome_vary_length = generate_palindrome_dataset(num_samples=5000, split="validation", fixed=False)
test_palindrome_vary_length = generate_palindrome_dataset(num_samples=20000, split="test", fixed=False)

palindrome_dataset_vary_length_dict = DatasetDict({
    "train": train_palindrome_vary_length,
    "validation": validation_palindrome_vary_length,
    "test": test_palindrome_vary_length
})

for split in palindrome_dataset_vary_length_dict:
    print(f"palindrome_dataset_vary_length {split} columns: {palindrome_dataset_vary_length_dict[split].column_names}")
    print(f"Sample {split}: {palindrome_dataset_vary_length_dict[split][0]}")
palindrome_dataset_vary_length_dict.save_to_disk("palindrome_dataset_vary_length")