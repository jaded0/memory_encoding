"""Dataset loading and preprocessing.

Synthetic datasets (names containing palindrome_dataset, long_range_memory_dataset or
resequence) load from synth_datasets/ and are preprocessed on the fly; they are small.

Hugging Face datasets (e.g. roneneldan/tinystories) are preprocessed once, by
`python preprocess.py <name>` (on the cluster: setup_cluster/prepare_datasets.sbatch),
and saved to the processed-data directory ($EPHEMERAL_DATA_DIR, default
./processed_datasets/). Training only loads that saved copy; if it is missing it
fails rather than spend the better part of an hour preprocessing inside a job.

Rows store the filtered text and the character indices (uint8). The one-hot tensors
the model consumes are built per batch by OneHotCollate, since storing them costs
~n_characters * 4 bytes per character (~274 GB for TinyStories).
"""
import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
import time

import torch.utils.data
from datasets import Features, Sequence, Value, load_dataset, load_from_disk
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence

from reproducibility import ResumableRandomSampler, make_torch_generator, seed_data_worker
from utils import collate_fn, filter_text, get_charset, initialize_charset, text_to_indices

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
    "palindrome_dataset_vary_length": "train",
    "1_palindrome_dataset_vary_length": "train",
    "2_palindrome_dataset_vary_length": "train",
    "3_palindrome_dataset_vary_length": "train",
    "4_palindrome_dataset_vary_length": "train",
    "1_small_palindrome_dataset_vary_length": "train",
    "2_small_palindrome_dataset_vary_length": "train",
    "3_small_palindrome_dataset_vary_length": "train",
    "4_small_palindrome_dataset_vary_length": "train",
}

# Bump whenever this file changes what a processed dataset contains. The saved name also
# carries a hash of the charset and of the utils.py preprocessing functions, so edits there
# invalidate old saves automatically.
PREPROCESS_VERSION = 1

# Rows kept from the start of the split, before preprocessing.
MAX_ROWS = {"roneneldan/tinystories": 1_000_000}

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SETUP_HINT = (
    "Prepare it once before training:\n"
    "  cluster: setup_cluster/download_datasets.sh on the login node, then\n"
    "           sbatch setup_cluster/prepare_datasets.sbatch   (see setup_cluster/README.md)\n"
    "  local:   python preprocess.py {name}"
)


class ProcessedDatasetMissing(FileNotFoundError):
    pass


def is_synthetic(dataset_name):
    return any(tag in dataset_name for tag in ("palindrome_dataset", "long_range_memory_dataset", "resequence"))


def processed_data_dir():
    return os.environ.get("EPHEMERAL_DATA_DIR") or os.path.join(REPO_DIR, "processed_datasets")


def _short_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def preprocessing_code_hash():
    """Hash of the utils.py functions whose output is saved."""
    return _short_hash("".join(inspect.getsource(fn) for fn in (filter_text, text_to_indices)))


def processed_dataset_name(dataset_name, limit=None):
    """Directory name that changes whenever anything affecting the saved rows changes."""
    split = dataset_keys[dataset_name]
    rows = MAX_ROWS.get(dataset_name)
    if limit is not None:
        rows = limit if rows is None else min(rows, limit)
    return "__".join([
        dataset_name.replace("/", "--"),
        split,
        f"rows-{rows if rows is not None else 'all'}",
        f"charset-{_short_hash(get_charset(dataset_name))}",
        f"code-{preprocessing_code_hash()}",
        f"v{PREPROCESS_VERSION}",
    ])


def processed_dataset_path(dataset_name, limit=None, data_dir=None):
    return os.path.join(data_dir or processed_data_dir(), processed_dataset_name(dataset_name, limit))


def default_num_proc(cap=16):
    """CPUs this process may use (SLURM allocation or affinity mask), capped."""
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_NTASKS"):
        if os.environ.get(var, "").isdigit():
            available = min(available, int(os.environ[var]))
            break
    return max(1, min(available, cap))


def preprocess_rows(dataset, dataset_name, num_proc=None, keep_in_memory=False):
    """Raw split -> rows of {'text': filtered text, 'tensor': uint8 character indices}."""
    n_characters = initialize_charset(dataset_name)[3]
    assert n_characters <= 256, "character indices are stored as uint8"
    if dataset_name == "brucewlee1/htest-palindrome":
        dataset = dataset.filter(lambda example: example["correct_options_idx"][0] == 0)
    map_kwargs = {
        "batched": True,
        "fn_kwargs": {"dataset_name": dataset_name},
        "num_proc": num_proc if num_proc and num_proc > 1 else None,
        "keep_in_memory": keep_in_memory,
    }
    print("mapping the filter")
    dataset = dataset.map(filter_text, remove_columns=dataset.column_names, **map_kwargs)
    print("mapping text to indices")
    features = Features({"text": Value("string"), "tensor": Sequence(Value("uint8"))})
    dataset = dataset.map(text_to_indices, features=features, **map_kwargs)
    return dataset


def prepare_dataset(dataset_name, limit=None, num_proc=None, data_dir=None, overwrite=False):
    """Preprocess a Hugging Face dataset from the local HF cache and save it. Returns its path."""
    if is_synthetic(dataset_name):
        raise ValueError(f"{dataset_name} is synthetic; it loads from synth_datasets/ and needs no preparation")
    path = processed_dataset_path(dataset_name, limit, data_dir)
    if os.path.isdir(path) and not overwrite:
        print(f"{dataset_name}: already prepared at {path}")
        return path
    num_proc = num_proc or default_num_proc()
    start = time.time()
    try:
        dataset = load_dataset(dataset_name, split=dataset_keys[dataset_name])
    except Exception as error:
        raise RuntimeError(
            f"Could not load raw {dataset_name} (offline: {os.environ.get('HF_DATASETS_OFFLINE')}). "
            "On the cluster, download it first on the login node: setup_cluster/download_datasets.sh"
        ) from error
    rows = MAX_ROWS.get(dataset_name)
    if limit is not None:
        rows = limit if rows is None else min(rows, limit)
    if rows is not None:
        dataset = dataset.select(range(min(rows, len(dataset))))
    print(f"{dataset_name}: preprocessing {len(dataset)} rows with num_proc={num_proc}")
    # In memory: the saved copy below is the only artifact, nothing lands in the HF cache.
    dataset = preprocess_rows(dataset, dataset_name, num_proc=num_proc, keep_in_memory=True)

    # Write next to the destination, then rename, so a killed job never leaves a partial save.
    tmp_path = f"{path}.tmp-{os.getpid()}"
    shutil.rmtree(tmp_path, ignore_errors=True)
    dataset.save_to_disk(tmp_path)
    info = {
        "dataset_name": dataset_name,
        "split": dataset_keys[dataset_name],
        "rows": len(dataset),
        "charset": get_charset(dataset_name),
        "code_hash": preprocessing_code_hash(),
        "preprocess_version": PREPROCESS_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(tmp_path, "preprocess_info.json"), "w") as handle:
        json.dump(info, handle, indent=2)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.rename(tmp_path, path)
    print(f"{dataset_name}: saved {len(dataset)} rows to {path} in {time.time() - start:.0f}s")
    return path


def load_processed_dataset(dataset_name, data_dir=None):
    path = processed_dataset_path(dataset_name, data_dir=data_dir)
    if not os.path.isdir(path):
        parent = os.path.dirname(path)
        prefix = dataset_name.replace("/", "--") + "__"
        others = sorted(n for n in os.listdir(parent) if n.startswith(prefix)) if os.path.isdir(parent) else []
        stale = ("\nOther (stale) versions there: " + ", ".join(others)) if others else ""
        raise ProcessedDatasetMissing(
            f"Processed dataset for {dataset_name} not found at {path}{stale}\n"
            + SETUP_HINT.format(name=dataset_name)
        )
    print(f"loaded processed dataset {path}")
    return load_from_disk(path)


class OneHotCollate:
    """Batches rows of {'text', 'tensor'} into (texts, padded indices, padded one-hot).

    Produces exactly what utils.collate_fn produced from stored one-hot rows: int64 indices
    padded with 0 and float32 one-hot padded with all-zero rows.
    """

    def __init__(self, n_characters):
        self.n_characters = n_characters

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        indices = [torch.tensor(item['tensor'], dtype=torch.long) for item in batch]
        tensors = pad_sequence(indices, batch_first=True)
        onehot_tensors = pad_sequence(
            [one_hot(line, num_classes=self.n_characters).float() for line in indices], batch_first=True
        )
        return texts, tensors, onehot_tensors


# Load dataset
def load_and_preprocess_data(dataset_name, batch_size=4, drop_last=True, seed=None):
    if is_synthetic(dataset_name):
        dataset = load_from_disk(f"synth_datasets/{dataset_name}")[dataset_keys[dataset_name]]
        print(f"loaded dataset {dataset_name}")
        dataset = preprocess_rows(dataset, dataset_name)
    else:
        dataset = load_processed_dataset(dataset_name)
    print(f"{dataset_name} columns:", dataset.column_names)
    print("Sample data:", dataset[0]['text'][:200])

    # Shuffle the dataset (in memory: nothing is written next to the saved data)
    if seed is None:
        dataset = dataset.shuffle(keep_in_memory=True)
    else:
        dataset = dataset.shuffle(seed=seed, keep_in_memory=True)

    # Create a DataLoader
    if is_synthetic(dataset_name):
        dataset = list(dataset)
    collate = OneHotCollate(initialize_charset(dataset_name)[3])
    return make_dataloader(dataset, batch_size, drop_last=drop_last, seed=seed, collate_fn=collate)


def make_dataloader(dataset, batch_size, drop_last=True, seed=None, num_workers=10, collate_fn=collate_fn):
    """Shuffling DataLoader; when seeded, its position can be checkpointed (see DataStream)."""
    dataloader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if seed is None:
        dataloader_kwargs["shuffle"] = True
    else:
        # Same draws as shuffle=True with this generator, but resumable mid-epoch.
        generator = make_torch_generator(seed)
        dataloader_kwargs["generator"] = generator
        dataloader_kwargs["sampler"] = ResumableRandomSampler(dataset, generator=generator)
        dataloader_kwargs["worker_init_fn"] = seed_data_worker

    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess Hugging Face datasets into the processed-data directory.")
    parser.add_argument("datasets", nargs="+", help="e.g. roneneldan/tinystories")
    parser.add_argument("--num_proc", type=int, default=None, help="map workers (default: allocated CPUs, max 16)")
    parser.add_argument("--data_dir", default=None, help="default: $EPHEMERAL_DATA_DIR or ./processed_datasets")
    parser.add_argument("--limit", type=int, default=None,
                        help="testing only: keep the first N rows (saved under a different name)")
    parser.add_argument("--overwrite", action="store_true", help="re-prepare even if already saved")
    args = parser.parse_args(argv)
    for name in args.datasets:
        if name not in dataset_keys:
            parser.error(f"unknown dataset {name}; known: {', '.join(dataset_keys)}")
        prepare_dataset(name, limit=args.limit, num_proc=args.num_proc, data_dir=args.data_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
