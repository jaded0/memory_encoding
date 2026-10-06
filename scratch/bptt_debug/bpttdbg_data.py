"""The repo's own loader, on the first N rows (the full split takes minutes to map)."""
import os
from datasets import load_from_disk
from preprocess import preprocess_rows, make_dataloader, OneHotCollate, dataset_keys, is_synthetic
from utils import initialize_charset


def subset_loader(dataset_name, batch_size, seed, n_rows=200_000):
    root = os.environ.get('SYNTH_ROOT', 'synth_datasets')
    ds = load_from_disk(f"{root}/{dataset_name}")[dataset_keys.get(dataset_name, 'train')]
    ds = ds.select(range(min(n_rows, len(ds))))
    ds = preprocess_rows(ds, dataset_name, keep_in_memory=True)
    ds = ds.shuffle(seed=seed, keep_in_memory=True)
    ds = list(ds)
    return make_dataloader(ds, batch_size, drop_last=True, seed=seed, num_workers=0,
                           collate_fn=OneHotCollate(initialize_charset(dataset_name)[3]))
