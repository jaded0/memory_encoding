"""Real TinyStories sample, real train.main/flags/checkpoints; no full-data prep.

This harness only changes the loader to a tiny pre-staged JSON train sample and
caps CUDA allocations below 5 GB. It is NOT a production data loader or result.
"""
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.getcwd())
import torch
import train
from preprocess import OneHotCollate, make_dataloader
from utils import initialize_charset

torch.cuda.set_per_process_memory_fraction(.25, 0)  # ~4094 MiB allocator on omen
torch.set_num_threads(3)


def sample_loader(dataset, batch_size, drop_last=True, seed=None, max_seq_len=0):
    charset, c2i, _, n = initialize_charset(dataset)
    texts = json.load(open(os.environ['B2FU_SMOKE_TRAIN']))
    items = [{'text': t, 'tensor': [c2i[c] for c in t if c in c2i]} for t in texts]
    return make_dataloader(items, batch_size, drop_last=drop_last, seed=seed,
                           num_workers=0, collate_fn=OneHotCollate(n, max_seq_len))


if __name__ == '__main__':
    with patch.object(train, 'load_and_preprocess_data', side_effect=sample_loader):
        train.main()
    print('CUDA_PEAK_ALLOCATED_BYTES', torch.cuda.max_memory_allocated())
    print('CUDA_PEAK_RESERVED_BYTES', torch.cuda.max_memory_reserved())
