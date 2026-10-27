"""Frozen round-1 lin checkpoint screening; no training and no slow updates.

Reuses round-1 o2_keys.run verbatim, adding PR and cosine-tail diagnostics there.
Each point is separately timeout-bounded by smoke.sh. GPU memory fraction <= .30.
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time

sys.path.insert(0, os.getcwd())
import torch
from kr_load_tasks import CHARSET, generate_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--mode", choices=("load", "gap"), required=True)
    ap.add_argument("--load", type=int, required=True)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selective-writes", action="store_true",
                    help="Suppress post-store/pre-query writes, retaining inputs and forgetting")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.30)
    spec = importlib.util.spec_from_file_location("o2_keys", "scratch/order2_2026-10-22/o2_keys.py")
    o2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(o2)
    original = o2.read_checkpoint

    def read(path):
        checkpoint = original(path)
        checkpoint["config"]["fused_update"] = False  # GPU0 is a 1080 Ti: unfused parity path
        return checkpoint

    def batches(dataset, batch_size, nb, device, split):
        rng = random.Random(20261027)  # paired streams across load points
        for _ in range(nb):
            texts = [generate_sample(rng, args.mode, args.load) for _ in range(batch_size)]
            indices = torch.tensor([[CHARSET.index(c) for c in t] for t in texts], device=device)
            yield texts, torch.nn.functional.one_hot(indices, len(CHARSET)).float()

    o2.read_checkpoint = read
    o2.load_heldout_batches = batches
    if args.selective_writes:
        original_mask = o2.update_mask

        def selective(protocol, texts, dataset, onehot):
            mask = original_mask(protocol, texts, dataset, onehot)
            mask[:, 2:] = False
            return mask

        o2.update_mask = selective
    start = time.monotonic()
    result = o2.run(args.checkpoint, args.batches, "cuda", [4.0])
    result.update(mode=args.mode, load=args.load, selective_writes=args.selective_writes,
                  elapsed_seconds=time.monotonic() - start,
                  peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                  checkpoint_sha256=__import__("hashlib").sha256(open(args.checkpoint, "rb").read()).hexdigest(),
                  code_commit=__import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  caveat="Frozen round-1 scaffold, OOD screening; not a trained-load capacity result.")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
