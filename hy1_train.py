"""HY1 opt-in training/timing, FP32 eager Adam and chunk-detached KV history.

The data file must be the B2-split TinyStories dict {train,validation,chars}.
All arms use the same seed/order/exact teacher-forced target-token budget.
No document cropping: traverse each entire story before the next; only the last
story is interrupted if needed to hit an exact requested budget. Loss excludes
the first char (no BOS) and no synthetic EOS/document-bridging labels are added.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from hy1_model import HY1Config, HY1Model, encode_text


def save_checkpoint(path, model, optimizer, chars, tokens, steps, state=None, provenance=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"config": asdict(model.config), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "chars": chars,
                "tokens": tokens, "steps": steps,
                "document_state": None if state is None else {
                    "fast": None if state.fast is None else state.fast.detach().cpu(),
                    "cache": [(k.cpu(), v.cpu()) for k, v in state.cache],
                    "position": state.position},
                "provenance": provenance or {}}, temporary)
    temporary.replace(path)


def train_chunk(model, optimizer, x, y, state):
    optimizer.zero_grad(set_to_none=True)
    logits, state, pending = model.forward_chunk(x, state)
    loss = F.cross_entropy(logits, y)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    model.enforce_exclusive()
    # Pending logits/features belong to the SAME pre-optimizer forward pass.
    state = model.commit(state, pending, y)
    return loss.detach().item(), state, float(norm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--arm", choices=["A0", "A1", "A2", "A3"], required=True)
    parser.add_argument("--tokens", type=int, default=1000000)
    parser.add_argument("--steps", type=int, default=None, help="optional early step ceiling; budget must be same across arms")
    parser.add_argument("--out", default=None)
    parser.add_argument("--timing-docs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=110601)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.tokens < 2 or args.timing_docs < 0 or args.lr > 1e-3:
        parser.error("tokens>=2, timing-docs>=0, lr<=1e-3 required")
    if not args.timing_docs and not args.out:
        parser.error("--out required outside timing mode")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    chars = data["chars"]
    config = HY1Config(vocab_size=len(chars), arm=args.arm, seed=args.seed)
    model = HY1Model(config).to(args.device)
    # Explicit unfused/non-foreach path works on sm61 as well as CPU/sm75.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, foreach=False, fused=False)
    indices = list(range(len(data["train"])))
    if not indices:
        raise ValueError("empty training data")
    random.Random(args.seed).shuffle(indices)
    if args.timing_docs:
        indices = indices[:args.timing_docs]
    # Pre-encode timing documents to exclude preprocessing/import overhead.
    timing_tokens = {i: encode_text(data["train"][i], chars) for i in indices} if args.timing_docs else {}
    # One disposable synthetic warmup exercises forward/backward and CUDA init;
    # restore fresh model+optimizer so timed data/order match production exactly.
    if args.timing_docs:
        warm = torch.arange(65, device=args.device) % len(chars)
        train_chunk(model, optimizer, warm[:-1], warm[1:], model.new_state())
        model = HY1Model(config).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, foreach=False, fused=False)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokens = steps = docs = 0
    loss_sum = 0.
    midpoint_saved = False
    threshold = math.ceil(args.tokens/2)
    midpoint_seconds = None
    while tokens < args.tokens:
        for index in indices:
            document = timing_tokens[index] if args.timing_docs else encode_text(data["train"][index], chars)
            document = document.to(args.device)
            state = model.new_state()
            docs += 1
            start = 0
            while start < len(document)-1 and tokens < args.tokens:
                length = min(config.chunk_size, len(document)-1-start, args.tokens-tokens)
                # Midpoint saved at first completed chunk >= half budget;
                # never change the C64 write schedule for checkpointing.
                x, y = document[start:start+length], document[start+1:start+1+length]
                loss, state, norm = train_chunk(model, optimizer, x, y, state)
                tokens += length
                steps += 1
                start += length
                loss_sum += loss*length
                if not midpoint_saved and tokens >= threshold:
                    midpoint_saved = True
                    midpoint_seconds = time.perf_counter()-started
                    if not args.timing_docs:
                        save_checkpoint(Path(args.out)/"midpoint.pt", model, optimizer, chars, tokens, steps, state,
                                        {"data": args.data, "lr": args.lr, "document_index": index, "document_offset": start,
                                         "chunk_tbptt": True, "dtype": "float32", "unfused_adam": True})
                if steps % 100 == 0:
                    print(json.dumps({"steps": steps, "tokens": tokens, "mean_loss": loss_sum/tokens,
                                      "last_loss": loss, "grad_norm": norm}), flush=True)
                if args.steps is not None and steps >= args.steps:
                    break
            if tokens >= args.tokens or (args.steps is not None and steps >= args.steps):
                break
        if args.timing_docs or (args.steps is not None and steps >= args.steps):
            break
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    report = {"arm": args.arm, "seed": args.seed, "tokens": tokens, "steps": steps, "docs": docs,
              "seconds": seconds, "sec_per_token": seconds/tokens, "tokens_per_second": tokens/seconds,
              "mean_loss": loss_sum/tokens, "midpoint_seconds": midpoint_seconds,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else 0,
              "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
              "unfused_adam": True, "dtype": "float32", **model.state_bytes()}
    if not args.timing_docs:
        save_checkpoint(Path(args.out)/"final.pt", model, optimizer, chars, tokens, steps, state,
                        {"data": args.data, "lr": args.lr, "document_index": index, "document_offset": start,
                         "chunk_tbptt": True, "dtype": "float32", "unfused_adam": True})
        Path(args.out, "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main()
