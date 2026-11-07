# HY1 opt-in pilot API
`from hy1_model import HY1Config, HY1Model, evaluate_document, encode_text, load_model`
- `HY1Config(vocab_size, arm="A0", seed=110601, ...)`; defaults 4 layers/d256/4 heads/window64/chunk64. A2 window128, A3 unlimited.
- `model = HY1Model(config).to(device)`; `model.new_state()` resets fast + KV; state is explicit, not held on model.
- `losses, state = evaluate_document(model, text_or_1d_token_tensor, chars=chars, state=None, wipe_every_chunk=False)` returns CPU per-position loss tensor of length len(encoded_doc)-1 and final state. Loss index i predicts encoded_doc[i+1]; input encoded_doc[i]. Passing state carries **fast only** across docs: KV and positions reset at every evaluate_document call. Default state=None resets fast. Frozen slow weights; fast DFA writes still occur. Wipe zeroes fast at start of EVERY chunk, not just document start.
- `encode_text(text, chars)` uses preprocess.filter_text({text: [text]}, TinyStories) then indices, exactly existing preprocessing. chars is ordered str.
- `load_model(checkpoint_path, device="cpu") -> (model, checkpoint_dict)`; checkpoint has config, model, chars, tokens, steps, optimizer.
- Low-level `model.forward_chunk(tokens_1d, state) -> (logits [T,V], next_state, pending)` does not write fast; `model.commit(next_state,pending,targets_1d)` commits detached writes only after entire chunk. `state.fast` vector or None, `state.cache` list, `state.position`.
- CLI `hy1_train.py --data .../data.pt --arm A1 --tokens N --out DIR [--timing-docs 3] [--device cuda]`; midpoint.pt/final.pt. Train batches one full document via chunks, optimizer per chunk. Same seed/data order/budget across arms. Timing mode does not launch production.
- Fast exact algorithm and benchmark report will be added here when tested.

## Implemented algorithm / caveats (final)
- `encode_text` actually calls `preprocess.filter_text({"text": [text]}, "roneneldan/tinystories")`, checks provided chars against `utils.initialize_charset` order, then encodes. This is the exact B2 filtering, including min-length padding, not the initially abbreviated signature above.
- Fast entries: exactly round(0.1*256*1024)=26,214 random layer0 MLP output-matrix entries. Slow value+gradient zero there, including enforcement after every Adam step. No new trainable parameters. Initial slow values outside mask are bit-identical across all arms. Private RNG selects mask and B_fast after all slow initialization.
- Fixed separate feedback `B_fast[vocab,d]` is Rademacher/sqrt(vocab). For completed chunk length T, E=softmax(logits)-onehot(labels), D=E@B_fast, G=(D.T@GELU(fc1(norm2(layer0_attention_residual))))/T, then select mask entries. U=-G/max(RMS(G),1e-12). F=.95F+1*U; clip F RMS to10. Thus alpha=1 is deliberately HIGH against slow-weight initial RMS.02; scale invariance comes from normalization of every DFA write, NOT a small learning rate. All writes detached.
- Fast read r=F@hidden is scaled per token by min(1,.1*RMS(slowMLP)/max(RMS(r),1e-12)), scale detached; fast weights fixed throughout chunk. Separate B is not tied to output weight or any training feedback matrix. Slow gradients still flow through reads wrt hidden, never through writes.
- W64 includes current token and 63 preceding positions per layer. Four local layers give max4*(W-1) token receptive field; >W means beyond direct per-layer attention, NOT beyond all attention. Cached layer representations also transitively carry context.
- KV is real sliding history across chunks; A3 retains entire document KV. KV is detached at chunk boundaries: chunk TBPTT, not full-document BPTT. Slow Adam updates per chunk mean past caches use the then-current slow weights (standard online stale-cache tradeoff). A3 is whole-story attention, NOT whole-story gradient.
- Default evaluate_document resets fast+KV; explicit state carries ONLY fast at document boundary; no cross-document attention. Wipe option erases fast before every chunk but still computes writes afterward, so returned final fast may be nonzero. Scores within that next chunk always used wiped zero state.
- Optimizer is plain torch.optim.Adam(lr3e-4, foreach=False, fused=False), FP32. No SDPA, compile, Triton, fused attention or fused optimizer. Nonfinite logits/loss/gradnorm/fast state fail fast.
- Data order is fixed shuffled full-document list from seed110601. No ordinary document cropping. Final exact token budget may stop part-way through the last story. Doc-end tails and final-budget tails can be shorter than C64. Midpoint checkpoint is first natural completed chunk at/after half-budget, and stores actual tokens (no artificial midpoint chunk splitting).
- Checkpoint includes document_state={fast: compact vector, cache: detached KV list, position}, plus config/model/optimizer/chars/tokens/steps/provenance. Scoring normally resets this state. There is no resume CLI; checkpoints are evaluation-ready and contain optimizer/document state for potential future resume implementation.
- Fast payload=104,856 bytes FP32; temporary dense read matrix=1,048,576 bytes. Fixed index buffer=209,712 bytes int64 and slow mask=1,048,576 bytes FP32 are model metadata, not carried adaptive state. B_fast metadata=4*vocab*256 bytes. Additional persistent KV for A2 versus A1=524,288 bytes FP32; A1 fast is 20.0% of that. No fraction-match arm added.
- Tests: CPU 12 passed/1 CUDA skip; CUDA 13 passed on EACH physical GPU including the final off-by-one test. Tests establish causality, chunk-delayed detached writes, exact W boundary, full-vs-chunked attention equivalence A0/A2/A3, read/state caps, exclusive zero gradients+weights, reset/carry/wipe behavior, no added parameters/shared slow initialization/private RNG, finite failures, checkpoint reload and unfused CUDA path.

## Timing (2026-11-06, 3 complete shuffled training docs, 2355 targets/38 steps)
Explicit `CUDA_DEVICE_ORDER=PCI_BUS_ID` is REQUIRED if GPU0 must mean nvidia-smi GPU0 (1080Ti). Without it this host reverses the CUDA ordinal mapping; first trial logs named gpu0 actually report RTX2080Ti. Use `pci_gpu*.log` below.

| physical GPU | A0 sec/token | A1 sec/token | A2 sec/token | A3 sec/token |
|---|---:|---:|---:|---:|
| 0 GTX1080Ti | .000247711 | .000256955 | .000242275 | .000243778 |
| 1 RTX2080Ti | .000252302 | .000269613 | .000256545 | .000256699 |

CUDA initialization and one discarded warmup chunk excluded; full forward/backward/Adam/DFA included; preprocessing preencoded for timing. Timing docs are complete, not cropped (2355 targets in total). CUDA synchronized at timing boundaries. Peak allocated A0/A1/A2/A3=73.53/101.09/74.58/90.43MB, reserved <=121.64MB. This is a very short microbenchmark, not a sustained throughput guarantee; allow 2x margin for CPU contention and corpus variation.

Commands, each via `ssh manticore bash -lc`:
```
cd /home/jaden/hy1_2026-11-06/code
for gpu in 0 1; do
  for arm in A0 A1 A2 A3; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu /home/jaden/miniforge3/envs/hebby/bin/python hy1_train.py --data /home/jaden/hy1_2026-11-06/data.pt --arm $arm --tokens 1000000 --timing-docs 3 --device cuda
  done
done
CUDA_VISIBLE_DEVICES= /home/jaden/miniforge3/envs/hebby/bin/python -m pytest tests/test_hy1.py -q
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /home/jaden/miniforge3/envs/hebby/bin/python -m pytest tests/test_hy1.py -q
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /home/jaden/miniforge3/envs/hebby/bin/python -m pytest tests/test_hy1.py -q
```
Raw logs: `/home/jaden/hy1_2026-11-06/timing/pci_gpu{0,1}_{A0,A1,A2,A3}.log`.
A 129-target checkpoint smoke (not production) wrote `timing/checkpoint_smoke/{midpoint,final}.pt`, verified actual midpoint128/final129 tokens and compact fast shape(26214,), and evaluated text after reload.
Recommended conservative sizing: 50m targets/arm, GPU0 A0+A2 ~6.8h, GPU1 A1+A3 ~7.3h raw; ~15h with2x margin, below20h. Main owns actual prereg/sizing/launch. No production launched or merge/commit performed by this worker.
