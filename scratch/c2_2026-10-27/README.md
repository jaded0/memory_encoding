# C2 ready: backprop step parity and the host-agnostic B_fast arm (prep, 2026-10-26)

Prep agent on omen. Nothing has been launched beyond the smoke tests below. Spec: proposals page `### C2 ·`. Pre-registration draft: `~/Documents/brain/ephemeral-lowrank/queue_2026-10-27/prereg_drafts/C2.md`.

## What's ready

- **Branch `c2-backprop-parity`**, based on origin/main 364a07f and pushed. Code commit `fdbea66`. Launchers and scoring were added in `38843c0`, `e7e195a` and `e01f658`, and this README in the commit after those. Launch from the branch head.
- **`--backprop_step_parity`** (ephemeral + `--updater backprop`; off by default). With it on, the step is `−lr·α·g_b` on the fast entries and `−lr·g_b` on the slow ones, the same form as DFA's. The derivation is in README "Backprop step parity".
  - Where the factors come from: `step_loss.mean()` gives each copy `g_b/B`, then `scale_ephemeral_grads` applies α and `ephemeral_update` applies α again. So today's step is lr·α²/B on the fast entries and lr/B on the slow ones.
  - The fix: backward on the batch sum, skip `scale_ephemeral_grads`, and divide the bias gradients by B so the bias step stays the batch mean.
  - It's refused with DFA, BPTT and the rnn baseline. A resume with a different setting is refused.
- **Host-agnostic arm: `--decouple_fast_feedback L` under backprop.** It needs `--backprop_step_parity`. Backprop trains every slow entry and bias. Layer L's fast entries take DC2's B_fast write, through the same `override_fast_gradients` code DFA uses. No gradient flows through the write. `--grad_norm_clip` and `--fast_write_from_readout` stay refused.
- **Tests.** There are 13 new tests in `tests/test_backprop_step_parity.py`:
  - both step forms (α²/B and parity);
  - on a tiny model, the readout step and a B_fast write equal the DFA path's exactly;
  - the slow steps are unchanged by decoupling;
  - the fast state is a constant to the slow gradient;
  - the clip norm, and the refused combinations.

  One existing assertion moved from "decouple + backprop refused" to BPTT (`test_fast_layer_flags.py`).
  - The suite has 419 passed. The one failure, `test_characterization`, is a Torch label mismatch on omen (`2.5.1+cu121` vs `2.5.1`) and fails on main too.
  - Default traces from this branch are **byte-identical to main's** on omen for every golden case.
- **Launchers:** `c2_arms.sh` (the arm table), `c2.sbatch` (ORC array 0–14), `run_local.sh` (local GPU), and `c2_score.py`.

## Arms (400k, seeds 3141 2718 4241, the stabilizer pilot's output_tanh recipe on 3-char reversal)

| Tasks | Arm | Flags beyond the recipe |
|---|---|---|
| 0–2 | S1-off | `--updater backprop` |
| 3–5 | S1-on | `+ --backprop_step_parity true` |
| 6–8 | S2-a1e4 | `+ --backprop_step_parity true --decouple_fast_feedback 0,1,2` |
| 9–11 | S2-a1e3 | same, `--plasticity 1e3` |
| 12–14 | S2-a1e2 | same, `--plasticity 1e2` |

## Launch

On ORC, once:
```
mkdir -p ~/c2_2026-10-27/{logs,runs}; cd ~/c2_2026-10-27
git clone <memory_encoding remote> code && git -C code checkout <commit>
ln -s <path to synth_datasets> code/synth_datasets
```
Then:
```
sbatch --export=ALL,CODE_COMMIT=<commit> code/scratch/c2_2026-10-27/c2.sbatch            # all 15
sbatch --array=0-5 --export=ALL,CODE_COMMIT=<commit> code/scratch/c2_2026-10-27/c2.sbatch  # Step 1 only
```
Score:
```
python code/scratch/c2_2026-10-27/c2_score.py ~/c2_2026-10-27/runs --json c2.jsonl
```

To run locally (deckard, manticore or omen), one run after another:
```
scratch/c2_2026-10-27/run_local.sh <code> <out> 400000 <tasks...>
```

## GPU-hours and placement

Measured on omen's 4090: S1 arms run at about 20 it/s, so a 400k run takes about 5.6 h. S2 arms run at about 13.5 it/s, about 8.2 h a run, because of the B_fast override. In total that's **about 34 GPU-h for Step 1 and about 74 GPU-h for Step 2, roughly 110 GPU-h**. That is well above the spec's 15–20, which assumed DFA speed. Collapsed runs stop early.

Suggested placement: an ORC array (12 h limit, requeues on timeout). If fairshare is tight, run tasks 0–5 first, since Step 1 gates any backprop claim. deckard's 2 A6000s are the fallback.

## Smoke evidence (omen, code e7e195a, seed 3141, 2,000 iterations each, `/tmp/c2_smoke`)

All five arms exited 0. A resume from 2,000 to 2,500 worked, and a resume with parity switched off was refused.

| Arm | Loss at 2k | recall_acc at 2k | it/s | Readout share in B_fast,2 | Trunk-2 fast entries at clamp |
|---|---:|---:|---:|---:|---:|
| S1-off | 2.015 | 0.181 | 19.8 | 0.008 | 3.2% (mean \|f\| 0.56) |
| S1-on | 1.976 | 0.000 | 20.6 | 0.010 | 0% (mean \|f\| 0.002) |
| S2-a1e4 | 1.815 | 0.065 | 14.0 | 0.152 | 0% |
| S2-a1e3 | 1.987 | 0.000 | 13.5 | 0.010 | 0% |
| S2-a1e2 | 1.987 | 0.000 | 13.1 | 0.010 | 0% |

Random share is 0.009. I don't read anything into 2k iterations.

## Open questions for Jaden (each one has the conservative option already chosen)

1. **BPTT.** The parity flag refuses BPTT. BPTT's fast writes never reach a forward pass, but it still carries 1/B on the slow weights. Should a 1/B fix for BPTT be added too?
2. **Which layers take B_fast.** I chose `--decouple_fast_feedback 0,1,2`, so no fast write anywhere is gradient-valued. The literal DC2 transplant would be `2` only, which leaves layers 0–1 with gradient-valued writes. Add that arm?
3. **What the α sweep is for.** The spec's "1–2 orders below" reasoning applies to gradient-valued fast writes, and that's S1-on, not S2, whose writes are DFA-valued. S2 keeps α 1e4 as the anchor and adds 1e3 and 1e2. S1-on at α 1e3 and 1e2 isn't prepped: 6 runs, about 34 GPU-h, and the arm table takes it in two lines.
4. **No same-recipe DFA comparator for S2.** That would be DFA `--decouple_fast_feedback 0,1,2 --output_tanh`, fused, about 2 h a run. DC2's earlier shares come from the R-break recipe without tanh.
5. **No held-out eval.** `heldout.py` needs DFA. S2's writes are all B_fast DFA writes, so allowing it there would be valid, but it needs a code change.
6. The optional old-recipe run, 3 × 5M (about 60 GPU-h at DFA speed, more under backprop), isn't prepped.
7. The README "Known issues" now notes the opt-in fix. **The default stays the old step** until these runs decide it.
8. I couldn't run `sbatch --test-only` from omen, because ORC asks for interactive authentication. `bash -n` passes on every launcher.
