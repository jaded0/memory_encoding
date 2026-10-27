# C6 follow-ups 2 and 6: ready to launch (prep only, nothing launched)

Prepared 2026-10-26/27 night by the C6 prep agent on omen. Approved by Jaden 2026-10-26 23:20 ("Yes to c6 followups"). In the queue, C6 comes after the S12 follow-ups. Spec: `### C6 ·` in the vault's `ephemeral weights experiment proposals.md`, follow-ups 2 and 6. Pre-registration draft: vault `ephemeral-lowrank/queue_2026-10-27/prereg_drafts/C6.md` ("Batch C6F").

## What's ready

Branch `c6-followups` (base origin/main 364a07f). Commit 71347cc (see the end).

- **No model code change was needed.** `--fast_layers 0` already composes with every `--fast_structure`. A trunk layer outside `--fast_layers` gets ephemeral_fraction 0, and any layer with fraction 0 is "exclusive" (all-slow), whatever the structure. `tests/test_fast_layers_structure.py` (new, 7 tests, 20 subtests) pins this for exclusive, additive_masked and additive_dense on a 3-layer trunk. It covers layer typing, where the writes land, the fused (eager) step against the unfused one, the CLI with checkpoint and resume, the loop tracer, and `load_eval.zero_fast` on the slow layers. The default path is untouched (no file outside `tests/`, `scratch/` and one README line changed).
- **Note on "layer 0 only":** `--fast_layers 0` keeps `i2h`'s fast weights, because i2h is not a trunk layer. That makes it the training-time twin of load_eval's read-time `only0`, which also keeps i2h. Making i2h slow too would need a new flag (open question 1).
- **Follow-up 6: extension, not from scratch.** All 8 Batch FS checkpoints at 300k exist on ORC (`~/fslt_2026-10-22/runs_L/L-{A3,M3}-s*/checkpoint_00300000.pth`). A3 ×5 and M3 ×3 resume from them with `--resume_checkpoint` into a new directory (`runs_F6/`), as BFR2 did, so the Batch FS run directories stay untouched. M3 seeds 4241 and 8128 never existed and run from scratch to 600k. That gives 5 matched seed pairs.
- Launchers (this directory):
  - `tasks.sh`: the 19-task table. Tasks 0-8 are follow-up 2: A3f0 / D3f0 / M3f0 × seeds 2718 3141 1618, 300k. Tasks 9-18 are follow-up 6.
  - `run_c6.sh train|eval TASK`: the Batch FS run_L.sh flags verbatim, plus `--fast_layers` where the task has it. It is resumable and logs the code commit to `<run>/c6_stages.log`. Before using any checkpoint, `check_start.py` refuses one from another arm: it checks structure, alpha, seed and fast_layers, because train.py's resume does not check the last three.
  - `c6_train.sbatch`: one-day chunks with `--signal=B:USR1@600` through `sweeps/forward_signals.sh`. A run stops cleanly with a checkpoint and the next chunk continues it.
  - `c6_eval.sbatch`: Batch FS's `load_eval.py` (unchanged) on every kept 50k checkpoint.
  - `submit.sh TASKS CHUNKS`: chains the chunks with `--dependency=aftercorr`, then the eval array.
  - `setup_orc.sh [REF]`: git archive to `~/c6fu_2026-10-27/code` plus `COMMIT`. It refuses to replace code once runs exist.
- Scoring: `collect_L.py` (Batch FS, unchanged) on `runs_L0` and `runs_F6`, then `score_c6.py FS.jsonl L0.jsonl F6.jsonl`. The metrics are the same as Batch FS part L: held-out strict per load, AREA, K50, read-time ablations, blowup and memory onset. FS.jsonl is the existing `~/fslt_2026-10-22/collect_L_orc_300k.jsonl` (full-model references and the 50k-300k history of the extended runs). The scorer was exercised on stand-in rows built from that file. It reports a missing part instead of scoring it.

## Launch (ORC, from manticore, which has the ssh master)

```bash
# 1. code (after the main session merges or picks the ref); ORC currently has a staging copy at 9b1fd1f, replaced by this step
cd ~/memory_encoding && git fetch && bash scratch/c6_followups_2026-10-27/setup_orc.sh origin/c6-followups   # or the merge commit
# 2. submit (each line prints the chained job ids; ../submitted.log records them)
ssh orc 'bash -lc "cd ~/c6fu_2026-10-27/code && bash scratch/c6_followups_2026-10-27/submit.sh 9-18 4"'   # follow-up 6
ssh orc 'bash -lc "cd ~/c6fu_2026-10-27/code && bash scratch/c6_followups_2026-10-27/submit.sh 0-8 3"'    # follow-up 2
# 3. score when the eval arrays finish
ssh orc 'bash -lc "conda activate hebby; cd ~/c6fu_2026-10-27/code && S=scratch/fast_structure_2026-10-22 && \
  python $S/collect_L.py ../runs_L0 ../collect_L0.jsonl && python $S/collect_L.py ../runs_F6 ../collect_F6.jsonl && \
  python scratch/c6_followups_2026-10-27/score_c6.py ~/fslt_2026-10-22/collect_L_orc_300k.jsonl ../collect_L0.jsonl ../collect_F6.jsonl"'
```

If a chunk dies (node failure, or the hard limit kills it before USR1 lands), its `aftercorr` successors never start. Rerun the same `submit.sh` line: every run resumes from its own latest checkpoint, and finished runs exit within a minute.

## GPU-hours and placement

From Batch FS's A100 timings (sacct: 150k took ~11 h for A3 and ~14 h for M3, evaluations included; median 3.9 and 3.2 it/s):

| part | runs | per run | A100-h |
|---|---|---|---|
| Z (follow-up 2) A3f0 | 3 × 300k | ~22 h | ~65 |
| Z D3f0, M3f0 | 6 × 300k | ~22-27 h (fewer additive layers than D3/M3, so probably faster) | ~150 |
| X (follow-up 6) A3 extension | 5 × 300k more | ~22 h | ~110 |
| X M3 extension | 3 × 300k more | ~27 h | ~80 |
| X M3 new seeds | 2 × 600k | ~55 h | ~110 |
| evaluations | 9 × 6 + 8 × 6 + 2 × 12 checkpoints | ~0.3 h each | ~40 |
| **total** | 19 runs | | **~550** |

The proposals table's "~8-14 h per run" is per 150k chunk, not per run. Placement: **ORC** (qos cs, A100) for everything, as Batch FS did. That's 19 concurrent GPUs at peak, so it shares fairshare with the other queued arrays. Wall-clock with no queue wait is about 1-1.2 days for Z and the extensions, and about 2.4 days for the two new M3 seeds, which are the critical path. Not deckard or omen: 150k mqarw takes ~28 h on one A6000, and omen's 4090 ran M3 at ~1-1.5 it/s in the smoke tests.

If fairshare is tight, submit follow-up 6 first. Its extensions answer the 5-seed question in ~1 day, and its two new M3 seeds take longest.

## Smoke-test evidence (omen RTX 4090, each step < 10 min and < 2 GB, run only while the GPU was otherwise idle)

`smoke.sh STEP OUT OLD_CODE` (logs were in omen `/tmp/c6_smoke/`). Losses are compared at print precision (4 decimals), every 25 iterations.
- **equivM / equivA**: M3 (task 17) and A3 (task 9 from scratch) for 150 its with this branch and with Batch FS's ORC code (a195879). Losses were **identical** at every print (M3 25:4.1701 … 150:4.1501; A3 25:4.1725 … 150:4.1518). So the extensions and the new M3 seeds train exactly as Batch FS's runs did.
- **fu2**: tasks 0, 3, 6 (A3f0, D3f0, M3f0) to 100 its (TRAIN EXIT 0 ×3). Then load_eval (2 loads × 2 batches) and collect_L.py: `read_all == strict` True and `only0 == all` True in all three, and the fast/slow drive ratio of trunk layers 1-2 is exactly 0 (layer 0: 1.36 / 0.41 / 1.29 for A3f0 / D3f0 / M3f0 at 50 its). On the 4090, at 2.2 / 1.8 / 1.8 it/s. (Smoke-only quirk: with 50-its spacing both evals map to `load_0k.json`, so the second was skipped. At 50k spacing the names are distinct.)
- **ext**: task 17 resumed from the equivM run's checkpoint at 50 with `--resume_checkpoint` into a new directory (`check_start: ok`), then trained to 150. The 75-150 losses were **identical** to the uninterrupted run's. That is the follow-up 6 extension path.
- **chunk**: USR1 sent to the outer shell at iteration 75 passed through both forward_signals layers. Result: "Received SIGUSR1: stopping before iteration 76", TRAIN EXIT 124, and the script still exited 0. Chunk 2 resumed and finished at 150. A spare chunk 3 reported "already finished, nothing to train". The losses after the stop were **identical** to the uninterrupted run's.
- CPU suite: `pytest tests` on omen: 413 passed, 10 skipped, 1 failed. The failure is the known environment one (C8): the golden traces were recorded under torch `2.5.1` and omen has `2.5.1+cu121`. It fails the same way on main.
- ORC: `submit.sh 0-18 3 --test-only` was accepted for both sbatch files ("to start ... on cs-1-1"). `check_start.py` accepted the real 300k checkpoints of L-M3-s2718 and L-A3-s8128 and refused a wrong seed (rc 1). All 8 init checkpoints were found. Data: omen's regenerated `mqarw_r1_52_q8` and `mqarw_52_q8` are md5-identical to ORC's in every split.

## Open questions for Jaden

1. "Trunk layer 0 only" as prepared keeps i2h's fast weights. That matches the read-time only0 result (~0.99 at K 52), which also kept them. Do you also want an arm with i2h slow ("only layer 0 fast anywhere")? That needs a small new flag and 3 more runs.
2. Follow-up 6 extends the existing 3 M3 + 5 A3 runs (exact continuation) and adds 2 new M3 seeds from scratch. OK, or do you prefer 5 fresh seeds per arm (~2× the compute, all from scratch)?
3. If both arms are still rising at 600k (X4), do you want 1.2M? Nothing like that is pre-approved or pre-registered.

## Branch and commit

`c6-followups` at **71347cc** (code, tests, launchers, scorer; the following commit only writes this hash). Pushed to origin, not merged. ORC `~/c6fu_2026-10-27/code` holds a staging copy of 9b1fd1f (identical launch path); `setup_orc.sh` replaces it at launch.
