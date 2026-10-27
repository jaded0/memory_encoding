# S12 ready: the four approved S12 follow-ups (prep, 2026-10-26/27)

## Launch review, 2026-10-27

Follow-up 1 is fixed at the approved 600k; follow-up 3 uses the specified plain KR arm only.
Review corrected `swap`: a zero donor read with a nonzero own read now uses seeded centred,
norm-matched noise. The earlier smoke's swap numbers used the incomplete matching and are
not results of the corrected protocol. Scoring preserves norm/fallback diagnostics. Null
intervention gaps are not an identified causal memory/calibration decomposition.
Launchers now fail on train/eval errors or missing requested checkpoints. `run_short_orc.sh`
runs ratios then both null tasks directly on the checkpoint host; its DONE/FAILED markers
support the shared watcher without an agent polling. The local plain-ln checkpoint subset
uses `GPU=0 bash run_null_manticore.sh`. See the vault's `S12 launched.md` for authoritative
launch IDs, commit, final smoke evidence, and scoring commands.

Prep agent on manticore (GPU 1 for smoke tests). Nothing has been launched beyond the smoke tests below. Spec: proposals page
`### S12 ·`, "Follow-ups: all approved by Jaden 2026-10-26". Background: [[ephemeral weights bounded fast read BFR 2026-10-22]],
[[ephemeral weights BFR2 ln durability 2026-10-23]]. Pre-registration draft: `~/Documents/brain/ephemeral-lowrank/queue_2026-10-27/prereg_drafts/S12.md`
("Batch S12F": L1-L6, N1-N5, A1-A6).

## What's ready

- **Branch `s12-followups`**, based on origin/main 364a07f and pushed. Code commit `ebec980` (flag, eval script, tests); launchers/README in later commits. Launch from the branch head; `stage_orc.sh` writes the commit it staged to `code/COMMIT`, and every run log prints it.
- **`--absrun_floor c`** (follow-up 3; `--fast_read_norm absrun` only): absrun divides by `max(1, n / c)` instead of `max(1, n)`. 0 (default)
  is off and leaves absrun bit for bit; `c = 1` is the same divisor. Refused with any other mode or a negative value. Reaches the model from
  the config, so `heldout.py` and resumes read it from the checkpoint. README "Fast read normalization" has the entry.
  Tests: `tests/test_absrun_floor.py` (6): off/None/0/1 equal absrun exactly (losses and every weight); every read through `train()` equals
  `S x + F x / max(1, coef·mass/c) + b` against an explicit key history, with rows where absrun alone would bind and the floor doesn't;
  the scale is monotone in c; config plumbing; refusals.
- **`ln_null_eval.py`** (follow-up 2): a valid no_fast for ln. At held-out time each ln layer's `LN(F x)` is replaced by
  - `swap`: `LN(F_d x)` from a donor episode (the next held-out batch row for row, run in lockstep under strict), rescaled to the own read's norm;
    when the donor read is zero but the own read is nonzero, seeded centred Gaussian noise supplies the norm-matched fallback;
  - `swap_raw`: `LN(F_d x)` as is;
  - `noise`: centred Gaussian noise rescaled to the own read's norm, fresh per step (seeded).
  It also recomputes heldout.py's strict and no_fast on the same batches, and writes heldout.py's JSON layout plus per-layer norm checks and fallback counts (retained by scoring).
  Differences are null-intervention gaps, not identified causal memory/calibration shares: nulling also changes activations and subsequent writes.
  Tests: `tests/test_ln_null_eval.py` (7). The main one: with the episode as its own donor, swap and swap_raw reproduce heldout.py's strict exactly.
- **Full suite:** 424 passed (CPU, `CUDA_VISIBLE_DEVICES=`), including the golden/identity tests; baseline on 364a07f was 411.

## Launchers (`scratch/s12_2026-10-27/`)

| file | follow-up | runs | where |
|---|---|---|---|
| `s12_lndc2_lc.sbatch` | 1: ln + DC2 on LC | array 0-4: `S12-LC-lnDC2-s{620101..620509}`, 600k (BFR's LC ln flags + `--decouple_fast_feedback 2`); heldout.py + ln_null_eval at the end | ORC, 12 h, resumable |
| `s12_null.sbatch` | 2: null eval | array 0 = KR (plain/DC2 ln × 5 at 30k, 60k, 100k; DC2 also 10k), 1 = LC (ln × 5 at 150k, 600k, 1.2M) | ORC, eval only |
| `run_null_manticore.sh` | 2: null eval | plain ln KR 10k/20k/30k (10k/20k exist only on manticore) | manticore GPU 1 |
| `s12_absfloor_lc.sbatch` | 3: absrun_floor LC | array 0-5: c ∈ {10, 100} × seeds 620101 620203 620307, 300k; heldout.py + bfr_ratio at 150k/300k | ORC, 6 h, resumable |
| `s12_absfloor_kr.sbatch` | 3: absrun_floor KR | array 0-5: plain absrun + c ∈ {10, 100} × seeds 2718 3141 1618, 30k (bfr_kr.sbatch's plain_absrun flags); heldout.py + bfr_ratio at 10k/30k | ORC, 3 h |
| `s12_ratio_r2.sbatch` | 4: BFR2 ratios | bfr_ratio.py on runs2 LC 1.2M and runs2_kr 100k → `~/bfr_2026-10-22/ratios_r2/` | ORC, eval only |
| `s12_score.py` | scoring | `lc`, `kr`, `null` → jsonl, same definitions as bfr_lc.py / bfr2_score.py / kr_collect.py | ORC |
| `stage_orc.sh` | setup | `~/s12_2026-10-27/{code,code_kr}` = git archive + the BFR data links (3pal; RN's KR data), COMMIT file | manticore → ORC |

1.2M for follow-up 1: change `N_ITERS=${N_ITERS:-600000}` to 1200000 in `s12_lndc2_lc.sbatch` (one line); 1.2M needs one resubmit of the same file.

## Launch (ORC; after Jaden's go)

```
# once, on manticore from a checkout with the branch fetched (already staged by the prep agent at the branch head; FORCE=1 only before any run starts):
bash scratch/s12_2026-10-27/stage_orc.sh origin/s12-followups
# then on ORC:
cd ~/s12_2026-10-27/code/scratch/s12_2026-10-27
sbatch s12_lndc2_lc.sbatch       # follow-up 1
sbatch s12_absfloor_lc.sbatch    # follow-up 3, LC
sbatch s12_absfloor_kr.sbatch    # follow-up 3, KR
sbatch s12_null.sbatch           # follow-up 2 (eval only)
sbatch s12_ratio_r2.sbatch       # follow-up 4 (eval only)
# manticore, follow-up 2's local part:  GPU=1 bash scratch/s12_2026-10-27/run_null_manticore.sh
# a task that hit the time limit: resubmit only its id, e.g. `sbatch --array=1,4 s12_lndc2_lc.sbatch` (it resumes from latest_checkpoint.pth).
```

Collect (ORC): `cd ~/s12_2026-10-27 && conda activate hebby && S=code/scratch/s12_2026-10-27/s12_score.py && python $S lc > s12_lc.jsonl &&
python $S kr > s12_kr.jsonl && python $S null null > s12_null.jsonl` (copy manticore's `~/s12_null/*.json` into `null/` first, or pass both dirs).
Ratios: `python code/scratch/bfr_2026-10-22/ratio_summary.py ~/bfr_2026-10-22/ratios_r2` (follow-up 4) and `... ratios` (follow-up 3).

## Cost and placement

| part | GPU-h | placement |
|---|---|---|
| 1: ln + DC2 LC, 5 × 600k | ~30-45 A100-h (round-1 ln: 600k in 5-10 h per node; 2 of 5 needed a resume) | ORC (12 h tasks); ×2 for 1.2M |
| 3: absrun_floor LC, 6 × 300k | ~12-18 A100-h (absrun 600k fit in < 6 h) | ORC |
| 3: absrun_floor KR, 6 × 30k | ~6 A100-h | ORC (or deckard: 2 per A6000) |
| 2: null eval, 50 ORC checkpoints + 15 manticore | ~2-3 A100-h + ~45 min on the 2080 Ti (≈ 3 min per KR checkpoint at 128 batches) | ORC + manticore |
| 4: BFR2 ratios, 15 checkpoints | < 0.5 A100-h | ORC |

## Smoke tests (manticore GPU 1, 2026-10-26/27)

Every launcher was run as the real sbatch file (bash, `SLURM_ARRAY_TASK_ID` set, `BASE=~/s12_smoke`, `N_ITERS=2000`, `CKPT_EVERY=1000`)
by `smoke_manticore.sh` against a git archive of `ebec980` (the code is unchanged since), 3pal data = ORC's copy, KR data = RN's. All exit 0, each < 10 min:

| launcher (task) | wall | train | final evals | checkpoint config |
|---|---|---|---|---|
| `s12_lndc2_lc` (0, s620101) | 395 s | 2000 its, loss 1.10 at 2k, no onset | heldout.py EXIT 0 (strict 0.55); ln_null_eval EXIT 0 (swap 0.11, noise 0.05, no_fast 0.00) | ln, DC2 '2' |
| `s12_absfloor_lc` (0, c = 10) | 293 s | 2000 its, last10 1.84 | heldout EXIT 0; bfr_ratio EXIT 0 (trunk-2 eff/slow 2.6 at 2k) | absrun, floor 10.0 |
| `s12_absfloor_lc` (3, c = 100) | 232 s | 2000 its, last10 1.81 | heldout EXIT 0; bfr_ratio EXIT 0 (eff/slow 3.9) | absrun, floor 100.0 |
| `s12_absfloor_kr` (0, c = 10) | 306 s | 2000 its, no blowup | heldout EXIT 0; bfr_ratio EXIT 0 | absrun, floor 10.0 |
| `s12_absfloor_kr` (3, c = 100) | 306 s | 2000 its, no blowup | heldout EXIT 0; bfr_ratio EXIT 0 | absrun, floor 100.0 |

`s12_score.py lc` and `kr` read these outputs (logs copied to the `logs/<job>_<id>_<task>.out` layout) and `s12_score.py null` read the
null smoke JSON: all rows parse. A 2k smoke says nothing about the predictions.

Follow-up 2's one allowed smoke eval: `smoke_null_manticore.sh`, `ln_null_eval.py` on KR plain ln s1414 at 30k (round 1, a drifted seed),
32 held-out batches, GPU 1, 42 s. strict 1.00, no_fast 0.00, swap 0.13, swap_raw 0.13, noise 0.02 (recall); per-layer norm checks
as designed: swap/noise norm = own norm at trunk 1/2 (31.9, 32.1), trunk 0 donor read 0 at 394 own-nonzero reads (unseen one-hot keys).
At 128 batches a KR checkpoint takes ~3 min on the 2080 Ti.

Review correction (2026-10-27): the smoke above used the old `swap`, which was not fully norm-matched at those 394 L0 reads.
Corrected `swap` uses the seeded centred-noise fallback there; those smoke scores are not results of the corrected protocol.
CPU regression tests: 28 passed (`test_ln_null_eval.py`, `test_absrun_floor.py`, `test_fast_read_norm.py`), including the L0 fallback and diagnostic retention.

ORC dry run (2026-10-27): `stage_orc.sh origin/s12-followups` staged `~/s12_2026-10-27/{code,code_kr}` at `fe7a5e8` (printed `STAGED fe7a5e8…`;
later commits touch only this README). `sbatch --test-only` accepted all five sbatch files; its start estimate for `s12_lndc2_lc` was
2026-11-03 on cs-1-4 (the cs queue is deep; 51 of our jobs were queued or running), so expect a wait.

## Open questions for Jaden

1. Follow-up 1: 600k first (prereg'd), or straight to 1.2M? The launcher does either; BFR2's D1-D3 transfer unchanged to 600k-1.2M.
2. Follow-up 3, KR: plain absrun only (6 runs, the spec's ~6 GPU-h). Add the DC2 absrun arm too (+6 runs)? Round 1 had both.
3. Follow-up 2's donor is the *next* held-out episode under strict. An alternative donor is the same episode's state from a different
   (shuffled) key set; I chose a real episode to retain real donor-state statistics. `swap_raw` vs `swap` can differ through rescaling,
   own-zero gating, and the zero-donor noise fallback. Fine?
4. Placement if the ORC queue stays a week deep: follow-up 2 can run on manticore instead. `~/bfr_kr_2026-10-22/orc_ckpt/` already holds the
   KR 10k/30k and LC 150k/600k checkpoints; the BFR2 ones (KR 60k/100k, LC 1.2M; 25 files, ~8 GB) would need pulling. Follow-up 3's KR part
   fits deckard (2 per A6000, ~1 h each). Follow-ups 1 and 3-LC are A100-sized and stay on ORC.
5. Disclosure: the one allowed smoke eval of follow-up 2 (plain ln s1414 30k, 32 batches) gave strict 1.00, no_fast 0.00, swap 0.13, noise 0.02
   before the prereg was finished; the draft excludes that checkpoint from scoring.
