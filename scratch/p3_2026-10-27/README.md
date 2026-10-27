# P3 — expansive write gain (2026-10-27)

Approved: control plus `--write_gain_power {1,2,4}` × seeds 2718, 3141, 1618;
100k key-recall iterations each. No optional sensitivity arms are launched.
The exact recipe and predictions are in the vault's `queue_2026-10-27/prereg_drafts/P3.md`.

## Placement and launch

Use omen's 16-GB RTX 4090 Laptop GPU, two sequential lanes (12 runs, roughly
12–20 hours total including evaluations; no ORC allocation). Pull main in
`~/memory_encoding`, then make a detached worktree at `~/p3_2026-10-27/code` so
later merges cannot change queued code. The run tree's `synth_datasets` symlink
points at `~/p3_2026-10-27/synth_datasets`, a copy of manticore's key-recall data:
omen's existing key-recall dataset differs and is left untouched. Verify
`cd ~/p3_2026-10-27 && sha256sum -c data.sha256` before starting.

```bash
cd ~/p3_2026-10-27/code
systemd-run --user --unit=p3-20261027 --collect \
  --property=WorkingDirectory="$PWD" \
  --setenv=GPU=0 --setenv=CONCURRENCY=2 \
  --setenv=OUT="$HOME/p3_2026-10-27/runs" \
  /bin/bash scratch/p3_2026-10-27/p3_local_queue.sh \
  ctl:2718 g1:2718 g2:2718 g4:2718 \
  ctl:3141 g1:3141 g2:3141 g4:3141 \
  ctl:1618 g1:1618 g2:1618 g4:1618
```

Each run records its commit, host, GPU and launch time. Infrastructure errors
fail that run but do not skip other arms. Nonfinite training is a scientific
outcome: evaluate the last saved checkpoint, if available, and retain the
`nan` status. Such evaluations are explicitly pre-failure, not final-step
measurements. `QUEUE_DONE` means execution/scoring finished; `QUEUE_FAILED`
means at least one execution/scoring error. Neither means the hypothesis passed.
The queue is registered with the existing watcher; do not babysit it with an agent.

## Evaluation and scoring

The launcher automatically runs `heldout.py` (observed/strict/no_fast; 128
batches) and salience decomposition at 50k and 100k (128 batches each). If
training stops, use the last saved checkpoint in addition to an existing 50k.
SG refuses a nonfinite or >1e-4 median reconstruction error. Its
`store_largest_no_gain` is a **fixed-trajectory** counterfactual, not a fresh
ungained model replay. The control reproduces all original cheap-check SG keys.

```bash
ssh omen 'cd ~/p3_2026-10-27/code; source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby; python scratch/p3_2026-10-27/p3_collect.py ~/p3_2026-10-27/runs'
ssh omen 'cat ~/p3_2026-10-27/runs/status.txt; systemctl --user status p3-20261027 --no-pager'
# Example exact per-checkpoint re-scoring (select an idle GPU):
python heldout.py --checkpoint RUN/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 --json RUN/final_eval.json
python scratch/p3_2026-10-27/p3_sg.py RUN/checkpoint_00050000.pth RUN/checkpoint_00100000.pth --out RUN/sg --batches 128 --device cuda
```

## Prep checks

CPU full suite, default/golden identity, gained-step parity/history/resume tests,
and analytical SG control-parity/log-split tests are required before merging.
A <=10-minute full-width compiled g4 smoke runs on manticore GPU 1 with 200
iterations and two batches per evaluation; evidence and final commit are in
the vault's `P3 launched.md`. Earlier prep-only 1000-iteration smoke results
were disclosed in the pre-registration; they are not hypothesis outcomes.
