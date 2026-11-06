# K15 scratch launchers (preparation, not authorization)

No script here arms itself. The primary owns commit/push, preregistration, diagnostics,
approval, detached arming and watchers. Do not modify K13 or start work on its GPUs.

## Important design finding: KRX dropped

On base 0f512f8, `EphemeralRNN.set_order2` accepts trunk indices 0 through num_layers-1
only. `trained_layers()` also contains `i2h` and `i2o`. `i2h` is constructed with the
nonzero ephemeral fraction and is a fast layer, even with recurrence disabled;
`i2o` has an all-false ephemeral mask because it is `is_last_layer=True`.
Thus `--order2_layers 0,1,2 --order2_mode replace` does NOT replace every fast layer.
The brief requires dropping KRX, not substituting trunk-only replacement. Default
staging therefore creates exactly 20 new bigkey tasks, no KRX. Unused conditional KRX scaffolding is retained for future work, but gate.py refuses
krx_supported=true on this branch; a separately verified every-fast-layer implementation would require a new code revision;
do not pass `--krx-supported` on this branch merely because trunk replacement works.
The BFR dataset would be `long_range_memory_dataset` copied without regeneration from
`~/bfr_kr_2026-10-22/code/synth_datasets/long_range_memory_dataset` on manticore.

## Layout / preparation after commit

From manticore:

```sh
bash scratch/k15_2026-11-05/stage_deckard.sh FULL_PUSHED_COMMIT
```

This uses the skeleton `deckard:~/k15_2026-11-05/` if present, refuses any existing
checkout/PREPARED/ARMED or nonempty status, requires 100 GiB free, clones a fresh
detached checkout, copies K13's bigkey dataset, verifies all file SHA256 digests,
and writes queue state, queue.txt, and an **unapproved** manifest. It never launches.
K13 data and manifest are read only. An interrupted staging operation requires manual
inspection; rerunning never overwrites an existing checkout.

Before arming, the primary must set manifest `checks_passed=true`, `checks_reference`
(test/smoke/beta finding), `prereg_reference`, and `launch_approved=true`, after all
checks and preregistration. `sink=fixed` is frozen by the launcher/gate. Run
`python scratch/k15_2026-11-05/gate.py ~/k15_2026-11-05/manifest.json` from code.
Do not change manifest after any run starts (exact manifest equality is required for
resume). The gate also requires a clean matching code commit and unchanged data.
Primary should repeat the >=100 GiB check immediately before arming.

## Lanes / queue contract

Each `deckard_lanes.sh i` locks its own lane, GPU `i % 2`, and refuses a completed
lane. Actual K13 emitted format is exactly `LANE DONE lane<i> <ISO timestamp>`
(confirmed from its checked-in launcher; no DONE existed at inspection). It waits
for that exact anchored line plus no K13 GPU PID, checking both command line and
process cwd. Unreadable live process state or nvidia-smi failure blocks. Four lanes
give two lanes/GPU. Do not run two copies of a lane.

Queue state mutations/claims/completions use one flock and atomic JSON replacement;
`queue.txt` is its readable pending projection. Initial arm order is sum2-t4,
sink2-t4, sum2-t16, par2-t16, cycling the five registered seeds inside each arm.
There are 20 new bigkey tasks. K13 `GIVEUP par2 sSEED ...` causes one and only one
new par2-t4 task per seed under the same lock (`REUSE-RERUN`). Reuse is never
rerun for a merely unfinished K13 cell. No KRX may be claimed until all bigkey
tasks, retries and reuse reruns are terminal AND all four K13 DONE markers exist
(so a late K13 GIVEUP cannot slip behind the barrier). A dead lane's active lease
blocks progress deliberately; inspect before manual recovery, never blindly relaunch.

**Retry policy:** at most three requeues / four total attempts, including timeouts;
then GIVEUP. This explicitly chooses the brief's "requeue at most 3 times" wording
rather than K13's three-total-attempt implementation. A run's training timeout is
72 hours cumulative across attempts, never reset by a retry; scoring has 2 hours.
Every attempt emits START and EXIT; all lanes emit LANE DONE only when the global
queue is terminal. Nonzero status/GIVEUP must be reported, not scored as success.

Each run waits for 40 GiB free before claim, uses min_free_disk_gb=15, saves latest
every 5k and numbered checkpoints every 25k, and resumes latest only at the same
commit/manifest. Bigkey recipe, final heldout test strict scoring and K13 analysis
match confirmation. Additional `diagnostics.py CHECKPOINT --device cuda --batches 2
--out FILE --analysis RUN/analysis.json` is required before EVAL EXIT 0 (signature
confirmed by diagnostics owner). KRX, if ever supported, omits optional `--analysis`.

## Checks / observed state

`python scratch/k15_2026-11-05/test_launchers.py` tests queue order/count, global
barrier, late reuse append-once, timeout retries, duplicate active lease refusal, immutable-data/approval/dirty-checkout and unsupported-KRX/sink refusal (8 checks).
`bash -n` all shell launchers. No GPU work is used for these checks.
Read-only deckard inspection on 2026-11-05 showed 411 GiB free, K13 still running,
no LANE DONE yet. Skeleton only is allowed before the primary's final commit.
