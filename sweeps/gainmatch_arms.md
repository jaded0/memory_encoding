# Gain-matched aligned-feedback arms

Run from `~/lowrank_deep/aligned_feedback` with the existing `scripts/run_arm.sh` recipe. These
commands use hidden width 256, 80k iterations, and the three pilot seeds; they are listed only and
must not be launched from this worktree.

```bash
for seed in 2718 3141 4241; do scripts/run_arm.sh gainmatch_random_h256_s${seed} 256 "$seed" 80000 --feedback_init random; done
for seed in 2718 3141 4241; do scripts/run_arm.sh gainmatch_aligned_h256_s${seed} 256 "$seed" 80000 --feedback_init aligned; done
for seed in 2718 3141 4241; do scripts/run_arm.sh gainmatch_aligned_spectrum_matched_h256_s${seed} 256 "$seed" 80000 --feedback_init aligned_spectrum_matched; done
for seed in 2718 3141 4241; do scripts/run_arm.sh gainmatch_random_spectrum_of_J_h256_s${seed} 256 "$seed" 80000 --feedback_init random_spectrum_of_J; done
```
