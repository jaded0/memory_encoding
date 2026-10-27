# C6 follow-ups 2 and 6 (2026-10-27): the task table, sourced by the launchers. Sets, for task $1:
#   name st alpha seed runs target fl init   (fl: --fast_layers value or ''; init: checkpoint to start from, '' = scratch)
# Recipe = Batch FS part L (run_L.sh); gain matched as there (c_eff = lr * alpha * phi_eff = 0.3; phi_eff = 0.1 exclusive/masked, 1 dense).
# Paths are relative to the code directory (ORC ~/c6fu_2026-10-27/code; omen smoke: the worktree).
#  0-8   follow-up 2, fast weights in trunk layer 0 only (--fast_layers 0; i2h keeps its fast weights, as in load_eval's only0), 300k from scratch:
#        0-2 A3f0 exclusive 3e4 | 3-5 D3f0 additive_dense 3e3 | 6-8 M3f0 additive_masked 3e4; seeds 2718 3141 1618 (= the Batch FS 3-seed set)
#  9-18  follow-up 6, M3 vs A3 at 5 seeds to 600k:
#        9-13 A3 exclusive 3e4 and 14-16 M3 additive_masked 3e4: resumed from Batch FS's checkpoint_00300000.pth (exact continuation) into runs_F6
#        17-18 M3 seeds 4241 8128: from scratch to 600k (Batch FS ran M3 on 3 seeds only)
C6_SEEDS=(2718 3141 1618 4241 8128)
C6_FS_RUNS=${C6_FS_RUNS:-$HOME/fslt_2026-10-22/runs_L}   # Batch FS part L runs (ORC)
c6_task() {
  local t=$1
  fl=''; init=''
  if [ $t -lt 9 ]; then
    local a=$((t / 3)); seed=${C6_SEEDS[$((t % 3))]}; runs=../runs_L0; target=300000; fl=0
    case $a in 0) arm=A3f0; st=exclusive; alpha=30000;; 1) arm=D3f0; st=additive_dense; alpha=3000;; 2) arm=M3f0; st=additive_masked; alpha=30000;; esac
  elif [ $t -lt 14 ]; then
    arm=A3; st=exclusive; alpha=30000; seed=${C6_SEEDS[$((t - 9))]}; runs=../runs_F6; target=600000
    init=$C6_FS_RUNS/L-A3-s$seed/checkpoint_00300000.pth
  elif [ $t -lt 17 ]; then
    arm=M3; st=additive_masked; alpha=30000; seed=${C6_SEEDS[$((t - 14))]}; runs=../runs_F6; target=600000
    init=$C6_FS_RUNS/L-M3-s$seed/checkpoint_00300000.pth
  elif [ $t -lt 19 ]; then
    arm=M3; st=additive_masked; alpha=30000; seed=${C6_SEEDS[$((t - 14))]}; runs=../runs_F6; target=600000
  else
    echo "no task $t" >&2; return 2
  fi
  name=L-$arm-s$seed
}
