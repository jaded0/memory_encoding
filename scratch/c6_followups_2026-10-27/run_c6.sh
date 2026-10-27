#!/bin/bash
# C6 follow-ups 2 and 6: one task of tasks.sh. Usage (from the code directory): run_c6.sh train|eval TASK
#   train: trains the run toward its target (Batch FS part L flags, run_L.sh), resumable. Starts from scratch, from the task's init
#          checkpoint (--resume_checkpoint, the follow-up 6 extensions) or from its own latest_checkpoint.pth (--resume). A SIGUSR1
#          (SLURM's pre-limit warning, forwarded by the sbatch script) stops it cleanly with a checkpoint; the next chunk continues.
#   eval:  load_eval.py (Batch FS's scorer, unchanged) on every kept checkpoint at a multiple of 50k without a load_<k>k.json, and on
#          the last state of a run that stopped early (load_final.json), as run_L.sh / run_L300.sh did.
# Smoke-test overrides (defaults = the Batch FS recipe): C6_TARGET C6_RUNS C6_KEEP C6_SAVE C6_PRINT C6_TRACE C6_EVAL_ARGS,
#   C6_INIT (another start checkpoint; 'none' = from scratch).
mode=$1; task=$2
here=scratch/c6_followups_2026-10-27
source $here/tasks.sh; c6_task $task || exit 2
source sweeps/forward_signals.sh
target=${C6_TARGET:-$target}; runs=${C6_RUNS:-$runs}; init=${C6_INIT:-$init}; [ "$init" = none ] && init=''
keep=${C6_KEEP:-50000}; save=${C6_SAVE:-5000}; print=${C6_PRINT:-2500}; trace=${C6_TRACE:-25000}
d=$runs/$name; mkdir -p $d
commit=$(cat COMMIT 2>/dev/null || git rev-parse HEAD)
stamp() { echo "$(date -Is) $(hostname) task $task $mode commit $commit target $target: $*" >> $d/c6_stages.log; }
finished() {  # the run reached its target, or train.py ended it (non-finite loss / high-loss early stop: a resume would replay it)
  [ -f $d/train.log ] && grep -q -E "Interval metrics \(ending @ iter $target[,) ]|Training terminated due to" $d/train.log
}

if [ $mode = train ]; then
  if finished; then stamp "already finished, nothing to train"; echo "$name finished"; exit 0; fi
  if [ -f $d/latest_checkpoint.pth ]; then start=$d/latest_checkpoint.pth; res="--resume true"
  elif [ -n "$init" ]; then start=$init; res="--resume_checkpoint $init"
  else start=''; res="--resume true"; fi
  if [ -n "$start" ]; then  # refuse a checkpoint from another arm (train.py checks the structure, not --fast_layers, alpha or seed)
    python $here/check_start.py $start --fast_structure $st --plasticity $alpha --seed $seed --fast_layers "$fl" || { stamp "start check FAILED ($start)"; exit 3; }
  fi
  stamp "start ($res)"
  forward_signals python -u train.py --dataset mqarw_r1_52_q8 --model_type ephemeral --updater dfa --fast_structure $st \
    --plasticity $alpha --learning_rate 1e-4 --ephemeral_fraction 0.1 --forget_rate 0.01 \
    --weight_clamp 0 --ephemeral_update_clamp 0 --fast_weight_clamp 0 --slow_weight_decay 0 \
    --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one --positional_encoding_dim 0 \
    --enable_recurrence False --residual_connection False --layer_norm False --fused_update True \
    --deterministic true --track false --seed $seed --n_iters $target --print_freq $print \
    --checkpoint_dir $d --checkpoint_save_freq $save --checkpoint_keep_every $keep \
    --trace_loop_every $trace ${fl:+--fast_layers $fl} $res >> $d/train.log 2>&1
  rc=$?; stamp "TRAIN EXIT $rc"; echo "$name TRAIN EXIT $rc"
  exit 0
fi

if [ $mode = eval ]; then
  for ((it = keep; it <= target; it += keep)); do
    ck=$d/checkpoint_$(printf %08d $it).pth
    [ -f $ck ] || continue
    out=$d/load_$((it / 1000))k.json
    [ -f $out ] && continue
    python -u scratch/fast_structure_2026-10-22/load_eval.py --checkpoint $ck --json $out $C6_EVAL_ARGS >> $d/eval.log 2>&1
    rc=$?; stamp "EVAL $it EXIT $rc"; echo "$name EVAL $it EXIT $rc"
  done
  # Stopped before its target (divergence): evaluate its last state.
  if [ ! -f $d/checkpoint_$(printf %08d $target).pth ] && [ -f $d/latest_checkpoint.pth ] && [ ! -f $d/load_final.json ]; then
    if finished; then
      python -u scratch/fast_structure_2026-10-22/load_eval.py --checkpoint $d/latest_checkpoint.pth \
        --json $d/load_final.json --ablations all $C6_EVAL_ARGS >> $d/eval.log 2>&1
      rc=$?; stamp "EVAL final EXIT $rc"; echo "$name EVAL final EXIT $rc"
    else
      stamp "not finished: no final eval (resubmit the train chunks)"; echo "$name NOT FINISHED"
    fi
  fi
  exit 0
fi
echo "usage: run_c6.sh train|eval TASK" >&2; exit 2
