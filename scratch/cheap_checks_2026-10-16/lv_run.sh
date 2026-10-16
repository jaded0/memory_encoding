#!/bin/bash
# LV measurement runner on deckard. usage: lv_run.sh GROUP GPU   (GROUP: v2 | plain | m)
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
O=~/fastlayers_2026-10-13; mkdir -p $O/lv; export CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=4 LV_BATCHES=4 SCRATCH_FORCE_FUSED=1
ck() { ls $1/checkpoint_00{1,2}*.pth $1/checkpoint_00300000.pth 2>/dev/null | sort -u; }
case $1 in
  v2) cd $O/code_m
    SCRATCH_RG_SCALE=0.1 python $O/lv_measure.py $(ck $O/runs_v2/RGs0.1-s620101) >> $O/lv/lv_all.jsonl 2>> $O/lv/err_v2.log
    SCRATCH_RG_SCALE=0.03 python $O/lv_measure.py $(ck $O/runs_v2/RGs0.03-s620101) >> $O/lv/lv_all.jsonl 2>> $O/lv/err_v2.log
    python $O/lv_measure.py $O/v2ck/Rbreak-*/checkpoint_*.pth >> $O/lv/lv_all.jsonl 2>> $O/lv/err_v2.log ;;
  plain) cd ~/battery_2026-10/code
    for s in 620101 620203 620307; do d=~/battery_2026-10/runs/A1-R-break-plain-s$s
      python $O/lv_measure.py $d/checkpoint_00{05,06,07,08,09,10,11,12,13,14,15,20}0000.pth >> $O/lv/lv_all.jsonl 2>> $O/lv/err_plain.log; done ;;
  m) cd $O/code_m
    for d in $O/lv_tmp/runs_m/*/; do s=$(grep -o 'SCRATCH_RG_SCALE=[0-9.]*' $d/scratch_env.txt | cut -d= -f2)
      SCRATCH_RG_SCALE=$s python $O/lv_measure.py $(ck $d) >> $O/lv/lv_all.jsonl 2>> $O/lv/err_m.log; done ;;
esac
echo "$(date +%T) DONE $1" >> $O/lv/status.txt
