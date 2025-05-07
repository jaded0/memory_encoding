#!/bin/bash

# Cancel all Slurm jobs for user jaded79
scancel -u jaded79

# Delete all slurm-* files in .
rm ./slurm-*
rm ./slurm_logs/*
rm ./*.out

# Delete all offline-run-* directories in ./wandb/
rm -r ./wandb/offline-run-*

rm *.log

echo "All Slurm jobs canceled, slurm-* files deleted, and offline-run-* directories removed."

