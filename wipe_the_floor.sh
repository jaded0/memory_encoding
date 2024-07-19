#!/bin/bash

# Cancel all Slurm jobs for user jaded79
scancel -u jaded79

# Delete all slurm-* files in /home/jaded79/memory_encoding
rm /home/jaded79/memory_encoding/slurm-*

# Delete all offline-run-* directories in /home/jaded79/memory_encoding/wandb/
rm -r /home/jaded79/memory_encoding/wandb/offline-run-*

rm *.log

echo "All Slurm jobs canceled, slurm-* files deleted, and offline-run-* directories removed."

