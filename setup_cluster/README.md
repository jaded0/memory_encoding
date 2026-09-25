# First-time cluster setup

Run once per cluster account, and again after changing the preprocessing
code or adding a Hugging Face dataset. Saves made before the 2026-09 change
to the code hash (`code-e0abe65b67` in the directory name) are no longer found:
run it again once. From the repo root on the **login node**:

```bash
setup_cluster/setup.sh            # download, then sbatch the prepare job
setup_cluster/setup.sh --redo     # re-download and overwrite the processed data
setup_cluster/setup.sh --wait     # also wait for the job and print its log tail
```

Extra arguments are dataset names (default `roneneldan/tinystories`). The
synthetic datasets in `synth_datasets/` need no setup.

`setup.sh` runs the two steps below. You can run them by hand too:

1. `setup_cluster/download_datasets.sh [--redo] [dataset ...]` (login node).
   This only downloads the raw datasets into `~/.cache/huggingface`. The only
   CPU work is HF's parquet-to-arrow conversion, which takes about 11 s of CPU, so
   it stays well inside the login node's 1 CPU-hour limit.
2. `sbatch setup_cluster/prepare_datasets.sbatch [dataset ...]` (from the repo
   root; `REDO=1` in the environment overwrites). This is a `--qos=test` job
   with Hugging Face offline. It runs `python preprocess.py` with one map
   worker per CPU and saves each dataset to the processed-data directory. It
   then runs 20 GPU iterations of the `slurm_run.sh` config, with W&B off and
   a checkpoint in `/tmp/$SLURM_JOB_ID` that is deleted afterwards. To smoke
   test a different launcher, set `CONFIG=run_training.sh`. The log is
   `prepare_datasets_<jobid>.out`.

## Where the data lives

- The raw download is in `~/.cache/huggingface` (about 2 GB for TinyStories).
- The processed data is in `$EPHEMERAL_DATA_DIR`, which defaults to
  `./processed_datasets/`. Each dataset gets one directory, and its name
  includes the split, the row count, a hash of the charset, a hash of the
  preprocessing code (comments and docstrings excluded; see the main README)
  and `PREPROCESS_VERSION`. When any of these changes,
  a training job can't find the directory and stops, telling you to rerun
  setup. It never loads stale rows, and it never preprocesses inside a SLURM
  job (unless `EPHEMERAL_AUTO_PREPROCESS=1` is set). A local run outside
  SLURM prepares the missing directory itself; see the main README.

The saved rows hold text and uint8 character indices. The one-hot tensors are
built per batch, and the batches are identical to the old stored one-hot
pipeline.
