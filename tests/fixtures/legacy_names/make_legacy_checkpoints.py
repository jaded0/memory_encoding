"""Makes the old-name checkpoints in this directory. Only meaningful at commit 1775121 (before
the state-dict and CLI renames); it was run there, from the repository root, as

    CUDA_VISIBLE_DEVICES="" python tests/fixtures/legacy_names/make_legacy_checkpoints.py tests/fixtures/legacy_names

For each case, iter3.pth is a checkpoint after 3 iterations, and iter5.pth is what the old code
saved after resuming it with --resume for 2 more. The flags are the old names, as a frozen
run_used.sh would pass them. tests/test_legacy_checkpoints.py resumes iter3.pth with today's
code and requires today's checkpoint to equal iter5.pth after mapping the names.
"""
import os, shutil, sys, tempfile
sys.path.insert(0, os.getcwd())
from tests.test_seed_resume import run_main

OUT = sys.argv[1]
CASES = {
    "dfa": ["--updater", "dfa", "--normalize", "True"],
    "backprop": ["--updater", "backprop"],
    "rnn_backprop": ["--model_type", "rnn", "--updater", "backprop"],
}
COMMON = ["--seed", "11", "--deterministic", "True", "--learning_rate", "0.05",
          "--plast_clip", "3", "--plast_proportion", "0.5", "--grad_clip", "0.2",
          "--clip_weights", "0.5", "--forget_rate", "0.25",
          "--plast_learning_rate", "0.005", "--imprint_rate", "0"]
for name, extra in CASES.items():
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        run_main(*COMMON, *extra, "--n_iters", "3", "--checkpoint_save_freq", "3", checkpoint_dir=d)
        shutil.copy(os.path.join(d, "latest_checkpoint.pth"), os.path.join(OUT, name, "iter3.pth"))
        run_main(*COMMON, *extra, "--resume", "--n_iters", "5", "--checkpoint_save_freq", "5", checkpoint_dir=d)
        shutil.copy(os.path.join(d, "latest_checkpoint.pth"), os.path.join(OUT, name, "iter5.pth"))
print("ok")
