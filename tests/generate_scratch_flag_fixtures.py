"""Writes tests/fixtures/scratch_flags.pt: tests/scratch_flags_reference.py's fixed run, computed
by other checkouts, for tests/test_fast_layer_flags.py.

    python tests/generate_scratch_flag_fixtures.py --main <main checkout> --code-v2 <code_v2 dir>

--main: main before the port (4697e4e); its "plain" run is what the flags-off model must still
give. --code-v2: the 10-13 scratch snapshot the L01/DC2/aimed-write runs used
(deckard ~/fastlayers_2026-10-13/code_v2: main 5242f3c + patch_fast_layers.py, the decouple and
readout-gradient patches, patch_eval_fix.py and patch_force_fused.py). Its scratch flags exist only
in the fused step, so it is run with the eager fused step (enable_fused_update(compile=False)).
Each checkout runs in its own process, on the CPU with one thread."""
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "scratch_flags_reference.py"
DEFAULT_OUTPUT = HERE / "fixtures" / "scratch_flags.pt"

CHILD = r"""
import importlib.util, sys, torch
torch.set_num_threads(1)
source, reference, out, which = sys.argv[1:5]
sys.path.insert(0, source)
from ephemeral_model import EphemeralRNN
from reproducibility import seed_everything
from train import train
spec = importlib.util.spec_from_file_location("scratch_flags_reference", reference)
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
results = {}
for name, (_, v2_kwargs, _) in ref.CASES.items():
    if which == "main" and name != "plain":
        continue
    model = ref.build(EphemeralRNN, seed_everything, **(v2_kwargs if which == "code_v2" else {}))
    if which == "code_v2":
        model.enable_fused_update(compile=False)
    results[name] = ref.run(model, train)
torch.save(results, out)
"""


def run_checkout(source, which, out):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    if which == "code_v2":
        env["SCRATCH_RG_SCALE"] = "0.1"  # read once at import: the RG2x0.1 case
    subprocess.run([sys.executable, "-c", CHILD, str(source), str(REFERENCE), str(out), which],
                   check=True, env=env, cwd=source)
    return torch.load(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--code-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    tmp = args.output.with_suffix(".tmp")
    payload = {
        "generated_with": {"python": platform.python_version(), "torch": str(torch.__version__),
                           "device": "cpu", "torch_threads": 1},
        "main": run_checkout(args.main.resolve(), "main", tmp),
        "code_v2": run_checkout(args.code_v2.resolve(), "code_v2", tmp),
    }
    tmp.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
