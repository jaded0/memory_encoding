import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from tests.characterization import all_trace_keys, run_characterization


DEFAULT_OUTPUT = Path(__file__).parent / "fixtures" / "training_traces.json"


def main():
    parser = argparse.ArgumentParser(description="Generate intentional training behavior baselines.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = {
        "schema_version": 1,
        "generated_with": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "torch_threads": 1,
        },
        "traces": {
            key: run_characterization(updater, case=case)
            for key, updater, case in all_trace_keys()
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
