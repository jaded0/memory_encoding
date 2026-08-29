import os
import random

import numpy as np
import torch


MAX_SEED = (2**63) - 1


def _validate_seed(seed):
    if seed is None:
        return
    if seed < 0 or seed > MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")


def seed_everything(seed=None, deterministic=False):
    """Seed supported RNGs without changing the unseeded default path."""
    if deterministic and seed is None:
        raise ValueError("--deterministic requires --seed")
    if seed is None:
        return

    _validate_seed(seed)
    if deterministic:
        workspace_config = os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
        )
        if workspace_config not in (":4096:8", ":16:8"):
            raise ValueError(
                "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 "
                "for deterministic mode"
            )

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def make_torch_generator(seed=None):
    if seed is None:
        return None

    _validate_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_data_worker(_worker_id):
    """Seed non-Torch RNGs from the worker seed assigned by DataLoader."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state():
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint):
    """Restore all RNG states present in a new or legacy checkpoint."""
    python_state = checkpoint.get("python_rng_state")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = checkpoint.get("numpy_rng_state")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = checkpoint.get("torch_rng_state")
    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu())

    cuda_states = checkpoint.get("cuda_rng_state_all")
    if cuda_states is not None and torch.cuda.is_available():
        for device_index, cuda_state in enumerate(cuda_states[:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(cuda_state.cpu(), device_index)
