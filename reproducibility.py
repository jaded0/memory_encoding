import itertools
import os
import random
import secrets
import subprocess

import numpy as np
import torch


MAX_SEED = (2**63) - 1


def _validate_seed(seed):
    if seed is None:
        return
    if seed < 0 or seed > MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")


def generate_seed():
    """A fresh seed from the OS entropy pool. Never time- or job-ID-based, so
    simultaneous array-job starts cannot collide."""
    return secrets.randbits(63)


def resolve_seed(seed, deterministic, checkpoint=None):
    """Decide this run's (seed, deterministic, source) in one place.

    The seed is fixed on the first fresh start and stored in every checkpoint;
    on resume the checkpoint is the only source of truth. --seed/--deterministic
    (None = not given) may repeat the checkpoint's values but never change them.
    """
    if checkpoint is None:
        deterministic = bool(deterministic)
        if seed is not None:
            _validate_seed(seed)
            return seed, deterministic, "from --seed"
        return generate_seed(), deterministic, "generated"

    saved = checkpoint.get("config", {})
    saved_seed = saved.get("seed")
    saved_deterministic = saved.get("deterministic", False)
    if seed is not None and seed != saved_seed:
        raise ValueError(
            f"--seed {seed} conflicts with the resumed checkpoint's seed "
            f"{saved_seed}; omit --seed to resume"
        )
    if deterministic is not None and deterministic != saved_deterministic:
        raise ValueError(
            f"--deterministic {deterministic} conflicts with the resumed "
            f"checkpoint's deterministic={saved_deterministic}; omit it to resume"
        )
    source = "from checkpoint" if saved_seed is not None else "legacy unseeded checkpoint"
    return saved_seed, saved_deterministic, source


def record_seed_in_slurm(seed):
    """Best effort: show the seed in `squeue -o %k` / `sacct` via the job comment.
    Every failure (no SLURM, no scontrol, permissions, timeout) is ignored."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or seed is None:
        return
    try:
        subprocess.run(
            ["scontrol", "update", f"JobId={job_id}", f"Comment=seed={seed}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
        )
    except Exception:
        pass


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


class ResumableRandomSampler(torch.utils.data.RandomSampler):
    """RandomSampler whose next epoch can start part-way through.

    It draws exactly what RandomSampler draws from the generator; `skip` only
    drops the first indices of the next epoch, so no skipped data is loaded.
    """
    skip = 0

    def __iter__(self):
        skip, self.skip = self.skip, 0
        return itertools.islice(super().__iter__(), skip, None)


class DataStream:
    """Endless batches from a DataLoader whose position survives checkpoints.

    The position is the sampler generator's state when the current epoch began
    plus the number of batches consumed from it. Restoring it replays that
    epoch's permutation (and worker seeds) and skips what was already trained
    on, so a resumed run continues the data stream instead of restarting it.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.generator = getattr(dataloader, "generator", None)
        self.epoch_start_state = None
        self.batches_into_epoch = 0
        self._batches = None

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._batches is None:
                if self.generator is not None:
                    self.epoch_start_state = self.generator.get_state()
                self._batches = iter(self.dataloader)
            try:
                batch = next(self._batches)
            except StopIteration:
                self._batches = None
                self.batches_into_epoch = 0
                continue
            self.batches_into_epoch += 1
            return batch

    def state_dict(self):
        if self.generator is None:
            return None
        return {
            "epoch_start_generator_state": self.epoch_start_state,
            "batches_into_epoch": self.batches_into_epoch,
        }

    def load_state_dict(self, state):
        """Continue from a saved position; False if there is none to continue."""
        if self.generator is None or state is None:
            return False
        if not isinstance(self.dataloader.sampler, ResumableRandomSampler):
            raise TypeError("DataStream can only resume a ResumableRandomSampler")
        if state["epoch_start_generator_state"] is not None:
            self.generator.set_state(state["epoch_start_generator_state"])
            self.dataloader.sampler.skip = (
                state["batches_into_epoch"] * self.dataloader.batch_size
            )
            self.batches_into_epoch = state["batches_into_epoch"]
        return True


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
