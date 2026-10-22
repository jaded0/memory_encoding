"""GPU check (the CPU suite skips it without a C++ compiler): the compiled fused additive step
agrees with the eager one to rounding, at a small size and at hidden 1024, for both additive
structures and exclusive. Run from the code directory: python scratch/fast_structure_2026-10-22/compiled_check.py"""
import contextlib
import io
import sys

import torch

sys.path.insert(0, ".")
import train as train_module  # noqa: E402
from ephemeral_model import EphemeralRNN  # noqa: E402
from reproducibility import seed_everything  # noqa: E402

device = "cuda"
charset = "abcdefgh"


def build(structure, hidden, compile_):
    seed_everything(5, deterministic=False)
    with contextlib.redirect_stdout(io.StringIO()):
        model = EphemeralRNN(len(charset), hidden, len(charset), 3, charset, updater="dfa", batch_size=4,
                             plasticity=3000.0 if structure == "additive_dense" else 30000.0,
                             forget_rate=0.01, ephemeral_fraction=0.1, enable_recurrence=False,
                             fast_structure=structure).to(device)
    model.enable_fused_update(compile=compile_)
    return model


def run(model, steps=4, length=12):
    g = torch.Generator().manual_seed(1)
    cfg = {"updater": "dfa", "criterion": torch.nn.CrossEntropyLoss(reduction="none"), "input_mode": "last_one",
           "pe_matrix": None, "learning_rate": 1e-4, "ephemeral_update_clamp": 0, "grad_norm_clip": 0}
    for _ in range(steps):
        idx = torch.randint(0, len(charset), (4, length), generator=g)
        onehot = torch.nn.functional.one_hot(idx, len(charset)).float().to(device)
        with contextlib.redirect_stdout(io.StringIO()):
            train_module.train_batch(None, onehot, model, cfg, {"training_instance": 0, "log_norms_now": False})


worst = 0.0
for hidden in (16, 1024):
    for structure in ("exclusive", "additive_masked", "additive_dense"):
        eager, compiled = build(structure, hidden, False), build(structure, hidden, True)
        run(eager)
        run(compiled)
        rel = 0.0
        for key, value in eager.state_dict().items():
            other = compiled.state_dict()[key]
            if value.dtype.is_floating_point and value.abs().max() > 0:
                rel = max(rel, ((other - value).abs().max() / value.abs().max()).item())
        fast = [l.fast_weights().abs().max().item() for l in eager.linear_layers]
        print(f"hidden {hidden} {structure}: max relative state difference compiled vs eager {rel:.2e}; "
              f"max |fast| per trunk layer {['%.3g' % f for f in fast]}")
        worst = max(worst, rel)
print("PASS" if worst < 1e-4 else "FAIL", worst)
