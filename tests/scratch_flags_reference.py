"""A short, fixed DFA run of a tiny three-layer EphemeralRNN, for tests/test_fast_layer_flags.py
and tests/generate_scratch_flag_fixtures.py.

It takes the model class and train() as arguments, so the generator can run it against another
checkout (main before the port, or the 10-13 code_v2 snapshot) and the test against this one.
It records each batch's losses through the real train(), then three held-out style fast-only
steps (fast_only_dfa_step), and the final per-sequence weights and biases of every trained layer."""
import contextlib
import io

import torch
import torch.nn.functional as F

CHARSET = list("abcd")
N_BATCHES = 30
SEQUENCE_LENGTH = 6
BATCH_SIZE = 3
LEARNING_RATE = 0.1

# name -> (EphemeralRNN kwargs on main after the port, kwargs in code_v2, SCRATCH_RG_SCALE there)
CASES = {
    "plain": ({}, {}, None),
    "L01": ({"fast_layers": [0, 1]}, {"fast_layers": [0, 1]}, None),
    "DC2": ({"decouple_fast_feedback": [2]}, {"decouple_fast_feedback": [2]}, None),
    "RG2x0.1": ({"fast_write_from_readout": [2], "fast_write_from_readout_scale": 0.1},
                {"fast_write_from_readout": [2]}, "0.1"),
}


def batches():
    generator = torch.Generator().manual_seed(1013)
    return [torch.randint(0, len(CHARSET), (BATCH_SIZE, SEQUENCE_LENGTH), generator=generator)
            for _ in range(N_BATCHES + 1)]


def build(model_class, seed_everything, **kwargs):
    seed_everything(11, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return model_class(len(CHARSET), 4, len(CHARSET), 3, CHARSET, plasticity=3.0,
                           batch_size=BATCH_SIZE, forget_rate=0.25, ephemeral_fraction=0.5,
                           enable_recurrence=False, weight_clamp=1.0, **kwargs)


def run(model, train, log_norms_every=0):
    """The fixed run. log_norms_every N > 0 sets state['log_norms_now'] on every N-th batch, as
    train.py's print steps do (those take the unfused step even under --fused_update)."""
    config = {"updater": "dfa", "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
              "input_mode": "last_one", "pe_matrix": None, "learning_rate": LEARNING_RATE,
              "ephemeral_update_clamp": 0, "grad_norm_clip": 0}
    data = batches()
    losses = []
    for i, batch in enumerate(data[:-1]):
        log_now = log_norms_every > 0 and i % log_norms_every == 0
        with contextlib.redirect_stdout(io.StringIO()):
            _, loss, *_ = train(batch, F.one_hot(batch, len(CHARSET)).float(), model, config,
                                {"training_instance": i, "log_norms_now": log_now})
        losses.append(float(loss))
    # Held-out style fast-only steps on the last batch (heldout.py's fast_only_dfa_step).
    held = data[-1]
    onehot = F.one_hot(held, len(CHARSET)).float()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    hidden = model.initHidden(BATCH_SIZE)
    with torch.no_grad():
        for t in range(3):
            output, hidden = model(onehot[:, t], hidden)
            with torch.enable_grad():
                output.requires_grad_(True)
                loss = criterion(output, onehot[:, t + 1])
                error = torch.autograd.grad(loss, output, torch.ones_like(loss))[0]
            model.fast_only_dfa_step(error, LEARNING_RATE, 0)
    state = {}
    for i, layer in enumerate(model.trained_layers()):
        state[f"{i}.weights"] = layer.per_sample_weights.detach().clone()
        if layer.bias is not None:
            state[f"{i}.bias"] = layer.bias.detach().clone()
    return {"losses": losses, "state": state}
