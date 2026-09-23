from contextlib import redirect_stdout
from io import StringIO

import torch

from ephemeral_model import EphemeralRNN
from train import train
from reproducibility import seed_everything


UPDATERS = ("dfa", "backprop", "bptt")
CHARACTERIZATION_SEED = 1729

# The base case (case=None) is the original trace, stored under the bare updater name.
# Each extra case overrides some settings and may run several consecutive train() calls on
# the same model; it is stored as "<updater>/<case>".
SEQUENCES = (
    [[0, 1, 2, 3, 0], [3, 1, 0, 2, 3]],
    [[2, 0, 3, 1, 2], [1, 3, 2, 0, 1]],
)
BASE_SETTINGS = {
    "normalize": False,
    "clip_weights": 0,
    "learning_rate": 0.01,
    "num_sequences": 1,
}
CASES = {
    # clip_weights 0.2 binds after normalize; 1 would not (a unit-norm tensor has no entry
    # above 1). lr 1.0 makes the second BPTT sequence's hidden-layer step (~lr**2) large
    # enough for the rel 1e-6 comparison instead of falling under abs_tol 1e-7.
    "normalize_clip_2seq": {
        "normalize": True,
        "clip_weights": 0.2,
        "learning_rate": 1.0,
        "num_sequences": 2,
    },
}


def trace_key(updater, case=None):
    return updater if case is None else f"{updater}/{case}"


def all_trace_keys():
    for case in (None, *CASES):
        for updater in UPDATERS:
            yield trace_key(updater, case), updater, case


def _tensor_values(tensor):
    tensor = tensor.detach().cpu()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "values": tensor.reshape(-1).tolist(),
    }


def _tensor_summary(tensor):
    tensor = tensor.detach().cpu()
    flat = tensor.to(torch.float64).reshape(-1)
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "sum": flat.sum().item(),
        "norm": torch.linalg.vector_norm(flat).item(),
        "min": flat.min().item() if flat.numel() else None,
        "max": flat.max().item() if flat.numel() else None,
        "first_values": flat[:8].tolist(),
    }


def _named_ephemeral_layers(model):
    for index, layer in enumerate(model.linear_layers):
        yield f"linear_layers.{index}", layer
    yield "i2h", model.i2h
    yield "i2o", model.i2o
    yield "self_grad", model.self_grad


def _instrument_updates(model):
    events = {
        name: {"forget": [], "update": []}
        for name, _layer in _named_ephemeral_layers(model)
    }

    for name, layer in _named_ephemeral_layers(model):
        original_forget = layer.apply_forget_step
        original_update = layer.apply_update

        def record_forget(original=original_forget, event_log=events[name]["forget"], layer=layer):
            before = _tensor_summary(layer.candidate_weights)
            original()
            event_log.append({
                "before": before,
                "after": _tensor_summary(layer.candidate_weights),
            })

        def record_update(
            learning_rate,
            grad_clip,
            state,
            original=original_update,
            event_log=events[name]["update"],
            layer=layer,
        ):
            before = _tensor_summary(layer.candidate_weights)
            gradient = (
                _tensor_summary(layer.candidate_weights.grad)
                if layer.candidate_weights.grad is not None
                else None
            )
            original(learning_rate, grad_clip, state)
            event_log.append({
                "before": before,
                "gradient": gradient,
                "after": _tensor_summary(layer.candidate_weights),
            })

        layer.apply_forget_step = record_forget
        layer.apply_update = record_update

    scale_events = []
    original_scale = model.scale_ephemeral_grads

    def record_scale(plast_clip):
        raw = {
            name: (
                _tensor_summary(layer.candidate_weights.grad)
                if layer.candidate_weights.grad is not None
                else None
            )
            for name, layer in _named_ephemeral_layers(model)
        }
        original_scale(plast_clip)
        scaled = {
            name: (
                _tensor_summary(layer.candidate_weights.grad)
                if layer.candidate_weights.grad is not None
                else None
            )
            for name, layer in _named_ephemeral_layers(model)
        }
        scale_events.append({"raw": raw, "scaled": scaled})

    model.scale_ephemeral_grads = record_scale
    return {"layers": events, "gradient_scaling": scale_events}


def _module_snapshot(layer):
    gradient = layer.candidate_weights.grad
    return {
        "weight": _tensor_summary(layer.weight),
        "candidate_weights": _tensor_values(layer.candidate_weights),
        "candidate_gradient": _tensor_values(gradient) if gradient is not None else None,
        "bias": _tensor_values(layer.bias) if layer.bias is not None else None,
        "mask": _tensor_values(layer.mask),
        "plasticity": _tensor_values(layer.plasticity),
        "forgetting_factor": _tensor_values(layer.forgetting_factor),
        "feedback_weights": _tensor_summary(layer.feedback_weights),
        "plasticity_feedback_weights": _tensor_summary(layer.plasticity_feedback_weights),
        "in_traces": _tensor_values(layer.in_traces),
        "out_traces": _tensor_values(layer.out_traces),
        "last_high_plast_update_norm": layer.last_high_plast_update_norm.item(),
        "last_low_plast_update_norm": layer.last_low_plast_update_norm.item(),
        "t": layer.t.item(),
    }


def _onehot(sequence_indices, charset):
    return torch.nn.functional.one_hot(
        sequence_indices, num_classes=len(charset)
    ).to(torch.float32)


def run_characterization(updater, seed=CHARACTERIZATION_SEED, case=None):
    if updater not in UPDATERS:
        raise ValueError(f"unknown updater: {updater}")
    if case is not None and case not in CASES:
        raise ValueError(f"unknown case: {case}")
    settings = {**BASE_SETTINGS, **(CASES[case] if case is not None else {})}

    torch.set_num_threads(1)
    seed_everything(seed, deterministic=True)

    charset = list("abcd")
    batch_size = 2
    sequences = [
        torch.tensor(SEQUENCES[index], dtype=torch.long)
        for index in range(settings["num_sequences"])
    ]

    with redirect_stdout(StringIO()):
        model = EphemeralRNN(
            input_size=len(charset) * 2,
            hidden_size=4,
            output_size=len(charset),
            num_layers=1,
            charset=charset,
            normalize=settings["normalize"],
            residual_connection=False,
            clip_weights=settings["clip_weights"],
            updater=updater,
            plast_clip=3.0,
            batch_size=batch_size,
            forget_rate=0.25,
            plast_proportion=0.5,
            enable_recurrence=True,
        )

    model.train()
    events = _instrument_updates(model)
    optimizer = (
        None
        if updater == "dfa"
        else torch.optim.SGD(model.parameters(), lr=settings["learning_rate"])
    )
    config = {
        "updater": updater,
        "criterion": torch.nn.CrossEntropyLoss(reduction="mean"),
        "input_mode": "last_two",
        "pe_matrix": None,
        "self_grad": 0.0,
        "learning_rate": settings["learning_rate"],
        "grad_clip": 0.2,
        "plast_clip": 3.0,
    }
    state = {"training_instance": 0, "log_norms_now": True}

    calls = []
    for sequence_indices in sequences:
        with redirect_stdout(StringIO()):
            output, loss, _step_preds, _step_losses, step_outputs, step_labels = train(
                sequence_indices,
                _onehot(sequence_indices, charset),
                model,
                config,
                state,
                optimizer=optimizer,
                log_outputs=True,
            )
        calls.append({
            "sequence_indices": _tensor_values(sequence_indices),
            "loss": loss,
            "final_output": _tensor_values(output),
            "step_outputs": [_tensor_values(value) for value in step_outputs],
            "step_labels": [_tensor_values(value) for value in step_labels],
        })

    configuration = {
        "batch_size": batch_size,
        "sequence_length": sequences[0].shape[1],
        "input_mode": "last_two",
        "hidden_size": 4,
        "num_layers": 1,
        "normalize": settings["normalize"],
        "residual_connection": False,
        "clip_weights": settings["clip_weights"],
        "learning_rate": settings["learning_rate"],
        "grad_clip": 0.2,
        "plast_clip": 3.0,
        "forget_rate": 0.25,
        "plast_proportion": 0.5,
        "enable_recurrence": True,
    }
    trace = {
        "schema_version": 1,
        "updater": updater,
        "seed": seed,
        "configuration": configuration,
    }
    if case is None:
        # Original single-call layout, kept flat so the base traces stay byte-identical.
        trace.update(calls[0])
    else:
        trace["case"] = case
        configuration["num_sequences"] = settings["num_sequences"]
        trace["calls"] = calls
    trace.update({
        "state": state,
        "modules": {
            name: _module_snapshot(layer)
            for name, layer in _named_ephemeral_layers(model)
        },
        "events": events,
    })
    return trace
