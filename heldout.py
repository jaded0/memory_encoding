"""Held-out evaluation of a DFA EphemeralRNN with its slow weights frozen.

Each episode starts as a training sequence does (start_sequence_wipe: the batch's slow copies
become their mean and the fast entries zero). Each step then predicts, is scored, and only
then sees its target, which writes the fast entries with the training DFA step
(EphemeralRNN.fast_only_dfa_step: the same helpers, clamps and forgetting as train.py). Slow
entries, biases and the output head i2o stay bit for bit. Three protocols:

  observed  every target writes, as in training (teacher-forced writes during the answer)
  strict    no writes from the step that predicts the first recall target onward; the fast
            entries still forget each step, so "no writes" is not "no change"
  no_fast   no fast weights at all: they stay at their wiped zeros (the slow scaffold alone)

Metrics are metrics.IntervalMetrics over the episodes (recall_acc, recall_acc_lag_<k>, ...)
plus first_answer_acc, the accuracy on each episode's first recall target, which no answer
write can have helped in any protocol.

Used by train.py --heldout_eval_every N, or on a saved checkpoint:
    python heldout.py --checkpoint PATH [--dataset NAME] [--protocols observed strict no_fast]
"""
import argparse
import json

import torch
from datasets import load_from_disk

from ephemeral_model import EphemeralRNN, dfa_output_error
from metrics import IntervalMetrics, recall_targets
from preprocess import OneHotCollate, is_synthetic, preprocess_rows
from reproducibility import capture_rng_state, restore_rng_state
from utils import initialize_charset, load_checkpoint, model_input, read_checkpoint, upgrade_legacy_config

PROTOCOLS = ("observed", "strict", "no_fast")


@torch.no_grad()
def evaluate_held_out(model, onehot, update_mask, config):
    """Runs one batch of episodes, onehot [B, T, vocab], prequentially with frozen slow weights.
    update_mask [B, T-1] (bool) selects the steps whose target writes the fast entries; None
    means no fast weights. config supplies input_mode, pe_matrix, learning_rate,
    ephemeral_update_clamp and grad_norm_clip, as in training. Returns the predictions and
    per-sequence losses, [T-1, B] each (IntervalMetrics's layout). Changes the model's state."""
    if not isinstance(model, EphemeralRNN):
        raise ValueError("held-out evaluation needs an EphemeralRNN (SimpleRNN has no fast weights)")
    model.check_fast_only_step()
    batch, steps = onehot.shape[0], onehot.shape[1] - 1
    if batch != model.batch_size:
        raise ValueError(f"batch size {batch} does not match the model's {model.batch_size}")
    if update_mask is not None and (update_mask.dtype != torch.bool or update_mask.shape != (batch, steps)):
        raise ValueError(f"update_mask must be boolean [{batch}, {steps}]")
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    model.start_sequence_wipe()
    hidden = model.initHidden(batch)
    preds, losses = [], []
    for i in range(steps):
        output, hidden = model(model_input(onehot, i, config['input_mode'], config['pe_matrix']), hidden)
        loss, output_error = dfa_output_error(output, onehot[:, i + 1], criterion)
        preds.append(output.argmax(dim=1))
        losses.append(loss.detach())
        if update_mask is not None:
            # A masked row gets zero error: its fast entries only forget, as a padding step does.
            model.fast_only_dfa_step(output_error * update_mask[:, i, None], config['learning_rate'],
                                     config['ephemeral_update_clamp'], config.get('grad_norm_clip', 0))
    return torch.stack(preds), torch.stack(losses)


def first_recall_steps(texts, dataset, steps):
    """The step predicting each episode's first recall target, [B] (steps if it has none)."""
    firsts = [min(recall_targets(text, dataset)[0], default=steps + 1) - 1 for text in texts]
    return torch.tensor([min(first, steps) for first in firsts])


def update_mask(protocol, texts, dataset, onehot):
    batch, steps = onehot.shape[0], onehot.shape[1] - 1
    if protocol == "no_fast":
        return None
    mask = torch.ones(batch, steps, dtype=torch.bool)
    if protocol == "strict":
        mask &= torch.arange(steps) < first_recall_steps(texts, dataset, steps)[:, None]
    elif protocol != "observed":
        raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")
    return mask.to(onehot.device)


def evaluate_protocols(model, batches, config, dataset, protocols=PROTOCOLS, prefix="heldout"):
    """Every protocol over batches [(texts, onehot)], as {f'{prefix}_{protocol}/{metric}': value}.
    The model's whole state is restored afterwards, so a training run continues unchanged."""
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    was_training = model.training
    model.eval()
    results = {}
    try:
        for protocol in protocols:
            interval = IntervalMetrics(dataset)
            first_hits = first_count = 0
            for texts, onehot in batches:
                preds, losses = evaluate_held_out(model, onehot, update_mask(protocol, texts, dataset, onehot), config)
                interval.update(texts, onehot, preds, losses)
                steps = preds.shape[0]
                first = first_recall_steps(texts, dataset, steps).to(preds.device)
                has = first < steps
                rows = torch.arange(len(texts), device=preds.device)[has]
                hit = preds[first[has], rows] == onehot[rows, first[has] + 1].argmax(-1)
                first_hits, first_count = first_hits + int(hit.sum()), first_count + int(has.sum())
            summary = interval.summary()
            if first_count:
                summary["first_answer_acc"] = first_hits / first_count
            results.update({f"{prefix}_{protocol}/{key}": value for key, value in summary.items()})
    finally:
        model.load_state_dict(saved)
        model.train(was_training)
    return results


def load_heldout_batches(dataset, batch_size, n_batches, device, split="validation"):
    """The first n_batches full batches (0 = all) of a synthetic dataset's held-out split, in
    stored order, as [(texts, onehot)]. Uses no random numbers, so a training run's RNG stream
    is unchanged."""
    if not is_synthetic(dataset):
        raise ValueError(f"held-out evaluation reads synth_datasets/<name>/{split}; {dataset} is not synthetic")
    rng = capture_rng_state()  # in case the datasets library draws any
    rows = preprocess_rows(load_from_disk(f"synth_datasets/{dataset}")[split], dataset)
    restore_rng_state(rng)
    count = len(rows) // batch_size if n_batches <= 0 else min(n_batches, len(rows) // batch_size)
    if count == 0:
        raise ValueError(f"{dataset} {split} has fewer than {batch_size} rows")
    collate = OneHotCollate(initialize_charset(dataset)[3])
    batches = []
    for b in range(count):
        texts, _, onehot = collate([rows[i] for i in range(b * batch_size, (b + 1) * batch_size)])
        batches.append((texts, onehot.to(device)))
    return batches


def main(argv=None):
    parser = argparse.ArgumentParser(description="Held-out, frozen-slow-weight evaluation of a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None, help="default: the checkpoint's")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    parser.add_argument("--batches", type=int, default=0, help="batches to evaluate (0 = the whole split)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", default=None, help="also write the results here")
    args = parser.parse_args(argv)

    from train import build_model, build_parser, positional_encoding  # train.py imports this module
    checkpoint = read_checkpoint(args.checkpoint)
    # A flag newer than the checkpoint takes its default, as it did when the run was trained.
    defaults = {key: value for key, value in vars(build_parser().parse_args([])).items()
                if not key.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(checkpoint.get("config", {}))}
    charset, _, _, n_characters = initialize_charset(config["dataset"])
    model = build_model(config, charset, n_characters)
    model, _, next_iter, _, _ = load_checkpoint(args.checkpoint, model, config, device=args.device,
                                                checkpoint=checkpoint)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], args.device)
    dataset = args.dataset or config["dataset"]
    batches = load_heldout_batches(dataset, config["batch_size"], args.batches, args.device, args.split)
    results = evaluate_protocols(model, batches, config, dataset, args.protocols)
    header = {"checkpoint": args.checkpoint, "iteration": next_iter - 1, "dataset": dataset,
              "split": args.split, "episodes": len(batches) * config["batch_size"]}
    print(json.dumps(header))
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({**header, **results}, handle, indent=2)


if __name__ == "__main__":
    main()
