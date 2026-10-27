"""S12 follow-up 2 (2026-10-27): a valid no_fast baseline for --fast_read_norm ln.

Under ln a layer reads S x + LN(F x) + b, and LN(F x) has norm ~sqrt(out) whenever anything was written.
heldout.py's no_fast replaces it with 0 at every step (in training only an empty store reads 0), so its recall
(0.00 on key recall, below chance) mixes "no memory" with "miscalibrated". This script keeps a
fast read of the trained size and removes only its content. Null protocols (every trunk layer with
ln fast entries, or --layers):

  swap      LN(F_d x) rescaled to |LN(F x)|: the read from a *different* sequence's fast state F_d
            (a donor episode run in lockstep under --donor_protocol, default strict, so F_d holds a
            real held-out episode's writes at the same step), with the own read's norm per row/step
            (0 where the own read is 0).
  swap_raw  LN(F_d x) as it comes (norm ~sqrt(out) wherever the donor read is nonzero, even where
            the own read is 0).
  noise     norm-matched Gaussian noise: e ~ N(0, I), centred over the outputs (LN's output has
            mean 0), rescaled to |LN(F x)| per row/step; fresh each step, seeded by --seed.

The own fast state F is written as under --own_protocol (default strict: writes stop at the step
that predicts the first recall target, as heldout.py's strict) with the errors of the null
forward; only its read's norm is used. Slow entries, biases and i2o are frozen as in heldout.py.
The donor of batch k is batch k+1 (row for row; with one batch, the same batch rolled by one row),
padded with zero (padding) steps or cut to the target's length.

In the same call it runs heldout.py's own protocols (default strict and no_fast) on the same
batches, so strict - swap / strict - noise is the memory and swap/noise - no_fast is the
calibration share of no_fast's gap. Output: heldout.py's JSON layout (heldout_<protocol>/<metric>)
plus per-layer read-norm diagnostics (null_<protocol>/L<i>_...).

Run from a code directory of main >= b26e88d (it imports train/heldout from the cwd and reads
synth_datasets/ there), as bfr_ratio.py:
    cd <code>; python <this dir>/ln_null_eval.py --checkpoint RUN/checkpoint_00030000.pth --batches 128 --json out.json
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

from ephemeral_model import dfa_output_error, fast_read_layer_norm  # noqa: E402
from heldout import evaluate_protocols, first_recall_steps, load_heldout_batches, update_mask  # noqa: E402
from metrics import IntervalMetrics  # noqa: E402
from utils import model_input  # noqa: E402

NULLS = ("swap", "swap_raw", "noise")


def ln_layers(model, layers=None):
    """[(name, layer)] of the layers whose fast read is ln: trunk layers by index (layers=None: every
    trunk layer with ln, and i2h if it has ln)."""
    out = [(f"L{i}", l) for i, l in enumerate(model.linear_layers)
           if l.fast_read_norm == 'ln' and (layers is None or i in layers)]
    if layers is None and model.i2h.fast_read_norm == 'ln':
        out.append(("i2h", model.i2h))
    return out


class NullRead:
    """Replaces one ln layer's fast read LN(F x) (patched in as the layer's _normalized_read)."""

    def __init__(self, layer, kind, donor_layer=None, generator=None):
        self.layer, self.kind, self.donor_layer, self.generator = layer, kind, donor_layer, generator
        # [reads with own > 0, sum |own|, sum |null| there, null > 0 where own = 0, null = 0 where own > 0]
        self.stats = None

    def __call__(self, input, input_unsq):
        layer = self.layer
        weights, mask = layer.per_sample_weights, layer.ephemeral_mask.unsqueeze(0)
        slow = torch.bmm(weights.masked_fill(mask, 0), input_unsq).squeeze(2)
        own = fast_read_layer_norm(torch.bmm(weights.masked_fill(~mask, 0), input_unsq).squeeze(2))
        own_norm = own.norm(dim=1, keepdim=True)
        if self.kind in ("swap", "swap_raw"):
            donor = self.donor_layer.per_sample_weights.masked_fill(~mask, 0)
            rep = fast_read_layer_norm(torch.bmm(donor, input_unsq).squeeze(2))
            if self.kind == "swap":
                rep = rep * (own_norm / rep.norm(dim=1, keepdim=True).clamp_min(1e-30))
        elif self.kind == "noise":
            noise = torch.randn(own.shape, generator=self.generator, device=own.device, dtype=own.dtype)
            noise = noise - noise.mean(dim=1, keepdim=True)
            rep = noise * (own_norm / noise.norm(dim=1, keepdim=True).clamp_min(1e-30))
        else:
            raise ValueError(f"unknown null {self.kind!r}; choose from {NULLS}")
        rep_norm = rep.norm(dim=1, keepdim=True)
        on = own_norm[:, 0] > 0
        step = torch.stack([on.sum(), (own_norm[:, 0] * on).sum(), (rep_norm[:, 0] * on).sum(),
                            ((rep_norm[:, 0] > 0) & ~on).sum(), ((rep_norm[:, 0] == 0) & on).sum()]).double()
        self.stats = step if self.stats is None else self.stats + step  # stays on the device: no sync per step
        return slow + rep


def donor_batch(batches, k, self_donor=False):
    """The donor (texts, onehot) of batch k: batch k+1, or the same batch rolled by one row.
    self_donor (tests only): batch k itself, row for row, so swap_raw must reproduce own_protocol."""
    if self_donor:
        return batches[k]
    if len(batches) > 1:
        return batches[(k + 1) % len(batches)]
    texts, onehot = batches[k]
    return texts[1:] + texts[:1], torch.roll(onehot, -1, dims=0)


def fit_steps(onehot, mask, steps):
    """A donor onehot [B, T', V] and mask [B, T'-1] cut or zero-padded to steps + 1 / steps."""
    have = onehot.shape[1] - 1
    if have >= steps:
        return onehot[:, :steps + 1], mask[:, :steps]
    pad = onehot.new_zeros(onehot.shape[0], steps - have, onehot.shape[2])
    return torch.cat([onehot, pad], 1), torch.cat([mask, mask.new_zeros(mask.shape[0], steps - have)], 1)


@torch.no_grad()
def evaluate_null(model, batches, config, dataset, kind, own_protocol="strict", donor_protocol="strict",
                  seed=0, layers=None, self_donor=False):
    """One null protocol over batches [(texts, onehot)]: (summary dict as evaluate_protocols's, per-layer
    stats). The model's state is restored afterwards."""
    if model.fast_iterations != 1:
        raise ValueError("ln_null_eval assumes one fast write per forward (--fast_backward_per_forward 1)")
    targets = ln_layers(model, layers)
    if not targets:
        raise ValueError("no layer with --fast_read_norm ln: this baseline is defined for ln reads only")
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    was_training = model.training
    model.eval()
    donor = copy.deepcopy(model) if kind != "noise" else None
    device = batches[0][1].device
    generator = torch.Generator(device=device).manual_seed(seed) if kind == "noise" else None
    donor_layers = dict(ln_layers(donor, layers)) if donor is not None else {}
    reads = {name: NullRead(layer, kind, donor_layers.get(name), generator) for name, layer in targets}
    for name, layer in targets:
        layer._normalized_read = reads[name]
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    lr, clamp, clip = config['learning_rate'], config['ephemeral_update_clamp'], config.get('grad_norm_clip', 0)
    interval = IntervalMetrics(dataset)
    first_hits = first_count = 0
    try:
        for k, (texts, onehot) in enumerate(batches):
            steps = onehot.shape[1] - 1
            own_mask = update_mask(own_protocol, texts, dataset, onehot)
            model.start_sequence_wipe()
            hidden = model.initHidden(onehot.shape[0])
            if donor is not None:
                d_texts, d_onehot = donor_batch(batches, k, self_donor)
                d_mask = update_mask(donor_protocol, d_texts, dataset, d_onehot)
                d_onehot, d_mask = fit_steps(d_onehot, d_mask, steps)
                donor.start_sequence_wipe()
                d_hidden = donor.initHidden(onehot.shape[0])
            preds, losses = [], []
            for i in range(steps):
                if donor is not None:  # the donor reads F_d(t) first; the target then reads the same F_d(t)
                    d_out, d_hidden = donor(model_input(d_onehot, i, config['input_mode'], config['pe_matrix']), d_hidden)
                output, hidden = model(model_input(onehot, i, config['input_mode'], config['pe_matrix']), hidden)
                loss, error = dfa_output_error(output, onehot[:, i + 1], criterion)
                preds.append(output.argmax(dim=1))
                losses.append(loss.detach())
                if own_mask is not None:
                    model.fast_only_dfa_step(error * own_mask[:, i, None], lr, clamp, clip)
                if donor is not None:
                    _, d_error = dfa_output_error(d_out, d_onehot[:, i + 1], criterion)
                    donor.fast_only_dfa_step(d_error * d_mask[:, i, None], lr, clamp, clip)
            preds, losses = torch.stack(preds), torch.stack(losses)
            interval.update(texts, onehot, preds, losses)
            first = first_recall_steps(texts, dataset, steps).to(preds.device)
            has = first < steps
            rows = torch.arange(len(texts), device=preds.device)[has]
            hit = preds[first[has], rows] == onehot[rows, first[has] + 1].argmax(-1)
            first_hits, first_count = first_hits + int(hit.sum()), first_count + int(has.sum())
    finally:
        for _, layer in targets:
            del layer._normalized_read
        model.load_state_dict(saved)
        model.train(was_training)
    summary = interval.summary()
    if first_count:
        summary["first_answer_acc"] = first_hits / first_count
    stats = {}
    for name, read in reads.items():
        n, own_sum, rep_sum, rep_on_off, rep_off_on = read.stats.tolist() if read.stats is not None else [0.0] * 5
        stats.update({f"{name}_own_norm_mean": own_sum / max(n, 1), f"{name}_null_norm_mean": rep_sum / max(n, 1),
                      f"{name}_own_nonzero_reads": n, f"{name}_null_nonzero_where_own_zero": rep_on_off,
                      f"{name}_null_zero_where_own_nonzero": rep_off_on})
    return summary, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batches", type=int, default=128, help="held-out batches (128 = the batches' heldout.json used)")
    ap.add_argument("--nulls", nargs="+", default=list(NULLS), choices=NULLS)
    ap.add_argument("--reference", nargs="*", default=["strict", "no_fast"],
                    help="heldout.py protocols to run on the same batches (default strict no_fast)")
    ap.add_argument("--own_protocol", default="strict", choices=["strict", "observed"])
    ap.add_argument("--donor_protocol", default="strict", choices=["strict", "observed"])
    ap.add_argument("--layers", default="all", help="'all', or a comma list of trunk layers to null (others keep their read)")
    ap.add_argument("--seed", type=int, default=0, help="noise seed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    from train import build_model, build_parser, positional_encoding
    from utils import initialize_charset, load_checkpoint, read_checkpoint, upgrade_legacy_config
    checkpoint = read_checkpoint(args.checkpoint)
    defaults = {k: v for k, v in vars(build_parser().parse_args([])).items() if not k.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(checkpoint.get("config", {}))}
    charset, _, _, n_characters = initialize_charset(config["dataset"])
    model = build_model(config, charset, n_characters)
    model, _, next_iter, _, _ = load_checkpoint(args.checkpoint, model, config, device=args.device, checkpoint=checkpoint)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], args.device)
    dataset = config["dataset"]
    batches = load_heldout_batches(dataset, config["batch_size"], args.batches, args.device)
    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    out = {"checkpoint": args.checkpoint, "iteration": next_iter - 1, "dataset": dataset,
           "fast_read_norm": config.get("fast_read_norm", "off"),
           "decouple_fast_feedback": config.get("decouple_fast_feedback", ""),
           "episodes": len(batches) * config["batch_size"], "own_protocol": args.own_protocol,
           "donor_protocol": args.donor_protocol, "layers": args.layers, "seed": args.seed}
    if args.reference:
        out.update(evaluate_protocols(model, batches, config, dataset, args.reference))
    for kind in args.nulls:
        summary, stats = evaluate_null(model, batches, config, dataset, kind, args.own_protocol,
                                       args.donor_protocol, args.seed, layers)
        out.update({f"heldout_{kind}/{k}": v for k, v in summary.items()})
        out.update({f"null_{kind}/{k}": v for k, v in stats.items()})
    print(json.dumps({k: out[k] for k in ("checkpoint", "iteration", "episodes")}))
    for key, value in out.items():
        if key.startswith("heldout_") and key.endswith(("/recall_acc", "/loss", "/recall_loss", "/other_loss")):
            print(f"  {key}: {value:.4f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
