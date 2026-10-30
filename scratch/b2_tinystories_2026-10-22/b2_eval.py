"""B2 (2026-10-22): held-out measurements of a TinyStories checkpoint, or key PR of a synthetic-task checkpoint.

Pre-registered in PREREG_fastlayers.md, "Batch B2". Run from a memory_encoding checkout (it imports train.py).

    python b2_eval.py tiny  --checkpoint C --names names.json --stories 4096 --out OUT   (measures a-d)
    python b2_eval.py curve --checkpoint C --stories 256 --out OUT                        (held-out CE only)
    python b2_eval.py synth --checkpoint C --batches 64 --out OUT                         (key PR on its own task)
    python b2_eval.py horizon --checkpoint C --stories 1024 --wipe_every 4,8,16,32,64,128 --out OUT   (round 2)

Inference-only forgetting intervention (all scoring modes): --forget_rate_override 0.03 (or 0.05, 0.1).
Omit it to retain checkpoint forgetting (e.g. DEEP's 0.01). Slow weights and checkpoint files stay unchanged;
JSON inference_forgetting records checkpoint/effective rates and every layer's actual rates.

Every protocol freezes the slow weights and starts each sequence from wiped fast entries (heldout.py):
  ON     every step writes the fast entries with the true next character (heldout.py "observed").
  OFF    no fast weights (they stay at their wiped zeros); with recurrence off this is a character bigram model.
  RESET  for a mention starting at s: the fast entries are zeroed at s - 32 (R), then written as in ON. With
         recurrence off and frozen slow weights nothing else carries over, so this equals running the snippet
         text[s-32 : end of mention] from a fresh wipe. With recurrence on, replay the full story prefix and
         wipe only fast entries at s - 32, preserving the hidden state (including earlier fast-write effects).
  WIPE R (round 2, horizon mode) ON, but the fast entries are zeroed before every step i > 0 with i % R == 0, so a
         prediction sees at most R characters of fast-written context (R - 1 - (i % R) fewer on average). Only the
         fast entries are zeroed; slow entries and hidden state stay as they are. R = 1 == OFF is asserted only
         with recurrence off. Hidden state can carry context beyond a fast wipe, so R does not bound total
         recurrent context. R >= the length equals ON bit for bit (tests/test_b2_horizon.py).

SimpleRNN/BPTT checkpoints have no fast updates: ON, OFF and RESET all retain ordinary hidden recurrence,
but reset hidden state at each story/pass. Fast-key PR is unavailable (reported as empty per-protocol objects
in tiny mode, or an empty pr object in synth mode).

Keys: the input x_t of each trunk layer (layer.in_traces after the forward), what its fast entries are written
with and read by. PR = (sum l)^2 / sum l^2 of the second moment of unit-norm keys (uncentered: the 09-30
real_model definition), and of the centered second moment.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())  # run from the memory_encoding checkout
R = 32
WORD = re.compile(r"[A-Za-z]+")
WINDOWS = (9, 64, 511)


# ----------------------------------------------------------------------------- model
def load_model(path, device):
    from train import build_model, build_parser, positional_encoding
    from utils import initialize_charset, load_checkpoint, read_checkpoint, upgrade_legacy_config
    checkpoint = read_checkpoint(path)
    defaults = {k: v for k, v in vars(build_parser().parse_args([])).items() if not k.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(checkpoint.get("config", {}))}
    charset, _, _, n_characters = initialize_charset(config["dataset"])
    model = build_model(config, charset, n_characters)
    model, _, next_iter, _, _ = load_checkpoint(path, model, config, device=device, checkpoint=checkpoint)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], device)
    model.eval()
    return model, config, charset, next_iter - 1


def has_fast_weights(model):
    """SimpleRNN checkpoints have neither fast state nor a fast-only updater."""
    return callable(getattr(model, "fast_only_dfa_step", None))


def forget_rate_value(value):
    rate = float(value)
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise argparse.ArgumentTypeError("inference forget rate must be finite and in [0, 1]")
    return rate


def inference_forgetting(model, config, override=None):
    """Change actual updater attributes, not just config (also works on unfused CPU/sm61).

    The in-memory config reports the effective rate; metadata preserves the training recipe.
    With no override this is strictly read-only.
    Include i2h/i2o and any auxiliary fast layers, not only the trunk; the root attribute is also
    kept consistent even though fast_only_dfa_step reads each layer's forget_rate.
    """
    checkpoint_rate = config.get("forget_rate")
    if override is not None:
        override = forget_rate_value(override)
        if not has_fast_weights(model):
            raise ValueError("--forget_rate_override requires a model with fast weights")
    modules = {name or "model": module for name, module in model.named_modules()
               if hasattr(module, "forget_rate")}
    original = {name: module.forget_rate for name, module in modules.items()}
    if override is not None:
        for module in modules.values():
            module.forget_rate = override
        config["forget_rate"] = override
    return {"checkpoint_forget_rate": checkpoint_rate, "override": override,
            "effective_forget_rate": override if override is not None else checkpoint_rate,
            "original_module_rates": original,
            "effective_module_rates": {name: module.forget_rate for name, module in modules.items()}}


def wipe_fast(model):
    """Zero every fast entry (trunk, i2h, i2o) and leave the slow entries alone: the wipe_fast part of
    EphemeralLinear.start_sequence_wipe without its batch mean of the slow entries. getattr because the B2 runtime
    copies (2bc949a) predate --fast_structure, the read-norm key mass and --order2_layers."""
    if not has_fast_weights(model):
        return
    for layer in (*model.linear_layers, model.i2h, model.i2o):
        if getattr(layer, "is_additive", False):
            layer.fast_state.data.zero_()
            continue
        layer.weight.data.masked_fill_(layer.ephemeral_mask, 0)
        layer.per_sample_weights.data.masked_fill_(layer.ephemeral_mask.unsqueeze(0), 0)
        if getattr(layer, "fast_key_mass", None) is not None:
            layer.fast_key_mass.zero_()
        if hasattr(layer, "fast_keys"):
            layer.fast_keys = layer.fast_key_weights = None
        if getattr(layer, "order2_fast", None) is not None:
            layer.order2_fast.zero_()


@torch.no_grad()
def run_pass(model, onehot, config, write, keys=False, wipe_every=0, wipe_at=None):
    """One batch [B, T, V] from a fresh wipe. Returns per-step losses [B, T-1] (target t+1 predicted at step t)
    and, if keys, a list per trunk layer of [T-1, B, D] inputs. wipe_every R > 0 zeroes the fast entries before
    every step i > 0 with i % R == 0 (the WIPE R protocol). wipe_at is a single fast-only RESET boundary;
    neither wipe resets hidden. SimpleRNN never writes fast entries and returns [] for requested fast keys."""
    from ephemeral_model import dfa_output_error
    from utils import model_input
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    fast = has_fast_weights(model)
    if fast:
        model.start_sequence_wipe()
    hidden = model.initHidden(onehot.shape[0])
    steps = onehot.shape[1] - 1
    losses = []
    collected = [[] for _ in model.linear_layers] if keys and fast else []
    for i in range(steps):
        if (wipe_every and i and i % wipe_every == 0) or i == wipe_at:
            wipe_fast(model)
        output, hidden = model(model_input(onehot, i, config["input_mode"], config["pe_matrix"]), hidden)
        if keys:
            for store, layer in zip(collected, model.linear_layers):
                store.append(layer.in_traces.data.clone())
        loss, error = dfa_output_error(output, onehot[:, i + 1], criterion)
        losses.append(loss.detach())
        if write and fast:
            model.fast_only_dfa_step(error, config["learning_rate"], config["ephemeral_update_clamp"],
                                     config.get("grad_norm_clip", 0))
    out = torch.stack(losses, 1)
    if keys:
        return out, [torch.stack(store) for store in collected]
    return out


def onehot_batch(texts, c2i, n_characters, length, batch_size, device):
    """Texts -> [batch_size, length, V] one-hot (all-zero rows pad short texts and missing rows), valid [B, T]."""
    hot = torch.zeros(batch_size, length, n_characters, device=device)
    valid = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
    for row, text in enumerate(texts):
        idx = torch.tensor([c2i[c] for c in text[:length]], device=device)
        hot[row, torch.arange(len(idx), device=device), idx] = 1
        valid[row, :len(idx)] = True
    return hot, valid


# ----------------------------------------------------------------------------- PR
def pr_of(eigs):
    eigs = eigs.clamp_min(0).double()
    return float(eigs.sum() ** 2 / (eigs ** 2).sum().clamp_min(1e-300))


class CorpusPR:
    def __init__(self, dim, device):
        self.m = torch.zeros(dim, dim, dtype=torch.float64, device=device)
        self.s = torch.zeros(dim, dtype=torch.float64, device=device)
        self.n = 0

    def add(self, unit):  # [N, D]
        x = unit.double()
        self.m += x.T @ x
        self.s += x.sum(0)
        self.n += x.shape[0]

    def result(self):
        second = self.m / self.n
        mean = self.s / self.n
        eu = torch.linalg.eigvalsh(second)
        ec = torch.linalg.eigvalsh(second - torch.outer(mean, mean))
        top = eu.flip(0)
        return {"pr_uncentered": pr_of(eu), "pr_centered": pr_of(ec), "n_keys": self.n,
                "top1_share": float(top[0] / top.sum()), "mean_dir_norm": float(mean.norm())}


def window_prs(unit, valid_steps, w):
    """unit [T, B, D] unit-norm keys; per (story, non-overlapping window fully inside the valid steps): uncentered
    and centered PR. Returns two lists."""
    unc, cen = [], []
    steps = unit.shape[0]
    for b in range(unit.shape[1]):
        n = int(valid_steps[b])
        count = n // w
        if count == 0:
            continue
        x = unit[:count * w, b].reshape(count, w, -1).double()  # [count, w, D]
        g = x @ x.transpose(1, 2)
        e = torch.linalg.eigvalsh(g).clamp_min(0)
        unc += (e.sum(1) ** 2 / (e ** 2).sum(1)).tolist()
        xc = x - x.mean(1, keepdim=True)
        gc = xc @ xc.transpose(1, 2)
        ec = torch.linalg.eigvalsh(gc).clamp_min(0)
        cen += (ec.sum(1) ** 2 / (ec ** 2).sum(1).clamp_min(1e-300)).tolist()
    return unc, cen


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------------------------------------------------------- text analysis
def position_classes(text):
    """Per character position j of text: 0 non-letter, 1 letter of a word's first occurrence in the text,
    2 letter of a repeated word (an earlier complete occurrence exists)."""
    cls = np.zeros(len(text), dtype=np.int8)
    seen = set()
    for match in WORD.finditer(text):
        word = match.group()
        cls[match.start():match.end()] = 2 if word in seen else 1
        seen.add(word)
    return cls


def mention_pairs(text, names):
    """(kind, word, (s1, e1), (s2, e2)) for each name / control word whose first two whole-word occurrences
    satisfy the pre-registered spacing (s1 >= R, s2 - R >= e1)."""
    occ = {}
    for match in WORD.finditer(text):
        occ.setdefault(match.group(), []).append((match.start(), match.end()))
    out = []
    for word, spans in occ.items():
        if len(spans) < 2:
            continue
        if word in names:
            kind = "name"
        elif word.islower() and len(word) >= 4:
            kind = "control"
        else:
            continue
        (s1, e1), (s2, e2) = spans[0], spans[1]
        if s1 >= R and s2 - R >= e1:
            out.append((kind, word, (s1, e1), (s2, e2)))
    return out


def mention_surprisal(losses_row, start, end, offset=0):
    """losses_row[k] is the loss of predicting character k + 1 + offset. Returns (first char, continuation) nats."""
    first = float(losses_row[start - 1 - offset])
    cont = float(losses_row[start - offset:end - 1 - offset].sum())
    return first, cont


# ----------------------------------------------------------------------------- modes
def load_validation(n, length, path=None):
    """The first n validation stories, charset-filtered, cropped. path: a JSON list of filtered stories saved by
    `b2_eval.py dumpval` (compute nodes have no network)."""
    if path:
        texts = json.load(open(path))[:n]
        return [t[:length] for t in texts]
    from datasets import load_dataset
    from utils import filter_text
    raw = load_dataset("roneneldan/tinystories", split="validation")
    raw = raw.select(range(min(n, len(raw))))
    texts = filter_text({"text": raw["text"]}, "roneneldan/tinystories")["text"]
    return [t[:length] for t in texts]


def tiny(args):
    device = args.device
    model, config, charset, iteration = load_model(args.checkpoint, device)
    forgetting = inference_forgetting(model, config, getattr(args, "inference_forget_rate", None))
    c2i = {c: i for i, c in enumerate(charset)}
    batch = config["batch_size"]
    length = args.length
    texts = load_validation(args.stories, length, args.val_file)
    names = set(json.load(open(args.names))["names"]) if args.names else set()
    n_layers = len(model.linear_layers) if has_fast_weights(model) else 0
    dim = model.linear_layers[0].in_features
    want_keys = args.mode == "tiny"
    corpus = {p: [CorpusPR(dim, device) for _ in range(n_layers)] for p in ("on", "off")} if want_keys else None
    wins = {p: {w: {l: ([], []) for l in range(n_layers)} for w in WINDOWS} for p in ("on", "off")}
    loss_on, loss_off, classes, lengths = [], [], [], []
    pairs = []  # dicts
    t0 = time.time()
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        hot, valid = onehot_batch(chunk, c2i, len(charset), length, batch, device)
        valid_steps = valid[:, 1:].sum(1)  # steps with a real target
        for protocol in ("on", "off"):
            if want_keys:
                losses, keys = run_pass(model, hot, config, write=(protocol == "on"), keys=True)
                for l, k in enumerate(keys):
                    unit = k / k.norm(dim=2, keepdim=True).clamp_min(1e-12)  # [T-1, B, D]
                    mask = valid[:, 1:].T  # [T-1, B]
                    corpus[protocol][l].add(unit[mask])
                    for w in WINDOWS:
                        u, c = window_prs(unit, valid_steps, w)
                        wins[protocol][w][l][0].extend(u)
                        wins[protocol][w][l][1].extend(c)
            else:
                losses = run_pass(model, hot, config, write=(protocol == "on"))
            losses = (losses * valid[:, 1:]).cpu().numpy()
            (loss_on if protocol == "on" else loss_off).append(losses[:len(chunk)])
        for row, text in enumerate(chunk):
            classes.append(position_classes(text))
            lengths.append(len(text))
            if names:
                for kind, word, m1, m2 in mention_pairs(text, names):
                    rec = {"story": start + row, "kind": kind, "word": word, "m1": m1, "m2": m2}
                    for tag, lo in (("on", loss_on[-1][row]), ("off", loss_off[-1][row])):
                        rec[f"{tag}1"] = mention_surprisal(lo, *m1)
                        rec[f"{tag}2"] = mention_surprisal(lo, *m2)
                    pairs.append(rec)
        if start // batch % 20 == 0:
            print(f"stories {start + len(chunk)}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    loss_on = np.concatenate(loss_on)
    loss_off = np.concatenate(loss_off)

    # RESET snippets are equivalent only without recurrence. Recurrent RESET replays the full prefix,
    # batching mentions with the same reset boundary so wipe_fast never touches another row's state.
    if pairs:
        jobs = [(i, which) for i, rec in enumerate(pairs) for which in (1, 2)]
        if getattr(model, "enable_recurrence", False):
            reset_mentions(model, config, texts, pairs, jobs, c2i, len(charset), batch, device)
            print(f"reset full prefixes {len(jobs)} {time.time() - t0:.0f}s", flush=True)
            jobs = []
        for js in range(0, len(jobs), batch):
            part = jobs[js:js + batch]
            snippets, spans = [], []
            for i, which in part:
                rec = pairs[i]
                s, e = rec[f"m{which}"]
                snippets.append(texts[rec["story"]][s - R:e])
                spans.append((s, e))
            hot, valid = onehot_batch(snippets, c2i, len(charset), max(len(x) for x in snippets), batch, device)
            losses = (run_pass(model, hot, config, write=True) * valid[:, 1:]).cpu().numpy()
            for row, ((i, which), (s, e)) in enumerate(zip(part, spans)):
                pairs[i][f"reset{which}"] = mention_surprisal(losses[row], s, e, offset=s - R)
        if jobs:
            print(f"reset snippets {len(jobs)} {time.time() - t0:.0f}s", flush=True)

    # ---- summaries
    cls = np.zeros_like(loss_on, dtype=np.int8) - 1
    for b, c in enumerate(classes):
        cls[b, :len(c) - 1] = c[1:]  # class of the predicted character
    res = {"checkpoint": args.checkpoint, "iteration": iteration, "stories": len(texts), "length": length,
           "arm_config": {k: config.get(k) for k in ("learning_rate", "plasticity", "fast_layers", "layer_norm",
                                                       "fast_read_norm", "decouple_fast_feedback", "seed",
                                                       "max_seq_len", "ephemeral_fraction", "forget_rate",
                                                       "model_type", "updater", "enable_recurrence",
                                                       "train_wipe_chars", "optimizer", "grad_norm_clip",
                                                       "residual_connection")}}
    ce = {}
    for name, sel in (("all", cls >= 0), ("nonletter", cls == 0), ("novel_word_letter", cls == 1),
                      ("repeat_word_letter", cls == 2)):
        n = int(sel.sum())
        ce[name] = {"n": n, "on": float(loss_on[sel].sum() / n), "off": float(loss_off[sel].sum() / n)}
        ce[name]["on_bits"] = ce[name]["on"] / math.log(2)
        ce[name]["off_bits"] = ce[name]["off"] / math.log(2)
    res["ce"] = ce
    res["inference_forgetting"] = forgetting
    if want_keys:
        res["pr"] = {}
        for p in ("on", "off"):
            res["pr"][p] = {}
            for l in range(n_layers):
                entry = corpus[p][l].result()
                for w in WINDOWS:
                    u, c = wins[p][w][l]
                    entry[f"win{w}_pr_uncentered"] = float(np.mean(u)) if u else float("nan")
                    entry[f"win{w}_pr_centered"] = float(np.mean(c)) if c else float("nan")
                    entry[f"win{w}_n"] = len(u)
                    if l > 0:
                        entry[f"win{w}_spearman_vs_layer0"] = spearman(u, wins[p][w][0][0])
                res["pr"][p][f"layer{l}"] = entry
    if pairs:
        res["remention"] = summarize_pairs(pairs)
    with open(args.out + ".json", "w") as handle:
        json.dump(res, handle, indent=1)
    if args.mode == "tiny":
        np.savez_compressed(args.out + ".npz", loss_on=loss_on.astype(np.float16), loss_off=loss_off.astype(np.float16),
                            cls=cls)
        with open(args.out + "_pairs.jsonl", "w") as handle:
            for rec in pairs:
                handle.write(json.dumps(rec) + "\n")
    print(json.dumps(res["ce"]["all"]), f"{time.time() - t0:.0f}s")


def reset_mentions(model, config, texts, pairs, jobs, c2i, n_characters, batch, device):
    """Score recurrent RESET mentions from story-start hidden state, not snippet-start hidden state.

    Stop after each mention (later characters cannot affect its causal predictions). Grouping by boundary
    allows the existing all-row fast wipe, including every supported fast-state representation, unchanged.
    """
    groups = {}
    for i, which in jobs:
        s, _ = pairs[i][f"m{which}"]
        groups.setdefault(s - R, []).append((i, which))
    for boundary, group in groups.items():
        for js in range(0, len(group), batch):
            part = group[js:js + batch]
            prefixes = [texts[pairs[i]["story"]][:pairs[i][f"m{which}"][1]] for i, which in part]
            hot, valid = onehot_batch(prefixes, c2i, n_characters, max(map(len, prefixes)), batch, device)
            losses = (run_pass(model, hot, config, write=True, wipe_at=boundary) * valid[:, 1:]).cpu().numpy()
            for row, (i, which) in enumerate(part):
                pairs[i][f"reset{which}"] = mention_surprisal(losses[row], *pairs[i][f"m{which}"])


def summarize_pairs(pairs):
    out = {}
    for kind in ("name", "control"):
        recs = [r for r in pairs if r["kind"] == kind and "reset1" in r]
        if not recs:
            continue
        arr = lambda key, part: np.array([r[key][part] for r in recs])
        chars = np.array([r["m1"][1] - r["m1"][0] - 1 for r in recs])  # continuation characters per mention
        summary = {"pairs": len(recs), "mean_cont_chars": float(chars.mean())}
        for part, label in ((0, "first"), (1, "cont")):
            for key in ("on1", "on2", "off1", "off2", "reset1", "reset2"):
                summary[f"{label}_{key}"] = float(arr(key, part).mean())
            m = (arr("reset2", part) - arr("on2", part)) - (arr("reset1", part) - arr("on1", part))
            drop = arr("on1", part) - arr("on2", part)
            summary[f"{label}_M"] = float(m.mean())
            summary[f"{label}_M_sem"] = float(m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 1 else float("nan")
            summary[f"{label}_armeni_drop"] = float(drop.mean())
            summary[f"{label}_off_drop"] = float((arr("off1", part) - arr("off2", part)).mean())
            summary[f"{label}_reset2_minus_on2"] = float((arr("reset2", part) - arr("on2", part)).mean())
        summary["cont_M_per_char"] = float(((arr("reset2", 1) - arr("on2", 1)) - (arr("reset1", 1) - arr("on1", 1))).sum()
                                           / chars.sum())
        out[kind] = summary
    return out


CLASSES = (("all", lambda c: c >= 0), ("nonletter", lambda c: c == 0), ("novel_word_letter", lambda c: c == 1),
           ("repeat_word_letter", lambda c: c == 2))


def horizon(args):
    """Round 2 (PREREG_fastlayers.md, "Batch B2 round 2"): held-out CE under ON, OFF and WIPE R for each R, by
    position class. Without recurrence, the first batch also checks WIPE 1 == OFF bit for bit.
    With recurrence, report a fast-only intervention without imposing the OFF equivalence."""
    device = args.device
    model, config, charset, iteration = load_model(args.checkpoint, device)
    forgetting = inference_forgetting(model, config, getattr(args, "inference_forget_rate", None))
    c2i = {c: i for i, c in enumerate(charset)}
    batch = config["batch_size"]
    texts = load_validation(args.stories, args.length, args.val_file)
    periods = [int(x) for x in args.wipe_every.split(",")]
    protocols = ["on", "off"] + [f"wipe{r}" for r in periods]
    sums = {p: {name: 0.0 for name, _ in CLASSES} for p in protocols}
    counts = {name: 0 for name, _ in CLASSES}
    t0 = time.time()
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        hot, valid = onehot_batch(chunk, c2i, len(charset), args.length, batch, device)
        cls = np.full((batch, args.length - 1), -1, dtype=np.int8)
        for row, text in enumerate(chunk):
            c = position_classes(text)
            cls[row, :len(c) - 1] = c[1:]  # class of the predicted character
        masks = {name: sel(cls) for name, sel in CLASSES}
        for name, m in masks.items():
            counts[name] += int(m.sum())
        for p in protocols:
            r = int(p[4:]) if p.startswith("wipe") else 0
            losses = (run_pass(model, hot, config, write=(p != "off"), wipe_every=r) * valid[:, 1:]).cpu().numpy()
            if p == "off" and start == 0 and not getattr(model, "enable_recurrence", False):
                check = (run_pass(model, hot, config, write=True, wipe_every=1) * valid[:, 1:]).cpu().numpy()
                assert np.array_equal(check, losses), "WIPE 1 differs from OFF"
            for name, m in masks.items():
                sums[p][name] += float(losses[m].sum())
        if start // batch % 8 == 0:
            print(f"stories {start + len(chunk)}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    ce = {p: {name: sums[p][name] / counts[name] for name, _ in CLASSES} for p in protocols}
    res = {"checkpoint": args.checkpoint, "iteration": iteration, "stories": len(texts), "length": args.length,
           "wipe_every": periods, "n": counts, "ce": ce,
           "arm_config": {k: config.get(k) for k in ("plasticity", "fast_layers", "layer_norm", "seed",
                                                      "model_type", "updater", "enable_recurrence",
                                                      "train_wipe_chars", "forget_rate")}}
    res["inference_forgetting"] = forgetting
    gain = ce["off"]["all"] - ce["on"]["all"]
    res["H"] = {str(r): (ce["off"]["all"] - ce[f"wipe{r}"]["all"]) / gain if gain else float("nan") for r in periods}
    res["seconds"] = time.time() - t0
    with open(args.out + ".json", "w") as handle:
        json.dump(res, handle, indent=1)
    print(json.dumps({p: round(ce[p]["all"], 4) for p in protocols}), json.dumps(res["H"]), f"{res['seconds']:.0f}s")


def synth(args):
    from heldout import load_heldout_batches
    device = args.device
    model, config, charset, iteration = load_model(args.checkpoint, device)
    forgetting = inference_forgetting(model, config, getattr(args, "inference_forget_rate", None))
    batches = load_heldout_batches(config["dataset"], config["batch_size"], args.batches, device)
    n_layers = len(model.linear_layers) if has_fast_weights(model) else 0
    dim = model.linear_layers[0].in_features
    corpus = [CorpusPR(dim, device) for _ in range(n_layers)]
    episode = {l: ([], []) for l in range(n_layers)}
    for texts, onehot in batches:
        valid = onehot[:, 1:].sum(-1) > 0  # [B, T-1]
        _, keys = run_pass(model, onehot, config, write=True, keys=True)
        for l, k in enumerate(keys):
            unit = k / k.norm(dim=2, keepdim=True).clamp_min(1e-12)
            corpus[l].add(unit[valid.T])
            for b in range(unit.shape[1]):
                n = int(valid[b].sum())
                u, c = window_prs(unit[:n, b:b + 1], torch.tensor([n]), n)
                episode[l][0].extend(u)
                episode[l][1].extend(c)
    res = {"checkpoint": args.checkpoint, "iteration": iteration, "dataset": config["dataset"],
           "episodes": len(batches) * config["batch_size"], "pr": {}, "inference_forgetting": forgetting}
    for l in range(n_layers):
        entry = corpus[l].result()
        entry["episode_pr_uncentered"] = float(np.mean(episode[l][0]))
        entry["episode_pr_centered"] = float(np.mean(episode[l][1]))
        res["pr"][f"layer{l}"] = entry
    with open(args.out + ".json", "w") as handle:
        json.dump(res, handle, indent=1)
    print(json.dumps(res["pr"]))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tiny", "curve", "synth", "dumpval", "horizon"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="output prefix")
    parser.add_argument("--names", default=None)
    parser.add_argument("--stories", type=int, default=4096)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--val_file", default=None, help="JSON list of filtered validation stories (dumpval)")
    parser.add_argument("--wipe_every", default="4,8,16,32,64,128", help="horizon: comma-separated wipe periods R")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--forget_rate_override", "--inference_forget_rate", dest="inference_forget_rate",
                        type=forget_rate_value, default=None,
                        help="inference-only fraction forgotten per write, in [0,1]; default: checkpoint rate")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "dumpval" and args.inference_forget_rate is not None:
        parser.error("--forget_rate_override is only valid for model scoring modes")
    if args.mode == "dumpval":
        with open(args.out, "w") as handle:
            json.dump(load_validation(args.stories, 10**9), handle)
    elif args.mode == "synth":
        synth(args)
    elif args.mode == "horizon":
        horizon(args)
    else:
        if args.mode == "curve":
            args.names = None
        tiny(args)


if __name__ == "__main__":
    main()
