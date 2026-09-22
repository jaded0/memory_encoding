"""Per-interval training metrics, including accuracy on targets that can only be predicted from memory.

A "recall target" is a position whose next character is determined by an earlier character of the same
sequence and cannot be inferred from the current input alone. Its *lag* is the number of steps between the
step where the remembered character was the input and the step that must predict it. By construction,
``text[target] == text[target - lag - 1]``.
"""
import torch

from utils import get_charset


def recall_targets(text, dataset_name):
    """Return ({target_index: lag}, end_index) for one sequence.

    ``end_index`` is the first padding position after a variable-length palindrome's mirrored half. Predicting
    it requires knowing the half-length, so it is tracked separately from the recalled characters. A model that
    always predicts padding gets it right, so read it together with the recall accuracy.
    """
    if "palindrome_dataset_vary_length" in dataset_name:
        middle = text.index(".")
        targets = {middle + j: 2 * j - 1 for j in range(1, middle + 1)}
        end = 2 * middle + 1
        return targets, (end if end < len(text) else None)
    if dataset_name == "palindrome_dataset":
        half = len(text) // 2
        return {half + j: 2 * j - 1 for j in range(1, half + 1)}, None
    if "long_range_memory" in dataset_name:
        store, query = text.index("?"), text.index("!")
        return {query + 1: query - store - 1}, None
    if "resequence" in dataset_name:
        period = len(set(text))  # characters are sampled without replacement, so distinct count = period
        if period < 2:
            return {}, None  # a single repeated character is predictable from the current input
        return {i: period - 1 for i in range(period, len(text))}, None
    return {}, None


def recall_chance(dataset_name):
    """Accuracy of guessing uniformly among the characters a recall target can take."""
    charset = get_charset(dataset_name)
    if "palindrome_dataset_vary_length" in dataset_name:
        return 1 / len([c for c in charset if c not in ". "])
    if "long_range_memory" in dataset_name:
        return 1 / len(charset[3:])
    if dataset_name == "palindrome_dataset" or "resequence" in dataset_name:
        return 1 / len(charset)
    return None


MAX_LAG = 60  # recall lags are bucketed by key = lag + 2 (0 = padding, 1 = non-recall target)
KEYS = MAX_LAG + 2
MASK_CACHE_LIMIT = 100_000


class IntervalMetrics:
    """Accumulates metrics over a logging interval with a handful of kernels and no device syncs.

    Each target position gets a key (0 padding, 1 ordinary target, lag + 2 recall target); per-key counts,
    correct predictions, and losses are reduced with ``scatter_add_`` into one stats vector that stays on device
    until ``summary()``. Per-sequence masks are cached, since synthetic tasks have few distinct sequences.
    """

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.has_recall = recall_chance(dataset_name) is not None
        self._mask_cache = {}
        self.reset()

    def reset(self):
        self.stats = None
        self.iterations = 0

    def _masks(self, text):
        cached = self._mask_cache.get(text)
        if cached is None:
            keys = torch.ones(len(text) - 1, dtype=torch.long)
            positions, end_index = recall_targets(text, self.dataset_name)
            for target, lag in positions.items():
                if target < len(text):
                    if lag > MAX_LAG:
                        raise ValueError(f"recall lag {lag} exceeds MAX_LAG={MAX_LAG}")
                    keys[target - 1] = lag + 2
            cached = (keys, end_index - 1 if end_index is not None else -1)
            if len(self._mask_cache) < MASK_CACHE_LIMIT:
                self._mask_cache[text] = cached
        return cached

    def update(self, texts, onehot, preds, losses):
        """Add one batch.

        texts: list of B strings; onehot: [B, T, V]; preds, losses: [T-1, B] per-step predictions and
        per-sample cross-entropy.
        """
        device = preds.device
        preds, losses = preds.t(), losses.t()  # [B, T-1]
        targets = onehot[:, 1:].argmax(-1)
        valid = onehot[:, 1:].sum(-1) > 0  # padding rows are all-zero
        correct = (preds == targets) & valid
        batch, steps = valid.shape
        self.iterations += 1

        if self.has_recall:
            masks = [self._masks(text) for text in texts]
            key_rows = [row for row, _ in masks]
            if all(len(row) == steps for row in key_rows):
                keys = torch.stack(key_rows)
            else:
                keys = torch.zeros(batch, steps, dtype=torch.long)
                for b, row in enumerate(key_rows):
                    keys[b, :len(row)] = row[:steps]
            end = torch.tensor([e for _, e in masks], dtype=torch.long)
            keys, end = keys.to(device, non_blocking=True) * valid, end.to(device, non_blocking=True)
        else:
            keys, end = valid.long(), torch.full((batch,), -1, dtype=torch.long, device=device)

        # scatter_add_ into fixed-size buffers: unlike bincount on CUDA, it needs no device sync
        flat = keys.flatten()
        per_key = torch.zeros(3, KEYS, device=device)
        values = torch.stack([valid.flatten().float(), correct.flatten().float(), (losses * valid).flatten()])
        per_key.scatter_add_(1, flat.expand(3, -1), values)
        counts, hits, loss_sums = per_key

        rows = torch.arange(batch, device=device)
        recall = keys >= 2
        has_recall = recall.any(1)
        has_end = end >= 0
        end_hit = correct[rows, end.clamp(min=0)] & has_end
        extras = torch.stack([
            correct[rows, valid.sum(1) - 1].sum(), torch.tensor(batch, device=device),
            ((correct | ~recall).all(1) & has_recall).sum(), has_recall.sum(),
            end_hit.sum(), has_end.sum(),
        ]).float()
        stats = torch.cat([counts, hits, loss_sums, extras])
        self.stats = stats if self.stats is None else self.stats + stats

    def summary(self):
        if self.stats is None:
            return {}
        stats = self.stats.cpu()
        counts, hits, loss_sums, extras = stats[:KEYS], stats[KEYS:2 * KEYS], stats[2 * KEYS:3 * KEYS], stats[3 * KEYS:]
        final_hit, final_count, exact, seq_count, end_hit, end_count = extras.tolist()
        ratio = lambda num, den: float(num) / float(den) if den else None
        tokens, recall_count = counts[1:].sum(), counts[2:].sum()
        out = {
            "loss": ratio(loss_sums[1:].sum(), tokens),
            "token_acc": ratio(hits[1:].sum(), tokens),
            "final_char_acc": ratio(final_hit, final_count),
            "recall_acc": ratio(hits[2:].sum(), recall_count),
            "recall_loss": ratio(loss_sums[2:].sum(), recall_count),
            "other_loss": ratio(loss_sums[1], counts[1]) if recall_count else None,
            "recall_seq_exact": ratio(exact, seq_count),
            "recall_end_acc": ratio(end_hit, end_count),
        }
        for key in range(2, KEYS):
            if counts[key]:
                out[f"recall_acc_lag_{key - 2}"] = float(hits[key] / counts[key])
        return {k: v for k, v in out.items() if v is not None}
