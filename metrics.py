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


class IntervalMetrics:
    """Accumulates metrics over a logging interval. Tensors stay on device until ``summary()``."""

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.reset()

    def reset(self):
        self.sums = {}
        self.lag_sums = {}
        self.iterations = 0

    def _add(self, store, key, value):
        store[key] = store.get(key, 0) + value

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
        self.iterations += 1

        self._add(self.sums, "loss", (losses * valid).sum())
        self._add(self.sums, "tokens", valid.sum())
        self._add(self.sums, "correct", correct.sum())
        last = valid.sum(1) - 1  # index of each sequence's final target
        rows = torch.arange(preds.shape[0], device=device)
        self._add(self.sums, "final_correct", correct[rows, last].sum())
        self._add(self.sums, "final_count", preds.shape[0])

        # Build masks on the CPU (the lags are known there) to avoid a device sync every iteration.
        recall = torch.zeros(valid.shape, dtype=torch.bool)
        lag = torch.full(valid.shape, -1, dtype=torch.long)
        end = torch.zeros(valid.shape, dtype=torch.bool)
        lags_seen = set()
        for b, text in enumerate(texts):
            positions, end_index = recall_targets(text, self.dataset_name)
            for target, target_lag in positions.items():
                if target >= len(text):
                    continue  # truncated sequence
                recall[b, target - 1] = True
                lag[b, target - 1] = target_lag
                lags_seen.add(target_lag)
            if end_index is not None:
                end[b, end_index - 1] = True
        if not lags_seen:
            return
        recall, lag, end = (t.to(device, non_blocking=True) for t in (recall, lag, end))

        self._add(self.sums, "recall_correct", (correct & recall).sum())
        self._add(self.sums, "recall_count", recall.sum())
        self._add(self.sums, "recall_loss", (losses * recall).sum())
        other = valid & ~recall
        self._add(self.sums, "other_loss", (losses * other).sum())
        self._add(self.sums, "other_count", other.sum())
        has_recall = recall.any(1)
        exact = ((correct | ~recall).all(1) & has_recall).sum()
        self._add(self.sums, "seq_exact", exact)
        self._add(self.sums, "seq_count", has_recall.sum())
        self._add(self.sums, "end_correct", (correct & end).sum())
        self._add(self.sums, "end_count", end.sum())
        for value in lags_seen:
            at_lag = lag == value
            self._add(self.lag_sums, (value, "correct"), (correct & at_lag).sum())
            self._add(self.lag_sums, (value, "count"), at_lag.sum())

    def summary(self):
        s = {k: float(v) for k, v in self.sums.items()}
        ratio = lambda num, den: s[num] / s[den] if s.get(den) else None
        out = {
            "loss": ratio("loss", "tokens"),
            "token_acc": ratio("correct", "tokens"),
            "final_char_acc": ratio("final_correct", "final_count"),
            "recall_acc": ratio("recall_correct", "recall_count"),
            "recall_loss": ratio("recall_loss", "recall_count"),
            "other_loss": ratio("other_loss", "other_count"),
            "recall_seq_exact": ratio("seq_exact", "seq_count"),
            "recall_end_acc": ratio("end_correct", "end_count"),
        }
        for value in sorted({lag for lag, _ in self.lag_sums}):
            out[f"recall_acc_lag_{value}"] = float(self.lag_sums[(value, "correct")]) / float(self.lag_sums[(value, "count")])
        return {k: v for k, v in out.items() if v is not None}
