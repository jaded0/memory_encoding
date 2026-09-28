"""Small deterministic token-level memory tasks for held-out evaluation.

The key-value episode is a dense token stream::

    key value ... key value [dedicated distractors...] QUERY queried-key answer

Inputs are every token except ``answer`` and targets are every token except the first, so the
answer is revealed only as the final target. ``unique`` samples distinct table keys and independent
values. ``replacement`` samples keys and values with replacement; the answer is the queried key's
latest assignment. This follows the core shape of Ba-style associative recall without claiming an
exact replication of a particular benchmark.

All randomness comes from a private CPU ``torch.Generator``. Tensors are dense and equal-length;
there is no padding or validity mask in this first implementation.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from heldout import HeldOutBatch, HeldOutResult, evaluate_held_out


STALE_SENTINEL = -1


@dataclass(frozen=True)
class TokenLayout:
    """Inspectable, disjoint token allocation for one generator configuration."""

    key_start: int
    key_stop: int
    value_start: int
    value_stop: int
    distractor_start: int
    distractor_stop: int
    query_marker: int
    vocab_size: int

    @property
    def key_ids(self):
        return range(self.key_start, self.key_stop)

    @property
    def value_ids(self):
        return range(self.value_start, self.value_stop)

    @property
    def distractor_ids(self):
        return range(self.distractor_start, self.distractor_stop)


@dataclass(frozen=True)
class KeyValueEpisodes:
    """A held-out batch and token metadata sufficient to audit every answer.

    Positions are zero-based positions in ``token_ids``. ``latest_source_position`` identifies the
    latest queried-key value token, and ``source_to_answer_lag`` is answer position minus that source
    position. ``stale_values`` is the immediately preceding queried-key assignment, or
    ``STALE_SENTINEL`` when no preceding assignment exists.
    """

    batch: HeldOutBatch
    layout: TokenLayout
    token_ids: torch.Tensor
    assignment_keys: torch.Tensor
    assignment_values: torch.Tensor
    query_keys: torch.Tensor
    answers: torch.Tensor
    stale_values: torch.Tensor
    query_assignment_counts: torch.Tensor
    latest_source_position: torch.Tensor
    source_to_answer_lag: torch.Tensor

    def to(self, device=None, dtype=None):
        """Return a moved copy, changing dtype only for the batch's floating one-hot tensors."""
        integer_names = ("token_ids", "assignment_keys", "assignment_values", "query_keys",
                         "answers", "stale_values", "query_assignment_counts",
                         "latest_source_position", "source_to_answer_lag")
        moved = {name: getattr(self, name).to(device=device) for name in integer_names}
        return KeyValueEpisodes(self.batch.to(device=device, dtype=dtype), self.layout, **moved)


@dataclass(frozen=True)
class KeyValueSummary:
    """Aggregate query metrics; every rate uses ``query_count`` as its denominator.

    ``full_output_uniform_chance`` is uniform guessing over every model output token.
    ``value_restricted_chance`` assumes the guesser knows the answer must be a value token and
    guesses uniformly only within that dedicated range.
    """

    query_count: int
    query_loss: float
    query_accuracy: float
    full_output_uniform_chance: float
    value_restricted_chance: float
    correct_count: int
    correct_rate: float
    stale_count: int
    stale_rate: float
    immediate_stale_count: int
    immediate_stale_rate: float
    older_stale_count: int
    older_stale_rate: float
    wrong_key_count: int
    wrong_key_rate: float
    other_value_count: int
    other_value_rate: float
    non_value_count: int
    non_value_rate: float
    stale_eligible_count: int
    stale_eligible_rate: float


@dataclass(frozen=True)
class KeyValueEvaluation:
    """Held-out outputs and per-row query diagnostics.

    ``correct``, ``stale``, ``wrong_key``, ``other_value``, and ``non_value`` are mutually
    exclusive and exhaustive. Priority is correct latest answer, any superseded queried-key value,
    a value seen under another key, another value-vocabulary token, then a non-value token.
    ``immediate_stale`` is an explicitly requested subset of ``stale``; ``older_stale`` is the
    disjoint remainder. Thus a correct token always stays correct even if seen elsewhere, and a
    stale token wins over overlap with another key.
    """

    result: HeldOutResult
    query_predictions: torch.Tensor
    query_targets: torch.Tensor
    correct: torch.Tensor
    stale: torch.Tensor
    immediate_stale: torch.Tensor
    older_stale: torch.Tensor
    wrong_key: torch.Tensor
    other_value: torch.Tensor
    non_value: torch.Tensor
    stale_eligible: torch.Tensor
    summary: KeyValueSummary


def _positive_int(name, value, *, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")


def generate_key_value_episodes(*, batch_size: int, table_size: int,
                                key_vocab_size: int, value_vocab_size: int,
                                mode: str = "unique", distractor_count: int = 0,
                                distractor_vocab_size: int = 1,
                                query_mode: str = "strict", seed: int = 0,
                                query_key_assignments: int | None = None) -> KeyValueEpisodes:
    """Generate deterministic key-value episodes as a :class:`HeldOutBatch`.

    Args:
        mode: ``unique`` or ``replacement``.
        distractor_count: Number of dedicated distractor tokens inserted between the table and
            query marker. Increasing it by one increases stream length and source-to-answer lag
            by one, without changing the vocabulary.
        distractor_vocab_size: Size of the fixed dedicated distractor-token range. Distractors are
            sampled with replacement from this range independently across rows and positions.
        query_mode: ``strict`` enables writes for support inputs only (table and distractors), and
            disables writes for the query-marker and queried-key inputs. ``observed`` permits every
            write, including the final answer-target write after that answer is scored.
        query_key_assignments: Replacement mode only. If provided, the queried key occurs exactly
            this many times in the table (one assignment plus ``n - 1`` overwrites). With at least
            two assignments and at least two value tokens, the final value is forced to differ from
            the immediately previous queried-key value, making the stale answer unambiguous.
    """
    for name, value in (("batch_size", batch_size), ("table_size", table_size),
                        ("key_vocab_size", key_vocab_size),
                        ("value_vocab_size", value_vocab_size)):
        _positive_int(name, value)
    _positive_int("distractor_count", distractor_count, allow_zero=True)
    _positive_int("distractor_vocab_size", distractor_vocab_size)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if mode not in ("unique", "replacement"):
        raise ValueError("mode must be 'unique' or 'replacement'")
    if query_mode not in ("strict", "observed"):
        raise ValueError("query_mode must be 'strict' or 'observed'")
    if mode == "unique" and table_size > key_vocab_size:
        raise ValueError("unique mode requires table_size <= key_vocab_size")
    if query_key_assignments is not None:
        _positive_int("query_key_assignments", query_key_assignments)
        if mode != "replacement":
            raise ValueError("query_key_assignments is supported only in replacement mode")
        if query_key_assignments > table_size:
            raise ValueError("query_key_assignments cannot exceed table_size")
        if key_vocab_size == 1 and query_key_assignments != table_size:
            raise ValueError("with one key token, the queried key must occupy every table assignment")

    key_start, key_stop = 0, key_vocab_size
    value_start, value_stop = key_stop, key_stop + value_vocab_size
    distractor_start = value_stop
    distractor_stop = distractor_start + distractor_vocab_size
    query_marker = distractor_stop
    layout = TokenLayout(key_start, key_stop, value_start, value_stop,
                         distractor_start, distractor_stop, query_marker, query_marker + 1)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    keys = torch.empty(batch_size, table_size, dtype=torch.long)
    values = torch.randint(value_vocab_size, (batch_size, table_size), generator=generator)
    query_keys = torch.empty(batch_size, dtype=torch.long)

    for row in range(batch_size):
        if mode == "unique":
            row_keys = torch.randperm(key_vocab_size, generator=generator)[:table_size]
            query_index = torch.randint(table_size, (), generator=generator).item()
            query_key = row_keys[query_index]
        elif query_key_assignments is None:
            row_keys = torch.randint(key_vocab_size, (table_size,), generator=generator)
            query_index = torch.randint(table_size, (), generator=generator).item()
            query_key = row_keys[query_index]
        else:
            query_key = torch.randint(key_vocab_size, (), generator=generator)
            positions = torch.randperm(table_size, generator=generator)[:query_key_assignments]
            row_keys = torch.empty(table_size, dtype=torch.long)
            selected = torch.zeros(table_size, dtype=torch.bool)
            selected[positions] = True
            row_keys[selected] = query_key
            remaining = table_size - query_key_assignments
            if remaining:
                # Sample uniformly from every key except query_key, preserving an exact count.
                other = torch.randint(key_vocab_size - 1, (remaining,), generator=generator)
                other += (other >= query_key).long()
                row_keys[~selected] = other
        keys[row] = row_keys
        query_keys[row] = query_key

        occurrences = torch.nonzero(row_keys == query_key, as_tuple=False).flatten()
        if query_key_assignments is not None and len(occurrences) >= 2 and value_vocab_size >= 2:
            previous, latest = occurrences[-2].item(), occurrences[-1].item()
            if values[row, latest] == values[row, previous]:
                # Uniformly choose one of the other values without another rejection draw.
                draw = torch.randint(value_vocab_size - 1, (), generator=generator)
                values[row, latest] = draw + (draw >= values[row, previous]).long()

    # Convert local key/value indices to their disjoint token ranges.
    assignment_keys = keys + layout.key_start
    assignment_values = values + layout.value_start
    query_keys = query_keys + layout.key_start
    answer = torch.empty(batch_size, dtype=torch.long)
    stale = torch.full((batch_size,), STALE_SENTINEL, dtype=torch.long)
    counts = torch.empty(batch_size, dtype=torch.long)
    source_positions = torch.empty(batch_size, dtype=torch.long)
    for row in range(batch_size):
        occurrences = torch.nonzero(assignment_keys[row] == query_keys[row], as_tuple=False).flatten()
        latest = occurrences[-1].item()
        counts[row] = len(occurrences)
        answer[row] = assignment_values[row, latest]
        source_positions[row] = 2 * latest + 1  # the value token, not its preceding key
        if len(occurrences) >= 2:
            stale[row] = assignment_values[row, occurrences[-2]]

    table = torch.stack((assignment_keys, assignment_values), dim=2).reshape(batch_size, -1)
    # Draw only after all assignments, queries, answers, and source metadata are fixed, so changing
    # distractor_count alone cannot perturb the task's associative content.
    distractors = torch.randint(distractor_vocab_size, (batch_size, distractor_count),
                                generator=generator) + layout.distractor_start
    marker = torch.full((batch_size, 1), layout.query_marker, dtype=torch.long)
    token_ids = torch.cat((table, distractors, marker, query_keys[:, None], answer[:, None]), dim=1)
    inputs_ids, targets_ids = token_ids[:, :-1], token_ids[:, 1:]
    inputs = F.one_hot(inputs_ids, layout.vocab_size).to(torch.float32)
    targets = F.one_hot(targets_ids, layout.vocab_size).to(torch.float32)

    steps = inputs_ids.shape[1]
    score_mask = torch.zeros(batch_size, steps, dtype=torch.bool)
    score_mask[:, -1] = True
    reset_mask = torch.zeros_like(score_mask)
    reset_mask[:, 0] = True
    update_mask = torch.ones_like(score_mask)
    if query_mode == "strict":
        marker_input_position = 2 * table_size + distractor_count
        update_mask[:, marker_input_position:] = False

    answer_position = token_ids.shape[1] - 1
    lag = answer_position - source_positions
    batch = HeldOutBatch(inputs, targets, score_mask, update_mask, reset_mask)
    return KeyValueEpisodes(batch, layout, token_ids, assignment_keys, assignment_values,
                            query_keys, answer, stale, counts, source_positions, lag)


def summarize_key_value(result: HeldOutResult, episodes: KeyValueEpisodes) -> KeyValueEvaluation:
    """Extract and classify the one scored key-value query prediction in every row.

    The five primary categories use the priority documented by :class:`KeyValueEvaluation`.
    Immediate stale is also reported as a subset of any stale; older stale is its complement.
    This performs structural and internal-consistency validation of result and episode tensors; it
    cannot prove that an otherwise consistent ``HeldOutResult`` was produced from these episodes.
    """
    batch = episodes.batch
    batch_size, steps = batch.score_mask.shape
    expected_shapes = {
        "predictions": (batch_size, steps),
        "losses": (batch_size, steps),
        "logits": (batch_size, steps, episodes.layout.vocab_size),
    }
    for name, shape in expected_shapes.items():
        if tuple(getattr(result, name).shape) != shape:
            raise ValueError(f"result.{name} must have shape {shape}")
    if result.final_hidden.ndim != 2 or result.final_hidden.shape[0] != batch_size:
        raise ValueError("result.final_hidden must be rank 2 with one row per episode")
    per_row_scores = batch.score_mask.sum(dim=1)
    if not torch.equal(per_row_scores, torch.ones_like(per_row_scores)):
        raise ValueError("key-value episodes must score exactly one transition per row")
    if not batch.score_mask[:, -1].all() or batch.score_mask[:, :-1].any():
        raise ValueError("key-value episodes must score only the final answer transition")
    if result.scored_count != batch_size:
        raise ValueError(f"result.scored_count must equal the query count ({batch_size})")
    metadata = (episodes.token_ids, episodes.assignment_keys, episodes.assignment_values, episodes.query_keys,
                episodes.answers, episodes.stale_values, episodes.query_assignment_counts,
                episodes.latest_source_position, episodes.source_to_answer_lag)
    if any(t.device != result.predictions.device for t in metadata):
        raise ValueError("result and episode metadata must be on the same device")
    batch_tensors = (batch.inputs, batch.targets, batch.score_mask, batch.update_mask,
                     batch.reset_mask)
    if any(t.device != result.predictions.device for t in batch_tensors):
        raise ValueError("result and held-out batch must be on the same device")
    if (result.losses.device != result.predictions.device or
            result.logits.device != result.predictions.device or
            result.final_hidden.device != result.predictions.device):
        raise ValueError("result tensors must be on the same device")
    if tuple(episodes.token_ids.shape) != (batch_size, steps + 1):
        raise ValueError(f"episodes.token_ids must have shape {(batch_size, steps + 1)}")
    if episodes.token_ids.dtype != torch.long:
        raise ValueError("episodes.token_ids must contain integer token IDs")
    if tuple(episodes.assignment_keys.shape) != tuple(episodes.assignment_values.shape):
        raise ValueError("assignment key/value metadata shapes must match")
    if episodes.assignment_keys.shape[0] != batch_size:
        raise ValueError("assignment metadata must have one row per episode")
    for name in ("query_keys", "answers", "stale_values", "query_assignment_counts",
                 "latest_source_position", "source_to_answer_lag"):
        if tuple(getattr(episodes, name).shape) != (batch_size,):
            raise ValueError(f"episodes.{name} must have shape ({batch_size},)")
    target_ids = batch.targets.argmax(dim=2)
    if not torch.equal(target_ids[:, -1], episodes.answers):
        raise ValueError("final scored targets do not match episode answers")
    if not torch.equal(episodes.token_ids[:, -2], episodes.query_keys):
        raise ValueError("token_ids must end with each queried key before its answer")
    if not torch.equal(episodes.token_ids[:, -1], episodes.answers):
        raise ValueError("token_ids must end with each correct answer")

    predictions = result.predictions[:, -1]
    if ((predictions < 0) | (predictions >= episodes.layout.vocab_size)).any():
        raise ValueError("query predictions must be valid output token IDs")
    if not torch.equal(result.predictions, result.logits.argmax(dim=-1)):
        raise ValueError("result.predictions must equal result.logits.argmax(-1)")
    expected_losses = -(batch.targets * F.log_softmax(result.logits, dim=-1)).sum(dim=-1)
    if result.logits.dtype == torch.float64:
        loss_rtol, loss_atol = 1e-10, 1e-12
    elif result.logits.dtype == torch.float32:
        loss_rtol, loss_atol = 1e-5, 1e-6
    else:
        # Half/bfloat formats need a tolerance scaled to their substantially larger epsilon.
        epsilon = torch.finfo(result.logits.dtype).eps
        loss_rtol, loss_atol = 10 * epsilon, 10 * epsilon
    if not torch.allclose(result.losses, expected_losses, rtol=loss_rtol, atol=loss_atol):
        raise ValueError("result.losses must match cross-entropy implied by logits and targets")
    targets = episodes.answers
    correct = predictions == targets
    stale = torch.zeros(batch_size, dtype=torch.bool, device=predictions.device)
    immediate = torch.zeros_like(stale)
    stale_eligible = torch.zeros_like(stale)
    seen_under_other_key = torch.zeros_like(stale)
    for row in range(batch_size):
        keys = episodes.assignment_keys[row]
        values = episodes.assignment_values[row]
        occurrences = torch.nonzero(keys == episodes.query_keys[row], as_tuple=False).flatten()
        if len(occurrences) == 0:
            raise ValueError("every query key must occur in its assignment table")
        if episodes.query_assignment_counts[row] != len(occurrences):
            raise ValueError("query_assignment_counts does not match the assignment table")
        latest = occurrences[-1]
        if values[latest] != episodes.answers[row]:
            raise ValueError("episode answer is not the latest queried-key assignment")
        expected_source = 2 * latest + 1
        if episodes.latest_source_position[row] != expected_source:
            raise ValueError("latest_source_position does not identify the latest queried value")
        answer_position = steps
        if episodes.source_to_answer_lag[row] != answer_position - expected_source:
            raise ValueError("source_to_answer_lag is inconsistent with latest_source_position")
        if len(occurrences) >= 2:
            stale_eligible[row] = True
            stale_values = values[occurrences[:-1]]
            stale[row] = (predictions[row] == stale_values).any()
            immediate[row] = predictions[row] == values[occurrences[-2]]
            if episodes.stale_values[row] != values[occurrences[-2]]:
                raise ValueError("stale_values does not identify the immediate stale value")
        elif episodes.stale_values[row] != STALE_SENTINEL:
            raise ValueError("stale_values must use the sentinel when no stale assignment exists")
        seen_under_other_key[row] = (
            (values == predictions[row]) & (keys != episodes.query_keys[row])).any()

    # Correctness has first priority; immediate is meaningful only within stale errors.
    stale &= ~correct
    immediate &= stale
    older = stale & ~immediate
    wrong_key = seen_under_other_key & ~correct & ~stale
    in_value_range = ((predictions >= episodes.layout.value_start) &
                      (predictions < episodes.layout.value_stop))
    other_value = in_value_range & ~correct & ~stale & ~wrong_key
    non_value = ~(correct | stale | wrong_key | other_value)
    if not torch.all((correct.to(torch.int8) + stale.to(torch.int8) + wrong_key.to(torch.int8) +
                      other_value.to(torch.int8) + non_value.to(torch.int8)) == 1):
        raise RuntimeError("key-value diagnostic categories are not exclusive and exhaustive")

    query_losses = result.losses[:, -1]
    query_loss = query_losses.mean().item()
    query_accuracy = correct.float().mean().item()
    if abs(result.scored_loss - query_loss) > 1e-6:
        raise ValueError("result.scored_loss does not match the episode's scored query losses")
    if abs(result.scored_accuracy - query_accuracy) > 1e-6:
        raise ValueError("result.scored_accuracy does not match the episode's scored predictions")

    def count_rate(mask):
        count = int(mask.sum().item())
        return count, count / batch_size

    correct_count, correct_rate = count_rate(correct)
    stale_count, stale_rate = count_rate(stale)
    immediate_count, immediate_rate = count_rate(immediate)
    older_count, older_rate = count_rate(older)
    wrong_count, wrong_rate = count_rate(wrong_key)
    other_count, other_rate = count_rate(other_value)
    non_value_count, non_value_rate = count_rate(non_value)
    eligible_count, eligible_rate = count_rate(stale_eligible)
    summary = KeyValueSummary(
        batch_size, query_loss, query_accuracy, 1 / episodes.layout.vocab_size,
        1 / (episodes.layout.value_stop - episodes.layout.value_start),
        correct_count, correct_rate, stale_count, stale_rate,
        immediate_count, immediate_rate, older_count, older_rate,
        wrong_count, wrong_rate, other_count, other_rate,
        non_value_count, non_value_rate, eligible_count, eligible_rate)
    return KeyValueEvaluation(result, predictions.detach().clone(), targets.detach().clone(),
                              correct, stale, immediate, older, wrong_key, other_value,
                              non_value, stale_eligible, summary)


def evaluate_key_value(model, episodes: KeyValueEpisodes, learning_rate: float,
                       update_clamp: float = 0.0, initial_state: str = "fresh",
                       initial_hidden: torch.Tensor | None = None) -> KeyValueEvaluation:
    """Evaluate and summarize episodes without implicitly moving model or episode tensors."""
    result = evaluate_held_out(model, episodes.batch, learning_rate, update_clamp=update_clamp,
                               initial_state=initial_state, initial_hidden=initial_hidden)
    return summarize_key_value(result, episodes)
