# train.py
import torch
from ephemeral_model import EphemeralRNN, SimpleRNN, dfa_output_error, parse_slow_update_every, parse_fast_backward_per_forward
import wandb
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess_data
from reproducibility import DataStream, capture_rng_state, record_seed_in_slurm, resolve_seed, seed_everything
from metrics import IntervalMetrics, recall_chance
from heldout import evaluate_protocols, load_heldout_batches
from loop_trace import NULL_TRACER, LoopTracer, summarize as summarize_traces
from utils import model_input, randomTrainingExample, timeSince, str2bool, initialize_charset, save_checkpoint, load_checkpoint, read_checkpoint, check_checkpoint_code_version, CHECKPOINT_CODE_VERSION, keep_numbered_checkpoint
import time
import math
import argparse
import sys
import itertools
import os
import psutil
import signal
try:
    from wandb_osh.hooks import TriggerWandbSyncHook  # <-- New!
except ImportError:
    TriggerWandbSyncHook = None

class TermColors:
    GREEN = '\033[92m'
    PURPLE = '\033[95m' # Magenta, often used for Purple
    WHITE = '\033[97m'
    RESET = '\033[0m'

def plot_ascii_bar_graph(data, title, max_width=40):
    if not data:
        return

    print(f"--- {title} ---")
    
    # Check if there's anything to plot
    valid_values = [v for v in data.values() if v > 0]
    if not valid_values:
        for label, value in sorted(data.items()):
            print(f"  {label}: {value:.6f}")
        print("  (all values are zero or negative)")
        return

    max_val = max(valid_values)
    max_label_len = max(len(label) for label in data.keys())

    for label, value in sorted(data.items()):
        bar_len = int((value / max_val) * max_width) if value > 0 else 0
        bar = '█' * bar_len
        print(f"  {label.ljust(max_label_len)} | {bar} ({value:.6f})")

# from memory_profiler import profile

trigger_sync = TriggerWandbSyncHook() if TriggerWandbSyncHook else None  # <--- New!

# --- W&B end-of-run markers ---
def wb_mark_end(reason: str, tags=None, exit_code: int | None = None):
    """Record an end reason in both tags and summary. Safe if tracking is off."""
    run = getattr(wandb, "run", None)
    if not (run and getattr(run, "summary", None) is not None):
        return
    # structured summary
    run.summary["end_reason"] = reason
    run.summary[f"end_is_{reason}"] = True
    if exit_code is not None:
        run.summary["end_exit_code_suggested"] = int(exit_code)  # read later at finish()

    # tags (filter-friendly)
    if tags:
        current = set(getattr(run, "tags", []))
        run.tags = list(current.union(set(tags)))

def parse_plasticity_schedule(text):
    """--plasticity_schedule "ITER:VALUE,ITER:VALUE,...": [(iteration, alpha), ...] sorted by iteration
    ('' = no schedule). alpha is VALUE from iteration ITER on, until the next entry."""
    entries = []
    for item in text.split(","):
        if item.strip():
            iteration, _, value = item.partition(":")
            entries.append((int(iteration), float(value)))
    if len({iteration for iteration, _ in entries}) != len(entries) or any(i < 0 or v < 0 for i, v in entries):
        raise ValueError(f"--plasticity_schedule {text!r}: iterations must be distinct and non-negative, values >= 0")
    return sorted(entries)


def plasticity_at(schedule, iteration, default):
    """The scheduled alpha at an iteration: the last entry at or before it, else default (--plasticity)."""
    value = default
    for start, alpha in schedule:
        if start <= iteration:
            value = alpha
    return value


def compile_cache_limit(schedule, shapes_per_alpha=6, base=64):
    """torch.compile cache entries the fused step needs under --plasticity_schedule: every distinct alpha is a new
    constant for each layer shape (trunk layers, i2h, i2o), so a 41-step ramp needs about 41 * 3 entries. Above the
    limit torch silently falls back to the eager step (X4 ran 4.4x slower from iteration ~100k with limit 64)."""
    distinct = len({alpha for _, alpha in schedule})
    return max(base, distinct * shapes_per_alpha + 16)


def high_loss_stop(count, loss, window, threshold=5.0):
    """--early_stop_window: (new count of consecutive intervals with loss > threshold, stop?).
    window 0 never stops."""
    count = count + 1 if loss > threshold else 0
    return count, window > 0 and count >= window


def train_batch(line_tensor, onehot_line_tensor, rnn, config, state, optimizer=None, log_outputs=False,
                tracer=None):
    """Trains on one batch of sequences with DFA, backprop or BPTT. tracer (a loop_trace.LoopTracer,
    ephemeral + DFA only) records the within-sequence feedback-loop traces without changing
    anything; the caller reads them with tracer.finish() afterwards."""
    updater = config['updater']
    criterion = config['criterion']
    batch_size = onehot_line_tensor.shape[0]
    hidden = rnn.initHidden(batch_size=batch_size)

    # For EphemeralRNN, reset the ephemeral weights at the start of the sequence. The slow
    # entries take the batch mean every sequence; with --wipe_every N > 1 the fast entries are
    # zeroed only on every N-th sequence (state['sequence_count'] counts the sequences started,
    # and is saved with the checkpoint), so between wipes each batch row's fast state carries
    # into the next sequence in that row. The hidden state starts at zero every sequence.
    if isinstance(rnn, EphemeralRNN):
        wipe_every = config.get('wipe_every', 1)
        if wipe_every <= 1:
            rnn.start_sequence_wipe()
        else:
            count = state.get('sequence_count', 0)
            rnn.start_sequence_wipe(wipe_fast=count % wipe_every == 0)
            state['sequence_count'] = count + 1

    # Summed on the device in float64, the same sums Python floats gave, so no step waits on a
    # host sync; read once at the end of the batch.
    if tracer is None:
        tracer = NULL_TRACER
    else:
        if updater != 'dfa' or not isinstance(rnn, EphemeralRNN):
            raise ValueError("loop tracing supports only the ephemeral model with the DFA updater")
        tracer.begin(onehot_line_tensor.size(1) - 1, batch_size)
    loss_total = torch.zeros((), dtype=torch.float64, device=onehot_line_tensor.device)
    losses = []  # For DFA (per-batch losses)
    step_preds, step_losses = [], []  # [T-1] x [B], for per-interval metrics
    num_steps = 0
    all_outputs = []
    all_labels = []

    for i in range(onehot_line_tensor.size()[1] - 1):
        # For BPTT, we keep gradients flowing through time by NOT detaching hidden state
        if updater != 'bptt':
            hidden = hidden.detach()

        hot_input_char_tensor = model_input(onehot_line_tensor, i, config['input_mode'], config["pe_matrix"])

        # Forward pass
        incoming_hidden = hidden
        output, hidden = rnn(hot_input_char_tensor, hidden)
        final_char = onehot_line_tensor[:, i+1, :]
        
        # Compute loss and update weights based on updater type
        if updater == 'dfa':
            # DFA-specific processing
            # Per-sequence output error dL/d(output), [B, vocab]: a new tensor (not a view of
            # output or of grad_outputs) that needs no grad. It was two names, global_error and
            # reward_update, bound to this one object; there was never a second tensor.
            loss, output_error = dfa_output_error(
                output, final_char, criterion, config.get('label_smoothing', 0.0))
            losses.append(loss.detach())

            # --fast_backward_per_forward 1/N: only every N-th character (the first, N+1-th, ...)
            # gets a fast update. The slow stream and the forgetting are unchanged.
            skip_fast = isinstance(rnn, EphemeralRNN) and i % rnn.fast_subsample != 0
            tracer.before_update(i, output, output_error, loss, hidden, fast_write=not skip_fast)

            # Apply DFA updates
            if isinstance(rnn, EphemeralRNN) and rnn.slow_update_every != 1:
                # --slow_update_every N or sequence: this step writes the fast entries; the slow
                # step is accumulated and applied every N steps and at the end of the sequence.
                rnn.windowed_dfa_step(output_error, config["learning_rate"], config["ephemeral_update_clamp"],
                                      config.get('grad_norm_clip', 0), log_norms=state.get('log_norms_now', False),
                                      skip_fast=skip_fast, defer_apply=rnn.split_step)
            elif skip_fast:
                rnn.skip_fast_dfa_step(output_error, config["learning_rate"], config["ephemeral_update_clamp"],
                                       config.get('grad_norm_clip', 0), log_norms=state.get('log_norms_now', False))
            elif isinstance(rnn, EphemeralRNN) and rnn.split_step:
                # --fast_backward_per_forward K >= 2: pass 1's fast half now, the slow half (from the
                # saved pass-1 errors and inputs) after the extra passes below.
                slow_half = rnn.split_dfa_step(output_error, config["learning_rate"], config["ephemeral_update_clamp"],
                                               config.get('grad_norm_clip', 0), log_norms=state.get('log_norms_now', False))
            elif isinstance(rnn, EphemeralRNN) and rnn.fused_layer_step is not None and not state.get('log_norms_now', False):
                # --fused_update: the same step as the branch below, one kernel per layer. Steps that
                # log update norms take the branch below, which materializes the update.
                rnn.fused_dfa_step(output_error, config["learning_rate"], config["ephemeral_update_clamp"],
                                   config.get('grad_norm_clip', 0))
            elif isinstance(rnn, EphemeralRNN):
                rnn.clear_dfa_gradients()
                # Every layer is given this same object, and all of them are populated before any
                # update runs. i2o keeps a reference to it (as _last_projected_error,
                # for the bias update), so it must not be modified in place until the updates
                # below are done: tests/test_dfa_error_signals.py checks that.
                # i2h is the forked state head. Direct feedback trains it even though the
                # current output reads the sibling emission path; this is an explicit local
                # surrogate for temporal credit, not BPTT through future states.
                for layer in rnn.linear_layers:
                    layer.populate_dfa_gradients(output_error)
                rnn.i2h.populate_dfa_gradients(output_error)
                rnn.i2o.populate_dfa_gradients(output_error)
                # --grad_norm_clip: each sequence's raw gradient, before alpha and the clamps.
                if config.get('grad_norm_clip', 0) > 0:
                    rnn.clip_grad_norm_per_sequence(config['grad_norm_clip'])
                # Apply the updates using the DFA-populated gradients
                for layer in rnn.linear_layers:
                    layer.apply_update(config["learning_rate"], config["ephemeral_update_clamp"], state)
                rnn.i2h.apply_update(config["learning_rate"], config["ephemeral_update_clamp"], state)
                rnn.i2o.apply_update(config["learning_rate"], config["ephemeral_update_clamp"], state)

                # Forget after the whole update (incl. the clamps), as in the paper
                rnn.apply_forget_step()
                
                # Clear gradients after the updates
                rnn.clear_dfa_gradients()
            elif isinstance(rnn, SimpleRNN):
                # The same DFA as above, minus the ephemeral entries, plasticity, forgetting and
                # wiping (see DFALinear): the hidden layers and i2h project output_error through
                # their fixed feedback matrices, i2o takes it directly, each gradient is the
                # outer product with the layer's input trace (averaged over the batch, since the
                # weights are shared), and w <- w - lr * grad, b <- b - lr * mean(error).
                if rnn.updater != 'dfa':
                    raise ValueError("SimpleRNN was built without DFA feedback matrices; pass updater='dfa'.")
                for layer in rnn.dfa_layers():
                    layer.populate_dfa_gradients(output_error)
                # --grad_norm_clip: the global norm of the shared (batch-mean) DFA gradients. Non-zero,
                # it breaks the exact match with the ephemeral model's DFA, which clips per sequence.
                if config.get('grad_norm_clip', 0) > 0:
                    norm = torch.nn.utils.clip_grad_norm_(rnn.parameters(), config['grad_norm_clip'])
                    rnn.grad_clip_stats.record(norm, config['grad_norm_clip'])
                for layer in rnn.dfa_layers():
                    layer.apply_dfa_update(config["learning_rate"])
                # The grads are left in place (the next step's zero_grad clears them) so that
                # get_all_norms logs them, as it does for the backprop baseline.

            if isinstance(rnn, EphemeralRNN) and rnn.split_step:
                # --fast_backward_per_forward K: K - 1 more fast-only steps on this character; the
                # loss and metrics above are the first pass's, the next hidden state the last's.
                extra_hidden = rnn.extra_fast_iterations(
                    hot_input_char_tensor, incoming_hidden, final_char, criterion, config["learning_rate"],
                    config["ephemeral_update_clamp"], config.get('grad_norm_clip', 0), tracer=tracer, step=i)
                if extra_hidden is not None:
                    hidden = extra_hidden
                # Then the slow half of pass 1's step, or the deferred window end.
                if rnn.slow_update_every == 1:
                    rnn.apply_slow_step(slow_half, config["learning_rate"], config["ephemeral_update_clamp"])
                else:
                    rnn.finish_slow_window(config["learning_rate"])

            # After the whole character step (every pass, and the slow half or window end), so every
            # slow change it made is already in the weights.
            tracer.after_update(i)

            state['training_instance'] += 1
            loss_total += loss.detach().mean().double()
            
        elif updater == 'backprop':
            # Backprop-specific processing
            if isinstance(rnn, EphemeralRNN):
                # EphemeralRNN with TRUE backprop - compute gradients through the network
                step_loss = criterion(output, final_char)
                losses.append(step_loss.detach())
                
                # Zero gradients before backward pass
                rnn.zero_grad()
                
                # Compute gradients through the entire network (true backprop)
                total_loss = step_loss.mean() if step_loss.dim() > 0 else step_loss
                total_loss.backward(retain_graph=False)

                # --grad_norm_clip: each sequence's raw gradient, before alpha and the clamps.
                if config.get('grad_norm_clip', 0) > 0:
                    rnn.clip_grad_norm_per_sequence(config['grad_norm_clip'])

                # Scale the ephemeral weights' gradients
                rnn.scale_ephemeral_grads(config["plasticity"])
                
                # Store gradient norms for logging
                if state.get('log_norms_now', False):
                    rnn.store_all_grad_norms()
                
                # # Store projected error for bias updates (backprop case)
                # # For backprop, we can extract the bias gradient that was already computed
                # # The bias gradient is equivalent to the sum of output gradients across batch
                # if hasattr(rnn.i2o, 'bias') and rnn.i2o.bias is not None and rnn.i2o.bias.grad is not None:
                #     # Use the bias gradient as the projected error (it's already summed across batch)
                #     bias_grad = rnn.i2o.bias.grad.clone()
                #     for layer in rnn.linear_layers:
                #         layer._last_projected_error = bias_grad.unsqueeze(0)  # Add batch dim for consistency
                #     rnn.i2h._last_projected_error = bias_grad.unsqueeze(0)
                #     rnn.i2o._last_projected_error = bias_grad.unsqueeze(0)
                
                # Apply the updates using the gradients computed by backprop
                for layer in rnn.linear_layers:
                    layer.apply_update(config["learning_rate"], config["ephemeral_update_clamp"], state)
                rnn.i2h.apply_update(config["learning_rate"], config["ephemeral_update_clamp"], state)
                rnn.i2o.apply_update(config["learning_rate"], config["ephemeral_update_clamp"], state)

                # Forget after the whole update (incl. the clamps), as in the paper
                rnn.apply_forget_step()
                
                # Clear gradients after the updates
                rnn.zero_grad()
                
                state['training_instance'] += 1
                loss_total += step_loss.detach().mean().double()
            else:
                # Standard SimpleRNN with backprop
                optimizer.zero_grad()
                step_loss = criterion(output, final_char)
                # With 'none' reduction, we get per-example losses, so take mean for backward
                total_loss = step_loss.mean() if step_loss.dim() > 0 else step_loss
                total_loss.backward()
                
                if config['grad_norm_clip'] > 0:
                    norm = torch.nn.utils.clip_grad_norm_(rnn.parameters(), config['grad_norm_clip'])
                    rnn.grad_clip_stats.record(norm, config['grad_norm_clip'])
                
                optimizer.step()
                rnn.apply_regularization()
                loss_total += step_loss.detach().mean().double()
            
        elif updater == 'bptt':
            # BPTT-specific processing - accumulate loss across sequence
            step_loss = criterion(output, final_char)
            
            # For BPTT, accumulate losses across the entire sequence
            if i == 0:
                # Initialize accumulated loss on first step
                accumulated_loss = step_loss
            else:
                # Add to accumulated loss (this maintains the computation graph)
                accumulated_loss = accumulated_loss + step_loss
            
            loss_total += step_loss.detach().mean().double()
            
            # Only backward and update on the last step to get full sequence gradients
            if i == onehot_line_tensor.size()[1] - 2:  # Last step
                if isinstance(rnn, EphemeralRNN):
                    # EphemeralRNN with BPTT - backward through entire accumulated loss
                    total_loss = accumulated_loss.mean() if accumulated_loss.dim() > 0 else accumulated_loss
                    total_loss.backward(retain_graph=False)

                    # --grad_norm_clip: each sequence's raw gradient (summed over the sequence's
                    # steps), before alpha. BPTT has no ephemeral_update_clamp.
                    if config.get('grad_norm_clip', 0) > 0:
                        rnn.clip_grad_norm_per_sequence(config['grad_norm_clip'])

                    # Scale the ephemeral weights' gradients
                    rnn.scale_ephemeral_grads(config["plasticity"])
                    
                    # Store gradient norms for logging
                    if state.get('log_norms_now', False):
                        rnn.store_all_grad_norms()
                    
                    # Manual optimizer step for EphemeralRNN
                    with torch.no_grad():
                        for param in rnn.parameters():
                            if param.grad is not None:
                                param.data -= config["learning_rate"] * param.grad
                                param.grad.zero_()
                    # The slow weights persist, so --weight_clamp applies
                    # as under the other updaters. --ephemeral_update_clamp does not: the fast
                    # entries it clamps are wiped before any forward pass reads them.
                    rnn.apply_regularization()

                    # Forget after the update (only once here), as in the paper
                    rnn.apply_forget_step()
                else:
                    # Standard SimpleRNN with BPTT
                    optimizer.zero_grad()
                    total_loss = accumulated_loss.mean() if accumulated_loss.dim() > 0 else accumulated_loss
                    total_loss.backward()
                    
                    if config['grad_norm_clip'] > 0:
                        norm = torch.nn.utils.clip_grad_norm_(rnn.parameters(), config['grad_norm_clip'])
                        rnn.grad_clip_stats.record(norm, config['grad_norm_clip'])
                    
                    optimizer.step()
                    rnn.apply_regularization()

        num_steps += 1
        step_preds.append(output.detach().argmax(dim=1))
        step_losses.append((loss if updater == 'dfa' else step_loss).detach())

        if log_outputs:
            if updater == 'dfa':
                all_outputs.append(output[0].detach())
                all_labels.append(final_char[0].detach())
            else:
                all_outputs.append(output[0])
                all_labels.append(final_char[0])

    if updater == 'dfa' and isinstance(rnn, EphemeralRNN) and rnn.slow_update_every != 1:
        # --slow_update_every: the sequence's last (or only) window is applied before it ends.
        rnn.apply_pending_slow_update(config["learning_rate"])

    # Calculate final loss
    if updater == 'dfa' and losses:
        stacked_losses = torch.stack(losses)
        loss_avg = stacked_losses.mean().item()
    else:
        loss_avg = (loss_total / num_steps).item() if num_steps > 0 else 0.0

    return output, loss_avg, torch.stack(step_preds), torch.stack(step_losses), all_outputs, all_labels


def train(line_tensor, onehot_line_tensor, rnn, config, state, optimizer=None, log_outputs=False, tracer=None):
    """Main training function that sets up criterion and calls train_batch."""
    # For ALL updaters, use 'none' reduction to preserve per-example gradients
    # This allows independent weight updates per sequence in the batch
    # This is critical for ephemeral weights to adapt independently per sequence
    if config['criterion'].reduction != 'none':
        print(f"Warning: Overriding criterion reduction to 'none' for {config['updater']} training.")
        config['criterion'] = type(config['criterion'])(reduction='none')
    
    return train_batch(line_tensor, onehot_line_tensor, rnn, config, state, optimizer, log_outputs, tracer)

def positional_encoding(pos_dim, device, max_len=2000):
    """The [max_len, pos_dim] sinusoidal encoding added to each step's input
    (--positional_encoding_dim; None when 0)."""
    if pos_dim <= 0:
        return None
    pe_matrix = torch.zeros(max_len, pos_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * (-math.log(10000.0) / pos_dim))
    pe_matrix[:, 0::2] = torch.sin(position * div_term)
    pe_matrix[:, 1::2] = torch.cos(position * div_term)
    return pe_matrix.to(device)


def build_optimizer(model, config):
    """The optimizer for backprop and BPTT. Only the SimpleRNN baseline steps through it; the
    ephemeral model's slow and fast weights take manual steps, and its SGD object is unused."""
    if config.get('optimizer', 'sgd') == 'adam':
        return torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    return torch.optim.SGD(model.parameters(), lr=config['learning_rate'])


def build_model(config, charset, n_characters):
    """The model a run's config describes (train.py's own config, or a checkpoint's)."""
    if config['input_mode'] not in ('last_one', 'last_two'):
        raise ValueError(f"Invalid input_mode: {config['input_mode']}")
    char_input_dim = n_characters * (2 if config['input_mode'] == 'last_two' else 1)
    input_size = char_input_dim + config['positional_encoding_dim']
    output_size = n_characters
    print(f"Model Input Size: {input_size}, Hidden Size: {config['n_hidden']}, Output Size: {output_size}")
    updater = config['updater']
    if config['model_type'] == 'rnn':
        print(f"Initializing SimpleRNN model with '{updater}' updater.")
        return SimpleRNN(input_size, config["n_hidden"], output_size, config["n_layers"],
                         dropout_rate=0, enable_recurrence=config['enable_recurrence'], updater=updater,
                         residual_connection=config['residual_connection'],
                         weight_clamp=config['weight_clamp'],
                         slow_weight_decay=config['slow_weight_decay'], output_tanh=config['output_tanh'],
                         layer_norm=config['layer_norm'], dfa_fprime=config.get('dfa_fprime', False))
    if config['model_type'] == 'ephemeral':
        print(f"Initializing EphemeralRNN model with '{updater}' updater.")
        return EphemeralRNN(
            input_size, config["n_hidden"], output_size, config["n_layers"], charset,
            residual_connection=config['residual_connection'],
            weight_clamp=config['weight_clamp'], updater=updater,
            plasticity=config["plasticity"], batch_size=config["batch_size"],
            forget_rate=config["forget_rate"], ephemeral_fraction=config["ephemeral_fraction"],
            enable_recurrence=config['enable_recurrence'],
            retain_sequence_bias_grads=config['grad_norm_clip'] > 0 and updater != 'dfa',
            slow_weight_decay=config['slow_weight_decay'], output_tanh=config['output_tanh'],
            fast_weight_clamp=config['fast_weight_clamp'], layer_norm=config['layer_norm'],
            dfa_fprime=config.get('dfa_fprime', False),
            slow_update_every=config.get('slow_update_every', 1),
            fast_backward_per_forward=config.get('fast_backward_per_forward', 1),
            readout_nlms=config.get('readout_nlms', False))
    raise ValueError(f"Unknown model_type: {config['model_type']}")


class _StoreWithAlias(argparse.Action):
    """Stores the value like 'store'. An action whose option strings are listed in `deprecated` is
    an old name for `canonical`: it still works and prints a one-line deprecation note. Giving both
    the new and an old name with different values is an error."""

    def __init__(self, option_strings, dest, deprecated=(), canonical=None, **kwargs):
        self.deprecated = tuple(deprecated)
        self.canonical = canonical or option_strings[0]
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.deprecated:
            print(f"DEPRECATED: {option_string} is now {self.canonical} (same meaning); the old name still works.")
        given = namespace.__dict__.setdefault('_given_flags', {})
        previous = given.get(self.dest)
        if previous is not None and previous != option_string and getattr(namespace, self.dest) != values:
            parser.error(f"{previous} and {option_string} are the same setting but were given different values.")
        given[self.dest] = option_string
        setattr(namespace, self.dest, values)


class _RemovedFlag(argparse.Action):
    """A removed boolean flag: false (what every run used) is accepted with a note, so old run
    scripts still run; true is an error, since the behaviour no longer exists. Stores nothing."""

    def __init__(self, option_strings, dest, reason='', **kwargs):
        self.reason = reason
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values:
            parser.error(f"{option_string} was removed: {self.reason} Drop the flag (false was the "
                         "default and is still accepted).")
        print(f"DEPRECATED: {option_string} was removed; false is its only setting, so it is ignored. Remove it.")


class _IgnoredFlag(argparse.Action):
    """Accepts a removed flag and its value so old run scripts still run, and prints a note."""

    def __call__(self, parser, namespace, values, option_string=None):
        print(f"DEPRECATED: {option_string} was unused and is now ignored; remove it.")


# Old flag names that became aliases, kept so old scripts and frozen checkpoints/<run>/run_used.sh
# copies keep running. --grad_clip is resolved in resolve_deprecated_args.
DEPRECATED_FLAG_ALIASES = {
    '--plasticity': ['--plast_clip'],
    '--ephemeral_fraction': ['--plast_proportion'],
    '--weight_clamp': ['--clip_weights'],
}
IGNORED_FLAGS = ('--plast_learning_rate', '--imprint_rate')
# Removed boolean flags (_RemovedFlag), with why; --normalize is the old name of --unit_norm_weights.
REMOVED_FLAGS = {
    ('--unit_norm_weights', '--normalize'):
        "it divided each weight slice by its whole L2 norm after every update, which leaves entries "
        "around 1e-3 and removes the memory (2 of 4,701 archive runs used it). For a normalization "
        "use --layer_norm.",
}


def _add_argument(parser, name, **kwargs):
    """Adds `name`, and each of its old names as a separate argument hidden from --help and usage
    (README "Renamed flags" is the reference for them). An old name stores into the same dest, so
    it behaves exactly like the new one apart from the deprecation note."""
    parser.add_argument(name, action=_StoreWithAlias, **kwargs)
    dest = name.lstrip('-')
    alias_kwargs = {key: value for key, value in kwargs.items() if key in ('type', 'nargs', 'const', 'choices')}
    for alias in DEPRECATED_FLAG_ALIASES.get(name, []):
        # default=SUPPRESS: the default comes from the new name's argument only.
        parser.add_argument(alias, dest=dest, action=_StoreWithAlias, deprecated=(alias,), canonical=name,
                            default=argparse.SUPPRESS, help=argparse.SUPPRESS, **alias_kwargs)


def build_parser():
    # Defaults match the configuration the run scripts actually use; the model hyperparameters
    # (lr, plasticity, forget_rate, hidden_size, ephemeral_fraction, dataset) are the bench_sweep point
    # that solves 3-char palindromes without recurrence.
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='rnn baseline under backprop or bptt: torch.optim.SGD or torch.optim.Adam '
                             '(default betas and eps) at --learning_rate. The ephemeral model and DFA '
                             'take manual per-sequence steps and support only sgd (see README).')
    _add_argument(parser, '--plasticity', type=float, default=1e5,
                  help='Plasticity alpha: the learning-rate multiplier on the ephemeral weights (slow weights have 1).')
    for flag in IGNORED_FLAGS:
        parser.add_argument(flag, action=_IgnoredFlag, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--forget_rate', type=float, default=0.01, help='Fraction of each ephemeral weight removed per step (w <- (1 - forget_rate) w).')
    parser.add_argument('--checkpoint_save_freq', type=int, default=10000,
                        help='How often to save a checkpoint (in iterations).')
    parser.add_argument('--residual_connection', type=str2bool, nargs='?', const=True, default=False, help='whether to have a skip connection')
    _add_argument(parser, '--ephemeral_update_clamp', type=float, default=0,
                  help='EphemeralRNN (DFA and backprop): clamp each alpha-scaled update of an ephemeral '
                       'weight to [-v, v] (0 = off). Ignored by BPTT and by the rnn baseline (no ephemeral weights).')
    _add_argument(parser, '--grad_norm_clip', type=float, default=0,
                  help='Gradient-norm clipping before each update, under every updater (0 = off). '
                       'rnn: clip_grad_norm_ on all parameters. ephemeral: each sequence\'s raw '
                       'gradient (its weight copies and bias shares), before plasticity scaling.')
    parser.add_argument('--slow_weight_decay', type=float, default=0,
                        help='Fraction of every slow weight removed after each update (0 = off): the '
                             'ephemeral model decays its slow entries in the forget step; SimpleRNN '
                             'decays all its weights. Biases are excluded.')
    parser.add_argument('--fast_weight_clamp', type=float, default=0,
                        help='EphemeralRNN: clamp only the ephemeral (fast) entries to [-v, v] after each '
                             'update, after --weight_clamp (0 = off). Ignored by the rnn baseline, which '
                             'has no fast entries.')
    parser.add_argument('--wipe_every', type=int, default=1,
                        help='EphemeralRNN: zero the fast entries at the start of every N-th sequence only '
                             '(1 = every sequence, the default). The slow entries are still averaged over '
                             'the batch every sequence; between wipes each batch row\'s fast entries carry '
                             'into its next sequence, erased only by --forget_rate (and --fast_weight_clamp). '
                             'The hidden state still starts at zero every sequence, and held-out evaluation '
                             'always starts from wiped fast entries.')
    parser.add_argument('--output_tanh', type=str2bool, nargs='?', const=True, default=False,
                        help='Both models: the output head i2o reads tanh of the shared trunk instead of '
                             'the trunk (removed from the default on 2026-09-24; see '
                             'docs/tapped_vs_forked_rnn_report.md).')
    parser.add_argument('--layer_norm', type=str2bool, nargs='?', const=True, default=False,
                        help='Both models: LayerNorm (no learnable gain or bias) on each trunk layer\'s '
                             'activations, after the GELU, so the next layer (and its DFA input trace) '
                             'reads the normalized features. The input and the recurrent state are not '
                             'normalized.')
    parser.add_argument('--readout_nlms', type=str2bool, nargs='?', const=True, default=False,
                        help='Ephemeral + DFA: NLMS-normalize each sequence\'s i2o slow update by '
                             '||x_out||^2 / width. This removes activation magnitude from the readout\'s '
                             'effective step while preserving its nominal scale at unit variance.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Ephemeral + DFA: train toward target*(1-eps) + eps/V, giving cross-entropy '
                             'a finite logit optimum (0 = off).')
    parser.add_argument('--dfa_fprime', type=str2bool, nargs='?', const=True, default=False,
                        help='DFA only, both models: multiply each non-output layer\'s projected error by '
                             'its activation derivative at the current pre-activation (Nokland 2016): '
                             'gelu\' for the trunk layers, tanh\' for i2h; i2o keeps the raw error.')
    parser.add_argument('--fused_update', type=str2bool, nargs='?', const=True, default=False,
                        help='Ephemeral + DFA only: compile each layer\'s DFA update, clamps and forgetting '
                             'into one kernel (torch.compile). The same math with different rounding, about '
                             '1e-7 relative per step. Needs compute capability 7.0+; a P100 falls back '
                             'to the unfused step. Norm-logging steps always run unfused.')
    parser.add_argument('--fast_backward_per_forward', type=parse_fast_backward_per_forward, default=1,
                        help='Ephemeral + DFA: backward (DFA) passes per forward pass for the FAST entries. '
                             '1 = one per character (unchanged). K >= 2: after the usual step, K-1 more '
                             'times re-run the forward pass on the same character with the updated fast '
                             'weights and take a fast-only DFA step (forgetting still once per character; '
                             'the slow stream, loss and metrics are those of the first pass). 1/N: only '
                             'every N-th character gets a fast update (forgetting every character). '
                             'Held-out evaluation stays 1:1. See README "Backward passes per forward".')
    parser.add_argument('--slow_update_every', type=parse_slow_update_every, default=1,
                        help='Ephemeral + DFA: how often the slow parameters (slow entries, i2o, biases) '
                             'take their DFA step, in steps (characters), or "sequence". 1 = every step '
                             '(the per-step path, unchanged). N > 1: the fast entries still change every '
                             'step; each sequence\'s slow gradients are summed over N steps and applied '
                             'together (and at the end of each sequence). sequence: slow parameters are '
                             'frozen within a sequence, and the batch mean of the per-sequence sums is '
                             'applied at its end. See README "Update rate of the slow weights".')
    parser.add_argument('--early_stop_window', type=int, default=10,
                        help='Stop when the interval loss has been > 5 for this many consecutive print_freq '
                             'intervals (default 10; 0 = never stop on loss; a NaN or inf loss always stops).')
    parser.add_argument('--plasticity_schedule', type=str, default='',
                        help='Ephemeral: "ITER:VALUE,ITER:VALUE,..." sets the plasticity alpha to VALUE from '
                             'iteration ITER on (piecewise constant, applied at iteration boundaries); '
                             '--plasticity holds before the first entry (use 0:VALUE to replace it). On resume '
                             'the value in force at the resumed iteration is applied, and a different schedule '
                             'than the checkpoint\'s is allowed. The active alpha is logged each interval. '
                             'Default empty = off.')
    parser.add_argument('--trace_loop_every', type=int, default=0,
                        help='Ephemeral + DFA: every N iterations (a multiple of --print_freq; 0 = off) record '
                             'the per-step, within-sequence feedback-loop traces of that iteration\'s batch '
                             '(trunk activation norm, fast-weight norm, max logit, the per-step loop gain, ...; '
                             'see loop_trace.py), log their summaries with the interval metrics, and write the '
                             'arrays to <checkpoint_dir>/traces/trace_<iter>.pt. Observation only: training is '
                             'bit-identical with it on or off.')
    parser.add_argument('--checkpoint_keep_every', type=int, default=0,
                        help='Also keep a numbered copy checkpoint_<iter>.pth every N iterations (0 = off; '
                             'latest_checkpoint.pth is saved as before). trace_replay.py reads them.')
    parser.add_argument('--checkpoint_keep_max', type=int, default=0,
                        help='With --checkpoint_keep_every: keep only the newest M numbered copies (0 = all).')
    parser.add_argument('--heldout_eval_every', type=int, default=0,
                        help='Ephemeral + DFA, synthetic datasets: every N iterations, evaluate '
                             '--heldout_batches batches of the validation split with the slow weights '
                             'frozen, under the observed, strict, no_fast and free_running protocols (heldout.py), '
                             'logged as heldout_<protocol>/<metric> with the next interval (0 = off). '
                             'The training run itself is unchanged.')
    parser.add_argument('--heldout_batches', type=int, default=4,
                        help='Validation batches per --heldout_eval_every evaluation (0 = the whole split).')
    # Old name for whichever of the two applies to --model_type; see resolve_deprecated_args.
    parser.add_argument('--grad_clip', type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--hidden_size', type=int, default=1024, help='Size of hidden layers in RNN')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in RNN')
    parser.add_argument('--n_iters', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--print_freq', type=int, default=50, help='Frequency of printing training progress')
    parser.add_argument('--model_type', type=str, default='ephemeral', choices=['rnn', 'ephemeral'], help='Model architecture to use.')
    parser.add_argument('--updater', type=str, default='dfa', choices=['dfa', 'backprop', 'bptt'], help='Weight update algorithm to use.')
    for flags, reason in REMOVED_FLAGS.items():
        parser.add_argument(*flags, action=_RemovedFlag, reason=reason, type=str2bool, nargs='?', const=True,
                            default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    _add_argument(parser, '--weight_clamp', type=float, default=0,
                  help='Clamp the weights to [-v, v] after each update (0 = off).')
    parser.add_argument('--track', type=str2bool, nargs='?', const=True, default=True, help='Whether to track progress online.')
    parser.add_argument('--dataset', type=str, default='3_palindrome_dataset_vary_length', help='The dataset used for training.')
    parser.add_argument('--notes', type=str, default='nothing to say', help='talk about this run')
    parser.add_argument('--wandb_project', type=str, default='ephemeral-weights', help='W&B project to log to.')
    parser.add_argument('--group', type=str, default="nothing_in_particular", help='Description of what sort of experiment is being run, here.')
    parser.add_argument('--tags', nargs='*', default=[], help="List of tags for WandB")
    parser.add_argument('--batch_size', type=int, default=16, help='how much to stuff in at once')
    parser.add_argument('--positional_encoding_dim', type=int, default=0,
                        help='Dimension for optional positional encoding (0 means off).')
    parser.add_argument('--input_mode', type=str, default='last_one', choices=['last_one', 'last_two'],
                        help='Input mode: use last one or last two characters.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints.')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Resume from this checkpoint (always resumes; errors if missing).')
    _add_argument(parser, '--ephemeral_fraction', type=float, default=0.1,
                  help='Fraction of each hidden layer\'s and i2h\'s weights that are ephemeral (i2o has none).')
    parser.add_argument('--enable_recurrence', type=str2bool, nargs='?', const=True, default=False, help='Whether to enable recurrent hidden state connections')
    parser.add_argument('--log_freq', type=int, default=None, help='Frequency for W&B sync triggers (overrides LOG_FREQ environment variable)')
    parser.add_argument('--resume', type=str2bool, nargs='?', const=True, default=False, help='Resume from <checkpoint_dir>/latest_checkpoint.pth if it exists.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed Python, NumPy, Torch, and data loading (unset = drawn from the OS on a fresh run, '
                             'read from the checkpoint on resume).')
    parser.add_argument('--resume_reseed', type=int, default=None,
                        help='Resume only: after loading the checkpoint, reseed the RNGs with this value and restart '
                             'the data stream from the first epoch of a loader built with it, so a replicate of an '
                             'intervention sees a different data order and noise (default off: exact continuation).')
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=None,
                        help='Require deterministic Torch operations (unset = off on a fresh run, '
                             'the checkpoint\'s value on resume).')
    return parser


def resolve_deprecated_args(args, parser):
    """Maps --grad_clip to the flag that applies to --model_type, as the old code did: the ephemeral
    model used it only as the element-wise update clamp, the rnn baseline only as clip_grad_norm_."""
    given = vars(args).pop('_given_flags', {})
    if 'grad_clip' in vars(args):
        grad_clip = vars(args).pop('grad_clip')
        target = 'ephemeral_update_clamp' if args.model_type == 'ephemeral' else 'grad_norm_clip'
        if target in given and getattr(args, target) != grad_clip:
            parser.error(f"--grad_clip {grad_clip} conflicts with --{target} {getattr(args, target)}.")
        setattr(args, target, grad_clip)
        print(f"DEPRECATED: --grad_clip is now --ephemeral_update_clamp (ephemeral) or --grad_norm_clip (rnn); "
              f"with --model_type {args.model_type} it sets --{target}.")
    return args


def check_argument_combinations(args, parser):
    if args.fast_backward_per_forward != 1 and (args.model_type != 'ephemeral' or args.updater != 'dfa'):
        parser.error("--fast_backward_per_forward other than 1 supports only --model_type ephemeral --updater dfa.")
    if args.slow_update_every != 1 and (args.model_type != 'ephemeral' or args.updater != 'dfa'):
        parser.error("--slow_update_every other than 1 supports only --model_type ephemeral --updater dfa.")
    if args.fused_update and (args.model_type != 'ephemeral' or args.updater != 'dfa'):
        parser.error("--fused_update supports only --model_type ephemeral --updater dfa.")
    if args.optimizer != 'sgd' and (args.model_type != 'rnn' or args.updater == 'dfa'):
        parser.error(f"--optimizer {args.optimizer} supports only --model_type rnn with --updater backprop "
                     "or bptt: the ephemeral model and DFA update their weights by hand, per sequence.")
    if args.early_stop_window < 0:
        parser.error("--early_stop_window must not be negative.")
    try:
        parse_plasticity_schedule(args.plasticity_schedule)
    except ValueError as exc:
        parser.error(str(exc))
    if args.plasticity_schedule and args.model_type != "ephemeral":
        parser.error("--plasticity_schedule needs --model_type ephemeral.")
    if args.trace_loop_every < 0 or args.checkpoint_keep_every < 0 or args.checkpoint_keep_max < 0:
        parser.error("--trace_loop_every, --checkpoint_keep_every and --checkpoint_keep_max must not be negative.")
    if args.trace_loop_every > 0:
        if args.model_type != 'ephemeral' or args.updater != 'dfa':
            parser.error("--trace_loop_every supports only --model_type ephemeral --updater dfa.")
        if args.print_freq <= 0 or args.trace_loop_every % args.print_freq != 0:
            parser.error("--trace_loop_every must be a multiple of --print_freq (the traces are logged "
                         "with the interval metrics).")
    if args.wipe_every < 1:
        parser.error("--wipe_every must be at least 1.")
    if args.wipe_every > 1 and args.model_type != 'ephemeral':
        parser.error("--wipe_every > 1 needs --model_type ephemeral (SimpleRNN has no fast weights).")
    if args.dfa_fprime and args.updater != 'dfa':
        parser.error("--dfa_fprime applies only to --updater dfa.")
    interventions = args.readout_nlms or args.label_smoothing != 0
    if interventions and (args.model_type != 'ephemeral' or args.updater != 'dfa'):
        parser.error("--readout_nlms and --label_smoothing support only "
                     "--model_type ephemeral --updater dfa.")
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label_smoothing must be in [0, 1).")
    if args.label_smoothing and (args.fast_backward_per_forward != 1 or args.heldout_eval_every > 0):
        parser.error("--label_smoothing does not support --fast_backward_per_forward other than 1 "
                     "or --heldout_eval_every; those auxiliary passes use the plain error.")
    if args.heldout_eval_every > 0 and (args.model_type != 'ephemeral' or args.updater != 'dfa'):
        parser.error("--heldout_eval_every supports only --model_type ephemeral --updater dfa "
                     "(see EphemeralRNN.check_fast_only_step).")
    return args


def parse_args(argv=None):
    parser = build_parser()
    return check_argument_combinations(resolve_deprecated_args(parser.parse_args(argv), parser), parser)


def main():
    # grab slurm jobid if it exists.
    job_id = os.environ.get("SLURM_JOB_ID") if os.environ.get("SLURM_JOB_ID") else "no_SLURM"
    print("SLURM Job ID:", job_id)
    
    parser = build_parser()
    args = check_argument_combinations(resolve_deprecated_args(parser.parse_args(), parser), parser)

    # Define the path to the latest checkpoint
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")

    # An explicit --resume_checkpoint always resumes; --resume picks up latest_checkpoint.pth if present.
    checkpoint_to_load = None
    if args.resume_checkpoint:
        if not os.path.isfile(args.resume_checkpoint):
            raise FileNotFoundError(f"Explicit resume checkpoint not found: {args.resume_checkpoint}")
        checkpoint_to_load = args.resume_checkpoint
        print(f"Attempting to resume from explicit checkpoint: {checkpoint_to_load}")
    elif args.resume and os.path.isfile(latest_checkpoint_path):
        checkpoint_to_load = latest_checkpoint_path
        print(f"Found latest checkpoint. Attempting to resume from: {checkpoint_to_load}")
    elif args.resume:
        print(f"--resume given but no checkpoint at {latest_checkpoint_path}. Starting from scratch.")
    else:
        print("Starting from scratch (pass --resume or --resume_checkpoint to resume).")
    # Read the checkpoint before anything consumes randomness: it decides the seed.
    checkpoint = read_checkpoint(checkpoint_to_load) if checkpoint_to_load else None
    if checkpoint is not None:
        # Refuse a checkpoint from other training mechanics before it decides anything.
        check_checkpoint_code_version(checkpoint, checkpoint_to_load)

    try:
        seed, deterministic, seed_source = resolve_seed(args.seed, args.deterministic, checkpoint)
        seed_everything(seed, deterministic=deterministic)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Seed: {seed} ({seed_source}), deterministic: {deterministic}")
    if seed is None:
        print("WARNING: resumed legacy unseeded run: no seed is set; the checkpoint's RNG states are "
              "restored, but data order does not continue exactly.")
    record_seed_in_slurm(seed)
    
    # Set log_freq: command line arg takes precedence over environment variable
    if args.log_freq is not None:
        log_freq = args.log_freq
        print(f"Log frequency set from command line: {log_freq}")
    else:
        log_freq = int(os.getenv("LOG_FREQ", "5000"))
        print(f"Log frequency set from environment (LOG_FREQ): {log_freq}")

    config = {
        "learning_rate": args.learning_rate,
        "plasticity": args.plasticity,
        "forget_rate": args.forget_rate,
        "checkpoint_save_freq": args.checkpoint_save_freq,
        # Use 'mean' for backprop, will be overridden to 'none' in train() for the ephemeral model
        "criterion": torch.nn.CrossEntropyLoss(reduction='mean'),
        "residual_connection": args.residual_connection,
        "ephemeral_update_clamp": args.ephemeral_update_clamp,
        "grad_norm_clip": args.grad_norm_clip,
        "n_hidden": args.hidden_size,
        "n_layers": args.num_layers,
        "track": args.track,
        "dataset": args.dataset,
        "model_type": args.model_type,
        "updater": args.updater,
        "batch_size": args.batch_size,
        "input_mode": args.input_mode,
        "ephemeral_fraction": args.ephemeral_fraction,
        "enable_recurrence": args.enable_recurrence,
        "seed": seed,
        "deterministic": deterministic,
    }
    # Record every other CLI argument too, so checkpoints carry it and a resume can diff it.
    config.update({key: value for key, value in vars(args).items() if key not in config})
    print(f"Input mode selected: {args.input_mode}") # Inform user

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True) # exist_ok=True for robustness

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    charset, char_to_idx, idx_to_char, n_characters = initialize_charset(args.dataset)
    config["charset_size"] = n_characters         # ← add this line
    print(f"Character set size: {n_characters}")

    # Use drop_last=True if batch size doesn't divide dataset size evenly
    dataloader = load_and_preprocess_data(
        args.dataset, args.batch_size, drop_last=True, seed=seed
    )

    config["pe_matrix"] = positional_encoding(args.positional_encoding_dim, device)
    optimizer = None
    start_iter = 1
    rnn = build_model(config, charset, n_characters)

    # Optimizer is only needed for backprop and bptt (regardless of model type)
    if args.updater in ['backprop', 'bptt']:
        optimizer = build_optimizer(rnn, config)

    state = {
        "training_instance": 0,
        "last_n_rewards": [0],
        "last_n_reward_avg": 0,
        "wandb_step": 0,  # Initialize wandb_step
        "log_norms_now": False,
    }

    # --- Resume from Checkpoint ---
    # The model and data stream are built (consuming seeded randomness exactly as the original
    # run did) before load_checkpoint overwrites the weights and restores the RNG states.
    data_stream = DataStream(dataloader)
    if checkpoint_to_load:
        # Any load failure aborts the run: silently restarting from scratch hides
        # the failure and mixes fresh weights into a "resumed" experiment.
        rnn, optimizer, start_iter, loaded_main_state, loaded_config = load_checkpoint(
            checkpoint_to_load, rnn, config, optimizer=optimizer, device=device, checkpoint=checkpoint
        )
        if not data_stream.load_state_dict(checkpoint.get("data_stream_state")) and seed is not None:
            print("WARNING: checkpoint has no data-stream position; data order restarts from the seed's first epoch.")
        checkpoint = None  # everything needed has been copied out; free the CPU copy
        state.update(loaded_main_state) # Update your main program state
        print(f"resumed, starting from iter: {start_iter}")
        if args.resume_reseed is not None:
            seed_everything(args.resume_reseed, deterministic=deterministic)
            dataloader = load_and_preprocess_data(
                args.dataset, args.batch_size, drop_last=True, seed=args.resume_reseed
            )
            data_stream = DataStream(dataloader)
            print(f"--resume_reseed {args.resume_reseed}: RNGs reseeded and the data stream restarted (not an exact continuation).")

        # Check if --plasticity has changed and update plasticity parameters if needed
        if isinstance(rnn, EphemeralRNN):
            # load_checkpoint maps an old checkpoint's plast_clip to plasticity.
            loaded_plasticity = loaded_config.get('plasticity', 1.0)
            current_plasticity = config.get('plasticity', 1.0)

            if loaded_plasticity != current_plasticity:
                print(f"Plasticity changed from {loaded_plasticity} to {current_plasticity}")
                print("Updating plasticity parameters in all layers...")
                rnn.set_plasticity(current_plasticity)
                print("Plasticity parameters updated successfully!")
            else:
                print(f"Plasticity unchanged: {current_plasticity}")

    elif torch.cuda.is_available(): # No checkpoint_to_load specified AT ALL, and cuda is available
        print("No checkpoint specified for loading. Moving model to GPU.")
        rnn = rnn.to(device)
    # else: model remains on CPU if no checkpoint and no CUDA

    # --fused_update compiles the DFA step with Triton, which needs compute capability 7.0.
    config["fused_update_active"] = False
    if args.fused_update:
        weights_device = next(rnn.parameters()).device
        if weights_device.type == "cuda" and torch.cuda.get_device_capability(weights_device)[0] < 7:
            print(f"--fused_update: {torch.cuda.get_device_name(weights_device)} is below compute "
                  "capability 7.0, which Triton needs; using the unfused step.")
        else:
            rnn.enable_fused_update()
            config["fused_update_active"] = True
            print("--fused_update: the DFA step is compiled per layer (first steps include compilation).")

    heldout_batches, heldout_metrics = None, {}
    tracer, trace_metrics = None, {}
    if args.trace_loop_every > 0:
        tracer = LoopTracer(rnn, args.learning_rate, args.ephemeral_update_clamp, args.grad_norm_clip)
        os.makedirs(os.path.join(args.checkpoint_dir, "traces"), exist_ok=True)
        print(f"Loop traces every {args.trace_loop_every} iterations, written to "
              f"{os.path.join(args.checkpoint_dir, 'traces')}.")
    if args.heldout_eval_every > 0:
        heldout_batches = load_heldout_batches(args.dataset, args.batch_size, args.heldout_batches,
                                               next(rnn.parameters()).device)
        print(f"Held-out evaluation every {args.heldout_eval_every} iterations on "
              f"{len(heldout_batches)} validation batches.")

    if args.track:
        # wandb initialization
        wandb_config = {
            "learning_rate": args.learning_rate,
            "plasticity": args.plasticity,
            "nominal_mean_lr": args.learning_rate * (1-args.ephemeral_fraction + args.ephemeral_fraction * args.plasticity),
            "nominal_ephemeral_lr": args.learning_rate * args.plasticity,
            "architecture": args.model_type,
            "updater": args.updater,
            "residual_connection": args.residual_connection,
            "ephemeral_update_clamp": args.ephemeral_update_clamp,
            "grad_norm_clip": args.grad_norm_clip,
            "fused_update": args.fused_update,
            "slow_update_every": args.slow_update_every,
            "fast_backward_per_forward": args.fast_backward_per_forward,
            "dfa_fprime": args.dfa_fprime,
            "slow_weight_decay": args.slow_weight_decay,
            "output_tanh": args.output_tanh,
            "layer_norm": args.layer_norm,
            "fast_weight_clamp": args.fast_weight_clamp,
            "wipe_every": args.wipe_every,
            "fused_update_active": config["fused_update_active"],
            "n_hidden": args.hidden_size,
            "n_layers": args.num_layers,
            "dataset": args.dataset,
            "epochs": 1, # This seems fixed, maybe adjust?
            "forget_rate": args.forget_rate,
            "weight_clamp": args.weight_clamp,
            "log_freq": log_freq,
            "batch_size": args.batch_size,
            "slurm_id": job_id,
            "positional_encoding_dim": args.positional_encoding_dim,
            "checkpoint_save_freq": args.checkpoint_save_freq,
            "input_mode": args.input_mode,
            "ephemeral_fraction": args.ephemeral_fraction,
            "enable_recurrence": args.enable_recurrence,
            "seed": seed,
            "seed_source": seed_source,
            "deterministic": deterministic,
            "recall_chance": recall_chance(args.dataset),
        }
        # A resumed checkpoint always starts a new W&B run; record where it came from.
        if checkpoint_to_load:
            wandb_config["resumed_from_checkpoint"] = checkpoint_to_load
            wandb_config["resumed_at_iter"] = start_iter
            print("Resuming a checkpoint: starting a new W&B run (W&B run resumption is not supported).")
        print(f"tags given to wandb: {args.tags}")
        wandb.init(project=args.wandb_project,
                group=args.group,
                notes=args.notes,
                tags=args.tags,
                config=wandb_config,
                )
        print(f"Initialized WandB with Run ID: {wandb.run.id}")


    # Training Loop
    start = time.time()
    
    # Flag to track if NaN has been detected
    nan_detected = False
    
    # Early stopping for high loss values - sliding window tracking
    EARLY_STOP_LOSS_THRESHOLD = 5.0
    EARLY_STOP_WINDOW_SIZE = args.early_stop_window
    plasticity_schedule = parse_plasticity_schedule(args.plasticity_schedule)
    if plasticity_schedule and args.fused_update:
        # Each alpha is a new constant for the compiled step (per layer shape); the default cache limit of 8
        # would silently fall back to the eager step after a few changes.
        limit = compile_cache_limit(plasticity_schedule)
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, limit)
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit, 2 * limit)
    loss_window = []  # Sliding window of average losses from print_freq intervals
    high_loss_count = 0  # Count of consecutive intervals with loss > threshold
    early_stopped = False

    # SLURM sends SIGUSR1 shortly before the wall-time limit (#SBATCH --signal=B:USR1@600, forwarded by
    # forward_signals.sh) and SIGTERM at the limit. Stop cleanly at the next iteration boundary.
    stop_signals = []
    def request_stop(signum, frame):
        stop_signals.append(signum)
    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    stopped_by_signal = None

    def checkpoint_state(next_iter):
        return {
            'iter': next_iter,
            'code_version': CHECKPOINT_CODE_VERSION,
            'model_state_dict': rnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'main_program_state': state,
            'config': config,
            'data_stream_state': data_stream.state_dict(),
            **capture_rng_state(),
        }

    try:
        # Metrics accumulated over each print_freq interval (whole batch, every step)
        interval = IntervalMetrics(args.dataset)
        interval_start = time.time()
        
        # These will be populated per iteration if log_outputs_for_train is true
        # and then cleared after print_freq, as per your original logic.
        all_outputs_for_print_freq = [] 
        all_labels_for_print_freq = []

        for iter in range(start_iter, args.n_iters + 1):
            if stop_signals:
                stopped_by_signal = signal.Signals(stop_signals[0])
                print(f"Received {stopped_by_signal.name}: stopping before iteration {iter}.")
                if args.checkpoint_save_freq > 0:
                    save_checkpoint(checkpoint_state(iter), args.checkpoint_dir, "latest_checkpoint.pth")
                else:
                    print("Checkpointing disabled (checkpoint_save_freq=0). No checkpoint saved.")
                if stopped_by_signal == signal.SIGUSR1:
                    wb_mark_end("time_limit", tags=["end:time_limit"], exit_code=124)
                else:
                    wb_mark_end("terminated", tags=["end:terminated"], exit_code=143)
                break

            if plasticity_schedule:
                scheduled = plasticity_at(plasticity_schedule, iter, args.plasticity)
                if scheduled != config["plasticity"]:
                    print(f"Plasticity schedule: alpha {config['plasticity']} -> {scheduled} at iter {iter}")
                    rnn.set_plasticity(scheduled)
                    config["plasticity"] = scheduled  # checkpoints record the alpha in force

            # Fetch next batch (the stream is endless and tracks its own position)
            sequence, line_tensor, onehot_line_tensor = next(data_stream)

            line_tensor = line_tensor.to(device)
            onehot_line_tensor = onehot_line_tensor.to(device)

            # Ensure batch size matches model expectation if using EphemeralRNN with fixed batch size param
            if args.updater != "backprop" and hasattr(rnn, 'batch_size') and onehot_line_tensor.shape[0] != rnn.batch_size:
                 print(f"Warning: Batch size mismatch ({onehot_line_tensor.shape[0]} vs {rnn.batch_size}). Skipping batch.")
                 continue # Skip this batch

            # Determine if detailed outputs are needed (for the frequent PRINT interval)
            log_outputs_for_train = (iter % args.print_freq == 0)

            # --- Train Step ---
            state["log_norms_now"] = (iter % args.print_freq == 0)
            # The train function returns step-by-step outputs for the first batch item if log_outputs=True
            tracing_now = tracer is not None and iter % args.trace_loop_every == 0
            output, loss, step_preds, step_losses, current_iter_all_outputs, current_iter_all_labels = train(
                line_tensor, onehot_line_tensor, rnn, config, state, optimizer, log_outputs=log_outputs_for_train,
                tracer=tracer if tracing_now else None
            )
            if tracing_now:
                traces = tracer.finish()
                torch.save({"iter": iter, "traces": traces},
                           os.path.join(args.checkpoint_dir, "traces", f"trace_{iter:08d}.pt"))
                trace_metrics = summarize_traces(traces)
            
            # Terminate on a non-finite loss (NaN or inf; isnan alone misses inf)
            if not math.isfinite(loss):
                print(f"Non-finite loss ({loss}) detected. Terminating training.")
                nan_detected = True
                wb_mark_end("nan_detected", tags=["end:nan", "NaN"], exit_code=1)
                break  # Exit the training loop
            
            # Store detailed outputs if they were generated FOR THIS ITERATION for print_freq
            if log_outputs_for_train:
                all_outputs_for_print_freq = current_iter_all_outputs
                all_labels_for_print_freq = current_iter_all_labels

            interval.update(sequence, onehot_line_tensor, step_preds, step_losses)

            if heldout_batches is not None and iter % args.heldout_eval_every == 0:
                # Logged with the next interval; evaluate_protocols restores the model's state.
                heldout_metrics = evaluate_protocols(rnn, heldout_batches, config, args.dataset)

            # ==============================================================
            # --- Frequent Detailed Console Logging Period (print_freq) ---
            # ==============================================================
            if iter % args.print_freq == 0:
                print(f'{iter} {iter / args.n_iters * 100:.2f}% ({timeSince(start)}) InstLoss: {loss:.4f}')

                # Check if detailed outputs were generated for this iteration
                if all_outputs_for_print_freq and all_labels_for_print_freq:
                    source_char_display_list = []
                    target_display_list = []
                    pred_t1_display_list = []
                    pred_t2_display_list = []

                    num_prediction_steps = len(all_outputs_for_print_freq)
                    num_correct_t1_for_seq = 0
                    num_correct_top2_for_seq = 0 # Top-2 inclusive (T1 or T2 correct)

                    for i in range(num_prediction_steps):
                        # Input character that led to this prediction (from the first item in batch)
                        source_char_idx = line_tensor[0, i].item()
                        source_char = idx_to_char.get(source_char_idx, '?') # Ensure idx_to_char is in scope
                        source_char_display_list.append(source_char)

                        # Target character for this step
                        step_target_onehot = all_labels_for_print_freq[i]
                        actual_target_idx = torch.argmax(step_target_onehot).item()
                        actual_target_char = idx_to_char.get(actual_target_idx, '?')

                        # Predictions for this step
                        step_output_logits = all_outputs_for_print_freq[i]
                        top_val, top_idx = torch.topk(step_output_logits, 2)
                        
                        predicted_idx_t1 = top_idx[0].item()
                        predicted_char_t1 = idx_to_char.get(predicted_idx_t1, '?')

                        predicted_idx_t2 = -1
                        predicted_char_t2 = " " 
                        if len(top_idx) > 1:
                            predicted_idx_t2 = top_idx[1].item()
                            predicted_char_t2 = idx_to_char.get(predicted_idx_t2, '?')
                        
                        # 1. Target Character ("Original" in your request)
                        if actual_target_idx == predicted_idx_t1:
                            # If T1 prediction matches the actual target
                            target_display_list.append(f"{TermColors.GREEN}{actual_target_char}{TermColors.RESET}")
                        elif len(top_idx) > 1 and predicted_idx_t2 == actual_target_idx:
                            # Else, if T1 did NOT match, but T2 exists and T2 matches the actual target
                            target_display_list.append(f"{TermColors.PURPLE}{actual_target_char}{TermColors.RESET}")
                        else:
                            # Else (T1 didn't match, AND (T2 didn't exist OR T2 also didn't match))
                            target_display_list.append(f"{TermColors.WHITE}{actual_target_char}{TermColors.RESET}")
                        

                        # 2. Pred T1 Character
                        if predicted_idx_t1 == actual_target_idx:
                            pred_t1_display_list.append(f"{TermColors.GREEN}{predicted_char_t1}{TermColors.RESET}")
                            num_correct_t1_for_seq += 1
                            num_correct_top2_for_seq += 1 # If T1 is correct, Top2 is correct
                        else:
                            pred_t1_display_list.append(f"{TermColors.RESET}{predicted_char_t1}{TermColors.RESET}")
                            # Check if T2 was correct for Top2 accuracy
                            if len(top_idx) > 1 and predicted_idx_t2 == actual_target_idx:
                                num_correct_top2_for_seq += 1
                        
                        # 3. Pred T2 Character
                        if len(top_idx) > 1: # If a T2 prediction exists
                            if predicted_idx_t2 == actual_target_idx:
                                pred_t2_display_list.append(f"{TermColors.GREEN}{predicted_char_t2}{TermColors.RESET}")
                            else: # T2 exists and is incorrect
                                pred_t2_display_list.append(f"{TermColors.WHITE}{predicted_char_t2}{TermColors.RESET}")
                        else: # No distinct T2 prediction
                            pred_t2_display_list.append(f"{TermColors.PURPLE} {TermColors.RESET}") # Purple space

                    # Print the assembled strings for the first sequence in the batch
                    # print(f"  Src : {''.join(source_char_display_list)}")
                    print(f"  Trg :  {''.join(target_display_list)}")
                    # print(f"  PrT1:  {''.join(pred_t1_display_list)}")
                    # print(f"  PrT2:  {''.join(pred_t2_display_list)}")
                    
                    # Calculate and print step-wise accuracy for THIS specific displayed sequence
                    seq_step_accuracy_t1 = (num_correct_t1_for_seq / num_prediction_steps) if num_prediction_steps > 0 else 0.0
                    seq_step_accuracy_top2 = (num_correct_top2_for_seq / num_prediction_steps) if num_prediction_steps > 0 else 0.0
                    print(f'  Seq Acc: T1 {seq_step_accuracy_t1:.4f}, Top2 {seq_step_accuracy_top2:.4f}')
                    print("-" * 40) # Separator

                    # Clear the detailed output lists after use for this print_freq iteration
                    all_outputs_for_print_freq.clear()
                    all_labels_for_print_freq.clear()
                # No WandB logging directly in this very frequent print_freq block

            # ==============================================================
            # --- Averaged Console Print & WandB Log Period (print_freq) ---
            # ==============================================================
            if args.print_freq > 0 and iter % args.print_freq == 0:
                metrics = interval.summary()
                metrics.update(rnn.grad_clip_stats.summary())
                metrics.update(heldout_metrics)
                metrics.update(trace_metrics)
                heldout_metrics, trace_metrics = {}, {}
                metrics["iters_per_sec"] = interval.iterations / (time.time() - interval_start)
                if plasticity_schedule:
                    metrics["plasticity"] = config["plasticity"]
                avg_loss_plot = metrics.get("loss", float("nan"))

                print(f'--- Interval metrics (ending @ iter {iter}, whole batch) ---')
                for key, value in metrics.items():
                    print(f'  {key}: {value:.4f}')
                print(f'-------------------------------------------')

                if args.track:
                    # Calculate and gather norms
                    model_norms = rnn.get_all_norms()

                    # --- ASCII Bar Graph for Gradient/Update Norms ---
                    ephemeral_update_norms = {
                        k.replace('_ephemeral_update_norm', ''): v
                        for k, v in model_norms.items()
                        if 'ephemeral_update_norm' in k
                    }
                    slow_update_norms = {
                        k.replace('_slow_update_norm', ''): v
                        for k, v in model_norms.items()
                        if 'slow_update_norm' in k
                    }
                    plot_ascii_bar_graph(ephemeral_update_norms, "Ephemeral Update/Grad Norms")
                    plot_ascii_bar_graph(slow_update_norms, "Slow Update/Grad Norms")
                    
                    # Calculate averages for each type
                    ephemeral_weights = [v for k, v in model_norms.items() if 'ephemeral_weight_norm' in k]
                    slow_weights = [v for k, v in model_norms.items() if 'slow_weight_norm' in k]
                    ephemeral_updates = [v for k, v in model_norms.items() if 'ephemeral_update_norm' in k]
                    slow_updates = [v for k, v in model_norms.items() if 'slow_update_norm' in k]
                    # Also keep track of backprop norms if needed (check if 'grad_norm' exists)
                    grad_norms = [v for k, v in model_norms.items() if 'grad_norm' in k]
                    all_weights = [v for k, v in model_norms.items() if 'weight_norm' in k] # For backprop or combined ephemeral

                    avg_ephemeral_w_norm = sum(ephemeral_weights) / len(ephemeral_weights) if ephemeral_weights else 0.0
                    avg_slow_w_norm = sum(slow_weights) / len(slow_weights) if slow_weights else 0.0
                    avg_ephemeral_u_norm = sum(ephemeral_updates) / len(ephemeral_updates) if ephemeral_updates else 0.0
                    avg_slow_u_norm = sum(slow_updates) / len(slow_updates) if slow_updates else 0.0
                    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0 # For backprop
                    avg_weight_norm = sum(all_weights) / len(all_weights) if all_weights else 0.0 # For backprop/combined

                    log_data = {
                        "iter": iter,
                        **metrics,
                        "avg_weight_norm": avg_weight_norm, # Combined / Backprop
                        "avg_grad_update_norm": avg_grad_norm or (avg_ephemeral_u_norm + avg_slow_u_norm), # Backprop, or ephemeral + slow
                        "avg_ephemeral_weight_norm": avg_ephemeral_w_norm,
                        "avg_slow_weight_norm": avg_slow_w_norm,
                        "avg_ephemeral_update_norm": avg_ephemeral_u_norm,
                        "avg_slow_update_norm": avg_slow_u_norm,
                        # **model_norms # Log all individual norms too
                    }
                    wandb.log(log_data, step=state["wandb_step"], commit=True)
                    # print some norm data too
                    print(f'  Avg Weight Norm: {avg_weight_norm:.4f}')
                    print(f'  Avg Grad/Update Norm: {avg_grad_norm:.4f}')
                
                # Early stopping logic - check loss after logging to WandB
                loss_window.append(avg_loss_plot)
                if len(loss_window) > EARLY_STOP_WINDOW_SIZE:
                    loss_window.pop(0)  # Keep only the last EARLY_STOP_WINDOW_SIZE values
                
                # Check if current loss exceeds threshold
                high_loss_count, early_stop_now = high_loss_stop(high_loss_count, avg_loss_plot,
                                                                 EARLY_STOP_WINDOW_SIZE, EARLY_STOP_LOSS_THRESHOLD)
                if EARLY_STOP_WINDOW_SIZE > 0 and avg_loss_plot > EARLY_STOP_LOSS_THRESHOLD:
                    print(f"  High loss detected ({avg_loss_plot:.4f} > {EARLY_STOP_LOSS_THRESHOLD}). Count: {high_loss_count}/{EARLY_STOP_WINDOW_SIZE}")

                # Early stop if we've had high loss for the required number of intervals
                if early_stop_now:
                    print(f"Early stopping: Loss has been > {EARLY_STOP_LOSS_THRESHOLD} for {EARLY_STOP_WINDOW_SIZE} consecutive intervals.")
                    early_stopped = True
                    wb_mark_end("high_loss_early_stop", tags=["end:high_loss", "early_stop"], exit_code=1)
                    break  # Exit the training loop
                
                state["wandb_step"] += 1
                interval.reset()
                interval_start = time.time()

            # ==============================================================
            # --- W&B Offline Sync Trigger ---
            # ==============================================================
            is_offline = os.getenv("WANDB_MODE") == "offline"
            if args.print_freq > 0 and iter % (log_freq) == 0 and args.track and is_offline: # Trigger less often
                print("Triggering W&B sync...")
                # the following code is pointless because environment variables don't change for a running process. I'll want to do it with signal handler or file based checks. 
                # whatever I do can't slow anything down. Logging needs to be hyper lightweight.
                # if int(os.getenv("LOG_FREQ", "5000")) != log_freq:
                #     print(f"Log frequency has been updated to {log_freq} in the environment variable.")
                #     log_freq = int(os.getenv("LOG_FREQ", "5000")) # Update log_freq if changed in env
                try:
                    if trigger_sync:
                        trigger_sync()
                except Exception as e:
                    print(f"Error during W&B sync: {e}")

            # ==============================================================
            # --- Checkpointing ---
            # ==============================================================
            if args.checkpoint_save_freq > 0 and iter % args.checkpoint_save_freq == 0:
                save_checkpoint(checkpoint_state(iter + 1), args.checkpoint_dir, "latest_checkpoint.pth") # Overwrites latest
            if args.checkpoint_keep_every > 0 and iter % args.checkpoint_keep_every == 0:
                keep_numbered_checkpoint(checkpoint_state(iter + 1), args.checkpoint_dir, iter, args.checkpoint_keep_max)

        # End of training loop - mark normal completion if no early stopping occurred
        if args.track and wandb.run and not nan_detected and not early_stopped and not stopped_by_signal:
            wb_mark_end("completed", tags=["end:completed"], exit_code=0)

        # If NaN was detected, we should not log the normal completion
        if nan_detected:
            print("Training terminated due to NaN loss.")
            sys.exit(1)
        elif early_stopped:
            print("Training terminated due to high loss early stopping.")
            sys.exit(1)
        elif stopped_by_signal:
            # 124 = timeout convention (SLURM's pre-limit warning); 143 = 128 + SIGTERM
            sys.exit(124 if stopped_by_signal == signal.SIGUSR1 else 143)


    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Attempting to save final checkpoint...")
        wb_mark_end("user_interrupt", tags=["end:user_interrupt"], exit_code=130)
        # Optionally save a final checkpoint on interrupt
        if args.checkpoint_dir and args.checkpoint_save_freq > 0: # Ensure dir is specified and checkpointing is enabled
            final_checkpoint_state = checkpoint_state(iter + 1 if 'iter' in locals() else start_iter)
            save_checkpoint(final_checkpoint_state, args.checkpoint_dir, "interrupt_checkpoint.pth")
            save_checkpoint(final_checkpoint_state, args.checkpoint_dir, "latest_checkpoint.pth") # also update latest
        elif args.checkpoint_save_freq == 0:
            print("Checkpointing disabled (checkpoint_save_freq=0). No checkpoint saved on interrupt.")
        print("Finishing up...")
        sys.exit(130)  # conventional exit code for SIGINT, so wrappers don't count this as success
    except Exception:
        # Re-raise so the process exits non-zero and wrapper scripts see the failure.
        wb_mark_end("crashed", tags=["end:crashed"], exit_code=1)
        raise


    finally: # Ensure wandb finishes even on error/interrupt
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if args.track and wandb.run is not None:
            print("Finishing W&B run...")
            wandb.finish()
            print("W&B run finished.")




if __name__ == '__main__':
    main()
