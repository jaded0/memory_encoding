"""Observation-only traces of the fast <-> slow feedback loop inside a sequence.

Nothing here changes training: LoopTracer reads the model's tensors and writes its own buffers,
and the default (NULL_TRACER, or no --trace_loop_every) leaves train.train_batch exactly as it
was. See the vault note "ephemeral weights feedback-loop instrumentation design 2026-09-30".

The collection interface is these four calls, made by train.train_batch around one DFA step per
character (the only place that knows about tracing):

    tracer.begin(num_steps, batch)            once, after the sequence wipe
    tracer.before_update(step, output, output_error, loss, hidden, fast_write=True)
                                              after the forward and the loss, before any weight
                                              changes: activations, the weights that produced
                                              this output, the write this step is about to make
    tracer.after_update(step)                 right after the main update (before the extra
                                              --fast_backward_per_forward passes): what changed
    tracer.finish()                           after the sequence (and its last slow window):
                                              the only device-to-host read; returns the arrays

Static-buffer rules, so a CUDA-graph or fused rewrite can keep the interface: every buffer is a
preallocated [T, ...] device tensor written by index (no .item(), no branching on values), the
write is computed from (output_error, layer traces) by the model's own dfa_step_errors and
ephemeral_update, so it does not depend on which update path (unfused, fused, windowed,
fast-backward) runs, and only finish() synchronizes.

Per-step quantities, per batch row, "fast layers" L = the trunk layers and i2h (i2o has no fast
entries); l indexes them:
  act_norm[T,B,L]     |x_l|, the layer's input (the last is the trunk output i2h and i2o read)
  fast_norm[T,B,L]    |F_l| Frobenius, ephemeral entries only, as the forward used them
  fast_write[T,B,L]   |lr * clamp(alpha * p x^T)| on the ephemeral entries: the write of this step
                      (with --grad_norm_clip, the clipped one; 0 on a step with no fast update)
  fast_drive[T,B,L]   |(F_l) x_l|, slow_drive[T,B,L] |(S_l) x_l|: each part's share of the
                      layer's pre-activation (bias excluded)
  fast_delta[T,B,L]   |F after - F before| over the step (forgetting included)
  slow_delta[T,B,L+1] |S after - S before| over the step (i2o last); what was applied, so zero
                      between window ends under --slow_update_every N / sequence
  max_logit, logit_norm, loss, hidden_norm   [T,B]
  slow_total_delta[B,L+1]   |S at the end of the sequence - S at its start| (includes the last window)
and the derived write_norm[T,B] = sqrt(sum_l fast_write^2) and
loop_gain[T,B] = write_norm[t] / write_norm[t-1] (NaN where the previous write is zero).
"""
import math

import torch

from ephemeral_model import dfa_per_sample_gradient, ephemeral_update

PER_STEP_LAYER = ("act_norm", "fast_norm", "fast_write", "fast_drive", "slow_drive", "fast_delta")


class NullTracer:
    """The default: every call is a no-op, so the training loop needs no 'if tracing' branches."""

    def begin(self, num_steps, batch):
        pass

    def before_update(self, step, output, output_error, loss, hidden, fast_write=True):
        pass

    def after_update(self, step):
        pass

    def finish(self):
        return None


NULL_TRACER = NullTracer()


def row_norm(tensor):
    """Frobenius norm of each batch row of [B, ...], float32."""
    return torch.linalg.vector_norm(tensor.flatten(1).float(), dim=1)


class LoopTracer:
    """Collects the traces of one batch of sequences (see the module docstring). Construct once;
    begin() starts a sequence. learning_rate, update_clamp and grad_norm_clip are the training
    step's (config['learning_rate'], ['ephemeral_update_clamp'], ['grad_norm_clip'])."""

    def __init__(self, model, learning_rate, update_clamp=0, grad_norm_clip=0):
        self.model = model
        self.learning_rate = learning_rate
        self.update_clamp = update_clamp
        self.grad_norm_clip = grad_norm_clip
        self.fast_layers = [*model.linear_layers, model.i2h]
        self.layers = [*self.fast_layers, model.i2o]
        self.buffers = {}
        self.snapshot = None
        self.start_slow = None

    @torch.no_grad()
    def begin(self, num_steps, batch):
        device = self.model.i2o.per_sample_weights.device
        n_fast = len(self.fast_layers)
        shapes = {name: (num_steps, batch, n_fast) for name in PER_STEP_LAYER}
        shapes["slow_delta"] = (num_steps, batch, n_fast + 1)
        for name in ("max_logit", "logit_norm", "loss", "hidden_norm"):
            shapes[name] = (num_steps, batch)
        self.buffers = {name: torch.zeros(shape, device=device) for name, shape in shapes.items()}
        self.snapshot = None
        self.start_slow = None

    @torch.no_grad()
    def before_update(self, step, output, output_error, loss, hidden, fast_write=True):
        buffers, model = self.buffers, self.model
        output = output.detach()
        buffers["max_logit"][step] = output.max(dim=1).values.float()
        buffers["logit_norm"][step] = row_norm(output)
        buffers["loss"][step] = loss.detach().float()
        buffers["hidden_norm"][step] = row_norm(hidden)
        if fast_write:
            projected, _ = model.dfa_step_errors(output_error, self.grad_norm_clip)
        for l, layer in enumerate(self.fast_layers):
            inputs, weights, mask = layer.in_traces.data, layer.per_sample_weights.data, layer.ephemeral_mask
            fast, slow = weights * mask, weights * ~mask
            buffers["act_norm"][step, :, l] = row_norm(inputs)
            buffers["fast_norm"][step, :, l] = row_norm(fast)
            buffers["fast_drive"][step, :, l] = row_norm(torch.bmm(fast, inputs.unsqueeze(2)))
            buffers["slow_drive"][step, :, l] = row_norm(torch.bmm(slow, inputs.unsqueeze(2)))
            if fast_write:
                update = ephemeral_update(dfa_per_sample_gradient(projected[l], inputs), layer.plasticity,
                                          mask, self.update_clamp, layer.is_last_layer)
                buffers["fast_write"][step, :, l] = self.learning_rate * row_norm(update * mask)
        self.snapshot = [layer.per_sample_weights.data.clone() for layer in self.layers]
        if self.start_slow is None:
            self.start_slow = [snap * ~layer.ephemeral_mask for snap, layer in zip(self.snapshot, self.layers)]

    @torch.no_grad()
    def after_update(self, step):
        for l, (layer, before) in enumerate(zip(self.layers, self.snapshot)):
            delta, mask = layer.per_sample_weights.data - before, layer.ephemeral_mask
            if l < len(self.fast_layers):
                self.buffers["fast_delta"][step, :, l] = row_norm(delta * mask)
            self.buffers["slow_delta"][step, :, l] = row_norm(delta * ~mask)
        self.snapshot = None

    @torch.no_grad()
    def finish(self):
        """The traces as CPU float32 tensors (see the module docstring), plus the derived
        write_norm and loop_gain."""
        traces = {name: buffer.cpu() for name, buffer in self.buffers.items()}
        traces["slow_total_delta"] = torch.stack([
            row_norm(layer.per_sample_weights.data * ~layer.ephemeral_mask - start)
            for layer, start in zip(self.layers, self.start_slow)], dim=1).cpu()
        traces.update(derived_traces(traces))
        self.buffers, self.start_slow = {}, None
        return traces


def derived_traces(traces):
    """write_norm [T,B] and loop_gain [T,B] from fast_write [T,B,L]."""
    write = torch.linalg.vector_norm(traces["fast_write"], dim=2)
    gain = torch.full_like(write, float("nan"))
    previous = write[:-1]
    gain[1:] = torch.where(previous > 0, write[1:] / previous.clamp_min(1e-30), gain[1:])
    return {"write_norm": write, "loop_gain": gain}


def longest_run_above_one(gain):
    """Per row of loop_gain [T,B], the longest run of consecutive steps with gain > 1."""
    best = torch.zeros(gain.shape[1], dtype=torch.long)
    run = torch.zeros(gain.shape[1], dtype=torch.long)
    for step in range(gain.shape[0]):
        above = gain[step] > 1  # NaN compares False, which also ends a run
        run = torch.where(above, run + 1, torch.zeros_like(run))
        best = torch.maximum(best, run)
    return best


def summarize(traces, prefix="trace"):
    """Scalar summaries of one batch's traces for the console and W&B: loop-gain statistics, the
    final step's trunk activation and fast norms (batch means), the largest logit anywhere, and
    the final-step ratio of fast to slow drive (mean over layers)."""
    gain = traces["loop_gain"]
    finite = gain[torch.isfinite(gain)]
    last = lambda name: traces[name][-1].float()  # noqa: E731
    total_fast = last("fast_norm").square().sum(dim=1).sqrt()
    summary = {
        f"{prefix}/loop_gain_median": float(finite.median()) if finite.numel() else math.nan,
        f"{prefix}/loop_gain_p90": float(finite.quantile(0.9)) if finite.numel() else math.nan,
        f"{prefix}/frac_gain_gt1": float((finite > 1).float().mean()) if finite.numel() else math.nan,
        f"{prefix}/longest_run_gt1": float(longest_run_above_one(gain).float().mean()),
        f"{prefix}/trunk_act_norm_last": float(traces["act_norm"][-1, :, -1].mean()),
        f"{prefix}/trunk_act_norm_max": float(traces["act_norm"][:, :, -1].max()),
        f"{prefix}/fast_norm_last": float(total_fast.mean()),
        f"{prefix}/max_logit_max": float(traces["max_logit"].max()),
        f"{prefix}/fast_over_slow_drive_last": float(
            (last("fast_drive") / last("slow_drive").clamp_min(1e-30)).mean()),
        f"{prefix}/slow_total_delta": float(traces["slow_total_delta"].square().sum(dim=1).sqrt().mean()),
    }
    return summary
