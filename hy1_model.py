"""Opt-in HY1 char decoder. No existing model/defaults are changed.

Eager FP32 only: explicit matmul/softmax attention, no SDPA/compile/Triton.
Attention caches cross chunk boundaries, but are detached (chunk TBPTT).
Four local layers can access up to 4*(W-1) prior positions, not merely W.

A1 reserves exactly round(fraction*d*hidden) random layer-0 output-MLP
entries, represented by a nontrainable vector F. Their slow weights stay zero.
B_fast is a separate fixed [vocab,d] Rademacher/sqrt(vocab) feedback matrix.
After a completed chunk, E=softmax(logits)-onehot(target), D=E@B_fast,
G=(D.T@GELU(mlp_input_projection))/T restricted to reserved entries.
U=-G/RMS(G) (zero if G=0); F <- .95*F + alpha*U, alpha=1 by default.
F is then RMS-clipped at 10. Alpha=1 is high relative to slow initial weight
RMS (~.02), deliberately made safe by the read bound, not a tiny write rate.
For each token r=F@hidden is rescaled by min(1,.1*RMS(slow)/RMS(r)).
Both this scale and every write/cache/state are detached, never meta-trained.
F is fixed during a chunk; labels can only affect later chunks. Fast state
is reset per document unless explicitly supplied to evaluate_document.
"""
from dataclasses import asdict, dataclass, field
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class HY1Config:
    vocab_size: int
    arm: str = "A0"
    seed: int = 110601
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    hidden_dim: int = 1024
    window: int = 64
    chunk_size: int = 64
    fast_fraction: float = .1
    fast_alpha: float = 1.
    fast_decay: float = .95
    fast_read_cap: float = .1
    fast_state_rms_cap: float = 10.

    def __post_init__(self):
        if self.arm not in ("A0", "A1", "A2", "A3"):
            raise ValueError("arm must be A0/A1/A2/A3")
        if self.d_model % self.n_heads or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by n_heads")
        if self.window < 1 or self.chunk_size < 1 or self.vocab_size < 2:
            raise ValueError("invalid dimensions")
        if not 0 < self.fast_fraction < 1:
            raise ValueError("fast_fraction must be in (0,1)")

    @property
    def attention_window(self):
        return None if self.arm == "A3" else self.window * (2 if self.arm == "A2" else 1)


@dataclass
class HY1State:
    fast: Optional[torch.Tensor] = None
    cache: list = field(default_factory=list)
    position: int = 0


@dataclass
class PendingWrite:
    logits: torch.Tensor
    hidden: Optional[torch.Tensor]


def rms(x):
    return x.square().mean(dim=-1, keepdim=True).sqrt()


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3*c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.norm2 = nn.LayerNorm(c.d_model)
        self.fc1 = nn.Linear(c.d_model, c.hidden_dim)
        self.fc2 = nn.Linear(c.hidden_dim, c.d_model)

    def forward(self, x, cache, c, fast_matrix=None, slow_mask=None):
        t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(t, 3, c.n_heads, d//c.n_heads).unbind(1)
        q, k, v = (z.transpose(0, 1) for z in (q, k, v))
        old = 0
        if cache is not None:
            old = cache[0].shape[1]
            k = torch.cat((cache[0], k), dim=1)
            v = torch.cat((cache[1], v), dim=1)
        qi = torch.arange(t, device=x.device)[:, None] + old
        ki = torch.arange(k.shape[1], device=x.device)[None, :]
        allowed = ki <= qi
        if c.attention_window is not None:
            allowed = allowed & (ki > qi-c.attention_window)
        scores = q @ k.transpose(-1, -2) / math.sqrt(d//c.n_heads)
        probs = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        x = x + self.proj((probs @ v).transpose(0, 1).reshape(t, d))
        h = F.gelu(self.fc1(self.norm2(x)))
        weight = self.fc2.weight if slow_mask is None else self.fc2.weight * slow_mask
        slow = F.linear(h, weight, self.fc2.bias)
        if fast_matrix is not None:
            read = F.linear(h, fast_matrix)
            scale = (c.fast_read_cap*rms(slow.detach())/rms(read.detach()).clamp_min(1e-12)).clamp(max=1)
            slow = slow + read*scale
        keep = k.shape[1] if c.attention_window is None else min(k.shape[1], c.attention_window-1)
        new_cache = (k[:, -keep:].detach(), v[:, -keep:].detach()) if keep else (k[:, :0].detach(), v[:, :0].detach())
        return x+slow, new_cache, h


class HY1Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Local RNG fork: identical slow initialization and no global RNG side effects.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
            self.norm = nn.LayerNorm(config.d_model)
            self.output = nn.Linear(config.d_model, config.vocab_size)
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.bias)
        # Separate generator leaves ALL slow initialization identical across arms.
        g = torch.Generator().manual_seed(config.seed+1)
        n = config.d_model*config.hidden_dim
        count = round(n*config.fast_fraction) if config.arm == "A1" else 0
        indices = torch.randperm(n, generator=g)[:count]
        mask = torch.ones(n)
        mask[indices] = 0
        self.register_buffer("fast_indices", indices)
        self.register_buffer("slow_mask", mask.view(config.d_model, config.hidden_dim))
        b = (torch.randint(0, 2, (config.vocab_size, config.d_model), generator=g)*2-1).float()/math.sqrt(config.vocab_size)
        self.register_buffer("B_fast", b)
        if config.arm == "A1":
            self.blocks[0].fc2.weight.register_hook(lambda grad: grad*self.slow_mask)
        self.enforce_exclusive()

    @torch.no_grad()
    def enforce_exclusive(self):
        if self.config.arm == "A1":
            self.blocks[0].fc2.weight.mul_(self.slow_mask)

    def new_state(self, fast=None):
        if self.config.arm == "A1":
            if fast is None:
                fast = self.embedding.weight.new_zeros(self.fast_indices.numel())
            else:
                fast = fast.detach().clone().to(self.embedding.weight)
        else:
            fast = None
        return HY1State(fast=fast)

    def forward_chunk(self, tokens, state=None):
        if tokens.ndim != 1 or not 0 < len(tokens) <= self.config.chunk_size:
            raise ValueError("expected 1D tokens of length 1..chunk_size")
        state = self.new_state() if state is None else state
        c = self.config
        positions = torch.arange(state.position, state.position+len(tokens), device=tokens.device).float()
        freq = torch.exp(torch.arange(0, c.d_model, 2, device=tokens.device).float()*(-math.log(10000.)/c.d_model))
        phase = positions[:, None]*freq
        pe = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
        x = self.embedding(tokens)+pe
        dense = None
        if state.fast is not None:
            dense = x.new_zeros(c.d_model*c.hidden_dim).scatter(0, self.fast_indices, state.fast).view(c.d_model, c.hidden_dim)
        caches, hidden = [], None
        for i, block in enumerate(self.blocks):
            cache = state.cache[i] if state.cache else None
            x, kv, h = block(x, cache, c, dense if i == 0 else None,
                             self.slow_mask if i == 0 and c.arm == "A1" else None)
            caches.append(kv)
            if i == 0 and c.arm == "A1":
                hidden = h.detach()
        logits = self.output(self.norm(x))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite HY1 logits")
        return logits, HY1State(state.fast, caches, state.position+len(tokens)), PendingWrite(logits.detach(), hidden)

    @torch.no_grad()
    def commit(self, state, pending, targets):
        if self.config.arm != "A1":
            return state
        if targets.shape != pending.logits.shape[:1]:
            raise ValueError("targets must match pending chunk")
        error = pending.logits.softmax(-1)
        error.scatter_add_(1, targets[:, None], -torch.ones_like(targets[:, None], dtype=error.dtype))
        dfa = error @ self.B_fast
        grad = (dfa.T @ pending.hidden)/len(targets)
        write = -grad.flatten()[self.fast_indices]
        write = write / write.square().mean().sqrt().clamp_min(1e-12)
        fast = self.config.fast_decay*state.fast + self.config.fast_alpha*write
        fast = fast * (self.config.fast_state_rms_cap/fast.square().mean().sqrt().clamp_min(1e-12)).clamp(max=1)
        if not torch.isfinite(fast).all():
            raise FloatingPointError("non-finite HY1 fast state")
        return HY1State(fast.detach(), state.cache, state.position)

    def state_bytes(self):
        """FP32 values, excluding fixed mask/index/feedback/model metadata."""
        c = self.config
        return {"fast_bytes": self.fast_indices.numel()*4,
                "A2_extra_KV_bytes": c.n_layers*2*c.window*c.d_model*4}


def encode_text(text, chars):
    # Delegate filtering/padding to the exact B2 preprocessor, not a lookalike.
    from preprocess import filter_text
    from utils import initialize_charset
    expected = initialize_charset("roneneldan/tinystories")[0]
    if chars != expected:
        raise ValueError("chars must match heldout TinyStories/B2 charset and order")
    filtered = filter_text({"text": [text]}, "roneneldan/tinystories")["text"][0]
    stoi = {ch: i for i, ch in enumerate(chars)}
    return torch.tensor([stoi[ch] for ch in filtered], dtype=torch.long)


@torch.no_grad()
def evaluate_document(model, document, chars=None, state=None, wipe_every_chunk=False):
    """Loss[i] predicts document[i+1]. Carries fast only, NEVER KV across docs.

    Slow parameters are frozen by no_grad, but labels drive chunk-end DFA writes.
    state=None starts a fresh document. Returned states contain detached tensors.
    """
    device = model.embedding.weight.device
    tokens = encode_text(document, chars) if isinstance(document, str) else document
    tokens = tokens.to(device=device, dtype=torch.long)
    state = model.new_state(None if state is None else state.fast)
    was_training = model.training
    model.eval()
    losses = []
    try:
        for start in range(0, len(tokens)-1, model.config.chunk_size):
            if wipe_every_chunk and state.fast is not None:
                state = HY1State(torch.zeros_like(state.fast), state.cache, state.position)
            x = tokens[start:min(start+model.config.chunk_size, len(tokens)-1)]
            y = tokens[start+1:start+1+len(x)]
            logits, state, pending = model.forward_chunk(x, state)
            loss = F.cross_entropy(logits, y, reduction="none")
            if not torch.isfinite(loss).all():
                raise FloatingPointError("non-finite eval loss")
            losses.append(loss.cpu())
            state = model.commit(state, pending, y)
    finally:
        model.train(was_training)
    return torch.cat(losses) if losses else torch.empty(0), state


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = HY1Model(HY1Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.enforce_exclusive()
    model.eval()
    return model, checkpoint
