"""Key geometry and read-time layer ablation on key recall checkpoints (checks K1-K3, PREREG 2026-10-15).

Run from a code_v2kr checkout:  python ../kr_keys.py CKPT [CKPT ...]
Episodes look like '0000?10!1': the store key is the '?' step s (the value is text[s+1]), the query key is
the '!' step q, and the answer is text[q+1], predicted at step q. Replays held-out episodes under the
observed protocol (slow frozen, the model's own DFA writes on the fast entries every step, as keygeom4.py),
records each trunk layer's input x_t, and at q reports per layer:
  cos_sq  = cos(x_s, x_q)
  cos_oth = mean cos(x_t, x_q) over the other earlier steps t < q, t != s
  sel     = (x_s . x_q) / rms_{t<q, t!=s}(x_t . x_q)
and the answer accuracy with each layer's fast entries zeroed at the query step (read-time ablation)."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from ephemeral_model import dfa_output_error  # noqa: E402
from heldout import load_heldout_batches  # noqa: E402
from utils import initialize_charset, load_checkpoint, model_input, read_checkpoint, upgrade_legacy_config  # noqa: E402
from train import build_model, build_parser, positional_encoding  # noqa: E402

ABL = {"all": (1, 1, 1), "drop0": (0, 1, 1), "drop1": (1, 0, 1), "drop2": (1, 1, 0), "none": (0, 0, 0)}
torch.set_num_threads(int(os.environ.get("KK_THREADS", "8")))


def load(path):
    ck = read_checkpoint(path)
    defaults = {k: v for k, v in vars(build_parser().parse_args([])).items() if not k.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(ck.get("config", {}))}
    charset, _, _, n = initialize_charset(config["dataset"])
    model = build_model(config, charset, n)
    model, _, it, _, _ = load_checkpoint(path, model, config, device="cpu", checkpoint=ck)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], "cpu")
    return model.eval(), config, charset, it - 1


@torch.no_grad()
def run(path, nb):
    model, config, charset, it = load(path)
    layers = list(model.linear_layers)
    cap = {}
    hooks = [l.register_forward_hook(lambda m, inp, out, k=k: cap.__setitem__(k, inp[0].detach().double())) for k, l in enumerate(layers)]
    crit = torch.nn.CrossEntropyLoss(reduction="none")
    st = {k: {"cos_sq": [], "cos_oth": [], "sel": []} for k in range(len(layers))}
    acc = {name: [] for name in ABL}
    for texts, onehot in load_heldout_batches(config["dataset"], config["batch_size"], nb, "cpu", "validation"):
        B, T = onehot.shape[0], onehot.shape[1]
        sq = [(t.index("?"), t.index("!")) for t in texts]
        model.start_sequence_wipe(); hidden = model.initHidden(B)
        xs = {k: [] for k in range(len(layers))}
        for i in range(T - 1):
            inp = model_input(onehot, i, config["input_mode"], config["pe_matrix"])
            rows = [b for b, (s, q) in enumerate(sq) if q == i]
            if rows:
                saved = [l.per_sample_weights.clone() for l in layers]
                for name, keep in ABL.items():
                    for l, w, kp in zip(layers, saved, keep):
                        l.per_sample_weights.copy_(w if kp else torch.where(l.ephemeral_mask.unsqueeze(0), torch.zeros_like(w), w))
                    o, _ = model(inp, hidden)
                    for b in rows:
                        acc[name].append(int(o[b].argmax()) == int(onehot[b, i + 1].argmax()))
                for l, w in zip(layers, saved):
                    l.per_sample_weights.copy_(w)
            out, hidden = model(inp, hidden)
            for k in xs:
                xs[k].append(cap[k])
            if rows:
                for k in xs:
                    for b in rows:
                        s, q = sq[b]
                        X = torch.stack([xs[k][t][b] for t in range(q + 1)])
                        xq = X[q]; dots = X[:q] @ xq; norms = X[:q].norm(dim=1) * xq.norm() + 1e-12
                        oth = [t for t in range(q) if t != s]
                        st[k]["cos_sq"].append(float(dots[s] / norms[s]))
                        st[k]["cos_oth"].append(float((dots[oth] / norms[oth]).mean()))
                        st[k]["sel"].append(float(dots[s] / (dots[oth].pow(2).mean().sqrt() + 1e-12)))
            _, err = dfa_output_error(out, onehot[:, i + 1], crit)
            projected, _ = model.dfa_step_errors(err, 0)
            for layer, e in zip(model.trained_layers(), projected):
                if layer.is_last_layer:
                    continue
                model._fast_entry_step(layer, e, config["learning_rate"], config["ephemeral_update_clamp"])
    for h in hooks:
        h.remove()
    fast = {f"L{k}": bool(l.ephemeral_mask.any()) for k, l in enumerate(layers)}
    return {"ckpt": "/".join(path.split("/")[-2:]), "iter": it, "n": len(acc["all"]), "fast_layers": fast,
            "acc": {k: round(float(np.mean(v)), 4) for k, v in acc.items()},
            "layers": {f"L{k}": {m: round(float(np.median(v)), 4) for m, v in d.items()} for k, d in st.items()}}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(run(p, int(os.environ.get("NB", "16")))), flush=True)
