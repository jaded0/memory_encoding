"""Trunk input norms (TN in PREREG_fastlayers.md, 2026-10-16). Run from the code dir that trained the checkpoint (code_m for the
batch-M aimed-write runs). usage: tn_norms.py CHECKPOINT [CHECKPOINT ...]  (env TN_BATCHES, default 4; TN_THREADS)
Forward pass only, fast state wiped at sequence start, no writes (as stepgain2.py). Per trunk layer i, over every valid step of the
held-out validation batches: median / p90 of |x_in,i|^2, median |D (e B_i)|, and the single-write self-read ratio
s*c*phi_i*|x_in,i|^2*|D (e B_i)| / |W_i,slow x_in,i| (s = aimed scale from the run name, RGs<s> / M2-* = 0.1, else 1; B_i is the
layer's fast feedback matrix when it has one; None when the layer has no fast entries)."""
import sys, os, re, json, torch
sys.path.insert(0, os.getcwd())
from ephemeral_model import dfa_activation_derivative
from heldout import load_heldout_batches
from utils import initialize_charset, load_checkpoint, model_input, read_checkpoint, upgrade_legacy_config
from train import build_model, build_parser, positional_encoding
torch.set_num_threads(int(os.environ.get("TN_THREADS", "6")))
nb = int(os.environ.get("TN_BATCHES", "4"))

def aimed_scale(run):
    m = re.search(r"RGs([0-9.]+)", run)
    if m:
        return float(m.group(1))
    return 0.1 if run.startswith("M2-") else 1.0

def q(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(p * len(a)))] if a else None

for path in sys.argv[1:]:
    ck = read_checkpoint(path)
    defaults = {k: v for k, v in vars(build_parser().parse_args([])).items() if not k.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(ck.get("config", {}))}
    charset, _, _, n = initialize_charset(config["dataset"])
    model = build_model(config, charset, n)
    model, _, it, _, _ = load_checkpoint(path, model, config, device="cpu", checkpoint=ck)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], "cpu")
    model.eval()
    run = path.split("/")[-2]
    c = config["learning_rate"] * config["plasticity"]
    layers = list(model.linear_layers)
    phi = [float(L.ephemeral_mask.float().mean()) for L in layers]
    stats = {i: {"x2": [], "p": [], "ratio": []} for i in range(len(layers))}
    with torch.no_grad():
        for texts, onehot in load_heldout_batches(config["dataset"], config["batch_size"], nb, "cpu", "validation"):
            model.start_sequence_wipe(); h = model.initHidden(onehot.shape[0])
            for t in range(onehot.shape[1] - 1):
                out, h = model(model_input(onehot, t, config["input_mode"], config["pe_matrix"]), h)
                e = torch.softmax(out, 1) - onehot[:, t + 1]
                valid = onehot[:, t + 1].sum(1) > 0
                for i, L in enumerate(layers):
                    x = L.in_traces.data
                    fb = getattr(L, "scratch_fast_feedback", None)
                    B = L.feedback_weights.data if fb is None else fb
                    D = dfa_activation_derivative(L.pre_activation(), "gelu")
                    p = (e @ B) * D
                    W = L.per_sample_weights.data
                    Ws = torch.where(L.ephemeral_mask, torch.zeros_like(W), W)
                    slow_out = torch.bmm(Ws, x.unsqueeze(2)).squeeze(2).norm(dim=1)
                    x2 = x.pow(2).sum(1)
                    s = aimed_scale(run) if i == len(layers) - 1 else 1.0
                    stats[i]["x2"].extend(x2[valid].tolist())
                    stats[i]["p"].extend(p.norm(dim=1)[valid].tolist())
                    if phi[i] > 0:
                        r = s * c * phi[i] * x2 * p.norm(dim=1) / (slow_out + 1e-12)
                        stats[i]["ratio"].extend(r[valid].tolist())
    row = {"run": run, "iter": it - 1}
    for i, st in stats.items():
        row[f"x2_{i}_p50"] = round(q(st["x2"], .5), 3); row[f"x2_{i}_p90"] = round(q(st["x2"], .9), 3)
        row[f"p_{i}_p50"] = round(q(st["p"], .5), 5)
        row[f"ratio_{i}_p50"] = None if not st["ratio"] else round(q(st["ratio"], .5), 4)
        row[f"ratio_{i}_p90"] = None if not st["ratio"] else round(q(st["ratio"], .9), 4)
    print(json.dumps(row), flush=True)
