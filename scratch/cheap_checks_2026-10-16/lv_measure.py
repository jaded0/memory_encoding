"""Large and visible (LV in PREREG_fastlayers.md, 2026-10-16). Run from the code dir that trained the checkpoint (code_m for the
fastlayers runs, the battery code for battery runs). usage: lv_measure.py CHECKPOINT [CHECKPOINT ...]
env: LV_BATCHES (default 4), LV_DEVICE (default cuda if available), SCRATCH_RG_SCALE (the aimed scale; read at import by code_m).

Slow weights frozen. Each held-out episode starts with start_sequence_wipe (fast entries zero) and every step writes the fast entries
with the training DFA step (heldout.py's "observed" protocol, fast_only_dfa_step, including the aimed write). At every valid step,
in the forward pass BEFORE that step's write, for each trunk layer i with fast entries, with pre_i = s_i + D_i
(s_i: slow entries times input plus bias; D_i: fast entries times input, i.e. this sequence's fast read):
  S_i = |D_i| / |s_i|                                   size of the fast read relative to the slow pre-activation
  I_i = |z - z(-i)|                                     logit impact: logits recomputed with D_i removed from layer i only
  V_i = |J_i D_i| / (|D_i| |J_i|_F / sqrt(n_i))         visibility: gain of the write direction to the logits vs a random direction
with z the logits and J_i = d z / d pre_i (exact GELU derivative; downstream layers keep their fast entries). Prints one JSON row per
checkpoint: median and p90 over valid steps with a nonzero fast read.
Exploratory additions (not pre-registered, added after scoring LV1-LV5): Ic_i = the same impact on mean-centred logits (softmax ignores
a shift of all logits), dL_i = loss without D_i minus loss with it (> 0: the fast read helps), zc = |centred logits|."""
import sys, os, re, json, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
from ephemeral_model import dfa_output_error
from heldout import load_heldout_batches
from utils import initialize_charset, load_checkpoint, model_input, read_checkpoint, upgrade_legacy_config
from train import build_model, build_parser, positional_encoding

nb = int(os.environ.get("LV_BATCHES", "4"))
device = os.environ.get("LV_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def gelu_prime(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2))) + x * torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def q(a, p):
    if not a:
        return None
    a = sorted(a)
    return a[min(len(a) - 1, int(p * len(a)))]


def downstream(layers, out_layer, i, pre_i):
    """Logits from layer i's pre-activation through the later trunk layers and i2o (full per-sample weights)."""
    a = F.gelu(pre_i)
    for L in layers[i + 1:]:
        a = F.gelu(torch.bmm(L.per_sample_weights.data, a.unsqueeze(2)).squeeze(2) + L.bias)
    return torch.bmm(out_layer.per_sample_weights.data, a.unsqueeze(2)).squeeze(2) + out_layer.bias


def jacobian(layers, out_layer, i):
    """d logits / d pre_i, [B, vocab, n_i], at the current forward pass."""
    J = out_layer.per_sample_weights.data
    for L in reversed(layers[i + 1:]):
        J = torch.bmm(J * gelu_prime(L.pre_activation())[:, None, :], L.per_sample_weights.data)
    return J * gelu_prime(layers[i].pre_activation())[:, None, :]


for path in sys.argv[1:]:
    ck = read_checkpoint(path)
    defaults = {k: v for k, v in vars(build_parser().parse_args([])).items() if not k.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(ck.get("config", {}))}
    assert not config.get("layer_norm") and not config.get("output_tanh") and not config.get("residual_connection"), path
    charset, _, _, n = initialize_charset(config["dataset"])
    model = build_model(config, charset, n)
    model, _, it, _, _ = load_checkpoint(path, model, config, device=device, checkpoint=ck)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], device)
    model.eval()
    layers, out_layer = list(model.linear_layers), model.i2o
    fast = [bool(L.ephemeral_mask.any()) for L in layers]
    st = {i: {"S": [], "I": [], "V": [], "lin": [], "Ic": [], "dL": []} for i in range(len(layers)) if fast[i]}
    znorm = []
    recon, losses = 0.0, []
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for texts, onehot in load_heldout_batches(config["dataset"], config["batch_size"], nb, device, "validation"):
            model.start_sequence_wipe()
            h = model.initHidden(onehot.shape[0])
            for t in range(onehot.shape[1] - 1):
                out, h = model(model_input(onehot, t, config["input_mode"], config["pe_matrix"]), h)
                valid = onehot[:, t + 1].sum(1) > 0
                for i in st:
                    L = layers[i]
                    x = L.in_traces.data.unsqueeze(2)
                    W = L.per_sample_weights.data
                    m = L.ephemeral_mask.unsqueeze(0) if L.ephemeral_mask.dim() == 2 else L.ephemeral_mask
                    D = torch.bmm(torch.where(m, W, torch.zeros_like(W)), x).squeeze(2)
                    s = torch.bmm(torch.where(m, torch.zeros_like(W), W), x).squeeze(2) + L.bias
                    pre = L.pre_activation()
                    z = downstream(layers, out_layer, i, pre)
                    recon = max(recon, float((z - out).abs().max()))
                    zm = downstream(layers, out_layer, i, pre - D)
                    I = (z - zm).norm(dim=1)
                    dz = (z - zm) - (z - zm).mean(dim=1, keepdim=True)
                    Ic = dz.norm(dim=1)
                    tgt = onehot[:, t + 1].argmax(1)
                    dL = F.cross_entropy(zm, tgt, reduction="none") - F.cross_entropy(z, tgt, reduction="none")
                    J = jacobian(layers, out_layer, i)
                    lin = torch.bmm(J, D.unsqueeze(2)).squeeze(2).norm(dim=1)
                    dn = D.norm(dim=1)
                    V = lin / (dn * J.flatten(1).norm(dim=1) / math.sqrt(pre.shape[1]) + 1e-30)
                    keep = valid & (dn > 0)
                    st[i]["S"] += (dn / (s.norm(dim=1) + 1e-12))[keep].tolist()
                    st[i]["I"] += I[keep].tolist()
                    st[i]["V"] += V[keep].tolist()
                    st[i]["lin"] += lin[keep].tolist()
                    st[i]["Ic"] += Ic[keep].tolist()
                    st[i]["dL"] += dL[keep].tolist()
                znorm += (out - out.mean(dim=1, keepdim=True)).norm(dim=1)[valid].tolist()
                loss, err = dfa_output_error(out, onehot[:, t + 1], criterion)
                losses += loss[valid].tolist()
                model.fast_only_dfa_step(err, config["learning_rate"], config["ephemeral_update_clamp"],
                                         config.get("grad_norm_clip", 0))
    row = {"run": os.path.basename(os.path.dirname(os.path.normpath(path))), "iter": it - 1, "scale": float(os.environ.get("SCRATCH_RG_SCALE", "1")),
           "recon_maxabs": recon, "loss_mean": sum(losses) / max(1, len(losses)),
           "zc_p50": q(znorm, .5), "zc_p90": q(znorm, .9)}
    for i, d in st.items():
        for k, v in d.items():
            row[f"{k}{i}_p50"] = q(v, .5)
            row[f"{k}{i}_p90"] = q(v, .9)
        row[f"n{i}"] = len(d["S"])
    print(json.dumps(row), flush=True)
