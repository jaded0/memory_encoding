"""Re-forward drift test, warm-start arm (PREREG_fastlayers.md, "Re-forward drift test (2026-10-16)", ADDENDUM 1).
Turns a K = 1 checkpoint into a --fast_backward_per_forward K checkpoint at matched lr*alpha*K: the fast entries'
plasticity (alpha) is divided by K in the state dict and in the stored config, and the config records K, so
`train.py --resume_checkpoint OUT --fast_backward_per_forward K --plasticity alpha/K` continues the same run (same
iteration, data stream and RNG state) with K re-forward passes per character.
Run from a code dir. usage: warmstart.py SRC.pth K OUT.pth"""
import sys, os, torch
sys.path.insert(0, os.getcwd())
from utils import read_checkpoint

src, K, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ck = read_checkpoint(src)
cfg, sd = ck["config"], ck["model_state_dict"]
assert cfg.get("fast_backward_per_forward", 1) == 1, "source must be a K = 1 checkpoint"
alpha = float(cfg["plasticity"])
for key in [k for k in sd if k.endswith(".plasticity")]:
    mask = sd[key.replace(".plasticity", ".ephemeral_mask")]
    vals = sd[key][mask].unique()
    assert vals.numel() <= 1 and (vals.numel() == 0 or float(vals) == alpha), (key, vals)
    sd[key] = torch.where(mask, sd[key] / K, sd[key])
cfg["plasticity"] = alpha / K
cfg["fast_backward_per_forward"] = K
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
torch.save(ck, out)
print(f"{src} -> {out}: iter {ck['iter']}, alpha {alpha:g} -> {alpha / K:g}, K = {K}")
