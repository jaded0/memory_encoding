"""Slow-only (s=0) vs full (s=1,0.3,0.1) replay of checkpoints, adapted from deckard overnight int_replay.py (same conventions as blowup analysis section 5). usage: replay_arm.py OUT.json CKPT..."""
import sys, os, json, torch, time
sys.path.insert(0, os.path.expanduser("~/causal_cap_2026-10-02/code"))
torch.set_num_threads(int(os.environ.get("NT","8")))
import trace_replay as tr
from heldout import load_heldout_batches
def q(x,p):
    x=x[torch.isfinite(x)]
    return float(torch.quantile(x.flatten().float(),p)) if x.numel() else float("nan")
out = sys.argv[1]; paths = sys.argv[2:]
res = json.load(open(out)) if os.path.exists(out) else {}
batches = None
def spec(w, iters=30):
    v = torch.randn(w.shape[1], generator=torch.Generator().manual_seed(0))
    for _ in range(iters):
        u = w @ v; u = u / u.norm(); v = w.T @ u; sv = v.norm(); v = v / sv
    return float(sv)
for p in paths:
    if p in res: continue
    t0 = time.time()
    model, config, state, it = tr.load_model(p, "cpu", False)
    if batches is None:
        batches = load_heldout_batches(config["dataset"], config["batch_size"], 8, "cpu", "validation")
    r = {"iter": it, "lr": config["learning_rate"], "alpha": float(config["plasticity"]), "tanh": config.get("output_tanh"), "wd": config.get("slow_weight_decay")}
    with torch.no_grad():
        r["spec"] = [spec((l.per_sample_weights.data * ~l.ephemeral_mask).mean(0).float()) for l in [*model.linear_layers, model.i2h]]
        r["spec_i2o"] = spec(model.i2o.per_sample_weights.data.mean(0).float())
    alpha = r["alpha"]
    for s in [1.0, 0.0, 0.3, 0.1]:
        model.set_plasticity(alpha * s)
        tb = tr.replay_checkpoint(model, config, state, batches)
        cat = lambda k: torch.cat([x[k] for x in tb], dim=1)       # T may differ across batches?
        d = {}
        try:
            g = cat("loop_gain"); an = cat("act_norm")[..., -1]; ml = cat("max_logit").abs()
            sd = cat("slow_drive")[..., :3].square().sum(2).sqrt(); fd = cat("fast_drive")[..., :3].square().sum(2).sqrt()
        except RuntimeError:
            g = torch.cat([x["loop_gain"].flatten() for x in tb]); an = torch.cat([x["act_norm"][..., -1].flatten() for x in tb]); ml = torch.cat([x["max_logit"].abs().flatten() for x in tb])
            sd0 = torch.cat([x["slow_drive"][0][..., :3].square().sum(1).sqrt() for x in tb]); sd = None
            sd = torch.cat([x["slow_drive"][..., :3].square().sum(2).sqrt().flatten() for x in tb]); fd = torch.cat([x["fast_drive"][..., :3].square().sum(2).sqrt().flatten() for x in tb])
        # step-0 and last-step quantities computed per batch to be safe
        sd0 = torch.cat([x["slow_drive"][0][..., :3].square().sum(1).sqrt() for x in tb])
        ratio_last = torch.cat([(x["fast_drive"][-1][..., :3].square().sum(1).sqrt() / x["slow_drive"][-1][..., :3].square().sum(1).sqrt().clamp_min(1e-30)) for x in tb])
        ratio_all = fd.flatten() / sd.flatten().clamp_min(1e-9)
        d = dict(loss=float(torch.stack([x["loss"].mean() for x in tb]).mean()), loss_med=q(torch.cat([x["loss"].flatten() for x in tb]), .5),
                 act_med=q(an, .5), act_p95=q(an, .95), act_max=float(an.max()), logit_max=float(ml.max()), logit_med=q(ml, .5), logit_p95=q(ml, .95),
                 sdrive0_med=q(sd0, .5), ratio_last_med=q(ratio_last, .5), ratio_all_med=q(ratio_all, .5),
                 gain_med=q(g, .5), frac_gt1=float((g[torch.isfinite(g)] > 1).float().mean()),
                 act0_med=q(torch.cat([x["act_norm"][0][..., -1] for x in tb]), .5))
        r["s%g" % s] = d
    res[p] = r
    json.dump(res, open(out, "w"), indent=1)
    print(it, p.split("/")[-2], "t=%.0fs" % (time.time() - t0), "spec", [round(x, 1) for x in r["spec"]], "s1", {k: round(v, 2) for k, v in r["s1"].items()}, "s0 act", round(r["s0"]["act_med"], 1), flush=True)
