"""P6 read-only checkpoint replay; run cwd=training-code archive, --code that archive."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def distribution(x):
    a = np.asarray(x, dtype=float)
    if not a.size:
        return dict(n=0, median=None, q25=None, q75=None, mean=None)
    return dict(n=int(a.size), median=float(np.median(a)), q25=float(np.quantile(a,.25)),
                q75=float(np.quantile(a,.75)), mean=float(a.mean()))


def analyze(E, A, valid, stores, task):
    """Arrays [episodes,steps,N]. Padding excluded. Energy participation is selector invariant."""
    N = E.shape[-1]
    total = E.sum(-1)
    active = valid & (total > 0)
    sm = np.zeros(valid.shape, bool)
    for b,ss in enumerate(stores):
        sm[b,ss] = True
    pr = total**2 / np.maximum(N*(E**2).sum(-1),1e-300)
    result = dict(N=N,real_steps=int(valid.sum()),zero_energy_steps=int((valid & ~active).sum()),
                  store_steps=int(sm.sum()),pr_all=distribution(pr[active]),
                  pr_store=distribution(pr[active & sm]),selections={})
    for variant,scores in (("energy",E),("activation",A)):
        order = np.argsort(-scores,axis=-1,kind="stable")
        result["selections"][variant] = {}
        for k in (16,64,256):
            assert k <= N
            inds = order[...,:k]
            share = np.take_along_axis(E,inds,-1).sum(-1)/np.maximum(total,1e-300)
            pairs = []
            counts = np.zeros(N,dtype=np.int64)
            for b,ss in enumerate(stores):
                for s in ss:
                    counts += np.bincount(inds[b,s],minlength=N)
                compare = ([(ss[0],t) for t in np.flatnonzero(valid[b]) if t != ss[0]] if task == "kr"
                           else [(s,t) for j,s in enumerate(ss) for t in ss[j+1:]])
                for s,t in compare:
                    n = np.intersect1d(inds[b,s],inds[b,t],assume_unique=True).size
                    pairs.append(n/(2*k-n))
            result["selections"][variant][str(k)] = dict(
                all=distribution(share[active]),store=distribution(share[active & sm]),
                jaccard=distribution(pairs),baseline=k/(2*N-k),ever_selected=float((counts>0).mean()),
                top5_share=float(np.sort(counts)[-int(np.ceil(.05*N)):].sum()/counts.sum()),
                selection_count=int(counts.sum()),counts=counts.tolist())
    return result


@torch.no_grad()
def measure(a):
    sys.path.insert(0,a.code)
    from ephemeral_model import dfa_output_error
    from heldout import load_heldout_batches
    from train import build_model,build_parser,positional_encoding
    from utils import initialize_charset,load_checkpoint,model_input,read_checkpoint,upgrade_legacy_config
    from reproducibility import seed_everything
    from metrics import recall_targets
    ck = read_checkpoint(a.checkpoint)
    config = {**vars(build_parser().parse_args([])),**upgrade_legacy_config(ck.get("config",{}))}
    seed_everything(int(config.get("seed") or 0),deterministic=False)
    charset,_,_,n = initialize_charset(config["dataset"])
    model = build_model(config,charset,n)
    model,_,it,_,_ = load_checkpoint(a.checkpoint,model,config,device=a.device,checkpoint=ck)
    del ck
    model = model.to(a.device).eval()
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"],a.device)
    assert model.fast_iterations == 1
    assert not getattr(model,"layer_norm",False), "hook must include layer norm if configured"
    named = [(f"L{i}",l) for i,l in enumerate(model.linear_layers)]+[("i2h",model.i2h)]
    named = [(name,l) for name,l in named if bool(l.ephemeral_mask.any())]
    activation = {}
    def capture(name,out):
        activation[name] = (torch.tanh(out) if name == "i2h" else F.gelu(out)).abs()
    hooks = [l.register_forward_hook(lambda m,inp,out,name=name:capture(name,out)) for name,l in named]
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    combined = {name:[] for name,_ in named}
    recalls,episodes,store_counts = [],0,[]
    split = "validation" if a.task == "kr" else "test"
    batches = load_heldout_batches(config["dataset"],config["batch_size"],a.batches,a.device,split)
    for texts,onehot in batches:
        B,T = onehot.shape[:2]
        valid = (onehot[:,1:].sum(-1)>0).cpu().numpy()
        stores = [[t.index("?")] if a.task == "kr" else [s-1 for s in range(2,t.index("?"),3) if t[s] != "."] for t in texts]
        targets = [recall_targets(t,config["dataset"])[0] for t in texts]
        store_counts.extend(map(len,stores))
        model.start_sequence_wipe()
        hidden = model.initHidden(B)
        history = {name:[[],[]] for name,_ in named}
        for step in range(T-1):
            out,hidden = model(model_input(onehot,step,config["input_mode"],config["pe_matrix"]),hidden)
            _,err = dfa_output_error(out,onehot[:,step+1],criterion)
            projected = dict(zip(map(id,model.trained_layers()),model.fast_write_projected(err,config.get("grad_norm_clip",0))))
            for name,l in named:
                p = projected[id(l)]
                E = p.double().square()*(l.in_traces.double().square() @ l.ephemeral_mask.double().T)
                history[name][0].append(E.cpu().numpy())
                history[name][1].append(activation[name].cpu().numpy())
            for b in range(B):
                if step+1 in targets[b]:
                    recalls.append(int(out[b].argmax()) == int(onehot[b,step+1].argmax()))
            model.fast_only_dfa_step(err,config["learning_rate"],config["ephemeral_update_clamp"],config.get("grad_norm_clip",0))
        for name,l in named:
            E,A = (np.stack(h,axis=1) for h in history[name])
            combined[name].append((E,A,valid,stores))
        episodes += B
    for h in hooks:
        h.remove()
    results = {}
    for name,l in named:
        chunks = combined[name]
        maxT = max(x[0].shape[1] for x in chunks)
        def pad(x):
            return np.pad(x,[(0,0),(0,maxT-x.shape[1])]+([(0,0)] if x.ndim==3 else []))
        E,A,valid = [np.concatenate([pad(x[j]) for x in chunks]) for j in range(3)]
        valid = valid.astype(bool)
        stores = sum([x[3] for x in chunks],[])
        results[name] = analyze(E,A,valid,stores,a.task)
        np.savez_compressed(Path(a.out).with_suffix("."+name+".npz"),E=E,A=A,valid=valid,
                            store_mask=np.array([[t in ss for t in range(maxT)] for ss in stores]))
    result = dict(checkpoint=a.checkpoint,checkpoint_sha256=hashlib.file_digest(open(a.checkpoint,"rb"),"sha256").hexdigest(),
                  iteration=it-1,task=a.task,dataset=config["dataset"],split=split,code=a.code,episodes=episodes,
                  observed_recall=float(np.mean(recalls)),store_counts=distribution(store_counts),layers=results,
                  config={k:config.get(k) for k in ("seed","n_hidden","n_layers","batch_size","input_mode","fast_read_norm",
                          "fast_structure","decouple_fast_feedback","grad_norm_clip","weight_clamp","ephemeral_update_clamp")})
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(a.out,"episodes",episodes,"recall",result["observed_recall"],flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--code",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--task",choices=["kr","mqar"],required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--batches",type=int,default=8)
    a = ap.parse_args()
    torch.set_num_threads(4)
    measure(a)
