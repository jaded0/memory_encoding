"""Minimal from-scratch nn.RNN/GRU baseline on the repo's dataset tensors."""
import sys, argparse, time, torch, random, numpy as np
import torch.nn as nn
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from utils import initialize_charset
from bpttdbg_data import subset_loader
from metrics import recall_targets
p = argparse.ArgumentParser()
p.add_argument('--dataset', default='long_range_memory_dataset')
p.add_argument('--cell', default='gru'); p.add_argument('--opt', default='adam')
p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--hidden', type=int, default=128)
p.add_argument('--iters', type=int, default=3000); p.add_argument('--bs', type=int, default=16)
p.add_argument('--seed', type=int, default=2718); p.add_argument('--every', type=int, default=500)
a = p.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
charset, _, _, n = initialize_charset(a.dataset)
dl = subset_loader(a.dataset, a.bs, a.seed)
cell = {'gru': nn.GRU, 'rnn': nn.RNN, 'lstm': nn.LSTM}[a.cell](n, a.hidden, batch_first=True)
head = nn.Linear(a.hidden, n)
params = list(cell.parameters()) + list(head.parameters())
opt = (torch.optim.Adam if a.opt == 'adam' else torch.optim.SGD)(params, lr=a.lr)
it = 0; corr = tot = 0; t0 = time.time(); losses = []
while it < a.iters:
    for texts, idx, oh in dl:
        it += 1
        h, _ = cell(oh[:, :-1])
        logits = head(h)
        tgt = oh[:, 1:]
        loss = -(tgt * logits.log_softmax(-1)).sum(-1).sum(1).mean()  # same as train.py: sum over steps, mean over batch
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        preds = logits.argmax(-1)
        for b, t in enumerate(texts):
            for pos, lag in recall_targets(t, a.dataset)[0].items():
                tot += 1; corr += int(preds[b, pos - 1].item() == idx[b, pos].item())
        if it % a.every == 0:
            print(f"{it} loss {np.mean(losses):.3f} recall {corr / max(tot, 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
            corr = tot = 0; losses = []
        if it >= a.iters:
            break
