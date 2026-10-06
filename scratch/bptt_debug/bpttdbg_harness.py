import sys, argparse, time, torch, random, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import train as T
from utils import initialize_charset
from bpttdbg_data import subset_loader
from metrics import recall_targets
p = argparse.ArgumentParser()
p.add_argument('--dataset', default='long_range_memory_dataset')
p.add_argument('--model_type', default='rnn'); p.add_argument('--updater', default='bptt')
p.add_argument('--opt', default='sgd'); p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--hidden', type=int, default=128); p.add_argument('--layers', type=int, default=3)
p.add_argument('--iters', type=int, default=3000); p.add_argument('--bs', type=int, default=16)
p.add_argument('--clip', type=float, default=0); p.add_argument('--rec', type=int, default=1)
p.add_argument('--seed', type=int, default=2718); p.add_argument('--every', type=int, default=500)
p.add_argument('--plasticity', type=float, default=1e5)
p.add_argument('--gradprobe', type=int, default=0); p.add_argument('--residual', type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
charset, c2i, i2c, n = initialize_charset(a.dataset)
dl = subset_loader(a.dataset, a.bs, a.seed)
config = dict(learning_rate=a.lr, plasticity=a.plasticity, forget_rate=0.01, ephemeral_update_clamp=0,
  grad_norm_clip=a.clip, n_hidden=a.hidden, n_layers=a.layers, dataset=a.dataset, model_type=a.model_type,
  updater=a.updater, batch_size=a.bs, input_mode='last_one', ephemeral_fraction=0.1, enable_recurrence=bool(a.rec),
  positional_encoding_dim=0, residual_connection=bool(a.residual), unit_norm_weights=False, weight_clamp=0,
  slow_weight_decay=0, output_tanh=False, fast_weight_clamp=0, criterion=torch.nn.CrossEntropyLoss(reduction='none'),
  pe_matrix=None)
rnn = T.build_model(config, charset, n)
opt = None
if a.updater in ('backprop', 'bptt'):
    opt = (torch.optim.Adam if a.opt == 'adam' else torch.optim.SGD)(rnn.parameters(), lr=a.lr)
state = dict(training_instance=0, last_n_rewards=[0], last_n_reward_avg=0, wandb_step=0, log_norms_now=False)
it = 0; corr = tot = 0; losses = []; t0 = time.time()
insz = n
while it < a.iters:
    for texts, idx, oh in dl:
        it += 1
        out, loss, preds, sl, _, _ = T.train(idx, oh, rnn, config, state, opt)
        losses.append(loss)
        for b, t in enumerate(texts):
            tg, _ = recall_targets(t, a.dataset)
            for pos, lag in tg.items():
                tot += 1; corr += int(preds[pos - 1, b].item() == idx[b, pos].item())
        if it % a.every == 0:
            msg = f"{it} loss {np.mean(losses):.4f} recall {corr / max(tot, 1):.3f} ({time.time() - t0:.0f}s)"
            if a.gradprobe and a.model_type == 'rnn':
                W = rnn.linear_layers[0].weight
                g = W.grad
                if g is not None:
                    msg += f" |g_in| {g[:, :insz].norm():.2e} |g_hid| {g[:, insz:].norm():.2e} |W_hid| {W[:, insz:].norm():.2e}"
            print(msg, flush=True)
            corr = tot = 0; losses = []
        if it >= a.iters:
            break
