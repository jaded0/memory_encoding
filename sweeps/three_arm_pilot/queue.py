#!/usr/bin/env python
"""Three-arm pilot queue on Deckard GPU 1. Runs jobs with bounded concurrency, stops a run
(SIGTERM to its own pid, which saves a checkpoint) once recall_acc >= 0.99 for 3 consecutive
intervals, then runs heldout.py strict + no_fast on its latest checkpoint."""
import json, os, re, signal, subprocess, sys, time

ROOT = os.path.expanduser('~/lowrank_deep/three_arm')
CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 4
FUSED = sys.argv[2] if len(sys.argv) > 2 else 'false'
ITERS = int(os.environ.get("ITERS", "120000"))
SEEDS = [2718, 3141]
CONFIGS = [('A', 'exclusive', a) for a in ('3e4', '1e4')] + \
          [('B', 'additive_masked', a) for a in ('3e4', '1e4', '3e3')] + \
          [('C', 'additive_dense', a) for a in ('3e4', '1e4', '3e3')]
JOBS = [(f'{arm}_a{a}_s{seed}', structure, a, seed) for seed in SEEDS for arm, structure, a in CONFIGS]
if len(sys.argv) > 3:
    JOBS = [j for j in JOBS if j[0] in sys.argv[3].split(',')]


def log_path(name):
    return f'{ROOT}/runs/{name}/train.log'


def parse(name):
    """[(iter, recall_acc, loss)] per interval from the log."""
    rows, it, rec, loss = [], None, None, None
    try:
        text = open(log_path(name)).read()
    except FileNotFoundError:
        return rows
    for line in text.splitlines():
        m = re.match(r'--- Interval metrics \(ending @ iter (\d+)', line)
        if m:
            it, rec, loss = int(m.group(1)), None, None
        m = re.match(r'\s+recall_acc: (\S+)', line)
        if m and it is not None:
            rec = float(m.group(1))
        m = re.match(r'\s+loss: (\S+)', line)
        if m and it is not None:
            loss = float(m.group(1))
        if m is None and it is not None and re.match(r'\s+iters_per_sec:', line):
            rows.append((it, rec if rec is not None else 0.0, loss))
    return rows


def run_heldout(name):
    ck = f'{ROOT}/runs/{name}/ckpt/latest_checkpoint.pth'
    out = f'{ROOT}/runs/{name}/heldout.json'
    if not os.path.exists(ck):
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', HF_DATASETS_OFFLINE='1')
    cmd = (f'source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby; cd {ROOT}/code; '
           f'python heldout.py --checkpoint {ck} --protocols strict no_fast observed --batches 8 --json {out}')
    subprocess.run(['bash', '-c', cmd], env=env, stdout=open(f'{ROOT}/runs/{name}/heldout.log', 'w'),
                   stderr=subprocess.STDOUT)


def alive(p):
    if hasattr(p, 'poll'):
        return p.poll() is None
    try:
        os.kill(p, 0)
        return os.path.exists(f'/proc/{p}')
    except OSError:
        return False


def pid_of(name):
    out = subprocess.run(['pgrep', '-f', f'runs/{name}/ckpt'], capture_output=True, text=True).stdout.split()
    pids = sorted(int(x) for x in out)
    return pids[0] if pids else None


SKIP = set(os.environ.get('SKIP', '').split(','))
running = {}  # name -> [Popen or pid, stopped_early, started]
pending = []
for j in JOBS:
    if j[0] in SKIP:
        continue
    pid = pid_of(j[0])
    if pid:
        running[j[0]] = [pid, False, time.time()]
        print('adopted', j[0], pid, flush=True)
    else:
        pending.append(j)
done = []
while pending or running:
    while pending and len(running) < CONC:
        name, structure, alpha, seed = pending.pop(0)
        os.makedirs(f'{ROOT}/runs/{name}', exist_ok=True)
        p = subprocess.Popen(['setsid', f'{ROOT}/run_arm.sh', name, structure, alpha, str(seed), str(ITERS),
                              '--fused_update', FUSED], stdout=open(log_path(name), 'a'),
                             stderr=subprocess.STDOUT, preexec_fn=None)
        running[name] = [p, False, time.time()]
        print(time.strftime('%H:%M:%S'), 'started', name, flush=True)
    time.sleep(15)
    for name in list(running):
        p, stopped, started = running[name]
        rows = parse(name)
        if alive(p):
            if not stopped and len(rows) >= 3 and all(r[1] >= 0.99 for r in rows[-3:]):
                # setsid forks nothing here (exec), so p.pid is the python process after exec chain
                os.kill(p.pid if hasattr(p, 'pid') else p, signal.SIGTERM)
                running[name][1] = True
                print(time.strftime('%H:%M:%S'), 'early stop (recall>=0.99 x3)', name, flush=True)
            continue
        status = 'early_stop_recall' if stopped else 'finished_or_stopped'
        txt = open(log_path(name)).read()
        if 'Non-finite loss' in txt:
            status = 'nan'
        elif 'Early stopping: Loss' in txt:
            status = 'loss_diverged_early_stop'
        run_heldout(name)
        done.append((name, status))
        print(time.strftime('%H:%M:%S'), 'done', name, status, 'last', rows[-1] if rows else None, flush=True)
        del running[name]
print('ALL DONE', flush=True)
