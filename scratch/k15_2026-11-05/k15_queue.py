"""Locked K15 queue, retries and global bigkey/KRX barrier (CPU only)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile

BIG_SEEDS = (2718, 3141, 1618, 1414, 1732)
KR_SEEDS = (2718, 3141, 1618, 1414, 5772)
BIG_ARMS = ('sum2-t4', 'sink2-t4', 'sum2-t16', 'par2-t16')


def initial(krx=True):
    # Seed round-robin within each arm, in the registered arm priority order.
    tasks = [f'bigkey {a} {s}' for a in BIG_ARMS for s in BIG_SEEDS]
    if krx:
        tasks += [f'krx {a} {s}' for a in ('rep2', 'sum2-rep') for s in KR_SEEDS]
    return tasks


def atomic(path, text):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.queue-')
    with os.fdopen(fd, 'w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(name, path)


def event(root, text):
    import datetime
    with (root / 'status.txt').open('a') as f:
        f.write(f'{text} {datetime.datetime.now().astimezone().isoformat()}\n')


def operate(root, k13, action, lane, rc=None):
    root, k13 = Path(root), Path(k13)
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'queue_state.json'
        state = json.loads(path.read_text())
        # State is authoritative; queue.txt is a human-readable pending projection.
        old = k13.read_text()
        for seed in BIG_SEEDS:
            task = f'bigkey par2-t4 {seed}'
            if re.search(rf'^GIVEUP par2 s{seed}(?: |$)', old, re.M) and task not in state['reuse_added']:
                state['pending'].insert(0, task)
                state['reuse_added'].append(task)
                event(root, f'REUSE-RERUN par2-t4 s{seed} K13-GIVEUP')
        key = str(lane)
        if action == 'finish':
            task = state['active'].pop(key)
            phase, arm, seed = task.split()
            attempt = state['attempts'][task]
            event(root, f'EXIT {rc} {arm} s{seed} lane{lane} phase={phase} attempt{attempt}')
            if rc:
                if attempt < 4:  # at most three requeues, four total attempts
                    state['pending'].append(task)
                    event(root, f'REQUEUE {arm} s{seed} after attempt{attempt}')
                else:
                    state['giveup'].append(task)
                    event(root, f'GIVEUP {arm} s{seed} after attempt{attempt}')
            else:
                state['done'].append(task)
            result = 'OK'
        else:
            if key in state['active']:
                raise RuntimeError('Lane has an unfinished lease; inspect before any manual recovery')
            big_active = any(t.startswith('bigkey ') for t in state['active'].values())
            big_pending = [t for t in state['pending'] if t.startswith('bigkey ')]
            all_k13_done = all(re.search(rf'^LANE DONE lane{i} \S+$', old, re.M) for i in range(4))
            if big_pending:
                task = big_pending[0]
            elif big_active or not all_k13_done:
                task = None
            else:
                task = state['pending'][0] if state['pending'] else None
            if task:
                state['pending'].remove(task)
                state['active'][key] = task
                attempt = state['attempts'].get(task, 0) + 1
                state['attempts'][task] = attempt
                phase, arm, seed = task.split()
                event(root, f'START {arm} s{seed} lane{lane} gpu{lane % 2} attempt{attempt} phase={phase}')
                result = task
            elif state['pending'] or state['active'] or not all_k13_done:
                result = 'WAIT'
            else:
                result = 'DONE'
        atomic(path, json.dumps(state, indent=2) + '\n')
        atomic(root / 'queue.txt', ''.join(t + '\n' for t in state['pending']))
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['claim', 'finish'])
    p.add_argument('--root', required=True)
    p.add_argument('--k13-status', required=True)
    p.add_argument('--lane', type=int, choices=range(4), required=True)
    p.add_argument('--rc', type=int)
    a = p.parse_args()
    if a.action == 'finish' and a.rc is None:
        p.error('finish requires --rc')
    print(operate(a.root, a.k13_status, a.action, a.lane, a.rc))


if __name__ == '__main__':
    main()
