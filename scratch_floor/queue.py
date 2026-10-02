import subprocess, time, os, re, signal, sys, shlex
root = os.path.expanduser("~/lowrank_deep/floor")
maxpar = int(sys.argv[1]) if len(sys.argv) > 1 else 4
started, running = set(), {}   # running: name -> (proc, gpu)

def recalls(name):
    try:
        txt = open(f"{root}/runs/{name}/train.log").read()
    except Exception:
        return []
    return [float(x) for x in re.findall(r"^  recall_acc: ([0-9.]+)", txt, re.M)]

def read_jobs():
    jobs = []
    for line in open(f"{root}/jobs.txt"):
        line = line.strip()
        if line and not line.startswith("#"):
            jobs.append(shlex.split(line))
    return jobs

while True:
    for name, (p, gpu) in list(running.items()):
        r = recalls(name)
        done = p.poll() is not None
        if not done and len(r) >= 3 and all(x >= 0.99 for x in r[-3:]):
            os.killpg(p.pid, signal.SIGTERM); time.sleep(5)
            open(f"{root}/runs/{name}/EARLYSTOP", "w").write("recall>=0.99 x3\n"); done = True
        if done:
            print(time.ctime(), "finished", name, flush=True); del running[name]
    pending = [j for j in read_jobs() if j[0] not in started]
    if os.path.exists(f"{root}/STOP"):
        pending = []
    while pending and len(running) < maxpar:
        job = pending.pop(0); name = job[0]
        gpus = [g for g in (0, 1)]
        gpu = min(gpus, key=lambda g: sum(1 for _, (_, gg) in running.items() if gg == g))
        os.makedirs(f"{root}/runs/{name}", exist_ok=True)
        log = open(f"{root}/runs/{name}/train.log", "w")
        running[name] = (subprocess.Popen(["bash", f"{root}/run_arm.sh", name, job[1], job[2], job[3], str(gpu)] + job[4:],
                         stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid), gpu)
        started.add(name); print(time.ctime(), "start", name, "gpu", gpu, flush=True)
    if not running and not pending and os.path.exists(f"{root}/STOP"):
        break
    time.sleep(20)
