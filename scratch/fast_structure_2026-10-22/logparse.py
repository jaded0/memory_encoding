"""Parse train.py logs: the interval-metric blocks ("--- Interval metrics (ending @ iter N, ...) ---"
followed by "  key: value" lines)."""
import math
import re

BLOCK = re.compile(r"--- Interval metrics \(ending @ iter (\d+)")
LINE = re.compile(r"^\s+([\w/.\-]+): (\S+)$")


def intervals(path):
    """[(iteration, {metric: float})] in log order. A resumed run repeats iterations: the later wins."""
    out, current = {}, None
    with open(path, errors="replace") as handle:
        for line in handle:
            match = BLOCK.search(line)
            if match:
                current = {}
                out[int(match.group(1))] = current
                continue
            if current is not None:
                match = LINE.match(line.rstrip("\n"))
                if match:
                    try:
                        current[match.group(1)] = float(match.group(2))
                    except ValueError:
                        pass
                elif line.startswith("---") or not line.strip():
                    current = None if line.startswith("-----") else current
    return sorted(out.items())


def blowup_onset(rows, threshold=5.0, run=5):
    """First iteration of `run` adjacent windows with loss > threshold (or nonfinite), else None (V2 definition)."""
    streak = []
    for iteration, metrics in rows:
        loss = metrics.get("loss", float("nan"))
        if not math.isfinite(loss) or loss > threshold:
            streak.append(iteration)
            if len(streak) >= run:
                return streak[0]
        else:
            streak = []
    return None


def memory_onset(rows, threshold=0.5):
    for iteration, metrics in rows:
        if metrics.get("recall_acc", 0) > threshold:
            return iteration
    return None
