"""Conservative four-lane planning estimate, not an authorization or GPU benchmark.

ROOT must contain pilot/, timing/{par2,spar}.{log,status}. Run after both bounded
timing smokes finish. Rates are A6000 lane seconds; no empirical A100 speed claim.
"""
import argparse
import json
from pathlib import Path
import re
import statistics


def estimate(root, iterations):
    root = Path(root).resolve()
    pilot = root / 'pilot/mqar2_16_q4-lin-s2718'
    run = json.loads((pilot / 'pilot_run.json').read_text())
    lin_rate = run['training_seconds'] / run['requested_iterations']
    measurements = {}
    for arm in ('par2', 'spar'):
        status = (root / f'timing/{arm}.status').read_text()
        match = re.fullmatch(r'TIMING EXIT 0 ELAPSED (\d+) REQUESTED_ITERS 1000\s*', status)
        if not match:
            raise ValueError(f'{arm} timing incomplete or failed: {status}')
        log = root / f'timing/{arm}.log'
        rates = [float(v) for v in re.findall(r'^\s+iters_per_sec:\s+([\d.eE+-]+)', log.read_text(), re.M)]
        if len(rates) < 3 or min(rates) <= 0:
            raise ValueError(f'{arm} missing measured iteration rates')
        measurements[arm] = dict(elapsed_seconds=int(match[1]), requested_iterations=1000,
                                 steady_seconds_per_iter=1 / statistics.median(rates[2:]),
                                 seconds_per_iter=max(int(match[1]) / 1000,
                                                      1 / statistics.median(rates[2:])),
                                 evidence=str(log))
    worst = max(measurements, key=lambda a: measurements[a]['seconds_per_iter'])
    arms = {}
    for arm in ('lin', 'par2', 'par1', 'rep2', 'spar'):
        if arm == 'lin':
            rate, evidence = lin_rate * 1.2, str(pilot / 'train.log')
            explanation = 'Measured full pilot walltime/200000, including validation/checkpoints; 20% reserve. Pilot shared GPU with K4 until K4 finished.'
        else:
            source = arm if arm in measurements else worst
            rate = max(lin_rate, measurements[source]['seconds_per_iter'] * 2) * 1.2
            evidence = measurements[source]['evidence']
            explanation = f'Measured solo {source} timing; 2x for two lanes/GPU, 20% reserve, at least pilot lin rate. par1/rep2 conservatively charged at slowest measured transformed arm.'
        arms[arm] = dict(seconds_per_iter=rate, scoring_seconds=1800,
                         evidence=evidence, basis='conservative_estimate', explanation=explanation)
    cost = max(iterations * a['seconds_per_iter'] + a['scoring_seconds'] for a in arms.values())
    seed_cost = sum(iterations * a['seconds_per_iter'] + a['scoring_seconds'] for a in arms.values())
    return dict(hardware='deckard 2x RTX A6000; four lanes, two/GPU', dataset='mqar2_16_q4',
                iterations=iterations, core_runs=25, launch_approved=False, arms=arms,
                measurements=measurements, pilot_lin_seconds_per_iter=lin_rate,
                most_expensive_measured_arm=worst,
                scoring_basis='1800s/run reserve: measured K16 full pilot analysis is about 37s; final scoring adds 128 batches/protocol and transformed-arm overhead. Not directly timed final scoring.',
                conversion=dict(a100_equivalent_hours_per_a6000_lane_hour=1,
                                basis='Conservative budget-accounting parity ONLY, not an empirical hardware speed ratio. ORC pilot never ran; no measured A100 comparison exists. Cap is not declared passed.',
                                cap_a100_hours=40),
                ideal_four_lane_wall_hours=25 * cost / 4 / 3600,
                seven_wave_wall_hours=7 * cost / 3600,
                seed_block_paired_wall_hours=10 * cost / 3600,
                arm_specific_ideal_four_lane_wall_hours=5 * seed_cost / 4 / 3600,
                arm_specific_seed_block_paired_wall_hours=2 * seed_cost / 3600,
                envelope_basis='Primary fields conservatively charge all25 runs at the slowest arm, matching frozen gate policy. arm_specific fields sum the individual arm estimates instead.',
                schedule='Five complete five-arm seed blocks on four lanes; one lane gets two seed blocks (10 runs); others 5. Fully paired partial results.',
                total_a6000_lane_hours=25 * cost / 3600,
                lin_only_unpadded_ideal_lower_bound_hours=25 * iterations * lin_rate / 4 / 3600,
                within_48_hours=10 * cost / 3600 <= 48,
                selection_reason='K16 only scientific pass. Onset 15000; add 180000 to earliest point within .02 of final (195000). Final two differ .0059 <= .03. No arbitrary 200k default.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--iterations', type=int, default=195000)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    Path(args.out).write_text(json.dumps(estimate(args.root, args.iterations), indent=2, allow_nan=False) + '\n')
