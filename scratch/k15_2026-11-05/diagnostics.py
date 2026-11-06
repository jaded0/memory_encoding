"""Unscored K15 post-run diagnostics, using canonical strict held-out replay.

python scratch/k15_2026-11-05/diagnostics.py RUN/latest_checkpoint.pth \
    --device cpu --batches 2 --out RUN/diagnostics.json [--analysis RUN/analysis.json]

Histories are actual order2_write calls, not an analytic cosine proxy. They include
zero-error/masked calls: G receives a zero-valued write there, but the key-only z
still receives its key. Each key entry carries every subsequent decay; lr*alpha is its value-side scale.
Clamps invalidate the interpretation as an exact mixture of unclamped error values;
key shares remain exact diagnostics of the key-only denominator, not of G's values.
The off arm reports a hypothetical key-sum normalization, NOT its actual read.
No training, checkpoint modification, launch, or scoring decision is performed.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from ephemeral_model import order2_kernel_norm
from heldout import evaluate_held_out, load_heldout_batches, update_mask
from train import build_model, build_parser, positional_encoding
from utils import initialize_charset, load_checkpoint, read_checkpoint, upgrade_legacy_config


def load(path, device):
    checkpoint = read_checkpoint(path)
    defaults = {k: v for k, v in vars(build_parser().parse_args([])).items()
                if not k.startswith('_')}
    config = {**defaults, **upgrade_legacy_config(checkpoint.get('config', {}))}
    # Same eager order2_step as production; avoid Triton availability dependence.
    config['fused_update'] = False
    charset, _, _, n = initialize_charset(config['dataset'])
    model = build_model(config, charset, n)
    model, _, iteration, _, _ = load_checkpoint(path, model, config, device=device,
                                                checkpoint=checkpoint)
    config['pe_matrix'] = positional_encoding(config['positional_encoding_dim'], device)
    return model.to(device).eval(), config, iteration - 1


def k13_module():
    spec = importlib.util.spec_from_file_location(
        'k15_k13_analysis', ROOT / 'scratch/k13_bigkey_2026-10-27/analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(values):
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'min': None, 'max': None}
    t = torch.tensor(values, dtype=torch.float64)
    return {'count': len(values), 'mean': float(t.mean()),
            'median': float(t.quantile(.5)), 'min': float(t.min()), 'max': float(t.max())}


def relative_error(expected, actual):
    return float((expected - actual).norm() / actual.norm().clamp_min(1e-30))


@torch.no_grad()
def replay(model, config, batches, dataset, k13):
    """Instrument only this loaded model; always restore methods/hooks on exit."""
    if model.fast_iterations != 1:
        raise ValueError('K15 recipe requires fast_iterations=1; no multi-pass substitution')
    layers = {i: l for i, l in enumerate(model.linear_layers) if l.order2 is not None}
    if not layers or any(l.order2 != 'order2' for l in layers.values()):
        raise ValueError('diagnostics require actual order2 layers, not sparse/order1')
    histories, records, stats = {}, {}, {}
    for i in layers:
        records[i] = []
        stats[i] = dict(update_elements=0, update_clipped=0, weight_elements=0,
                        weight_bound=0, fast_weight_bound=0, final_bound=0,
                        final_elements=0, z_relative_errors=[], G_read_relative_errors=[])
    state = {'step': -1, 'targets': {}, 'batch': 0}
    handles, originals = [], {}

    def model_pre(_model, _inputs):
        state['step'] += 1

    def write(i, original, layer, error, learning_rate, update_clamp, forget=True):
        alpha = float(layer.order2_alpha())
        factor = float(learning_rate) * alpha
        rho = 1.0 - float(layer.forget_rate) if forget else 1.0
        if not math.isfinite(factor) or factor < 0 or not 0 <= rho <= 1:
            raise ValueError('nonnegative finite key factors and decay in [0,1] required')
        history = histories[i]
        for entry in history:
            entry['factor'] *= rho
        history.append({'step': state['step'], 'factor': rho, 'value_lr_alpha': factor,
                        'features': layer.order2_trace.detach().clone(),
                        'error': error.detach().clone()})
        # Check clamps at the exact stages order2_step applies them (before decay).
        update = -alpha * error.unsqueeze(2) * layer.order2_trace.unsqueeze(1)
        s = stats[i]
        s['update_elements'] += update.numel()
        if update_clamp > 0:
            s['update_clipped'] += int((update.abs() > update_clamp).sum())
            update.clamp_(-update_clamp, update_clamp)
        candidate = layer.order2_fast + learning_rate * update
        s['weight_elements'] += candidate.numel()
        if layer.weight_clamp > 0:
            s['weight_bound'] += int((candidate.abs() >= layer.weight_clamp).sum())
            candidate.clamp_(-layer.weight_clamp, layer.weight_clamp)
        if layer.fast_weight_clamp > 0:
            s['fast_weight_bound'] += int((candidate.abs() >= layer.fast_weight_clamp).sum())
        return original(error, learning_rate, update_clamp, forget)

    def read(i, layer, inputs, _output):
        q = state['step']
        targets = state['targets'].get(q, [])
        if not targets:
            return
        history = histories[i]
        if not history:
            raise ValueError('recall has no preceding order-two write calls')
        query = layer.order2_trace.detach().double()
        features = torch.stack([h['features'] for h in history]).double()  # [writes,B,F]
        factors = query.new_tensor([h['factor'] for h in history])
        weighted = features * factors[:, None, None]
        expected_z = weighted.sum(0)
        z = getattr(layer, 'order2_z', None)
        if z is not None:
            err = relative_error(expected_z, z.double())
            stats[i]['z_relative_errors'].append(err)
            if not torch.allclose(expected_z, z.double(), atol=1e-5, rtol=2e-4):
                raise RuntimeError('explicit history differs from actual z (factor/reset/order mismatch)')
        for b, source, answer in targets:
            masses = weighted[:, b] @ query[b]
            tolerance = 1e-10 * max(1.0, float(masses.abs().max()))
            if float(masses.min()) < -tolerance:
                raise RuntimeError('order-two key mass is negative beyond rounding tolerance')
            masses = masses.clamp_min(0)
            if not torch.isfinite(masses).all():
                raise RuntimeError('nonfinite key masses; checkpoint replay is unhealthy')
            key_mass = float(masses.sum())
            sink = 0.0
            if layer.order2_sink == 'fixed':
                sink = float(layer.order2_fraction * inputs[0][b].double().square().sum()
                             / order2_kernel_norm(layer.order2, layer.order2_temperature))
            elif layer.order2_sink != 'off':
                raise ValueError('unsupported sink; learned is unavailable in the fixed K15 recipe')
            # Match forward's epsilon, not a renormalization after adding the sink.
            denominator = max(1e-12, key_mass + sink)
            weights = masses / denominator
            conditional = masses / key_mass if key_mass > 0 else torch.zeros_like(masses)
            store_indices = [j for j, h in enumerate(history) if h['step'] == source]
            if len(store_indices) != 1:
                raise RuntimeError('desired store must correspond to exactly one actual write call')
            si = store_indices[0]
            errors = torch.stack([h['error'][b] for h in history]).double()
            value_factors = masses.new_tensor([h['value_lr_alpha'] for h in history])
            predicted_read = -(errors * (masses * value_factors)[:, None]).sum(0)
            actual_read = layer.order2_fast[b].double() @ query[b]
            gerr = relative_error(predicted_read, actual_read)
            stats[i]['G_read_relative_errors'].append(gerr)
            nonzero = errors.abs().sum(1) > 0
            active_mass = float(masses[nonzero].sum())
            positive = conditional > 0
            entropy = float(-(conditional[positive] * conditional[positive].log()).sum())
            positive_w = weights > 0
            entropy_subprob = float(-(weights[positive_w] * weights[positive_w].log()).sum())
            sink_share = sink / denominator
            entropy_with_sink = entropy_subprob - (sink_share * math.log(sink_share)
                                                   if sink_share > 0 else 0.0)
            records[i].append({
                'batch': state['batch'], 'row': b, 'query_step': q,
                'source_step': source, 'answer_index': answer,
                'normalization': layer.order2_read_norm,
                'hypothetical_normalization': layer.order2_read_norm == 'off',
                'key_mass': key_mass, 'sink_mass': sink, 'denominator': denominator,
                'weight_sum': float(weights.sum()), 'sink_share': sink_share,
                'epsilon_deficit': max(0.0, 1.0 - float(weights.sum()) - sink_share),
                'store_write_share': float(weights[si]),
                'store_write_share_conditional_on_keys': float(conditional[si]),
                'per_write_entropy_nats_conditional_on_keys': entropy,
                'per_write_entropy_nats_subprobability': entropy_subprob,
                'entropy_nats_including_sink': entropy_with_sink,
                'zero_error_write_share': float(weights[~nonzero].sum()),
                'store_share_among_nonzero_error_writes':
                    (float(masses[si]) if bool(nonzero[si]) else 0.0) / active_mass
                    if active_mass > 0 else None,
                'G_unclamped_reconstruction_relative_error': gerr,
                'per_write': [{'step': h['step'], 'decayed_key_factor': h['factor'],
                               'value_lr_alpha': h['value_lr_alpha'],
                               'mass': float(masses[j]), 'weight': float(weights[j]),
                               'error_nonzero': bool(nonzero[j])}
                              for j, h in enumerate(history)]})

    try:
        handles.append(model.register_forward_pre_hook(model_pre))
        for i, layer in layers.items():
            originals[i] = layer.order2_write
            original = originals[i]
            layer.order2_write = types.MethodType(
                lambda self, error, learning_rate, update_clamp, forget=True, i=i, original=original:
                write(i, original, self, error, learning_rate, update_clamp, forget), layer)
            handles.append(layer.register_forward_hook(
                lambda layer, inp, out, i=i: read(i, layer, inp, out)))
        episodes = 0
        for batch, (texts, onehot) in enumerate(batches):
            histories.clear()
            histories.update({i: [] for i in layers})
            targets = {}
            for b, text in enumerate(texts):
                for answer, source in k13.metadata(text, dataset):
                    targets.setdefault(answer - 1, []).append((b, source - 1, answer))
            state.update(step=-1, targets=targets, batch=batch)
            mask = update_mask('strict', texts, dataset, onehot)
            mask &= onehot[:, 1:].sum(-1) > 0
            evaluate_held_out(model, onehot, mask, config)
            episodes += len(texts)
            for i, layer in layers.items():
                # Stored G has already decayed; transform the bound accordingly.
                bounds = [c for c in (layer.weight_clamp, layer.fast_weight_clamp) if c > 0]
                stats[i]['final_elements'] += layer.order2_fast.numel()
                if bounds:
                    bound = min(bounds) * (1 - layer.forget_rate)
                    stats[i]['final_bound'] += int((layer.order2_fast.abs() >= bound).sum())
    finally:
        for handle in handles:
            handle.remove()
        for i, original in originals.items():
            layers[i].order2_write = original
    result = {}
    for i, rows in records.items():
        s = stats[i]
        clamp_bound = s['weight_bound'] + s['fast_weight_bound']
        result[f'L{i}'] = {
            'targets': len(rows),
            'summary': {key: summary([r[key] for r in rows]) for key in (
                'store_write_share', 'store_write_share_conditional_on_keys',
                'per_write_entropy_nats_conditional_on_keys',
                'per_write_entropy_nats_subprobability', 'entropy_nats_including_sink',
                'sink_mass', 'sink_share', 'weight_sum', 'zero_error_write_share')},
            'clamps': {
                'update_clipped_fraction': s['update_clipped'] / max(1, s['update_elements']),
                'weight_bound_fraction_before_forget': s['weight_bound'] / max(1, s['weight_elements']),
                'fast_weight_bound_fraction_before_forget': s['fast_weight_bound'] / max(1, s['weight_elements']),
                'G_bound_fraction_at_batch_end': s['final_bound'] / max(1, s['final_elements']),
                'exact_unclamped_value_interpretation_valid_on_replay':
                    not (s['update_clipped'] or clamp_bound),
                'population': 'all strict replay writes, not historical training writes'},
            'z_relative_error': summary(s['z_relative_errors']),
            'G_unclamped_reconstruction_relative_error': summary(s['G_read_relative_errors']),
            'records': rows}
    return {'episodes': episodes, 'layers': result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='latest_checkpoint.pth (never best checkpoint)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--analysis', help='reuse K13 analysis.json kernels/layers for key ratios')
    args = parser.parse_args(argv)
    if args.batches < 1:
        parser.error('--batches must be positive')
    torch.set_num_threads(4)
    start = time.monotonic()
    model, config, iteration = load(args.checkpoint, args.device)
    dataset = args.dataset or config['dataset']
    if initialize_charset(dataset)[0] != initialize_charset(config['dataset'])[0]:
        raise ValueError('evaluation charset differs from checkpoint')
    k13 = k13_module()
    batches = load_heldout_batches(dataset, config['batch_size'], args.batches, args.device, 'test')
    result = replay(model, config, batches, dataset, k13)
    if args.analysis:
        analysis = json.loads(Path(args.analysis).read_text())
        source_hash = analysis.get('checkpoint_sha256')
        if source_hash and source_hash != hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest():
            raise ValueError('--analysis checkpoint hash does not match this checkpoint')
        matches = [r for r in analysis.get('results', []) if r.get('dataset') == dataset]
        if len(matches) != 1:
            raise ValueError('--analysis must contain exactly one matching dataset')
        ratios = {'source': str(Path(args.analysis).resolve()), 'reused': True,
                  'source_checkpoint': analysis.get('checkpoint'), 'source_split': analysis.get('split'),
                  'layers': matches[0].get('layers'), 'kernels': matches[0].get('kernels')}
    else:
        # Only the strict replay's K13 observer/geometry, not the scoring or ablation suite.
        observer, kernels = k13.sparse.make_observer(k13.o2, layers=(0, 1, 2))
        geometry, pooled = {}, {}
        for texts, onehot in batches:
            k13.strict_replay(model, config, texts, onehot, dataset, observer, geometry, pooled)
        ratios = {'source': 'K13 strict_replay on these test batches', 'reused': False,
                  'layers': k13.summarize_geometry(geometry, pooled),
                  'kernels': k13.summarize_kernels(kernels)}
    # K13 ratios can diverge when the median wrong-key mass is zero. Preserve
    # that fact as null + explicit paths, rather than writing non-standard JSON.
    nonfinite_ratios = []
    def finite_ratios(value, path='trunk_key_ratios_K13'):
        if isinstance(value, dict):
            return {k: finite_ratios(v, path + '.' + k) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_ratios(v, path + f'[{i}]') for i, v in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite_ratios.append(path)
            return None
        return value
    ratios = finite_ratios(ratios)
    output = {'checkpoint': str(Path(args.checkpoint).resolve()), 'iteration': iteration,
              'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                       text=True).strip(),
              'dataset': dataset, 'split': 'test', 'device': args.device,
              'unscored': True, 'training_modified': False,
              'definition': 'actual psi history, key-side factor 1, add then same G decay; lr*alpha is value-side; masked zero-error calls retain keys',
              'clamp_caveat': 'Binding update/weight/fast-weight clamps invalidate exact error-value mixture interpretation; shares describe key-only mass. Replay clamp statistics are not historical training clamp statistics.',
              'entropy_definition': 'natural log; conditional-on-keys and subprobability entropy both reported; including-sink treats sink as an extra category',
              'trunk_key_ratios_K13': ratios, 'nonfinite_K13_ratio_fields': nonfinite_ratios, **result,
              'elapsed_seconds': time.monotonic() - start}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'out': str(path), 'iteration': iteration, 'episodes': result['episodes']}))


if __name__ == '__main__':
    main()
