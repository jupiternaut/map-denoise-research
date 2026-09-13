"""Exposed-development evaluation; estimator calls finish before loading truth."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RUNS = Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
INPUTS = RUNS/'repair-v2-oyuie4pl/synthetics/identifiable'
sys.path[:0] = [str(PROJECT), str(PROJECT/'exploration_v3')]
from evaluate_v2 import synthetic_geometry
from metrics import structure_metrics

PRIMARY = ('dual_g4_s912101_b4', 'dual_g8_s912101_b4', 'ghost_s912101_b4')
CASES = tuple(f'{family}_s{seed}_b{bias}' for seed in (912101, 912113, 912127)
              for family in ('ghost', 'dual_g2', 'dual_g4', 'dual_g8') for bias in (0, 4))
METRICS = ('surface_accuracy_mean_mm', 'matched_point_rms_mm',
           'fitted_gap_at_same_xy_error_mm')

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def csv_save(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def load_model():
    spec = importlib.util.spec_from_file_location('v7_legal_reassociation', HERE/'algorithm/reassociation.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def aggregates(rows):
    output = {}
    subsets = {'all_public': rows, 'dual_public': [r for r in rows if r['gap_mm'] > 0]}
    subsets.update({f'gap_{gap}': [r for r in rows if r['gap_mm'] == gap] for gap in (0, 2, 4, 8)})
    for name, subset in subsets.items():
        methods = {}
        for method in sorted({r['method'] for r in subset}):
            values = [r for r in subset if r['method'] == method]
            record = {'input_count': len(values)}
            for metric in (*METRICS, 'input_edit_rms_mm', 'measured_fit_seconds',
                           'shared_freeze_plus_fit_seconds', 'group_count', 'supported_fraction'):
                numbers = [r[metric] for r in values if metric in r and r[metric] is not None]
                if numbers: record[metric] = float(np.mean(numbers))
            methods[method] = record
        output[name] = methods
    return output

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary-only', action='store_true')
    parser.add_argument('--budgets', type=int, nargs='+', default=[1, 3, 6])
    args = parser.parse_args()
    if socket.gethostname() != 'liekkas': raise RuntimeError('wrong target host')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE'):
        if os.environ.get(key) != '1': raise RuntimeError(key+' must be 1')
    if any(b < 1 for b in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError('distinct positive budgets required')
    cases = PRIMARY if args.primary_only else CASES
    model = load_model()
    sources = [Path(__file__), HERE/'PROTOCOL.md', *sorted((HERE/'algorithm').glob('*.py')),
               PROJECT/'exploration_v5/slope_pooling.py', PROJECT/'exploration_v4/surface_pooling.py',
               PROJECT/'exploration_v3/graph_surface.py', PROJECT/'evaluate_v2.py',
               PROJECT/'exploration_v3/metrics.py']
    before = {str(p): digest(p) for p in sources}
    dest = Path(tempfile.mkdtemp(prefix='reassociation-v7-primary-' if args.primary_only else 'reassociation-v7-development-', dir=RUNS))
    print(dest, flush=True)
    for name in ('source', 'outputs', 'states'):
        (dest/name).mkdir()
    for source in sources:
        target = dest/'source'/source.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    save(dest/'SOURCE_MANIFEST.json', before)
    save(dest/'CONFIG.json', dict(host=socket.gethostname(), cases=cases, budgets=args.budgets,
         data_role='previously exposed development', sigma_mm=1., gt_in_estimator=False,
         nonfinite_diagnostic_encoding='null; primary scores must be finite',
         timing_scope='one shared upstream freeze per case; each reported arm independently refits'))
    started = time.perf_counter()
    rows, manifest, replays = [], [], []
    for case in cases:
        input_path = INPUTS/(case+'.npz')
        ev_path = INPUTS/'evaluation'/(case+'.eval.npz')
        with np.load(input_path, allow_pickle=False) as data:
            world, scans = data['xyz_world'].copy(), data['scan_id'].copy()
            point_ids = data['source_point_index'].copy()
        original_world, original_scans = world.copy(), scans.copy()
        tick = time.perf_counter()
        state = model.freeze(world, scans, 1.)
        freeze_seconds = time.perf_counter()-tick
        state_arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
        with (dest/'states'/(case+'.npz')).open('xb') as stream:
            np.savez_compressed(stream, **state_arrays)
        calls = [('original', 0, sharing) for sharing in ('independent', 'shared')]
        calls += [(variant, b, sharing) for b in args.budgets
                  for variant in ('local_multistart', 'reassociate') for sharing in ('independent', 'shared')]
        completed = [('identity', world.copy(), {}, {'support_mask': np.zeros(len(world), bool)}, 0.)]
        for variant, budget, sharing in calls:
            name = f'{variant}_b{budget}_{sharing}'
            tick = time.perf_counter()
            result, info, artifacts = model.fit_frozen(state, variant=variant, budget=max(budget, 1), sharing=sharing)
            elapsed = time.perf_counter()-tick
            if result.shape != world.shape or not np.isfinite(result).all():
                raise AssertionError('invalid output '+name)
            if not np.array_equal(world, original_world) or not np.array_equal(scans, original_scans):
                raise AssertionError('estimator modified input')
            support = np.asarray(artifacts['support_mask'], bool)
            if not np.array_equal(result[~support], world[~support]):
                raise AssertionError('unsupported output changed')
            completed.append((name, result, info, artifacts, elapsed))
        # Evaluator boundary: no case truth was loaded before all legal estimates completed.
        with np.load(ev_path, allow_pickle=False) as data:
            ev = {k: data[k].copy() for k in data.files if k != 'json'}
            if 'json' in data: ev.update(json.loads(data['json'].tobytes().decode()))
        meta = json.loads((INPUTS/(case+'.json')).read_text())
        labels = np.asarray(ev['gt_layer'])
        for name, result, info, artifacts, elapsed in completed:
            support = np.asarray(artifacts['support_mask'], bool)
            row = dict(case=case, method=name, seed=int(meta['seed']), gap_mm=float(meta['gap_mm']),
                       bias_rms_mm=float(meta['bias_rms_mm']), n_points=len(world),
                       primary=case in PRIMARY, stage='public_development',
                       supported_fraction=float(support.mean()), measured_fit_seconds=elapsed,
                       freeze_seconds=freeze_seconds if name != 'identity' else 0.,
                       shared_freeze_plus_fit_seconds=freeze_seconds+elapsed if name != 'identity' else 0.,
                       input_edit_rms_mm=float(1000*np.sqrt(np.mean(np.sum((result-world)**2, axis=1)))))
            # Preserve all implementation counters; do not assume one iteration is equal cost.
            row.update({'info_'+k: clean(v) for k, v in info.items()
                        if isinstance(v, (str, int, float, bool, np.generic))})
            groups = artifacts.get('group_ids')
            composition = []
            if groups is not None:
                groups = np.asarray(groups)
                used = np.unique(groups[groups >= 0])
                row['group_count'] = len(used)
                wrong, assigned = 0, 0
                for group in used:
                    take = groups == group
                    counts = {str(int(k)): int(np.sum(labels[take] == k)) for k in np.unique(labels)}
                    wrong += int(take.sum())-max(counts.values())
                    assigned += int(take.sum())
                    composition.append(dict(group=int(group), counts=counts))
                row['evaluation_only_weightless_group_impurity'] = wrong/max(assigned, 1)
                if len(np.unique(labels)) == 1:
                    row['single_surface_fragment_groups'] = len(used)
            row.update(synthetic_geometry(result, {}, ev))
            row.update(structure_metrics(result, ev))
            for metric in METRICS:
                if metric in row and not np.isfinite(row[metric]):
                    raise AssertionError('nonfinite primary metric')
            target = dest/'outputs'/(case+'__'+name+'.npz')
            arrays = {k: np.asarray(v) for k, v in artifacts.items() if isinstance(v, (np.ndarray, list, tuple))}
            with target.open('xb') as stream:
                np.savez_compressed(stream, xyz_world=result, scan_id=scans, source_point_index=point_ids, **arrays)
            row.update(output=str(target), output_sha256=digest(target), ok=True)
            save(target.with_suffix('.json'), dict(info=info, row=row, evaluation_only_composition=composition))
            rows.append(row)
            if name.startswith('original_b0_'):
                sharing = name.rsplit('_', 1)[1]
                old = RUNS/'association-v6-bgcq8zff/outputs'/(case+'__original_'+sharing+'.npz')
                with np.load(old, allow_pickle=False) as data:
                    prior = data['xyz_world']
                delta = float(np.max(abs(result-prior)))
                replays.append(dict(case=case, method=name, reference=str(old), max_abs_m=delta,
                                    bitwise_equal=bool(np.array_equal(result, prior))))
                if delta > 1e-12: raise AssertionError('fixed-arm replay mismatch')
        manifest.append(dict(case=case, input=str(input_path), input_sha256=digest(input_path),
                             evaluation=str(ev_path), evaluation_sha256=digest(ev_path)))
        print(case, str(len(completed))+'/'+str(len(completed)), flush=True)
    csv_save(dest/'RESULTS.csv', rows)
    save(dest/'AGGREGATES.json', aggregates(rows))
    save(dest/'INPUT_MANIFEST.json', manifest)
    save(dest/'REPRODUCTION.json', replays)
    after = {p: digest(p) for p in before}
    save(dest/'SOURCE_AFTER.json', after)
    if before != after: raise AssertionError('source changed during experiment')
    summary = dict(run_dir=str(dest), host=socket.gethostname(), input_count=len(cases),
                   outputs=len(rows), primary_only=args.primary_only, new_confirmation=False,
                   source_unchanged=True, original_replays=len(replays),
                   elapsed_seconds=time.perf_counter()-started,
                   peak_parent_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   resource_scope='one process CPU run; no GPU; parent RSS only',
                   algorithm_success='not determined by successful execution; see comparative report')
    save(dest/'SUMMARY.json', summary)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__': run()
