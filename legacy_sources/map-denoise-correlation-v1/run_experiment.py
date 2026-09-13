"""Frozen-method paired probes; outputs are persisted before geometry truth is read."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Nonfinite JSON value')
    return value


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(plain(value), handle, ensure_ascii=False, indent=2, allow_nan=False)


def selected_k(info):
    if 'k' in info:
        return int(info['k'])
    nested = info.get('fit', {}).get('0', {})
    return int(nested['k']) if 'k' in nested else None


def evaluate(output, observed, truth, gap, frame):
    clean = truth['clean_xyz_mm']
    labels = truth['labels'].astype(int)
    err = output - clean
    normal = err[:, 2]
    result = {
        'normal_mae_mm': float(np.mean(np.abs(normal))),
        'normal_rmse_mm': float(np.sqrt(np.mean(normal**2))),
        'point_rmse_mm': float(np.sqrt(np.mean(np.sum(err**2, axis=1)))),
        'signed_normal_mean_mm': float(normal.mean()),
        'centered_normal_mae_mm': float(np.mean(np.abs(normal - normal.mean()))),
        'displacement_mae_mm': float(np.mean(np.linalg.norm(output - observed, axis=1))),
        'frame_mean_error_std_mm': float(np.std([normal[frame == f].mean() for f in np.unique(frame)])),
        'output_normal_std_mm': float(np.std(output[:, 2])),
        'layer_gap_hat_mm': None,
        'layer_gap_error_mm': None,
        'gap_retention': None,
        'half_gap_error_fraction': None,
        'layer_separation_lost': None,
    }
    if gap > 0 and set(np.unique(labels)) == {0, 1}:
        # Subtract the known per-point clean shape only in the evaluator. This
        # also avoids mistaking unequal curved-surface sampling for true gap.
        gap_delta = normal[labels == 1].mean() - normal[labels == 0].mean()
        gap_hat = float(gap + gap_delta)
        result.update(layer_gap_hat_mm=gap_hat, layer_gap_error_mm=float(abs(gap_delta)),
                      gap_retention=gap_hat / gap,
                      half_gap_error_fraction=float(np.mean(np.abs(normal) > gap / 2)),
                      layer_separation_lost=bool(gap_hat < gap / 2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modes', default='base_sigma,total_scale')
    parser.add_argument('--methods')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if platform.node() != 'liekkas':
        raise RuntimeError('Target host mismatch')
    import operators

    methods = tuple(args.methods.split(',')) if args.methods else operators.METHODS
    if not set(methods).issubset(operators.METHODS):
        raise ValueError('Unknown method')
    modes = args.modes.split(',')
    if not set(modes).issubset({'base_sigma', 'total_scale'}):
        raise ValueError('Unknown noise-scale protocol')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    entries = manifest['entries'][:args.limit] if args.limit else manifest['entries']
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'outputs').mkdir()
    source_hashes = operators.source_hashes()
    source_hashes.update({str(p): sha(p) for p in
                         [Path(__file__), ROOT / 'PROTOCOL.md', ROOT / 'generate_cases.py']})
    started = time.perf_counter()
    warm_start = time.perf_counter()
    # Warm every method on the same unscored small input. This is not a timing
    # benchmark; per-call timings later are descriptive single executions.
    rng = np.random.default_rng(7719)
    warm_xyz = np.c_[rng.normal(0, 15, (64, 2)), rng.normal(0, 1, 64)]
    warm_frame = np.repeat(np.arange(4), 16)
    warm_info = []
    for method in methods:
        t0 = time.perf_counter()
        operators.estimate(method, warm_xyz, warm_frame, 1.)
        warm_info.append({'method': method, 'seconds': time.perf_counter() - t0})
    run_manifest = {
        'host': platform.node(), 'python': sys.version, 'numpy': np.__version__,
        'data': str(args.data.resolve()), 'data_manifest_sha256': sha(args.data / 'manifest.json'),
        'methods': methods, 'modes': modes, 'entries': len(entries),
        'source_hashes': source_hashes, 'warmup_seconds': time.perf_counter() - warm_start,
        'warmup_methods': warm_info,
        'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'scope': 'exploratory paired perturbation probe; same frozen methods; not an external SOTA benchmark',
        'timing_scope': 'single warmed method call including adapter; no file I/O or evaluation; not a rigorous speed ranking',
    }
    write_json(args.out / 'manifest.json', run_manifest)
    rows = []
    for index, entry in enumerate(entries):
        input_path = args.data / entry['input_path']
        input_hash = sha(input_path)
        if entry.get('input_sha256') and entry['input_sha256'] != input_hash:
            raise ValueError('Input hash mismatch')
        with np.load(input_path, allow_pickle=False) as data:
            if set(data.files) != {'xyz_mm', 'frame', 'sigma_mm'}:
                raise ValueError('Unexpected legal input fields')
            xyz = data['xyz_mm'].copy()
            frame = data['frame'].copy()
            base_sigma = float(data['sigma_mm'])
        for mode in modes:
            sigma = base_sigma if mode == 'base_sigma' else float(np.hypot(base_sigma, entry['amplitude_mm']))
            for method in methods:
                row = {**entry, 'method': method, 'sigma_protocol': mode,
                       'supplied_sigma_mm': sigma, 'input_sha256': input_hash}
                t0 = time.perf_counter()
                try:
                    before = xyz.copy()
                    output, info = operators.estimate(method, xyz, frame, sigma)
                    elapsed = time.perf_counter() - t0
                    if not np.array_equal(xyz, before):
                        raise AssertionError('Input mutated')
                    if output.shape != xyz.shape or not np.isfinite(output).all():
                        raise ValueError('Invalid output')
                    tag = f'{entry["id"]}__{mode}__{method}'
                    out_path = args.out / 'outputs' / f'{tag}.npz'
                    np.savez_compressed(out_path, xyz_mm=output)
                    write_json(out_path.with_suffix('.json'), info)
                    # Ground truth is first read only after estimator output is saved.
                    with np.load(args.data / entry['eval_path'], allow_pickle=False) as data:
                        truth = {k: data[k].copy() for k in data.files}
                    with np.load(out_path, allow_pickle=False) as data:
                        saved_output = data['xyz_mm']
                    metrics = evaluate(saved_output, xyz, truth, float(entry['gap_mm']), frame)
                    row.update(execution='ok', status=info.get('status'), selected_k=selected_k(info),
                               seconds=elapsed, n_points=len(xyz), metrics=metrics,
                               output_path=str(out_path.relative_to(args.out)), output_sha256=sha(out_path))
                except Exception as error:
                    row.update(execution='error', seconds=time.perf_counter() - t0, error=repr(error))
                rows.append(row)
                with (args.out / 'records.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(plain(row), ensure_ascii=False, allow_nan=False) + '\n')
        if (index + 1) % 9 == 0 or index + 1 == len(entries):
            print(json.dumps({'cases_done': index + 1, 'total_cases': len(entries),
                              'records': len(rows), 'errors': sum(r['execution'] != 'ok' for r in rows)}), flush=True)
    sources_unchanged = all(sha(p) == h for p, h in source_hashes.items())
    inputs_unchanged = all(sha(args.data / e['input_path']) ==
                           next(r['input_sha256'] for r in rows if r['id'] == e['id']) for e in entries)
    summary = {'records': rows, 'elapsed_seconds': time.perf_counter() - started,
               'errors': sum(r['execution'] != 'ok' for r in rows),
               'sources_unchanged': sources_unchanged, 'inputs_unchanged': inputs_unchanged,
               'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(args.out / 'results.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)
    if summary['errors'] or not sources_unchanged or not inputs_unchanged:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
