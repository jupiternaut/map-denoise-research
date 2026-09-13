"""Isolated, fixed-protocol outer-iteration probe; never edits estimators."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from paths import RUNS, require_liekkas

PROTOCOL = HERE / 'TRANSPORT_BUDGET_PROTOCOL.md'
BASE = RUNS / 'repair-v2-oyuie4pl' / 'synthetics' / 'identifiable'
BASE_NAMES = ('ghost_s912101_b4', 'dual_g2_s912101_b4',
              'dual_g4_s912101_b4', 'dual_g8_s912101_b4')
OUTER_BUDGETS = (3, 6, 12)
VARIANTS = ('balanced', 'unbalanced')
FROZEN_PARAMETERS = {'sigma_mm': 1.0, 'sinkhorn_iterations': 24}


def clean(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, Path): return str(v)
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [clean(x) for x in v]
    return v


def write_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(clean(obj), f, indent=2, ensure_ascii=False, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_points(path, xyz):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, xyz_world=xyz)


def worker(args):
    """No evaluator or truth loaded/imported in this process."""
    import measure_surface
    import graph_surface
    assert Path(measure_surface.__file__).resolve() == HERE / 'measure_surface.py'
    assert Path(graph_surface.__file__).resolve() == HERE / 'graph_surface.py'
    original = dict(measure_surface.PARAMETERS)
    assert original['sinkhorn_iterations'] == 24
    outdir = args.worker_run / 'outputs'
    prefix = f'{args.variant}_o{args.outer}'
    before = {str(HERE / n): sha(HERE / n) for n in ('measure_surface.py', 'graph_surface.py')}
    summary = {'variant': args.variant, 'outer_iterations': args.outer,
               'pid': os.getpid(), 'original_parameters': original,
               'frozen_source_hash_before': before, 'calls': 0, 'ok': 0}
    started = time.perf_counter()
    try:
        measure_surface.PARAMETERS['outer_iterations'] = args.outer
        # Fixed ordinary points, no true normals, labels, offsets or reference.
        xy = np.array([(x, y) for x in (-.01, 0., .01) for y in (-.01, 0., .01)])
        tiny = np.tile(np.c_[xy, np.zeros(len(xy))], (2, 1))
        tiny_frames = np.repeat([0, 1], len(xy))
        warm_start = time.perf_counter()
        w, _ = measure_surface.estimate(tiny, tiny_frames, 1., variant=args.variant)
        graph_surface.estimate(w, tiny_frames, 1., variant='local_only')
        summary['warmup_s'] = time.perf_counter()-warm_start
        for file in sorted((args.worker_run / 'inputs').glob('*.npz')):
            summary['calls'] += 1
            artifact = outdir / f'{file.stem}__{prefix}'
            try:
                with np.load(file, allow_pickle=False) as a:
                    xyz, frame = a['xyz_world'].copy(), a['scan_id'].copy()
                original_xyz, original_frame = xyz.copy(), frame.copy()
                t = time.perf_counter()
                raw, info = measure_surface.estimate(xyz, frame, 1., variant=args.variant)
                transport_s = time.perf_counter()-t
                if raw.shape != xyz.shape or not np.isfinite(raw).all():
                    raise ValueError('invalid correction output')
                if not np.array_equal(xyz, original_xyz) or not np.array_equal(frame, original_frame):
                    raise ValueError('correction mutated its input')
                raw_before = raw.copy()
                t = time.perf_counter()
                post, local_info = graph_surface.estimate(raw, frame, 1., variant='local_only')
                local_s = time.perf_counter()-t
                if post.shape != xyz.shape or not np.isfinite(post).all():
                    raise ValueError('invalid local-filter output')
                if not np.array_equal(raw, raw_before) or not np.array_equal(frame, original_frame):
                    raise ValueError('local filter mutated its input')
                save_points(str(artifact)+'__raw.npz', raw)
                save_points(str(artifact)+'__local.npz', post)
                write_json(str(artifact)+'.json', {
                    'ok': True, 'case': file.stem, 'variant': args.variant,
                    'outer_iterations': args.outer, 'sinkhorn_iterations': 24,
                    'transport_seconds': transport_s, 'local_filter_seconds': local_s,
                    'pipeline_seconds': transport_s+local_s,
                    'transport': info, 'local_filter': local_info,
                    'input_array_unchanged': True, 'n_input': len(xyz),
                    'n_output_raw': len(raw), 'n_output_local': len(post)})
                summary['ok'] += 1
            except Exception as exc:
                write_json(str(artifact)+'.json', {
                    'ok': False, 'case': file.stem, 'variant': args.variant,
                    'outer_iterations': args.outer, 'error': repr(exc),
                    'traceback': traceback.format_exc()})
    finally:
        measure_surface.PARAMETERS.clear()
        measure_surface.PARAMETERS.update(original)
        summary.update(parameters_restored=measure_surface.PARAMETERS == original,
                       elapsed_s=time.perf_counter()-started,
                       peak_worker_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       frozen_source_hash_after={p: sha(p) for p in before})
        write_json(args.worker_run / f'WORKER_{prefix}.json', summary)
    assert summary['frozen_source_hash_after'] == before
    assert summary['parameters_restored']


def aggregate(rows):
    result = []
    for sampling in ('full', 'partial_overlap'):
        for variant in VARIANTS:
            for outer in OUTER_BUDGETS:
                for stage in ('raw', 'local'):
                    group = [r for r in rows if r['sampling'] == sampling and
                             r['variant'] == variant and r['outer_iterations'] == outer and
                             r['stage'] == stage and r['ok']]
                    row = dict(sampling=sampling, variant=variant,
                               outer_iterations=outer, stage=stage, successful=len(group), expected=4)
                    for metric in ('surface_accuracy_mean_mm', 'surface_accuracy_rms_mm',
                                   'matched_point_rms_mm', 'source_surface_tilt_mean_deg',
                                   'reference_sample_coverage_1mm', 'method_seconds',
                                   'estimated_bias_rms_mm', 'effective_transport_mass',
                                   'balanced_or_relaxed_row_l1_mean', 'balanced_or_relaxed_row_l1_max'):
                        if group: row['mean_'+metric] = float(np.mean([r[metric] for r in group]))
                    result.append(row)
    return result


def run():
    from schema import read_patch, read_evaluation
    from evaluate_v2 import synthetic_geometry, point_rms_mm
    from metrics import sampling_indices, subset_evaluation, structure_metrics
    started = time.perf_counter()
    dest = Path(tempfile.mkdtemp(prefix='transport-budget-v3-', dir=RUNS))
    print(dest, flush=True)
    (dest/'inputs').mkdir(); (dest/'outputs').mkdir(); (dest/'source').mkdir()
    sources = [HERE/n for n in ('transport_budget_probe.py', 'TRANSPORT_BUDGET_PROTOCOL.md',
                                'measure_surface.py', 'graph_surface.py', 'metrics.py')]
    sources += [PROJECT/n for n in ('schema.py', 'evaluate_v2.py', 'hashutil.py', 'transforms.py', 'paths.py')]
    cases, protected = [], list(sources)
    for name in BASE_NAMES:
        source = BASE/f'{name}.json'
        points, meta = read_patch(source)
        evaluation = read_evaluation(source)
        assert meta['seed'] == 912101 and meta['bias_rms_mm'] == 4
        assert evaluation.get('identifiable', True)
        npz = Path(meta['npz'])
        protected += [source, npz, npz.parent/'evaluation'/f'{name}.eval.npz',
                      npz.parent/'evaluation'/f'{name}.eval.json']
        for sampling in ('full', 'partial_overlap'):
            idx = np.arange(len(points['xyz_world'])) if sampling == 'full' else sampling_indices(
                points['xyz_world'], points['scan_id'], 'partial_overlap')
            case_name = name if sampling == 'full' else name+'__partial_overlap'
            xyz, frame = points['xyz_world'][idx].copy(), points['scan_id'][idx].copy()
            ev = subset_evaluation(evaluation, idx, len(points['xyz_world']))
            with (dest/'inputs'/f'{case_name}.npz').open('xb') as f:
                np.savez_compressed(f, xyz_world=xyz, scan_id=frame)
            case = dict(case=case_name, base_case=name, sampling=sampling,
                        source=str(source), n_points=len(xyz), n_frames=len(np.unique(frame)),
                        source_indices=idx, family=meta['family'], gap_mm=meta['gap_mm'],
                        seed=meta['seed'], bias_rms_mm=meta['bias_rms_mm'])
            write_json(dest/'inputs'/f'{case_name}.json', case)
            cases.append(dict(case, xyz=xyz, frame=frame, evaluation=ev))
    before = {str(p): sha(p) for p in protected}
    write_json(dest/'PROTECTED_BEFORE.json', before)
    write_json(dest/'SOURCE_MANIFEST.json', {str(p): before[str(p)] for p in sources})
    for p in sources: shutil.copyfile(p, dest/'source'/p.name)
    write_json(dest/'CONFIGURATIONS.json', dict(
        variants=VARIANTS, outer_iterations=OUTER_BUDGETS, **FROZEN_PARAMETERS,
        input_count=8, expected_correction_calls=48, expected_output_count=96,
        protocol_sha256=sha(PROTOCOL), frozen_before_workers=True,
        legal_input_sha256={str(p): sha(p) for p in (dest/'inputs').glob('*.npz')}))
    worker_rows = []
    for variant in VARIANTS:
        for outer in OUTER_BUDGETS:
            t = time.perf_counter()
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            cp = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                 '--worker-run', str(dest), '--variant', variant, '--outer', str(outer)],
                                env=env, capture_output=True, text=True, check=False)
            execution = dict(variant=variant, outer_iterations=outer, returncode=cp.returncode,
                             subprocess_wall_s=time.perf_counter()-t,
                             stdout=cp.stdout, stderr=cp.stderr)
            write_json(dest/f'PROCESS_{variant}_o{outer}.json', execution)
            worker_rows.append(execution)
            print(f'{variant} outer={outer}: returncode={cp.returncode}', flush=True)
    rows = []
    for case in cases:
        for variant in VARIANTS:
            for outer in OUTER_BUDGETS:
                base = dest/'outputs'/f"{case['case']}__{variant}_o{outer}"
                metadata_path = Path(str(base)+'.json')
                d = json.loads(metadata_path.read_text()) if metadata_path.exists() else {'ok':False, 'error':'worker output missing'}
                for stage in ('raw', 'local'):
                    row = {k: v for k, v in case.items() if k not in ('xyz', 'frame', 'evaluation', 'source_indices')}
                    row.update(variant=variant, outer_iterations=outer, sinkhorn_iterations=24,
                               stage=stage, sigma_mm=1., ok=d['ok'])
                    if not d['ok']:
                        row['error'] = d.get('error', 'unknown worker failure'); rows.append(row); continue
                    path = Path(str(base)+f'__{stage}.npz')
                    with np.load(path, allow_pickle=False) as a: out = a['xyz_world']
                    info = d['transport']; pairs = info['scan_pair_details']
                    history = info['outer_history']
                    increments = np.diff(np.vstack([np.zeros(len(info['bias_mm']))] +
                                                   [h['bias_mm'] for h in history]), axis=0)
                    max_steps = np.max(np.abs(increments), axis=1)
                    row.update(synthetic_geometry(out, info, case['evaluation']))
                    row.update(structure_metrics(out, case['evaluation']))
                    disp = np.linalg.norm(out-case['xyz'], axis=1)*1000
                    row.update(method_seconds=d['transport_seconds'] if stage == 'raw' else d['pipeline_seconds'],
                               transport_seconds=d['transport_seconds'], local_filter_seconds=d['local_filter_seconds'],
                               input_edit_rms_mm=point_rms_mm(out, case['xyz']),
                               input_edit_p95_mm=float(np.quantile(disp, .95)),
                               moved_fraction=float(np.mean(disp > 1e-6)),
                               estimated_bias_rms_mm=info['bias_rms_mm'],
                               estimated_bias_max_mm=float(np.max(np.abs(info['bias_mm']))),
                               effective_transport_mass=info['effective_transport_mass'],
                               usable_transport_mass=info['usable_transport_mass'],
                               supported_scan_fraction=info['supported_scan_fraction'],
                               component_changes=sum(a['components'] != b['components'] for a, b in zip(history, history[1:])),
                               largest_observed_bias_increment_mm=float(max_steps.max()),
                               updates_at_nominal_1p6mm_increment=int(np.sum(np.isclose(max_steps, 1.6, atol=1e-9, rtol=0))),
                               balanced_or_relaxed_row_l1_mean=float(np.mean([p['row_marginal_l1'] for p in pairs])),
                               balanced_or_relaxed_row_l1_max=float(max(p['row_marginal_l1'] for p in pairs)),
                               balanced_or_relaxed_column_l1_max=float(max(p['column_marginal_l1'] for p in pairs)),
                               output=str(path), output_sha256=sha(path))
                    write_json(Path(str(base)+f'__{stage}.metrics.json'), row)
                    rows.append(row)
    with (dest/'RESULTS.csv').open('x', newline='') as f:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    after = {p: sha(p) for p in before}
    write_json(dest/'PROTECTED_AFTER.json', after)
    workers = [json.loads(p.read_text()) for p in dest.glob('WORKER_*.json')]
    summary = dict(run_dir=dest, evidence='exposed development, 4 worlds with 2 sampling views; no reserved seeds',
                   input_count=len(cases), correction_calls=sum(w['calls'] for w in workers),
                   output_rows=len(rows), successful_output_rows=sum(r['ok'] for r in rows),
                   parameters_restored=all(w['parameters_restored'] for w in workers),
                   protected_files_unchanged=before == after, protected_file_count=len(before),
                   elapsed_s=time.perf_counter()-started,
                   actual_method_call_seconds=sum(r['transport_seconds']+r['local_filter_seconds'] for r in rows if r['ok'] and r['stage']=='raw'),
                   worker_subprocess_wall_s=sum(w['subprocess_wall_s'] for w in worker_rows),
                   parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   largest_worker_peak_rss_kib=max((w['peak_worker_rss_kib'] for w in workers), default=None),
                   memory_scope='separate process high-water marks, not a simultaneous pipeline RAM estimate',
                   threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                   python=sys.executable, aggregate=aggregate(rows))
    write_json(dest/'SUMMARY.json', summary)
    if before != after: raise RuntimeError('protected sources or original inputs changed')
    print(json.dumps(clean({k:v for k,v in summary.items() if k!='aggregate'}), indent=2), flush=True)


if __name__ == '__main__':
    require_liekkas()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-run', type=Path)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--outer', type=int, choices=OUTER_BUDGETS)
    options = parser.parse_args()
    if options.worker_run:
        if options.variant is None or options.outer is None: parser.error('worker requires variant and outer')
        worker(options)
    else:
        run()
