"""Paired perturbation response on six exact Oxford patches; no accuracy truth.

Original outputs of each method are its own response reference. Real point
labels are unavailable, so shuffled interventions permute globally, not within
true surfaces. The original coordinates and frozen estimator sources are read
only. Every execution creates a new result directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

import numpy as np


WORK = Path(__file__).resolve().parent
SOURCE = Path('/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600/t4/run-selwrbu7')
OLD = Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2/experiment.py')
FAST = Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-t2-boundary-v1/baseline/scalar_reference.py')
ADAPTER_REFERENCE = Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-t2-boundary-v1/real_smoke.py')
NAMES = tuple(f'{side}-{radius}-sample.npz' for side in ('left', 'center', 'right') for radius in ('0.15', '0.30'))
METHODS = ('identity', 'xyz_mixture', 'scalar_profile_hard')
SEEDS = (911071, 911083)
AMPLITUDES_MM = (5., 15.)
SIGMA_MM = 10.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def adapt(path):
    # Exact coordinate construction used by the preceding real_smoke.py.
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data['points'], dtype=float)
        normal = np.asarray(data['normal'], dtype=float).reshape(3)
        original_frame = np.asarray(data['frame_index'])
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Invalid source XYZ')
    if original_frame.shape != (len(xyz),) or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 0:
        raise ValueError('Invalid source metadata')
    normal /= np.linalg.norm(normal)
    tangent = np.array([1., 0., 0.]) if abs(normal[0]) < .8 else np.array([0., 1., 0.])
    tangent -= normal * np.dot(tangent, normal)
    tangent /= np.linalg.norm(tangent)
    basis = np.stack((tangent, np.cross(normal, tangent), normal), axis=1)
    origin = xyz.mean(axis=0)
    local = (xyz-origin) @ basis * 1000.
    frame_ids, frame = np.unique(original_frame, return_inverse=True)
    roundtrip = float(np.max(np.linalg.norm(local/1000. @ basis.T+origin-xyz, axis=1))*1000.)
    return {'world': xyz, 'local': local, 'frame': frame, 'original_frame': original_frame,
            'frame_ids': frame_ids, 'basis': basis, 'origin': origin, 'normal': normal,
            'counts': np.bincount(frame), 'roundtrip_max_mm': roundtrip}


def selected_k(info):
    if 'k' in info:
        return int(info['k'])
    if 'fit' in info and '0' in info['fit']:
        return int(info['fit']['0']['k'])
    return None


def interventions(scene_index, frame, counts):
    n = len(frame)
    yield {'arm': 'original', 'seed': None, 'amplitude_mm': 0., 'repeat': 0,
           'delta': np.zeros(n), 'frame_bias': np.zeros(len(counts)),
           'permutation': np.arange(n), 'histogram_equal_to_correlated': True}
    for seed in SEEDS:
        rng = np.random.default_rng(np.random.SeedSequence([seed, scene_index]))
        unit_bias = rng.normal(size=len(counts))
        unit_bias -= np.average(unit_bias, weights=counts)
        unit_bias /= np.sqrt(np.average(unit_bias**2, weights=counts))
        for amplitude in AMPLITUDES_MM:
            frame_bias = unit_bias * amplitude
            correlated = frame_bias[frame]
            yield {'arm': 'frame_shared', 'seed': seed, 'amplitude_mm': amplitude,
                   'repeat': 0, 'delta': correlated, 'frame_bias': frame_bias,
                   'permutation': np.arange(n), 'histogram_equal_to_correlated': True}
            for repeat in (1, 2):
                shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, scene_index, int(amplitude), repeat]))
                permutation = shuffle_rng.permutation(n)
                shuffled = correlated[permutation]
                exact = bool(np.array_equal(np.sort(correlated), np.sort(shuffled)))
                if not exact:
                    raise AssertionError('Permutation changed the intervention multiset')
                yield {'arm': 'shuffled', 'seed': seed, 'amplitude_mm': amplitude,
                       'repeat': repeat, 'delta': shuffled, 'frame_bias': frame_bias,
                       'permutation': permutation, 'histogram_equal_to_correlated': exact}


def aggregate(records):
    rows = []
    for method in METHODS:
        for amplitude in AMPLITUDES_MM:
            for arm in ('frame_shared', 'shuffled'):
                selected = [r for r in records if r['method'] == method and r['amplitude_mm'] == amplitude and r['arm'] == arm]
                ok = [r for r in selected if r['execution'] == 'ok']
                eligible = [r for r in ok if r['k_flip_from_original'] is not None]
                row = {'method': method, 'amplitude_mm': amplitude, 'arm': arm,
                       'n_outputs': len(selected), 'n_ok': len(ok),
                       'n_k_comparable': len(eligible),
                       'n_k_flips': sum(r['k_flip_from_original'] for r in eligible)}
                for key in ('response_normal_mae_mm', 'response_normal_rms_mm',
                            'response_point_mae_mm', 'response_point_rms_mm', 'elapsed_s'):
                    row['mean_'+key] = float(np.mean([r[key] for r in ok])) if ok else None
                row['max_response_point_rms_mm'] = max((r['response_point_rms_mm'] for r in ok), default=None)
                rows.append(row)
    return rows


def main():
    started_all = time.perf_counter()
    if platform.node() != 'liekkas':
        raise RuntimeError('Wrong host; designated source must not be substituted')
    paths = [SOURCE/name for name in NAMES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Designated source is missing: {path}')
    sources = (Path(__file__), WORK/'PROTOCOL.md', OLD, FAST, ADAPTER_REFERENCE)
    before_inputs = {str(p): sha(p) for p in paths}
    before_sources = {str(p): sha(p) for p in sources}
    old = load_module('_real_probe_frozen_experiment', OLD)
    fast = load_module('_real_probe_frozen_scalar_reference', FAST)
    destination = WORK/'real_results'
    destination.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='execution-', dir=destination))
    for directory in ('inputs', 'interventions', 'outputs', 'adaptation'):
        (run/directory).mkdir()
    manifest = {'scope': 'real perturbation response only, no clean geometry truth or accuracy inference',
                'host': platform.node(), 'input_sha256_before': before_inputs,
                'source_sha256_before': before_sources, 'methods': list(METHODS),
                'sigma_mm': SIGMA_MM, 'noise_scope': 'fixed uncalibrated 10 mm assumption, not changed with intervention RMS',
                'seeds': list(SEEDS), 'amplitudes_mm': list(AMPLITUDES_MM),
                'shuffle_repeats': 2, 'shuffle_scope': 'global point permutation without real surface labels; not the synthetic within-layer shuffle; not iid',
                'gauge': 'frame bias centered and RMS-normalized with point-count weights; original basis and origin fixed across arms',
                'reference': 'each method output on the unperturbed original input; original coordinates are not clean truth',
                'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                'numpy_version': np.__version__, 'python': sys.version,
                'expected_inputs': 78, 'expected_method_outputs': 234, 'expected_source_point_records': 415,
                'source_point_count_scope': '415 observed point records across six overlapping local samples, not necessarily unique map points',
                'output_source_hash_reference': 'source_sha256_before applies to every output', 'patches': []}
    write_json(run/'manifest.json', manifest)
    records = []
    cases = []
    print(json.dumps({'run': str(run)}), flush=True)
    for scene_index, path in enumerate(paths):
        data = adapt(path)
        patch = path.stem.removesuffix('-sample')
        frame = data['frame']
        counts = data['counts']
        patch_info = {'patch': patch, 'input_path': str(path), 'input_sha256': before_inputs[str(path)],
                      'n_points': len(frame), 'n_frames': len(counts), 'points_per_frame': counts.tolist(),
                      'original_frame_ids': data['frame_ids'].tolist(), 'minimum_points_per_frame': int(counts.min()),
                      'roundtrip_max_mm': data['roundtrip_max_mm']}
        manifest['patches'].append(patch_info)
        np.savez_compressed(run/'adaptation'/f'{patch}.npz', xyz_mm=data['local'], frame=frame,
                            original_frame=data['original_frame'], original_frame_ids=data['frame_ids'],
                            basis=data['basis'], origin_m=data['origin'], normal=data['normal'])
        references = {}
        for intervention in interventions(scene_index, frame, counts):
            arm, seed, amplitude, repeat = (intervention[k] for k in ('arm', 'seed', 'amplitude_mm', 'repeat'))
            case_id = f'{patch}__original' if arm == 'original' else f'{patch}__a{int(amplitude)}__s{seed}__{arm}__r{repeat}'
            pair_id = None if arm == 'original' else f'{patch}__a{int(amplitude)}__s{seed}'
            xyz = data['local'].copy()
            xyz[:, 2] += intervention['delta']
            input_path = run/'inputs'/f'{case_id}.npz'
            # Algorithm input never contains applied offsets or any reference geometry.
            np.savez_compressed(input_path, xyz_mm=xyz, frame=frame, sigma_mm=SIGMA_MM)
            delta_path = run/'interventions'/f'{case_id}.npz'
            np.savez_compressed(delta_path, perturbation_mm=intervention['delta'],
                                frame_bias_mm=intervention['frame_bias'], permutation=intervention['permutation'])
            within_frame_var = [np.var(intervention['delta'][frame == fid]) for fid in range(len(counts))]
            case = {'id': case_id, 'pair_id': pair_id, 'patch': patch, 'arm': arm, 'seed': seed,
                    'amplitude_mm': amplitude, 'repeat': repeat, 'input_path': str(input_path),
                    'input_sha256': sha(input_path), 'intervention_path': str(delta_path),
                    'intervention_sha256': sha(delta_path),
                    'perturbation_mean_mm': float(intervention['delta'].mean()),
                    'perturbation_rms_mm': float(np.sqrt(np.mean(intervention['delta']**2))),
                    'perturbation_mae_mm': float(np.abs(intervention['delta']).mean()),
                    'point_weighted_within_frame_variance_mm2': float(np.average(within_frame_var, weights=counts)),
                    'histogram_equal_to_correlated': intervention['histogram_equal_to_correlated'],
                    'has_geometry_ground_truth': False}
            cases.append(case)
            for method in METHODS:
                row = {'case_id': case_id, 'pair_id': pair_id, 'patch': patch, 'arm': arm,
                       'seed': seed, 'amplitude_mm': amplitude, 'repeat': repeat, 'method': method,
                       'n_input': len(frame), 'input_sha256': case['input_sha256'],
                       'source_sha256': before_sources, 'has_geometry_ground_truth': False}
                started = time.perf_counter()
                try:
                    with np.load(input_path, allow_pickle=False) as saved_input:
                        inp = old.Input(saved_input['xyz_mm'], saved_input['frame'], np.zeros(len(frame), dtype=int),
                                        float(saved_input['sigma_mm']), 50.)
                    fn = fast.estimate if method == 'scalar_profile_hard' else old.estimate
                    output, bias, info = fn(inp, method)
                    elapsed = time.perf_counter()-started
                    output_path = run/'outputs'/f'{case_id}__{method}.npz'
                    world = np.asarray(output)/1000. @ data['basis'].T+data['origin']
                    np.savez_compressed(output_path, xyz_mm=output, points=world,
                                        frame=frame, frame_index=data['original_frame'], bias_mm=bias)
                    write_json(output_path.with_suffix('.json'), {'method': method, 'info': info, 'source_sha256': before_sources})
                    # Reload serialized geometry before measuring any response.
                    with np.load(output_path, allow_pickle=False) as saved_output:
                        actual = saved_output['xyz_mm']
                        actual_world = saved_output['points']
                        finite = bool(np.isfinite(actual).all() and np.isfinite(actual_world).all() and np.isfinite(saved_output['bias_mm']).all())
                        frame_preserved = bool(np.array_equal(saved_output['frame_index'], data['original_frame']))
                    k = selected_k(info)
                    if arm == 'original':
                        references[method] = (actual.copy(), actual_world.copy(), k, str(output_path))
                    reference, reference_world, reference_k, reference_path = references[method]
                    difference = actual-reference
                    point_response = np.linalg.norm(difference, axis=1)
                    world_response = np.linalg.norm(actual_world-reference_world, axis=1)*1000.
                    row.update({'execution': 'ok', 'elapsed_s': elapsed, 'status': info.get('status'),
                                'selected_k': k, 'original_selected_k': reference_k,
                                'k_flip_from_original': bool(k != reference_k) if k is not None and reference_k is not None else None,
                                'n_output': len(actual), 'same_point_count': len(actual) == len(frame),
                                'all_finite': finite, 'frame_index_preserved': frame_preserved,
                                'response_reference_output': reference_path,
                                'response_normal_mae_mm': float(np.abs(difference[:, 2]).mean()),
                                'response_normal_rms_mm': float(np.sqrt(np.mean(difference[:, 2]**2))),
                                'response_point_mae_mm': float(point_response.mean()),
                                'response_point_rms_mm': float(np.sqrt(np.mean(point_response**2))),
                                'response_point_max_mm': float(point_response.max()),
                                'response_world_point_rms_mm': float(np.sqrt(np.mean(world_response**2))),
                                'output_normal_span_mm': float(np.ptp(actual[:, 2])),
                                'output': str(output_path), 'output_sha256': sha(output_path), 'info': info})
                except Exception as exc:
                    row.update({'execution': 'error', 'elapsed_s': time.perf_counter()-started, 'error': repr(exc)})
                records.append(row)
                with (run/'records.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps({'patch': patch, 'points': len(frame), 'completed_outputs': len(records)}), flush=True)
    manifest['input_sha256_after'] = {str(p): sha(p) for p in paths}
    manifest['source_sha256_after'] = {str(p): sha(p) for p in sources}
    manifest['source_inputs_unchanged'] = before_inputs == manifest['input_sha256_after']
    manifest['sources_unchanged'] = before_sources == manifest['source_sha256_after']
    manifest['total_source_point_records'] = sum(p['n_points'] for p in manifest['patches'])
    manifest['actual_inputs'] = len(cases)
    manifest['actual_method_records'] = len(records)
    manifest['successful_outputs'] = sum(r['execution'] == 'ok' for r in records)
    manifest['failed_outputs'] = sum(r['execution'] != 'ok' for r in records)
    manifest['elapsed_total_s'] = time.perf_counter()-started_all
    manifest['peak_process_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest['resource_scope'] = 'one sequential CPU process including serialization; method timings exclude output serialization, no GPU'
    checks = {'source_points_are_415': manifest['total_source_point_records'] == 415,
              'inputs_are_78': len(cases) == 78, 'outputs_are_234': len(records) == 234,
              'all_success': manifest['failed_outputs'] == 0,
              'all_output_checks': all(r.get('all_finite') and r.get('same_point_count') and r.get('frame_index_preserved') for r in records),
              'all_intervention_histograms_exact': all(c['histogram_equal_to_correlated'] for c in cases),
              'all_intervention_means_below_1e_12_mm': all(abs(c['perturbation_mean_mm']) < 1e-12 for c in cases),
              'all_intervention_rms_within_1e_12_mm': all(abs(c['perturbation_rms_mm']-c['amplitude_mm']) < 1e-12 for c in cases),
              'source_inputs_unchanged': manifest['source_inputs_unchanged'], 'sources_unchanged': manifest['sources_unchanged']}
    results = {'manifest': manifest, 'checks': checks, 'cases': cases, 'aggregate': aggregate(records), 'records': records}
    write_json(run/'manifest.json', manifest)
    write_json(run/'results.json', results)
    write_json(run/'aggregate.json', results['aggregate'])
    print(json.dumps({'completed_run': str(run), 'checks': checks, 'errors': manifest['failed_outputs']}), flush=True)
    if not all(checks.values()):
        raise AssertionError('One or more validation checks failed; saved outputs retained')


if __name__ == '__main__':
    main()
