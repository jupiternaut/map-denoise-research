"""Independent V24 arithmetic audit; does not import run/field_ops/confidence.

The PLY input is read through Open3D, whereas production uses plyfile. All
queries use the same frozen evaluation support. This is an implementation
check, not independent-scene confirmation or a physical geometry certificate.
"""
from pathlib import Path
import hashlib
import json
import socket
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


ROOT = Path('/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1')
CACHE = Path('/srv/slam-research/grf/map-denoise/runs/reconstruction-v22-qayc8gft')
FIELDS = ('local_plane64', 'quadratic64', 'multiscale_full')
BUDGETS = (0.025, 0.05, 0.1)
SEEDS = tuple(range(92401, 92406))
ATOL = 1e-10


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def rms(array):
    return float(np.linalg.norm(array) / np.sqrt(len(array)))


def close(first, second, label, tolerance=ATOL):
    if first is None or second is None:
        assert first is None and second is None, (label, first, second)
        return 0.
    a, b = np.asarray(first), np.asarray(second)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), label
    difference = float(np.max(np.abs(a-b))) if a.size else 0.
    assert difference <= tolerance, (label, difference, tolerance)
    return difference


def expected_records(points, native, alpha):
    """Rebuild the algebra from cached fields, never call the tested module."""
    result = {}
    for name, output in native.items():
        result[f'native__{name}'] = dict(output=output, kind='native', field=name,
                                        allocation='cached', status='OK')
    delta = {name: native[name]-points for name in FIELDS}
    lengths = {name: rms(value) for name, value in delta.items()}
    cap = min(lengths.values())
    if cap <= 1e-12:
        cap = 0.
    previous = []
    for requested in BUDGETS:
        actual = min(requested, cap)
        duplicate = actual in previous
        previous.append(actual)
        specs = [(name, 'uniform', None, delta[name]) for name in FIELDS]
        specs.append(('multiscale_full', 'alpha', None,
                      delta['multiscale_full'] * alpha[:, None]))
        for seed in SEEDS:
            order = np.random.default_rng(seed).permutation(len(alpha))
            specs.append(('multiscale_full', 'shuffle', seed,
                          delta['multiscale_full'] * alpha[order, None]))
        for name, allocation, seed, vector in specs:
            name_key = f'budget{requested:g}__{name}__{allocation}'
            if seed is not None:
                name_key += f'_{seed}'
            length = rms(vector)
            if actual == 0:
                scale, output = 0., points.copy()
            elif length <= 1e-12:
                scale, output = None, None
            else:
                scale = actual / length
                output = points + vector * scale
            result[name_key] = dict(output=output, kind='budget', field=name,
                allocation=allocation, seed=seed, requested_budget_mm=requested,
                actual_budget_mm=actual, duplicate_budget=duplicate,
                direction_informative=actual > 0, native_field_rms_mm=length,
                normalization_factor=scale,
                status='OK' if output is not None else 'INFEASIBLE')
    for step in (0.5, 1.):
        result[f'probe_t{step:g}'] = dict(output=points + step*delta['multiscale_full'],
            kind='probe', field='multiscale_full', allocation='fixed_step',
            step=step, status='OK')
    return result, lengths, cap


def group_summary(alpha, gain, mask):
    a, g = alpha[mask], gain[mask]
    return dict(n=len(a), mean_alpha=float(np.mean(a)), min_alpha=float(np.min(a)),
        max_alpha=float(np.max(a)), mean_signed_gain=float(np.mean(g)),
        beneficial_fraction=float(np.count_nonzero(g > 0)/len(g)),
        harmful_fraction=float(np.count_nonzero(g < 0)/len(g)),
        unchanged_fraction=float(np.count_nonzero(g == 0)/len(g)))


def check_dict(actual, expected, label):
    maximum = 0.
    for name, wanted in expected.items():
        got = actual[name]
        if isinstance(wanted, dict):
            maximum = max(maximum, check_dict(got, wanted, f'{label}.{name}'))
        elif isinstance(wanted, (str, bool)):
            assert got == wanted, (label, name, got, wanted)
        else:
            maximum = max(maximum, close(got, wanted, f'{label}.{name}'))
    return maximum


def check_confidence(saved, alpha, baseline, probes):
    """Use SciPy ranking plus explicit comparisons for bins/tie handling."""
    maximum = 0.
    quartiles = np.percentile(alpha, [25, 50, 75])
    cuts = np.unique(quartiles)
    maximum = max(maximum, close(saved['bin_boundaries'], cuts, 'confidence.cuts'))
    maximum = max(maximum, check_dict(saved['alpha_quantiles'],
        dict(zip(('q25', 'q50', 'q75'), quartiles)), 'confidence.quantiles'))
    # Count boundaries strictly below each value, preserving all ties together.
    bin_number = np.sum(alpha[:, None] > cuts[None, :], axis=1)
    lower = alpha <= quartiles[0]
    upper = alpha >= quartiles[2]
    overlap = lower & upper
    lower = lower & ~overlap
    upper = upper & ~overlap
    maximum = max(maximum, check_dict(saved['high_low_counts'],
        dict(low=int(lower.sum()), high=int(upper.sum()),
             overlapping_ties_excluded=int(overlap.sum())), 'confidence.tail_counts'))
    assert saved['n'] == len(alpha)
    for name, errors in probes.items():
        gain = baseline-errors
        record = saved['probes'][name]
        rho = None if len(alpha) < 2 or np.ptp(alpha) == 0 or np.ptp(gain) == 0 else float(spearmanr(alpha, gain).statistic)
        maximum = max(maximum, close(record['spearman_rank_correlation'], rho,
                                     f'confidence.{name}.rho'))
        maximum = max(maximum, check_dict(record['overall'],
            group_summary(alpha, gain, np.ones(len(alpha), bool)), f'confidence.{name}.overall'))
        present = np.unique(bin_number)
        assert len(record['bins']) == len(present)
        for actual_bin, number in zip(record['bins'], present):
            wanted = dict(bin_id=int(number), **group_summary(alpha, gain, bin_number == number))
            maximum = max(maximum, check_dict(actual_bin, wanted, f'confidence.{name}.bin{number}'))
        comparison = record['high_low_comparison']
        if not lower.any() or not upper.any():
            assert comparison is None
        else:
            low_stats = group_summary(alpha, gain, lower)
            high_stats = group_summary(alpha, gain, upper)
            maximum = max(maximum, check_dict(comparison,
                dict(low=low_stats, high=high_stats,
                     high_minus_low_mean_signed_gain=high_stats['mean_signed_gain']-low_stats['mean_signed_gain']),
                f'confidence.{name}.tails'))
    return maximum


def main():
    assert socket.gethostname() == 'liekkas', 'wrong target host'
    assert Path(__file__).resolve().parent == ROOT/'field_budget_v24', 'wrong project'
    if len(sys.argv) != 2:
        raise SystemExit('usage: python -B field_budget_v24/verify.py /absolute/run/directory')
    dest = Path(sys.argv[1]).resolve()
    assert dest.parent == CACHE.parent and dest.name.startswith('field-budget-v24-'), dest
    assert not (dest/'VERIFICATION.json').exists(), 'verification output already exists'
    start = time.perf_counter()
    sealed = read(dest/'SEALED.json')
    assert sealed['version'] == 24 and sealed['cache'] == str(CACHE)
    assert sealed['construction_used_reference'] is False
    locks = {key: read(dest/f'{key}.json') for key in ('HISTORY_LOCK', 'SOURCE_LOCK')}
    for label, lock in locks.items():
        for path, digest in lock.items():
            assert sha(path) == digest, (label, path)
    for path, digest in locks['SOURCE_LOCK'].items():
        assert sha(dest/'source'/Path(path).name) == digest, path
    assert sealed['protocol_sha256'] == sha(dest/'source/PROTOCOL.md')
    rows = read(dest/'RESULTS.json')
    index = {(r['case'], r['method']): r for r in rows}
    assert len(index) == len(rows), 'duplicate result row'
    confidence_index = {r['case']: r for r in read(dest/'CONFIDENCE.json')}
    prior = {}
    old_results = {}
    for phase in ('development', 'confirmation'):
        for bundle in read(CACHE/f'{phase}_SEALED.json'):
            prior[bundle['job']['case']] = bundle
        for row in read(CACHE/f'{phase}_RESULTS.json'):
            old_results[(row['case'], row['method'])] = row
    assert {c['case'] for c in sealed['cases']} == set(prior)
    assert set(confidence_index) == set(prior)
    checked_outputs = checked_metrics = checked_native = checked_budgets = 0
    checked_confidence = infeasible = 0
    max_coordinates = max_metric = max_budget = max_confidence = max_distance = 0.
    reference_counts = {}
    for scene, manifest in read(dest/'REFERENCE_MANIFEST.json').items():
        assert sha(manifest['path']) == manifest['sha256']
        reference = np.asarray(o3d.io.read_point_cloud(manifest['path']).points)
        assert reference.ndim == 2 and reference.shape[1] == 3 and np.isfinite(reference).all()
        reference_counts[scene] = len(reference)
        tree = cKDTree(reference)
        for case in [c for c in sealed['cases'] if c['scene'] == scene]:
            name = case['case']
            assert case['evidence_role'] == 'exposed_diagnostic'
            for field in ('input', 'alpha', 'evaluation'):
                assert sha(case[field]) == case[f'{field}_sha256']
            old = prior[name]
            assert sha(case['original_input']) == old['job']['sha256']
            assert sha(case['input']) == old['job']['sha256']
            assert sha(case['evaluation']) == sha(CACHE/'evaluation'/f'{name}.npz')
            points = np.load(case['input'])
            assert points.shape == (1024, 3) and np.isfinite(points).all()
            alpha = np.load(case['alpha'])
            diagnostic = read(case['original_diagnostic'])['operator']['consensus']
            np.testing.assert_array_equal(alpha, np.asarray(diagnostic['alpha']))
            assert alpha.shape == (1024,) and np.all((alpha >= 0) & (alpha <= 1))
            native = {}
            native_records = {}
            for record in old['records']:
                assert record['status'] == 'OK' and sha(record['path']) == record['sha256']
                native[record['method']] = np.load(record['path'])
                native_records[record['method']] = record
            normal = np.asarray(diagnostic['normal'])
            full = native['multiscale_full']-points
            close(full, np.asarray(diagnostic['unshrunk_correction'])[:, None]*normal, f'{name}.full')
            close(native['multiscale_consensus']-points, alpha[:, None]*full, f'{name}.alpha')
            expected, native_rms, cap = expected_records(points, native, alpha)
            close(case['diagnostic']['shared_cap_mm'], cap, f'{name}.cap')
            check_dict(case['diagnostic']['native_field_rms_mm'], native_rms, f'{name}.rms')
            assert len(case['records']) == len(expected)
            assert {r['method'] for r in case['records']} == set(expected)
            evaluation = np.load(case['evaluation'])
            keep, ids = evaluation['input_mask'], evaluation['reference_ids']
            assert keep.dtype == bool and keep.shape == (1024,) and keep.any()
            assert ids.ndim == 1 and len(ids) and np.all((ids >= 0) & (ids < len(reference)))
            stored_distances = np.load(dest/'distances'/f'{name}.npz')
            recomputed_distances = {}
            for record in case['records']:
                method = record['method']
                wanted = expected[method]
                row = index[(name, method)]
                assert row['scene'] == scene and row['status'] == record['status'] == wanted['status']
                for key, value in wanted.items():
                    if key == 'output':
                        continue
                    if isinstance(value, (str, bool)):
                        assert record[key] == value, (name, method, key)
                    else:
                        close(record[key], value, f'{name}.{method}.{key}')
                if wanted['output'] is None:
                    assert 'path' not in record and method not in stored_distances.files
                    infeasible += 1
                    continue
                assert sha(record['path']) == record['sha256']
                output = np.load(record['path'])
                assert output.shape == points.shape and np.isfinite(output).all()
                max_coordinates = max(max_coordinates, close(output, wanted['output'], f'{name}.{method}.coordinates'))
                move = np.linalg.norm(output-points, axis=1)
                movement = dict(displacement_rms_mm=rms(output-points),
                    displacement_p95_mm=float(np.quantile(move, .95)),
                    displacement_max_mm=float(np.max(move)),
                    moved_fraction=float(np.count_nonzero(move > 1e-7)/len(move)))
                check_dict(record, movement, f'{name}.{method}.movement_record')
                check_dict(row, movement, f'{name}.{method}.movement_row')
                if wanted['kind'] == 'budget':
                    max_budget = max(max_budget, close(movement['displacement_rms_mm'],
                        wanted['actual_budget_mm'], f'{name}.{method}.actual_budget'))
                    if wanted['allocation'] == 'uniform':
                        assert record['normalization_factor'] <= 1.+1e-12
                    checked_budgets += 1
                if wanted['kind'] == 'native':
                    original = native_records[wanted['field']]
                    assert record['sha256'] == original['sha256'], (name, method, 'native changed bytes')
                    checked_native += 1
                accuracy = tree.query(output[keep], workers=1)[0]
                completeness = cKDTree(output).query(reference[ids], workers=1)[0]
                precision = float(np.count_nonzero(accuracy <= 1.)/len(accuracy))
                recall = float(np.count_nonzero(completeness <= 1.)/len(completeness))
                metrics = dict(n_input=int(keep.sum()), n_reference=len(ids),
                    accuracy_mm=float(np.mean(accuracy)), p95_mm=float(np.quantile(accuracy, .95)),
                    completeness_mm=float(np.mean(completeness)), precision=precision, recall=recall,
                    fscore=2.*precision*recall/(precision+recall) if precision+recall else 0.)
                max_metric = max(max_metric, check_dict(row, metrics, f'{name}.{method}.metrics'))
                if wanted['kind'] == 'native':
                    max_metric = max(max_metric, check_dict(old_results[(name, wanted['field'])],
                        metrics, f'{name}.{method}.cached_metrics'))
                max_distance = max(max_distance, close(stored_distances[method], accuracy,
                    f'{name}.{method}.stored_distances'))
                recomputed_distances[method] = accuracy
                checked_metrics += 1
                checked_outputs += 1
            assert set(stored_distances.files) == set(recomputed_distances)
            stored_confidence = np.load(dest/'confidence'/f'{name}.npz')
            close(stored_confidence['alpha'], alpha[keep], f'{name}.scored_alpha')
            for saved_key, method in [('baseline_errors', 'native__identity'),
                ('probe_t05_errors', 'probe_t0.5'), ('probe_t1_errors', 'probe_t1')]:
                close(stored_confidence[saved_key], recomputed_distances[method], f'{name}.{saved_key}')
            saved = read(dest/'confidence'/f'{name}.json')
            assert saved == confidence_index[name]
            max_confidence = max(max_confidence, check_confidence(saved, alpha[keep],
                recomputed_distances['native__identity'],
                {m: recomputed_distances[m] for m in ('probe_t0.5', 'probe_t1')}))
            checked_confidence += 1
            print('VERIFIED_CASE', name, len(expected), flush=True)
        del tree, reference
    assert checked_outputs + infeasible == len(rows)
    summary = read(dest/'RUN_SUMMARY.json')
    assert summary['n_outputs'] == len(rows) and summary['n_patches'] == len(prior)
    assert summary['failed_outputs'] == infeasible
    # Check the exact same lock again, including all historical cache bytes.
    for label, lock in locks.items():
        for path, digest in lock.items():
            assert sha(path) == digest, (label, path)
    result = dict(status='PASS', n_patches=len(prior), checked_outputs=checked_outputs,
        checked_geometry_metrics=checked_metrics, native_outputs_byte_identical=checked_native,
        budget_outputs_checked=checked_budgets, infeasible_outputs=infeasible,
        signed_confidence_patches_checked=checked_confidence,
        max_coordinate_difference_mm=max_coordinates, max_metric_difference=max_metric,
        max_budget_difference_mm=max_budget, max_point_distance_difference_mm=max_distance,
        max_confidence_difference=max_confidence, tolerance=ATOL,
        reference_points=reference_counts, historical_files_checked=len(locks['HISTORY_LOCK']),
        construction_sources_checked=len(locks['SOURCE_LOCK']), all_locked_files_unchanged=True,
        verifier_sha256=sha(__file__), verifier_seconds=time.perf_counter()-start,
        independent_reader='Open3D (production uses plyfile)',
        forbidden_construction_imports_used=False,
        scope='Independent arithmetic audit on existing exposed patches; not new physical validation')
    with (dest/'VERIFICATION.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
