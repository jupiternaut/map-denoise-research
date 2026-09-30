"""Independent sealed-experiment verifier: fixed-layout PLY + exhaustive distances.

Reads original evidence without importing experiment source. Writes only the
requested review/INDEPENDENT_NUMERIC_CHECK.json. Does not rerun any prediction.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
import zlib
import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/colmap-transfer-20260930T180000Z')
assert socket.gethostname() == 'liekkas'
assert Path(__file__).resolve().parent == ROOT / 'review'


def read_json(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def npz(p):
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def read_csv(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def eq(a, b):
    if a is None:
        assert b is None or b == '', (a, b)
    elif isinstance(a, bool):
        assert a == (b is True or b == 'True'), (a, b)
    elif isinstance(a, str):
        assert a == b, (a, b)
    elif isinstance(a, int):
        assert a == int(b), (a, b)
    else:
        assert np.isclose(a, float(b), rtol=1e-12, atol=1e-9), (a, b)


def check_row(expected, saved):
    assert set(expected) == set(saved), (set(expected), set(saved))
    for k, v in expected.items():
        try:
            eq(v, saved[k])
        except AssertionError as e:
            raise AssertionError((k, e.args))


def stats(d, mask):
    x = d[mask]
    assert np.isfinite(x).all()
    return {'n': int(len(x)), 'mse_mm2': float(np.sum(x ** 2) / len(x)) if len(x) else None,
            'mae_mm': float(np.sum(x) / len(x)) if len(x) else None,
            'median_mm': float(np.median(x)) if len(x) else None,
            'correct_1mm': int(np.count_nonzero(x <= 1)), 'wrong_5mm': int(np.count_nonzero(x > 5))}


def read_reference(entry, metadata):
    path = Path(entry['path'])
    assert path == ROOT / 'references' / f"stl{entry['scene']:03d}_total.ply"
    crc, size, h = 0, 0, hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block); size += len(block); crc = zlib.crc32(block, crc)
    assert size == entry['bytes'] == metadata['bytes']
    assert crc == entry['crc32'] == metadata['crc32']
    assert h.hexdigest() == entry['sha256']
    assert entry['member'] == metadata['name']
    assert entry['url'] == 'https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip'
    with path.open('rb') as f:
        lines = []
        while True:
            line = f.readline().strip().decode('ascii')
            lines.append(line)
            if line == 'end_header':
                break
            assert len(lines) < 100
        offset = f.tell()
    assert lines[:2] == ['ply', 'format binary_little_endian 1.0']
    count = int(lines[2].split()[-1])
    assert lines[2].startswith('element vertex ')
    assert lines[3:] == ['property float x', 'property float y', 'property float z',
                         'property float nx', 'property float ny', 'property float nz',
                         'property uchar red', 'property uchar green', 'property uchar blue', 'end_header']
    assert size == offset + 27 * count
    raw = np.memmap(path, mode='r', dtype='uint8', offset=offset)
    xyz = np.ndarray((count, 3), dtype='<f4', buffer=raw, strides=(27, 4)).astype(np.float64)
    assert np.isfinite(xyz).all()
    return xyz, {'scene': entry['scene'], 'vertices': count, 'bytes': size, 'crc32': crc,
                 'sha256': h.hexdigest(), 'header_bytes': offset, 'vertex_stride_bytes': 27,
                 'mtime_ns': path.stat().st_mtime_ns, 'metadata_member': metadata}


def brute_nearest(queries, reference):
    """Exact exhaustive search over every reference vertex, no nearest-neighbor tree."""
    minimum = np.full(len(queries), np.inf)
    nearest = np.full(len(queries), -1, dtype=np.int64)
    for q in range(0, len(queries), 32):
        stop = min(q + 32, len(queries))
        for start in range(0, len(reference), 65536):
            squared = cdist(queries[q:stop], reference[start:start + 65536], 'sqeuclidean')
            ids = np.argmin(squared, axis=1)
            values = squared[np.arange(stop - q), ids]
            better = values < minimum[q:stop]
            minimum[q:stop][better] = values[better]
            nearest[q:stop][better] = start + ids[better]
    assert np.isfinite(minimum).all() and (nearest >= 0).all()
    return np.sqrt(minimum), nearest


def summarize(rows, method):
    selected = [r for r in rows if r['method'] == method and r['scope'] == 'cpu_valid_common']
    return {'method': method, 'roi_equal_mse_mm2': float(np.mean([r['mse_mm2'] for r in selected])),
            'roi_equal_mae_mm': float(np.mean([r['mae_mm'] for r in selected])),
            'common_n': sum(r['n'] for r in selected), 'correct_1mm': sum(r['correct_1mm'] for r in selected),
            'wrong_5mm': sum(r['wrong_5mm'] for r in selected)}


tick = time.monotonic()
seal_path = ROOT / 'PREDICTIONS_SEALED.json'
assert seal_path.is_file()
seal = read_json(seal_path)
lock = read_json(ROOT / 'RUN_LOCK.json')
plan = read_json(ROOT / 'PLAN.json')
mvs = read_json(ROOT / 'MVS_LOCK.json')
inputs = read_json(ROOT / 'INPUT_MANIFEST.json')
manifest = read_json(ROOT / 'references/MANIFEST.json')
selection = read_json(ROOT / 'SCENE_SELECTION.json')
bindings = {}
for mapping in [lock['files'], lock['dependencies'], plan['source_files'], mvs['files'], seal['files'],
                inputs['source_bindings'], selection['source_bindings']]:
    for p, digest in mapping.items():
        path = ROOT / p
        assert sha(path) == digest, path
        if str(path) in bindings:
            assert bindings[str(path)] == digest
        bindings[str(path)] = digest
assert sha(seal_path) == manifest['prediction_seal_sha256']
assert seal['created_utc'] < manifest['created_utc']
assert seal_path.stat().st_mtime_ns < (ROOT / 'references/MANIFEST.json').stat().st_mtime_ns
assert set(plan['scenes']) == {'118', '122'}
assert set(e['scene'] for e in manifest['files']) == {118, 122}
metadata_path = Path('/srv/slam-research/grf/map-denoise/datasets/external-confirmation-20260928T213020Z/METADATA.json')
metadata = read_json(metadata_path)['public_metadata']['reference']['scenes']
result_saved = read_json(ROOT / 'evaluation/RESULTS.json')
saved_points = read_csv(ROOT / 'evaluation/POINT_METRICS.csv')
saved_rows = read_csv(ROOT / 'evaluation/ROI_METRICS.csv')
saved_tail = read_csv(ROOT / 'evaluation/TAIL_METRICS.csv')
saved_brute = read_json(ROOT / 'evaluation/BRUTE_FORCE_CHECK.json')
artifact_names = ['evaluation/RESULTS.json', 'evaluation/POINT_METRICS.csv', 'evaluation/ROI_METRICS.csv',
                  'evaluation/TAIL_METRICS.csv', 'evaluation/BRUTE_FORCE_CHECK.json', 'references/MANIFEST.json',
                  'PREDICTIONS_SEALED.json', 'EVALUATION.log', 'MVS_RUN.log']
artifact_hashes = {name: sha(ROOT / name) for name in artifact_names}
rows, tails, points, reference_checks, all_errors, branch_contributions = [], [], [], [], [], []
distances_by_roi = {}
primary_methods = ['CPU', 'photo_fallback', 'geo_fallback']
exhaustive_queries = 0
max_point_error = 0.
for sid, scene in plan['scenes'].items():
    entry = next(e for e in manifest['files'] if e['scene'] == int(sid))
    expected_member = next(m for m in metadata[sid]['members'] if m['name'] == entry['member'])
    gt, gt_record = read_reference(entry, expected_member)
    assert seal_path.stat().st_mtime_ns < gt_record['mtime_ns']
    reference_checks.append(gt_record)
    arrays = []
    for roi in scene['rois']:
        old = npz(ROOT / 'cpu' / (roi['id'] + '.npz'))
        new = npz(ROOT / 'predictions' / (roi['id'] + '.npz'))
        uv = np.asarray(roi['pixel_xy'])
        assert np.array_equal(old['pixel_xy'], uv) and np.array_equal(new['pixel_xy'], uv)
        arrays.append((roi, old, new))
    qdata, slots = [], []
    for roi, old, new in arrays:
        for method in ['CPU', 'photo', 'geo']:
            valid = old['valid'] if method == 'CPU' else new[method + '_valid']
            xyz = old['xyz_mm'] if method == 'CPU' else new[method + '_xyz_mm']
            assert valid.dtype == bool
            assert np.array_equal(valid, np.isfinite(xyz).all(axis=1))
            qdata.extend(xyz[valid]); slots.append((roi['id'], method, valid.copy()))
    computed, nearest = brute_nearest(np.asarray(qdata), gt)
    exhaustive_queries += len(qdata)
    cursor = 0
    for name, method, valid in slots:
        d = np.full(len(valid), np.nan); nn = np.full(len(valid), -1, dtype=int)
        count = int(valid.sum()); d[valid] = computed[cursor:cursor + count]; nn[valid] = nearest[cursor:cursor + count]
        distances_by_roi.setdefault(name, {})[method] = (d, valid, nn)
        cursor += count
    assert cursor == len(qdata)
    for roi, old, new in arrays:
        name = roi['id']; uv = np.asarray(roi['pixel_xy']); n = len(uv)
        kd, cv, cnn = distances_by_roi[name]['CPU']
        for arm in ['photo', 'geo']:
            d, v, nn = distances_by_roi[name][arm]
            fd = np.where(cv, np.where(v, d, kd), np.nan)
            fn = np.where(cv, np.where(v, nn, cnn), -1)
            distances_by_roi[name][arm + '_fallback'] = (fd, cv.copy(), fn)
            good, bad = cv & (kd <= 1), cv & (kd > 5)
            improved = int(np.count_nonzero(cv & v & (d < kd - 1e-9)))
            worsened = int(np.count_nonzero(cv & v & (d > kd + 1e-9)))
            unchanged = int(np.count_nonzero(cv & ((~v) | (np.abs(d - kd) <= 1e-9))))
            assert improved + worsened + unchanged == int(cv.sum())
            tails.append({'scene': int(sid), 'roi': name, 'method': arm, 'baseline_valid': int(cv.sum()),
                          'improved': improved, 'worsened': worsened, 'unchanged': unchanged,
                          'old_good_1mm': int(good.sum()), 'old_good_harmed_beyond_1mm': int(np.count_nonzero(good & v & (d > 1))),
                          'old_wrong_5mm': int(bad.sum()), 'rescued_5mm': int(np.count_nonzero(bad & v & (d <= 5))),
                          'rescued_1mm': int(np.count_nonzero(bad & v & (d <= 1))),
                          'newly_supported': int(np.count_nonzero((~cv) & v)),
                          'newly_supported_correct_1mm': int(np.count_nonzero((~cv) & v & (d <= 1)))})
            for branch, subset in [('mvs_replaced', cv & v), ('cpu_retained', cv & ~v)]:
                branch_contributions.append({'roi': name, 'scene': int(sid), 'method': arm + '_fallback',
                    'branch': branch, 'n': int(subset.sum()), 'sum_squared_error_mm2': float(np.sum(fd[subset] ** 2)),
                    'roi_equal_mse_contribution_mm2': float(np.sum(fd[subset] ** 2) / (4 * cv.sum()))})
        for method in ['CPU', 'photo', 'photo_fallback', 'geo', 'geo_fallback']:
            d, v, nn = distances_by_roi[name][method]
            for scope, mask in [('all_valid', v), ('cpu_valid_common', v & cv)]:
                stat = stats(d, mask)
                rows.append({'scene': int(sid), 'roi': name, 'method': method, 'scope': scope, 'requested': n,
                             **stat, 'coverage': stat['n'] / n, 'correct_per_requested': stat['correct_1mm'] / n,
                             'paired_cpu_mse_mm2': stats(kd, mask & cv)['mse_mm2']})
            for j, (u, w) in enumerate(uv):
                points.append({'scene': int(sid), 'roi': name, 'query': j, 'u': int(u), 'v': int(w), 'method': method,
                               'valid': bool(v[j]), 'distance_mm': float(d[j]) if v[j] else '',
                               'cpu_valid': bool(cv[j]), 'cpu_distance_mm': float(kd[j]) if cv[j] else ''})
            if method in primary_methods:
                arm = method.split('_')[0]
                for j in np.flatnonzero(cv):
                    replaced = method != 'CPU' and bool(new[arm + '_valid'][j])
                    xyz = new[arm + '_xyz_mm'][j] if replaced else old['xyz_mm'][j]
                    all_errors.append({'scene': int(sid), 'roi': name, 'query': int(j), 'u': int(uv[j, 0]), 'v': int(uv[j, 1]),
                        'method': method, 'distance_mm': float(d[j]), 'cpu_distance_mm': float(kd[j]),
                        'branch': 'mvs_replaced' if replaced else 'cpu_retained', 'xyz_mm': xyz.tolist(),
                        'nearest_reference_vertex': int(nn[j]), 'nearest_reference_xyz_mm': gt[nn[j]].tolist(),
                        'roi_n': int(cv.sum()), 'roi_equal_mse_contribution_mm2': float(d[j] ** 2 / (4 * cv.sum())),
                        'roi_equal_mse_gain_contribution_mm2': float((kd[j] ** 2 - d[j] ** 2) / (4 * cv.sum()))})
    print('EXHAUSTIVE_SCENE_COMPLETE', sid, len(gt), len(qdata), round(time.monotonic() - tick, 2), flush=True)
    del gt

assert len(rows) == len(saved_rows) == 40
assert len(tails) == len(saved_tail) == 8
assert len(points) == len(saved_points) == 2560
for calculated, saved in zip(rows, saved_rows):
    check_row(calculated, saved)
for calculated, saved in zip(tails, saved_tail):
    check_row(calculated, saved)
for calculated, saved in zip(points, saved_points):
    check_row(calculated, saved)
    if calculated['valid']:
        max_point_error = max(max_point_error, abs(calculated['distance_mm'] - float(saved['distance_mm'])))
summary = [summarize(rows, method) for method in primary_methods]
scenes = {sid: [summarize([r for r in rows if r['scene'] == int(sid)], m) for m in primary_methods] for sid in plan['scenes']}
for calculated, saved in zip(summary, result_saved['summary']):
    check_row(calculated, saved)
for sid in scenes:
    for calculated, saved in zip(scenes[sid], result_saved['scenes'][sid]):
        check_row(calculated, saved)
assert result_saved['tail'] == tails
assert result_saved['fixed_query_count'] == 512 and result_saved['baseline_valid'] == 484
assert result_saved['references'] == {str(e['scene']): e for e in manifest['files']}
for entry in saved_brute:
    d, v, _ = distances_by_roi[entry['roi']][entry['method']]
    assert v[entry['query']]
    eq(float(d[entry['query']]), entry['bruteforce_mm']); eq(float(d[entry['query']]), entry['kdtree_mm'])
    eq(abs(entry['bruteforce_mm'] - entry['kdtree_mm']), entry['difference_mm'])
top_errors, concentrated, gains_losses = {}, {}, {}
for summary_row in summary:
    method = summary_row['method']
    selected = sorted([r for r in all_errors if r['method'] == method], key=lambda r: r['roi_equal_mse_contribution_mm2'], reverse=True)
    total = summary_row['roi_equal_mse_mm2']
    for r in selected:
        r['fraction_of_roi_equal_mse'] = r['roi_equal_mse_contribution_mm2'] / total
    top_errors[method] = selected[:10]
    concentrated[method] = {f'top{k}_fraction_of_roi_equal_mse': sum(r['roi_equal_mse_contribution_mm2'] for r in selected[:k]) / total for k in [1, 3, 5, 10]}
    if method != 'CPU':
        by_gain = sorted(selected, key=lambda r: r['roi_equal_mse_gain_contribution_mm2'])
        gains_losses[method] = {'largest_harms': by_gain[:5], 'largest_improvements': list(reversed(by_gain[-5:]))}
tail_totals = {arm: {k: sum(r[k] for r in tails if r['method'] == arm) for k in tails[0] if k not in ('scene', 'roi', 'method')} for arm in ['photo', 'geo']}
relative = {m['method']: (summary[0]['roi_equal_mse_mm2'] - m['roi_equal_mse_mm2']) / summary[0]['roi_equal_mse_mm2'] for m in summary[1:]}
scene_relative = {sid: {r['method']: (ss[0]['roi_equal_mse_mm2'] - r['roi_equal_mse_mm2']) / ss[0]['roi_equal_mse_mm2'] for r in ss[1:]} for sid, ss in scenes.items()}
for p, digest in bindings.items():
    assert sha(p) == digest, p
for p, digest in artifact_hashes.items():
    assert sha(ROOT / p) == digest, p
output = {'status': 'PASS_DETERMINISTIC_CHECKS_ONLY', 'semantic_review': 'separate same-family provisional review',
    'created_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(), 'root': str(ROOT),
    'verifier_sha256': sha(__file__), 'elapsed_seconds': time.monotonic() - tick,
    'distance_algorithm': 'All finite CPU/photo/geo points versus every reference vertex using scipy cdist sqeuclidean in blocks; no KD tree; fallback recomputed from validity only.',
    'reference_reader': 'Independent strict binary little-endian 27-byte vertex layout, direct strided NumPy read after exact header/size checks.',
    'unique_bound_files_verified_before_and_after': len(bindings), 'artifact_hashes': artifact_hashes,
    'reference_checks': reference_checks, 'seal_created_utc': seal['created_utc'], 'reference_manifest_created_utc': manifest['created_utc'],
    'seal_sha256': sha(seal_path), 'point_rows_verified': len(points), 'roi_rows_verified': len(rows),
    'tail_rows_verified': len(tails), 'exhaustive_point_queries': exhaustive_queries,
    'maximum_absolute_saved_distance_difference_mm': max_point_error, 'summary': summary, 'scenes': scenes,
    'relative_roi_equal_mse_improvement': relative, 'scene_relative_roi_equal_mse_improvement': scene_relative,
    'tail_totals': tail_totals, 'branch_contributions': branch_contributions,
    'largest_error_concentration': concentrated, 'largest_errors': top_errors, 'gain_loss_attribution': gains_losses,
    'limits': ['Local artifact/hash/timestamp evidence is not trusted external timestamping or OS-level read tracing.',
               'No inference rerun and no proof of full hardware calibration independence.',
               'No post-hoc exclusion or alternate subset used; largest-error analysis is descriptive only.']}
destination = ROOT / 'review/INDEPENDENT_NUMERIC_CHECK.json'
with destination.open('x') as f:
    json.dump(output, f, indent=2, allow_nan=False); f.write('\n')
print('VERIFIED', exhaustive_queries, 'raw points;', len(points), 'output rows; max difference', max_point_error, 'mm', flush=True)
print(json.dumps({'summary': summary, 'relative': relative, 'scene_relative': scene_relative,
                  'tails': tail_totals, 'concentration': concentrated}, indent=2), flush=True)
