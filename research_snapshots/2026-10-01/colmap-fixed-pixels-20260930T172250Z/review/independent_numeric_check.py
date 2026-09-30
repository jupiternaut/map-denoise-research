"""Read-only independent input/score audit; writes only its own review JSON.

No author module is imported. --identity does not open GT. --numeric requires
all source/prediction seals before opening official reference PLY files.
"""
import argparse
import csv
import hashlib
import json
import socket
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/colmap-fixed-pixels-20260930T172250Z')
OLD = ROOT.parent / 'camera-pairing-replay-20260930T162130Z'
BASE = ROOT.parent / 'upstream-photo-holdout-20260930T113213Z'


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def check_files(root, manifest):
    for name, expected in manifest.items():
        path = Path(name) if Path(name).is_absolute() else root / name
        assert digest(path) == expected, str(path)
    return len(manifest)


def identity():
    assert socket.gethostname() == 'liekkas'
    assert ROOT.resolve() == ROOT
    lock = read_json(ROOT / 'RUN_LOCK.json')
    counts = {'new_run_lock': check_files(ROOT, lock['files'])}
    origin_seal = read_json(BASE / 'SEALED.json')
    counts['original_seal'] = check_files(BASE, origin_seal['files'])
    counts['original_external_sources'] = check_files(BASE, origin_seal['external_sources'])
    counts['replay_prediction_seal'] = check_files(OLD, read_json(OLD / 'PREDICTIONS_SEALED.json')['files'])
    seal = ROOT / 'PREDICTIONS_SEALED.json'
    if seal.exists():
        counts['new_prediction_seal'] = check_files(ROOT, read_json(seal)['files'])
    dependencies = {str(BASE / f): digest(BASE / f) for f in ['evaluate.py', 'audit_sources.py']}
    for name in ['evaluate.py', 'audit_sources.py']:
        assert dependencies[str(BASE / name)] == origin_seal['files'][name]
    plan = read_json(OLD / 'PLAN_CORRECTED.json')
    manifest = read_json(OLD / 'QUERY_MANIFEST.json')['records']
    max_p = max_ray = 0.0
    image_count = 0
    views = set()
    workspaces = sorted({j['workspace'] for j in lock['jobs']})
    assert len(workspaces) == 5 and len(lock['jobs']) == 15
    for workspace in workspaces:
        ws = Path(workspace)
        sid = ws.parent.name.removeprefix('scan')
        spec = plan['scenes'][sid]
        camera = {}
        for line in (ws / 'sparse/cameras.txt').read_text().splitlines():
            v = line.split()
            assert v[1] == 'PINHOLE'
            camera[int(v[0])] = (tuple(map(int, v[2:4])), np.array([
                [float(v[4]), 0, float(v[6])], [0, float(v[5]), float(v[7])], [0, 0, 1]]))
        names = []
        for line in (ws / 'sparse/images.txt').read_text().splitlines():
            if not line.strip():
                continue
            v = line.split()
            name = v[9]
            names.append(name)
            values = np.array(v[1:8], float)
            rotation = Rotation.from_quat([*values[1:4], values[0]]).as_matrix()
            translation = values[4:7]
            size, calibration = camera[int(v[8])]
            expected = np.array(spec['cameras'][name]['P'])
            projected = calibration @ np.column_stack([rotation, translation])
            max_p = max(max_p, float(abs(projected - expected).max()))
            uv = np.array([[0, 0], [388, 290], [776, 580], [123, 321]], float)
            depth = np.array([559., 680., 786., 700.])
            homogeneous = np.column_stack([uv, np.ones(len(uv))])
            world = (homogeneous @ np.linalg.inv(calibration).T * depth[:, None] - translation) @ rotation
            center = -np.linalg.solve(expected[:, :3], expected[:, 3])
            expected_world = center + homogeneous @ np.linalg.inv(expected[:, :3]).T * depth[:, None]
            max_ray = max(max_ray, float(abs(world - expected_world).max()))
            with Image.open(spec['cameras'][name]['image_path']) as image:
                rgb = np.asarray(image.convert('RGB').resize(size, Image.Resampling.BILINEAR))
            # Preserve the exact original matmul operation: np.dot's different
            # accumulation order can change uint8 truncation on integer ties.
            legacy = np.clip(rgb @ np.array([.299, .587, .114]), 0, 255).astype(np.uint8)
            with Image.open(ws / 'images' / name) as image:
                actual = np.array(image)
            assert actual.shape == (581, 777) and np.array_equal(actual, legacy), (ws, name)
            image_count += 1
            views.add((sid, name))
            config = (ws / 'configs' / f'{name}.cfg').read_text().splitlines()
            assert config == [name, ', '.join(n for n in [spec['reference']] + spec['Q'] if n != name)]
        assert names == [spec['reference']] + spec['Q']
        assert (ws / 'sparse/points3D.txt').read_bytes() == b''
    query_count = valid_count = 0
    for sid, spec in plan['scenes'].items():
        for roi in spec['rois']:
            name = roi['id']
            a = arrays(OLD / 'initialization' / f'scan{sid}' / name / 'input.npz')
            m = sorted([r for r in manifest if r['roi'] == name], key=lambda r: r['query_old_id'])
            assert [r['query_old_id'] for r in m] == list(range(128))
            assert np.array_equal(a['fixed_query_pixel_xy'], [r['pixel_xy'] for r in m])
            assert np.array_equal(a['construction_success'], [r['u1_valid'] for r in m])
            q = a['query_old_ids']
            assert np.array_equal(q, np.flatnonzero(a['construction_success']))
            assert np.array_equal(a['reference_pixel_xy'][a['query_ids']], a['fixed_query_pixel_xy'][q])
            for r in m:
                assert r['pixel_id'] == name + ':' + hashlib.sha256(np.array(r['pixel_xy'], np.float64).tobytes()).hexdigest()
            query_count += len(m)
            valid_count += len(q)
    assert query_count == 512 and valid_count == 497 and max_p < 1e-7 and max_ray < 1e-8
    return dict(status='PASS', seals=counts, dependencies=dependencies,
                dependencies_match_original_seal=True, gray_image_copies_equal=image_count,
                distinct_scene_view_pairs=len(views), workspaces=len(workspaces), fixed_queries=query_count,
                cpu_valid=valid_count, serialized_camera_max_abs_P_difference=max_p,
                serialized_camera_max_abs_ray_mm=max_ray, gt_accessed=False)


def read_ply(path):
    """Independent minimal PLY scalar-vertex reader, without author imports."""
    types = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
             'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
             'ushort': 'u2', 'uint16': 'u2', 'short': 'i2', 'int16': 'i2',
             'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4'}
    with Path(path).open('rb') as handle:
        assert handle.readline().strip() == b'ply'
        fields = []
        count = None
        current = None
        fmt = None
        while True:
            raw = handle.readline()
            assert raw, 'Truncated header'
            parts = raw.decode('ascii').split()
            if not parts:
                continue
            if parts[0] == 'end_header':
                break
            if parts[0] == 'format':
                fmt = parts[1]
            if parts[0] == 'element':
                if count is None:
                    assert parts[1] == 'vertex', 'Expected vertex as first element'
                current = parts[1]
                if current == 'vertex':
                    count = int(parts[2])
            if parts[0] == 'property' and current == 'vertex':
                assert parts[1] in types
                fields.append((parts[2], types[parts[1]]))
        assert count and all(c in dict(fields) for c in ['x', 'y', 'z'])
        if fmt == 'ascii':
            raw = np.loadtxt(handle, max_rows=count)
            indexes = [list(dict(fields)).index(c) for c in ['x', 'y', 'z']]
            xyz = raw[:, indexes]
        else:
            assert fmt in ['binary_little_endian', 'binary_big_endian']
            endian = '<' if fmt == 'binary_little_endian' else '>'
            raw = np.fromfile(handle, dtype=[(key, endian + value) for key, value in fields], count=count)
            assert len(raw) == count
            xyz = np.column_stack([raw[c] for c in ['x', 'y', 'z']])
    xyz = xyz.astype(np.float64)
    assert xyz.shape == (count, 3) and np.isfinite(xyz).all()
    return xyz


def depth_map(path):
    with Path(path).open('rb') as handle:
        text = bytearray()
        while text.count(b'&') != 3:
            item = handle.read(1)
            assert item
            text.extend(item)
        width, height, channels = map(int, text.decode().strip('&').split('&'))
        raw = np.fromfile(handle, '<f4')
        assert channels == 1 and raw.size == width * height
        return raw.reshape(height, width)


def numbers(distances):
    return dict(n=len(distances), mse_mm2=float(np.mean(np.square(distances))) if len(distances) else None,
                mae_mm=float(np.mean(distances)) if len(distances) else None,
                median_mm=float(np.median(distances)) if len(distances) else None,
                correct_1mm=int(np.count_nonzero(distances <= 1)),
                wrong_5mm=int(np.count_nonzero(distances > 5)))


def numeric(preflight):
    assert 'new_prediction_seal' in preflight['seals']
    definitions = {
        24: ('/srv/slam-research/grf/map-denoise/datasets/published-outputs-v2-reference/stl024_total.ply',
             '/srv/slam-research/grf/map-denoise/datasets/published-outputs-v2-reference/MANIFEST.json'),
        37: ('/srv/slam-research/grf/map-denoise/datasets/reconstruction-v22-scan37/stl037_total.ply',
             '/srv/slam-research/grf/map-denoise/datasets/reconstruction-v22-scan37/DOWNLOAD_RETRY_1789282524663424354.json')}
    plan = read_json(OLD / 'PLAN_CORRECTED.json')
    reference_records = {}
    all_rows, tails, repeat, details, brute_checks = [], [], {}, {}, []
    for sid, spec in plan['scenes'].items():
        gt_path, metadata_path = definitions[int(sid)]
        metadata = read_json(metadata_path)
        metadata_rows = metadata if isinstance(metadata, list) else metadata['files']
        row = [r for r in metadata_rows if r.get('path') == gt_path and r.get('member') == f'Points/stl/stl{int(sid):03d}_total.ply']
        assert len(row) == 1 and row[0]['url'] == 'https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip'
        assert digest(gt_path) == row[0]['sha256']
        gt = read_ply(gt_path)
        tree = KDTree(gt)
        reference_records[sid] = dict(path=gt_path, sha256=digest(gt_path), vertices=len(gt), official_member=row[0]['member'])
        for roi in spec['rois']:
            name = roi['id']
            initial = arrays(OLD / 'initialization' / f'scan{sid}' / name / 'input.npz')
            new = arrays(ROOT / 'predictions' / f'{name}.npz')
            uv = initial['fixed_query_pixel_xy']
            assert np.array_equal(new['pixel_xy'], uv) and np.array_equal(new['query_old_ids'], np.arange(128))
            q = initial['query_old_ids']
            cpu_mask = np.zeros(128, bool)
            cpu_mask[q] = True
            cpu_xyz = initial['points_mm'][initial['query_ids']]
            candidate = arrays(OLD / 'predictions/U1' / f'{name}__native' / 'candidates/CANDIDATES.npz')
            assert np.array_equal(candidate['query_old_ids'], q)
            assert np.array_equal(candidate['geometry_mm'][:, 0], cpu_xyz)
            candidate_errors = np.stack([tree.query(candidate['geometry_mm'][:, i], workers=1)[0] for i in range(3)], axis=1)
            cpu = np.full(128, np.nan)
            cpu[q] = candidate_errors[:, 0]
            pool = np.full(128, np.nan)
            pool[q] = candidate_errors.min(axis=1)
            errors = {'CPU': cpu, 'old_pool_oracle': pool}
            masks = {'CPU': cpu_mask, 'old_pool_oracle': cpu_mask}
            dense_p = np.array(spec['cameras'][spec['reference']]['P'])
            center = -np.linalg.solve(dense_p[:, :3], dense_p[:, 3])
            rays = np.linalg.solve(dense_p[:, :3], np.column_stack([uv, np.ones(128)]).T).T
            for method, variant, suffix in [('photo', 'photo', 'photometric'), ('geo', 'geo', 'geometric'), ('geo_unfiltered', 'geo', 'photometric')]:
                xyz = new[method + '_xyz_mm']
                valid = new[method + '_valid']
                z = new[method + '_depth_mm']
                assert xyz.shape == (128, 3) and valid.dtype == bool and valid.shape == (128,)
                assert np.array_equal(valid, np.isfinite(xyz).all(axis=1))
                assert np.array_equal(valid, np.isfinite(z) & (z > 0))
                disk = depth_map(ROOT / 'workspaces' / f'scan{sid}' / variant / 'stereo/depth_maps' / f'{spec["reference"]}.{suffix}.bin')
                assert np.array_equal(z, disk[uv[:, 1].astype(int), uv[:, 0].astype(int)])
                assert np.allclose(xyz[valid], (center + rays * z[:, None])[valid], rtol=0, atol=1e-8)
                d = np.full(128, np.nan)
                d[valid] = tree.query(xyz[valid], workers=1)[0]
                errors[method], masks[method] = d, valid
                # Index-by-index construction, independent of evaluator's masked replacement.
                fallback = np.full(128, np.nan)
                enhanced = np.full(128, np.nan)
                for i in q:
                    fallback[i] = d[i] if valid[i] else cpu[i]
                    enhanced[i] = min(pool[i], d[i]) if valid[i] else pool[i]
                errors[method + '_fallback'], masks[method + '_fallback'] = fallback, cpu_mask
                errors[method + '_oracle'], masks[method + '_oracle'] = enhanced, cpu_mask
                hard_ids = [i for i in q if pool[i] > 5]
                good_ids = [i for i in q if cpu[i] <= 1]
                tails.append(dict(scene=int(sid), roi=name, method=method, old_hard=len(hard_ids),
                    hard_supported=sum(bool(valid[i]) for i in hard_ids),
                    hard_better=sum(bool(valid[i] and d[i] < pool[i]) for i in hard_ids),
                    hard_rescued_5mm=sum(bool(valid[i] and d[i] <= 5) for i in hard_ids),
                    hard_rescued_1mm=sum(bool(valid[i] and d[i] <= 1) for i in hard_ids),
                    old_good_1mm=len(good_ids), old_good_moved_beyond_1mm=sum(bool(valid[i] and d[i] > 1) for i in good_ids),
                    improved=sum(bool(valid[i] and d[i] < cpu[i] - 1e-9) for i in q),
                    worsened=sum(bool(valid[i] and d[i] > cpu[i] + 1e-9) for i in q),
                    unchanged_or_missing=sum(bool(not valid[i] or abs(d[i] - cpu[i]) <= 1e-9) for i in q),
                    newly_supported_15=sum(bool(valid[i]) for i in range(128) if not cpu_mask[i]),
                    newly_supported_correct_1mm=sum(bool(valid[i] and d[i] <= 1) for i in range(128) if not cpu_mask[i])))
            combined = np.full(128, np.nan)
            for i in q:
                choices = [pool[i]] + [errors[m][i] for m in ['photo', 'geo'] if masks[m][i]]
                combined[i] = min(choices)
            errors['enriched_pool_oracle'], masks['enriched_pool_oracle'] = combined, cpu_mask
            for method, d in errors.items():
                v = masks[method]
                for scope in ['all_valid', 'cpu_valid_common']:
                    selected = v if scope == 'all_valid' else v & cpu_mask
                    all_rows.append(dict(scene=int(sid), roi=name, method=method, scope=scope, requested=128,
                        **numbers(d[selected]), coverage=float(v.mean()), correct_per_requested=int(np.count_nonzero(v & (d <= 1))) / 128,
                        paired_cpu_mse_mm2=numbers(cpu[selected & cpu_mask])['mse_mm2']))
            if sid == '24':
                repeat[name] = dict(same_valid=bool(np.array_equal(new['photo_valid'], new['photo_repeat_valid'])),
                    same_depth=bool(np.array_equal(new['photo_depth_mm'], new['photo_repeat_depth_mm'])),
                    max_depth_difference_mm=float(np.max(abs(new['photo_depth_mm'] - new['photo_repeat_depth_mm']))))
                repeat_disk = depth_map(ROOT / 'workspaces/scan24/photo_repeat/stereo/depth_maps/0022.png.photometric.bin')
                assert np.array_equal(new['photo_repeat_depth_mm'], repeat_disk[uv[:, 1].astype(int), uv[:, 0].astype(int)])
            details[name] = {m: dict(distance_mm=[float(x) if np.isfinite(x) else None for x in d], valid=masks[m].tolist()) for m, d in errors.items()}
            # Three query identities per ROI, fixed by ID not by performance;
            # every candidate ray is compared to the full GT by brute force.
            for local_index in [0, len(q) // 2, len(q) - 1]:
                point = cpu_xyz[local_index]
                brute = float(np.sqrt(np.min(np.sum((gt - point) ** 2, axis=1))))
                kd = float(cpu[q[local_index]])
                assert abs(brute - kd) < 1e-10
                brute_checks.append(dict(roi=name, query_old_id=int(q[local_index]), brute_mm=brute, tree_mm=kd, abs_difference=abs(brute-kd)))
    summaries = []
    for method in sorted({r['method'] for r in all_rows}):
        full = [r for r in all_rows if r['method'] == method and r['scope'] == 'all_valid']
        common = [r for r in all_rows if r['method'] == method and r['scope'] == 'cpu_valid_common']
        def equal_mean(rows, key):
            return float(np.mean([r[key] for r in rows])) if all(r['n'] for r in rows) else None
        summaries.append(dict(method=method, roi_equal_mse_mm2=equal_mean(full, 'mse_mm2'),
            coverage_n=sum(r['n'] for r in full), requested=512, correct_1mm=sum(r['correct_1mm'] for r in full),
            wrong_5mm=sum(r['wrong_5mm'] for r in full), common_n=sum(r['n'] for r in common),
            common_roi_equal_mse_mm2=equal_mean(common, 'mse_mm2'), common_cpu_roi_equal_mse_mm2=equal_mean(common, 'paired_cpu_mse_mm2')))
    compared = 0
    max_abs = 0.0
    def compare(a, b, location):
        nonlocal compared, max_abs
        if isinstance(a, dict):
            assert set(a) == set(b), location
            for key in a:
                compare(a[key], b[key], location + '/' + key)
        elif isinstance(a, list):
            assert len(a) == len(b), location
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, location + '/' + str(i))
        elif a is None:
            assert b is None or b == '', location
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            difference = abs(a - float(b))
            max_abs = max(max_abs, difference)
            compared += 1
            assert difference <= 1e-9 + abs(a) * 1e-12, (location, a, b)
        else:
            assert a == b, (location, a, b)
    author = read_json(ROOT / 'evaluation/RESULTS.json')
    compare(summaries, author['summary'], 'summary')
    compare(tails, author['tail'], 'tail')
    compare(repeat, author['repeat'], 'repeat')
    compare(details, read_json(ROOT / 'evaluation/POINT_DISTANCES.json'), 'point_distances')
    with (ROOT / 'evaluation/ROI_METRICS.csv').open() as handle:
        author_rows = list(csv.DictReader(handle))
    row_key = lambda r: (r['roi'], r['method'], r['scope'])
    expected_rows = {row_key(r): r for r in all_rows}
    assert len(expected_rows) == len(author_rows) == 96
    for row in author_rows:
        compare(expected_rows[row_key(row)], row, 'roi_metrics/' + str(row_key(row)))
    hard_totals = {m: sum(r['old_hard'] for r in tails if r['method'] == m) for m in ['photo', 'geo', 'geo_unfiltered']}
    assert set(hard_totals.values()) == {75}
    assert all(r['coverage_n'] == 497 for r in summaries if r['method'].endswith('_fallback'))
    return dict(status='PASS', reference_type='official dataset GT, custom nearest-point two-scene development evaluation',
        references=reference_records, numeric_fields_compared=compared, max_absolute_numeric_difference=max_abs,
        roi_metric_records=len(all_rows), summaries=summaries, hard_strata_totals=hard_totals,
        tail=tails, repeat=repeat, brute_force_checks=brute_checks, gt_accessed=True,
        nearest_neighbor_implementation='independent PLY reader + scipy KDTree; 12 full-reference brute-force checks',
        evaluator_imported=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', action='store_true')
    parser.add_argument('--numeric', action='store_true')
    args = parser.parse_args()
    assert args.identity != args.numeric
    preflight = identity()
    if args.identity:
        print(json.dumps(preflight, indent=2))
    else:
        result = dict(reviewer_status='existing', review_independence='same-family', acceptance_status='provisional',
                      identity=preflight, numeric=numeric(preflight), script_sha256=digest(__file__),
                      audited_seal_sha256=digest(ROOT / 'PREDICTIONS_SEALED.json'),
                      audited_run_lock_sha256=digest(ROOT / 'RUN_LOCK.json'),
                      author_results_sha256=digest(ROOT / 'evaluation/RESULTS.json'))
        with (ROOT / 'review/INDEPENDENT_NUMERIC_CHECK.json').open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        print(json.dumps({k: result['numeric'][k] for k in ['status', 'numeric_fields_compared', 'max_absolute_numeric_difference', 'roi_metric_records', 'hard_strata_totals']}, indent=2))
