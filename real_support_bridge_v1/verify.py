"""Independent saved-array checks for the real-support suitability assay.

Does not rerun RANSAC, select geometric models or certify physical surface labels.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import socket
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def close(actual, expected, atol=1e-9):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def projected(points, matrix):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3], homogeneous[:, 2]


def check_box(uv, depth, box):
    left, top, right, bottom = box
    assert np.all(depth > 0)
    assert np.all((uv[:, 0] >= left) & (uv[:, 0] < right))
    assert np.all((uv[:, 1] >= top) & (uv[:, 1] < bottom))


def pair_assessment(descriptors, center):
    if len(descriptors) < 2:
        return dict(provisional_parallel_pair=False,
                    reason='fewer than two supported planes')
    first, second = descriptors[:2]
    n1, n2 = np.asarray(first['normal']), np.asarray(second['normal'])
    cosine = float(np.dot(n1, n2))
    angle = float(np.rad2deg(np.arccos(np.clip(abs(cosine), 0, 1))))
    intercept1 = -(np.dot(center, n1) + first['offset'])
    intercept2 = (-(np.dot(center, n2) + second['offset']) / cosine
                  if abs(cosine) > 1e-9 else None)
    separation = None if intercept2 is None else abs(float(intercept2-intercept1))
    fractions = [first['fraction'], second['fraction']]
    criteria = dict(both_at_least_15pct=all(f >= .15 for f in fractions),
                    union_at_least_70pct=sum(fractions) >= .70,
                    normals_within_5deg=angle <= 5,
                    separation_at_least_05mm=separation is not None and separation >= .5)
    return dict(provisional_parallel_pair=all(criteria.values()), criteria=criteria,
                angle_deg=angle, line_separation_at_center_mm=separation,
                top_two_coverage=sum(fractions), fractions=fractions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    assert socket.gethostname() == 'liekkas'
    assert run.parent == Path('/srv/slam-research/grf/map-denoise/runs')
    report = json.loads((run/'ASSAY.json').read_text())
    inputs = json.loads((run/'INPUTS_FIXED_BEFORE_REFERENCE.json').read_text())
    lock = json.loads((run/'SOURCE_LOCK.json').read_text())
    assert all(Path(path).is_file() and sha(path) == digest for path, digest in lock.items())
    jobs = {job['name']: job for job in inputs['jobs']}
    matrix = np.asarray(inputs['P'], dtype=float)
    close(-np.linalg.solve(matrix[:, :3], matrix[:, 3]), inputs['camera_center'])
    region_records = []
    source_indices = {}
    reference_indices = {}
    max_projection_difference = 0.
    max_residual_summary_difference = 0.
    for row in report['regions']:
        name = row['name']
        job = jobs[name]
        assert job['box'] == row['box']
        input_path = run/'inputs'/f'{name}.npz'
        assert input_path == Path(job['input']) == Path(row['input'])
        with np.load(input_path, allow_pickle=False) as data:
            points = data['points_mm'].copy()
            ids = data['source_ids'].copy()
            uv = data['uv'].copy()
        with np.load(run/'reference'/f'{name}.npz', allow_pickle=False) as data:
            reference = data['points_mm'].copy()
            ref_ids = data['source_ids'].copy()
            ref_uv = data['uv'].copy()
            frame, origin = data['frame'].copy(), data['origin'].copy()
            assignments = {key: data[array].copy() for key, array in
                           [('0.15', 'labels_015'), ('0.3', 'labels_030'), ('0.6', 'labels_060')]}
        assert points.shape == (row['mesh_points'], 3)
        assert reference.shape == (row['reference_points'], 3)
        assert row['mesh_points'] == job['mesh_points']
        for array, indices in [(points, ids), (reference, ref_ids)]:
            assert np.isfinite(array).all()
            assert np.issubdtype(indices.dtype, np.integer)
            assert indices.shape == (len(array),) and np.all(indices >= 0)
            assert len(np.unique(indices)) == len(indices)
        for array, saved_uv in [(points, uv), (reference, ref_uv)]:
            recomputed, depth = projected(array, matrix)
            close(recomputed, saved_uv, atol=1e-7)
            check_box(recomputed, depth, row['box'])
            max_projection_difference = max(max_projection_difference,
                                           float(np.max(abs(recomputed-saved_uv))))
        close(origin, points.mean(axis=0))
        close(frame.T @ frame, np.eye(3))
        covariance = np.cov(points.T)
        transformed = frame.T @ covariance @ frame
        close(transformed-np.diag(np.diag(transformed)), np.zeros((3, 3)))
        assert np.all(np.diff(np.diag(transformed)) <= 1e-9)
        height = (reference-origin) @ frame[:, 2]
        close(float(np.sqrt(np.mean(height**2))), row['reference_pca_height_rms_mm'])
        counts = [row['reference_range_band_counts'][str(t)] for t in (5, 10, 20)]
        assert 0 <= counts[0] <= counts[1] <= counts[2] == len(reference)
        tolerance_records = []
        for key, labels in assignments.items():
            descriptors = row['descriptors'][key]
            assert labels.shape == (len(reference),)
            assert np.issubdtype(labels.dtype, np.integer)
            assert set(np.unique(labels)).issubset(set(range(len(descriptors))) | {-1})
            recomputed_descriptors = []
            plane_records = []
            for label, descriptor in enumerate(descriptors):
                n = np.asarray(descriptor['normal'], dtype=float)
                close(np.linalg.norm(n), 1., atol=1e-10)
                take = labels == label
                count = int(take.sum())
                assert count == descriptor['n'] and count >= 30
                residuals = reference[take] @ n + descriptor['offset']
                rms = float(np.sqrt(np.mean(residuals**2)))
                p95 = float(np.quantile(abs(residuals), .95))
                fraction = count/len(reference)
                close(rms, descriptor['residual_rms_mm'])
                close(p95, descriptor['residual_p95_mm'])
                close(fraction, descriptor['fraction'])
                close(reference[take].mean(axis=0), descriptor['center'])
                max_residual_summary_difference = max(max_residual_summary_difference,
                    abs(rms-descriptor['residual_rms_mm']), abs(p95-descriptor['residual_p95_mm']))
                recomputed_descriptors.append(dict(normal=n, offset=descriptor['offset'], fraction=fraction))
                plane_records.append(dict(label=label, points=count, residual_rms_mm=rms,
                    residual_p95_mm=p95, residual_max_mm=float(abs(residuals).max()),
                    assigned_points_outside_nominal_tolerance=int(np.sum(abs(residuals) > float(key)))))
            rebuilt = pair_assessment(recomputed_descriptors, reference.mean(axis=0))
            expected = row['assessments'][key]
            assert rebuilt['provisional_parallel_pair'] == expected['provisional_parallel_pair']
            if len(recomputed_descriptors) >= 2:
                assert rebuilt['criteria'] == expected['criteria']
                for field in ('angle_deg', 'line_separation_at_center_mm', 'top_two_coverage', 'fractions'):
                    if rebuilt[field] is None:
                        assert expected[field] is None
                    else:
                        close(rebuilt[field], expected[field])
            else:
                assert rebuilt['reason'] == expected['reason']
            assert sum(p['points'] for p in plane_records) + int(np.sum(labels == -1)) == len(reference)
            tolerance_records.append(dict(tolerance_mm=float(key), planes=plane_records,
                unassigned_points=int(np.sum(labels == -1)), assessment=rebuilt))
        source_indices[name], reference_indices[name] = ids, ref_ids
        region_records.append(dict(name=name, input_points=len(points), reference_points=len(reference),
            input_reference_nearest_distance_scope='not independently rescored by this verification',
            reference_range_band_counts_scope='monotonic counts checked; raycasting was not rerun',
            tolerances=tolerance_records))
    assert set(jobs) == set(source_indices)
    overlap = [dict(regions=[first, second],
        shared_mesh_source_ids=int(len(np.intersect1d(source_indices[first], source_indices[second]))),
        shared_reference_source_ids=int(len(np.intersect1d(reference_indices[first], reference_indices[second]))))
        for first, second in itertools.combinations(source_indices, 2)]
    assert all(Path(path).is_file() and sha(path) == digest for path, digest in lock.items())
    output = dict(status='PASS', run=str(run), host=socket.gethostname(),
        checked_regions=len(region_records), checked_plane_descriptions=sum(
            len(t['planes']) for row in region_records for t in row['tolerances']),
        source_lock_checked_files=len(lock), source_lock_unchanged=True,
        max_projection_difference_pixels=max_projection_difference,
        max_residual_summary_difference_mm=max_residual_summary_difference,
        regions=region_records, roi_source_overlap=overlap,
        scope='Saved-array arithmetic, projection, descriptive plane summaries and frozen-source verification only; '
              'not RANSAC replay, physical layer identity validation, visibility certification, or algorithm quality evidence.',
        overlap_warning='Regions are from one exposed scene and may share original points; not independent samples.',
        verifier_sha256=sha(__file__))
    with (run/'VERIFICATION.json').open('x') as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
    print(json.dumps({key: output[key] for key in (
        'status', 'checked_regions', 'checked_plane_descriptions', 'source_lock_checked_files',
        'max_projection_difference_pixels', 'max_residual_summary_difference_mm', 'roi_source_overlap')}, indent=2))


if __name__ == '__main__':
    main()
