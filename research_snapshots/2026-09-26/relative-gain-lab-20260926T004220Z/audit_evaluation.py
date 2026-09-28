"""Independent replay mask, metric, aggregation and geometric spot checks."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import open3d as o3d
from scipy.io import loadmat
from audit_array_checks import metrics, verify_matching

ROOT = Path(__file__).resolve().parent
CLOSEOUT = Path('/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z')
REFERENCE = Path('/srv/slam-research/grf/map-denoise/datasets/closeout-confirmation-v1/evaluation_only')
EDGES = np.array([1e-7, .25, .5, 1., 2., 3., 4., 6.000001])
SPOT_CASES = ('scan55_roi0__native', 'scan65_roi0__minus3', 'scan69_roi0__plus3')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def points(path):
    return np.asarray(o3d.io.read_point_cloud(str(path)).points).copy()


def cloud(p):
    c = o3d.geometry.PointCloud()
    c.points = o3d.utility.Vector3dVector(p)
    return c


def membership(p, roi, obs):
    box = np.all((p >= roi['lo']) & (p <= roi['hi']), axis=1)
    index = np.rint((p - obs['BB'].reshape(-1, 3)[0]) / float(obs['Res'].ravel()[0])).astype(int)
    valid = np.all((index >= 0) & (index < obs['ObsMask'].shape), axis=1)
    seen = np.zeros(len(p), dtype=bool)
    seen[valid] = obs['ObsMask'][tuple(index[valid].T)].astype(bool)
    return box & seen


def geometry_check(case, errors):
    sid = int(case.split('_')[0][4:])
    roi_id = case.split('__')[0]
    source = CLOSEOUT/'confirmation'/f'scan{sid}'
    roi = next(r for r in json.loads((source/'ROIS.json').read_text()) if r['id'] == roi_id)
    ref_source = REFERENCE/f'scan{sid}'
    laser = points(ref_source/f'stl{sid:03d}_total.ply')
    obs = loadmat(ref_source/f'ObsMask{sid}_10.mat')
    ref = cloud(laser[membership(laser, roi, obs)]).voxel_down_sample(0.8)
    native = points(source/(roi_id+'__native')/'identity.ply')
    assert np.array_equal(membership(native, roi, obs), errors['support'])
    del laser
    maximum = 0.0
    saved_points = {}
    for arm, key in [('identity', 'd0'), ('A_all', 'd1')]:
        p = points(source/case/(arm+'.ply'))
        saved_points[arm] = p
        distance = np.asarray(cloud(p).compute_point_cloud_distance(ref))
        discrepancy = float(np.max(np.abs(distance-errors[key])))
        assert discrepancy <= 1e-9, (case, arm, discrepancy)
        maximum = max(maximum, discrepancy)
    displacement2 = np.sum((saved_points['A_all']-saved_points['identity'])**2, axis=1)
    assert np.array_equal(displacement2, errors['movement_squared'])
    return {'case': case, 'independent_nn_max_abs_mm': maximum,
            'support_rows': int(errors['support'].sum()), 'backend': 'Open3D C++ NN versus evaluator scipy cKDTree'}


def main():
    evaluation = ROOT/'evaluation'
    seal = json.loads((evaluation/'SEALED.json').read_text())
    for name, expected in seal['files'].items():
        assert sha(evaluation/name) == expected
    training = json.loads((ROOT/'AUDIT_CALIBRATION.json').read_text())
    assert sha(ROOT/'training/SEALED.json') == training['training_seal_sha256']
    model_lock = json.loads((ROOT/'training/MODEL_LOCK.json').read_text())
    assert sha(ROOT/'training/policies.joblib') == model_lock['policy_sha256']
    assert sha(ROOT/'training/THRESHOLDS.json') == model_lock['thresholds_sha256']
    thresholds = json.loads((ROOT/'training/THRESHOLDS.json').read_text())
    rows = list(csv.DictReader((evaluation/'METRICS.csv').open()))
    lookup = {(row['roi']+'__'+row['condition'], row['arm']): row for row in rows}
    assert len(lookup) == len(rows)
    cases = sorted({key[0] for key in lookup})
    assert len(cases) == 60
    fields = ('source_MSE_mm2', 'source_MAE_mm', 'source_p95_mm', 'improved_fraction', 'harmed_fraction',
              'accepted_fraction', 'accepted_support_fraction', 'moved_fraction', 'move_RMS_mm',
              'benefit_sum_mm2', 'harm_sum_mm2')
    maximum = {key: 0.0 for key in fields}
    diagnostic_count, exported_count, geometry, recalculated = 0, 0, [], {}
    for case in cases:
        sid = int(case.split('_')[0][4:])
        source = ROOT/'inference'/f'scan{sid}'/case
        with np.load(evaluation/case/'point_errors.npz') as z:
            errors = {key: z[key] for key in z.files}
        with np.load(source/'decisions.npz') as z:
            masks = {key: z[key] for key in z.files}
        with np.load(evaluation/case/'diagnostic_decisions.npz') as z:
            diagnostic = {key: z[key] for key in z.files}
        assert not set(masks) & set(diagnostic)
        masks.update(diagnostic)
        with np.load(source/'scores.npz') as scores:
            for method, settings in thresholds.items():
                for setting, spec in settings.items():
                    expected = (np.zeros(len(errors['d0']), dtype=bool) if spec['action'] == 'keep' else
                                np.ones(len(errors['d0']), dtype=bool) if spec['action'] == 'all' else
                                scores[method] > spec['threshold'])
                    assert np.array_equal(masks[method+'__'+setting], expected)
            assert np.array_equal(masks['frozen_gain'], scores['frozen_gain'] > 0)
        historical = CLOSEOUT/'confirmation'/f'scan{sid}'/case
        with np.load(historical/'DECISIONS.npz') as old:
            assert np.array_equal(masks['frozen_gain'], old['post_A_keep'] != 0)
        assert np.array_equal(masks['oracle_fixed_A'], errors['d1'] < errors['d0'])
        displacement = np.sqrt(errors['movement_squared'])
        for arm, accept in masks.items():
            if arm.startswith('random__'):
                prefix, _, seed = arm.rpartition('__seed')
                target, mode = prefix.removeprefix('random__').rsplit('__', 1)
                assert int(seed) in range(10)
                verify_matching(masks[target], accept, displacement, errors['support'], EDGES if mode == 'bin' else None)
                diagnostic_count += 1
            computed = metrics(errors['d0']**2, errors['d1']**2, accept, displacement, errors['support'])
            original = lookup[(case, arm)]
            assert int(original['n_source']) == errors['support'].sum()
            assert int(original['n_rows']) == len(accept)
            for field in fields:
                diff = abs(computed[field]-float(original[field]))
                maximum[field] = max(maximum[field], diff)
                assert np.isclose(computed[field], float(original[field]), rtol=1e-12, atol=1e-8), (case, arm, field, diff)
            recalculated[(case, arm)] = computed
        oracle_mse = recalculated[(case, 'oracle_fixed_A')]['source_MSE_mm2']
        assert all(recalculated[(case, arm)]['source_MSE_mm2'] >= oracle_mse-1e-12 for arm in masks)
        p, a = points(historical/'identity.ply'), points(historical/'A_all.ply')
        for arm in ('frozen_gain', 'normalized_gain__balanced', 'normalized_gain__native_priority'):
            q = points(source/(arm+'.ply'))
            assert np.array_equal(q, np.where(masks[arm][:, None], a, p))
            exported_count += 1
        if case in SPOT_CASES:
            geometry.append(geometry_check(case, errors))
    summary = json.loads((evaluation/'SUMMARY.json').read_text())
    summary_cells = 0
    for condition, methods in summary['exposed_replay'].items():
        for arm, saved in methods.items():
            selected = [case for case in cases if case.endswith('__'+condition)]
            assert len(selected) == 12
            for field in fields:
                means = [np.mean([recalculated[(case, arm)][field] for case in selected if case.startswith(f'scan{sid}_')])
                         for sid in (55, 65, 69)]
                assert np.isclose(np.mean(means), saved[field], rtol=1e-12, atol=1e-8)
            delta = np.array([recalculated[(case, arm)]['source_MSE_mm2']-
                              recalculated[(case, 'identity')]['source_MSE_mm2'] for case in selected])
            assert saved['wins'] == int(np.sum(delta < -1e-12))
            assert saved['losses'] == int(np.sum(delta > 1e-12))
            assert saved['ties'] == int(np.sum(np.abs(delta) <= 1e-12))
            summary_cells += 1
    output = {'status': 'PASS', 'cases': len(cases), 'metric_rows': len(rows),
              'random_masks_exactly_matched': diagnostic_count, 'exported_ply_verified': exported_count,
              'summary_cells': summary_cells, 'maximum_metric_discrepancy': maximum,
              'geometry_spot_checks': geometry, 'evaluation_seal_sha256': sha(evaluation/'SEALED.json'),
              'training_models_thresholds_unchanged_after_replay': True}
    with (ROOT/'AUDIT_EVALUATION.json').open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
