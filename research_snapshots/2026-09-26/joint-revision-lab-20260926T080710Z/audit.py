"""Independent audit of development alignment and the frozen joint selector.

This audit never opens replay reference geometry or new replay metric outcomes.
It writes only its own AUDIT.json and AUDIT.md after the model has been sealed.
"""
from pathlib import Path
import ast
import json
import socket
import sys
import unittest

from common import (ROOT, OLD, BASE, CROSS, CLOSEOUT, RESERVED, check_host,
                    development_metadata, load_observations, np, sha,
                    verify_seal, save_json)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def development_alignment():
    """Compare every sampled row against both independently frozen datasets."""
    verify_seal(BASE / 'data')
    verify_seal(CROSS / 'data')
    with np.load(BASE / 'data/train.npz', allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    with np.load(CROSS / 'data/train.npz', allow_pickle=False) as z:
        b = {k: z[k] for k in z.files}
    meta = ('scene', 'condition', 'case_id', 'row_id', 'calibration_support', 'e0')
    for key in meta:
        require(np.array_equal(a[key], b[key]), 'A/B metadata mismatch: ' + key)
    require(len(a['e0']) == 79594, 'changed development row count')
    require(a['calibration_support'].sum() == 78598, 'changed calibration support')
    require(set(a['scene']) == {24, 37}, 'nondevelopment supervision')
    require(len(np.unique(a['case_id'])) == 24, 'changed development cases')
    for candidate in (a, b):
        require(np.allclose(candidate['gain'], candidate['e0'] - candidate['e1'],
                            atol=1e-12, rtol=0), 'raw gain is not e0-e1')
    d = development_metadata()
    require(np.array_equal(d['losses'], np.column_stack([a['e0'], a['e1'], b['e1']])),
            'wrong KEEP/A/B loss alignment')
    raw_seal = json.loads((OLD / 'real_results/SEALED.json').read_text())['files']
    label_seal = json.loads((OLD / 'selector_results/SEALED.json').read_text())['files']
    records = []
    for rec in d['records']:
        pick = d['case_id'] == rec['case_id']
        ids = d['row_id'][pick]
        path = OLD / 'real_results' / rec['case']
        for name in ('identity.ply', 'A_all.ply', 'B_all.ply', 'features.npz'):
            require(sha(path / name) == raw_seal[rec['case'] + '/' + name],
                    'archived candidate or feature changed')
        x, geometry = load_observations(path, rows=ids, support=True)
        require(x.shape[:2] == (len(ids), 2), 'candidate feature shape')
        require(geometry.shape == (len(ids), 3, 3), 'geometry shape')
        require(np.isfinite(x).all() and np.isfinite(geometry).all(), 'nonfinite input')
        require(np.array_equal(x[:, 0, :64], a['X'][pick]), 'A cached features changed')
        require(np.array_equal(x[:, 1, :64], b['X'][pick]), 'B cached features changed')
        for slot, source, candidate in ((0, RESERVED, 'A'), (1, CROSS, 'B')):
            with np.load(source / 'evidence' / f"scan{rec['scene']}" /
                         rec['case'] / 'PAIRED.npz', allow_pickle=False) as z:
                require(np.array_equal(ids, z['row_ids']), candidate + ' evidence row order')
                require(np.array_equal(x[:, slot, 64:96], z['R']),
                        candidate + ' reserved features changed')
        with np.load(ROOT / 'evidence' / f"scan{rec['scene']}" /
                     rec['case'] / 'SUPPORT.npz', allow_pickle=False) as z:
            require(np.array_equal(ids, z['row_ids']), 'new support row order')
            require(np.array_equal(x[:, 0, 96:], z['A']), 'A support changed')
            require(np.array_equal(x[:, 1, 96:], z['B']), 'B support changed')
        scene = int(rec['scene'])
        fold = f'train{scene}_test{37 if scene == 24 else 24}'
        labels = OLD / 'selector_results' / fold / (rec['case'] + '_training_labels.npz')
        require(sha(labels) == label_seal[str(labels.relative_to(OLD / 'selector_results'))],
                'archived labels changed')
        with np.load(labels, allow_pickle=False) as z:
            require(np.array_equal(ids, z['row_ids']), 'historical label row order')
            for slot, label in ((1, 'A_gain'), (2, 'B_gain')):
                require(np.allclose(d['losses'][pick, 0] - d['losses'][pick, slot],
                                    z[label], atol=1e-8, rtol=0),
                        'historical raw mm2 gain mismatch: ' + label)
        records.append({'case': rec['case'], 'rows': len(ids), 'feature_dim': x.shape[2]})
    return {'status': 'PASS', 'rows': len(a['e0']),
            'calibration_rows': int(a['calibration_support'].sum()),
            'scenes': [24, 37], 'case_count': len(records),
            'metadata_equal': list(meta),
            'cached_A_B_different_rows': int(np.any(a['X'] != b['X'], axis=1).sum()),
            'loss_columns': ['KEEP_e0', 'A_e1', 'B_e1'], 'loss_unit': 'mm2',
            'records': records}


def historical_contract():
    paths = [OLD / 'selector_pipeline.py', OLD / 'SELECTOR_PROTOCOL.md',
             CLOSEOUT / 'prepare_release.py',
             CLOSEOUT / 'package/v28_closeout/runtime.py',
             CLOSEOUT / 'package/v28_closeout/models/MODEL_LOCK.json']
    lock = json.loads(paths[-1].read_text())
    require(all(lock['models'][name]['rows'] == 79594 for name in ('post_A', 'post_B')),
            'unexpected historical package training rows')
    return {'source_sha256': {str(p): sha(p) for p in paths},
            'old_decision': 'argmax(0, post_A, post_B); ties KEEP then A',
            'old_target': 'raw source NN squared-error gain e0-eA/B in mm2',
            'old_features': 'separate candidate-specific cached64 post features',
            'old_fit': 'two independent unweighted squared-error HGB regressors',
            'package_rows_per_regressor': 79594,
            'novelty_limit': 'KEEP/A/B selection and raw mm2 gains already existed. '
                            'This experiment changes shared pair context, candidate support '
                            'and case/base-loss weighting; it is a method ablation, '
                            'not established literature novelty.'}


def no_replay_reference_interface():
    """A bounded source review, not a proof of arbitrary dependency behavior."""
    names = ('router.py', 'infer.py', 'support_features.py', 'extract_support.py')
    findings = {}
    forbidden = ('stl055', 'stl065', 'stl069', 'evaluation_references',
                 'nearest_distances', 'ObsMask55', 'ObsMask65', 'ObsMask69')
    for name in names:
        text = (ROOT / name).read_text()
        ast.parse(text, filename=name)
        require(not any(token in text for token in forbidden), 'replay truth marker in ' + name)
        findings[name] = sha(ROOT / name)
    return {'status': 'PASS', 'source_sha256': findings,
            'scope': 'Reviewed array scorer and observation extraction/inference sources; '
                     'no replay-reference path, nearest-reference query or evaluator import. '
                     'Development loss labels are consumed only by training/calibration.'}


def run_router_tests():
    suite = unittest.defaultTestLoader.loadTestsFromName('test_router')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), 'independent pure router tests failed')
    return {'status': 'PASS', 'tests_run': result.testsRun}


def sealed_model_contract():
    import joblib
    from router import (PARAMS, METHODS, SETTINGS, case_weights, contextual_features,
                        predict_shared, routes, materialize)
    from train import calibration_row
    lock = json.loads((ROOT / 'training/MODEL_LOCK.json').read_text())
    pre = json.loads((ROOT / 'training/PRE_FIT_LOCK.json').read_text())
    require(lock['sources'] == pre['sources'], 'prefit/final source lock mismatch')
    for path, digest in lock['sources'].items():
        require(sha(path) == digest, 'method source changed after lock: ' + path)
    require(sha(ROOT / 'training/models.joblib') == lock['model_sha'], 'model hash')
    require(sha(ROOT / 'training/THRESHOLDS.json') == lock['thresholds_sha'], 'threshold hash')
    require(lock['dev_rows'] == 79594 and lock['fit_rows'] == 78598, 'model row counts')
    d = development_metadata()
    fit = d['calibration_support']
    gains = d['losses'][:, :1] - d['losses'][:, 1:]
    weights = case_weights(d['case_id'][fit], d['e0'][fit])
    expected_separate = np.average(gains[fit], axis=0, weights=weights)
    expected_shared = np.average(gains[fit].reshape(-1), weights=np.repeat(weights, 2))
    models = joblib.load(ROOT / 'training/models.joblib')
    require(set(models) == {'independent_absolute', 'joint_common', 'joint_support'},
            'unexpected fitted model family')
    dimensions = {'independent_absolute': 96, 'joint_common': 196, 'joint_support': 260}
    baselines = {}
    for name, artifact in models.items():
        estimators = artifact if isinstance(artifact, list) else [artifact]
        require(len(estimators) == (2 if name == 'independent_absolute' else 1), 'model count')
        baselines[name] = []
        for candidate, model in enumerate(estimators):
            require(model.n_features_in_ == dimensions[name], 'fitted feature width')
            require(model.n_iter_ == PARAMS['max_iter'], 'fitted iteration count')
            require(all(model.get_params()[key] == value for key, value in PARAMS.items()),
                    'HGB parameter mismatch')
            expected = expected_separate[candidate] if name == 'independent_absolute' else expected_shared
            actual = float(np.asarray(model._baseline_prediction).reshape(-1)[0])
            require(np.isclose(actual, expected, rtol=1e-12, atol=1e-12),
                    'model initial prediction disagrees with weighted raw mm2 gain')
            baselines[name].append(actual)
    geometry = np.empty((len(fit), 3, 3), dtype=float)
    sample_x, sample_geometry = [], []
    for rec in d['records']:
        keep = d['case_id'] == rec['case_id']
        x, geo = load_observations(OLD / 'real_results' / rec['case'], d['row_id'][keep], True)
        geometry[keep] = geo
        pick = np.unique(np.linspace(0, len(x) - 1, min(64, len(x)), dtype=int))
        sample_x.append(x[pick])
        sample_geometry.append(geo[pick])
    x, geo = np.concatenate(sample_x), np.concatenate(sample_geometry)
    swapped_geometry = geo[:, [0, 2, 1]]
    for name, width in (('joint_common', 96), ('joint_support', 128)):
        normal = predict_shared(models[name], contextual_features(x[:, :, :width], geo))
        swapped = predict_shared(models[name], contextual_features(x[:, ::-1, :width], swapped_geometry))
        require(np.array_equal(normal[:, ::-1], swapped), 'actual-model swap scores')
        output = materialize(geo, routes(normal, geo))
        swapped_output = materialize(swapped_geometry, routes(swapped, swapped_geometry))
        require(np.array_equal(output, swapped_output), 'actual-model swap geometry')
    choices = json.loads((ROOT / 'training/THRESHOLDS.json').read_text())
    require(set(choices) == set(METHODS), 'threshold method set')
    with np.load(ROOT / 'training/OOF_SCORES.npz', allow_pickle=False) as z:
        for method in METHODS:
            score = z[method]
            require(score.shape == (79594, 2), 'OOF candidate/row alignment')
            require(not np.isnan(score).any() and not np.isposinf(score).any(), 'invalid OOF scores')
            require(set(choices[method]) == set(SETTINGS), 'threshold policy set')
            require(choices[method]['natural']['threshold'] == 0. and
                    choices[method]['natural']['action'] == 'threshold', 'natural policy is not zero utility')
            for setting, specification in choices[method].items():
                recomputed = calibration_row(d, geometry, score, specification['threshold'], specification['action'])
                for key, value in recomputed.items():
                    expected = specification[key]
                    require(value == expected if isinstance(value, str) else
                            np.isclose(value, expected, atol=1e-12, rtol=1e-12),
                            'calibration reproduction: ' + method + '/' + setting + '/' + key)
    return {'status': 'PASS', 'model_dimensions': dimensions,
            'initial_prediction_mm2': baselines, 'weighted_raw_gain_mean_verified': True,
            'fit_rows': int(fit.sum()), 'fit_rows_by_scene': {
                str(scene): int(np.sum(fit & (d['scene'] == scene))) for scene in (24, 37)},
            'actual_model_swap_rows': len(x), 'actual_model_swap_scores_and_outputs_exact': True,
            'calibration_policies_recomputed': len(METHODS) * len(SETTINGS),
            'all_natural_policies_are_strict_zero_threshold': True,
            'source_hashes_verified': len(lock['sources'])}


def main():
    check_host()
    require(Path.cwd().resolve() == ROOT, 'audit target workspace mismatch')
    require((ROOT / 'training/SEALED.json').exists(), 'model must be sealed before final audit')
    verify_seal(ROOT / 'training')
    for scene in (24, 37):
        verify_seal(ROOT / 'evidence' / f'scan{scene}')
    result = {'status': 'PASS', 'host': socket.gethostname(), 'workspace': str(ROOT),
              'historical_contract': historical_contract(),
              'development_alignment': development_alignment(),
              'pure_router_tests': run_router_tests(),
              'sealed_model_contract': sealed_model_contract(),
              'reference_interface_review': no_replay_reference_interface(),
              'new_replay_outcomes_read_by_audit': False,
              'replay_scenes_are_exposed': [55, 65, 69],
              'method_files_sha256': {n: sha(ROOT / n) for n in
                                      ('common.py', 'router.py', 'train.py', 'infer.py')},
              'training_seal_sha256': sha(ROOT / 'training/SEALED.json')}
    save_json(ROOT / 'AUDIT.json', result)
    report = (
        '# Independent implementation audit\n\n'
        'Status: PASS. Exact target: liekkas, `' + str(ROOT) + '`.\n\n'
        'All 79,594 development rows from scenes 24/37 and all 78,598 calibration '
        'rows retain their original case and row identities. A and B losses reproduce '
        'the archived squared-millimetre gain labels. Candidate cached, reserved and '
        'new support features match their own artifacts; B does not reuse A features.\n\n'
        'All 14 independent router tests pass. The fitted HGB initial predictions equal '
        'the independently recomputed weighted raw-gain means on 78,598 eligible rows. '
        'The shared models have 196/260 inputs; all 15 stored policy calibration summaries '
        'reproduce, and all natural policies use strict zero threshold. Actual shared-model '
        'scores and selected coordinates are exactly swap-equivariant on 1,536 sampled '
        'development rows. The audited '
        'method sources and sealed model hashes are recorded in AUDIT.json. No new '
        'replay outcomes or replay reference geometry were opened by this audit.\n\n'
        'V28 already selected argmax(KEEP=0, post_A, post_B) from independent raw-gain '
        'regressors. This run changes the shared pair context, candidate-specific support '
        'and weighting. It does not establish a first KEEP/A/B method or literature '
        'novelty. Scenes 55/65/69 are exposed replay evidence, and scene-fold scores '
        'used for threshold selection are tuning evidence.\n'
    )
    with (ROOT / 'AUDIT.md').open('x') as stream:
        stream.write(report)
    print(json.dumps({'status': result['status'], 'rows': 79594,
                      'tests': result['pure_router_tests']['tests_run']}))


if __name__ == '__main__':
    main()
