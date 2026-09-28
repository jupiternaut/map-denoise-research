"""Audit locked development calibration and every replay decision/output.

No replay error arrays or reference geometry are opened by this script.
"""
from common import *
import joblib
from threadpoolctl import threadpool_limits


def stats(data, scores, threshold=None, action='threshold'):
    accepted = (np.ones(len(scores), bool) if action == 'all' else
                np.zeros(len(scores), bool) if action == 'keep' else scores > threshold)
    relative, edits, native, native0, injected = [], [], [], [], []
    for cid in np.unique(data['case_id']):
        use = (data['case_id'] == cid) & data['calibration_support']
        e0 = data['e0'][use]
        after = np.where(accepted[use], data['e1'][use], e0)
        ratio = after.mean()/max(e0.mean(), 1e-6)
        relative.append(ratio)
        edits.append(accepted[use].mean())
        if data['condition'][use][0] == 'native':
            native.append(after.mean()); native0.append(e0.mean())
        else:
            injected.append(ratio)
    return dict(threshold=threshold if action == 'threshold' else None, action=action,
        objective_relative_MSE=float(np.mean(relative)), accepted_fraction=float(np.mean(edits)),
        native_MSE=float(np.mean(native)), native_identity_MSE=float(np.mean(native0)),
        injected_relative_MSE=float(np.mean(injected)))


def assert_stats(expected, saved):
    for key, value in expected.items():
        if isinstance(value, float):
            assert abs(value-saved[key]) <= 1e-12, (key, value, saved[key])
        else:
            assert value == saved[key], (key, value, saved[key])


def main():
    check_host(); verify_seal(ROOT/'training'); verify_seal(PREV/'data')
    with np.load(PREV/'data/train.npz', allow_pickle=False) as src:
        data = {k: src[k] for k in src.files}
    assert len(data['X']) == 79594 and data['calibration_support'].sum() == 78598
    assert set(np.unique(data['scene'])) == {24, 37}
    assert len(np.unique(data['case_id'])) == 24
    thresholds = json.loads((ROOT/'training/THRESHOLDS.json').read_text())
    models = joblib.load(ROOT/'training/models.joblib')
    names = {'cached64': 64, 'fit_aug': 96, 'reserved_aug': 96, 'both_aug': 128}
    assert set(thresholds) == set(models) == set(names)
    with np.load(ROOT/'training/OOF_SCORES.npz', allow_pickle=False) as src:
        oof = {k: src[k] for k in src.files}
    with np.load(PREV/'training/OOF_SCORES.npz', allow_pickle=False) as old:
        assert np.array_equal(oof['cached64'], old['normalized_gain'])
    calibration = []
    for name, width in names.items():
        model = models[name]
        assert model.n_features_in_ == width
        expected_params = dict(loss='squared_error', learning_rate=.08, max_iter=80,
            max_leaf_nodes=7, max_depth=3, min_samples_leaf=80,
            l2_regularization=1., random_state=SEED, early_stopping=False)
        assert all(model.get_params()[key] == value for key, value in expected_params.items())
        score = oof[name]
        assert score.shape == (len(data['X']),) and np.isfinite(score).all()
        grid = sorted(set(np.quantile(score[data['calibration_support']], np.linspace(0, 1, 101)).tolist()+[0.]))
        rows = [stats(data, score, t) for t in grid]+[stats(data, score, action=a) for a in ('all', 'keep')]
        key = lambda row: (row['objective_relative_MSE'], row['accepted_fraction'],
                           0 if row['action'] != 'threshold' else 1,
                           -row['threshold'] if row['threshold'] is not None else 0)
        assert_stats(min(rows, key=key), thresholds[name]['balanced'])
        assert_stats(stats(data, score, 0.), thresholds[name]['natural'])
        feasible = [r for r in rows if r['native_MSE'] <= r['native_identity_MSE']+1e-12 and
                    r['injected_relative_MSE'] <= .95+1e-12]
        saved = thresholds[name]['native_priority']
        if feasible:
            assert saved['feasible'] is True
            assert_stats(min(feasible, key=key), saved)
        else:
            assert saved['feasible'] is False and saved['action'] == 'keep' and saved['keep_all'] is True
        calibration.append(dict(method=name, feature_count=width, grid_size=len(rows),
                                priority_feasible=bool(feasible)))
    records = []
    for sid in SCENES:
        verify_seal(ROOT/'inference'/f'scan{sid}')
        for path in cases_for_scene(sid):
            dest = ROOT/'inference'/f'scan{sid}'/path.name
            X = cached_features(path)
            with np.load(ROOT/'evidence'/f'scan{sid}'/path.name/'PAIRED.npz', allow_pickle=False) as src:
                F, R = src['F'], src['R']
            arrays = dict(cached64=X, fit_aug=np.column_stack((X, F)),
                          reserved_aug=np.column_stack((X, R)), both_aug=np.column_stack((X, F, R)))
            with np.load(dest/'decisions.npz', allow_pickle=False) as src:
                masks = {k: src[k] for k in src.files}
            with np.load(dest/'scores.npz', allow_pickle=False) as src:
                scores = {k: src[k] for k in src.files}
            assert all(m.dtype == bool and m.shape == (len(X),) for m in masks.values())
            for name in names:
                assert np.array_equal(models[name].predict(arrays[name]), scores[name])
                for setting, spec in thresholds[name].items():
                    expected = (np.ones(len(X), bool) if spec['action'] == 'all' else
                                np.zeros(len(X), bool) if spec['action'] == 'keep' else
                                scores[name] > spec['threshold'])
                    assert np.array_equal(expected, masks[name+'__'+setting])
            assert np.array_equal(masks['fit_pair_margin'], (F[:, 12:16].sum(1) >= 2) & (F[:, 24] > 0))
            assert np.array_equal(masks['reserved_pair_margin'], (R[:, 12:16].sum(1) >= 2) & (R[:, 24] > 0))
            with np.load(PREV/'inference'/f'scan{sid}'/path.name/'decisions.npz', allow_pickle=False) as old:
                for name in ('identity', 'A_all', 'frozen_gain'):
                    assert np.array_equal(old[name], masks[name])
                assert np.array_equal(old['normalized_gain__balanced'], masks['previous_normalized'])
            assert np.array_equal(masks['cached64__balanced'], masks['previous_normalized'])
            for suffix in ('fit', 'reserved'):
                assert np.array_equal(masks['previous_'+suffix+'_veto'],
                    masks['previous_normalized'] & masks[suffix+'_pair_margin'])
            p, a = read_points(path/'identity.ply'), read_points(path/'A_all.ply')
            outputs = sorted(dest.glob('*.ply'))
            assert len(outputs) == 4
            for file in outputs:
                assert np.array_equal(read_points(file), np.where(masks[file.stem][:, None], a, p))
            records.append(dict(case=path.name, rows=len(X), masks=len(masks), exact_exports=len(outputs)))
        print('AUDITED SELECTION', sid, flush=True)
    assert len(records) == 60
    save_json(ROOT/'AUDIT_SELECTION.json', dict(status='PASS', calibration=calibration,
        replay_cases=records, case_count=len(records), exact_exports=sum(r['exact_exports'] for r in records),
        replay_labels_used=False, reference_geometry_read=False,
        source_sha256={str(ROOT/name): sha(ROOT/name) for name in
                       ('audit_selection.py', 'train_selectors.py', 'infer_selectors.py', 'common.py')}))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
