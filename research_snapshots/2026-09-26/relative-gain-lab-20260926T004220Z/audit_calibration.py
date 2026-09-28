"""Independently recompute the sealed OOF threshold grid and selection."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from audit_array_checks import threshold_rows

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equivalent(a, b):
    assert a['action'] == b['action'], (a, b)
    if a['action'] == 'threshold':
        assert a['threshold'] == b['threshold'], (a, b)


def main():
    folder = ROOT/'training'
    seal = json.loads((folder/'SEALED.json').read_text())
    for name, expected in seal['files'].items():
        assert sha(folder/name) == expected
    lock = json.loads((folder/'MODEL_LOCK.json').read_text())
    for name, expected in lock['sources_sha256'].items():
        assert sha(Path(name)) == expected
    saved = json.loads((folder/'THRESHOLDS.json').read_text())
    with np.load(ROOT/'data/train.npz') as data_file:
        data = {key: data_file[key] for key in data_file.files}
    table = list(csv.DictReader((folder/'THRESHOLD_GRID.csv').open()))
    results = []
    with np.load(folder/'OOF_SCORES.npz') as score_file:
        for method in score_file.files:
            score = score_file[method]
            assert score.shape == data['gain'].shape and np.isfinite(score).all()
            expected_thresholds = set(np.quantile(score[data['calibration_support']], np.linspace(0, 1, 101)))
            if lock['defaults'][method] is not None:
                expected_thresholds.add(lock['defaults'][method])
            rows = [row for row in table if row['method'] == method]
            assert set(float(row['threshold']) for row in rows if row['action'] == 'threshold') == expected_thresholds
            assert sum(row['action'] == 'all' for row in rows) == 1
            assert sum(row['action'] == 'keep' for row in rows) == 1
            th = [(-np.inf if row['action'] == 'all' else np.inf if row['action'] == 'keep'
                   else float(row['threshold'])) for row in rows]
            computed = threshold_rows(score, data['e0'], data['e1'], data['case_id'], data['condition'],
                                      np.ones(len(score)), data['calibration_support'], th)
            independent = []
            for original, row in zip(rows, computed):
                mapped = dict(action=original['action'], threshold=None if original['action'] != 'threshold' else row['threshold'],
                              objective_relative_MSE=row['objective'], accepted_fraction=row['accepted_fraction'],
                              native_MSE=row['native_MSE'], native_identity_MSE=row['native_identity_MSE'],
                              injected_relative_MSE=row['injected_relative_MSE'])
                for key in ('objective_relative_MSE', 'accepted_fraction', 'native_MSE', 'native_identity_MSE', 'injected_relative_MSE'):
                    assert np.isclose(mapped[key], float(original[key]), rtol=1e-13, atol=1e-13), (method, key)
                independent.append(mapped)
            key = lambda row: (row['objective_relative_MSE'], row['accepted_fraction'],
                               row['action'] == 'threshold', -row['threshold'] if row['threshold'] is not None else 0)
            equivalent(min(independent, key=key), saved[method]['balanced'])
            feasible = [row for row in independent if row['native_MSE'] <= row['native_identity_MSE'] + 1e-12
                        and row['injected_relative_MSE'] <= 0.95 + 1e-12]
            if feasible:
                equivalent(min(feasible, key=key), saved[method]['native_priority'])
                assert saved[method]['native_priority']['feasible']
            else:
                assert saved[method]['native_priority']['action'] == 'keep'
                assert not saved[method]['native_priority']['feasible']
            results.append({'method': method, 'grid_rows': len(rows), 'native_feasible': bool(feasible)})

        # Two standalone refits check opposite held-out directions without using
        # the orchestrator, fit_all, or policy classes.
        refits = []
        for method, train_scene in [('normalized_gain', 24), ('candidate_error_hgb', 37)]:
            train = data['scene'] == train_scene
            test = ~train
            cids = data['case_id'][train]
            counts = {cid: int(np.sum(cids == cid)) for cid in set(cids)}
            w = np.array([1.0 / counts[cid] for cid in cids])
            w /= w.mean()
            target = (data['gain']/(data['e0'] + data['e1'] + 0.01)) if method == 'normalized_gain' else data['e1']
            model = HistGradientBoostingRegressor(loss='squared_error', learning_rate=.08, max_iter=80,
                max_leaf_nodes=7, max_depth=3, min_samples_leaf=80, l2_regularization=1.,
                early_stopping=False, random_state=20260926)
            model.fit(data['X'][train].astype(float), target[train], sample_weight=w)
            prediction = model.predict(data['X'][test].astype(float))
            if method == 'candidate_error_hgb':
                prediction = -prediction
            error = float(np.max(np.abs(prediction-score_file[method][test])))
            assert error <= 1e-12, (method, error)
            refits.append({'method': method, 'train_scene': train_scene, 'max_score_difference': error})
    output = {'status': 'PASS', 'methods': results, 'threshold_grid_rows': len(table),
              'independent_opposite_fold_refits': refits, 'training_seal_sha256': sha(folder/'SEALED.json'),
              'limitations': ['OOF values are threshold-tuning evidence, not unbiased test performance.']}
    with (ROOT/'AUDIT_CALIBRATION.json').open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
