"""Development-only scene-fold fitting and common operating-point selection."""
from common import *
import csv
import joblib
import time
from threadpoolctl import threadpool_limits
import baseline_models
import innovation_models


def case_weights(case_id):
    unique, inverse, counts = np.unique(case_id, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w / w.mean()


def fit_all(data, ids):
    args = [data[k][ids] for k in ('X', 'gain', 'e0', 'e1')]
    w = case_weights(data['case_id'][ids])
    b = baseline_models.train_policies(*args, w, SEED)
    n = innovation_models.train_policies(*args, w, SEED)
    if set(b) & set(n):
        raise ValueError('duplicate policy names')
    return {**b, **n}


def threshold_statistics(data, score, threshold, action='threshold'):
    accept = (np.zeros(len(score), dtype=bool) if action == 'keep' else
              np.ones(len(score), dtype=bool) if action == 'all' else score > threshold)
    values, native, native_base, injected, fractions = [], [], [], [], []
    for cid in np.unique(data['case_id']):
        ids = (data['case_id'] == cid) & data['calibration_support']
        if not ids.any():
            raise ValueError('empty calibration case')
        e0, e1 = data['e0'][ids], data['e1'][ids]
        after = np.where(accept[ids], e1, e0)
        mse, before = float(after.mean()), float(e0.mean())
        relative = mse / max(before, 1e-6)
        values.append(relative)
        fractions.append(float(accept[ids].mean()))
        condition = data['condition'][ids][0]
        if condition == 'native':
            native.append(mse)
            native_base.append(before)
        else:
            injected.append(relative)
    return dict(threshold=float(threshold) if action == 'threshold' else None, action=action,
                objective_relative_MSE=float(np.mean(values)),
                native_MSE=float(np.mean(native)), native_identity_MSE=float(np.mean(native_base)),
                injected_relative_MSE=float(np.mean(injected)),
                accepted_fraction=float(np.mean(fractions)))


def calibrate(data, score, default):
    if not np.isfinite(score).all():
        raise ValueError('nonfinite calibration scores')
    s = score[data['calibration_support']]
    thresholds = list(np.quantile(s, np.linspace(0., 1., 101)))
    if default is not None:
        thresholds.append(float(default))
    thresholds = sorted(set(thresholds))
    rows = [threshold_statistics(data, score, threshold) for threshold in thresholds]
    rows += [threshold_statistics(data, score, None, action) for action in ('all', 'keep')]
    key = lambda row: (row['objective_relative_MSE'], row['accepted_fraction'],
                       0 if row['action'] != 'threshold' else 1,
                       -row['threshold'] if row['threshold'] is not None else 0)
    balanced = dict(min(rows, key=key), feasible=True, keep_all=False)
    feasible = [r for r in rows if r['native_MSE'] <= r['native_identity_MSE'] + 1e-12
                and r['injected_relative_MSE'] <= .95 + 1e-12]
    if feasible:
        native = dict(min(feasible, key=key), feasible=True, keep_all=False)
    else:
        native = dict(threshold=None, action='keep', feasible=False, keep_all=True,
                      reason='No locked threshold meets native-MSE and 5% recovery constraints')
    choices = {'balanced': balanced, 'native_priority': native}
    if default is not None:
        choices['natural'] = dict(threshold_statistics(data, score, default), feasible=True, keep_all=False)
    return choices, rows


def main():
    check_host()
    verify_seal(ROOT/'data')
    out = ROOT/'training'
    out.mkdir(exist_ok=False)
    sources = {str(ROOT/name): sha(ROOT/name) for name in
               ('PROTOCOL.md', 'common.py', 'train_experiment.py', 'baseline_models.py', 'innovation_models.py')}
    sources[str(ROOT/'data/SEALED.json')] = sha(ROOT/'data/SEALED.json')
    save_json(out/'PRE_FIT_LOCK.json', dict(source_sha256=sources, seed=SEED,
                training_scenes=[24,37], replay_scenes=[55,65,69],
                status='protocol_and_implementations_locked_before_fitting',
                primary_exploratory_method='normalized_gain', threads=1, gpu=False))
    with np.load(ROOT/'data/train.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    start = time.monotonic()
    oof, defaults, timings = {}, {}, []
    for train_scene, test_scene in ((24, 37), (37, 24)):
        tick = time.monotonic()
        train, test = data['scene'] == train_scene, data['scene'] == test_scene
        models = fit_all(data, train)
        if oof and set(oof) != set(models):
            raise AssertionError('method list changed between folds')
        for name, model in models.items():
            if name not in oof:
                oof[name] = np.full(len(data['X']), np.nan)
                defaults[name] = model.default_threshold
            elif defaults[name] != model.default_threshold:
                raise AssertionError('natural threshold changed by fold')
            oof[name][test] = model.score(data['X'][test])
        timings.append(dict(train_scene=train_scene, prediction_scene=test_scene,
                            seconds=time.monotonic()-tick, rows_train=int(train.sum())))
        print('FOLD', train_scene, '->', test_scene, round(timings[-1]['seconds'], 2), 's', flush=True)
    save_npz(out/'OOF_SCORES.npz', **oof)
    thresholds, table = {}, []
    for name, score in oof.items():
        choices, rows = calibrate(data, score, defaults[name])
        thresholds[name] = choices
        table += [dict(method=name, **row) for row in rows]
        print('CALIBRATED', name, 'balanced', round(choices['balanced']['objective_relative_MSE'], 4),
              'native_feasible', choices['native_priority']['feasible'], flush=True)
    with (out/'THRESHOLD_GRID.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    save_json(out/'THRESHOLDS.json', thresholds)
    tick = time.monotonic()
    models = fit_all(data, np.ones(len(data['X']), dtype=bool))
    artifact = out/'policies.joblib'
    joblib.dump(models, artifact)
    timings.append(dict(train_scene='24+37', seconds=time.monotonic()-tick, rows_train=len(data['X'])))
    if any(sha(Path(p)) != value for p, value in sources.items()):
        raise AssertionError('locked source changed during training')
    save_json(out/'MODEL_LOCK.json', dict(methods=list(models), defaults=defaults,
                policy_sha256=sha(artifact), thresholds_sha256=sha(out/'THRESHOLDS.json'),
                sources_sha256=sources, timings=timings, wall_seconds=time.monotonic()-start,
                calibration_performance_is_tuning_not_test=True,
                inference_interface='score(X64) only', replay_labels_used=False))
    seal(out, sources)
    print('TRAINING SEALED', len(models), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
