"""Apply locked policies to exposed archived candidates, with no reference access."""
from common import *
import argparse
import joblib
import resource
import time
from threadpoolctl import threadpool_limits


def main():
    check_host()
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, required=True, choices=SCENES)
    args = parser.parse_args()
    verify_seal(ROOT/'training')
    model_lock = json.loads((ROOT/'training/MODEL_LOCK.json').read_text())
    for path, digest in model_lock['sources_sha256'].items():
        if sha(Path(path)) != digest:
            raise RuntimeError('locked policy source modified')
    models = joblib.load(ROOT/'training/policies.joblib')
    thresholds = json.loads((ROOT/'training/THRESHOLDS.json').read_text())
    old_model_path = CLOSEOUT/'package/v28_closeout/models/post_A.joblib'
    old_model_spec = json.loads(old_model_path.with_suffix('.json').read_text())
    if sha(old_model_path) != old_model_spec['artifact_sha256']:
        raise RuntimeError('old model checksum mismatch')
    old_model = joblib.load(old_model_path)
    raw = CLOSEOUT/'confirmation'/f'scan{args.scene}'
    cases = sorted(raw.glob('scan*__*'))
    if len(cases) != 20 or {p.name.split('__')[1] for p in cases} != set(CONDS):
        raise AssertionError('replay queue is not four ROIs by five conditions')
    old_seal = json.loads((raw/'SEALED.json').read_text())['files']
    out = ROOT/'inference'/f'scan{args.scene}'
    out.mkdir(parents=True, exist_ok=False)
    sources = {str(p): sha(p) for p in (ROOT/'infer_replay.py', ROOT/'common.py',
                ROOT/'training/SEALED.json', old_model_path, raw/'SEALED.json')}
    save_json(out/'LOCK.json', dict(scene=args.scene, sources_sha256=sources,
              reference_access=False, data_role='EXPOSED_REPLAY_NOT_NEW_CONFIRMATION'))
    start = time.monotonic()
    records = []
    for case in cases:
        dest = out/case.name
        dest.mkdir()
        tick = time.monotonic()
        for filename in ('FEATURES.npz', 'identity.ply', 'A_all.ply', 'DECISIONS.npz'):
            path = case/filename
            if sha(path) != old_seal[str(path.relative_to(raw))]:
                raise RuntimeError('archived case checksum mismatch')
        with np.load(case/'FEATURES.npz', allow_pickle=False) as z:
            X = z['A_all_post']
        scores = {name: policy.score(X) for name, policy in models.items()}
        scores['frozen_gain'] = old_model.predict(X)
        if any(s.shape != (len(X),) or not np.isfinite(s).all() for s in scores.values()):
            raise AssertionError('invalid policy output')
        decisions = {'identity': np.zeros(len(X), dtype=bool), 'A_all': np.ones(len(X), dtype=bool),
                     'frozen_gain': scores['frozen_gain'] > 0}
        with np.load(case/'DECISIONS.npz', allow_pickle=False) as z:
            if not np.array_equal(decisions['frozen_gain'], z['post_A_keep'] != 0):
                raise AssertionError('old policy not reproduced')
        for name, choices in thresholds.items():
            for setting, spec in choices.items():
                decisions[name+'__'+setting] = (
                    np.zeros(len(X), dtype=bool) if spec['action'] == 'keep' else
                    np.ones(len(X), dtype=bool) if spec['action'] == 'all' else
                    scores[name] > spec['threshold'])
        save_npz(dest/'scores.npz', **{k: v.astype(np.float64) for k,v in scores.items()})
        save_npz(dest/'decisions.npz', **decisions)
        p, a = read_points(case/'identity.ply'), read_points(case/'A_all.ply')
        if p.shape != a.shape or len(p) != len(X):
            raise AssertionError('input row mismatch')
        exported = []
        for arm in ('frozen_gain', 'normalized_gain__balanced', 'normalized_gain__native_priority'):
            write_points(dest/(arm+'.ply'), np.where(decisions[arm][:, None], a, p))
            exported.append(arm)
        record = dict(case=case.name, rows=len(X), input_path=str(case),
                      input_sha256={f:sha(case/f) for f in ('identity.ply','A_all.ply','FEATURES.npz')},
                      accepted_counts={k:int(v.sum()) for k,v in decisions.items()},
                      exported=exported, wall_seconds=time.monotonic()-tick,
                      reference_access=False)
        save_json(dest/'INFERENCE.json', record)
        records.append(record)
        print('INFERRED', case.name, len(decisions), 'arms', round(record['wall_seconds'],2),'s', flush=True)
    save_json(out/'SUMMARY.json', dict(records=records, wall_seconds=time.monotonic()-start,
              peak_own_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              gpu=False, reference_access=False))
    seal(out, sources)
    print('INFERENCE SEALED', args.scene, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
