"""Deploy locked X/F/R selectors without reference data access."""
from common import *
import argparse,time,joblib
from threadpoolctl import threadpool_limits
from paired_features import gate_pairs
from train_selectors import design


def main():
    check_host();ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,choices=SCENES,required=True)
    sid=ap.parse_args().scene
    verify_seal(ROOT/'training');verify_seal(ROOT/'evidence'/f'scan{sid}')
    verify_seal(PREV/'inference'/f'scan{sid}')
    lock=json.loads((ROOT/'training/MODEL_LOCK.json').read_text())
    if any(sha(p)!=h for p,h in lock['source_sha256'].items()):raise AssertionError('training source changed')
    models=joblib.load(ROOT/'training/models.joblib')
    thresholds=json.loads((ROOT/'training/THRESHOLDS.json').read_text())
    out=ROOT/'inference'/f'scan{sid}';out.mkdir(parents=True,exist_ok=False)
    sources={str(p):sha(p) for p in (ROOT/'infer_selectors.py',ROOT/'training/SEALED.json',ROOT/'evidence'/f'scan{sid}'/'SEALED.json')}
    save_json(out/'LOCK.json',dict(source_sha256=sources,reference_access=False))
    start=time.monotonic();records=[]
    for path in cases_for_scene(sid):
        tick=time.monotonic();dest=out/path.name;dest.mkdir()
        meta=json.loads((ROOT/'evidence'/f'scan{sid}'/path.name/'META.json').read_text())
        if any(sha(path/f)!=h for f,h in meta['source_sha256'].items()):raise AssertionError('candidate/features changed')
        X=cached_features(path)
        with np.load(ROOT/'evidence'/f'scan{sid}'/path.name/'PAIRED.npz',allow_pickle=False) as z:
            F,R=z['F'],z['R']
            if not np.array_equal(z['row_ids'],np.arange(len(X))):raise AssertionError('inference row mismatch')
        scores={name:model.predict(design(X,F,R)[name]) for name,model in models.items()}
        if any(not np.isfinite(v).all() for v in scores.values()):raise AssertionError('nonfinite score')
        with np.load(PREV/'inference'/f'scan{sid}'/path.name/'decisions.npz',allow_pickle=False) as z:
            decisions={k:z[k] for k in ('identity','A_all','frozen_gain')}
            decisions['previous_normalized']=z['normalized_gain__balanced']
        for name,points in thresholds.items():
            for setting,spec in points.items():decisions[name+'__'+setting]=apply_threshold(scores[name],spec)
        if not np.array_equal(decisions['cached64__balanced'],decisions['previous_normalized']):
            raise AssertionError('matched cached64 no longer reproduces previous normalized')
        decisions['fit_pair_margin']=gate_pairs(F)
        decisions['reserved_pair_margin']=gate_pairs(R)
        decisions['previous_fit_veto']=decisions['previous_normalized']&decisions['fit_pair_margin']
        decisions['previous_reserved_veto']=decisions['previous_normalized']&decisions['reserved_pair_margin']
        save_npz(dest/'decisions.npz',**decisions);save_npz(dest/'scores.npz',**scores)
        p,a=read_points(path/'identity.ply'),read_points(path/'A_all.ply')
        for arm in ('fit_aug__balanced','reserved_aug__balanced','both_aug__balanced','previous_reserved_veto'):
            write_points(dest/(arm+'.ply'),np.where(decisions[arm][:,None],a,p))
        record=dict(case=path.name,source_path=str(path),rows=len(X),
             accepted={k:int(v.sum()) for k,v in decisions.items()},wall_seconds=time.monotonic()-tick,
             reference_access=False)
        save_json(dest/'META.json',record);records.append(record)
        print('INFERRED',path.name,len(decisions),'arms',flush=True)
    save_json(out/'SUMMARY.json',dict(records=records,wall_seconds=time.monotonic()-start))
    seal(out,sources);print('INFERENCE SEALED',sid,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
