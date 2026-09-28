"""21 fixed decisions; replay labels cannot enter this interface."""
from common import *
import argparse,time,joblib
from threadpoolctl import threadpool_limits
from paired_features import gate_pairs

def main():
    check_host();ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,choices=SCENES,required=True)
    sid=ap.parse_args().scene
    for folder in (ROOT/'training',ROOT/'evidence'/f'scan{sid}',PREV/'inference'/f'scan{sid}'):verify_seal(folder)
    lock=json.loads((ROOT/'training/MODEL_LOCK.json').read_text())
    if any(sha(p)!=h for p,h in lock['source_sha256'].items()):raise AssertionError('source changed')
    models=joblib.load(ROOT/'training/models.joblib');thresholds=json.loads((ROOT/'training/THRESHOLDS.json').read_text())
    out=ROOT/'inference'/f'scan{sid}';out.mkdir(parents=True,exist_ok=False)
    sources={str(p):sha(p) for p in (ROOT/'infer_cross.py',ROOT/'training/SEALED.json',
        ROOT/'evidence'/f'scan{sid}'/'SEALED.json',PREV/'inference'/f'scan{sid}'/'SEALED.json')}
    save_json(out/'LOCK.json',dict(source_sha256=sources,replay_reference_access=False))
    records=[];start=time.monotonic()
    for path in cases_for_scene(sid):
        tick=time.monotonic();dest=out/path.name;dest.mkdir()
        meta=json.loads((ROOT/'evidence'/f'scan{sid}'/path.name/'META.json').read_text())
        if any(sha(path/k)!=h for k,h in meta['source_sha256'].items()):raise AssertionError('B changed')
        X=cached_features(path,'B')
        with np.load(ROOT/'evidence'/f'scan{sid}'/path.name/'PAIRED.npz') as z:
            F,R=z['F'],z['R']
            if not np.array_equal(z['row_ids'],np.arange(len(X))):raise AssertionError('rows')
        scores={'B_F':models['B_F'].predict(np.column_stack([X,F])),
                'B_R':models['B_R'].predict(np.column_stack([X,R]))}
        if not all(np.isfinite(v).all() for v in scores.values()):raise AssertionError('scores')
        with np.load(PREV/'inference'/f'scan{sid}'/path.name/'decisions.npz') as old:
            decisions={k:old[k] for k in ('identity','A_all','frozen_gain','previous_normalized')}
            for k,orig in (('A_F','fit_aug'),('A_R','reserved_aug')):
                for setting in ('balanced','native_priority','natural'):decisions[k+'__'+setting]=old[orig+'__'+setting]
            decisions.update(A_F_pair=old['fit_pair_margin'],A_R_pair=old['reserved_pair_margin'])
        decisions['B_all']=np.ones(len(X),bool)
        for k in ('B_F','B_R'):
            for setting,spec in thresholds[k].items():decisions[k+'__'+setting]=apply_threshold(scores[k],spec)
        decisions.update(B_F_pair=gate_pairs(F),B_R_pair=gate_pairs(R))
        if len(decisions)!=21:raise AssertionError('arm count')
        p,b=read_points(path/'identity.ply'),read_points(path/'B_all.ply')
        for arm in ('B_F__balanced','B_R__balanced','B_R__native_priority'):
            write_points(dest/(arm+'.ply'),np.where(decisions[arm][:,None],b,p))
        save_npz(dest/'decisions.npz',**decisions);save_npz(dest/'scores.npz',**scores)
        rec=dict(case=path.name,rows=len(X),accepted={k:int(v.sum()) for k,v in decisions.items()},
                 wall_seconds=time.monotonic()-tick,reference_access=False)
        save_json(dest/'META.json',rec);records.append(rec);print('CROSS INFERRED',path.name,flush=True)
    save_json(out/'SUMMARY.json',dict(records=records,wall_seconds=time.monotonic()-start))
    seal(out,sources);print('CROSS INFERENCE SEALED',sid,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
