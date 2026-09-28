"""Observation-only deployment path. No training/evaluation loss files imported."""
from common import *
from router import *
import argparse,time,joblib
from threadpoolctl import threadpool_limits

def main():
    check_host();parser=argparse.ArgumentParser();parser.add_argument('--scene',type=int,choices=SCENES,required=True)
    sid=parser.parse_args().scene
    for folder in (ROOT/'training',ROOT/'evidence'/f'scan{sid}'):verify_seal(folder)
    out=ROOT/'inference'/f'scan{sid}';out.mkdir(parents=True,exist_ok=False)
    models=joblib.load(ROOT/'training/models.joblib')
    old_a=joblib.load(RESERVED/'training/models.joblib')['reserved_aug']
    old_b=joblib.load(CROSS/'training/models.joblib')['B_R']
    thresholds=json.loads((ROOT/'training/THRESHOLDS.json').read_text())
    sources={str(p):sha(p) for p in (ROOT/'common.py',ROOT/'router.py',ROOT/'infer.py',
         ROOT/'training/SEALED.json',ROOT/'evidence'/f'scan{sid}'/'SEALED.json',
         RESERVED/'training/models.joblib',CROSS/'training/models.joblib')}
    save_json(out/'LOCK.json',dict(sources=sources,replay_reference_access=False))
    start=time.monotonic();records=[]
    for case in cases_for_scene(sid):
        tick=time.monotonic();dest=out/case.name;dest.mkdir()
        features,geometry=load_observations(case,support=True)
        scores={'independent_absolute':np.column_stack([m.predict(features[:,c,:96])
                                      for c,m in enumerate(models['independent_absolute'])]),
                'normalized_max':np.column_stack([old_a.predict(features[:,0,:96]),old_b.predict(features[:,1,:96])]),
                'support_margin':support_scores(features)}
        for method,width in (('joint_common',96),('joint_support',128)):
            scores[method]=predict_shared(models[method],contextual_features(features[:,:,:width],geometry))
        decisions={k+'__'+setting:routes(scores[k],geometry,spec['threshold'],spec['action'])
                   for k in METHODS for setting,spec in thresholds[k].items()}
        assert len(decisions)==15
        save_npz(dest/'routes.npz',**decisions);save_npz(dest/'scores.npz',**scores)
        for arm in ('joint_common__balanced','joint_support__balanced'):
            write_points(dest/(arm+'.ply'),materialize(geometry,decisions[arm]))
        records.append(dict(case=case.name,rows=len(features),seconds=time.monotonic()-tick,
              routes={k:np.bincount(v,minlength=3).tolist() for k,v in decisions.items()}))
        print('INFERRED',case.name,flush=True)
    save_json(out/'SUMMARY.json',dict(records=records,seconds=time.monotonic()-start))
    seal(out,sources);print('INFERENCE SEALED',sid,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
