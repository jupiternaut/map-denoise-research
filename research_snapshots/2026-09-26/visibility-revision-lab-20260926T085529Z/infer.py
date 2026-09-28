"""Frozen models, input-only evidence; no evaluator distance imports."""
from common import *
from learner import *
import argparse,time,resource,joblib

def main(sid):
    check_host();verify_seal(OUT/'training');verify_seal(OUT/'evidence'/f'scan{sid}')
    out=OUT/'inference'/f'scan{sid}';out.mkdir(parents=True,exist_ok=False)
    lock=json.loads((OUT/'training/MODEL_LOCK.json').read_text())
    thresholds=json.loads((OUT/'training/THRESHOLDS.json').read_text())
    models={a:joblib.load(OUT/'training/arms'/a/'models.joblib') for a in ARMS}
    sources={str(p):sha(p) for p in (ROOT/'common.py',ROOT/'learner.py',ROOT/'infer.py',
        OUT/'training/SEALED.json',OUT/'evidence'/f'scan{sid}'/'SEALED.json')}
    save_json(out/'LOCK.json',dict(sources=sources,primary_arm=lock['primary_arm'],replay_labels=False))
    tick=time.monotonic();records=[]
    for case in cases_for_scene(sid):
        dest=out/case.name;dest.mkdir();t=time.monotonic()
        base,geo=load_observations(case);extra=new_evidence(case)
        scores={a:predict_pair(models[a],features_for(base,extra,a)) for a in ARMS}
        choice={a+'__'+p:routes(scores[a],geo,spec['threshold'],spec['action'])
                for a in ARMS for p,spec in thresholds[a].items()}
        assert set(choice)==set(lock['expected_route_arms'])
        save_npz(dest/'routes.npz',**choice);save_npz(dest/'scores.npz',**scores)
        for a in set((lock['primary_arm'],lock['recovery_arm'])):
            write_points(dest/(a+'.ply'),materialize(geo,choice[a]))
        records.append(dict(case=case.name,rows=len(geo),seconds=time.monotonic()-t,
            counts={a:np.bincount(r,minlength=3).tolist() for a,r in choice.items()}))
        print('INFERRED',case.name,flush=True)
    save_json(out/'SUMMARY.json',dict(records=records,seconds=time.monotonic()-tick,
        peak_own_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    seal(out,sources);print('INFERENCE SEALED',sid,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True,choices=SCENES)
    main(ap.parse_args().scene)
