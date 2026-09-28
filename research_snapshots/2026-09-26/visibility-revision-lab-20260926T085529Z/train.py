"""One arm per CPU worker; finalize locks winners before replay scoring."""
from common import *
from learner import *
import argparse,csv,time,resource,joblib

def load_development():
    from visibility_features import FEATURE_NAMES
    d=development_metadata();n=len(d['e0'])
    base=np.empty((n,2,96),np.float32);extra=np.empty((n,2,len(FEATURE_NAMES)),np.float32)
    geometry=np.empty((n,3,3),float)
    for record in d['records']:
        selected=d['case_id']==record['case_id'];case=OLD/'real_results'/record['case']
        base[selected],geometry[selected]=load_observations(case,d['row_id'][selected],False)
        extra[selected]=new_evidence(case,d['row_id'][selected])
    return d,base,extra,geometry

def source_hashes():
    paths=[ROOT/n for n in ('common.py','learner.py','train.py','infer.py','visibility_features.py','PROTOCOL.md')]
    paths += [OUT/'evidence'/f'scan{s}'/'SEALED.json' for s in (24,37)]
    paths += [BASE/'data/SEALED.json',CROSS/'data/SEALED.json',JOINT/'training/SEALED.json']
    return {str(p):sha(p) for p in paths}

def fit_arm(arm):
    for sid in (24,37):verify_seal(OUT/'evidence'/f'scan{sid}')
    out=OUT/'training/arms'/arm;out.mkdir(parents=True,exist_ok=False)
    sources=source_hashes();save_json(out/'PRE_FIT_LOCK.json',dict(sources=sources,arm=arm,
        parameters=parameters(arm),labels='development only, signed MSE gain mm2'))
    tick=time.monotonic();d,base,extra,geo=load_development();x=features_for(base,extra,arm)
    gains=d['losses'][:,:1]-d['losses'][:,1:];oof=np.full((len(x),2),np.nan);folds=[]
    for train_scene in (24,37):
        train=(d['scene']==train_scene)&d['calibration_support'];test=d['scene']!=train_scene
        weights=case_weights(d['case_id'][train],d['e0'][train])
        start=time.monotonic();models=fit_pair(x[train],gains[train],weights,arm)
        oof[test]=predict_pair(models,x[test])
        folds.append(dict(train_scene=train_scene,train_rows=int(train.sum()),test_rows=int(test.sum()),
            train_row_identity_sha=__import__('hashlib').sha256(np.flatnonzero(train).tobytes()).hexdigest(),
            test_row_identity_sha=__import__('hashlib').sha256(np.flatnonzero(test).tobytes()).hexdigest(),
            seconds=time.monotonic()-start))
        print('OOF',arm,train_scene,flush=True)
    assert np.isfinite(oof).all()
    policies,table=calibrate(d,geo,oof)
    save_npz(out/'OOF.npz',scores=oof)
    save_json(out/'THRESHOLDS.json',policies)
    with (out/'CALIBRATION.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    fit=d['calibration_support'];weights=case_weights(d['case_id'][fit],d['e0'][fit])
    models=fit_pair(x[fit],gains[fit],weights,arm);joblib.dump(models,out/'models.joblib')
    assert all(sha(path)==h for path,h in sources.items())
    save_json(out/'SUMMARY.json',dict(arm=arm,features=x.shape[2],fit_rows=int(fit.sum()),
        rows=len(x),folds=folds,seconds=time.monotonic()-tick,
        peak_own_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        baseline_initial_gain=[float(m._baseline_prediction[0,0]) for m in models]))
    seal(out,sources);print('ARM SEALED',arm,policies['balanced'],flush=True)

def finalize():
    out=OUT/'training';thresholds={};sources=source_hashes()
    for arm in ARMS:
        folder=out/'arms'/arm;sealed=verify_seal(folder)
        assert sealed['source']==sources
        thresholds[arm]=json.loads((folder/'THRESHOLDS.json').read_text())
    winners=choose_winners(thresholds)
    save_json(out/'THRESHOLDS.json',thresholds)
    save_json(out/'MODEL_LOCK.json',dict(primary_arm=winners['balanced']['route_arm'],
        recovery_arm=winners['recovery']['route_arm'],expected_route_arms=[a+'__'+p for a in ARMS for p in POLICIES],
        winners=winners,sources=sources,dev_scenes=[24,37],replay_labels_used=False))
    seal(out,sources);print('TRAINING LOCKED',winners,flush=True)

if __name__=='__main__':
    check_host();ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS);ap.add_argument('--finalize',action='store_true')
    args=ap.parse_args()
    if args.finalize:finalize()
    elif args.arm:fit_arm(args.arm)
    else:ap.error('choose --arm or --finalize')
