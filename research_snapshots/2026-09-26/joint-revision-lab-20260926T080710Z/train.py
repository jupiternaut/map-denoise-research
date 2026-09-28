"""Development-only shared utility fit. One protocol, no replay labels."""
from common import *
from router import *
import csv, time, joblib
from threadpoolctl import threadpool_limits

def calibration_row(d,geometry,score,threshold=0.,action='threshold'):
    route=routes(score,geometry,threshold,action)
    after=d['losses'][np.arange(len(route)),route]
    relative,native,base,perturbed,accepted=[],[],[],[],[]
    for cid in np.unique(d['case_id']):
        ids=(d['case_id']==cid)&d['calibration_support']
        before=float(d['losses'][ids,0].mean());mse=float(after[ids].mean())
        relative.append(mse/max(before,1e-6));accepted.append(float((route[ids]!=0).mean()))
        if d['condition'][ids][0]=='native':native.append(mse);base.append(before)
        else:perturbed.append(relative[-1])
    return dict(threshold=float(threshold),action=action,
                objective_relative_MSE=float(np.mean(relative)),native_MSE=float(np.mean(native)),
                native_identity_MSE=float(np.mean(base)),injected_relative_MSE=float(np.mean(perturbed)),
                accepted_fraction=float(np.mean(accepted)))

def calibrate(d,geometry,score):
    best=canonical_scores(score,geometry).max(axis=1)
    vals=best[d['calibration_support']&np.isfinite(best)]
    ts=sorted(set([0.]+list(np.quantile(vals,np.linspace(0,1,100))))) if len(vals) else [0.]
    table=[calibration_row(d,geometry,score,t) for t in ts]
    keep=calibration_row(d,geometry,score,0.,'keep');table.append(keep)
    key=lambda x:(x['objective_relative_MSE'],x['accepted_fraction'],-x['threshold'])
    balanced=dict(min(table,key=key),feasible=True)
    allowed=[r for r in table if r['native_MSE']<=r['native_identity_MSE']+1e-12
             and r['injected_relative_MSE']<=.95+1e-12]
    priority=dict(min(allowed,key=key),feasible=True) if allowed else dict(keep,feasible=False)
    return dict(balanced=balanced,native_priority=priority,
                natural=dict(calibration_row(d,geometry,score,0.),feasible=True)),table

def main():
    check_host();out=ROOT/'training';out.mkdir(exist_ok=False)
    for s in (24,37):verify_seal(ROOT/'evidence'/f'scan{s}')
    sources={str(ROOT/n):sha(ROOT/n) for n in ('common.py','router.py','train.py','infer.py','PROTOCOL.md')}
    for path in (BASE/'data/SEALED.json',CROSS/'data/SEALED.json',RESERVED/'training/SEALED.json',
                 CROSS/'training/SEALED.json',ROOT/'evidence/scan24/SEALED.json',ROOT/'evidence/scan37/SEALED.json'):
        sources[str(path)]=sha(path)
    save_json(out/'PRE_FIT_LOCK.json',dict(sources=sources,params=PARAMS,primary='joint_support__balanced',
              dev_scenes=[24,37],fit_support_rows=78598,replay_labels=False))
    d=development_metadata();n=len(d['e0']);x=np.empty((n,2,128),np.float32);geo=np.empty((n,3,3))
    for rec in d['records']:
        ids=d['case_id']==rec['case_id']
        features,points=load_observations(OLD/'real_results'/rec['case'],d['row_id'][ids],True)
        x[ids]=features;geo[ids]=points
    assert np.isfinite(x).all() and np.isfinite(geo).all()
    contexts={'joint_common':contextual_features(x[:,:,:96],geo),
              'joint_support':contextual_features(x,geo)}
    gains=d['losses'][:,:1]-d['losses'][:,1:]
    oof={k:np.full((n,2),np.nan) for k in METHODS}
    with np.load(RESERVED/'training/OOF_SCORES.npz') as z:oof['normalized_max'][:,0]=z['reserved_aug']
    with np.load(CROSS/'training/OOF_SCORES.npz') as z:oof['normalized_max'][:,1]=z['B_R']
    oof['support_margin']=support_scores(x)
    start=time.monotonic();timings=[]
    def fit(train_ids):
        w=case_weights(d['case_id'][train_ids],d['e0'][train_ids])
        models={'independent_absolute':fit_independent(x[train_ids,:,:96],gains[train_ids],w)}
        models.update({k:fit_shared(v[train_ids],gains[train_ids],w) for k,v in contexts.items()})
        return models
    for scene in (24,37):
        tick=time.monotonic();train_ids=(d['scene']==scene)&d['calibration_support'];test=d['scene']!=scene
        models=fit(train_ids)
        oof['independent_absolute'][test]=np.column_stack([m.predict(x[test,c,:96])
                                                          for c,m in enumerate(models['independent_absolute'])])
        for k,v in contexts.items():oof[k][test]=predict_shared(models[k],v[test])
        timings.append(dict(train_scene=scene,seconds=time.monotonic()-tick));print('OOF',scene,flush=True)
    thresholds={};table=[]
    for k,score in oof.items():
        choices,rows=calibrate(d,geo,score);thresholds[k]=choices
        table += [dict(method=k,**r) for r in rows]
        print('CALIBRATION',k,choices['balanced']['objective_relative_MSE'],choices['native_priority']['feasible'],flush=True)
    save_npz(out/'OOF_SCORES.npz',**oof)
    save_json(out/'THRESHOLDS.json',thresholds)
    with (out/'CALIBRATION.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    models=fit(d['calibration_support']);joblib.dump(models,out/'models.joblib')
    assert all(sha(p)==h for p,h in sources.items())
    save_json(out/'MODEL_LOCK.json',dict(sources=sources,model_sha=sha(out/'models.joblib'),
        thresholds_sha=sha(out/'THRESHOLDS.json'),dimensions=dict(independent_absolute=96,joint_common=196,joint_support=260),
        dev_rows=n,fit_rows=int(d['calibration_support'].sum()),timings=timings,
        seconds=time.monotonic()-start,labels_unit='squared mm',same_raw_gain_scale=True))
    seal(out,sources);print('TRAINING SEALED',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
