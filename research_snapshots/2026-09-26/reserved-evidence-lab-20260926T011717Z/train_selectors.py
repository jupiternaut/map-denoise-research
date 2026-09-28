"""Same normalized target/capacity: cached, fitted-view, reserved-view, both."""
from common import *
import csv,time,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from train_experiment import calibrate,case_weights

METHODS=('cached64','fit_aug','reserved_aug','both_aug')
PARAMS=dict(loss='squared_error',learning_rate=.08,max_iter=80,max_leaf_nodes=7,
            max_depth=3,min_samples_leaf=80,l2_regularization=1.,
            random_state=SEED,early_stopping=False)


def design(X,F,R):
    return dict(cached64=X,fit_aug=np.column_stack([X,F]),
                reserved_aug=np.column_stack([X,R]),both_aug=np.column_stack([X,F,R]))


def fit(X,data,mask):
    target=data['gain'][mask]/(data['e0'][mask]+data['e1'][mask]+.01)
    model=HistGradientBoostingRegressor(**PARAMS)
    model.fit(X[mask],target,sample_weight=case_weights(data['case_id'][mask]))
    return model


def main():
    check_host();verify_seal(PREV/'data');verify_seal(PREV/'training')
    for s in (24,37):verify_seal(ROOT/'evidence'/f'scan{s}')
    out=ROOT/'training';out.mkdir(exist_ok=False)
    paths=[ROOT/n for n in ('common.py','paired_features.py','extract_evidence.py','train_selectors.py','PROTOCOL.md')]
    paths += [PREV/'train_experiment.py',PREV/'common.py',PREV/'data/SEALED.json',
              ROOT/'evidence/scan24/SEALED.json',ROOT/'evidence/scan37/SEALED.json']
    sources={str(p):sha(p) for p in paths}
    save_json(out/'PRE_FIT_LOCK.json',dict(source_sha256=sources,params=PARAMS,methods=METHODS,
               development_scenes=[24,37],lead='reserved_aug',replay_labels_used=False))
    with np.load(PREV/'data/train.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    F,R=np.zeros((len(data['X']),32),np.float32),np.zeros((len(data['X']),32),np.float32)
    meta=json.loads((PREV/'data/TRAIN_MANIFEST.json').read_text())['records']
    for rec in meta:
        ids=data['case_id']==rec['case_id']
        with np.load(ROOT/'evidence'/f"scan{rec['scene']}"/rec['case']/'PAIRED.npz',allow_pickle=False) as z:
            if not np.array_equal(z['row_ids'],data['row_id'][ids]):raise AssertionError('training row mismatch')
            F[ids],R[ids]=z['F'],z['R']
    inputs=design(data['X'],F,R)
    save_npz(out/'NEW_FEATURES.npz',F=F,R=R)
    oof={m:np.full(len(F),np.nan) for m in METHODS};timings=[];start=time.monotonic()
    for train_scene,test_scene in ((24,37),(37,24)):
        train,test=data['scene']==train_scene,data['scene']==test_scene
        for name,X in inputs.items():
            tick=time.monotonic();model=fit(X,data,train);oof[name][test]=model.predict(X[test])
            timings.append(dict(method=name,train_scene=train_scene,seconds=time.monotonic()-tick))
            print('FOLD',name,train_scene,'->',test_scene,round(timings[-1]['seconds'],2),'s',flush=True)
    with np.load(PREV/'training/OOF_SCORES.npz',allow_pickle=False) as z:
        discrepancy=float(np.max(np.abs(oof['cached64']-z['normalized_gain'])))
    if discrepancy!=0:raise AssertionError(('cached matched model OOF differs',discrepancy))
    save_npz(out/'OOF_SCORES.npz',**oof)
    thresholds={};table=[]
    for name,scores in oof.items():
        choices,rows=calibrate(data,scores,0.)
        thresholds[name]=choices;table += [dict(method=name,**r) for r in rows]
        print('THRESHOLD',name,choices['balanced']['threshold'],choices['native_priority']['feasible'],flush=True)
    save_json(out/'THRESHOLDS.json',thresholds)
    with (out/'THRESHOLD_GRID.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    models={}
    for name,X in inputs.items():
        tick=time.monotonic();models[name]=fit(X,data,np.ones(len(X),bool))
        timings.append(dict(method=name,train_scene='24+37',seconds=time.monotonic()-tick))
    joblib.dump(models,out/'models.joblib')
    if any(sha(p)!=h for p,h in sources.items()):raise AssertionError('source changed while training')
    save_json(out/'MODEL_LOCK.json',dict(source_sha256=sources,models_sha256=sha(out/'models.joblib'),
        thresholds_sha256=sha(out/'THRESHOLDS.json'),feature_dims={k:v.shape[1] for k,v in inputs.items()},
        old_cached_OOF_max_difference=discrepancy,timings=timings,wall_seconds=time.monotonic()-start,
        oof_is_tuning_not_unbiased_test=True))
    seal(out,sources);print('TRAINING SEALED',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
