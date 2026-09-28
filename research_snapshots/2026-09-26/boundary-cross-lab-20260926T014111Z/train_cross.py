"""Train B_F/B_R with candidate-correct labels; A models remain byte-frozen."""
from common import *
import csv,time,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from train_experiment import calibrate,case_weights

PARAMS=dict(loss='squared_error',learning_rate=.08,max_iter=80,max_leaf_nodes=7,
    max_depth=3,min_samples_leaf=80,l2_regularization=1.,random_state=SEED,early_stopping=False)

def fit(X,d,ids):
    m=HistGradientBoostingRegressor(**PARAMS)
    m.fit(X[ids],d['gain'][ids]/(d['e0'][ids]+d['e1'][ids]+.01),
          sample_weight=case_weights(d['case_id'][ids]))
    return m

def main():
    check_host();verify_seal(ROOT/'data');verify_seal(PREV/'training')
    for s in (24,37):verify_seal(ROOT/'evidence'/f'scan{s}')
    out=ROOT/'training';out.mkdir(exist_ok=False)
    files=[ROOT/n for n in ('common.py','PROTOCOL.md','train_cross.py','prepare_data.py','extract_b.py','data/SEALED.json')]
    files += [BASE/'train_experiment.py',PREV/'paired_features.py',PREV/'training/SEALED.json',
              ROOT/'evidence/scan24/SEALED.json',ROOT/'evidence/scan37/SEALED.json']
    sources={str(p):sha(p) for p in files}
    save_json(out/'PRE_FIT_LOCK.json',dict(source_sha256=sources,params=PARAMS,primary='B_R__balanced',
        training_scenes=[24,37],reference_access='development supervision only',replay_labels_used=False))
    with np.load(ROOT/'data/train.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    meta=json.loads((BASE/'data/TRAIN_MANIFEST.json').read_text())['records']
    F,R=np.zeros((len(d['X']),32),np.float32),np.zeros((len(d['X']),32),np.float32)
    for rec in meta:
        ids=d['case_id']==rec['case_id']
        with np.load(ROOT/'evidence'/f"scan{rec['scene']}"/rec['case']/'PAIRED.npz') as z:
            if not np.array_equal(z['row_ids'],d['row_id'][ids]):raise AssertionError('row mismatch')
            F[ids],R[ids]=z['F'],z['R']
    Xs={'B_F':np.column_stack([d['X'],F]),'B_R':np.column_stack([d['X'],R])}
    save_npz(out/'NEW_FEATURES.npz',F=F,R=R)
    oof={k:np.full(len(F),np.nan) for k in Xs};timings=[];start=time.monotonic()
    for s,t in ((24,37),(37,24)):
        for k,X in Xs.items():
            tic=time.monotonic();m=fit(X,d,d['scene']==s);idx=d['scene']==t;oof[k][idx]=m.predict(X[idx])
            timings.append(dict(model=k,train_scene=s,seconds=time.monotonic()-tic));print('FOLD',k,s,t,flush=True)
    save_npz(out/'OOF_SCORES.npz',**oof)
    thresholds={};table=[]
    for k,scores in oof.items():
        choices,rows=calibrate(d,scores,0.);thresholds[k]=choices;table += [dict(method=k,**r) for r in rows]
    old=json.loads((PREV/'training/THRESHOLDS.json').read_text())
    thresholds.update(A_F=old['fit_aug'],A_R=old['reserved_aug'])
    save_json(out/'THRESHOLDS.json',thresholds)
    with (out/'THRESHOLD_GRID.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    models={}
    for k,X in Xs.items():
        tic=time.monotonic();models[k]=fit(X,d,np.ones(len(X),bool))
        timings.append(dict(model=k,train_scene='24+37',seconds=time.monotonic()-tic))
    joblib.dump(models,out/'models.joblib')
    if any(sha(p)!=h for p,h in sources.items()):raise AssertionError('locked source changed')
    save_json(out/'MODEL_LOCK.json',dict(source_sha256=sources,models_sha256=sha(out/'models.joblib'),
        thresholds_sha256=sha(out/'THRESHOLDS.json'),timings=timings,wall_seconds=time.monotonic()-start,
        feature_dims={k:X.shape[1] for k,X in Xs.items()},oof_is_tuning_not_test=True,
        A_models_unchanged=str(PREV/'training/models.joblib')))
    seal(out,sources);print('B MODELS SEALED',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
