"""Independent development calibration and all replay decisions/exports audit."""
from common import *
import joblib,time
from threadpoolctl import threadpool_limits


def stats(d,scores,threshold=None,action='threshold'):
    chosen=np.ones(len(scores),bool) if action=='all' else np.zeros(len(scores),bool) if action=='keep' else scores>threshold
    relative=[];edits=[];native=[];native0=[];injected=[]
    for cid in np.unique(d['case_id']):
        use=(d['case_id']==cid)&d['calibration_support']
        before=d['e0'][use];after=np.where(chosen[use],d['e1'][use],before)
        ratio=after.mean()/max(before.mean(),1e-6)
        relative.append(ratio);edits.append(chosen[use].mean())
        if d['condition'][use][0]=='native':native.append(after.mean());native0.append(before.mean())
        else:injected.append(ratio)
    return dict(threshold=threshold if action=='threshold' else None,action=action,
        objective_relative_MSE=float(np.mean(relative)),accepted_fraction=float(np.mean(edits)),
        native_MSE=float(np.mean(native)),native_identity_MSE=float(np.mean(native0)),
        injected_relative_MSE=float(np.mean(injected)))


def compare_stats(a,b):
    for key,value in a.items():
        if isinstance(value,float):assert abs(value-b[key])<=1e-12,(key,value,b[key])
        else:assert value==b[key],(key,value,b[key])


def main():
    check_host();start=time.monotonic()
    for folder in (ROOT/'data',ROOT/'training',PREV/'training'):verify_seal(folder)
    assert json.loads((ROOT/'AUDIT_DATA.json').read_text())['status']=='PASS'
    with np.load(ROOT/'data/train.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(ROOT/'training/NEW_FEATURES.npz',allow_pickle=False) as z:F,R=z['F'],z['R']
    manifest=json.loads((BASE/'data/TRAIN_MANIFEST.json').read_text())['records']
    for rec in manifest:
        use=d['case_id']==rec['case_id']
        with np.load(ROOT/'evidence'/f"scan{rec['scene']}"/rec['case']/'PAIRED.npz',allow_pickle=False) as z:
            assert np.array_equal(d['row_id'][use],z['row_ids'])
            assert np.array_equal(F[use],z['F']) and np.array_equal(R[use],z['R'])
    models=joblib.load(ROOT/'training/models.joblib')
    thresholds=json.loads((ROOT/'training/THRESHOLDS.json').read_text())
    previous=json.loads((PREV/'training/THRESHOLDS.json').read_text())
    assert thresholds['A_F']==previous['fit_aug'] and thresholds['A_R']==previous['reserved_aug']
    assert set(models)=={'B_F','B_R'} and set(thresholds)=={'A_F','A_R','B_F','B_R'}
    lock=json.loads((ROOT/'training/MODEL_LOCK.json').read_text())
    assert sha(ROOT/'training/models.joblib')==lock['models_sha256']
    assert sha(ROOT/'training/THRESHOLDS.json')==lock['thresholds_sha256']
    assert all(sha(Path(p))==h for p,h in lock['source_sha256'].items())
    with np.load(ROOT/'training/OOF_SCORES.npz',allow_pickle=False) as z:oof={k:z[k] for k in z.files}
    calibration=[]
    for name in ('B_F','B_R'):
        assert models[name].n_features_in_==96
        params=dict(loss='squared_error',learning_rate=.08,max_iter=80,max_leaf_nodes=7,
            max_depth=3,min_samples_leaf=80,l2_regularization=1.,random_state=SEED,early_stopping=False)
        assert all(models[name].get_params()[k]==v for k,v in params.items())
        score=oof[name];assert score.shape==(len(d['X']),) and np.isfinite(score).all()
        grid=sorted(set(np.quantile(score[d['calibration_support']],np.linspace(0,1,101)).tolist()+[0.]))
        rows=[stats(d,score,t) for t in grid]+[stats(d,score,action=a) for a in ('all','keep')]
        key=lambda r:(r['objective_relative_MSE'],r['accepted_fraction'],0 if r['action']!='threshold' else 1,
                      -r['threshold'] if r['threshold'] is not None else 0)
        compare_stats(min(rows,key=key),thresholds[name]['balanced'])
        compare_stats(stats(d,score,0.),thresholds[name]['natural'])
        feasible=[r for r in rows if r['native_MSE']<=r['native_identity_MSE']+1e-12 and r['injected_relative_MSE']<=.95+1e-12]
        chosen=thresholds[name]['native_priority']
        if feasible:
            assert chosen['feasible'] is True
            compare_stats(min(feasible,key=key),chosen)
        else:assert chosen['feasible'] is False and chosen['action']=='keep' and chosen['keep_all'] is True
        calibration.append(dict(method=name,dimensions=96,grid_size=len(rows),native_priority_feasible=bool(feasible)))
    records=[]
    for sid in SCENES:
        verify_seal(ROOT/'inference'/f'scan{sid}')
        for path in cases_for_scene(sid):
            target=ROOT/'inference'/f'scan{sid}'/path.name
            x=cached_features(path,'B')
            with np.load(ROOT/'evidence'/f'scan{sid}'/path.name/'PAIRED.npz',allow_pickle=False) as z:
                f,r=z['F'],z['R'];assert np.array_equal(z['row_ids'],np.arange(len(x)))
            with np.load(target/'scores.npz',allow_pickle=False) as z:scores={k:z[k] for k in z.files}
            with np.load(target/'decisions.npz',allow_pickle=False) as z:masks={k:z[k] for k in z.files}
            assert len(masks)==21 and set(scores)=={'B_F','B_R'}
            assert all(v.dtype==bool and v.shape==(len(x),) for v in masks.values())
            for name,pair in (('B_F',f),('B_R',r)):
                assert np.array_equal(models[name].predict(np.column_stack([x,pair])),scores[name])
                for setting,spec in thresholds[name].items():
                    selected=np.ones(len(x),bool) if spec['action']=='all' else np.zeros(len(x),bool) if spec['action']=='keep' else scores[name]>spec['threshold']
                    assert np.array_equal(selected,masks[name+'__'+setting])
                assert np.array_equal(masks[name+'_pair'],(pair[:,12:16].sum(1)>=2)&(pair[:,24]>0))
            with np.load(PREV/'inference'/f'scan{sid}'/path.name/'decisions.npz',allow_pickle=False) as z:
                for name in ('identity','A_all','frozen_gain','previous_normalized'):assert np.array_equal(masks[name],z[name])
                for name,previous_name in (('A_F','fit_aug'),('A_R','reserved_aug')):
                    for setting in ('balanced','natural','native_priority'):
                        assert np.array_equal(masks[name+'__'+setting],z[previous_name+'__'+setting])
                assert np.array_equal(masks['A_F_pair'],z['fit_pair_margin'])
                assert np.array_equal(masks['A_R_pair'],z['reserved_pair_margin'])
            assert not masks['identity'].any() and masks['A_all'].all() and masks['B_all'].all()
            p,b=read_points(path/'identity.ply'),read_points(path/'B_all.ply')
            files=sorted(target.glob('*.ply'))
            assert {f.stem for f in files}=={'B_F__balanced','B_R__balanced','B_R__native_priority'}
            for file in files:assert np.array_equal(read_points(file),np.where(masks[file.stem][:,None],b,p))
            records.append(dict(case=path.name,rows=len(x),masks=len(masks),exact_B_exports=len(files)))
        print('AUDIT B SELECTION',sid,flush=True)
    assert len(records)==60
    save_json(ROOT/'AUDIT_SELECTION.json',dict(status='PASS',calibration=calibration,replay_cases=records,
        case_count=len(records),exact_B_exports=sum(r['exact_B_exports'] for r in records),
        A_thresholds_and_decisions_reused_exactly=True,candidate_B_models_use_B64=True,
        replay_labels_read=False,reference_geometry_read=False,
        source_sha256={str(ROOT/'audit_selection.py'):sha(ROOT/'audit_selection.py')},wall_seconds=time.monotonic()-start))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
