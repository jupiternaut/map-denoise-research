"""Post-hoc diagnostic: available K1/K2 coordinates, not the broader POOL.

Written after the main 720-row evaluation was sealed. No model or selection
rule is altered. Ground truth is used only to measure candidate-set capacity.
"""
from common import *
import csv, time
from scipy.io import loadmat
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


def main():
    check_host()
    verify_seal(OUT/'evaluation')
    for sid in SCENES:
        verify_seal(OUT/'inference'/f'scan{sid}')
    dest=OUT/'posthoc_field_capacity'
    dest.mkdir(exist_ok=False)
    sources={str(ROOT/'posthoc_field_capacity.py'):sha(ROOT/'posthoc_field_capacity.py'),
             str(OUT/'evaluation/SEALED.json'):sha(OUT/'evaluation/SEALED.json')}
    with (OUT/'evaluation/METRICS.csv').open() as f:
        old={(int(r['scene']),r['roi'],r['condition'],r['arm']):r for r in csv.DictReader(f)}
    manifest_path=DATA/'closeout-confirmation-v1/REFERENCE_MANIFEST.json'
    sources[str(manifest_path)]=sha(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    rows=[];checks=[];start=time.monotonic()
    for sid in SCENES:
        raw=CLOSEOUT/'confirmation'/f'scan{sid}'
        frozen_file(raw/'ROIS.json',raw,sources)
        rois=json.loads((raw/'ROIS.json').read_text())
        folder=DATA/'closeout-confirmation-v1/evaluation_only'/f'scan{sid}'
        for rec in manifest['records']:
            path=Path(rec['path'])
            if path.parent==folder:
                if sha(path)!=rec['sha256']:raise AssertionError('reference changed')
                sources[str(path)]=rec['sha256']
        laser=read_points(folder/f'stl{sid:03d}_total.ply')
        obs=loadmat(folder/f'ObsMask{sid}_10.mat')
        for roi in rois:
            rid=roi['id'];npath=raw/(rid+'__native')/'identity.ply'
            frozen_file(npath,raw,sources);native=read_points(npath)
            support=io.in_box(native,roi['lo'],roi['hi'])&io.observed(native,obs)
            spath=CLOSEOUT/'evaluation'/(rid+'_native_support.npy')
            frozen_file(spath,CLOSEOUT/'evaluation',sources)
            if not support.any() or not np.array_equal(support,np.load(spath,allow_pickle=False)):
                raise AssertionError('native-row support changed')
            reference=io.voxel(laser[io.in_box(laser,roi['lo'],roi['hi'])&io.observed(laser,obs)])
            tree=cKDTree(reference)
            for cond in CONDS:
                case=rid+'__'+cond;inf=OUT/'inference'/f'scan{sid}'/case
                path=inf/'POOL.npz';frozen_file(path,inf.parent,sources)
                with np.load(path,allow_pickle=False) as z:
                    pool=z['points'];names=z['candidate_names'].tolist()
                idx={name:names.index(name) for name in ('keep','field_K1','field_K2a','field_K2b')}
                pp=pool[support]
                distances={name:tree.query(pp[:,j],workers=1)[0] for name,j in idx.items()}
                baseline=float(np.mean(distances['keep']**2))
                recorded=float(old[(sid,rid,cond,'identity')]['source_MSE_mm2'])
                if abs(baseline-recorded)>1e-12:raise AssertionError('identity MSE changed')
                for arm,fields,actual in (
                    ('oracle_K1_only',('keep','field_K1'),'single_field'),
                    ('oracle_K2_only',('keep','field_K2a','field_K2b'),'multi_field')):
                    ds=np.column_stack([distances[n] for n in fields])
                    choice=ds.argmin(axis=1);best=ds[np.arange(len(ds)),choice]
                    mse=float(np.mean(best**2));mae=float(best.mean())
                    actual_mse=float(old[(sid,rid,cond,actual)]['source_MSE_mm2'])
                    broad=float(old[(sid,rid,cond,'oracle_all_layers')]['source_MSE_mm2'])
                    if mse>actual_mse+1e-10 or broad>mse+1e-10:
                        raise AssertionError('candidate inclusion ordering failed')
                    # Verify the actual method truly emits only this candidate set.
                    outpath=inf/(actual+'.ply');frozen_file(outpath,inf.parent,sources)
                    out=read_points(outpath)[support]
                    error=np.min(np.stack([np.sum((out-pp[:,idx[n]])**2,axis=1) for n in fields]),axis=0)
                    maxerr=float(np.sqrt(error.max()))
                    if maxerr>1e-10:raise AssertionError('actual arm not in declared candidate set')
                    checks.append(dict(case=case,arm=arm,max_candidate_coordinate_error_mm=maxerr))
                    rows.append(dict(scene=sid,roi=rid,condition=cond,arm=arm,
                        n_source=int(support.sum()),n_reference=len(reference),
                        source_MSE_mm2=mse,source_MAE_mm=mae,identity_MSE_mm2=baseline,
                        corresponding_actual=actual,actual_MSE_mm2=actual_mse,
                        selection_gap_mm2=actual_mse-mse,
                        restricted_pool_gap_mm2=mse-broad,
                        keep_fraction=float(np.mean(choice==0)),
                        choice_counts=json.dumps(dict(zip(fields,np.bincount(choice,minlength=len(fields)).tolist())))))
                print('FIELD CAPACITY',case,flush=True)
    summary={}
    for condition in CONDS:
        summary[condition]={}
        for arm in ('oracle_K1_only','oracle_K2_only'):
            rr=[r for r in rows if r['condition']==condition and r['arm']==arm]
            if len(rr)!=12:raise AssertionError('incomplete condition')
            keys=('source_MSE_mm2','source_MAE_mm','identity_MSE_mm2','actual_MSE_mm2',
                  'selection_gap_mm2','restricted_pool_gap_mm2','keep_fraction')
            cell={k:float(np.mean([np.mean([r[k] for r in rr if r['scene']==s]) for s in SCENES])) for k in keys}
            cell['MSE_gain_percent']=100*(1-cell['source_MSE_mm2']/cell['identity_MSE_mm2'])
            summary[condition][arm]=cell
    with (dest/'METRICS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(dest/'SUMMARY.json',dict(status='PASS',post_hoc=True,registered_primary_unchanged=True,
        reason='Separate restricted field-coordinate capacity from selecting within the field family.',
        scope='Only KEEP/K1 or KEEP/K2a/K2b; pointwise evaluation-reference selector, not deployable.',
        reference_preprocessing='Same native-row fixed support; ROI/ObsMask laser; 0.8mm voxel.',
        aggregation='ROI equally then scene equally',cases=60,rows=len(rows),
        by_condition=summary,checks=checks,wall_seconds=time.monotonic()-start))
    if any(sha(p)!=h for p,h in sources.items()):raise AssertionError('source changed')
    seal(dest,sources)
    print('POSTHOC FIELD CAPACITY SEALED',len(rows),'rows',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
