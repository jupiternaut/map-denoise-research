"""Post-evaluation evidence replay only. Never changes a method or decision."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np
from run import ROOT,verify_lock,load,sha,dump,save_npz,unpack_aux,now
from baseline import cameras
from predictor import predict_pixels
from fixtures import render_world

def main():
    verify_lock();out=ROOT/'diagnostics/pixels';params=load(ROOT/'METHOD.json')
    grid=np.arange(300.,1001.,2.);cand=np.asarray(params['candidates'])
    ix=[int(np.argmin(abs(grid-z))) for z in cand]
    truth={t['id']:t for t in load(ROOT/'data/truth/metadata.json')}
    inputs=load(ROOT/'data/observed/inputs.json');rec=[];hashes={};maximum=0.;rois={}
    for name in ('ordinary_stage','oracle_stage'):
        for p,h in load(ROOT/name/'SEAL.json')['files'].items():assert sha(p)==h
        records=load(ROOT/name/'OBSERVATIONS.json')['rows']
        by_id=defaultdict(list)
        for r in records:
            if r['arm']!='N':by_id[r['id']].append(r)
        for inp in inputs:
            fid=inp['id'];cam=cameras(inp['cameras']);t=truth[fid]
            with np.load(inp['image_file']) as f:images=f['images']
            reproduced,diag=render_world(t['producer_parameters'],cam,return_diagnostics=True)
            np.testing.assert_array_equal(reproduced,images)
            for r in by_id[fid]:
                dynamic=r['arm'].endswith('D');inc=600 if dynamic else r['incumbent']
                with np.load(r['curve_file']) as f:sealed_fold=f['fold_loss'][:,ix]
                for fold,(_,ev) in enumerate(params['folds']):
                    aux=unpack_aux(ROOT/name/f'{fid}_aux{fold}.npz')
                    # Full grid is essential: ROI is defined from its complete
                    # projection union, never from the smaller exported subset.
                    p=predict_pixels(aux,cam[0],cam[ev],grid,inc,params,'dynamic' if dynamic else 'fixed')
                    roi=p['roi'];k=(fid,fold)
                    if k in rois:np.testing.assert_array_equal(rois[k],roi)
                    else:rois[k]=roi.copy()
                    observed=images[ev,roi[:,1],roi[:,0]];prediction=p['prediction'][ix]
                    residual=prediction-observed[None];loss=np.mean(residual**2,axis=1)
                    valid=bool(np.all(p['depth_valid']))
                    if valid:
                        delta=float(np.max(abs(loss-sealed_fold[fold])))
                        maximum=max(maximum,delta);assert delta<1e-9
                    alpha=diag['source_alpha'][ev,roi[:,1],roi[:,0]]
                    path=out/f'{fid}_{r["arm"]}_{int(inc)}_fold{fold}.npz'
                    hashes[str(path)]=save_npz(path,candidates=cand,roi=roi,prediction=prediction,
                        observed=observed,residual=residual,true_alpha_evaluation_only=alpha,
                        predicted_alpha=p['alpha'][ix],valid=np.asarray(valid))
                    for region,mask in [('foreground',alpha==1),('background',alpha==0),('boundary',(alpha>0)&(alpha<1))]:
                        rec.append(dict(id=fid,arm=r['arm'],incumbent=inc,fold=fold,region=region,
                            pixel_count=int(mask.sum()),mse=np.mean(residual[:,mask]**2,axis=1).tolist() if mask.any() else None,
                            valid=valid))
    rows=load(ROOT/'evaluation/ROWS.json')
    pairwise=[];reversals=[]
    for r in rows:
        rank=r.get('ranking')
        if rank and rank.get('valid'):
            scores=np.asarray(rank['candidate_losses']);errors=abs(cand-r['true_depth'])
            agree=disagree=tied=0
            for i in range(len(cand)):
                for j in range(i+1,len(cand)):
                    if abs(errors[i]-errors[j])<1e-9:continue
                    d=(scores[i]-scores[j])*(errors[i]-errors[j])
                    if abs(scores[i]-scores[j])<1e-9:tied+=1
                    elif d>0:agree+=1
                    else:disagree+=1
            pairwise.append(dict(id=r['id'],arm=r['arm'],initial_depth=r['initial_depth'],
                geometry_order_agreements=agree,disagreements=disagree,ties=tied,
                final_regret_mm=r['error']-float(errors.min()),
                argmin_regret_mm=rank['diagnostic_argmin_error_min']-float(errors.min())))
            if r['arm']=='ED' and r['initial_depth']==600 and r['error']>0:
                reversals.append({k:r[k] for k in ('id','mechanism','intervals','support_mean','selected_depth','ranking','error')})
    dump(out/'REGIONS.json',rec);dump(out/'PAIRWISE_REGRET.json',pairwise)
    dump(out/'SUMMARY.json',dict(created=now(),source_sha256=sha(__file__),n_pixel_artifacts=len(hashes),
        maximum_replay_loss_difference=maximum,all_arms_share_roi=True,producer_replay_exact=True,
        scope='post-evaluation diagnostics; no method or P change; source alpha is evaluation only',
        correct_incumbent_P_reversals=reversals))
    hashes.update({str(p):sha(p) for p in out.glob('*.json')})
    dump(out/'SEAL.json',dict(files=hashes))
    print(__import__('json').dumps(dict(n_pixel_artifacts=len(hashes),max_loss_difference=maximum,reversals=reversals)))
if __name__=='__main__':main()
