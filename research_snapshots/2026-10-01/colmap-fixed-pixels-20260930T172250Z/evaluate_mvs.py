"""After full prediction seal: fixed pixels against independent DTU reference."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import csv,json
import numpy as np
from scipy.spatial import cKDTree
from run_colmap import ROOT,OLD,BASE,sha,load,dump,npz
sys.path.insert(0,str(BASE))
import evaluate as prior

def metrics(d,mask):
    x=d[mask];good=x<=1
    return dict(n=int(len(x)),mse_mm2=float(np.mean(x*x)) if len(x) else None,
        mae_mm=float(np.mean(x)) if len(x) else None,median_mm=float(np.median(x)) if len(x) else None,
        correct_1mm=int(good.sum()),wrong_5mm=int((x>5).sum()))

def main():
    seal=load(ROOT/'PREDICTIONS_SEALED.json')
    for p,h in seal['files'].items():assert sha(ROOT/p)==h,p
    lock=load(ROOT/'RUN_LOCK.json')
    for p,h in lock['files'].items():assert sha(p)==h,p
    # Verify old immutable candidate/prediction seals before reading references.
    for p,h in load(OLD/'PREDICTIONS_SEALED.json')['files'].items():assert sha(OLD/p)==h,p
    references=prior.reference_metadata();trees={}
    for sid,meta in references.items():
        assert sha(meta['path'])==meta['sha256']
        trees[sid]=cKDTree(prior.read_ply_xyz(meta['path'],np))
    plan=load(OLD/'PLAN_CORRECTED.json');rows=[];pointrows=[];tail=[];repeat={};details={}
    for sid,s in plan['scenes'].items():
        tree=trees[int(sid)]
        for roi in s['rois']:
            name=roi['id'];a=npz(OLD/'initialization'/f'scan{sid}'/name/'input.npz')
            new=npz(ROOT/'predictions'/f'{name}.npz');uv=a['fixed_query_pixel_xy']
            assert np.array_equal(uv,new['pixel_xy'])
            q=a['query_old_ids'];keep=np.full((128,3),np.nan);keep[q]=a['points_mm'][a['query_ids']]
            valid=np.isfinite(keep).all(1);kd=np.full(128,np.nan);kd[valid]=tree.query(keep[valid],workers=1)[0]
            cand=npz(OLD/'predictions/U1'/(name+'__native')/'candidates/CANDIDATES.npz')
            assert np.array_equal(cand['query_old_ids'],q) and np.array_equal(cand['geometry_mm'][:,0],keep[q])
            cd=tree.query(cand['geometry_mm'].reshape(-1,3),workers=1)[0].reshape(len(q),3)
            oracle=np.full(128,np.nan);oracle[q]=np.min(cd,1)
            distances={'CPU':kd};valids={'CPU':valid}
            for arm in ['photo','geo','geo_unfiltered']:
                xyz=new[arm+'_xyz_mm'];v=new[arm+'_valid'];assert np.array_equal(v,np.isfinite(xyz).all(1))
                d=np.full(128,np.nan);d[v]=tree.query(xyz[v],workers=1)[0]
                distances[arm]=d;valids[arm]=v
                fb=kd.copy();fb[v&valid]=d[v&valid]
                distances[arm+'_fallback']=fb;valids[arm+'_fallback']=valid
                enriched=oracle.copy();enriched[v&valid]=np.minimum(enriched[v&valid],d[v&valid])
                distances[arm+'_oracle']=enriched;valids[arm+'_oracle']=valid
                hard=valid&(oracle>5);oldgood=valid&(kd<=1)
                tail.append(dict(scene=int(sid),roi=name,method=arm,old_hard=int(hard.sum()),
                    hard_supported=int((hard&v).sum()),hard_better=int((hard&v&(d<oracle)).sum()),
                    hard_rescued_5mm=int((hard&v&(d<=5)).sum()),hard_rescued_1mm=int((hard&v&(d<=1)).sum()),
                    old_good_1mm=int(oldgood.sum()),old_good_moved_beyond_1mm=int((oldgood&v&(d>1)).sum()),
                    improved=int((valid&v&(d<kd-1e-9)).sum()),worsened=int((valid&v&(d>kd+1e-9)).sum()),
                    unchanged_or_missing=int((valid&(~v|np.isclose(d,kd,rtol=0,atol=1e-9))).sum()),
                    newly_supported_15=int((~valid&v).sum()),newly_supported_correct_1mm=int((~valid&v&(d<=1)).sum())))
            combined=oracle.copy()
            for arm in ['photo','geo']:
                v=valids[arm]&valid;combined[v]=np.minimum(combined[v],distances[arm][v])
            distances['old_pool_oracle']=oracle;valids['old_pool_oracle']=valid
            distances['enriched_pool_oracle']=combined;valids['enriched_pool_oracle']=valid
            for method,d in distances.items():
                v=valids[method]
                for scope,mask in [('all_valid',v),('cpu_valid_common',v&valid)]:
                    rec=dict(scene=int(sid),roi=name,method=method,scope=scope,requested=128,**metrics(d,mask))
                    rec['coverage']=float(v.mean());rec['correct_per_requested']=int((v&(d<=1)).sum())/128
                    rec['paired_cpu_mse_mm2']=metrics(kd,mask&valid)['mse_mm2']
                    rows.append(rec)
                for i in range(128):
                    pointrows.append(dict(scene=int(sid),roi=name,query_old_id=i,u=uv[i,0],v=uv[i,1],
                        method=method,valid=bool(v[i]),distance_mm=float(d[i]) if v[i] else '',
                        cpu_valid=bool(valid[i]),cpu_distance_mm=float(kd[i]) if valid[i] else '',
                        old_pool_min_mm=float(oracle[i]) if valid[i] else ''))
            if sid=='24':
                repeat[name]=dict(same_valid=bool(np.array_equal(new['photo_valid'],new['photo_repeat_valid'])),
                    same_depth=bool(np.array_equal(new['photo_depth_mm'],new['photo_repeat_depth_mm'])),
                    max_depth_difference_mm=float(np.max(np.abs(new['photo_depth_mm']-new['photo_repeat_depth_mm']))))
            details[name]={m:dict(distance_mm=[float(x) if np.isfinite(x) else None for x in d],
                valid=valids[m].tolist()) for m,d in distances.items()}
    summary=[]
    for method in sorted(set(r['method'] for r in rows)):
        rr=[r for r in rows if r['method']==method and r['scope']=='all_valid']
        common=[r for r in rows if r['method']==method and r['scope']=='cpu_valid_common']
        summary.append(dict(method=method,roi_equal_mse_mm2=float(np.mean([r['mse_mm2'] for r in rr])) if all(r['n'] for r in rr) else None,
            coverage_n=sum(r['n'] for r in rr),requested=512,correct_1mm=sum(r['correct_1mm'] for r in rr),
            wrong_5mm=sum(r['wrong_5mm'] for r in rr),
            common_n=sum(r['n'] for r in common),common_roi_equal_mse_mm2=float(np.mean([r['mse_mm2'] for r in common])) if all(r['n'] for r in common) else None,
            common_cpu_roi_equal_mse_mm2=float(np.mean([r['paired_cpu_mse_mm2'] for r in common])) if all(r['n'] for r in common) else None))
    dest=ROOT/'evaluation';dest.mkdir(exist_ok=False)
    for filename,data in [('ROI_METRICS.csv',rows),('POINT_METRICS.csv',pointrows),('TAIL_METRICS.csv',tail)]:
        with (dest/filename).open('x') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    dump(dest/'RESULTS.json',dict(summary=summary,tail=tail,repeat=repeat,references=references,
        reference_type='dataset-provided DTU structured-light point samples; not official full-scene DTU benchmark',
        evaluation_type='real_gt, old two-scene development replay',fixed_query_count=512,baseline_valid=497,
        deployment_changed=False,gt_used_for_prediction=False))
    dump(dest/'POINT_DISTANCES.json',details)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
