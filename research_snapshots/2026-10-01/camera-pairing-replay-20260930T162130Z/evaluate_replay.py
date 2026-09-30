"""Post-seal paired-UV evaluation; dataset references never affect predictions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import csv,json,hashlib,socket
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parent
OLD=Path('/srv/slam-research/grf/map-denoise/runs/upstream-photo-holdout-20260930T113213Z')
sys.path.insert(0,str(OLD))
import evaluate as prior
ARMS=('KEEP','restore','construct_witness','heldout_witness')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def read(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def dump(path,x):
    with path.open('x') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2)
def main():
    assert socket.gethostname()=='liekkas'
    seal=json.loads((ROOT/'PREDICTIONS_SEALED.json').read_text())
    for p,h in seal['files'].items():assert sha(ROOT/p)==h,p
    protocol=json.loads((ROOT/'EVALUATION_PROTOCOL_LOCK.json').read_text())
    for p,h in protocol['files'].items():assert sha(ROOT/p)==h,p
    plan=json.loads((ROOT/'PLAN_CORRECTED.json').read_text())
    dest=ROOT/'evaluation';dest.mkdir(exist_ok=False)
    # No GT coordinates read before complete prediction hash verification above.
    meta=prior.reference_metadata()
    trees={};references={}
    for sid,m in meta.items():
        assert sha(m['path'])==m['sha256']
        references[sid]=prior.read_ply_xyz(m['path'],np);trees[sid]=cKDTree(references[sid])
    bundles={};failures=[]
    for u in ('U0','U1'):
        for case in plan['cases']:
            d=ROOT/'predictions'/u/case['case_id']
            if (d/'INCOMPLETE.json').exists():failures.append(json.loads((d/'INCOMPLETE.json').read_text()));continue
            cand=read(d/'candidates/CANDIDATES.npz');inp=read(d/'INPUT.npz')
            b=dict(candidates=cand,inputs=inp,outputs={},routes={},distances={})
            g=cand['geometry_mm'];rr=cand['restore_routes'];n=len(g)
            assert np.array_equal(g[:,0],inp['points_mm'][inp['query_ids']])
            for w in ('W0','W1'):
                output=read(d/w/'points_outputs.npz');routes=read(d/w/'routes.npz');ws=read(d/w/'witness.npz')
                for arm in ARMS:
                    r=routes[arm]
                    assert np.array_equal(output[arm],g[np.arange(n),r]),(u,case['case_id'],w,arm)
                    if arm.endswith('witness'):assert np.logical_or(r==0,r==rr).all()
                for prefix in ('construct','heldout'):
                    sc=ws[prefix+'_scores'];valid=np.isfinite(sc).all(2)
                    margin=np.where(valid,sc[:,:,1]-sc[:,:,0],0)
                    accepted=(valid.sum(1)>=3)&(((margin>0)&valid).sum(1)>=3)&(margin.sum(1)>0)
                    assert np.array_equal(accepted,routes[prefix+'_accepted'])
                    assert (ws[prefix+'_shared_pixels'][valid]>=40).all()
                b['outputs'][w]=output;b['routes'][w]=routes
                b['distances'][w]={arm:trees[case['scene']].query(output[arm],workers=1)[0] for arm in ARMS}
            b['candidate_distances']=trees[case['scene']].query(g.reshape(-1,3),workers=1)[0].reshape(n,3)
            bundles[u,case['case_id']]=b
    # Native footprints: paired scope uses symmetric union across both U KEEP inputs.
    footprints={};query_manifest=[]
    for case in plan['cases']:
        if case['condition']!='native':continue
        roi=case['roi'];sid=case['scene'];key=case['case_id'];old=bundles['U0',key]
        new=bundles.get(('U1',key));common=new['candidates']['query_old_ids'] if new else np.array([],int)
        fixed=old['inputs']['reference_pixel_xy'][old['inputs']['query_ids']]
        for i,uv in enumerate(fixed):
            query_manifest.append(dict(roi=roi,query_old_id=i,pixel_xy=uv.tolist(),
                pixel_id=roi+':'+hashlib.sha256(np.ascontiguousarray(uv).tobytes()).hexdigest(),u1_valid=bool(i in common)))
        paired=[]
        for u,b in [('U0',old),('U1',new)]:
            if b is None:continue
            keep=b['candidates']['geometry_mm'][:,0]
            footprints[u,roi,'all_valid']=prior.fixed_keep_footprint(keep,references[sid],trees[sid],np)[0]
            if len(common):paired.append(keep[common] if u=='U0' else keep)
        fp=prior.fixed_keep_footprint(np.concatenate(paired),references[sid],trees[sid],np)[0] if paired else np.empty((0,3))
        for u in ('U0','U1'):footprints[u,roi,'paired_common']=fp
    dump(ROOT/'QUERY_MANIFEST.json',dict(requested=512,records=query_manifest))
    rows=[]
    for case in plan['cases']:
        new=bundles.get(('U1',case['case_id'])); common=new['candidates']['query_old_ids'] if new else np.array([],int)
        for u in ('U0','U1'):
            b=bundles.get((u,case['case_id']))
            if b is None:continue
            for scope in ('all_valid','paired_common'):
                ix=np.arange(len(b['candidates']['geometry_mm'])) if scope=='all_valid' or u=='U1' else common
                if scope=='paired_common' and not len(common):continue
                cd=b['candidate_distances'][ix]; oracle=np.min(cd**2,axis=1)
                footprint=footprints[u,case['roi'],scope]
                for w in ('W0','W1'):
                    distances={arm:b['distances'][w][arm][ix] for arm in ARMS}
                    output={arm:b['outputs'][w][arm][ix] for arm in ARMS}
                    keep=output['KEEP'];keep_d=distances['KEEP'];metrics={};veto={}
                    original=b['routes'][w]['restore'][ix]
                    old_delta=keep_d-distances['restore']
                    for arm in ARMS:
                        prefix='construct' if arm=='construct_witness' else 'heldout' if arm=='heldout_witness' else None
                        valid=b['routes'][w][prefix+'_valid_views'][ix] if prefix else None
                        m=prior.arm_metrics(output[arm],keep,distances[arm],keep_d,footprint,cKDTree,np,valid)
                        m['selection_regret_mm2']=m['nearest_mse_mm2']-float(oracle.mean())
                        potential=float(np.mean(keep_d**2-oracle));actual=float(np.mean(keep_d**2-distances[arm]**2))
                        m['extracted_potential_fraction']=actual/potential if potential>0 else None
                        metrics[arm]=m
                        if prefix:
                            denied=(original!=0)&(b['routes'][w][arm][ix]==0)
                            bad=(original!=0)&(old_delta<-.1);good=(original!=0)&(old_delta>.1)
                            veto[arm]=dict(bad_original=int(bad.sum()),good_original=int(good.sum()),
                                bad_blocked=int((denied&bad).sum()),good_lost=int((denied&good).sum()),
                                bad_blocked_fraction=float((denied&bad).sum()/bad.sum()) if bad.any() else None,
                                good_lost_fraction=float((denied&good).sum()/good.sum()) if good.any() else None)
                    rows.append(dict(case_id=case['case_id'],scene=case['scene'],roi=case['roi'],condition=case['condition'],
                        upstream=u,witness=w,scope=scope,n=len(ix),requested=128,query_old_ids=b['candidates']['query_old_ids'][ix].tolist(),
                        footprint_n=len(footprint),oracle_mse_mm2=float(oracle.mean()),potential_mm2=float(np.mean(keep_d**2-oracle)),
                        arms=metrics,veto=veto))
    summary={};acceptance={}
    for scope in ('all_valid','paired_common'):
        summary[scope]={};acceptance[scope]={}
        for u in ('U0','U1'):
            for w in ('W0','W1'):
                cell=u+w;summary[scope][cell]={}
                for condition in ('native','minus3','plus3'):
                    select=[r for r in rows if r['scope']==scope and r['upstream']==u and r['witness']==w and r['condition']==condition]
                    if not select:continue
                    s=dict(roi_count=len(select),n=sum(r['n'] for r in select),
                        oracle_mse_mm2=float(np.mean([r['oracle_mse_mm2'] for r in select])),arms={})
                    for arm in ARMS:
                        s['arms'][arm]={k:float(np.mean([r['arms'][arm][k] for r in select])) if all(r['arms'][arm][k] is not None for r in select) else None for k in select[0]['arms'][arm]}
                    summary[scope][cell][condition]=s
                acceptance[scope][cell]={}
                if len(summary[scope][cell])<3:continue
                for arm in ('construct_witness','heldout_witness'):
                    native=summary[scope][cell]['native']['arms']; checks={}
                    for condition in ('minus3','plus3'):
                        m=summary[scope][cell][condition]['arms'];r=m['restore']['improvement_percent'];a=m[arm]['improvement_percent']
                        checks[condition]=dict(restore_gain_percent=r,gain_percent=a,retained=a/r if r>0 else None,status=('PASS' if a/r>=.9 else 'FAIL') if r>0 else 'PREMISE_ABSENT')
                    noninferior=native[arm]['nearest_mse_mm2']<=native['KEEP']['nearest_mse_mm2']
                    acceptance[scope][cell][arm]=dict(native_noninferior=noninferior,perturbed=checks,
                        joint_pass=noninferior and all(v['status']=='PASS' for v in checks.values()))
    report=dict(evaluation_type='real_gt',protocol='custom sampled DTU nearest-reference, not official full DTU benchmark',
        units='mm, mm^2',prediction_seal_sha256=sha(ROOT/'PREDICTIONS_SEALED.json'),references=meta,
        requested_base_queries=512,valid_u1_base_queries=sum(q['u1_valid'] for q in query_manifest),failures=failures,
        footprint_definition='all_valid: U native KEEP 1mm reference footprint; paired_common: symmetric U0/U1 union footprint on common fixed pixels; not whole-scene completeness',
        aggregation='equal ROI mean; mean per-ROI improvement percentage is not percentage reduction of mean MSE',
        records=rows,summary=summary,acceptance=acceptance)
    dump(dest/'RESULTS.json',report)
    fields=['case_id','scene','roi','condition','upstream','witness','scope','n','arm','mse_mm2','gain_percent','oracle_mm2','regret_mm2','good','bad','moved','precision1mm','coverage1mm']
    with (dest/'CASE_METRICS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            for arm,m in r['arms'].items():
                writer.writerow({**{k:r[k] for k in fields[:8]},'arm':arm,'mse_mm2':m['nearest_mse_mm2'],
                    'gain_percent':m['improvement_percent'],'oracle_mm2':r['oracle_mse_mm2'],'regret_mm2':m['selection_regret_mm2'],
                    'good':m['beneficial_count'],'bad':m['harmful_count'],'moved':m['moved_count'],
                    'precision1mm':m['precision_1mm'],'coverage1mm':m['fixed_keep_footprint_coverage_1mm']})
    print(json.dumps(dict(valid=report['valid_u1_base_queries'],summary=summary['paired_common'],acceptance=acceptance['paired_common'])),flush=True)

if __name__=='__main__':main()
