"""Read-only independent audit; prints JSON. No imports of run or policy code."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
import csv
import hashlib
import json
import math
import socket
import sys
import time
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/surface-owned-support-20261008T022918Z')
HISTORY = ROOT.parent/'selector-attribution-20261007T180539Z'
TRACK = ROOT.parent/'track-discrimination-20261007T160716Z'
ANCESTOR = ROOT.parent/'surface-evidence-20261001T140058Z/evaluation'
TRANSFER = ROOT.parent/'colmap-transfer-20260930T180000Z'
assert socket.gethostname() == 'liekkas'
assert Path(__file__).resolve().parent == ROOT/'audit'
assert sys.executable == '/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python'
started = time.monotonic()
def load(p): return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='') as s: return list(csv.DictReader(s))
def key(r): return r['roi'], int(r['query'])
def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda:s.read(1048576), b''): h.update(b)
    return h.hexdigest()
def close(x,y): return math.isclose(x,y,rel_tol=2e-12,abs_tol=2e-11)
def merge(xs):
    ans=[]
    for a,b in sorted(xs):
        if ans and a <= ans[-1][1]+1e-9: ans[-1][1]=max(b,ans[-1][1])
        else: ans.append([float(a),float(b)])
    return ans
def intervals(z,ok):
    starts=np.flatnonzero(ok & ~np.r_[False,ok[:-1]])
    stops=np.flatnonzero(ok & ~np.r_[ok[1:],False])
    return merge([[max(z[0],z[a]-1.5),min(z[-1],z[b]+1.5)] for a,b in zip(starts,stops)])
def inter(a,b):
    return merge([[max(l,p),min(r,q)] for l,r in a for p,q in b if max(l,p)<=min(r,q)])

lock=load(ROOT/'RUN_LOCK.json'); seal=load(ROOT/'PREDICTIONS_SEALED.json')
saved=load(ROOT/'evaluation/RESULTS.json'); request=load(ROOT/'REQUESTS.json')
obs=load(ROOT/'OBSERVATIONS.json'); inputs=load(ROOT/'INPUTS.json')
decisions=load(ROOT/'DECISIONS.json'); arms=load(ROOT/'PROTOCOL.json')['arms']
checks={**lock['files'],**lock['photos'],**seal['files'],**saved['evaluation_inputs'],**inputs['sources']}
hash_fail=[p for p,h in checks.items() if digest(p)!=h]
assert not hash_fail,hash_fail
assert lock['created_at'] < obs['created_at'] < decisions['created_at'] < seal['created_at'] < saved['created_at']
O={key(r):r for r in obs['rows']}; I={key(r):r for r in inputs['rows']}; D={key(r):r for r in decisions['rows']}
T={key(r):r for r in load(TRACK/'observation/TRACKS.json')['rows']}
assert len(O)==len(I)==len(D)==21 and set(O)==set(I)==set(D)==set(T)
assert sum(len(r['objects']) for r in I.values())==123
assert sum(o['candidate_id']>=0 for r in I.values() for o in r['objects'])==105
assert all(O[k]['evidence']['legacy_full9']['intervals']==T[k]['star_full_intervals'] for k in O)

# Independent bilinear sampler (no scipy.ndimage or author sampling code).
def sample(im,p):
    x,y=p[...,0],p[...,1]; good=np.isfinite(p).all(-1)&(x>=0)&(y>=0)&(x<=im.shape[1]-1)&(y<=im.shape[0]-1)
    x=np.nan_to_num(x); y=np.nan_to_num(y)
    a=np.clip(np.floor(x).astype(int),0,im.shape[1]-1); b=np.clip(np.floor(y).astype(int),0,im.shape[0]-1)
    aa=np.minimum(a+1,im.shape[1]-1); bb=np.minimum(b+1,im.shape[0]-1)
    u=x-a; v=y-b
    val=(1-u)*(1-v)*im[b,a]+u*(1-v)*im[b,aa]+(1-u)*v*im[bb,a]+u*v*im[bb,aa]
    return np.where(good,val,0.),good
def connected(p):
    mask=np.zeros((9,9),bool); mask[4,4]=True
    while True:
        before=mask.copy()
        for y in range(9):
            for x in range(9):
                if not before[y,x]: continue
                for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0<=yy<9 and 0<=xx<9 and abs(p[yy,xx]-p[y,x])<=20: mask[yy,xx]=True
        if np.array_equal(mask,before): return mask
images={}
for sn,scene in request['scenes'].items():
    for name,c in scene['cameras'].items():
        with Image.open(c['image_path']) as im:
            images[sn,name]=np.array(im.convert('L').resize((c['width'],c['height']),Image.Resampling.BILINEAR),float)
off=np.array([[x,y] for y in range(-4,5) for x in range(-4,5)])
max_score_error=0.; score_values=0; scan_count=0; rebuilt={}; mask_rows=[]; score_warp_delta={}; changed_support=[]
for k,o in O.items():
    scene=request['scenes'][str(o['scene'])]; ref=scene['cameras'][scene['reference']]
    z=np.arange(scene['depth_range_mm'][0],scene['depth_range_mm'][1]+0.01,1.)
    assert np.array_equal(z,o['depth_grid_mm'])
    xy=np.asarray(o['pixel_xy']); pixels=xy+off
    patch,refvalid=sample(images[str(o['scene']),scene['reference']],pixels)
    masks={'full9':np.ones((9,9),bool),'center3':np.pad(np.ones((3,3),bool),3),'connected9':connected(patch.reshape(9,9))}
    mask_rows.append(dict(roi=k[0],query=k[1],connected_count=int(masks['connected9'].sum()),std_full=float(patch.std()),std_connected=float(patch[masks['connected9'].ravel()].std()),std_center=float(patch[masks['center3'].ravel()].std())))
    inv=np.linalg.inv(np.asarray(ref['K_half'])); R=np.asarray(ref['R']); C=np.asarray(ref['center'])
    pixel_rays=np.c_[pixels,np.ones(81)]@inv.T@R
    center_ray=np.r_[xy,1.]@inv.T@R
    assert np.allclose(center_ray,I[k]['ray'],rtol=0,atol=1e-15)
    rebuilt[k]={}; per_scan={}
    for arm in arms[1:]:
        warp,kind=arm.split('_'); mask=masks[kind].ravel(); a=patch[mask]; count=int(mask.sum()); a=a-a.mean(); stda=np.sqrt(np.mean(a*a))
        accepted=[]
        for num,src in enumerate(scene['sources']):
            c=scene['cameras'][src]; score=np.full(len(z),np.nan); valid=np.zeros(len(z),bool)
            if count>=9 and refvalid[mask].all() and stda>=3:
                if warp=='translation': world=C+z[:,None]*center_ray
                else: world=C+z[:,None,None]*pixel_rays[None,mask,:]
                local=(world-np.asarray(c['center']))@np.asarray(c['R']).T
                uvw=local@np.asarray(c['K_half']).T; uv=uvw[...,:2]/uvw[...,2:]
                positive=local[...,2]>0
                if warp=='translation': uv=uv[:,None,:]+off[None,mask,:]
                else: positive=positive.all(1)
                b,good=sample(images[str(o['scene']),src],uv); b=b-b.mean(1,keepdims=True); stdb=np.sqrt(np.mean(b*b,1))
                valid=good.all(1)&positive&(z>0)&(stdb>=3)
                score[valid]=np.einsum('ij,j->i',b[valid],a)/(count*stda*stdb[valid])
            actual=o['evidence'][arm]['scans'][num]; reference=np.array([np.nan if v is None else v for v in actual['scores']])
            assert np.array_equal(np.isfinite(score),np.isfinite(reference)),(k,arm,src,'finite')
            err=float(np.max(np.abs(score[valid]-reference[valid]))) if valid.any() else 0.
            max_score_error=max(max_score_error,err); score_values+=int(valid.sum()); scan_count+=1
            assert err<1e-11,(k,arm,src,err)
            assert actual['mask_count']==count and np.array_equal(actual['mask'],masks[kind])
            assert actual['valid_count']==int(valid.sum()) and close(actual['anchor_std'],stda)
            ok=valid&(score>=.6); accepted.append(ok)
            assert int(ok.sum())==actual['accepted_count']
            per_scan[arm,src]=score
        shared=accepted[0]&accepted[1]
        rebuilt[k][arm]=intervals(z,shared)
        assert np.flatnonzero(shared).tolist()==o['evidence'][arm]['joint_sample_indices']
        assert rebuilt[k][arm]==o['evidence'][arm]['intervals'],(k,arm)
        if arm=='translation_full9':
            rebuilt[k]['legacy_full9']=inter(intervals(z,accepted[0]),intervals(z,accepted[1]))
            assert rebuilt[k]['legacy_full9']==o['evidence']['legacy_full9']['intervals']
    for kind in masks:
        ds=[]
        for src in scene['sources']:
            a=per_scan['translation_'+kind,src];b=per_scan['plane_'+kind,src];v=np.isfinite(a)&np.isfinite(b)
            if v.any(): ds.extend(np.abs(a[v]-b[v]).tolist())
        score_warp_delta.setdefault(kind,[]).extend(ds)
        if rebuilt[k]['translation_'+kind]!=rebuilt[k]['plane_'+kind]: changed_support.append([k[0],k[1],kind])

# Independent world-distance selector; exact represented-float arithmetic, no gain kernel.
def gain(a,b,C,r,z):
    q=[F(float(c))+F(float(z))*F(float(t)) for c,t in zip(C,r)]
    return sum((F(float(x))-t)**2-(F(float(y))-t)**2 for x,y,t in zip(a,b,q))
def lower(x):
    v=float(x)
    return math.nextafter(v,-math.inf) if F(v)>x else v
selection_count=0; score_count=0; min_positive=None; policy_margins=[]
for k,r in I.items():
    pool={o['candidate_id']:o['xyz_mm'] for o in r['objects']}; cur=r['current_id']
    for arm in arms:
        ev=rebuilt[k][arm]; d=D[k]['decisions'][arm]; target=None; winner=cur
        if cur not in pool: reason='NO_CURRENT_OUTPUT'
        elif not ev: reason='EMPTY_UNKNOWN'
        else:
            lengths=[b-a for a,b in ev];total=math.fsum(lengths)
            target=math.fsum(w*(a+(b-a)/2) for (a,b),w in zip(ev,lengths))/total if total else math.fsum(a for a,b in ev)/len(ev)
            gains={cid:lower(gain(pool[cur],p,r['center'],r['ray'],target)) for cid,p in pool.items()}
            eligible=[cid for cid,g in gains.items() if cid!=cur and g>0]
            winner=min(eligible,key=lambda cid:(-gains[cid],cid)) if eligible else cur
            reason='POSITIVE_POINT_GAIN' if eligible else 'NO_POSITIVE_POINT_GAIN'
            assert sorted(eligible)==d['eligible_ids']
            for s in d['scores']:
                score_count+=1;cid=s['candidate_id'];assert gains[cid]==s['point_gain_low_mm2']
            if eligible:
                sortedg=sorted((gains[cid] for cid in pool),reverse=True)
                policy_margins.append(sortedg[0]-sortedg[1])
                min_positive=min(gains[cid] for cid in eligible) if min_positive is None else min(min_positive,min(gains[cid] for cid in eligible))
        assert winner==d['selected_candidate_id'] and reason==d['reason'] and target==d['target_depth'],(k,arm)
        selection_count+=1

archive=rows(TRACK/'evaluation/POINT_METRICS.csv'); points=rows(ROOT/'evaluation/POINT_METRICS.csv')
photo={key(r):r for r in archive if r['arm']=='photo_U11'}
original={key(r):r for r in rows(ANCESTOR/'POINT_METRICS.csv') if r['arm']=='incumbent'}
candidates={(*key(r),int(r['candidate_id'])):r for r in rows(ANCESTOR/'CANDIDATE_DISTANCES.csv')}
def dist(r): return float(r['distance_mm']) if r['distance_mm'] else None
def source(k,cid): return original[k] if cid==-1 else candidates[(*k,cid)]
for k,r in I.items():
    for obj in r['objects']: assert obj['xyz_mm']==[float(source(k,obj['candidate_id'])[n]) for n in ('x_mm','y_mm','z_mm')]
    if r['current_id'] is not None: assert dist(photo[k])==dist(source(k,r['current_id']))
assert len(points)==4096 and len(photo)==512 and len({(r['arm'],*key(r)) for r in points})==4096
for p in points:
    k=key(p);a=p['arm'];expected=dist(photo[k])
    if a!='photo_U11' and k in D:
        cid=D[k]['decisions'][a]['selected_candidate_id'];expected=None if cid is None else dist(source(k,cid))
        assert (int(p['selected']) if p['selected'] else None)==cid
    assert dist(p)==expected,(a,k)
    assert p['primary']==original[k]['primary']
metrics=[];trans=[]
for arm in ['photo_U11']+arms:
    rr=[r for r in points if r['arm']==arm]; pp=[r for r in rr if r['primary']=='True'];assert len(rr)==512 and len(pp)==484
    roi={}
    for name in sorted({r['roi'] for r in pp}):
        ds=[dist(r) for r in pp if r['roi']==name];roi[name]=dict(n=len(ds),mse_mm2=math.fsum(d*d for d in ds)/len(ds),mae_mm=math.fsum(ds)/len(ds))
    dif=[dist(r)-dist(photo[key(r)]) for r in pp]
    m=dict(arm=arm,n=484,mse_mm2=math.fsum(v['mse_mm2'] for v in roi.values())/4,mae_mm=math.fsum(v['mae_mm'] for v in roi.values())/4,roi=roi,improved=sum(x<-1e-9 for x in dif),worsened=sum(x>1e-9 for x in dif),unchanged=sum(abs(x)<=1e-9 for x in dif),new_harm=sum(dist(photo[key(r)])<=1<dist(r) for r in pp),severe=sum(dist(r)>5 for r in pp),finite=sum(dist(r)is not None for r in rr),missing_finite=sum(r['primary']!='True' and dist(r)is not None for r in rr))
    baseline=m['mse_mm2'] if arm=='photo_U11' else metrics[0]['mse_mm2'];m['mse_reduction_pct_vs_photo']=100*(baseline-m['mse_mm2'])/baseline
    reported=next(x for x in saved['metrics'] if x['arm']==arm);assert m==reported,(arm,m,reported)
    metrics.append(m)
    for k,r in D.items():
        if arm!='photo_U11' and r['decisions'][arm]['selected_candidate_id']!=r['current_id']:
            cid=r['decisions'][arm]['selected_candidate_id'];trans.append(dict(arm=arm,roi=k[0],query=k[1],photo_mm=dist(photo[k]),output_mm=dist(source(k,cid)),photo_id=r['current_id'],selected=cid))
saved_trans=rows(ROOT/'evaluation/TRANSITIONS.csv')
assert len(trans)==len(saved_trans)
for a,b in zip(trans,saved_trans):
    assert all(str(v)==b[k] for k,v in a.items())

# Raw official reference recomputation, reading binary records independently.
raw=[];rawdist={};max_dist_err=0.
for ref in load(TRANSFER/'references/MANIFEST.json')['files']:
    assert digest(ref['path'])==ref['sha256']
    with open(ref['path'],'rb') as s:
        header=[]
        while True:
            line=s.readline().decode('ascii').strip();header.append(line)
            if line=='end_header':break
        assert header[1]=='format binary_little_endian 1.0'
        n=int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
        expected=['property float '+x for x in ['x','y','z','nx','ny','nz']]+['property uchar '+x for x in ['red','green','blue']]
        assert [x for x in header if x.startswith('property ')]==expected
        rec=np.fromfile(s,dtype=np.dtype([('v','<f4',(6,)),('rgb','u1',(3,))]),count=n)
    vertices=rec['v'][:,:3].astype(float);tree=cKDTree(vertices)
    objects=[(k,-1,r) for k,r in original.items() if int(r['scene'])==ref['scene']]+[(x[:2],x[2],r) for x,r in candidates.items() if int(r['scene'])==ref['scene']]
    finite=[(k,cid,r) for k,cid,r in objects if r['x_mm'] and r['y_mm'] and r['z_mm']]
    xyz=np.array([[float(r[c]) for c in ('x_mm','y_mm','z_mm')] for k,cid,r in finite]);ds=tree.query(xyz,workers=1)[0]
    for (k,cid,r),d in zip(finite,ds):
        err=abs(float(d)-dist(r));max_dist_err=max(max_dist_err,err);assert err<1e-10;rawdist[k,cid]=float(d)
    raw.append(dict(scene=ref['scene'],vertices=n,coordinates=len(finite),reference_sha256=ref['sha256']))
    del tree,vertices,rec
raw_point_count=0
for p in points:
    if dist(p) is None:continue
    k=key(p);arm=p['arm']
    cid=D[k]['decisions'][arm]['selected_candidate_id'] if k in D and arm!='photo_U11' else (I[k]['current_id'] if k in I else -1)
    assert abs(rawdist[k,cid]-dist(p))<1e-10;raw_point_count+=1

control=[]
for kind in ('full9','center3','connected9'):
    changes=[dict(roi=k[0],query=k[1],translation=D[k]['decisions']['translation_'+kind]['selected_candidate_id'],plane=D[k]['decisions']['plane_'+kind]['selected_candidate_id']) for k in D if D[k]['decisions']['translation_'+kind]['selected_candidate_id']!=D[k]['decisions']['plane_'+kind]['selected_candidate_id']]
    control.append(dict(mask=kind,changed_support_count=sum(x[2]==kind for x in changed_support),changed_decisions=changes,max_score_delta=max(score_warp_delta[kind],default=0),mean_absolute_score_delta=float(np.mean(score_warp_delta[kind]))))
assert all(D[k]['decisions'][w+'_full9']['selected_candidate_id']==D[k]['decisions'][w+'_connected9']['selected_candidate_id'] for k in D for w in ('translation','plane'))
decomposition=load(ROOT/'evaluation/NCC_DECOMPOSITION.json')
decomp_error=0.;center=np.array([abs(x)<=1 and abs(y)<=1 for x,y in off])
for row in decomposition['rows']:
    k=key(row);o=O[k];scene=request['scenes'][str(o['scene'])];ref=scene['cameras'][scene['reference']];src=scene['cameras'][row['source']]
    parts=rebuilt[k]['translation_full9'];length=sum(b-a for a,b in parts)
    z=sum((b-a)*(a+b)/2 for a,b in parts)/length if length else sum(a for a,b in parts)/len(parts)
    assert z==row['target_support_mean_mm']
    xy=np.array(o['pixel_xy']);a,_=sample(images[str(o['scene']),scene['reference']],xy+off)
    ray=np.r_[xy,1.]@np.linalg.inv(np.array(ref['K_half'])).T@np.array(ref['R'])
    world=np.array(ref['center'])+z*ray;local=(world-np.array(src['center']))@np.array(src['R']).T
    uvw=local@np.array(src['K_half']).T;uv=uvw[:2]/uvw[2];b,valid=sample(images[str(o['scene']),row['source']],uv+off)
    ac=a-a.mean();bc=b-b.mean();den=np.linalg.norm(ac)*np.linalg.norm(bc)
    cc=np.dot(a[center]-a[center].mean(),b[center]-b[center].mean())
    cr=np.dot(a[~center]-a[~center].mean(),b[~center]-b[~center].mean())
    cg=8.*(a[center].mean()-a[~center].mean())*(b[center].mean()-b[~center].mean())
    vals=dict(center_std=float(a[center].std()),full_std=float(a.std()),center_within_energy_fraction=float(np.sum((a[center]-a[center].mean())**2)/np.dot(ac,ac)),ncc=float(np.dot(ac,bc)/den),center_ncc_contribution=float(cc/den),ring_ncc_contribution=float(cr/den),between_ncc_contribution=float(cg/den))
    assert abs(vals['ncc']-vals['center_ncc_contribution']-vals['ring_ncc_contribution']-vals['between_ncc_contribution'])<1e-12
    assert bool(valid.all())==row['valid']
    for n,v in vals.items():
        err=abs(v-row[n]);decomp_error=max(decomp_error,err);assert err<1e-12,(k,n,err)
repair_keys=[k for k in D if k[0]=='scan118_upper_fold' and k[1] in (5,95)]
report=dict(host=socket.gethostname(),root=str(ROOT),python=sys.executable,check_type='independent runcheck; author modules not imported',hashes_checked=len(checks)+2,hash_failures=hash_fail,chronology_consistent=True,legacy_supports_exact=21,raw_image_scans_replayed=scan_count,finite_ncc_values_checked=score_values,max_abs_ncc_error=max_score_error,all_147_supports_exact=True,selections_checked=selection_count,exact_rational_point_scores_checked=score_count,minimum_positive_point_gain_mm2=min_positive,minimum_winner_margin_mm2=min(policy_margins),candidate_coordinate_identity_checks=123,evaluator_rows_verified=4096,raw_reference_coordinate_checks=raw,raw_distance_rows_verified=raw_point_count,max_abs_raw_distance_error_mm=max_dist_err,metrics=metrics,transitions=trans,connected_mask_counts=dict(sorted(Counter(x['connected_count'] for x in mask_rows).items())),mask_details=mask_rows,warp_controls=control,large_repair_selections={a:[D[k]['decisions'][a]['selected_candidate_id'] for k in repair_keys] for a in arms},read_log=obs['read_log'],blocked_negative_reads=obs['blocked_negative_reads'],wallclock_seconds=time.monotonic()-started)
report['posthoc_decomposition']=dict(rows_verified=len(decomposition['rows']),max_abs_field_error=decomp_error,formula_identity_pass=True,diagnostic_source_sha256=digest(ROOT/'diagnose_evidence.py'),diagnostic_output_sha256=digest(ROOT/'evaluation/NCC_DECOMPOSITION.json'))
print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
