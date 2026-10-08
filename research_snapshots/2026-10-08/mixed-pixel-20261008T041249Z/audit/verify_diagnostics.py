"""Independent arithmetic and ownership checks of post-evaluation artifacts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import socket
import numpy as np

ROOT=Path('/srv/slam-research/grf/map-denoise/runs/mixed-pixel-20261008T041249Z')
assert socket.gethostname()=='liekkas'
OUT=ROOT/'diagnostics/pixels'
hashes={}
checks=[]
def digest(path):
    path=Path(path);value=hashlib.sha256(path.read_bytes()).hexdigest();hashes[str(path)]=value
    return value
def read(path):
    digest(path);return json.loads(Path(path).read_text())
def check(name,value):checks.append(dict(name=name,passed=bool(value)))
truth={r['id']:r for r in read(ROOT/'data/truth/metadata.json')}
inputs={r['id']:r for r in read(ROOT/'data/observed/inputs.json')}
summary=read(OUT/'SUMMARY.json')
seal=read(OUT/'SEAL.json')
check('diagnostic_seal',all(digest(p)==h for p,h in seal['files'].items()))
check('diagnostic_source_hash',digest(ROOT/'diagnostics/replay.py')==summary['source_sha256'])
regions=read(OUT/'REGIONS.json')
regionmap={(r['id'],r['arm'],r['incumbent'],r['fold'],r['region']):r for r in regions}
per_fold_rois={}
alpha_cache={}
max_difference=0.
count=0
for stage in ('ordinary_stage','oracle_stage'):
    obs=read(ROOT/stage/'OBSERVATIONS.json')
    for row in obs['rows']:
        if row['arm']=='N':continue
        identifier=row['id'];arm=row['arm'];inc=600 if arm.endswith('D') else row['incumbent']
        with np.load(row['curve_file'],allow_pickle=False) as f:
            candidates=[450.,540.,600.,660.,900.]
            indices=[np.flatnonzero(f['grid']==z).item() for z in candidates]
            expected=f['fold_loss'][:,indices].copy()
            if truth[identifier]['mechanism']=='flat_equal':
                check('whole_grid_equal_color_'+Path(row['curve_file']).name,np.ptp(f['loss'])<=1e-12)
        for fold,ev in enumerate((2,1)):
            path=OUT/f'{identifier}_{arm}_{int(inc)}_fold{fold}.npz'
            with np.load(path,allow_pickle=False) as f:
                data={k:f[k].copy() for k in f.files}
            check('candidate_set_'+path.name,np.array_equal(data['candidates'],candidates))
            check('residual_identity_'+path.name,np.array_equal(data['residual'],data['prediction']-data['observed'][None]))
            key=(identifier,fold)
            if key in per_fold_rois:
                check('common_roi_'+path.name,np.array_equal(data['roi'],per_fold_rois[key]))
            else:
                per_fold_rois[key]=data['roi']
            roi=data['roi'];w=truth[identifier]['producer_parameters']
            if key not in alpha_cache:
                camera=inputs[identifier]['cameras'][ev]
                offsets=np.array([(x/3,y/3) for y in (-1,0,1) for x in (-1,0,1)])
                rays=roi[:,None]+offsets
                xy1=np.concatenate((rays,np.ones(rays.shape[:-1]+(1,))),axis=-1)
                directions=xy1@np.linalg.inv(np.asarray(camera['K'])).T@np.asarray(camera['R'])
                center=np.asarray(camera['C'])
                t=(w['true_depth']-center[2])/directions[...,2]
                xyz=center+t[...,None]*directions
                uv=160*xyz[...,:2]/xyz[...,2,None]+64
                local=uv-np.asarray(w['center']);c=np.cos(w['angle']);s=np.sin(w['angle'])
                inside=(np.abs(c*local[...,0]+s*local[...,1])<=w['half_width'])&(np.abs(-s*local[...,0]+c*local[...,1])<=w['half_height'])
                if w['mechanism']=='textured_single':inside[:]=True
                alpha_cache[key]=inside.mean(axis=-1)
            alpha=alpha_cache[key]
            check('independent_truth_ownership_'+path.name,np.array_equal(alpha,data['true_alpha_evaluation_only']))
            if data['valid']:
                mse=np.mean(data['residual']**2,axis=1)
                difference=float(np.max(np.abs(mse-expected[fold])))
                max_difference=max(max_difference,difference)
                check('sealed_fold_reconstruction_'+path.name,difference<1e-9)
            if arm.endswith('F'):
                check('fixed_alpha_constant_'+path.name,np.all(data['predicted_alpha']==data['predicted_alpha'][0]))
            for region,mask in [('foreground',alpha==1),('background',alpha==0),('boundary',(alpha>0)&(alpha<1))]:
                r=regionmap[identifier,arm,inc,fold,region]
                check('region_count_'+path.name+'_'+region,r['pixel_count']==int(mask.sum()))
                if mask.any():
                    check('region_mse_'+path.name+'_'+region,np.allclose(np.mean(data['residual'][:,mask]**2,axis=1),r['mse'],rtol=1e-12,atol=1e-12))
                else:
                    check('region_empty_'+path.name+'_'+region,r['mse'] is None)
            count+=1
rows=read(ROOT/'evaluation/ROWS.json')
pairwise=read(OUT/'PAIRWISE_REGRET.json')
pairmap={(r['id'],r['arm'],r['initial_depth']):r for r in pairwise}
for row in rows:
    if not row.get('ranking') or not row['ranking'].get('valid'):continue
    values=row['ranking']['candidate_losses'];errors=[abs(z-row['true_depth']) for z in candidates]
    agreement=disagreement=tie=0
    for i in range(len(values)):
        for j in range(i+1,len(values)):
            if abs(errors[i]-errors[j])<1e-9:continue
            if abs(values[i]-values[j])<1e-9:tie+=1
            elif (values[i]-values[j])*(errors[i]-errors[j])>0:agreement+=1
            else:disagreement+=1
    actual=pairmap[row['id'],row['arm'],row['initial_depth']]
    expected=dict(id=row['id'],arm=row['arm'],initial_depth=row['initial_depth'],geometry_order_agreements=agreement,
                  disagreements=disagreement,ties=tie,final_regret_mm=row['error']-min(errors),
                  argmin_regret_mm=row['ranking']['diagnostic_argmin_error_min']-min(errors))
    check('pairwise_regret_'+str((row['id'],row['arm'],row['initial_depth'])),actual==expected)
check('diagnostic_count',count==summary['n_pixel_artifacts']==576)
payload=dict(passed=all(c['passed'] for c in checks),checks_run=len(checks),failures=[c for c in checks if not c['passed']],
    n_pixel_artifacts=count,maximum_independent_loss_difference=max_difference,
    scope='Read-only artifact arithmetic plus independent truth ownership; no inference or generator code imported.',
    verified_hashes=hashes,checks=checks)
with (ROOT/'audit/DIAGNOSTIC_VERIFICATION.json').open('x') as stream:json.dump(payload,stream,indent=2)
print(json.dumps({k:payload[k] for k in ('passed','checks_run','failures','n_pixel_artifacts','maximum_independent_loss_difference')},indent=2))
