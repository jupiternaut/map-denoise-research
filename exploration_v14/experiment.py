"""Effect-first controlled benchmark; estimator sees XYZ, provenance, supplied sigma only."""
from pathlib import Path
import sys, json, hashlib, tempfile, shutil, socket, time, importlib.util, traceback, csv
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT),str(ROOT/'exploration_v13'),str(ROOT/'exploration_v12'),str(ROOT/'exploration_v11')]
from generate_synthetics import make_ghost,make_dual
from schema import POINT_KEYS,fill_rays,write_patch
from evaluate_v2 import synthetic_geometry
from atlas import prepare,decode
from spatial_filter import filter_frozen
from local_filter import corrected_world
from external import estimate_frozen as external
RUNS=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
spec=importlib.util.spec_from_file_location('_v14_upstream',ROOT/'exploration_v7/algorithm/reassociation.py')
v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(p,x):
    with Path(p).open('x') as f:json.dump(serial(x),f,indent=2,allow_nan=False)
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def csvsave(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def make(seed,gap,bias,sigma,retain):
    p,m,e=make_ghost(seed,bias) if gap==0 else make_dual(seed,gap,bias)
    n=len(p['xyz_world']);labels=e['gt_layer'];rng=np.random.default_rng(np.random.SeedSequence([seed,1414]))
    eligible=np.ones(n,bool) if gap==0 else labels==1
    keep=~eligible|(rng.random(n)<retain)
    for k in POINT_KEYS:p[k]=p[k][keep].copy()
    for k in ('gt_layer','gt_clean_xyz_world','gt_eps_mm'):e[k]=e[k][keep].copy()
    eps=e['gt_eps_mm']*sigma;e['gt_eps_mm']=eps
    p['xyz_world']=e['gt_clean_xyz_world'].copy()
    p['xyz_world'][:,2]+=(eps+e['gt_bias_mm'][p['scan_id']])/1000
    p['xyz_local']=p['xyz_world']-p['scanner_origin_world'];fill_rays(p)
    support={int(k):dict(n=int(np.sum(e['gt_layer']==k)),stations=np.unique(p['scan_id'][e['gt_layer']==k]).tolist()) for k in np.unique(e['gt_layer'])}
    m.update(sigma_mm_generation=sigma,retention=retain,layer_support=support,
        sampling_note='First returns followed by controlled layer-dependent dropout; monolayer dropout affects all points.')
    e['layer_support']=support
    return p,m,e

def layer_scores(out,e):
    q=out*1000;labels=e['gt_layer'];means=[];nearest=[]
    for k,(z,x0,x1,y0,y1) in enumerate(e['surface_rectangles_mm']):
        projection=q.copy();projection[:,0]=np.clip(q[:,0],x0,x1);projection[:,1]=np.clip(q[:,1],y0,y1);projection[:,2]=z
        d=np.linalg.norm(q-projection,axis=1);nearest.append(d)
        if np.any(labels==k):means.append(float(d[labels==k].mean()))
    return dict(layer_balanced_source_surface_mae_mm=float(np.mean(means)),worst_layer_source_surface_mae_mm=max(means),
        independent_surface_mae_mm=float(np.min(nearest,axis=0).mean()))

def apply(state,sigma,method,prepared=None):
    world=state['world']
    if method=='identity':return world.copy(),{},np.zeros(len(world),bool)
    if 'design' not in state:return world.copy(),dict(status='UNSUPPORTED'),np.zeros(len(world),bool)
    q,mask=corrected_world(state)
    if method=='bias_only':
        out=world.copy();out[mask]=q[mask];return out,{},mask
    if method=='old_original':out,info,_=v7.fit_frozen(state,variant='original',budget=0,sharing='independent')
    elif method.startswith('spatial_'):out,info,_=filter_frozen(state,sigma,'spatial_free',int(method.split('_')[1]))
    elif method.startswith('atlas_'):
        mode,nn=method.rsplit('_',1);out,info,_=decode(state,prepared if prepared is not None else prepare(state,sigma),sigma,mode,int(nn))
    else:
        kind,scale=method.split('_');out,info,_=external(state,kind,float(scale))
    return out,info,mask

def jobs_for(dest,phase,seeds):
    jobs=[]
    for seed in seeds:
        for gap in (0,2,4,8):
            for sigma in (.5,1.,2.):
                for bias in (0,4):
                    for retain in (1.,.25):
                        case=f's{seed}_g{gap}_n{sigma}_b{bias}_r{retain}'
                        p,m,e=make(seed,gap,bias,sigma,retain);write_patch(dest/'inputs',case,p,m,e)
                        jobs.append(dict(phase=phase,case=case,seed=seed,gap=gap,sigma=sigma,bias=bias,retain=retain,n=len(p['xyz_world']),
                            input=str(dest/'inputs'/f'{case}.npz'),evaluation=str(dest/'inputs/evaluation'/f'{case}.eval.npz')))
    return jobs

def run_phase(dest,phase,seeds,methods):
    jobs=jobs_for(dest,phase,seeds);save(dest/f'{phase}_JOBS.json',jobs);records=[]
    for idx,j in enumerate(jobs):
        inp=load(j['input']);state=v7.freeze(inp['xyz_world'],inp['scan_id'],j['sigma'])
        prepared=prepare(state,j['sigma']) if 'design' in state else None
        for method in methods:
            start=time.perf_counter();record=dict(**j,method=method,input_sha256=sha(j['input']))
            try:
                out,info,mask=apply(state,j['sigma'],method,prepared)
                assert out.shape==inp['xyz_world'].shape and np.isfinite(out).all()
                np.testing.assert_array_equal(out[~mask],inp['xyz_world'][~mask])
                path=dest/'outputs'/f'{j["case"]}__{method}.npz'
                with path.open('xb') as f:np.savez_compressed(f,xyz_world=out,support_mask=mask)
                record.update(output=str(path),output_sha256=sha(path),support=float(mask.mean()),status=info.get('status','OK'),info=info)
            except Exception:record.update(status='FAILED',error=traceback.format_exc())
            record['seconds']=time.perf_counter()-start;records.append(record)
        print(phase,idx+1,len(jobs),j['case'],'sealed',flush=True)
    save(dest/f'{phase}_SEALED_BEFORE_GT.json',records)
    rows=[]
    for r in records:
        row={k:r[k] for k in ('phase','case','seed','gap','sigma','bias','retain','n','method','status','seconds')}
        if r['status']!='FAILED':
            out=load(r['output'])['xyz_world'];e=load(r['evaluation']);e.update(json.loads(e.pop('json').tobytes().decode()))
            row.update(synthetic_geometry(out,{},e));row.update(layer_scores(out,e));row['support']=r['support']
            assert abs(row['surface_accuracy_mean_mm']-row['independent_surface_mae_mm'])<1e-10
        rows.append(row)
    csvsave(dest/f'{phase}_RESULTS.csv',rows);save(dest/f'{phase}_ROWS.json',rows)
    return rows

def aggregate(rows):
    result={}
    keys=('surface_accuracy_mean_mm','matched_point_rms_mm','layer_balanced_source_surface_mae_mm','worst_layer_source_surface_mae_mm',
        'reference_sample_coverage_1mm','source_group_gap_retention','support')
    for method in sorted(set(r['method'] for r in rows)):
        part=[r for r in rows if r['method']==method];ok=[r for r in part if r['status']!='FAILED']
        result[method]=dict(cases=len(part),failed=len(part)-len(ok),unsupported=sum(r['status']=='UNSUPPORTED' for r in part))
        for key in keys:
            vals=[r[key] for r in ok if key in r and r[key] is not None]
            if vals:result[method][key]=float(np.mean(vals))
    return result

def main():
    assert socket.gethostname()=='liekkas'
    dest=Path(tempfile.mkdtemp(prefix='effect-v14-',dir=RUNS));print(dest,flush=True)
    for name in ('source','outputs'):(dest/name).mkdir()
    sources={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.md')}
    save(dest/'SOURCES_BEFORE.json',sources)
    for name in sources:
        target=dest/'source'/Path(name).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    fixed=['identity','bias_only','old_original','atlas_tied_256','atlas_relax_96']
    methods=fixed+[f'spatial_{n}' for n in (12,36,108)]+[f'{kind}_{s}' for kind in ('apss','rimls') for s in (.5,1.,2.,4.,8.)]
    dev=run_phase(dest,'development',(9140101,9140111),methods);agg=aggregate(dev);save(dest/'development_AGGREGATES.json',agg)
    chosen={}
    for prefix in ('spatial_','apss_','rimls_'):
        valid=[m for m in methods if m.startswith(prefix) and not agg[m]['failed']]
        if not valid:raise RuntimeError(f'No successful configuration: {prefix}')
        chosen[prefix]=min(valid,key=lambda m:(round(agg[m]['surface_accuracy_mean_mm'],12),agg[m]['matched_point_rms_mm'],m))
    save(dest/'SELECTION_LOCK.json',dict(chosen=chosen,rule='global development mean surface MAE; no timing objective',source_sha256=sha(HERE/'experiment.py')))
    print('LOCKED',chosen,flush=True)
    test=run_phase(dest,'confirmation',(9141101,9141111,9141121),fixed+list(chosen.values()))
    save(dest/'confirmation_AGGREGATES.json',aggregate(test))
    slices={}
    for field in ('gap','sigma','bias','retain','seed'):
        slices[field]={str(v):aggregate([r for r in test if r[field]==v]) for v in sorted(set(r[field] for r in test))}
    save(dest/'confirmation_SLICES.json',slices)
    assert all(sha(p)==h for p,h in sources.items())
    save(dest/'SUMMARY.json',dict(development_records=len(dev),confirmation_records=len(test),sources_unchanged=len(sources),
        failed=sum(r['status']=='FAILED' for r in dev+test),scope='controlled first-return family with dropout and variable noise; not independent real geometry'))
    print(json.dumps(aggregate(test),indent=2),flush=True)

if __name__=='__main__':main()
