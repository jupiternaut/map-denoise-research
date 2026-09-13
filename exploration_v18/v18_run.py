from pathlib import Path
import sys,json,tempfile,time,socket,shutil,resource,traceback,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import v18_operator as op
import v17_run as helper
v14=helper.v14
OLD=v14.RUNS/'adaptive-v17-mklh_h2j'

def create():
    assert socket.gethostname()=='liekkas'
    dest=Path(tempfile.mkdtemp(prefix='retention-v18-',dir=v14.RUNS))
    for name in ('source','inputs','evaluation','outputs','records','diagnostics'):(dest/name).mkdir()
    v14.save(dest/'V17_HASHES_BEFORE.json',helper.hashes((OLD,)))
    for p in ROOT.rglob('*.py'):
        q=dest/'source'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    shutil.copyfile(HERE/'PROTOCOL.md',dest/'source/PROTOCOL.md')
    return dest

def jobs(dest,phase):
    out=[]
    if phase=='development':
        for kind in ('confirmation','bridge'):
            records=json.loads((OLD/f'{kind}_SEALED_BEFORE_GT.json').read_text())
            for r in records:
                if r['method']!='old_spatial36':continue
                j={k:r[k] for k in ('case','seed','gap','sigma','input','evaluation','input_sha256','evaluation_sha256')}
                for k in ('dependence','bias','retain'):
                    if k in r:j[k]=r[k]
                j.update(phase=phase,kind='bridge' if kind=='bridge' else 'statistical',cached_diagnostic=r['diagnostic'],
                    cached_diagnostic_sha256=r['diagnostic_sha256'],cached_old=r['output'],cached_old_sha256=r['output_sha256'])
                out.append(j)
        return out
    if phase=='bridge':
        for j in v14.jobs_for(dest,phase,(9182101,9182111,9182121)):
            out.append(dict(j,kind='bridge',input_sha256=v14.sha(j['input']),evaluation_sha256=v14.sha(j['evaluation'])))
        return out
    for seed in range(9181101,9181201,10):
        for gap in (0.,2.,8.):
            for sigma in (1.,2.):
                for lam in ((0.,) if gap==0 else (0.,.5,1.)):
                    case=f's{seed}_g{gap:g}_n{sigma:g}_l{lam:g}';inp,truth=helper.gen.make(seed,gap,sigma,lam)
                    ip=dest/'inputs'/f'{case}.npz';ep=dest/'evaluation'/f'{case}.npz'
                    helper.npz(ip,inp);helper.npz(ep,truth)
                    out.append(dict(phase=phase,kind='statistical',case=case,seed=seed,gap=gap,sigma=sigma,dependence=lam,
                        input=str(ip),evaluation=str(ep),input_sha256=v14.sha(ip),evaluation_sha256=v14.sha(ep)))
    return out

def worker(job,dest):
    dest=Path(dest);start=time.perf_counter();records=[]
    try:
        assert v14.sha(job['input'])==job['input_sha256'];inp=v14.load(job['input']);state=None
        if job['kind']=='bridge':
            state=v14.v7.freeze(inp['xyz_world'],inp['scan_id'],job['sigma']);x,y=state['design'],state['corrected']
        else:x,y=inp['design'],inp['height_mm']
        kwargs={}
        if job['phase']=='development':
            assert v14.sha(job['cached_diagnostic'])==job['cached_diagnostic_sha256']
            assert v14.sha(job['cached_old'])==job['cached_old_sha256']
            kwargs=dict(cached_pool=json.loads(Path(job['cached_diagnostic']).read_text())['pool'],cached_old=v14.load(job['cached_old']))
        models,diagnostic=op.construct(x,y,job['sigma'],**kwargs)
        diagnostic.update(input_fields=['measured_design','height','sigma'],kind=job['kind'])
        dp=dest/'diagnostics'/f'{job["case"]}.json';v14.save(dp,diagnostic);dh=v14.sha(dp)
        models['identity']=None
        for name in op.METHODS:
            model=models[name];out=inp['xyz_world'].copy();art={}
            if model is not None:
                art={k:np.asarray(model[k]) for k in ('k','means','slope','gate','groups','posterior','prediction')}
                levels=model['means'][None,:]+(x[:,1:]@model['slope'])[:,None]
                art['distance_to_fitted_mm']=np.min(abs(model['prediction'][:,None]-levels),axis=1)
                if state is None:out[:,2]=model['prediction']/1000.
                else:
                    take=np.flatnonzero(state['support']);rows=state['order'][take];disp=model['prediction'][take]-state['local'][take,2]
                    if name=='old_spatial36':out[rows]+=disp[:,None]*state['normal']/1000.
                    else:out[rows]+=(disp/1000.)[:,None]*state['normal']
            else:art.update(k=np.array(0),means=np.array([]),slope=np.zeros(2))
            if state is not None:art.update({k:state[k] for k in ('order','basis','center','normal','design','local','corrected','support','common_scale')})
            art['xyz_world']=out;path=dest/'outputs'/f'{job["case"]}__{name}.npz';helper.npz(path,art)
            records.append(dict(**job,method=name,status='OK',output=str(path),output_sha256=v14.sha(path),
                diagnostic=str(dp),diagnostic_sha256=dh,replaced_r=diagnostic['replaced_r'],selected_family=diagnostic['selected_family'] if name.startswith('retained') else None))
    except Exception:records=[dict(**job,method=m,status='FAILED',error=traceback.format_exc()) for m in op.METHODS]
    for r in records:r.update(shared_case_seconds=time.perf_counter()-start,worker_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    v14.save(dest/'records'/f'{job["case"]}.json',records)
    return records

def score(dest,phase,records):
    rows=[]
    for r in records:
        row={k:r[k] for k in ('phase','kind','case','seed','gap','sigma','method','status','shared_case_seconds')}
        for k in ('dependence','bias','retain','selected_family','replaced_r'):
            if k in r:row[k]=r[k]
        if r['status']!='OK':rows.append(row);continue
        assert v14.sha(r['output'])==r['output_sha256']
        inp,truth,art=map(v14.load,(r['input'],r['evaluation'],r['output']))
        if 'json' in truth:truth.update(json.loads(truth.pop('json').tobytes().decode()))
        q=art['xyz_world']*1000.;reference=truth['gt_clean_xyz_world']*1000.;labels=truth['gt_layer']
        if r['kind']=='bridge':
            distances=[]
            for z,x0,x1,y0,y1 in truth['surface_rectangles_mm']:
                distances.append(np.sqrt((q[:,0]-np.clip(q[:,0],x0,x1))**2+(q[:,1]-np.clip(q[:,1],y0,y1))**2+(q[:,2]-z)**2))
            d=np.asarray(distances).T
        else:
            d=abs(q[:,2,None]-truth['true_means_mm']);np.testing.assert_array_equal(art['xyz_world'][:,:2],inp['xyz_world'][:,:2])
        row.update(surface_mae_mm=float(d.min(1).mean()),balanced_source_mae_mm=float(np.mean([d[labels==k,k].mean() for k in np.unique(labels)])),
            matched_rms_mm=float(np.sqrt(np.mean(np.sum((q-reference)**2,axis=1)))),coverage_1mm=float(np.mean(cKDTree(q).query(reference)[0]<=1.+1e-9)),
            model_k=None if r['method']=='identity' else int(art['k']))
        if r['gap']:
            gap=float(q[labels==1,2].mean()-q[labels==0,2].mean())
            row.update(source_gap_mm=gap,source_gap_error_mm=abs(gap-r['gap']))
        if r['method']!='identity':
            if r['kind']=='bridge':
                vec=art['basis'][:,2]-art['basis'][:,:2]@(art['slope']/art['common_scale'])
                plane_scale=float(np.linalg.norm(vec));vertical_scale=abs(vec[2]);support=art['support']
                unsupported=art['order'][~support];np.testing.assert_array_equal(art['xyz_world'][unsupported],inp['xyz_world'][unsupported])
            else:plane_scale=float(np.sqrt(1+np.sum((art['slope']/50.)**2)));vertical_scale=1.;support=np.ones(len(q),bool)
            row['distance_to_fitted_surface_mm']=float(np.mean(art['distance_to_fitted_mm'][support])/plane_scale)
            if r['gap']:
                fg=float(np.ptp(art['means'])/max(vertical_scale,1e-12))
                row.update(fitted_gap_mm=fg,fitted_gap_error_mm=abs(fg-r['gap']))
        rows.append(row)
    v14.save(dest/f'{phase}_ROWS.json',rows);v14.csvsave(dest/f'{phase}_RESULTS.csv',rows)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','confirmation','bridge']);p.add_argument('--dest');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    dest=Path(args.dest) if args.dest else create();print('RUN',dest,flush=True)
    if args.phase!='development':
        lock=json.loads((dest/'CONFIRMATION_LOCK.json').read_text())
        for path,h in lock.items():assert v14.sha(path)==h,path
    jj=jobs(dest,args.phase);v14.save(dest/f'{args.phase}_JOBS.json',jj);records=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(worker,j,str(dest)) for j in jj]),1):
            records.extend(f.result())
            if i%20==0 or i==len(jj):print(args.phase,i,'/',len(jj),'failed',sum(r['status']!='OK' for r in records),flush=True)
    records.sort(key=lambda r:(r['case'],r['method']));v14.save(dest/f'{args.phase}_SEALED_BEFORE_GT.json',records)
    v14.save(dest/f'{args.phase}_TIMING.json',dict(wall_seconds=time.perf_counter()-start,workers=args.workers,
        parent_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,max_single_worker_rss_kib=max(r['worker_rss_kib'] for r in records),
        note='Development reuses cached fits; fresh phases fit. Separate process lifetime RSS maxima, not aggregate peak. Shared candidate cost, CPU only.'))
    score(dest,args.phase,records)
    if args.phase=='development':
        sources=[HERE/'v18_operator.py',HERE/'PROTOCOL.md',ROOT/'exploration_v17/v17_operator.py',ROOT/'exploration_v16/models.py',ROOT/'exploration_v15/constant_solver.py',ROOT/'exploration_v11/spatial_filter.py']
        v14.save(dest/'CONFIRMATION_LOCK.json',{str(p):v14.sha(p) for p in sources})
    print('DONE',args.phase,len(records),flush=True)

if __name__=='__main__':main()
