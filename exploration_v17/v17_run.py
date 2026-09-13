"""Run frozen-input V17 development / confirmation / first-return bridge."""
from pathlib import Path
import sys,importlib.util,json,time,tempfile,socket,shutil,resource,traceback,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import v17_operator as op

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v14=module('v17_support14',ROOT/'exploration_v14/experiment.py')
gen=module('v17_generator16',ROOT/'exploration_v16/generator.py')
metrics=module('v17_metrics16',ROOT/'exploration_v16/metrics.py')
OLD16=v14.RUNS/'mechanism-v16-hky6n1fv'
OLD15=v14.RUNS/'challenge-v15-tchfksi_'

def npz(path,data):
    with Path(path).open('xb') as f:np.savez_compressed(f,**data)

def hashes(roots):return {str(p):v14.sha(p) for root in roots for p in root.rglob('*') if p.is_file()}

def create():
    assert socket.gethostname()=='liekkas'
    dest=Path(tempfile.mkdtemp(prefix='adaptive-v17-',dir=v14.RUNS))
    for d in ('source','inputs','evaluation','outputs','records','diagnostics'):(dest/d).mkdir()
    for p in ROOT.rglob('*.py'):
        q=dest/'source'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    shutil.copyfile(HERE/'PROTOCOL.md',dest/'source/PROTOCOL.md')
    v14.save(dest/'OLD_HASHES_BEFORE.json',hashes((OLD15,OLD16)))
    v14.save(dest/'RUN_IDENTITY.json',dict(host=socket.gethostname(),project=str(ROOT),created=time.time(),protocol_sha256=v14.sha(HERE/'PROTOCOL.md')))
    return dest

def jobs(dest,phase):
    if phase=='development':
        old=json.loads((OLD16/'JOBS_LOCK.json').read_text())
        return [dict(j,phase=phase) for j in old if j['seed'] in (9161101,9161111,9161121)]
    if phase=='bridge':
        out=v14.jobs_for(dest,phase,(9172101,9172111,9172121))
        return [dict(j,input_sha256=v14.sha(j['input']),evaluation_sha256=v14.sha(j['evaluation'])) for j in out]
    out=[]
    for seed in range(9171101,9171201,10):
        for gap in (0.,2.,8.):
            for sigma in (1.,2.):
                for lam in ((0.,) if gap==0 else (0.,.5,1.)):
                    case=f's{seed}_g{gap:g}_n{sigma:g}_l{lam:g}'
                    inp,truth=gen.make(seed,gap,sigma,lam)
                    ip=dest/'inputs'/f'{case}.npz';ep=dest/'evaluation'/f'{case}.npz'
                    npz(ip,inp);npz(ep,truth)
                    out.append(dict(case=case,phase=phase,seed=seed,gap=gap,sigma=sigma,dependence=lam,n=len(inp['height_mm']),
                        input=str(ip),evaluation=str(ep),input_sha256=v14.sha(ip),evaluation_sha256=v14.sha(ep)))
    return out

def worker(job,dest):
    dest=Path(dest);started=time.perf_counter();records=[]
    try:
        assert v14.sha(job['input'])==job['input_sha256']
        inp=v14.load(job['input']);state=None
        if job['phase']=='bridge':
            state=v14.v7.freeze(inp['xyz_world'],inp['scan_id'],job['sigma'])
            x,y=state['design'],state['corrected']
        else:x,y=inp['design'],inp['height_mm']
        outputs,diagnostic=op.construct(x,y,job['sigma'])
        dp=dest/'diagnostics'/f'{job["case"]}.json';v14.save(dp,diagnostic)
        for method,m in outputs.items():
            art={k:np.asarray(m[k]) for k in ('k','means','slope','groups','gate','posterior','prediction')}
            out=inp['xyz_world'].copy()
            if state is None:out[:,2]=m['prediction']/1000.
            else:
                take=np.flatnonzero(state['support']);rows=state['order'][take]
                disp=m['prediction'][take]-state['local'][take,2]
                if method=='old_spatial36':out[rows]+=disp[:,None]*state['normal']/1000.
                else:out[rows]+=(disp/1000.)[:,None]*state['normal']
                art.update({k:state[k] for k in ('order','basis','center','normal','design','local','corrected','support','common_scale')})
                np.testing.assert_array_equal(out[state['order'][~state['support']]],inp['xyz_world'][state['order'][~state['support']]])
            assert out.shape==inp['xyz_world'].shape and np.isfinite(out).all()
            art['xyz_world']=out;path=dest/'outputs'/f'{job["case"]}__{method}.npz';npz(path,art)
            records.append(dict(**job,method=method,status='OK',output=str(path),output_sha256=v14.sha(path),
                selected_family=m.get('selected_family','S' if m['k']==1 else ('C' if method=='old_constant64' else 'R')),
                diagnostic=str(dp),diagnostic_sha256=v14.sha(dp)))
    except Exception:
        records=[dict(**job,method=m,status='FAILED',error=traceback.format_exc()) for m in op.METHODS]
    seconds=time.perf_counter()-started
    for r in records:r.update(shared_case_seconds=seconds,worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    v14.save(dest/'records'/f'{job["case"]}.json',records)
    return records

def score(dest,phase,records):
    rows=[]
    for r in records:
        row={k:r[k] for k in ('phase','case','seed','gap','sigma','method','status','shared_case_seconds')}
        for k in ('dependence','bias','retain','selected_family'):
            if k in r:row[k]=r[k]
        if r['status']=='OK':
            assert v14.sha(r['output'])==r['output_sha256']
            inp,truth,art=map(v14.load,(r['input'],r['evaluation'],r['output']))
            if phase!='bridge':row.update(metrics.score_new(inp,truth,art))
            else:
                truth.update(json.loads(truth.pop('json').tobytes().decode()))
                geometry=v14.synthetic_geometry(art['xyz_world'],dict(k=int(art['k'])),truth)
                layer=v14.layer_scores(art['xyz_world'],truth)
                row.update(geometry);row.update(layer)
                row.update(surface_mae_mm=layer['independent_surface_mae_mm'],balanced_source_mae_mm=layer['layer_balanced_source_surface_mae_mm'],
                    worst_source_mae_mm=layer['worst_layer_source_surface_mae_mm'],model_k=int(art['k']),
                    matched_rms_mm=float(1000*np.sqrt(np.mean(np.sum((art['xyz_world']-truth['gt_clean_xyz_world'])**2,axis=1)))))
                replay,_=metrics.score_replay(inp,truth,art,art,art['xyz_world']);row.update(replay)
                row['source_geometry_scope']='V17 new-seed synthetic first returns; frozen upstream; not real geometry'
                if not r['gap']:row['single_false_split']=int(art['k']==2)
        rows.append(row)
    v14.save(dest/f'{phase}_ROWS.json',rows);v14.csvsave(dest/f'{phase}_RESULTS.csv',rows)
    return rows

def freeze(dest,rows):
    candidates={}
    for method in ('adaptive_bic','adaptive_cv'):
        rr=[r for r in rows if r['method']==method];dual=[r for r in rr if r['gap'] and r['status']=='OK']
        failures=sum(r['status']!='OK' for r in rr);false=sum(r.get('single_false_split',0) for r in rr if not r['gap'])
        candidates[method]=dict(failures=failures,single_false_splits=false,eligible=not failures and not false,
            dual_balanced_source_mae_mm=float(np.mean([r['balanced_source_mae_mm'] for r in dual])))
    eligible=[m for m in candidates if candidates[m]['eligible']]
    primary=min(eligible,key=lambda m:candidates[m]['dual_balanced_source_mae_mm']) if eligible else None
    inference_files=[HERE/'v17_operator.py',ROOT/'exploration_v16/models.py',ROOT/'exploration_v15/constant_solver.py',ROOT/'exploration_v11/spatial_filter.py']
    v14.save(dest/'CONFIRMATION_LOCK.json',dict(primary=primary,development=candidates,
        sources={str(p):v14.sha(p) for p in inference_files},protocol_sha256=v14.sha(HERE/'PROTOCOL.md'),
        statement='Global selector choice uses exposed development GT only; no per-case test GT; both reported regardless.'))
    print('FROZEN',primary,candidates,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','confirmation','bridge']);p.add_argument('--dest');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();dest=Path(args.dest) if args.dest else create();print('RUN',dest,flush=True)
    if args.phase!='development':
        lock=json.loads((dest/'CONFIRMATION_LOCK.json').read_text())
        for path,sha in lock['sources'].items():assert v14.sha(path)==sha
        assert v14.sha(HERE/'PROTOCOL.md')==lock['protocol_sha256']
    jj=jobs(dest,args.phase);v14.save(dest/f'{args.phase}_JOBS_LOCK.json',jj)
    start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,j,str(dest)) for j in jj]
        for i,f in enumerate(as_completed(futures),1):
            records.extend(f.result())
            if i%5==0 or i==len(jj):print(args.phase,i,'/',len(jj),'failed',sum(r['status']!='OK' for r in records),flush=True)
    records.sort(key=lambda r:(r['case'],r['method']))
    v14.save(dest/f'{args.phase}_SEALED_BEFORE_GT.json',records)
    v14.save(dest/f'{args.phase}_TIMING.json',dict(wall_seconds=time.perf_counter()-start,workers=args.workers,
        parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        max_single_worker_peak_rss_kib=max(r['worker_peak_rss_kib'] for r in records),scope='fit/save phase wall; RSS are separate lifetime maxima, not aggregate; shared case pool cannot sum method times; CPU only'))
    rows=score(dest,args.phase,records)
    if args.phase=='development':freeze(dest,rows)
    print('DONE',args.phase,len(records),'outputs',flush=True)

if __name__=='__main__':main()
