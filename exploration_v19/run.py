"""Shared V19 driver. Estimators receive observations/cache, never evaluation arrays."""
from pathlib import Path
import os,sys,json,tempfile,socket,time,argparse,traceback,resource,shutil,subprocess
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from common import HERE,ROOT,OLD,v14,hist,v18,sha,save,load,npz,make,compact,unpack,score,import_path

def create():
    assert socket.gethostname()=='liekkas'
    dest=Path(tempfile.mkdtemp(prefix='parallel-v19-',dir=v14.RUNS))
    for d in ('inputs','evaluation','cache','outputs','records','source'):(dest/d).mkdir()
    tracked=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().rstrip('\0').split('\0')
    save(dest/'CHECKPOINT_BEFORE.json',{str(ROOT/p):sha(ROOT/p) for p in tracked})
    save(dest/'V18_BEFORE.json',hist.helper.hashes((OLD,)))
    save(dest/'IDENTITY.json',dict(host=socket.gethostname(),project=str(ROOT),python=sys.executable,created=time.time()))
    print('RUN',dest,flush=True);return dest

def lock(dest):
    files=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.md')]
    files += [ROOT/f'exploration_v{i}/{f}' for i,f in ((18,'v18_operator.py'),(17,'v17_operator.py'),(16,'models.py'),(15,'constant_solver.py'),(11,'spatial_filter.py'))]
    sources={str(p):sha(p) for p in files}
    save(dest/'CONFIRMATION_LOCK.json',sources)
    for p in files:
        q=dest/'source'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)

def check_lock(dest):
    for p,h in json.loads((dest/'CONFIRMATION_LOCK.json').read_text()).items():assert sha(p)==h,p

def jobs(dest,phase):
    out=[];seeds=(9190101,) if phase=='development' else (9191101,9191111,9191121,9191131)
    for seed in seeds:
        for gap in (0.,2.,8.):
            for sig in (1.,2.):
                for ratio in ((.5,) if gap==0 else (.2,.5,.8)):
                    for dep in ((0.,) if gap==0 else (0.,1.)):
                        case=f's{seed}_g{gap:g}_n{sig:g}_p{ratio:g}_l{dep:g}'
                        inp,truth=make(seed,gap,sig,ratio,dep)
                        ip=dest/'inputs'/f'{case}.npz';ep=dest/'evaluation'/f'{case}.npz';npz(ip,inp);npz(ep,truth)
                        j=dict(phase=phase,case=case,seed=seed,gap=gap,sigma=sig,supplied_sigma=sig,ratio=ratio,dependence=dep,
                               sigma_factor=1.,kind='statistical',input=str(ip),evaluation=str(ep))
                        out.append(j)
                        if phase=='confirmation' and gap==2 and sig==2 and ratio in (.2,.5):
                            for factor in (.8,1.2):out.append(dict(j,case=case+f'_sf{factor}',supplied_sigma=sig*factor,sigma_factor=factor,kind='sigma_stress'))
    if phase=='development':
        old=json.loads((OLD/'bridge_SEALED_BEFORE_GT.json').read_text())
        for r in old:
            if r['method']=='retained_map' and r['seed']==9182101:
                out.append(dict(**{k:r[k] for k in ('case','seed','gap','sigma','bias','retain','input','evaluation')},
                    phase=phase,kind='bridge',supplied_sigma=r['sigma'],sigma_factor=1.,cached_diagnostic=r['diagnostic']))
    else:
        for j in v14.jobs_for(dest,'confirmation',(9192101,9192111)):
            out.append(dict(j,kind='bridge',supplied_sigma=j['sigma'],sigma_factor=1.))
    for j in out:
        j.update(input_sha256=sha(j['input']),evaluation_sha256=sha(j['evaluation']))
    return out

def cache_worker(j,dest):
    start=time.perf_counter();inp=load(j['input']);state=None
    assert sha(j['input'])==j['input_sha256']
    if j['kind']=='bridge':
        raw=v14.v7.freeze(inp['xyz_world'],inp['scan_id'],j['supplied_sigma'])
        x,y=raw['design'],raw['corrected']
        state={k:raw[k] for k in ('order','basis','center','normal','local','support','common_scale')}
    else:x,y=inp['design'],inp['height_mm']
    if 'cached_diagnostic' in j:
        d=json.loads(Path(j['cached_diagnostic']).read_text());m=v18.arrays(d['retained'])
    else:
        _,d=v18.construct(x,y,j['supplied_sigma']);m=d['retained']
    target=Path(dest)/'cache'/f'{j["case"]}.json'
    save(target,dict(x=x,y=y,sigma=j['supplied_sigma'],baseline=compact(m),state=state,truth_fields_used=[]))
    return dict(j,cache=str(target),cache_sha256=sha(target),fit_seconds=time.perf_counter()-start)

def prepare(dest,phase,workers):
    if phase=='confirmation':check_lock(dest)
    jj=jobs(dest,phase);start=time.perf_counter();out=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(cache_worker,j,str(dest)) for j in jj]),1):
            out.append(f.result())
            if i%20==0 or i==len(jj):print('cache',phase,i,'/',len(jj),flush=True)
    out.sort(key=lambda j:j['case']);save(dest/f'{phase}_JOBS.json',out)
    save(dest/f'{phase}_CACHE_TIMING.json',dict(wall_seconds=time.perf_counter()-start,workers=workers,n=len(out)))

def work(j,dest,track):
    start=time.perf_counter();c,x,y,baseline=unpack(j)
    try:
        module=import_path('track_'+track,HERE/f'track_{track}/operator.py')
        results=module.run(x,y,c['sigma'],baseline)
        if track=='b':results['identity']=dict(identity=True,k=0,means=[],slope=[0.,0.],representation='points')
        records=[]
        for name,m in results.items():
            path=Path(dest)/'outputs'/f'{j["case"]}__{track}_{name}.json';save(path,m)
            records.append(dict(case=j['case'],method=f'{track}_{name}',track=track,status='OK',output=str(path),output_sha256=sha(path)))
    except Exception:
        records=[dict(case=j['case'],method=track+'_FAILED',track=track,status='FAILED',error=traceback.format_exc())]
    for r in records:r.update(seconds=time.perf_counter()-start,worker_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(Path(dest)/'records'/f'{j["case"]}__{track}.json',records);return records

def run(dest,phase,track,workers):
    if phase=='confirmation':check_lock(dest)
    jj=json.loads((dest/f'{phase}_JOBS.json').read_text());start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,j,str(dest),track) for j in jj]),1):
            records.extend(f.result())
            if i%10==0 or i==len(jj):print(track,phase,i,'/',len(jj),'failed',sum(r['status']!='OK' for r in records),flush=True)
    records.sort(key=lambda r:(r['case'],r['method']))
    save(dest/f'{phase}_{track}_SEALED.json',records)
    save(dest/f'{phase}_{track}_TIMING.json',dict(wall_seconds=time.perf_counter()-start,workers=workers,
         parent_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,max_single_worker_rss_kib=max(r['worker_rss_kib'] for r in records),
         scope='cached baseline excluded; lifetime RSS separate, not summed peak; CPU only'))
    # Evaluation starts only after every output of this line has been sealed.
    jobmap={j['case']:j for j in jj};rows=[]
    for r in records:
        j=jobmap[r['case']];row={k:v for k,v in j.items() if k in ('case','kind','phase','seed','gap','sigma','supplied_sigma','sigma_factor','ratio','dependence','bias','retain')}
        row.update({k:r[k] for k in ('method','status','seconds')})
        if r['status']=='OK':
            assert sha(r['output'])==r['output_sha256'];assert sha(j['evaluation'])==j['evaluation_sha256']
            row.update(score(j,json.loads(Path(r['output']).read_text())))
        else:row['error']=r['error']
        rows.append(row)
    save(dest/f'{phase}_{track}_ROWS.json',rows);v14.csvsave(dest/f'{phase}_{track}_RESULTS.csv',rows)
    print('DONE',phase,track,len(rows),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('create','prepare','lock','run'));p.add_argument('--dest');p.add_argument('--phase',default='development',choices=('development','confirmation'));p.add_argument('--track',choices=('a','b'));p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.action=='create':create();return
    dest=Path(a.dest)
    if a.action=='lock':lock(dest)
    elif a.action=='prepare':prepare(dest,a.phase,a.workers)
    else:run(dest,a.phase,a.track,a.workers)
if __name__=='__main__':main()
