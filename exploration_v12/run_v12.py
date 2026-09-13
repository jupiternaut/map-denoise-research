import argparse,csv,importlib.util,json,shutil,socket,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'exploration_v9'),str(ROOT/'exploration_v10'),str(ROOT/'exploration_v11'),str(ROOT),str(ROOT/'exploration_v3')]
from run_v9 import RUNS,read,save,npz,csvsave,sha,memory
from run_v10 import protect
from spatial_filter import filter_frozen
from local_filter import estimate_frozen as local
from external import estimate_frozen as external

def main(fresh=False):
    assert socket.gethostname()=='liekkas'
    dest=Path(tempfile.mkdtemp(prefix='local-v12-'+('confirmation-' if fresh else 'development-'),dir=RUNS));print(dest,flush=True)
    for d in ('outputs','source','states'):(dest/d).mkdir()
    history=protect()
    for base in (ROOT/'exploration_v10',ROOT/'exploration_v11',RUNS/'spatial-v11-checkpoint-49g5y2ej'):
        history.update({str(p):sha(p) for p in base.rglob('*') if p.is_file()})
    save(dest/'HISTORY_BEFORE.json',history)
    hashes={str(p):sha(p) for p in HERE.iterdir() if p.is_file()};save(dest/'SOURCES.json',hashes)
    for p in hashes:shutil.copyfile(p,dest/'source'/Path(p).name)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    (dest/'TESTS.log').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stderr
    spec=importlib.util.spec_from_file_location('_v12_v7',ROOT/'exploration_v7/algorithm/reassociation.py');v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)
    jobs=[];base=RUNS/'spatial-v11-confirmation-xjro7mgm'
    if fresh:
        from generate_synthetics import make_ghost,make_dual
        from schema import write_patch
        base=dest
        for seed in (9132011,9132021,9132033):
            for gap in (0,2,4,8):
                for bias in (0,4):
                    case=f'ghost_s{seed}_b{bias}' if gap==0 else f'dual_g{gap}_s{seed}_b{bias}'
                    points,meta,ev=make_ghost(seed,bias) if gap==0 else make_dual(seed,gap,bias)
                    write_patch(base/'inputs',case,points,meta,ev)
                    state=v7.freeze(points['xyz_world'],points['scan_id'],1.)
                    npz(base/'states'/(case+'.npz'),**{k:v for k,v in state.items() if isinstance(v,np.ndarray)})
        del points,meta,ev,state
    for p in sorted((base/'inputs').glob('*.npz')):
        jobs.append(dict(case=p.stem,domain='synthetic',input=str(p),state=str(base/'states'/(p.stem+'.npz')),evaluation=str(base/'inputs/evaluation'/(p.stem+'.eval.npz'))))
    previous=json.loads((RUNS/'spatial-v11-real-6tqt9xxy/SEALED_BEFORE_SCORING.json').read_text())
    for r in ([] if fresh else previous):
        if r['method']=='identity':jobs.append(dict(case=r['case']+'_'+r['condition'],patch=r['case'],condition=r['condition'],domain='real',input=r['input'],original=r['original_input']))
    save(dest/'JOBS.json',jobs);records=[];start=time.perf_counter()
    for job in jobs:
        data=read(job['input']);world=data['xyz_world']
        if job['domain']=='synthetic':state=read(job['state']);scans=data['scan_id']
        else:
            scans=read(job['original'])['scan_id'];state=v7.freeze(world,scans,1.)
            sp=dest/'states'/(job['case']+'.npz');npz(sp,**{k:v for k,v in state.items() if isinstance(v,np.ndarray)});job['state']=str(sp)
        methods=['identity','old_original','old_multistart','whole_spatial']+[f'local_{b}_{k}' for b in ('plane','spatial') for k in (96,256)]+[f'{b}_{s}' for b in ('apss','rimls') for s in (1,2)]+['raw_apss_2','raw_rimls_2']
        for method in methods:
            t=time.perf_counter()
            try:
                if method=='identity':out=world.copy();info={}
                elif method.startswith('old_'):
                    # Reconstruct the full upstream for old methods on synthetic cached states.
                    if 'v4_info' not in state:oldstate=v7.freeze(world,scans,1.)
                    else:oldstate=state
                    t=time.perf_counter();out,info,_=v7.fit_frozen(oldstate,variant='original' if method=='old_original' else 'local_multistart',budget=0 if method=='old_original' else 6,sharing='independent')
                elif 'design' not in state:out=world.copy();info=dict(status='UNSUPPORTED')
                elif method=='whole_spatial':out,info,_=filter_frozen(state,1.,'spatial_free',12)
                elif method.startswith('local_'):
                    _,backend,k=method.split('_');out,info,_=local(state,1.,int(k),backend)
                else:
                    fields=method.split('_');out,info,_=external(state,fields[-2],float(fields[-1]),raw=fields[0]=='raw')
                elapsed=time.perf_counter()-t;path=dest/'outputs'/f'{job["case"]}__{method}.npz';npz(path,xyz_world=out)
                record=dict(**job,method=method,output=str(path),output_sha256=sha(path),input_sha256=sha(job['input']),seconds=elapsed,info=info,status='OK')
            except Exception as e:record=dict(**job,method=method,status='FAILED',error=repr(e))
            records.append(record)
        print(job['case'],'sealed',flush=True)
    save(dest/'SEALED_BEFORE_GT.json',records)
    from evaluate_v2 import synthetic_geometry
    from metrics import structure_metrics
    rows=[]
    for r in records:
        if r['status']!='OK':continue
        out=read(r['output'])['xyz_world'];row={k:r[k] for k in ('case','domain','method','output','output_sha256','seconds')}
        if r['domain']=='synthetic':
            ev=read(r['evaluation']);ev.update(json.loads(ev.pop('json').tobytes().decode()))
            row['gap_mm']=ev['true_gap_mm'];row.update(synthetic_geometry(out,{},ev));row.update(structure_metrics(out,ev))
        else:
            ref=read(r['original'])['xyz_world'];current=read(r['input'])['xyz_world']
            row.update(condition=r['condition'],patch=r['patch'],measured_reference_rms_mm=float(np.sqrt(np.mean(np.sum((out-ref)**2,axis=1)))*1000),input_edit_rms_mm=float(np.sqrt(np.mean(np.sum((out-current)**2,axis=1)))*1000))
        rows.append(row)
    csvsave(dest/'RESULTS.csv',rows);aggregate={}
    for group in ('synthetic','real_zero','real_shift3mm'):
        aggregate[group]={}
        for method in methods:
            part=[r for r in rows if r['method']==method and (r['domain']=='synthetic' if group=='synthetic' else r['domain']=='real' and r['condition']==group[5:])]
            keys=('surface_accuracy_mean_mm','matched_point_rms_mm','source_group_gap_error_mm','reference_sample_coverage_1mm','measured_reference_rms_mm','seconds')
            aggregate[group][method]={k:float((np.mean if group=='synthetic' else np.median)([r[k] for r in part if k in r])) for k in keys if any(k in r for r in part)}
    save(dest/'AGGREGATES.json',aggregate)
    assert all(sha(p)==h for p,h in history.items());assert all(sha(p)==h for p,h in hashes.items())
    save(dest/'SUMMARY.json',dict(jobs=len(jobs),outputs=len(rows),fresh_seeds=fresh,failures=[r for r in records if r['status']!='OK'],elapsed_seconds=time.perf_counter()-start,memory_KiB=memory(),historical_files_unchanged=len(history)))
    print(json.dumps(aggregate,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fresh',action='store_true');main(parser.parse_args().fresh)
