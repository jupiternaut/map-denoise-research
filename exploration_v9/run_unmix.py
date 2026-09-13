"""Additional predeclared moment-unmixing repair on the same legal inputs."""
import os,shutil,socket,subprocess,sys,tempfile,time,json
from pathlib import Path
import numpy as np
from run_v9 import HERE,PROJECT,RUNS,PARENT,INPUTS,BUDGETS,METRICS,sha,read,save,npz,csvsave,memory
from unmix import unmix_frozen

def main():
    assert socket.gethostname()=='liekkas'
    for k in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='1',k
    dest=Path(tempfile.mkdtemp(prefix='selection-unmix-v9-',dir=RUNS));print(dest,flush=True)
    for d in ('source','outputs'):(dest/d).mkdir()
    sources=[HERE/n for n in ('unmix.py','test_unmix.py','run_unmix.py','run_v9.py','UNMIX_PROTOCOL.md')]
    sources.extend([PROJECT/'exploration_v8/compensation.py',PROJECT/'evaluate_v2.py',PROJECT/'exploration_v3/metrics.py'])
    hashes={str(p):sha(p) for p in sources};save(dest/'SOURCES.json',hashes)
    for p in sources:
        target=dest/'source'/p.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_unmix.py','-v'],capture_output=True,text=True)
    with (dest/'TESTS.log').open('x') as f:f.write(tests.stdout+tests.stderr)
    assert tests.returncode==0,tests.stderr
    start=time.perf_counter();memstart=memory();records=[];protected={}
    cases=sorted(p.stem for p in (PARENT/'states').glob('*.npz'));assert len(cases)==24
    for case in cases:
        statepath=PARENT/'states'/(case+'.npz');inputpath=INPUTS/(case+'.npz')
        state=read(statepath);inputs=read(inputpath)
        for p in (statepath,inputpath):protected[str(p)]=sha(p)
        np.testing.assert_array_equal(state['world'],inputs['xyz_world'])
        for budget in BUDGETS:
            candidatepath=PARENT/'outputs'/f'{case}__reassociate_b{budget}_independent.npz'
            candidate=read(candidatepath);protected[str(candidatepath)]=sha(candidatepath)
            for mode in ('offset','affine'):
                out,info,art=unmix_frozen(state,candidate,1.,mode=mode)
                assert np.isfinite(out).all();target=dest/'outputs'/f'{case}__b{budget}_{mode}.npz'
                npz(target,xyz_world=out,scan_id=inputs['scan_id'],source_point_index=inputs['source_point_index'],**art)
                r=dict(case=case,budget=budget,mode=mode,method=f'b{budget}_{mode}',
                    output=str(target),output_sha256=sha(target),state=str(statepath),input=str(inputpath),
                    candidate=str(candidatepath),candidate_sha256=sha(candidatepath),info=info)
                save(target.with_suffix('.json'),r);records.append(r)
        print(case,'6/6 saved',flush=True)
    generation=time.perf_counter()-start
    save(dest/'SEALED_BEFORE_GT.json',records);save(dest/'PROTECTED.json',protected)
    print(len(records),'outputs sealed; evaluation begins',flush=True)
    sys.path[:0]=[str(PROJECT),str(PROJECT/'exploration_v3')]
    from evaluate_v2 import synthetic_geometry
    from metrics import structure_metrics
    rows=[]
    for r in records:
        ev=read(INPUTS/'evaluation'/(r['case']+'.eval.npz'));ev.update(json.loads(ev.pop('json').tobytes().decode()))
        meta=json.loads((INPUTS/(r['case']+'.json')).read_text());data=read(r['output']);world=read(r['input'])['xyz_world']
        row={k:r[k] for k in ('case','budget','mode','method','output','output_sha256')}
        row.update(gap_mm=meta['gap_mm'],seed=meta['seed'],bias_rms_mm=meta['bias_rms_mm'],
                   supported_fraction=float(data['support_mask'].mean()),
                   input_edit_rms_mm=float(1000*np.sqrt(np.mean(np.sum((data['xyz_world']-world)**2,axis=1)))))
        row.update({k:v for k,v in r['info'].items() if isinstance(v,(int,float,str)) and k not in row})
        row.update(synthetic_geometry(data['xyz_world'],{},ev));row.update(structure_metrics(data['xyz_world'],ev));rows.append(row)
    csvsave(dest/'RESULTS.csv',rows);aggregates={}
    for label,gaps in (('all',(0,2,4,8)),('single',(0,)),('dual',(2,4,8)),('gap2',(2,)),('gap4',(4,)),('gap8',(8,))):
        aggregates[label]={}
        for method in sorted(set(r['method'] for r in rows)):
            part=[r for r in rows if r['method']==method and r['gap_mm'] in gaps]
            aggregates[label][method]=dict(n=len(part),**{k:float(np.mean([r[k] for r in part if k in r]))
                for k in (*METRICS,'seconds') if any(k in r for r in part)})
    save(dest/'AGGREGATES.json',aggregates)
    for p,h in {**hashes,**protected}.items():assert sha(p)==h,p
    report=dict(run=str(dest),outputs=len(records),conditions=24,seeds=3,new_confirmation=False,
       all_outputs_before_gt=True,generation_with_io_seconds=generation,total_with_eval_seconds=time.perf_counter()-start,
       protected_files=len(protected),protected_unchanged=True,memory_start_KiB=memstart,memory_end_KiB=memory(),
       timing_scope='cached candidate action; excludes original upstream and candidate search')
    save(dest/'SUMMARY.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
