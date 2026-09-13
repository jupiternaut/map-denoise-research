"""V9: preserve all candidates, seal outputs before evaluating geometry."""
import csv,hashlib,json,os,shutil,socket,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
from action_ablation import action_frozen
from crossfit import candidate_search,project_candidate

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
RUNS=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
PARENT=RUNS/'reassociation-v7-development-vbwoojgf'
INPUTS=RUNS/'repair-v2-oyuie4pl/synthetics/identifiable'
PREVIOUS=RUNS/'selection-compensation-v8-ey291bta'
BUDGETS=(1,3,6)
METRICS=('surface_accuracy_mean_mm','matched_point_rms_mm','within_source_layer_rms_mm',
    'source_group_gap_mm','source_group_gap_error_mm','fitted_gap_at_same_xy_error_mm',
    'source_surface_tilt_mean_deg','reference_sample_coverage_1mm','input_edit_rms_mm')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(p,x):
    with p.open('x') as f:json.dump(clean(x),f,indent=2,allow_nan=False)
def npz(p,**x):
    with p.open('xb') as f:np.savez_compressed(f,**x)
def csvsave(p,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
def memory():
    return {l.split()[0].rstrip(':'):int(l.split()[1]) for l in Path('/proc/self/status').read_text().splitlines()
            if l.startswith(('VmHWM:','VmRSS:'))}

def main():
    assert socket.gethostname()=='liekkas'
    for k in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='1',k
    dest=Path(tempfile.mkdtemp(prefix='crossfit-actions-v9-',dir=RUNS));print(dest,flush=True)
    for d in ('source','candidates','outputs'):(dest/d).mkdir()
    sources=sorted(HERE.glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'MATH_NOTE.md',PROJECT/'exploration_v8/compensation.py',
       PROJECT/'exploration_v7/algorithm/reassociation.py',PROJECT/'evaluate_v2.py',PROJECT/'exploration_v3/metrics.py']
    sourcehash={str(p):sha(p) for p in sources};save(dest/'SOURCES.json',sourcehash)
    for p in sources:
        target=dest/'source'/p.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    with (dest/'TESTS.log').open('x') as f:f.write(tests.stdout+tests.stderr)
    assert tests.returncode==0,tests.stderr
    math=subprocess.run([sys.executable,str(HERE/'math_diagnosis.py')],capture_output=True,text=True)
    assert math.returncode==0,math.stderr
    save(dest/'MATH_DIAGNOSIS.json',json.loads(math.stdout))
    started=time.perf_counter();mem_start=memory();protected={};records=[];search_records=[];replays=[]
    cases=sorted(p.stem for p in (PARENT/'states').glob('*.npz'));assert len(cases)==24
    for case in cases:
        statepath=PARENT/'states'/(case+'.npz');inputpath=INPUTS/(case+'.npz')
        seedpath=PARENT/'outputs'/f'{case}__original_b0_independent.npz'
        state=read(statepath);inputs=read(inputpath);seed=read(seedpath)
        np.testing.assert_array_equal(state['world'],inputs['xyz_world'])
        for p in (statepath,inputpath,seedpath):protected[str(p)]=sha(p)
        for budget in BUDGETS:
            frozenpath=PARENT/'outputs'/f'{case}__reassociate_b{budget}_independent.npz'
            frozen=read(frozenpath);protected[str(frozenpath)]=sha(frozenpath)
            for stage in ('frozen','alltrain','crossfit'):
                if stage=='frozen':
                    candidate=frozen;candidatepath=frozenpath
                    search=dict(seconds=0.,scope='reused V7 search; cached action only',fallback_count=0)
                    actions=('none','constant','slope','full')
                else:
                    candidate,search=candidate_search(state,seed,1.,budget=budget,folds=1 if stage=='alltrain' else 2)
                    candidatepath=dest/'candidates'/f'{case}__{stage}_b{budget}.npz'
                    npz(candidatepath,**candidate)
                    search_records.append(dict(case=case,stage=stage,path=str(candidatepath),
                                              sha256=sha(candidatepath),**search))
                    actions=('none','constant','full','candidate')
                for action in actions:
                    if action=='candidate':out,info,art=project_candidate(state,candidate)
                    else:out,info,art=action_frozen(state,candidate,1.,mode=action)
                    if stage=='frozen' and action in ('none','full'):
                        if action=='none':ref=frozen['xyz_world']
                        else:
                            oldpath=PREVIOUS/'outputs'/f'{case}__b{budget}_conditional.npz'
                            protected[str(oldpath)]=sha(oldpath);ref=read(oldpath)['xyz_world']
                        error=float(np.max(abs(out-ref))*1000);assert error<1e-8
                        replays.append(dict(case=case,budget=budget,action=action,max_error_mm=error))
                    target=dest/'outputs'/f'{case}__{stage}_b{budget}_{action}.npz'
                    npz(target,xyz_world=out,scan_id=inputs['scan_id'],source_point_index=inputs['source_point_index'],**art)
                    r=dict(case=case,budget=budget,stage=stage,action=action,method=f'{stage}_b{budget}_{action}',
                        output=str(target),output_sha256=sha(target),state=str(statepath),state_sha256=sha(statepath),
                        candidate=str(candidatepath),candidate_sha256=sha(candidatepath),input=str(inputpath),input_sha256=sha(inputpath),
                        search=search,info=info,candidate_parameter_count=int(candidate['coefficients'].size))
                    save(target.with_suffix('.json'),r);records.append(r)
        print(case,'36/36 sealed',flush=True)
    generation=time.perf_counter()-started
    save(dest/'SEALED_BEFORE_GT.json',records);save(dest/'SEARCH_RECORDS.json',search_records)
    save(dest/'PROTECTED.json',protected);save(dest/'REPLAYS.json',replays)
    print(len(records),'outputs sealed; GT evaluation starts',flush=True)
    sys.path[:0]=[str(PROJECT),str(PROJECT/'exploration_v3')]
    from evaluate_v2 import synthetic_geometry
    from metrics import structure_metrics
    rows=[]
    for r in records:
        ev=read(INPUTS/'evaluation'/(r['case']+'.eval.npz'));ev.update(json.loads(ev.pop('json').tobytes().decode()))
        meta=json.loads((INPUTS/(r['case']+'.json')).read_text());data=read(r['output']);world=read(r['input'])['xyz_world']
        row={k:r[k] for k in ('case','budget','stage','action','method','output','output_sha256')}
        row.update(gap_mm=meta['gap_mm'],seed=meta['seed'],bias_rms_mm=meta['bias_rms_mm'],
            search_seconds=r['search']['seconds'],action_seconds=r['info']['seconds'],
            cached_total_seconds=r['search']['seconds']+r['info']['seconds'],fallback_count=r['search']['fallback_count'],
            group_count=int(len(np.unique(data['group_ids'][data['group_ids']>=0]))),
            candidate_parameter_count=r['candidate_parameter_count'],
            final_fit_parameter_count=r['info'].get('fit_parameter_count',0),final_fit_rank=r['info'].get('fit_rank',0),
            supported_fraction=float(data['support_mask'].mean()),
            actual_changed_fraction=float(np.mean(np.any(data['xyz_world']!=world,axis=1))),
            input_edit_rms_mm=float(1000*np.sqrt(np.mean(np.sum((data['xyz_world']-world)**2,axis=1)))))
        row.update(synthetic_geometry(data['xyz_world'],{},ev));row.update(structure_metrics(data['xyz_world'],ev))
        rows.append(row)
    csvsave(dest/'RESULTS.csv',rows)
    aggregates={}
    for label,gaps in (('all',(0,2,4,8)),('single',(0,)),('dual',(2,4,8)),('gap2',(2,)),('gap4',(4,)),('gap8',(8,))):
        aggregates[label]={}
        for method in sorted(set(r['method'] for r in rows)):
            part=[r for r in rows if r['method']==method and r['gap_mm'] in gaps]
            aggregates[label][method]=dict(n=len(part),**{m:float(np.mean([r[m] for r in part if m in r]))
                for m in (*METRICS,'cached_total_seconds','supported_fraction','actual_changed_fraction') if any(m in r for r in part)})
    save(dest/'AGGREGATES.json',aggregates)
    for p,h in {**protected,**sourcehash}.items():assert sha(p)==h,p
    summary=dict(run=str(dest),host=socket.gethostname(),conditions=len(cases),seeds=3,outputs=len(records),
        new_confirmation=False,original_and_v8_replays=len(replays),generation_with_io_seconds=generation,
        total_with_eval_seconds=time.perf_counter()-started,memory_start_KiB=mem_start,memory_end_KiB=memory(),
        fallback_groups=sum(r['fallback_count'] for r in search_records),protected_files=len(protected),
        all_outputs_before_gt=True,protected_and_sources_unchanged=True,
        timing_scope='cached upstream, includes new candidate training/action and IO/eval, excludes prior V7 upstream')
    save(dest/'SUMMARY.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
