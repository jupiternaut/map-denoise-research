"""Small spectral/filter experiment: frozen outputs before evaluation."""
import csv,hashlib,json,os,shutil,socket,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'exploration_v9'))
from run_v9 import RUNS,PARENT,INPUTS,read,save,npz,csvsave,memory,sha
from spectral_filter import filter_frozen,null_gain_threshold

def protect():
    before=json.loads((RUNS/'session-v9-bzr612se/BEFORE.json').read_text())
    for p,h in before.items():assert sha(p)==h,p
    for base in (ROOT/'exploration_v9',RUNS/'session-v9-bzr612se',RUNS/'crossfit-actions-v9-dcz583qe',
                 RUNS/'crossfit-actions-v9-wi65b5sr',RUNS/'selection-unmix-v9-qe05wnli'):
        for p in base.rglob('*'):
            if p.is_file():before[str(p)]=sha(p)
    return before

def main():
    assert socket.gethostname()=='liekkas'
    for k in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='1',k
    dest=Path(tempfile.mkdtemp(prefix='spectral-filter-v10-',dir=RUNS));print(dest,flush=True)
    for d in ('source','outputs'):(dest/d).mkdir()
    protected=protect();save(dest/'HISTORY_BEFORE.json',protected)
    sources=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md',ROOT/'evaluate_v2.py',ROOT/'exploration_v3/metrics.py']
    hashes={str(p):sha(p) for p in sources};save(dest/'SOURCES.json',hashes)
    for p in sources:
        target=dest/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    with (dest/'TESTS.log').open('x') as f:f.write(tests.stdout+tests.stderr)
    assert tests.returncode==0,tests.stderr
    start=time.perf_counter();records=[];calibrations={};memstart=memory()
    cases=sorted(p.stem for p in (PARENT/'states').glob('*.npz'));assert len(cases)==24
    for case in cases:
        statepath=PARENT/'states'/(case+'.npz');inputpath=INPUTS/(case+'.npz')
        state=read(statepath);inputs=read(inputpath)
        np.testing.assert_array_equal(state['world'],inputs['xyz_world'])
        counts=tuple(int(n) for n in np.unique(inputs['scan_id'],return_counts=True)[1])
        if counts not in calibrations:
            t=time.perf_counter();threshold,draws=null_gain_threshold(counts)
            calibrations[counts]=dict(counts=counts,threshold=threshold,seconds=time.perf_counter()-t,draws=draws)
        for iterations in (12,36):
            for method in ('pooled_em','spectrum_fixed'):
                out,info,art=filter_frozen(state,1.,method,iterations)
                target=dest/'outputs'/f'{case}__{method}_i{iterations}.npz'
                npz(target,scan_id=inputs['scan_id'],source_point_index=inputs['source_point_index'],**art)
                r=dict(case=case,method=f'{method}_i{iterations}',variant=method,iterations=iterations,
                   input=str(inputpath),input_sha256=sha(inputpath),state=str(statepath),state_sha256=sha(statepath),
                   output=str(target),output_sha256=sha(target),info=info)
                save(target.with_suffix('.json'),r);records.append(r)
        print(case,'4 outputs saved',flush=True)
    generation=time.perf_counter()-start
    save(dest/'SEALED_BEFORE_GT.json',records);save(dest/'NULL_CALIBRATIONS.json',list(calibrations.values()))
    print('All outputs saved; evaluation begins',flush=True)
    sys.path[:0]=[str(ROOT),str(ROOT/'exploration_v3')]
    from evaluate_v2 import synthetic_geometry
    from metrics import structure_metrics
    rows=[]
    for r in records:
        ev=read(INPUTS/'evaluation'/(r['case']+'.eval.npz'));ev.update(json.loads(ev.pop('json').tobytes().decode()))
        meta=json.loads((INPUTS/(r['case']+'.json')).read_text());data=read(r['output']);info=r['info']
        row={k:r[k] for k in ('case','method','variant','iterations','output','output_sha256')}
        row.update(gap_mm=meta['gap_mm'],seed=meta['seed'],bias_rms_mm=meta['bias_rms_mm'],
             chosen_k=info['k'],true_k=ev['n_true_layers'],model_gap_mm=info['fitted_gap_mm'],
             model_gap_error_mm=abs(info['fitted_gap_mm']-meta['gap_mm']),
             spectrum_gap_mm=info['spectrum']['gap_mm'] if info['spectrum'] else None,
             seconds=info['seconds'],supported_fraction=info['supported_fraction'])
        row.update(synthetic_geometry(data['xyz_world'],{},ev));row.update(structure_metrics(data['xyz_world'],ev));rows.append(row)
    csvsave(dest/'RESULTS.csv',rows);a={}
    metrics=('surface_accuracy_mean_mm','matched_point_rms_mm','source_group_gap_error_mm',
             'fitted_gap_at_same_xy_error_mm','model_gap_error_mm','seconds','reference_sample_coverage_1mm')
    for label,gaps in (('all',(0,2,4,8)),('single',(0,)),('gap2',(2,)),('gap4',(4,)),('gap8',(8,))):
        a[label]={}
        for method in sorted({r['method'] for r in rows}):
            part=[r for r in rows if r['method']==method and r['gap_mm'] in gaps]
            a[label][method]=dict(n=len(part),wrong_k=sum(r['chosen_k']!=r['true_k'] for r in part),
                 **{k:float(np.mean([r[k] for r in part if k in r])) for k in metrics if any(k in r for r in part)})
    save(dest/'AGGREGATES.json',a)
    after={p:sha(p) for p in protected};assert after==protected
    assert {p:sha(p) for p in hashes}==hashes
    save(dest/'HISTORY_VERIFICATION.json',dict(files=len(protected),unchanged=True,missing=[],changed=[]))
    save(dest/'SUMMARY.json',dict(conditions=24,seeds=3,outputs=len(records),all_outputs_before_gt=True,new_confirmation=False,
        generation_with_io_seconds=generation,total_seconds=time.perf_counter()-start,
        null_calibration_seconds=sum(v['seconds'] for v in calibrations.values()),
        memory_start_KiB=memstart,memory_end_KiB=memory(),historical_files_unchanged=len(protected),
        timing_scope='cached upstream; calibration included in generation, method seconds are warmed-null-cache'))
    print(json.dumps(a['all'],indent=2),flush=True)
if __name__=='__main__':main()
