"""Seal all legal compensation outputs before loading evaluation truth."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import numpy as np
from compensation import compensate_frozen

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
RUNS=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
PARENT=RUNS/'reassociation-v7-development-vbwoojgf'
SOFT=RUNS/'decision-v7-nk7sm4yf'
INPUTS=RUNS/'repair-v2-oyuie4pl/synthetics/identifiable'
MODES=('none','conditional','random');BUDGETS=(1,3,6)
METRICS=('surface_accuracy_mean_mm','matched_point_rms_mm','fitted_gap_at_same_xy_error_mm',
         'within_source_layer_rms_mm')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def payload(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def save(p,v):
    with Path(p).open('x') as f:json.dump(clean(v),f,indent=2,ensure_ascii=False,allow_nan=False)
def csv_save(p,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def memory():
    result={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmHWM:','VmRSS:')):
            k,v,_=line.split();result[k.rstrip(':')+'_KiB']=int(v)
    return result

def main():
    if socket.gethostname()!='liekkas':raise RuntimeError('wrong target host')
    for k in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(k)!='1':raise RuntimeError(k+' must be 1')
    cases=sorted(p.stem for p in (PARENT/'states').glob('*.npz'))
    assert len(cases)==24
    dest=Path(tempfile.mkdtemp(prefix='selection-compensation-v8-',dir=RUNS));print(dest,flush=True)
    for name in ('source','outputs'):(dest/name).mkdir()
    sources=[*sorted(HERE.glob('*.py')),HERE/'PROTOCOL.md',PROJECT/'evaluate_v2.py',PROJECT/'exploration_v3/metrics.py']
    source_hashes={str(p):digest(p) for p in sources};save(dest/'SOURCE_MANIFEST.json',source_hashes)
    for p in sources:
        target=dest/'source'/p.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    with (dest/'TESTS.log').open('x') as f:f.write(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError('tests failed')
    started=time.perf_counter();protected={};records=[];replay=[];initial_memory=memory()
    # Pure generation phase: no evaluation path or previous score table is opened.
    for case in cases:
        state_path=PARENT/'states'/(case+'.npz');input_path=INPUTS/(case+'.npz')
        state=payload(state_path);inputs=payload(input_path)
        np.testing.assert_array_equal(state['world'],inputs['xyz_world'])
        for p in (state_path,input_path):protected[str(p)]=digest(p)
        for budget in BUDGETS:
            a_path=PARENT/'outputs'/f'{case}__reassociate_b{budget}_independent.npz'
            artifacts=payload(a_path);protected[str(a_path)]=digest(a_path)
            for mode in MODES:
                out,info,a=compensate_frozen(state,artifacts,1.,mode=mode)
                if mode=='none':
                    error=float(np.max(abs(out-artifacts['xyz_world']))*1000.)
                    assert error<1e-8;replay.append(dict(case=case,budget=budget,max_error_mm=error))
                target=dest/'outputs'/f'{case}__b{budget}_{mode}.npz'
                with target.open('xb') as f:np.savez_compressed(f,xyz_world=out,scan_id=inputs['scan_id'],
                         source_point_index=inputs['source_point_index'],**a)
                record=dict(case=case,budget=budget,mode=mode,output=str(target),output_sha256=digest(target),
                            state=str(state_path),state_sha256=digest(state_path),original_artifacts=str(a_path),
                            original_artifacts_sha256=digest(a_path),input=str(input_path),input_sha256=digest(input_path),info=info)
                save(target.with_suffix('.json'),record);records.append(record)
        print(case,'9/9 generated',flush=True)
    generation_seconds=time.perf_counter()-started
    save(dest/'OUTPUTS_SEALED_BEFORE_GT.json',records);save(dest/'PROTECTED_BEFORE.json',protected)
    save(dest/'NONE_REPLAY.json',replay)
    print('216 outputs sealed; evaluation starts',flush=True)
    sys.path[:0]=[str(PROJECT),str(PROJECT/'exploration_v3')]
    from evaluate_v2 import synthetic_geometry
    from metrics import structure_metrics
    old={(r['case'],r['method']):r for r in csv.DictReader((PARENT/'RESULTS.csv').open())}
    soft={(r['case'],int(r['budget'])):r for r in csv.DictReader((SOFT/'RESULTS.csv').open())}
    rows=[];pairs=[]
    for record in records:
        case,budget,mode=record['case'],record['budget'],record['mode']
        evpath=INPUTS/'evaluation'/(case+'.eval.npz')
        with np.load(evpath,allow_pickle=False) as f:
            ev={k:f[k].copy() for k in f.files if k!='json'}
            if 'json' in f:ev.update(json.loads(f['json'].tobytes().decode()))
        meta=json.loads((INPUTS/(case+'.json')).read_text())
        data=payload(record['output']);world=payload(record['input'])['xyz_world'];out=data['xyz_world']
        row=dict(case=case,budget=budget,mode=mode,method=f'b{budget}_{mode}',gap_mm=meta['gap_mm'],seed=meta['seed'],
             bias_rms_mm=meta['bias_rms_mm'],output=record['output'],output_sha256=record['output_sha256'],
             stage='exposed development, cached conditional action',supported_fraction=float(data['support_mask'].mean()),
             postprocess_seconds=record['info']['seconds'],mean_abs_subtracted_noise_mm=record['info']['mean_absolute_subtracted_noise_mm'],
             max_abs_subtracted_noise_mm=record['info']['max_absolute_subtracted_noise_mm'],
             group_count=record['info']['group_count'],fit_rank=record['info']['fit_rank'],
             low_probability_fallbacks=record['info']['low_probability_fallbacks'],
             input_edit_rms_mm=float(1000*np.sqrt(np.mean(np.sum((out-world)**2,axis=1)))))
        row.update(synthetic_geometry(out,{},ev));row.update(structure_metrics(out,ev))
        for m in METRICS:
            if m in row:assert np.isfinite(row[m])
        rows.append(row)
        if mode=='conditional':
            references={'v7_hard':old[(case,f'reassociate_b{budget}_independent')],
                        'v7_soft':soft[(case,budget)],'original_independent':old[(case,'original_b0_independent')],
                        'local_multistart':old[(case,f'local_multistart_b{budget}_independent')]}
            for name,ref in references.items():
                pair=dict(case=case,budget=budget,gap_mm=row['gap_mm'],comparator=name)
                for m in METRICS:
                    if m in row and ref.get(m) not in ('',None):pair[m]=row[m]-float(ref[m])
                pairs.append(pair)
    csv_save(dest/'RESULTS.csv',rows);csv_save(dest/'PAIRED_DIFFERENCES.csv',pairs)
    aggregates={};contrasts={}
    for label,allowed in (('all',(0,2,4,8)),('single',(0,)),('dual',(2,4,8)),('gap2',(2,)),('gap4',(4,)),('gap8',(8,))):
        aggregates[label]={};contrasts[label]={}
        for budget in BUDGETS:
            for mode in MODES:
                part=[r for r in rows if r['budget']==budget and r['mode']==mode and r['gap_mm'] in allowed]
                aggregates[label][f'b{budget}_{mode}']=dict(n=len(part),**{m:float(np.mean([r[m] for r in part if m in r]))
                    for m in (*METRICS,'mean_abs_subtracted_noise_mm','postprocess_seconds') if any(m in r for r in part)})
            for name in ('v7_hard','v7_soft','original_independent','local_multistart'):
                part=[p for p in pairs if p['budget']==budget and p['comparator']==name and p['gap_mm'] in allowed]
                contrasts[label][f'b{budget}_vs_{name}']={m:dict(n=sum(m in p for p in part),
                    mean_delta=float(np.mean([p[m] for p in part if m in p])),wins=sum(p[m]<-1e-8 for p in part if m in p),
                    ties=sum(abs(p[m])<=1e-8 for p in part if m in p),losses=sum(p[m]>1e-8 for p in part if m in p))
                    for m in METRICS if any(m in p for p in part)}
    save(dest/'AGGREGATES.json',aggregates);save(dest/'CONTRASTS.json',contrasts)
    after={p:digest(p) for p in protected};assert after==protected;save(dest/'PROTECTED_AFTER.json',after)
    assert {p:digest(p) for p in source_hashes}==source_hashes
    summary=dict(run=str(dest),host=socket.gethostname(),inputs=24,outputs=len(rows),budgets=BUDGETS,
       sigma_mm=1.,all_outputs_saved_before_gt=True,new_confirmation=False,old_refit_replays=len(replay),
       generation_seconds=generation_seconds,total_seconds=time.perf_counter()-started,
       memory_start=initial_memory,memory_end=memory(),protected_files=len(protected),source_unchanged=True,
       scope='conditional model calibration; not blind denoising or recovered noise realization',
       timing_scope='cached-state action with fits, plus IO/evaluation; excludes V7 search and upstream')
    save(dest/'SUMMARY.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
