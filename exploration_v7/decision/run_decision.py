"""Versioned exposed-data action ablation; seal 72 outputs before reading GT."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time

sys.dont_write_bytecode=True
import numpy as np
from soft_map import project_soft_map

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
RUNS=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
PARENT=RUNS/'reassociation-v7-development-vbwoojgf'
INPUTS=RUNS/'repair-v2-oyuie4pl/synthetics/identifiable'
METRICS=('surface_accuracy_mean_mm','matched_point_rms_mm','fitted_gap_at_same_xy_error_mm')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,Path):return str(value)
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(path,value):
    with path.open('x') as f:json.dump(clean(value),f,indent=2,ensure_ascii=False,allow_nan=False)


def payload(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def main():
    if socket.gethostname()!='liekkas':raise RuntimeError('exact target host required')
    cases=sorted(p.stem for p in (PARENT/'states').glob('*.npz'))
    assert len(cases)==24
    dest=Path(tempfile.mkdtemp(prefix='decision-v7-',dir=RUNS));print(dest,flush=True)
    (dest/'outputs').mkdir();(dest/'source').mkdir()
    for path in HERE.glob('*.py'):shutil.copyfile(path,dest/'source'/path.name)
    save(dest/'SOURCE_MANIFEST.json',{str(p):digest(p) for p in HERE.glob('*.py')})
    started=time.perf_counter();records=[];protected={};pending=[]
    # No evaluator file, old evaluation CSV/JSON, or GT metadata is read here.
    for case in cases:
        state_path=PARENT/'states'/(case+'.npz');input_path=INPUTS/(case+'.npz')
        state=payload(state_path);inputs=payload(input_path)
        for path in (state_path,input_path):protected[str(path)]=digest(path)
        np.testing.assert_array_equal(state['world'],inputs['xyz_world'])
        for budget in (1,3,6):
            old_path=PARENT/'outputs'/f'{case}__reassociate_b{budget}_independent.npz'
            old=payload(old_path);protected[str(old_path)]=digest(old_path)
            tick=time.perf_counter()
            result,support=project_soft_map(state['world'],state['order'],state['design'],state['local'][:,2],
                state['active'],state['support'],state['normal'],old['group_ids'],old['coefficients'])
            projection_seconds=time.perf_counter()-tick
            np.testing.assert_array_equal(support,old['support_mask'])
            arrays={k:v for k,v in old.items() if k!='xyz_world'}
            output=dest/'outputs'/f'{case}__soft_M_map_b{budget}.npz'
            with output.open('xb') as f:np.savez_compressed(f,xyz_world=result,**arrays)
            record=dict(case=case,budget=budget,method=f'soft_M_map_b{budget}',output=str(output),
                output_sha256=digest(output),state=str(state_path),state_sha256=digest(state_path),
                original_soft_artifact=str(old_path),original_soft_artifact_sha256=digest(old_path),
                projection_only_seconds=projection_seconds,gt_fields_used=[],
                action='MAP component + saved soft-M plane; no hard-label WLS refit',
                preserved=['input XYZ','order','normal','scan bias','weights','active rows','support','MAP group IDs',
                           'responsibility','candidate mask','soft objective','soft-M coefficients'],
                timing_scope='cached artifact projection only; not a complete estimator or speed comparison')
            records.append(record);pending.append((case,budget,output))
    save(dest/'OUTPUTS_SEALED_BEFORE_GT.json',records)
    save(dest/'PROTECTED_BEFORE.json',protected)
    generation_seconds=time.perf_counter()-started
    print(f'{len(records)} outputs saved and hashed; evaluator boundary begins now',flush=True)

    # GT and old evaluation outcomes are permitted only after all outputs exist.
    sys.path[:0]=[str(PROJECT),str(PROJECT/'exploration_v3')]
    from evaluate_v2 import synthetic_geometry
    from metrics import structure_metrics
    reference=list(csv.DictReader((PARENT/'RESULTS.csv').open()))
    reference={(r['case'],r['method']):r for r in reference}
    rows=[];differences=[]
    for case,budget,path in pending:
        with np.load(INPUTS/'evaluation'/(case+'.eval.npz'),allow_pickle=False) as a:
            ev={k:a[k].copy() for k in a.files if k!='json'}
            if 'json' in a:ev.update(json.loads(a['json'].tobytes().decode()))
        meta=json.loads((INPUTS/(case+'.json')).read_text())
        data=payload(path);inputs=payload(INPUTS/(case+'.npz'))
        result=data['xyz_world'];support=data['support_mask']
        row=dict(case=case,budget=budget,method=f'soft_M_map_b{budget}',gap_mm=float(meta['gap_mm']),
            seed=int(meta['seed']),n_points=len(result),supported_fraction=float(support.mean()),
            group_count=len(np.unique(data['group_ids'][data['group_ids']>=0])),
            input_edit_rms_mm=float(1000*np.sqrt(np.mean(np.sum((result-inputs['xyz_world'])**2,axis=1)))),
            output=str(path),output_sha256=digest(path),stage='exposed development post-decision ablation')
        row.update(synthetic_geometry(result,{},ev));row.update(structure_metrics(result,ev))
        for key in METRICS:
            if key in row and not np.isfinite(row[key]):raise AssertionError('nonfinite score')
        rows.append(row)
        for comparator in (f'reassociate_b{budget}_independent','original_b0_independent'):
            old=reference[(case,comparator)]
            d=dict(case=case,budget=budget,gap_mm=row['gap_mm'],comparator=comparator)
            for key in METRICS:
                if key in row and old.get(key):d[key]=float(row[key])-float(old[key])
            differences.append(d)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (dest/'RESULTS.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    aggregated={};paired={}
    for gap in ('all',0.,2.,4.,8.):
        for budget in (1,3,6):
            values=[r for r in rows if r['budget']==budget and (gap=='all' or r['gap_mm']==gap)]
            key=f'gap_{gap}_b{budget}'
            aggregated[key]=dict(n=len(values),**{m:float(np.mean([r[m] for r in values if m in r])) for m in METRICS if any(m in r for r in values)})
            for comparator in (f'reassociate_b{budget}_independent','original_b0_independent'):
                ds=[d for d in differences if d['budget']==budget and d['comparator']==comparator and (gap=='all' or d['gap_mm']==gap)]
                paired[key+'__'+comparator]={m:dict(mean_new_minus_old=float(np.mean([d[m] for d in ds if m in d])),
                    wins=sum(d[m]<-1e-10 for d in ds if m in d),ties=sum(abs(d[m])<=1e-10 for d in ds if m in d),
                    losses=sum(d[m]>1e-10 for d in ds if m in d)) for m in METRICS if any(m in d for d in ds)}
    save(dest/'AGGREGATES.json',aggregated);save(dest/'PAIRED.json',paired);save(dest/'DIFFERENCES.json',differences)
    after={p:digest(p) for p in protected};assert after==protected;save(dest/'PROTECTED_AFTER.json',after)
    summary=dict(run_dir=str(dest),outputs=len(rows),inputs=len(cases),budgets=[1,3,6],
        all_outputs_saved_before_gt=True,kept_same_soft_states=True,same_support_and_map_labels=True,
        generation_seconds=generation_seconds,wall_seconds=time.perf_counter()-started,
        protected_files=len(protected),protected_unchanged=True,new_confirmation=False,
        scope='one post-decision ablation, not new upstream/association optimization, no end-to-end speed measurement')
    save(dest/'SUMMARY.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
