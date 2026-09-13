"""Run immutable development comparisons on the exact liekkas pilot inputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
sys.path.insert(0,str(PROJECT))
from paths import RUNS, PATCHES, OLD_CHECKPOINT, require_liekkas
from schema import read_patch, read_evaluation
from adapt import build_adapter, to_operator_mm, from_operator_mm
from evaluate_v2 import synthetic_geometry, point_rms_mm
from repair_pilot_v2 import old_manifest, visible_perturbation, real_metrics
from metrics import structure_metrics, sampling_indices, subset_evaluation

V2=RUNS/'repair-v2-oyuie4pl'
METHODS=('identity','xyz_mixture','fast','open3d_icp_then_xyz',
         'graph_local','graph_shared','measure_balanced','measure_unbalanced',
         'measure_balanced_local','measure_unbalanced_local')


def clean(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    return value


def save_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(clean(data),f,indent=2,ensure_ascii=False,allow_nan=False)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_manifest():
    result=old_manifest()
    result.update({str(p):sha(p) for p in V2.rglob('*') if p.is_file()})
    return result


def invoke(method,xyz,frame,sigma,ops):
    # Method boundary receives only legal arrays and supplied noise, copied.
    p=np.array(xyz,dtype=float,copy=True); f=np.array(frame,dtype=np.int64,copy=True)
    if method=='identity': return p,{'method':'identity'}
    if method in ('xyz_mixture','fast','open3d_icp_then_xyz'):
        adapter=build_adapter(p)
        result,info=ops.estimate(method,to_operator_mm(p,adapter),f,sigma)
        return from_operator_mm(result,adapter),info
    if method.startswith('graph_'):
        from graph_surface import estimate
        return estimate(p,f,sigma,variant='graph' if method=='graph_shared' else 'local_only')
    from measure_surface import estimate
    corrected,info=estimate(p,f,sigma,variant='unbalanced' if 'unbalanced' in method else 'balanced')
    if method.endswith('_local'):
        from graph_surface import estimate as local_estimate
        result,local_info=local_estimate(corrected,f,sigma,variant='local_only')
        return result,{'transport':info,'local_filter':local_info,'pipeline':'transport then identical graph_local filter'}
    return corrected,info


def synth_cases(smoke=False):
    files=sorted((V2/'synthetics').glob('*/*.json'))
    bases=[]
    for file in files:
        p,m=read_patch(file); ev=read_evaluation(file)
        if smoke and not(m['seed']==912101 and m['bias_rms_mm']==4 and m['family']!='ambiguous_two_explanations'):
            continue
        entry=dict(case=file.stem,base_case=file.stem,split='synthetic',sampling='full',
                   xyz=p['xyz_world'],frame=p['scan_id'],evaluation=ev,sigma=1.,meta=m,
                   source=str(file),rotation=np.eye(3),translation=np.zeros(3))
        bases.append(entry)
    yield from bases
    if smoke: return
    R=Rotation.from_euler('xyz',[17.,31.,-23.],degrees=True).as_matrix()
    translation=np.array([.37,-.22,.41])
    for base in bases:
        m=base['meta']
        if m['seed']!=912101 or m['bias_rms_mm']!=4: continue
        for mode in ('density','partial_overlap'):
            ind=sampling_indices(base['xyz'],base['frame'],mode)
            ev=subset_evaluation(base['evaluation'],ind,len(base['xyz']))
            yield dict(base,case=base['case']+'__'+mode,sampling=mode,
                       xyz=base['xyz'][ind],frame=base['frame'][ind],evaluation=ev)
        yield dict(base,case=base['case']+'__rigid_coordinates',sampling='rigid_coordinates',
                   xyz=base['xyz']@R.T+translation,rotation=R,translation=translation)


def real_cases():
    for file in sorted(PATCHES.glob('*/*.json')):
        p,m=read_patch(file)
        for mode in ('unpert','trans','trans_rot'):
            if mode=='unpert': xyz=p['xyz_world']; actual=0.
            else:
                current,change=visible_perturbation(p,m,mode,912401)
                xyz=current['xyz_world']; actual=change['actual_point_delta_rms_mm']
            yield dict(case=file.stem+'__'+mode,base_case=file.stem,split='real',sampling=mode,
                       xyz=xyz,frame=p['scan_id'],sigma=2.,meta=m,source=str(file),
                       reference=p['xyz_world'],actual_point_delta_rms_mm=actual)


def evaluator_negative_controls():
    file=next(p for p in (V2/'synthetics'/'identifiable').glob('*.json') if 'dual' in p.stem and 'g4_' in p.stem)
    p,m=read_patch(file); ev=read_evaluation(file); ref=ev['gt_clean_xyz_world']
    ramp=ref.copy(); A=np.c_[ref[:,:2],np.ones(len(ref))]
    ramp[:,2]=A@np.linalg.lstsq(A,ref[:,2],rcond=None)[0]
    collapsed=ref.copy(); collapsed[:,2]=np.mean(ref[:,2])
    shifted=ref+np.array([0.,0.,.1])
    rows=[]
    for name,out in [('gt',ref),('collapsed',collapsed),('ramp',ramp),('shift100mm',shifted)]:
        a=synthetic_geometry(out,{'K':1},ev); b=synthetic_geometry(out,{'K':2},ev)
        for key in a:
            if key not in ('reported_model_k',): assert a[key]==b[key],key
        row=dict(output=name,**a,**structure_metrics(out,ev)); rows.append(row)
    assert rows[0]['surface_accuracy_mean_mm']<1e-9
    assert rows[1]['surface_accuracy_mean_mm']>1.
    assert rows[2]['fitted_gap_at_same_xy_error_mm']>3.9
    assert rows[3]['surface_accuracy_mean_mm']>95.
    return rows


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def run():
    require_liekkas()
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    parser.add_argument('--out-parent',type=Path,default=RUNS)
    args=parser.parse_args()
    total_start=time.perf_counter()
    dest=Path(tempfile.mkdtemp(prefix='exploration-v3-',dir=args.out_parent))
    print(str(dest),flush=True)
    before=protected_manifest(); save_json(dest/'PROTECTED_BEFORE.json',before)
    source_dir=dest/'source'; source_dir.mkdir()
    sources=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))
    sources += [PROJECT/p for p in ('adapt.py','transforms.py','evaluate_v2.py','schema.py','perturb.py','repair_pilot_v2.py','hashutil.py','paths.py')]
    for p in sources:
        target=source_dir/p.name
        if target.exists(): raise RuntimeError('source snapshot basename collision')
        shutil.copyfile(p,target)
    save_json(dest/'SOURCE_MANIFEST.json',{str(p):sha(p) for p in sources})
    save_json(dest/'NEGATIVE_CONTROLS.json',evaluator_negative_controls())
    sys.path.insert(0,str(OLD_CHECKPOINT)); import operators
    assert Path(operators.__file__).resolve()==(OLD_CHECKPOINT/'operators.py').resolve()
    warm=operators.warmup(('identity','xyz_mixture','fast','open3d_icp_then_xyz'))
    cases=list(synth_cases(args.smoke))
    if not args.smoke: cases+=list(real_cases())
    # Warm new modules using the first legal observed input; no evaluation passed.
    for method in args.methods:
        if method.startswith(('graph_','measure_')):
            start=time.perf_counter()
            invoke(method,cases[0]['xyz'],cases[0]['frame'],cases[0]['sigma'],operators)
            warm['method_warmup_s'][method]=time.perf_counter()-start
    save_json(dest/'WARMUP.json',warm)
    save_json(dest/'BASELINE_SOURCES.json',operators.source_hashes())
    output_dir=dest/'outputs'; output_dir.mkdir()
    input_dir=dest/'inputs'; input_dir.mkdir()
    rows=[]; real_baselines={}; artifacts=[]
    for case in cases:
        cp=input_dir/(case['case']+'.npz')
        with cp.open('xb') as f:
            np.savez_compressed(f,xyz_world=case['xyz'],scan_id=case['frame'],
                                source=str(case['source']))
        eval_payload=case.get('evaluation',{})
        # Extra metadata is stored separately, never read in invoke().
        save_json(input_dir/(case['case']+'.json'),
                  {k:v for k,v in case.items() if k not in ('xyz','frame','evaluation','reference')})
        for method in args.methods:
            start=time.perf_counter()
            row=dict(case=case['case'],base_case=case['base_case'],split=case['split'],
                     sampling=case['sampling'],method=method,sigma_mm=case['sigma'],
                     n_points=len(case['xyz']),n_frames=len(np.unique(case['frame'])),
                     source=case['source'])
            try:
                out,info=invoke(method,case['xyz'],case['frame'],case['sigma'],operators)
                elapsed=time.perf_counter()-start
                out=np.asarray(out,dtype=float)
                if out.shape!=case['xyz'].shape or not np.isfinite(out).all():
                    raise ValueError('invalid output shape or nonfinite output')
                row.update(ok=True,method_seconds=elapsed)
                path=output_dir/(case['case']+'__'+method+'.npz')
                canonical=out
                if case['split']=='synthetic':
                    canonical=(out-case['translation'])@case['rotation']
                    row.update(family=case['meta']['family'],gap_mm=case['meta']['gap_mm'],
                               bias_rms_mm=case['meta']['bias_rms_mm'],seed=case['meta']['seed'])
                    row.update(synthetic_geometry(canonical,info,eval_payload))
                    row.update(structure_metrics(canonical,eval_payload))
                else:
                    base_key=(case['base_case'],method)
                    if case['sampling']=='unpert': real_baselines[base_key]=out.copy()
                    if base_key in real_baselines:
                        row.update(real_metrics(out,case['reference'],real_baselines[base_key]))
                    row.update(independent_real_geometry='not_measured',scene=case['meta']['scene'],
                               actual_point_delta_rms_mm=case['actual_point_delta_rms_mm'])
                displacement=np.linalg.norm(out-case['xyz'],axis=1)*1000
                row.update(input_edit_rms_mm=point_rms_mm(out,case['xyz']),
                           input_edit_p95_mm=float(np.quantile(displacement,.95)),
                           moved_fraction=float(np.mean(displacement>1e-6)))
                with path.open('xb') as f: np.savez_compressed(f,xyz_world=out,xyz_world_canonical=canonical)
                save_json(path.with_suffix('.json'),{'info':info,'row':row})
                row['output']=str(path)
                artifacts.append((case['case'],method,path))
            except Exception as exc:
                row.update(ok=False,error=repr(exc),method_seconds=time.perf_counter()-start)
                save_json(output_dir/(case['case']+'__'+method+'.error.json'),
                          {'error':repr(exc),'traceback':traceback.format_exc()})
            rows.append(row)
        print(f"{case['case']}: {sum(r['ok'] for r in rows[-len(args.methods):])}/{len(args.methods)}",flush=True)
    # Coordinate-equivariance diagnostic on common base observations.
    lookup={(r['case'],r['method']):r for r in rows}
    for row in rows:
        if row['sampling']=='rigid_coordinates' and row['ok']:
            base=lookup.get((row['base_case'],row['method']))
            if base and base['ok']:
                with np.load(row['output']) as a: x=a['xyz_world_canonical']
                with np.load(base['output']) as a: y=a['xyz_world_canonical']
                row['rigid_equivariance_rms_mm']=point_rms_mm(x,y)
    write_csv(dest/'RESULTS.csv',rows)
    after=protected_manifest(); save_json(dest/'PROTECTED_AFTER.json',after)
    if before!=after: raise RuntimeError('protected historical artifact changed')
    summary=dict(run_dir=dest,rows=len(rows),ok=sum(r['ok'] for r in rows),
                 synthetic_inputs=sum(c['split']=='synthetic' for c in cases),
                 real_inputs=sum(c['split']=='real' for c in cases),
                 original_synthetic_worlds=27 if not args.smoke else len(cases),
                 real_scenes=2 if not args.smoke else 0,real_patches=6 if not args.smoke else 0,
                 smoke=args.smoke,elapsed_s=time.perf_counter()-total_start,
                 method_total_s=sum(r['method_seconds'] for r in rows),
                 peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 protected_files=len(before),historical_files_unchanged=True,
                 threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                 python=sys.executable,methods=list(args.methods),
                 evidence='exposed development data; no independent real geometry; not confirmation')
    save_json(dest/'SUMMARY.json',summary)
    print(json.dumps(clean(summary),indent=2),flush=True)


if __name__=='__main__': run()
