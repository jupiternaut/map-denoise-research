"""Immutable same-input V4 comparisons and one frozen held-seed confirmation."""
from __future__ import annotations
import argparse
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

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
V3=PROJECT/'exploration_v3'
sys.path[:0]=[str(HERE),str(V3),str(PROJECT)]
from run_exploration import synth_cases, evaluator_negative_controls, save_json, clean, write_csv, protected_manifest as previous_manifest, invoke as invoke_v3
from paths import RUNS, OLD_CHECKPOINT, require_liekkas
from evaluate_v2 import synthetic_geometry, point_rms_mm
from metrics import structure_metrics
from schema import write_patch
from generate_synthetics import make_ghost,make_dual,HOLD_OUT_SEEDS

METHODS=('identity','fast','open3d_icp_then_xyz','graph_shared',
         'pool_independent','pool_global','pool_compatible',
         'graph_bias_only','graph_projected_bias_only',
         'zero_balanced_6','zero_unbalanced_6','warm_balanced_2','warm_unbalanced_2')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def algorithm_sources():
    sources=[HERE/p for p in ('run_v4.py','surface_pooling.py','warm_transport.py','PROTOCOL.md')]
    sources += [V3/p for p in ('graph_surface.py','measure_surface.py','metrics.py','run_exploration.py')]
    sources += [PROJECT/p for p in ('evaluate_v2.py','adapt.py','transforms.py','generate_synthetics.py','schema.py','paths.py')]
    sources += [OLD_CHECKPOINT/'operators.py']
    return {str(p):digest(p) for p in sources}


def protected_manifest():
    result=previous_manifest()
    for root in (V3,RUNS/'exploration-v3-lnx049yd',RUNS/'transport-budget-v3-mxi1kprj'):
        result.update({str(p):digest(p) for p in root.rglob('*') if p.is_file()})
    return result


def invoke(method,xyz,frame,sigma,ops):
    p=np.array(xyz,float,copy=True); f=np.array(frame,dtype=np.int64,copy=True)
    if method in ('identity','fast','open3d_icp_then_xyz','graph_shared'):
        return invoke_v3(method,p,f,sigma,ops)
    if method.startswith('pool_'):
        from surface_pooling import estimate
        return estimate(p,f,sigma,variant=method.removeprefix('pool_'))
    from warm_transport import estimate
    return estimate(p,f,sigma,variant=method)


def confirmation_cases(destination):
    result=[]
    for seed in HOLD_OUT_SEEDS:
        for gap in (0.,2.,4.,8.):
            for bias in (0.,4.):
                if gap==0:
                    points,meta,ev=make_ghost(seed,bias); name=f'ghost_s{seed}_b{int(bias)}'
                else:
                    points,meta,ev=make_dual(seed,gap,bias); name=f'dual_g{int(gap)}_s{seed}_b{int(bias)}'
                saved=write_patch(destination/'synthetics'/'identifiable',name,points,meta,ev)
                result.append(dict(case=name,base_case=name,split='synthetic',sampling='full',
                  xyz=points['xyz_world'],frame=points['scan_id'],evaluation=ev,sigma=1.,meta=meta,
                  source=saved['meta'],rotation=np.eye(3),translation=np.zeros(3)))
    return result


def run():
    require_liekkas()
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--confirm-from',type=Path)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    args=parser.parse_args()
    started=time.perf_counter()
    frozen=algorithm_sources()
    if args.confirm_from:
        if args.smoke:raise ValueError('confirmation cannot be smoke')
        previous=json.loads((args.confirm_from/'ALGORITHM_SOURCES.json').read_text())
        if frozen!=previous:raise RuntimeError('algorithm/protocol changed after development freeze')
        prior=json.loads((args.confirm_from/'SUMMARY.json').read_text())
        if prior['stage']!='development' or prior['ok']!=prior['rows']:
            raise RuntimeError('need a completed development run')
        if list(args.methods)!=prior['methods']:raise RuntimeError('confirm the same entire method set')
    label='confirm' if args.confirm_from else 'smoke' if args.smoke else 'dev'
    dest=Path(tempfile.mkdtemp(prefix=f'exploration-v4-{label}-',dir=RUNS))
    print(dest,flush=True)
    save_json(dest/'ALGORITHM_SOURCES.json',frozen)
    before=protected_manifest();save_json(dest/'PROTECTED_BEFORE.json',before)
    sources=dest/'source';sources.mkdir()
    for path in frozen:
        p=Path(path);rel=p.relative_to(PROJECT) if p.is_relative_to(PROJECT) else Path('legacy')/p.name
        target=sources/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    shutil.copyfile(HERE/'PROTOCOL.md',dest/'PROTOCOL.md')
    if args.confirm_from:
        save_json(dest/'CONFIRMATION_LOCK.json',{'development_run':str(args.confirm_from.resolve()),
          'seeds':list(HOLD_OUT_SEEDS),'code_and_protocol_identical':True,
          'interpretation':'new noise/sampling seeds in the same synthetic family, not new real scenes',
          'all_variants_confirmed':list(args.methods)})
        cases=confirmation_cases(dest)
    else:cases=list(synth_cases(args.smoke))
    save_json(dest/'NEGATIVE_CONTROLS.json',evaluator_negative_controls())
    sys.path.insert(0,str(OLD_CHECKPOINT));import operators
    assert Path(operators.__file__).resolve()==(OLD_CHECKPOINT/'operators.py').resolve()
    warm=operators.warmup(('identity','fast','open3d_icp_then_xyz'))
    # Always warm on an exposed, separately generated small plane, not on held seeds.
    rng=np.random.default_rng(77);f=np.repeat(np.arange(4),24)
    xyz=np.c_[rng.uniform(-30,30,(len(f),2)),rng.normal(0,1,len(f))]/1000
    for method in args.methods:
        start=time.perf_counter();invoke(method,xyz,f,1.,operators)
        warm['method_warmup_s'][method]=time.perf_counter()-start
    save_json(dest/'WARMUP.json',warm)
    inputs=dest/'inputs';inputs.mkdir();outputs=dest/'outputs';outputs.mkdir()
    rows=[]
    for c in cases:
        with (inputs/(c['case']+'.npz')).open('xb') as h:
            np.savez_compressed(h,xyz_world=c['xyz'],scan_id=c['frame'])
        save_json(inputs/(c['case']+'.json'),{k:v for k,v in c.items() if k not in ('xyz','frame','evaluation')})
        for method in args.methods:
            row=dict(case=c['case'],base_case=c['base_case'],source=c['source'],method=method,
                     stage='confirmation' if args.confirm_from else 'development',sampling=c['sampling'],
                     family=c['meta']['family'],gap_mm=c['meta']['gap_mm'],bias_rms_mm=c['meta']['bias_rms_mm'],
                     seed=c['meta']['seed'],n_points=len(c['xyz']),n_frames=len(np.unique(c['frame'])),sigma_mm=c['sigma'])
            start=time.perf_counter()
            try:
                out,info=invoke(method,c['xyz'],c['frame'],c['sigma'],operators)
                elapsed=time.perf_counter()-start
                out=np.asarray(out,float)
                if out.shape!=c['xyz'].shape or not np.isfinite(out).all():raise ValueError('invalid estimator output')
                canonical=(out-c['translation'])@c['rotation']
                row.update(ok=True,method_seconds=elapsed)
                row.update(synthetic_geometry(canonical,info,c['evaluation']))
                row.update(structure_metrics(canonical,c['evaluation']))
                distance=np.linalg.norm(out-c['xyz'],axis=1)*1000
                row.update(input_edit_rms_mm=point_rms_mm(out,c['xyz']),input_edit_p95_mm=float(np.quantile(distance,.95)),
                           moved_fraction=float(np.mean(distance>1e-6)))
                target=outputs/(c['case']+'__'+method+'.npz')
                with target.open('xb') as h:np.savez_compressed(h,xyz_world=out,xyz_world_canonical=canonical)
                save_json(target.with_suffix('.json'),{'info':info,'row':row})
                row['output']=str(target);row['output_sha256']=digest(target)
            except Exception as exc:
                row.update(ok=False,error=repr(exc),method_seconds=time.perf_counter()-start)
                save_json(outputs/(c['case']+'__'+method+'.error.json'),{'error':repr(exc),'traceback':traceback.format_exc()})
            rows.append(row)
        print(c['case'],f"{sum(r['ok'] for r in rows[-len(args.methods):])}/{len(args.methods)}",flush=True)
    lookup={(r['case'],r['method']):r for r in rows}
    for row in rows:
        if row['sampling']=='rigid_coordinates' and row['ok']:
            base=lookup.get((row['base_case'],row['method']))
            if base and base['ok']:
                with np.load(row['output']) as a,np.load(base['output']) as b:
                    row['rigid_equivariance_rms_mm']=point_rms_mm(a['xyz_world_canonical'],b['xyz_world_canonical'])
    write_csv(dest/'RESULTS.csv',rows)
    after=protected_manifest();save_json(dest/'PROTECTED_AFTER.json',after)
    if before!=after:raise RuntimeError('protected historical files changed')
    if frozen!=algorithm_sources():raise RuntimeError('algorithm changed during run')
    summary=dict(run_dir=dest,stage='confirmation' if args.confirm_from else 'development',smoke=args.smoke,
        rows=len(rows),ok=sum(r['ok'] for r in rows),inputs=len(cases),methods=list(args.methods),
        original_synthetic_cases=24 if args.confirm_from else 4 if args.smoke else 27,
        elapsed_s=time.perf_counter()-started,method_total_s=sum(r['method_seconds'] for r in rows),
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        protected_files=len(before),historical_files_unchanged=True,frozen_sources_unchanged=True,
        reserved_seeds_used=list(HOLD_OUT_SEEDS) if args.confirm_from else [],
        threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        cost_note='single-thread per process; concurrent research jobs may affect timings; not GPU or a controlled speed benchmark')
    save_json(dest/'SUMMARY.json',summary);print(json.dumps(clean(summary),indent=2),flush=True)


if __name__=='__main__':run()
