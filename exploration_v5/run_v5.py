"""Matched V5 sharing experiments; prior checkpoints remain read-only."""
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
V4=PROJECT/'exploration_v4'
V3=PROJECT/'exploration_v3'
sys.path[:0]=[str(HERE),str(V4),str(V3),str(PROJECT)]
from run_v4 import protected_manifest as protected_v4, algorithm_sources as sources_v4, digest
from run_exploration import synth_cases, evaluator_negative_controls, save_json, write_csv, clean, invoke as old_invoke
from paths import RUNS,OLD_CHECKPOINT,require_liekkas
from schema import write_patch
from generate_synthetics import make_ghost,make_dual
from evaluate_v2 import synthetic_geometry,point_rms_mm
from metrics import structure_metrics

METHODS=('identity','fast','open3d_icp_then_xyz','graph_shared','pool_independent',
         'pool_compatible','shared_group_slope','node_intercepts','official_gicp')
CONFIRM_SEEDS=(912601,912613,912627)
V4_RUNS=('exploration-v4-dev-e0c_n4hk','exploration-v4-confirm-1_0s40f5','real-components-v4-h3dnauhf')


def protected_manifest():
    result=protected_v4()
    for root in (V4,*(RUNS/name for name in V4_RUNS)):
        result.update({str(p):digest(p) for p in root.rglob('*') if p.is_file()})
    return result


def algorithm_sources():
    result=sources_v4()
    result.update({str(HERE/p):digest(HERE/p) for p in
                   ('run_v5.py','slope_pooling.py','registration_baselines.py','PROTOCOL.md','BASELINE_NOTES.md')})
    return result


def invoke(method,xyz,frame,sigma,ops):
    p=np.array(xyz,float,copy=True);f=np.array(frame,np.int64,copy=True)
    if method in ('identity','fast','open3d_icp_then_xyz','graph_shared'):
        return old_invoke(method,p,f,sigma,ops)
    if method=='pool_independent':
        import surface_pooling
        return surface_pooling.estimate(p,f,sigma,variant='independent')
    if method=='official_gicp':
        import registration_baselines
        return registration_baselines.estimate(p,f,sigma)
    import slope_pooling
    return slope_pooling.estimate(p,f,sigma,variant='v4_compatible' if method=='pool_compatible' else method)


def confirm_cases(dest):
    for seed in CONFIRM_SEEDS:
        for gap in (0.,2.,4.,8.):
            for bias in (0.,4.):
                if gap==0:
                    p,meta,ev=make_ghost(seed,bias);name=f'ghost_s{seed}_b{int(bias)}'
                else:
                    p,meta,ev=make_dual(seed,gap,bias);name=f'dual_g{int(gap)}_s{seed}_b{int(bias)}'
                saved=write_patch(dest/'synthetics',name,p,meta,ev)
                yield dict(case=name,base_case=name,sampling='full',xyz=p['xyz_world'],frame=p['scan_id'],
                    evaluation=ev,sigma=1.,meta=meta,source=saved['meta'],rotation=np.eye(3),translation=np.zeros(3))


def run():
    require_liekkas()
    parser=argparse.ArgumentParser();parser.add_argument('--confirm-from',type=Path)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    started=time.perf_counter();frozen=algorithm_sources()
    if args.confirm_from:
        if args.smoke:raise ValueError('cannot smoke confirmation')
        if json.loads((args.confirm_from/'ALGORITHM_SOURCES.json').read_text())!=frozen:
            raise RuntimeError('algorithm/protocol changed after development freeze')
        prior=json.loads((args.confirm_from/'SUMMARY.json').read_text())
        if prior['stage']!='development' or prior['ok']!=prior['rows']:
            raise RuntimeError('completed development outputs required')
    stage='confirmation' if args.confirm_from else 'smoke' if args.smoke else 'development'
    dest=Path(tempfile.mkdtemp(prefix='exploration-v5-'+stage+'-',dir=RUNS));print(dest,flush=True)
    save_json(dest/'ALGORITHM_SOURCES.json',frozen)
    before=protected_manifest();save_json(dest/'PROTECTED_BEFORE.json',before)
    for path in frozen:
        p=Path(path);rel=p.relative_to(PROJECT) if p.is_relative_to(PROJECT) else Path('legacy')/p.name
        target=dest/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    shutil.copyfile(HERE/'PROTOCOL.md',dest/'PROTOCOL.md')
    if args.confirm_from:
        save_json(dest/'CONFIRMATION_LOCK.json',dict(development_run=str(args.confirm_from.resolve()),
            seeds=CONFIRM_SEEDS,all_methods=METHODS,code_identical=True,
            scope='first use new seeds, same synthetic family, not unseen real geometry'))
    cases=list(confirm_cases(dest)) if args.confirm_from else list(synth_cases(args.smoke))
    save_json(dest/'NEGATIVE_CONTROLS.json',evaluator_negative_controls())
    sys.path.insert(0,str(OLD_CHECKPOINT));import operators
    assert Path(operators.__file__).resolve()==OLD_CHECKPOINT/'operators.py'
    warm=operators.warmup(('identity','fast','open3d_icp_then_xyz'))
    rng=np.random.default_rng(77);frame=np.repeat(np.arange(4),24)
    cloud=np.c_[rng.uniform(-30,30,(len(frame),2)),rng.normal(0,1,len(frame))]/1000
    for method in METHODS:
        t=time.perf_counter();invoke(method,cloud,frame,1.,operators)
        warm['method_warmup_s'][method]=time.perf_counter()-t
    save_json(dest/'WARMUP.json',warm)
    inputs=dest/'inputs';outputs=dest/'outputs';inputs.mkdir();outputs.mkdir();rows=[]
    for c in cases:
        with (inputs/(c['case']+'.npz')).open('xb') as h:
            np.savez_compressed(h,xyz_world=c['xyz'],scan_id=c['frame'])
        save_json(inputs/(c['case']+'.json'),{k:v for k,v in c.items() if k not in ('xyz','frame','evaluation')})
        for method in METHODS:
            row=dict(case=c['case'],base_case=c['base_case'],source=c['source'],method=method,stage=stage,
                sampling=c['sampling'],family=c['meta']['family'],gap_mm=c['meta']['gap_mm'],bias_rms_mm=c['meta']['bias_rms_mm'],
                seed=c['meta']['seed'],n_points=len(c['xyz']),n_frames=len(np.unique(c['frame'])),sigma_mm=c['sigma'])
            t=time.perf_counter()
            try:
                out,info=invoke(method,c['xyz'],c['frame'],c['sigma'],operators)
                elapsed=time.perf_counter()-t;out=np.asarray(out,float)
                if out.shape!=c['xyz'].shape or not np.isfinite(out).all():raise ValueError('invalid output')
                canonical=(out-c['translation'])@c['rotation']
                row.update(ok=True,method_seconds=elapsed)
                row.update(synthetic_geometry(canonical,info,c['evaluation']))
                row.update(structure_metrics(canonical,c['evaluation']))
                distances=np.linalg.norm(out-c['xyz'],axis=1)*1000
                row.update(input_edit_rms_mm=point_rms_mm(out,c['xyz']),input_edit_p95_mm=float(np.quantile(distances,.95)),
                           moved_fraction=float(np.mean(distances>1e-6)))
                target=outputs/(c['case']+'__'+method+'.npz')
                with target.open('xb') as h:np.savez_compressed(h,xyz_world=out,xyz_world_canonical=canonical)
                save_json(target.with_suffix('.json'),dict(info=info,row=row))
                row.update(output=str(target),output_sha256=digest(target))
            except Exception as exc:
                row.update(ok=False,error=repr(exc),method_seconds=time.perf_counter()-t)
                save_json(outputs/(c['case']+'__'+method+'.error.json'),dict(error=repr(exc),traceback=traceback.format_exc()))
            rows.append(row)
        print(c['case'],f"{sum(r['ok'] for r in rows[-len(METHODS):])}/{len(METHODS)}",flush=True)
    lookup={(r['case'],r['method']):r for r in rows}
    for r in rows:
        if r['sampling']=='rigid_coordinates' and r['ok']:
            base=lookup.get((r['base_case'],r['method']))
            if base and base['ok']:
                with np.load(r['output']) as a,np.load(base['output']) as b:
                    r['rigid_equivariance_rms_mm']=point_rms_mm(a['xyz_world_canonical'],b['xyz_world_canonical'])
    write_csv(dest/'RESULTS.csv',rows)
    after=protected_manifest();save_json(dest/'PROTECTED_AFTER.json',after)
    if before!=after:raise RuntimeError('historical files changed')
    if frozen!=algorithm_sources():raise RuntimeError('algorithm changed during run')
    summary=dict(run_dir=dest,stage=stage,rows=len(rows),ok=sum(r['ok'] for r in rows),inputs=len(cases),
        methods=METHODS,elapsed_s=time.perf_counter()-started,method_total_s=sum(r['method_seconds'] for r in rows),
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,protected_files=len(before),
        historical_files_unchanged=True,frozen_sources_unchanged=True,confirmation_seeds_used=CONFIRM_SEEDS if args.confirm_from else [],
        threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        cost_scope='single CPU process, potentially concurrent contention; not controlled hardware speed claim')
    save_json(dest/'SUMMARY.json',summary);print(json.dumps(clean(summary),indent=2),flush=True)


if __name__=='__main__':run()
