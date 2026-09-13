"""Development-only equivalence and cost experiment; no new geometry claim."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import cProfile
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pstats
import subprocess
import sys
import threading
import time
import numpy as np
import psutil
from accelerated import FusedEM, REFERENCE, OLD, make_denoiser, traced_denoise

HERE=Path(__file__).resolve().parent
RUN=Path('/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600/t5')
CONFIG={'fallback':'bilateral','em_iterations':32,'iterations':2}

def write_json(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False)

def measured(fn,p):
    process=psutil.Process(); values=[process.memory_info().rss]; stop=threading.Event()
    def poll():
        while not stop.wait(.005): values.append(process.memory_info().rss)
    monitor=threading.Thread(target=poll); monitor.start()
    start=time.perf_counter(); output,diag=fn(p,CONFIG); elapsed=time.perf_counter()-start
    values.append(process.memory_info().rss); stop.set(); monitor.join()
    return output,{'seconds':elapsed,'rss_start_bytes':values[0],'rss_sampled_peak_bytes':max(values),'rss_end_bytes':values[-1]}

def geometry_module():
    path=OLD.parents[2]/'parallel_geometry_v4'/'geometry.py'
    spec=importlib.util.spec_from_file_location('t5_geom',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def cases(geo):
    for seed in (1211,1212):
        for j,shape in enumerate(('plane','thin_assembly','cylinder','crease')):
            clean,normal,_=geo.sample_surface(shape,1500,np.random.default_rng(seed+j))
            reference,_,labels=geo.sample_surface(shape,8000,np.random.default_rng(seed+200+j))
            for corruption in ('iid_1p5mm','normal_bias_1p5mm'):
                p=geo.corrupt(clean,normal,corruption,np.random.default_rng(seed+90))
                yield f'{shape}-{corruption}-{seed}',p,(shape,reference,labels)
    rng=np.random.default_rng(910115)
    for ratio in (0.,1.5,2.5,3.,4.,6.,8.):
        p=np.column_stack((rng.uniform(-.06,.06,(600,2)),rng.normal(0,.001,600)+rng.choice([-.5,.5],600)*ratio*.001))
        yield f'gap-noise-ratio-{ratio:g}',p,None
    p=np.column_stack((rng.uniform(-.1,.1,(600,2)),np.zeros(600)))
    yield 'exact-plane',p,None
    yield 'line-degenerate',np.column_stack((np.linspace(-.1,.1,300),np.zeros((300,2)))),None
    yield 'all-duplicate',np.zeros((300,3)),None
    yield 'empty',np.zeros((0,3)),None
    yield 'single',np.ones((1,3)),None

def residual_checks(library):
    rng=np.random.default_rng(8713)
    rows=np.concatenate([rng.normal(0,1,(200,48)),
          rng.normal(0,1,(200,48))+rng.choice([-3.,3.],(200,48)),
          rng.normal(0,.001,(100,48)),np.zeros((25,48)),
          np.tile(np.r_[np.zeros(24),np.ones(24)],(25,1))])
    checks=[]
    ref=REFERENCE.fit_modes(rows,.01)
    configurations=[('default',{}),('bic-below',{'bic_gain':float(np.nextafter(ref['bic_gain'][210],-np.inf))}),
       ('bic-exact',{'bic_gain':float(ref['bic_gain'][210])}),
       ('bic-above',{'bic_gain':float(np.nextafter(ref['bic_gain'][210],np.inf))}),
       ('separation-exact',{'separation_sigma':float(ref['separation_sigma'][210])}),
       ('support-exact',{'min_count':3,'min_fraction':float(ref['responsibility'][210].sum(0).min()/48)})]
    for name,cfg in configurations:
        target=REFERENCE.fit_modes(rows,.01,cfg)
        for screened in (False,True):
            fit=FusedEM(library,screen=screened); got=fit(rows,.01,cfg)
            checks.append({'config':name,'screened':screened,'rows':len(rows),
             'branch_mismatches':int(np.sum(got['choose_two']!=target['choose_two'])),
             'max_resp_delta_on_selected':float(np.max(np.abs(got['responsibility'][target['choose_two']]-target['responsibility'][target['choose_two']]))) if target['choose_two'].any() else 0.,
             'stats':fit.stats})
    np.savez_compressed(RUN/'residual-tests.npz',rows=rows,reference_choice=ref['choose_two'])
    return checks

def main():
    RUN.mkdir(parents=True,exist_ok=False)
    library=RUN/'mixture_kernel.so'
    command=['g++','-O3','-std=c++17','-fPIC','-shared','-fno-fast-math',str(HERE/'mixture_kernel.cpp'),'-o',str(library)]
    start=time.perf_counter(); built=subprocess.run(command,capture_output=True,text=True,check=True)
    compilation_s=time.perf_counter()-start
    write_json(RUN/'build.json',{'command':command,'seconds':compilation_s,'stdout':built.stdout,'stderr':built.stderr})
    geo=geometry_module(); allcases=list(cases(geo)); profiler=cProfile.Profile()
    profiler.enable(); REFERENCE.denoise(allcases[0][1],CONFIG); profiler.disable()
    capture=io.StringIO(); pstats.Stats(profiler,stream=capture).sort_stats('cumtime').print_stats(25)
    (RUN/'reference-profile.txt').write_text(capture.getvalue())
    residual=residual_checks(library); write_json(RUN/'residual-checks.json',residual)
    backend={'reference':None,'fused':FusedEM(library),'fused_screen':FusedEM(library,screen=True)}
    functions={name:make_denoiser(fit) for name,fit in backend.items()}
    records=[]; comparison=[]; start_all=time.perf_counter()
    # This is first-call rather than a fresh-process benchmark; the profiler has warmed libraries.
    cold={}
    for name,fn in functions.items():
        _,cold[name]=measured(fn,allcases[0][1])
    for case_index,(case,p,metric) in enumerate(allcases):
        np.save(RUN/f'{case}-input.npy',p,allow_pickle=False)
        outputs={}; traces={}; case_records={}
        for name,fit in backend.items():
            out,diag,trace=traced_denoise(p,CONFIG,fit)
            # Save output before any GT metric calculation.
            np.save(RUN/f'{case}-{name}.npy',out,allow_pickle=False)
            outputs[name]=out; traces[name]=trace
            arr={f'{key}_{it}':value for it,t in enumerate(trace) for key,value in t.items()}
            np.savez_compressed(RUN/f'{case}-{name}-branches.npz',**arr)
            record={'case':case,'arm':name,'points':len(p),'diagnostics':diag,'timings':[]}
            if metric:
                shape,ref,labels=metric; record['metrics']=geo.evaluate(shape,out,ref,labels,len(p))
            records.append(record); case_records[name]=record
        # Reverse ordering across repetitions to reduce, not eliminate, concurrent load bias.
        for repeat in range(3):
            order=list(functions) if (repeat+case_index)%2==0 else list(reversed(functions))
            for name in order:
                _,timing=measured(functions[name],p); timing['repeat']=repeat
                case_records[name]['timings'].append(timing)
        for name in ('fused','fused_screen'):
            delta=np.linalg.norm(outputs[name]-outputs['reference'],axis=1)
            mismatch={key:sum(int(np.sum(a[key]!=b[key])) for a,b in zip(traces[name],traces['reference'])) for key in ('choose_two','split')}
            row={'case':case,'arm':name,'max_output_delta_m':float(delta.max()) if len(delta) else 0.,
                 'rms_output_delta_m':float(np.sqrt(np.mean(delta**2))) if len(delta) else 0.,'branch_mismatches':mismatch,
                 'reference_median_s':float(np.median([x['seconds'] for x in case_records['reference']['timings']])),
                 'candidate_median_s':float(np.median([x['seconds'] for x in case_records[name]['timings']]))}
            row['speedup']=row['reference_median_s']/row['candidate_median_s']; comparison.append(row)
        print(json.dumps({'case':case,'comparison':comparison[-2:]}),flush=True)
        write_json(RUN/f'case-{case_index:02d}.json',{'records':list(case_records.values()),'comparison':comparison[-2:]})
    substantial=[r for r in comparison if r['case'] not in ('empty','single','all-duplicate')]
    summary={}
    for name in ('fused','fused_screen'):
        selected=[r for r in substantial if r['arm']==name]
        summary[name]={'median_case_speedup':float(np.median([r['speedup'] for r in selected])),
          'aggregate_time_ratio':sum(r['reference_median_s'] for r in selected)/sum(r['candidate_median_s'] for r in selected),
          'min_speedup':min(r['speedup'] for r in selected),'max_speedup':max(r['speedup'] for r in selected),
          'max_output_delta_m':max(r['max_output_delta_m'] for r in selected),
          'choose_two_mismatches':sum(r['branch_mismatches']['choose_two'] for r in selected),
          'split_mismatches':sum(r['branch_mismatches']['split'] for r in selected)}
    manifest={'status':'exposed_development_equivalence_and_implementation_cost_only',
       'config':CONFIG,'case_count':len(allcases),'compiled_backend':'single-thread float64 C++ no fast-math',
       'compile_seconds':compilation_s,'first_call_not_cold_process':cold,'residual_checks':residual,
       'summary':summary,'records':records,'comparisons':comparison,
       'screen_stats':{name:fit.stats for name,fit in backend.items() if fit is not None},
       'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD,HERE/'accelerated.py',HERE/'mixture_kernel.cpp',HERE/'run.py')},
       'elapsed_s':time.perf_counter()-start_all,'environment':{'python':sys.version,'cpu_count':os.cpu_count(),'threads':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}}
    write_json(RUN/'results.json',manifest)
    print(json.dumps({'summary':summary,'run':str(RUN)}),flush=True)
    assert all(r['branch_mismatches']==0 for r in residual)
    assert all(r['max_output_delta_m']<1e-9 and not any(r['branch_mismatches'].values()) for r in comparison)

if __name__=='__main__': main()
