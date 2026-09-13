import argparse,csv,importlib.util,json,os,shutil,socket,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'exploration_v10'),str(ROOT)]
from run_v10 import protect,RUNS,PARENT,INPUTS,read,save,npz,csvsave,memory,sha
from spatial_filter import METHODS,filter_frozen
from spectral_filter import null_gain_threshold

def main(fresh=False):
    assert socket.gethostname()=='liekkas'
    dest=Path(tempfile.mkdtemp(prefix='spatial-v11-'+('confirmation-' if fresh else 'development-'),dir=RUNS));print(dest,flush=True)
    for folder in ('outputs','source','states'):(dest/folder).mkdir()
    history=protect()
    for base in (ROOT/'exploration_v10',RUNS/'spectral-filter-v10-4bn33_0y',RUNS/'spectral-filter-v10-dpamdvgb'):
        history.update({str(p):sha(p) for p in base.rglob('*') if p.is_file()})
    save(dest/'HISTORY_BEFORE.json',history)
    sources=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    source_hash={str(p):sha(p) for p in sources};save(dest/'SOURCES.json',source_hash)
    for p in sources:shutil.copyfile(p,dest/'source'/p.name)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    (dest/'TESTS.log').write_text(tests.stdout+tests.stderr);assert tests.returncode==0,tests.stderr
    inputs_root=INPUTS
    spec=importlib.util.spec_from_file_location('_v11_v7',ROOT/'exploration_v7/algorithm/reassociation.py')
    v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)
    if fresh:
        from generate_synthetics import make_ghost,make_dual
        from schema import write_patch
        inputs_root=dest/'inputs'
        for seed in (9131011,9131021,9131033):
            for gap in (0,2,4,8):
                for bias in (0,4):
                    case=f'ghost_s{seed}_b{bias}' if gap==0 else f'dual_g{gap}_s{seed}_b{bias}'
                    points,meta,ev=make_ghost(seed,bias) if gap==0 else make_dual(seed,gap,bias)
                    write_patch(inputs_root,case,points,meta,ev)
        del ev,points,meta
    start=time.perf_counter();records=[];upstream=[]
    t=time.perf_counter();null_gain_threshold((96,)*8);calibration=time.perf_counter()-t
    for ip in sorted(inputs_root.glob('*.npz')):
        case=ip.stem
        if not fresh and not (PARENT/'states'/(case+'.npz')).exists():continue
        inputs=read(ip)
        if fresh:
            t=time.perf_counter();state=v7.freeze(inputs['xyz_world'],inputs['scan_id'],1.);upstream.append(time.perf_counter()-t)
            npz(dest/'states'/(case+'.npz'),**{k:v for k,v in state.items() if isinstance(v,np.ndarray)})
        else:state=read(PARENT/'states'/(case+'.npz'))
        np.testing.assert_array_equal(state['world'],inputs['xyz_world'])
        for method in METHODS:
            for it in (12,36):
                out,info,art=filter_frozen(state,1.,method,it)
                path=dest/'outputs'/f'{case}__{method}_i{it}.npz';npz(path,**art)
                records.append(dict(case=case,method=f'{method}_i{it}',output=str(path),output_sha256=sha(path),input=str(ip),input_sha256=sha(ip),info=info))
        for variant,budget,label in (('original',0,'old_original'),('local_multistart',6,'old_multistart')):
            if fresh:
                out,info,art=v7.fit_frozen(state,variant=variant,budget=budget,sharing='independent')
                path=dest/'outputs'/f'{case}__{label}.npz';npz(path,xyz_world=out)
            else:
                oldname='original_b0_independent' if variant=='original' else 'local_multistart_b6_independent'
                path=PARENT/'outputs'/f'{case}__{oldname}.npz';info={}
            records.append(dict(case=case,method=label,output=str(path),output_sha256=sha(path),input=str(ip),input_sha256=sha(ip),info=info))
        print(case,'8 new outputs + 2 references sealed',flush=True)
    save(dest/'SEALED_BEFORE_GT.json',records)
    from evaluate_v2 import synthetic_geometry
    sys.path.insert(0,str(ROOT/'exploration_v3'))
    from metrics import structure_metrics
    rows=[]
    for r in records:
        ev=read(inputs_root/'evaluation'/(r['case']+'.eval.npz'));ev.update(json.loads(ev.pop('json').tobytes().decode()))
        meta=json.loads(Path(r['input']).with_suffix('.json').read_text());out=read(r['output'])['xyz_world'];info=r['info']
        row={k:r[k] for k in ('case','method','output','output_sha256')}
        row.update(gap_mm=meta['gap_mm'],seed=meta['seed'],bias_rms_mm=meta['bias_rms_mm'],chosen_k=info.get('k'),true_k=ev['n_true_layers'],seconds=info.get('seconds'),supported_fraction=info.get('supported_fraction'))
        row.update(synthetic_geometry(out,{},ev));row.update(structure_metrics(out,ev));rows.append(row)
    csvsave(dest/'RESULTS.csv',rows);aggregate={}
    for label,gaps in (('all',(0,2,4,8)),('single',(0,)),('gap2',(2,)),('gap4',(4,)),('gap8',(8,))):
        aggregate[label]={}
        for method in sorted(set(r['method'] for r in rows)):
            part=[r for r in rows if r['method']==method and r['gap_mm'] in gaps]
            keys=('surface_accuracy_mean_mm','matched_point_rms_mm','source_group_gap_error_mm','fitted_gap_at_same_xy_error_mm','reference_sample_coverage_1mm','seconds')
            aggregate[label][method]=dict(n=len(part),wrong_k=sum(r['chosen_k']!=r['true_k'] for r in part) if part[0]['chosen_k'] is not None else None,
                **{k:float(np.mean([r[k] for r in part if r.get(k) is not None])) for k in keys if any(r.get(k) is not None for r in part)})
    save(dest/'AGGREGATES.json',aggregate)
    assert all(sha(p)==h for p,h in history.items());assert all(sha(p)==h for p,h in source_hash.items())
    save(dest/'SUMMARY.json',dict(host=socket.gethostname(),conditions=len(rows)//10,new_outputs=len(rows)*8//10,
        fresh_seeds=fresh,all_outputs_before_gt=True,elapsed_seconds=time.perf_counter()-start,
        upstream_seconds=upstream,null_calibration_seconds=calibration,memory_KiB=memory(),historical_files_unchanged=len(history)))
    print(json.dumps(aggregate['all'],indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fresh',action='store_true');main(p.parse_args().fresh)
