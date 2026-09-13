import argparse,importlib.util,json,shutil,socket,sys,tempfile,time,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'exploration_v9'),str(ROOT/'exploration_v10'),str(ROOT/'exploration_v11'),str(ROOT/'exploration_v12'),str(ROOT),str(ROOT/'exploration_v3')]
from run_v9 import RUNS,read,save,npz,csvsave,sha,memory
from run_v10 import protect
from atlas import prepare,decode
from spatial_filter import filter_frozen
from local_filter import corrected_world,estimate_frozen as local_filter
from external import estimate_frozen as external

def main(fresh=False,curved=False):
    assert socket.gethostname()=='liekkas';dest=Path(tempfile.mkdtemp(prefix='atlas-v13-'+('curved-' if curved else 'confirmation-' if fresh else 'development-'),dir=RUNS));print(dest,flush=True)
    for d in ('outputs','source','states'):(dest/d).mkdir()
    history=protect()
    for base in (ROOT/'exploration_v10',ROOT/'exploration_v11',ROOT/'exploration_v12',RUNS/'local-v12-checkpoint-eg4790h7'):
        history.update({str(p):sha(p) for p in base.rglob('*') if p.is_file()})
    save(dest/'HISTORY_BEFORE.json',history);sources={str(p):sha(p) for p in HERE.iterdir() if p.is_file()}
    save(dest/'SOURCES.json',sources)
    for p in sources:shutil.copyfile(p,dest/'source'/Path(p).name)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    (dest/'TESTS.log').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stderr
    spec=importlib.util.spec_from_file_location('_v13_v7',ROOT/'exploration_v7/algorithm/reassociation.py');v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)
    jobs=[];refs={}
    if not fresh and not curved:
        for directory,domain in (('local-v12-confirmation-1ehbo99b','synthetic'),('local-v12-development-bqctbge_','real')):
            previous=json.loads((RUNS/directory/'SEALED_BEFORE_GT.json').read_text())
            for r in previous:
                if r['domain']!=domain:continue
                refs[(r['case'],r['method'])]=r
                if r['method']=='identity':jobs.append({k:v for k,v in r.items() if k not in ('method','output','output_sha256','seconds','info','status')})
    else:
        from generate_synthetics import make_ghost,make_dual
        from schema import write_patch
        from curved_data import make as make_curved
        for seed in ((9134101,9134111) if curved else (9133011,9133021,9133033)):
            for gap in ((0,2,8) if curved else (0,2,4,8)):
                for bias in (0,4):
                    for amp in ((0,2,8) if curved else (0,)):
                        case=(f'ghost_s{seed}_b{bias}' if gap==0 else f'dual_g{gap}_s{seed}_b{bias}')+(f'_a{amp}' if curved else '')
                        points,meta,ev=make_curved(seed,gap,bias,amp) if curved else (make_ghost(seed,bias) if gap==0 else make_dual(seed,gap,bias))
                        write_patch(dest/'inputs',case,points,meta,ev)
                        state=v7.freeze(points['xyz_world'],points['scan_id'],1.);sp=dest/'states'/(case+'.npz');npz(sp,**{k:v for k,v in state.items() if isinstance(v,np.ndarray)})
                        jobs.append(dict(case=case,domain='synthetic',curved=curved,amplitude_mm=amp,input=str(dest/'inputs'/(case+'.npz')),state=str(sp),evaluation=str(dest/'inputs/evaluation'/(case+'.eval.npz'))))
        del points,meta,ev,state
    save(dest/'JOBS.json',jobs);records=[];start=time.perf_counter()
    for job in jobs:
        state=read(job['state']);p=prepare(state,1.)
        choices=[('global_hard',0)]+[(mode,k) for mode in ('atlas_hard','atlas_tied','atlas_relax') for k in (96,256)]
        for mode,k in choices:
            out,info,art=decode(state,p,1.,mode,k);method=mode+(f'_{k}' if k else '')
            path=dest/'outputs'/f'{job["case"]}__{method}.npz';npz(path,xyz_world=out,**art)
            records.append(dict(**job,method=method,output=str(path),output_sha256=sha(path),info=info,status='OK'))
        out=state['world'].copy();out[p['mask']]=p['mm'][p['mask']]/1000;path=dest/'outputs'/f'{job["case"]}__bias_only.npz';npz(path,xyz_world=out)
        records.append(dict(**job,method='bias_only',output=str(path),output_sha256=sha(path),info={},status='OK'))
        for method in ('old_original','whole_spatial','local_spatial_96','apss_2','rimls_2'):
            if not fresh and not curved:
                r=refs[(job['case'],method)];records.append(dict(**job,method=method,output=r['output'],output_sha256=r['output_sha256'],info=r['info'],status='REFERENCE'))
            else:
                if method=='whole_spatial':out,info,_=filter_frozen(state,1.,'spatial_free',12)
                elif method=='local_spatial_96':out,info,_=local_filter(state,1.,96,'spatial')
                elif method=='old_original':
                    inp=read(job['input']);full=v7.freeze(inp['xyz_world'],inp['scan_id'],1.);out,info,_=v7.fit_frozen(full,variant='original',budget=0,sharing='independent')
                else:out,info,_=external(state,method.split('_')[0],2.)
                path=dest/'outputs'/f'{job["case"]}__{method}.npz';npz(path,xyz_world=out)
                records.append(dict(**job,method=method,output=str(path),output_sha256=sha(path),info=info,status='OK'))
        print(job['case'],'sealed',flush=True)
    save(dest/'SEALED_BEFORE_GT.json',records)
    from evaluate_v2 import synthetic_geometry
    from metrics import structure_metrics
    rows=[]
    for r in records:
        out=read(r['output'])['xyz_world'];row={k:r[k] for k in ('case','domain','method','output','output_sha256')}
        info=r['info'];row.update(prepare_seconds=info.get('prepare_seconds'),decode_seconds=info.get('decode_seconds'),association_change_fraction=info.get('association_change_fraction'),fallback_components=info.get('fallback_components'))
        if r['domain']=='synthetic':
            ev=read(r['evaluation']);ev.update(json.loads(ev.pop('json').tobytes().decode()))
            row['gap_mm']=ev['true_gap_mm']
            if r.get('curved'):
                from curved_data import scores
                row['amplitude_mm']=r['amplitude_mm'];row.update(scores(out,ev))
            else:row.update(synthetic_geometry(out,{},ev));row.update(structure_metrics(out,ev))
        else:
            ref=read(r['original'])['xyz_world'];row.update(condition=r['condition'],patch=r['patch'],measured_reference_rms_mm=float(np.sqrt(np.mean(np.sum((out-ref)**2,axis=1)))*1000))
        rows.append(row)
    csvsave(dest/'RESULTS.csv',rows);agg={}
    for group in ('synthetic','real_zero','real_shift3mm'):
        agg[group]={}
        for method in sorted(set(r['method'] for r in rows)):
            part=[r for r in rows if r['method']==method and (r['domain']=='synthetic' if group=='synthetic' else r['domain']=='real' and r['condition']==group[5:])]
            keys=('surface_accuracy_mean_mm','matched_point_rms_mm','source_group_gap_error_mm','reference_sample_coverage_1mm','measured_reference_rms_mm','prepare_seconds','decode_seconds','association_change_fraction')
            agg[group][method]={k:float((np.mean if group=='synthetic' else np.median)([r[k] for r in part if r.get(k) is not None])) for k in keys if any(r.get(k) is not None for r in part)}
    save(dest/'AGGREGATES.json',agg)
    assert all(sha(p)==h for p,h in history.items());assert all(sha(p)==h for p,h in sources.items())
    save(dest/'SUMMARY.json',dict(jobs=len(jobs),records=len(rows),fresh_seeds=fresh,curved=curved,elapsed_seconds=time.perf_counter()-start,memory_KiB=memory(),historical_files_unchanged=len(history)))
    print(json.dumps(agg,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fresh',action='store_true');parser.add_argument('--curved',action='store_true');args=parser.parse_args();main(args.fresh,args.curved)
