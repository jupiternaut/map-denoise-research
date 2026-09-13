"""Real-patch transfer check: measured-reference perturbation, not geometry GT."""
import importlib.util,json,shutil,socket,sys,tempfile,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'exploration_v9'),str(ROOT/'exploration_v6')]
from run_v9 import RUNS,read,npz,save,csvsave,sha,memory
from audit_metrics import rms_mm
from spatial_filter import filter_frozen

def main():
    assert socket.gethostname()=='liekkas'
    dest=Path(tempfile.mkdtemp(prefix='spatial-v11-real-',dir=RUNS));print(dest,flush=True)
    (dest/'outputs').mkdir();(dest/'source').mkdir()
    sources={str(p):sha(p) for p in (HERE/'spatial_filter.py',HERE/'check_real.py')}
    for p in sources:shutil.copyfile(p,dest/'source'/Path(p).name)
    save(dest/'SOURCES.json',sources)
    spec=importlib.util.spec_from_file_location('_v11_real_v7',ROOT/'exploration_v7/algorithm/reassociation.py')
    v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)
    records=[];references={};inputhash={};start=time.perf_counter()
    for p in sorted((RUNS/'patches').glob('*/*.npz')):
        inp=read(p);world=inp['xyz_world'];scans=inp['scan_id'];case=p.stem
        inputhash[str(p)]=sha(p);references[case]=world.copy()
        refstate=v7.freeze(world,scans,1.)
        axis=refstate.get('normal',np.array([0.,0.,1.]))
        _,s=np.unique(scans,return_inverse=True);counts=np.bincount(s)
        b=np.linspace(-1,1,len(counts));b-=counts@b/counts.sum();b*=3/np.sqrt(np.average(b*b,weights=counts))
        for condition in ('zero','shift3mm'):
            x=world.copy() if condition=='zero' else world+b[s,None]*axis[None,:]/1000
            state=refstate if condition=='zero' else v7.freeze(x,scans,1.)
            ip=dest/'outputs'/f'{case}_{condition}__input.npz';npz(ip,xyz_world=x)
            for method in ('identity','old_original','old_multistart','spatial_free','spatial_fixed'):
                t=time.perf_counter()
                if method=='identity':out=x.copy();info=dict(status='IDENTITY')
                elif method.startswith('old_'):
                    out,info,_=v7.fit_frozen(state,variant='original' if method=='old_original' else 'local_multistart',budget=0 if method=='old_original' else 6,sharing='independent')
                elif 'design' not in state:
                    out=x.copy();info=dict(status='UNSUPPORTED')
                else:out,info,_=filter_frozen(state,1.,method,36)
                path=dest/'outputs'/f'{case}_{condition}__{method}.npz';npz(path,xyz_world=out)
                records.append(dict(case=case,condition=condition,method=method,input=str(ip),original_input=str(p),
                    output=str(path),output_sha256=sha(path),seconds=time.perf_counter()-t,info=info))
        print(case,'done',flush=True)
    save(dest/'SEALED_BEFORE_SCORING.json',records);rows=[]
    zero={(r['case'],r['method']):read(r['output'])['xyz_world'] for r in records if r['condition']=='zero'}
    for r in records:
        x=read(r['input'])['xyz_world'];out=read(r['output'])['xyz_world'];ref=references[r['case']]
        rows.append(dict(case=r['case'],condition=r['condition'],method=r['method'],
            measured_reference_rms_mm=rms_mm(out-ref),input_edit_rms_mm=rms_mm(out-x),
            response_to_injection_rms_mm=rms_mm(out-zero[(r['case'],r['method'])]),
            injected_rms_mm=rms_mm(x-ref),seconds=r['seconds']))
    csvsave(dest/'RESULTS.csv',rows);agg={}
    for condition in ('zero','shift3mm'):
        agg[condition]={}
        for method in sorted(set(r['method'] for r in rows)):
            part=[r for r in rows if r['method']==method and r['condition']==condition]
            agg[condition][method]={k:float(np.median([r[k] for r in part])) for k in ('measured_reference_rms_mm','input_edit_rms_mm','response_to_injection_rms_mm','seconds')}
    assert all(sha(p)==h for p,h in inputhash.items());assert all(sha(p)==h for p,h in sources.items())
    save(dest/'AGGREGATES.json',agg);save(dest/'SUMMARY.json',dict(outputs=len(records),patches=6,
        sigma_mm=1.,sigma_note='fixed test setting, not calibrated real sensor noise',elapsed_seconds=time.perf_counter()-start,
        physical_geometry_gt=False,reference='original measured cloud only',memory_KiB=memory(),input_hashes=inputhash))
    print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
