import json,csv,tempfile,shutil
from pathlib import Path
import numpy as np
from normal_followup import estimate_normal,PRIOR
from source_filter import reference_id
from run import HERE,RUNS,PATCHES,rms,sha,ambiguity

def main():
    dest=Path(tempfile.mkdtemp(prefix='observation-plane-v1-',dir=RUNS));print('RUN',dest,flush=True)
    src=dest/'source';src.mkdir()
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,src/p.name)
    original=json.loads((PRIOR/'RESULTS.json').read_text());hashes={str(p):sha(p) for p in PRIOR.iterdir() if p.is_file()}
    candidates=[];diags=[]
    for d in original['diagnostics']:
        with np.load(PRIOR/f"{d['case']}.npz") as a:q=a['input'].copy();scan=a['scan_id'].copy()
        allout={}
        for valid in (False,True):
            for local in (False,True):
                outputs,info=estimate_normal(q,scan,valid,local)
                key=f'valid{int(valid)}_local{int(local)}'
                for name,out in outputs.items():allout[f'{key}_{name}']=out
                diags.append(dict(case=d['case'],patch=d['patch'],kind=d['kind'],configuration=key,**info))
        np.savez_compressed(dest/f"{d['case']}.npz",**allout)
        candidates.append((d,q,scan,allout))
    references={}
    for p in PATCHES.glob('*/*.npz'):
        with np.load(p) as a:references[p.stem]=a['xyz_world'].copy()
    rows=[]
    for d,q,scan,outputs in candidates:
        ref=references[d['patch']];moving=scan!=reference_id(scan)
        for name,out in outputs.items():
            rows.append(dict(case=d['case'],patch=d['patch'],kind=d['kind'],method=name,input_rms_mm=rms(q,ref),
                recovery_rms_mm=rms(out,ref),moving_rms_mm=rms(out[moving],ref[moving]),edit_rms_mm=rms(out,q)))
    summary=[]
    for kind in dict.fromkeys(r['kind'] for r in rows):
        for name in dict.fromkeys(r['method'] for r in rows):
            rr=[r for r in rows if r['kind']==kind and r['method']==name]
            summary.append(dict(kind=kind,method=name,rows=len(rr),mean_mm=float(np.mean([r['recovery_rms_mm'] for r in rr])),
                median_mm=float(np.median([r['recovery_rms_mm'] for r in rr])),
                wins=sum(r['recovery_rms_mm']<r['input_rms_mm']-1e-8 for r in rr),
                edited=sum(r['edit_rms_mm']>1e-8 for r in rr)))
    q,scan,refs=ambiguity();outputs,info=estimate_normal(q,scan,True,True)
    ar=[dict(world=world,method=name,rms_mm=rms(out,ref),gap_mm=float(np.linalg.norm(out[:512].mean(0)-out[512:].mean(0))*1000))
        for world,ref in refs.items() for name,out in outputs.items()]
    for path,h in hashes.items():assert sha(Path(path))==h
    result=dict(rows=rows,summary=summary,diagnostics=diags,ambiguity=ar,ambiguity_info=info,prior_unchanged=True,prior_hashes=hashes,
        scope='Exposed mechanism 2x2 ablation; real geometry references unavailable; 6 unique inputs in each rigid perturbation category, not 18')
    (dest/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (dest/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('SUMMARY',json.dumps(summary),flush=True)
    print('AMBIGUITY',ar,flush=True)
    print('COMPLETE',dest,flush=True)

if __name__=='__main__':main()
