"""Post-confirmation diagnostic: preserve gate/component orientation at seeding.
Original four methods remain unchanged. This is exposed-data repair, not new confirmation.
"""
import sys,tempfile,json,shutil
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'exploration_v9'),str(ROOT),str(ROOT/'exploration_v3')]
from run_v9 import RUNS,PARENT,INPUTS,read,npz,save,csvsave,sha
from spatial_filter import filter_frozen

def main():
    dest=Path(tempfile.mkdtemp(prefix='spatial-v11-seed-aligned-',dir=RUNS));(dest/'outputs').mkdir();records=[]
    shutil.copyfile(HERE/'spatial_filter.py',dest/'spatial_filter.py');print(dest,flush=True)
    for stage in ('development','confirmation'):
        parent=PARENT if stage=='development' else RUNS/'spatial-v11-confirmation-xjro7mgm'
        inputs=INPUTS if stage=='development' else parent/'inputs'
        for p in sorted((parent/'states').glob('*.npz')):
            state=read(p)
            for it in (12,36):
                out,info,art=filter_frozen(state,1.,'spatial_seed_aligned',it)
                target=dest/'outputs'/f'{stage}_{p.stem}_i{it}.npz';npz(target,**art)
                records.append(dict(stage=stage,case=p.stem,iterations=it,output=str(target),sha256=sha(target),inputs=str(inputs),info=info))
    save(dest/'SEALED_BEFORE_GT.json',records)
    from evaluate_v2 import synthetic_geometry
    from metrics import structure_metrics
    rows=[]
    for r in records:
        inputs=Path(r['inputs']);ev=read(inputs/'evaluation'/(r['case']+'.eval.npz'));ev.update(json.loads(ev.pop('json').tobytes().decode()))
        out=read(r['output'])['xyz_world'];row={k:r[k] for k in ('stage','case','iterations','output')}
        row.update(synthetic_geometry(out,{},ev));row.update(structure_metrics(out,ev));rows.append(row)
    csvsave(dest/'RESULTS.csv',rows)
    agg={}
    for stage in ('development','confirmation'):
        for it in (12,36):
            part=[r for r in rows if r['stage']==stage and r['iterations']==it]
            agg[f'{stage}_i{it}']={k:float(np.mean([r[k] for r in part])) for k in ('surface_accuracy_mean_mm','matched_point_rms_mm')}
    save(dest/'AGGREGATES.json',agg);print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
