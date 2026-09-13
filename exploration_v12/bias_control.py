"""No surface model: isolate shared frame-bias correction from denoising."""
import csv,json,sys,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'exploration_v9'),str(ROOT/'exploration_v6')]
from run_v9 import RUNS,read,save,npz,csvsave,sha
from audit_metrics import synthetic_scores,rms_mm
from local_filter import corrected_world

def main():
    parent=Path(sys.argv[1]);records=json.loads((parent/'SEALED_BEFORE_GT.json').read_text())
    target=Path(tempfile.mkdtemp(prefix='local-v12-bias-control-',dir=RUNS));(target/'outputs').mkdir();sealed=[];comparisons=[]
    for r in records:
        if r['status']!='OK':continue
        state=read(r['state']);q,mask=corrected_world(state);out=state['world'].copy();out[mask]=q[mask]
        candidate=read(r['output'])['xyz_world'];delta=np.linalg.norm(candidate-out,axis=1)*1000
        comparisons.append(dict(case=r['case'],domain=r['domain'],method=r['method'],
            additional_edit_rms_mm=rms_mm(candidate-out),fraction_more_than_001mm=float(np.mean(delta>.01))))
        if r['method']=='identity':
            p=target/'outputs'/(r['case']+'.npz');npz(p,xyz_world=out)
            sealed.append(dict(**{k:v for k,v in r.items() if k not in ('output','output_sha256','method','info')},output=str(p),output_sha256=sha(p),method='bias_only'))
    save(target/'SEALED_BEFORE_SCORING.json',sealed);rows=[]
    for r in sealed:
        out=read(r['output'])['xyz_world'];row={k:r[k] for k in ('case','domain','method','output')}
        if r['domain']=='synthetic':
            ev=read(r['evaluation']);ev.update(json.loads(ev.pop('json').tobytes().decode()))
            row.update(synthetic_scores(out,ev['gt_clean_xyz_world'],ev['surface_rectangles_mm'],ev['gt_layer'],ev['true_gap_mm']))
        else:row.update(condition=r['condition'],measured_reference_rms_mm=rms_mm(out-read(r['original'])['xyz_world']))
        rows.append(row)
    csvsave(target/'RESULTS.csv',rows);csvsave(target/'ADDITIONAL_EDITS.csv',comparisons)
    agg={}
    for group in ('synthetic','real_zero','real_shift3mm'):
        part=[r for r in rows if r['domain']=='synthetic'] if group=='synthetic' else [r for r in rows if r['domain']=='real' and r['condition']==group[5:]]
        agg[group]={k:float((np.mean if group=='synthetic' else np.median)([r[k] for r in part])) for k in ('surface_accuracy_mean_mm','matched_point_rms_mm','measured_reference_rms_mm') if k in part[0]}
    save(target/'AGGREGATES.json',agg);print(target);print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
