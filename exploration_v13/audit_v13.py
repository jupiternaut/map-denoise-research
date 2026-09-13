import csv,json,sys,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'exploration_v9'),str(ROOT/'exploration_v6'),str(ROOT/'exploration_v12')]
from run_v9 import RUNS,read,save,sha
from audit_metrics import synthetic_scores,rms_mm
from local_filter import corrected_world

def exact_curved(out,ev):
    # Independent scalar polynomial roots, instead of evaluator Newton steps.
    q=out*1000;a=ev['parabola_a_per_mm'];best=np.full(len(q),np.inf)
    for c,lo,hi,ylo,yhi in ev['surface_rectangles_mm']:
        for i,(x,y,z) in enumerate(q):
            if a==0:options=[np.clip(x,lo,hi)]
            else:
                roots=np.roots([2*a*a,0.,1+2*a*(c-z),-x])
                options=[lo,hi]+[r.real for r in roots if abs(r.imag)<1e-8 and lo<=r.real<=hi]
            d=min((u-x)**2+(a*u*u+c-z)**2 for u in options)+(np.clip(y,ylo,yhi)-y)**2
            best[i]=min(best[i],np.sqrt(d))
    return float(best.mean())

def main():
    target=Path(tempfile.mkdtemp(prefix='atlas-v13-audit-',dir=RUNS));reports=[]
    for name in ('atlas-v13-development-ff24px9y','atlas-v13-confirmation-q9ixevii','atlas-v13-curved-ugoaiy7_'):
        root=RUNS/name;records=json.loads((root/'SEALED_BEFORE_GT.json').read_text())
        rows={(r['case'],r['method']):r for r in csv.DictReader((root/'RESULTS.csv').open())};worst=0.;sampled_curved=0
        for record in records:
            assert sha(record['output'])==record['output_sha256'];data=read(record['output']);out=data['xyz_world']
            state=read(record['state']);_,mask=corrected_world(state)
            np.testing.assert_array_equal(out[~mask],state['world'][~mask]);row=rows[(record['case'],record['method'])]
            if record['domain']=='real':scores=dict(measured_reference_rms_mm=rms_mm(out-read(record['original'])['xyz_world']))
            else:
                ev=read(record['evaluation']);ev.update(json.loads(ev.pop('json').tobytes().decode()))
                if record.get('curved'):
                    scores=dict(matched_point_rms_mm=rms_mm(out-ev['gt_clean_xyz_world']))
                    # Two prechosen methods over every curvature condition.
                    if record['method'] in ('atlas_relax_96','whole_spatial'):
                        scores['surface_accuracy_mean_mm']=exact_curved(out,ev);sampled_curved+=1
                else:scores=synthetic_scores(out,ev['gt_clean_xyz_world'],ev['surface_rectangles_mm'],ev['gt_layer'],ev['true_gap_mm'])
            for k,v in scores.items():
                if k in row and row[k]:worst=max(worst,abs(v-float(row[k])))
        assert worst<1e-8
        history=json.loads((root/'HISTORY_BEFORE.json').read_text());assert all(sha(p)==h for p,h in history.items())
        for p,h in json.loads((root/'SOURCES.json').read_text()).items():assert sha(root/'source'/Path(p).name)==h
        reports.append(dict(run=name,outputs_checked=len(records),max_metric_difference_mm=worst,
            curved_surface_independent_roots_outputs=sampled_curved,support_unchanged=True,historical_files_unchanged=len(history)))
        print(name,'checked',flush=True)
    save(target/'AUDIT.json',reports);print(target);print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
