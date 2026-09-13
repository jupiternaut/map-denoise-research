"""Read-only numeric audit of sealed runs; writes a separate audit artifact."""
import csv,json,sys,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'exploration_v6'),str(HERE.parent/'exploration_v9')]
from audit_metrics import synthetic_scores
from run_v9 import RUNS,INPUTS,PARENT,sha,read,save

def main():
    results=[]
    for name in ('spectral-filter-v10-4bn33_0y','spectral-filter-v10-dpamdvgb'):
        root=RUNS/name;rows=list(csv.DictReader((root/'RESULTS.csv').open()))
        errors=[]
        for row in rows:
            assert sha(row['output'])==row['output_sha256']
            out=read(row['output']);ev=read(INPUTS/'evaluation'/(row['case']+'.eval.npz'))
            ev.update(json.loads(ev.pop('json').tobytes().decode()))
            score=synthetic_scores(out['xyz_world'],ev['gt_clean_xyz_world'],ev['surface_rectangles_mm'],ev['gt_layer'],float(row['gap_mm']))
            for key in ('surface_accuracy_mean_mm','matched_point_rms_mm','fitted_gap_at_same_xy_error_mm'):
                if key in score:
                    delta=abs(score[key]-float(row[key]));assert delta<1e-8,(key,delta);errors.append(delta)
            state=read(PARENT/'states'/(row['case']+'.npz'))
            pred=out['prediction_world_order_mm'][state['order']]
            expected=state['world'].copy();take=np.flatnonzero(state['support'])
            expected[state['order'][take]]+=(pred[take]-state['local'][take,2])[:,None]*state['normal']/1000.
            np.testing.assert_array_equal(expected,out['xyz_world'])
        protected=json.loads((root/'HISTORY_BEFORE.json').read_text())
        assert all(sha(p)==h for p,h in protected.items())
        # Original source snapshots, not later working copies, remain frozen.
        sources=json.loads((root/'SOURCES.json').read_text())
        for p,h in sources.items():assert sha(root/'source'/Path(p).relative_to(HERE.parent))==h
        results.append(dict(run=name,outputs=len(rows),max_metric_difference=max(errors),
            output_reconstruction_exact=True,historical_files_unchanged=len(protected),source_snapshots_unchanged=True))
    target=Path(tempfile.mkdtemp(prefix='spectral-v10-audit-',dir=RUNS))/'AUDIT.json'
    save(target,results);print(target);print(json.dumps(results,indent=2))
if __name__=='__main__':main()
