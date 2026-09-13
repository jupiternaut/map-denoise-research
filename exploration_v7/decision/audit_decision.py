"""Independent saved-action reconstruction and existing independent metrics."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'exploration_v6'))
from audit_metrics import synthetic_scores

INPUTS=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/repair-v2-oyuie4pl/synthetics/identifiable')


def load(path):
    with np.load(path,allow_pickle=False) as a:
        out={k:a[k].copy() for k in a.files if k!='json'}
        if 'json' in a:out.update(json.loads(a['json'].tobytes().decode()))
        return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=Path);args=parser.parse_args()
    dest=args.run_dir.resolve()
    records=json.loads((dest/'OUTPUTS_SEALED_BEFORE_GT.json').read_text())
    rows={(r['case'],int(r['budget'])):r for r in csv.DictReader((dest/'RESULTS.csv').open())}
    count=0;numeric=0;max_coord=0.;max_metric=0.;arrays_checked=0
    for record in records:
        out=load(record['output']);state=load(record['state']);old=load(record['original_soft_artifact'])
        assert hashlib.sha256(Path(record['output']).read_bytes()).hexdigest()==record['output_sha256']
        for key,value in old.items():
            if key!='xyz_world':np.testing.assert_array_equal(out[key],value);arrays_checked+=1
        expected=state['world'].copy();predicted=state['local'][:,2].copy()
        for group in range(len(out['coefficients'])):
            sorted_rows=np.flatnonzero(out['group_ids'][state['order']]==group)
            predicted[sorted_rows]=state['design'][sorted_rows]@out['coefficients'][group]
        selected=np.flatnonzero(state['support']);original=state['order'][selected]
        expected[original]+=np.outer(predicted[selected]-state['local'][selected,2],state['normal'])/1000.
        delta=float(np.max(abs(expected-out['xyz_world']))*1000.)
        assert delta<1e-8;max_coord=max(max_coord,delta)
        np.testing.assert_array_equal(out['xyz_world'][~out['support_mask']],state['world'][~out['support_mask']])
        ev=load(INPUTS/'evaluation'/(record['case']+'.eval.npz'))
        scores=synthetic_scores(out['xyz_world'],ev['gt_clean_xyz_world'],ev['surface_rectangles_mm'],ev['gt_layer'],ev.get('true_gap_mm'))
        row=rows[(record['case'],record['budget'])]
        for name,value in scores.items():
            if row.get(name):
                delta=abs(value-float(row[name]));assert delta<1e-8
                max_metric=max(max_metric,delta);numeric+=1
        count+=1
    protected=json.loads((dest/'PROTECTED_BEFORE.json').read_text())
    for path,expected in protected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    summary=dict(outputs=count,unchanged_noncoordinate_arrays=arrays_checked,independent_metric_comparisons=numeric,
        max_coordinate_difference_mm=max_coord,max_metric_difference_mm=max_metric,
        protected_files_unchanged=len(protected),passed=True,
        scope='same soft state and labels, changed action only; no evidence of independent real geometry gain')
    with (dest/'INDEPENDENT_AUDIT.json').open('x') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
