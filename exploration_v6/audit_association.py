"""Independent saved-output, partition, fit and metric audit of V6 oracle runs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from audit_metrics import synthetic_scores


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload(path):
    with np.load(path,allow_pickle=False) as f:
        result={k:f[k].copy() for k in f.files if k!='json'}
        if 'json' in f:result.update(json.loads(f['json'].tobytes().decode()))
        return result


def recompute_fit(state, labels, shared):
    active=state['active'];order=state['order']
    original=state['node_to_group'][state['assignment'][active]]
    if labels is None:groups=original
    else:_,groups=np.unique(np.c_[original,labels[order][active]],axis=0,return_inverse=True)
    count=int(groups.max())+1
    design=np.zeros((len(active),count+(2 if shared else 2*count)))
    design[np.arange(len(active)),groups]=1
    xy=state['design'][active,1:]
    for column in range(2):
        indices=np.full(len(active),count+column) if shared else count+2*groups+column
        design[np.arange(len(active)),indices]=xy[:,column]
    root_weight=np.sqrt(state['weights'][active])
    coef=np.linalg.lstsq(design*root_weight[:,None],state['corrected'][active]*root_weight,rcond=1e-12)[0]
    predicted=state['corrected'].copy();predicted[active]=design@coef
    ordered=state['ordered_world'].copy();support=state['support']
    ordered[support]+=(predicted[support]-state['local'][support,2])[:,None]*state['normal']/1000
    result=state['world'].copy();result[order]=ordered
    group_original=np.full(len(order),-1,dtype=int);group_original[order[active]]=groups
    return result,group_original


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('save',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    rows=list(csv.DictReader((run/'RESULTS.csv').open()))
    manifest={r['case']:r for r in json.loads((run/'INPUT_MANIFEST.json').read_text())}
    differences=[];fits=[];support_checks=0
    for row in rows:
        entry=manifest[row['case']]
        assert digest(entry['input'])==entry['input_sha256']
        assert digest(entry['evaluation'])==entry['evaluation_sha256']
        source=payload(entry['input']);ev=payload(entry['evaluation'])
        state=payload(run/'states'/(row['case']+'.npz'))
        saved=payload(row['output']);out=saved['xyz_world']
        assert digest(row['output'])==row['output_sha256']
        np.testing.assert_array_equal(saved['scan_id'],source['scan_id'])
        np.testing.assert_array_equal(saved['source_point_index'],source['source_point_index'])
        expected_mask=np.zeros(len(out),bool);expected_mask[state['order']]=state['support']
        np.testing.assert_array_equal(saved['support_mask'],expected_mask)
        np.testing.assert_array_equal(saved['active_original_indices'],state['order'][state['active']])
        np.testing.assert_array_equal(out[~expected_mask],source['xyz_world'][~expected_mask]);support_checks+=1
        oracle=row['arm'].startswith('oracle_');shared=row['arm'].endswith('_shared')
        fit,groups=recompute_fit(state,ev['gt_layer'] if oracle else None,shared)
        np.testing.assert_array_equal(groups,saved['group_assignment'])
        maxfit=float(np.max(abs(fit-out))*1000);assert maxfit<1e-8;fits.append(maxfit)
        scores=synthetic_scores(out,ev['gt_clean_xyz_world'],ev['surface_rectangles_mm'],
                                ev['gt_layer'],ev.get('true_gap_mm'))
        for key,value in scores.items():
            if row.get(key):
                difference=abs(float(row[key])-value);assert difference<1e-8
                differences.append(difference)
    for name in ('SOURCE_MANIFEST.json','PROTECTED_BEFORE.json'):
        for path,h in json.loads((run/name).read_text()).items():assert digest(path)==h,path
    summary=dict(run=str(run),outputs_checked=len(rows),support_and_id_checks=support_checks,
        independently_reconstructed_fits=len(fits),max_fit_coordinate_difference_mm=max(fits),
        independently_recomputed_metrics=len(differences),max_metric_difference_mm=max(differences),
        input_and_source_hashes_unchanged=True,
        scope='saved conditional fits and metrics verified; oracle is not a deployable estimator')
    with args.save.open('x') as f:json.dump(summary,f,indent=2)
    code_snapshot=args.save.parent/'audit_association_source.py'
    if not code_snapshot.exists():shutil.copyfile(Path(__file__),code_snapshot)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
