"""Independently recompute V6 direction scores, bounds and historical replay."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from audit_metrics import real_scores,synthetic_scores,axis_bounds,rms_mm

RUNS=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as f:
        result={k:f[k].copy() for k in f.files if k!='json'}
        if 'json' in f:result.update(json.loads(f['json'].tobytes().decode()))
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('save',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    rows=list(csv.DictReader((run/'RESULTS.csv').open()))
    cases={(c['split'],c['case']):c for c in json.loads((run/'CASES.json').read_text())}
    lookup={(r['split'],r['case'],r['method']):r for r in rows}
    differences=[];bounds=[];replays=0;support_checks=0
    for row in rows:
        assert row['ok']=='True',row
        c=cases[row['split'],row['case']]
        current=load(c['input']);ev=load(row['evaluation']);saved=load(row['output'])
        assert digest(row['output'])==row['output_sha256']
        np.testing.assert_array_equal(current['xyz_world'],load(row['original_input'])['xyz_world'])
        source=current if row['split']=='real' else load(c['source_points'])
        np.testing.assert_array_equal(saved['scan_id'],source['scan_id'])
        np.testing.assert_array_equal(saved['source_point_index'],source['source_point_index'])
        out=saved['xyz_world'];mask=saved['support_mask']
        info=json.loads(Path(row['output']).with_suffix('.json').read_text())['info']
        info_mask=np.ones(len(out),bool);info_mask[np.asarray(info['unsupported_point_indices'],int)]=False
        np.testing.assert_array_equal(mask,info_mask)
        np.testing.assert_array_equal(out[~mask],current['xyz_world'][~mask])
        assert abs(float(mask.mean())-float(row['supported_fraction']))<1e-12
        support_checks+=1
        if row['split']=='real':
            reference=ev['reference_measured_xyz_world']
            construction=json.loads(Path(row['evaluation_json']).read_text())
            zero=lookup['real',row['patch']+'__zero',row['method']]
            zero_output=load(zero['output'])['xyz_world']
            scores=real_scores(out,current['xyz_world'],reference,zero_output,construction['construction_normal_world'])
            axis=info['normal_world']
            if axis is not None:
                b=axis_bounds(out,current['xyz_world'],reference,axis,mask)
            else:
                assert not mask.any();np.testing.assert_array_equal(out,current['xyz_world'])
                b=dict(axis_only_floor_mm=rms_mm(current['xyz_world']-reference),
                       axis_and_support_floor_mm=rms_mm(current['xyz_world']-reference),
                       off_axis_edit_rms_mm=0.)
            scores.update(axis_pointwise_lower_bound_xyz_rms_mm=b['axis_only_floor_mm'],
                          support_axis_lower_bound_xyz_rms_mm=b['axis_and_support_floor_mm'],
                          output_off_axis_rms_mm=b['off_axis_edit_rms_mm'])
            bounds.append(dict(case=row['case'],method=row['method'],**b))
        else:
            scores=synthetic_scores(out,ev['gt_clean_xyz_world'],ev['surface_rectangles_mm'],
                                    ev['gt_layer'],ev.get('true_gap_mm'))
        for key,value in scores.items():
            if row.get(key):
                error=abs(value-float(row[key]));assert error<1e-8,(row['case'],key,error)
                differences.append(error)
        if row['method']=='difference':
            previous=RUNS/('real-transfer-v5-zrtui_f5' if row['split']=='real' else 'exploration-v5-development-vdbkdouk')
            np.testing.assert_array_equal(out,load(previous/'outputs'/(row['case']+'__pool_compatible.npz'))['xyz_world'])
            replays+=1
    for p,h in json.loads((run/'PROTECTED_BEFORE.json').read_text()).items():assert digest(p)==h,p
    for p,record in json.loads((run/'SOURCE_MANIFEST.json').read_text()).items():
        assert digest(p)==record['sha256'] and digest(record['snapshot'])==record['sha256']
    report=dict(run=str(run),outputs_checked=len(rows),point_id_and_support_checks=support_checks,
                exact_old_replays=replays,independently_recomputed_metrics=len(differences),
                max_metric_difference_mm=max(differences),axis_floor_cases=len(bounds),
                axis_bounds=bounds,source_and_input_hashes_unchanged=True,
                scope='input-direction intervention and measured-reference recovery, not physical truth')
    with args.save.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    snapshot=args.save.parent/'audit_direction_source.py'
    if not snapshot.exists():shutil.copyfile(Path(__file__),snapshot)
    print(json.dumps({k:v for k,v in report.items() if k!='axis_bounds'},indent=2))


if __name__=='__main__':main()
