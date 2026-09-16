"""Recheck saved outputs without rerunning experiments or changing old E0."""
import hashlib
import io
import json
import unittest

import numpy as np

from common import ROOT, write_json


def main():
    lock=json.loads((ROOT/'SOURCE_LOCK.json').read_text())
    from pathlib import Path
    old=Path(lock['root'])
    current={str(p.relative_to(old)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(old.rglob('*')) if p.is_file()}
    changed=[name for name,h in lock['files'].items() if current.get(name)!=h]
    added=sorted(set(current)-set(lock['files']))
    assert not changed and not added, (changed,added)
    proj=json.loads((ROOT/'results/projection/RESULTS.json').read_text())
    max_delta=0.; n_blocks=0
    for run in proj['runs']:
        a=np.load(ROOT/'results/projection'/(run['key']+'.npz'),allow_pickle=False)
        truth=a['evaluation_truth']
        for name,maskkey in [('original_bh','bh_mask'),('eligible_no_bh','eligible_mask')]:
            mask=a[maskkey]
            for arm in ['identity','projection','argmin']:
                theta=np.zeros(len(truth)) if arm=='identity' else np.where(mask,a[arm],0.)
                error=np.abs(theta-truth)
                measured=run['diagnostics'][name]['metrics'][arm]['ALL']
                delta=max(abs(float(error.mean())-measured['e_after_mean']),
                          abs(float(np.sqrt(np.mean(error**2)))-measured['e_after_rmse']))
                max_delta=max(max_delta,delta)
                assert int(np.count_nonzero(theta))==measured['n_moved']
                assert int(np.sum(error>np.abs(truth)+1e-9))==measured['n_dmg']
                assert delta<1e-12
                n_blocks+=1
    boundary=json.loads((ROOT/'results/boundary/RESULTS.json').read_text())
    grid_harm=0; linear_harm=0; maximum_step_difference=0.
    for run in boundary['runs']:
        a=np.load(ROOT/'results/boundary'/(run['key']+'.npz'),allow_pickle=False)
        truth=a['evaluation_truth'];selected=a['eligible_mask']&a['covered_evaluation']
        for arm,key in [('grid','grid_projection'),('linear','linear_projection')]:
            harm=int(np.sum((np.abs(a[key]-truth)>np.abs(truth)+1e-9)&selected))
            assert harm==run[arm+'_covered']['n_dmg']
            if arm=='grid': grid_harm+=harm
            else: linear_harm+=harm
        maximum_step_difference=max(maximum_step_difference,float(np.max(np.abs(a['grid_projection']-a['linear_projection']))))
    assert linear_harm==0 and maximum_step_difference<=.0500000001
    guards=json.loads((ROOT/'results/guards/metrics.json').read_text())
    guard_blocks=0
    for run in guards:
        a=np.load(ROOT/'results/guards'/(run['case_id']+'.npz'),allow_pickle=False)
        for arm,metrics in run['arms'].items():
            error=np.abs(a['z0']+a[arm+'_theta']-a['truth_z'])
            assert abs(float(error.mean())-metrics['source_sheet_mae_after_mm'])<1e-12
            assert int(np.count_nonzero(a[arm+'_theta']))==metrics['accepted_movement_count']
            assert int(a[arm+'_flags'].sum())==metrics['alarm_count']
            guard_blocks+=1
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    assert outcome.wasSuccessful(),stream.getvalue()
    result=dict(old_files_checked=len(current),old_files_changed=changed,old_files_added=added,
                tests_run=outcome.testsRun,tests_passed=outcome.wasSuccessful(),test_log=stream.getvalue(),
                projection_cases=len(proj['runs']),projection_overall_blocks_checked=n_blocks,
                projection_max_metric_delta=max_delta,guard_cases=len(guards),guard_arm_blocks_checked=guard_blocks,
                boundary_cases=len(boundary['runs']),grid_covered_harm_events=grid_harm,
                linear_covered_harm_events=linear_harm,max_endpoint_change_mm=maximum_step_difference)
    write_json(ROOT/'VERIFICATION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='test_log'},indent=2))


if __name__=='__main__':
    main()
