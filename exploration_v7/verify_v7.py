"""Audit saved V7 fits/metrics without importing any estimator."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'exploration_v6'))
from audit_metrics import synthetic_scores

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def payload(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files if k != 'json'}
        if 'json' in data: result.update(json.loads(data['json'].tobytes().decode()))
    return result

def fit_from_groups(state, world_groups, shared):
    active, order = state['active'], state['order']
    labels = np.asarray(world_groups)[order][active]
    _, groups = np.unique(labels, return_inverse=True)
    if (labels < 0).any(): raise AssertionError('unassigned active row')
    count = int(groups.max())+1
    design = np.zeros((len(active), count+(2 if shared else 2*count)))
    design[np.arange(len(active)), groups] = 1.
    for col in range(2):
        columns = np.full(len(active), count+col) if shared else count+2*groups+col
        design[np.arange(len(active)), columns] = state['design'][active, col+1]
    root_w = np.sqrt(state['weights'][active])
    beta = np.linalg.lstsq(design*root_w[:, None], state['corrected'][active]*root_w, rcond=1e-12)[0]
    predicted = state['corrected'].copy()
    predicted[active] = design@beta
    ordered = state['ordered_world'].copy()
    support = state['support']
    ordered[support] += (predicted[support]-state['local'][support, 2])[:, None]*state['normal']/1000.
    output = state['world'].copy()
    output[order] = ordered
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    rows = list(csv.DictReader((run/'RESULTS.csv').open()))
    manifest = {r['case']: r for r in json.loads((run/'INPUT_MANIFEST.json').read_text())}
    for path, value in json.loads((run/'SOURCE_MANIFEST.json').read_text()).items():
        if digest(path) != value: raise AssertionError('source changed: '+path)
    metric_diffs, fit_diffs, mass_diffs, objective_increases = [], [], [], []
    for row in rows:
        item = manifest[row['case']]
        assert digest(item['input']) == item['input_sha256']
        assert digest(item['evaluation']) == item['evaluation_sha256']
        assert digest(row['output']) == row['output_sha256']
        source, ev = payload(item['input']), payload(item['evaluation'])
        state = payload(run/'states'/(row['case']+'.npz'))
        result = payload(row['output']); out = result['xyz_world']
        np.testing.assert_array_equal(result['scan_id'], source['scan_id'])
        np.testing.assert_array_equal(result['source_point_index'], source['source_point_index'])
        support = result['support_mask'].astype(bool)
        np.testing.assert_array_equal(out[~support], source['xyz_world'][~support])
        if row['method'] != 'identity':
            expected = np.zeros(len(out), bool); expected[state['order']] = state['support']
            np.testing.assert_array_equal(support, expected)
            fit = fit_from_groups(state, result['group_ids'], row['method'].endswith('_shared'))
            diff = float(1000*np.max(abs(fit-out)))
            if diff > 1e-7: raise AssertionError('independent final-fit mismatch: '+str(diff))
            fit_diffs.append(diff)
            if 'responsibility' in result:
                responsibilities = result['responsibility']
                active = result['candidate_mask'].any(axis=1)
                assert responsibilities.shape == result['candidate_mask'].shape
                if active.any():
                    mass = float(np.max(abs(responsibilities[active].sum(axis=1)-1.)))
                    if mass > 1e-10: raise AssertionError('responsibility mass not one')
                    mass_diffs.append(mass)
                assert np.isfinite(responsibilities).all() and np.min(responsibilities) >= 0.
                np.testing.assert_array_equal(responsibilities[~result['candidate_mask']], 0.)
            if row['method'].startswith('reassociate_') and 'objective_trace' in result:
                trace = result['objective_trace']
                if trace.ndim == 1 and len(trace) > 1:
                    increase = float(np.max(np.diff(trace)))
                    objective_increases.append(increase)
                    if increase > 1e-7*max(1., abs(float(trace[0]))):
                        raise AssertionError('conditional soft objective increased')
        scores = synthetic_scores(out, ev['gt_clean_xyz_world'], ev['surface_rectangles_mm'],
                                  ev['gt_layer'], ev.get('true_gap_mm'))
        for key, value in scores.items():
            if row.get(key):
                delta = abs(value-float(row[key]))
                if delta > 1e-8: raise AssertionError('independent metric mismatch '+key)
                metric_diffs.append(delta)
    result = dict(outputs_checked=len(rows), independent_fits=len(fit_diffs),
         max_fit_coordinate_difference_mm=max(fit_diffs, default=0.),
         independently_recomputed_metrics=len(metric_diffs), max_metric_difference_mm=max(metric_diffs, default=0.),
         max_responsibility_mass_error=max(mass_diffs, default=0.),
         max_soft_objective_step_increase=max(objective_increases, default=0.),
         unchanged_input_and_source_hashes=True,
         scope='saved final fits, constraints and scores; no claim of physical accuracy or novelty')
    with (run/'INDEPENDENT_AUDIT.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
