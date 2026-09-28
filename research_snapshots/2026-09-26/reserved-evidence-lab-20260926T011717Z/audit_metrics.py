"""Independent metric arithmetic, equal-scene aggregation and random-control audit.

Run after selection and evaluation are sealed. This evaluator-only audit reads
archived error arrays, never uses them to change the frozen selectors.
"""
from common import *
import csv

FIELDS = ('source_MSE_mm2', 'source_MAE_mm', 'source_p95_mm',
          'improved_fraction', 'harmed_fraction', 'accepted_fraction',
          'accepted_support_fraction', 'moved_fraction', 'move_RMS_mm',
          'benefit_sum_mm2', 'harm_sum_mm2')


def independent_metrics(d0, d1, keep, movement, mask):
    revised = d0.copy()
    revised[mask] = d1[mask]
    active = revised[keep]
    original = d0[keep]
    squared_gain = original*original-active*active
    moved_squared = np.zeros(len(mask))
    moved_squared[mask] = movement[mask]
    return dict(source_MSE_mm2=float(np.dot(active, active)/len(active)),
        source_MAE_mm=float(sum(active)/len(active)), source_p95_mm=float(np.quantile(active, .95)),
        improved_fraction=float(np.count_nonzero(original-active > .1)/len(active)),
        harmed_fraction=float(np.count_nonzero(active-original > .1)/len(active)),
        accepted_fraction=float(np.count_nonzero(mask)/len(mask)),
        accepted_support_fraction=float(np.count_nonzero(mask[keep])/len(active)),
        moved_fraction=float(np.count_nonzero(moved_squared > 1e-14)/len(mask)),
        move_RMS_mm=float(np.sqrt(moved_squared.sum()/len(mask))),
        benefit_sum_mm2=float(squared_gain[squared_gain > 0].sum()),
        harm_sum_mm2=float(-squared_gain[squared_gain < 0].sum()))


def main():
    check_host(); verify_seal(ROOT/'evaluation')
    with (ROOT/'evaluation/METRICS.csv').open() as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        rows.append({**r, 'scene': int(r['scene']), **{k: float(r[k]) for k in FIELDS}})
    keys = [(r['scene'], r['roi'], r['condition'], r['arm']) for r in rows]
    assert len(set(keys)) == len(rows)
    lookup = dict(zip(keys, rows))
    summary = json.loads((ROOT/'evaluation/SUMMARY.json').read_text())
    arms = sorted({r['arm'] for r in rows})
    assert len(arms) == 41 and len(rows) == 60*len(arms)
    maxima = {k: 0. for k in FIELDS}
    random_checks = 0
    priority_counts = {}
    for sid in SCENES:
        for path in cases_for_scene(sid):
            rid, condition = path.name.split('__')
            with np.load(PREV/'evaluation'/path.name/'point_errors.npz', allow_pickle=False) as arr:
                d0, d1, support, movement = (arr[k] for k in ('d0', 'd1', 'support', 'movement_squared'))
            assert support.dtype == bool and support.any()
            assert all(np.isfinite(x).all() for x in (d0, d1, movement))
            with np.load(ROOT/'inference'/f'scan{sid}'/path.name/'decisions.npz', allow_pickle=False) as arr:
                masks = {k: arr[k] for k in arr.files}
            with np.load(ROOT/'evaluation'/path.name/'diagnostic_decisions.npz', allow_pickle=False) as arr:
                masks.update({k: arr[k] for k in arr.files})
            assert set(masks) == set(arms)
            assert np.array_equal(masks['oracle_fixed_A'], d1 < d0)
            target = masks['reserved_aug__balanced']
            bins = np.searchsorted(np.array([1e-7, .25, .5, 1., 2., 3., 4., 6.000001]),
                                   np.sqrt(movement), side='right')
            for arm, mask in masks.items():
                assert mask.dtype == bool and mask.shape == d0.shape
                if arm.startswith('random__'):
                    mode = arm.split('__')[1]
                    groups = support.astype(int)+2*(bins if mode == 'bin' else np.zeros(len(mask), int))
                    for g in np.unique(groups):
                        assert mask[groups == g].sum() == target[groups == g].sum()
                    random_checks += 1
                if arm.endswith('__native_priority'):
                    counts = priority_counts.setdefault(condition, {}).setdefault(arm,
                        dict(rows=0, supported_rows=0, accepted_rows=0, actual_moved_rows=0,
                             accepted_supported_rows=0, actual_moved_supported_rows=0))
                    actual = mask & (movement > 1e-14)
                    for key, value in dict(rows=len(mask), supported_rows=support.sum(),
                            accepted_rows=mask.sum(), actual_moved_rows=actual.sum(),
                            accepted_supported_rows=(mask & support).sum(),
                            actual_moved_supported_rows=(actual & support).sum()).items():
                        counts[key] += int(value)
                result = independent_metrics(d0, d1, support, movement, mask)
                saved = lookup[sid, rid, condition, arm]
                assert int(saved['n_rows']) == len(d0) and int(saved['n_source']) == support.sum()
                for key in FIELDS:
                    error = abs(result[key]-saved[key])
                    maxima[key] = max(maxima[key], error)
                    assert np.isclose(result[key], saved[key], rtol=1e-12, atol=1e-10), (path.name, arm, key, error)
                oracle_mse = np.minimum(d0[support]**2, d1[support]**2).mean()
                assert saved['source_MSE_mm2'] >= oracle_mse-1e-12
    assert random_checks == 1200
    for condition in CONDS:
        for arm in arms:
            selected = [r for r in rows if r['condition'] == condition and r['arm'] == arm]
            assert len(selected) == 12
            scene_means = []
            for sid in SCENES:
                cell = [r for r in selected if r['scene'] == sid]
                assert len(cell) == 4
                mean = {k: sum(r[k] for r in cell)/4 for k in FIELDS}
                scene_means.append(mean)
                for key in FIELDS:
                    assert np.isclose(mean[key], summary['per_scene'][str(sid)][condition][arm][key], rtol=1e-12, atol=1e-10)
            aggregate = summary['exposed_replay'][condition][arm]
            for key in FIELDS:
                assert np.isclose(sum(m[key] for m in scene_means)/3, aggregate[key], rtol=1e-12, atol=1e-10)
            diff = np.array([r['source_MSE_mm2']-lookup[r['scene'], r['roi'], condition, 'identity']['source_MSE_mm2'] for r in selected])
            assert aggregate['wins'] == int((diff < -1e-12).sum())
            assert aggregate['ties'] == int((abs(diff) <= 1e-12).sum())
            assert aggregate['losses'] == int((diff > 1e-12).sum())
            before = sum(lookup[r['scene'], r['roi'], condition, 'identity']['source_MSE_mm2'] for r in selected)
            after = sum(r['source_MSE_mm2'] for r in selected)
            assert np.isclose(100*(1-after/before), aggregate['MSE_gain_percent'], rtol=1e-12, atol=1e-10)
    save_json(ROOT/'AUDIT_METRICS.json', dict(status='PASS', rows=len(rows), methods=len(arms),
        cases=60, numeric_fields=list(FIELDS), maximum_numeric_discrepancy=maxima,
        matched_control_checks=random_checks, summary_cells=5*len(arms),
        native_priority_actual_edit_counts=priority_counts,
        aggregation='four ROIs per scene, three scenes equal weight',
        used_error_arrays_only_after_model_and_inference_lock=True,
        no_selector_changes=True, source_sha256=sha(ROOT/'audit_metrics.py')))


if __name__ == '__main__':
    main()
