"""Read-only exact endpoint and enclosure accounting after COMPLETE + PASS."""
from collections import Counter
from fractions import Fraction as Q
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/structured-v1'
UNITS, METHODS = ('raw', 'clip'), ('keep', 'wide', 'calibrated', 'joint')


def load(path):
    return json.loads(path.read_text())


def exact(value):
    return Q(value['numerator'], value['denominator']) if isinstance(value, dict) else Q(value)


def mean(values):
    return sum(values, Q(0)) / len(values)


def intervals(outer):
    return [tuple(map(exact, pair)) for pair in outer['retained_intervals']]


def included(inner, outer):
    # Registered union intervals are sorted and merged closed intervals.
    return all(any(c <= a <= b <= d for c, d in outer) for a, b in inner)


def analysis():
    status = load(OUTPUT / 'RUN_STATUS.json')
    verify = load(OUTPUT / 'verification/VERIFY.json')
    assert status['status'] == 'COMPLETE' and verify['status'] == 'PASS'
    rows = load(OUTPUT / 'evaluation/ROWS.json')
    coverage = load(OUTPUT / 'evaluation/COVERAGE.json')
    outers = load(OUTPUT / 'outers.json')
    summary = load(OUTPUT / 'evaluation/SUMMARY.json')
    assert len(rows) == 3888 and len(coverage) == 432 and len(outers) == 108
    main = []
    for unit in UNITS:
        for method in METHODS:
            rs = [r for r in rows if r['unit_id'] == unit and r['method'] == method]
            small = [r for r in rs if r['state_group'] == 'small']
            correct = [r for r in rs if r['state_group'] == 'correct']
            negative = [r for r in small if r['case'] == 'equal_full_negative']
            cov = [c for c in coverage if c['unit_id'] == unit and c['method'] == method]
            errors = [Q(r['ae_mm']) for r in small]
            initial = [Q(r['initial_ae_mm']) for r in small]
            mae, mse = mean(errors), mean([e * e for e in errors])
            mae0, mse0 = mean(initial), mean([e * e for e in initial])
            main.append({'unit': unit, 'method': method, 'offset_n': len(small),
                         'improved': sum(e < a for e, a in zip(errors, initial)),
                         'same': sum(e == a for e, a in zip(errors, initial)),
                         'harm': sum(e > a for e, a in zip(errors, initial)),
                         'mae_exact_mm': str(mae), 'mse_exact_mm2': str(mse),
                         'mae_mm': float(mae), 'mse_mm2': float(mse),
                         'mae_reduction_vs_keep_percent': float(100 * (mae0 - mae) / mae0),
                         'mse_reduction_vs_keep_percent': float(100 * (mse0 - mse) / mse0),
                         'correct_keep': sum(r['status'] == 'KEEP' for r in correct),
                         'negative_keep': sum(r['status'] == 'KEEP' for r in negative),
                         'covered': sum(c['covered'] for c in cov),
                         'generating_branch_covered': sum(c['generating_branch_covered'] for c in cov),
                         'latent_material_covered': None if method not in ('calibrated', 'joint') else sum(c['latent_material_covered'] for c in cov),
                         'latent_bias_covered': None if method != 'joint' else sum(c['latent_bias_covered'] for c in cov),
                         'incompatible': sum(r['status'] == 'INCOMPATIBLE' for r in rs),
                         'mean_hull_width_mm': float(mean([Q(c['hull_width_mm']) for c in cov if c['hull_width_mm'] is not None]))})
    pairs, exceptions, grouped_pairs = [], [], []
    lookup = {(r['unit_id'], r['base_task_id'], r['method']): r for r in rows}
    for unit in UNITS:
        ids = sorted({r['base_task_id'] for r in rows if r['unit_id'] == unit and r['state_group'] == 'small'})
        for left, right in (('wide', 'calibrated'), ('wide', 'joint'), ('calibrated', 'joint')):
            subset = [(lookup[unit, i, left], lookup[unit, i, right]) for i in ids]
            differences = [Q(a['ae_mm']) - Q(b['ae_mm']) for a, b in subset]
            sqdiff = [Q(a['se_mm2']) - Q(b['se_mm2']) for a, b in subset]
            pairs.append({'unit': unit, 'left': left, 'right': right, 'n': len(subset),
                          'right_better': sum(g > 0 for g in differences),
                          'same': sum(g == 0 for g in differences), 'right_worse': sum(g < 0 for g in differences),
                          'identical_outputs': sum(a['output'] == b['output'] for a, b in subset),
                          'mean_mae_gain_mm': float(mean(differences)), 'mean_mse_gain_mm2': float(mean(sqdiff)),
                          'newly_repaired_vs_keep': sum(Q(a['ae_mm']) == Q(a['initial_ae_mm']) and Q(b['ae_mm']) < Q(b['initial_ae_mm']) for a, b in subset),
                          'lost_repairs_vs_keep': sum(Q(a['ae_mm']) < Q(a['initial_ae_mm']) and Q(b['ae_mm']) == Q(b['initial_ae_mm']) for a, b in subset)})
            for a, b in subset:
                if Q(b['ae_mm']) > Q(a['ae_mm']):
                    exceptions.append({'unit': unit, 'left': left, 'right': right, 'base_task_id': a['base_task_id'],
                                       'left_row': a, 'right_row': b})
            for field in ('case', 'seed'):
                for value in sorted({a[field] for a, _b in subset}):
                    group = [(a, b) for a, b in subset if a[field] == value]
                    grouped_pairs.append({'unit': unit, 'left': left, 'right': right, 'grouping': field,
                                          'condition': value, 'n': len(group),
                                          'mean_mae_gain_mm': float(mean([Q(a['ae_mm']) - Q(b['ae_mm']) for a, b in group])),
                                          'mean_mse_gain_mm2': float(mean([Q(a['se_mm2']) - Q(b['se_mm2']) for a, b in group])),
                                          'new_repairs': sum(Q(a['ae_mm']) == Q(a['initial_ae_mm']) and Q(b['ae_mm']) < Q(b['initial_ae_mm']) for a, b in group),
                                          'lost_repairs': sum(Q(a['ae_mm']) < Q(a['initial_ae_mm']) and Q(b['ae_mm']) == Q(b['initial_ae_mm']) for a, b in group)})
    sets, calibration, joint_bias = [], [], []
    for unit in UNITS:
        records = [r for r in outers if r['unit_id'] == unit]
        for left, right in (('wide', 'calibrated'), ('wide', 'joint'), ('calibrated', 'joint')):
            counter = Counter()
            branch_counter = Counter()
            for record in records:
                a, b = (record['arms'][method]['outer'] for method in (left, right))
                ah, bh = tuple(map(exact, a['hull'])), tuple(map(exact, b['hull']))
                counter['same_hull' if ah == bh else 'right_hull_nested' if ah[0] <= bh[0] <= bh[1] <= ah[1] else 'right_hull_not_nested'] += 1
                counter['right_set_subset' if included(intervals(b), intervals(a)) else 'right_set_not_subset'] += 1
                for branch in ('none', 'partial', 'full'):
                    aa, bb = (record['arms'][method]['branches'][branch] for method in (left, right))
                    branch_intervals = []
                    for body in (aa, bb):
                        ids = set(body['retained_ids'])
                        pieces = sorted((exact(n['lo']), exact(n['hi'])) for n in body['trace'] if n['id'] in ids)
                        merged = []
                        for lo, hi in pieces:
                            if merged and lo <= merged[-1][1]:
                                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
                            else:
                                merged.append((lo, hi))
                        branch_intervals.append(merged)
                    branch_counter['right_subset' if included(branch_intervals[1], branch_intervals[0]) else 'right_not_subset'] += 1
            sets.append({'unit': unit, 'left': left, 'right': right, 'observations': len(records),
                         'counts': dict(counter), 'branch_counts': dict(branch_counter)})
        for view in range(3):
            values = [tuple(map(exact, r['arms']['calibrated']['calibration_intervals'][view])) for r in records]
            assert all(r['arms']['calibrated']['calibration'] == r['arms']['joint']['calibration'] for r in records)
            calibration.append({'unit': unit, 'view': view, 'n': len(values),
                                'point_intervals': sum(a == b for a, b in values),
                                'unchanged_prior': sum((a, b) == (Q(0), Q(1, 50)) for a, b in values),
                                'mean_t_width': float(mean([b - a for a, b in values])),
                                'distribution': dict(Counter(str((a, b)) for a, b in values))})
        checks = leaves = tighter = 0
        for record in records:
            prior = [tuple(map(exact, pair)) for pair in record['arms']['joint']['calibration_intervals']]
            for branch in record['arms']['joint']['branches'].values():
                ids = set(branch['retained_ids'])
                for node in branch['trace']:
                    if node['id'] not in ids:
                        continue
                    leaves += 1
                    for pair, (lo, hi) in zip(node['t_bounds'], prior):
                        aa, bb = map(exact, pair)
                        assert lo <= aa <= bb <= hi
                        checks += 1
                        tighter += (aa, bb) != (lo, hi)
        joint_bias.append({'unit': unit, 'retained_leaves': leaves, 't_intervals': checks, 'further_tightened': tighter})
    prior_output = ROOT.parent / 'lovelace-observation-bridge-20261011T232502Z/outputs/bridge-v1'
    prior_actions = {(a['task_id'], a['method']): a for a in load(prior_output / 'decisions.json')}
    prior_outers = {o['observation_id']: o for o in load(prior_output / 'outers.json')}
    reference_match = []
    for method in ('calibrated', 'joint'):
        values = [r for r in rows if r['unit_id'] == 'raw' and r['method'] == method]
        same_outputs = sum(exact(prior_actions['R0epsilon-' + r['base_task_id'], 'shared']['output']) == Q(r['output']) for r in values)
        same_hulls = sum(record['arms'][method]['outer']['hull'] == prior_outers['R0epsilon-' + record['observation_id'].split('-', 1)[1]]['arms']['shared']['outer']['hull']
                         for record in outers if record['unit_id'] == 'raw')
        reference_match.append({'method': method, 'historical_reference': 'R0epsilon/shared: unbiased raw image, noise .01',
                                'same_outputs': same_outputs, 'action_records': len(values), 'same_hulls': same_hulls, 'observations': 54})
    return {'target': 'liekkas:' + str(ROOT), 'main': main, 'pairs_offsets': pairs,
            'grouped_pairs': grouped_pairs, 'paired_exceptions': exceptions, 'unbiased_raw_reference_match': reference_match,
            'set_relations': sets, 'same_public_calibration': calibration, 'joint_retained_t': joint_bias,
            'seed_mean_sd': summary['seed_mean_sd'], 'costs': summary['costs'],
            'main_seconds': status['elapsed_seconds'], 'verification_seconds': verify['verification_seconds'],
            'main_plus_verification_seconds': verify['shared_budget']['elapsed_main_plus_verification_seconds'],
            'unresolved_budget_leaves': sum(r['arms'][m]['outer']['unresolved_budget_leaves'] for r in outers for m in METHODS),
            'scope': 'same54 synthetic source observations in two transformed images; no independent real-scene transfer'}


if __name__ == '__main__':
    result = analysis()
    with (ROOT / 'analysis/RESULTS.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('main', 'pairs_offsets', 'set_relations', 'joint_retained_t', 'costs')}, ensure_ascii=False))
