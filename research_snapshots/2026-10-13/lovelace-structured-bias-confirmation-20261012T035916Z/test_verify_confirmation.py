"""Nonconfirmation tests: no seed61004 data generation or private-label reads."""
import copy
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('independent_confirmation_tested', ROOT / 'verify_confirmation.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def public_fixture():
    observations = {f'c{i:03d}': {} for i in range(1, 19)}
    tasks = {f'{oid}s{state}': {} for oid in observations for state in range(9)}
    decisions = [{'task_id': tid, 'method': method, 'reused': False}
                 for oid in observations for method in v.METHODS for tid in (f'{oid}s{state}' for state in range(9))]
    return observations, tasks, decisions


def source_fixture():
    return '''
def execute_new(observation, radius, method, clipped):
    return frozen.execute_new(observation, radius, method, clipped)
def population():
    return load(DATA / 'observations.json'), load(DATA / 'tasks.json')
def complete_saved_actions(output, decisions):
    if len(decisions) != 486 or len({(r['task_id'], r['method']) for r in decisions}) != 486 or load(output / 'decisions.json') != decisions:
        raise RuntimeError('incomplete')
    if Counter((r['method'], r['reused']) for r in decisions) != {('keep', False): 162, ('calibrated', False): 162, ('joint', False): 162}:
        raise RuntimeError('origins')
def private_truths():
    return load(DATA / 'private/truth.json')
def _run(output, started, budget):
    entry = execute_new(item['observation'], item['radius'], method, item['clipped'])
    complete_saved_actions(output, decisions)
    truths = private_truths()
'''


def full_branch_fixture():
    image = [v.encoded(1)] * 432
    geometry = lambda *_args: ((v.Q(0),) * 6,) * 144
    mask = v._prior.public_background_mask(geometry)
    calibration = v._prior.calibration_from_public(image, True, mask)
    initial = [[v.encoded(0), v.encoded(1)]] * 3 + calibration['intervals']
    rows = v._prior.necessary_rows(image, geometry(), True, 'joint', calibration['intervals'])
    contraction = v._prior.independent_contraction(rows, initial)
    trace, queue = [], [(None, v.Q(540), v.Q(660))]
    next_child = 1
    while queue:
        parent, lo, hi = queue.pop(0)
        ident = len(trace)
        node = {'id': ident, 'parent': parent, 'lo': v.encoded(lo), 'hi': v.encoded(hi),
                'rounds': contraction['rounds'], 'updates': contraction['updates'],
                'darkness_bounds': contraction['bounds'][:3], 't_bounds': contraction['bounds'][3:]}
        if hi-lo > v.TOLERANCE:
            node.update(status='split', children=[next_child, next_child+1])
            next_child += 2
            middle = (lo+hi)/2
            queue.extend(((ident, lo, middle), (ident, middle, hi)))
        else:
            node['status'] = 'retained_tolerance'
        trace.append(node)
    outer = {'domain': list(map(v.encoded, v.DOMAIN)), 'epsilon': v.encoded(v.EPSILON), 'beta': v.encoded(v.BETA),
             'clipped': True, 'tolerance': v.encoded(v.TOLERANCE), 'max_boxes': 511, 'evaluated_boxes': 511,
             'trace': trace, 'retained_ids': list(range(255, 511)), 'hull': list(map(v.encoded, v.DOMAIN)),
             'shared_tests': 511, 'propagation_rounds': 8, 'mode': 'joint', 'rounds': 8, 'budget': 511,
             'channel_precheck': 'none', 'initial_bounds': initial, 'scope': v._prior.GROUP_SCOPE}
    return image, geometry, calibration, outer


class ConfirmationVerifierTests(unittest.TestCase):
    def test_negative_scientific_outcomes_are_preserved_and_depth_dicts_grouped(self):
        rows, coverage, truths = [], [], {}
        checked = {'hull': None, 'intervals': [], 'retained_branches': [], 'evaluated': 0, 'budget_leaves': 0}
        oid_index = 0
        for case, branch, colors in v._historical.CASES:
            for depth in v._historical.TRUTHS:
                oid_index += 1
                oid = f'c{oid_index:03d}'
                private = {'observation_id': oid, 'unit_id': 'clip', 'case': case, 'seed': 99000,
                           'truth': v.encoded(depth), 'rear_branch': branch, 'colors': list(map(v.encoded, colors)),
                           'actual_view_bias': list(map(v.encoded, (-v.BETA, v.Q(0), v.BETA)))}
                truths[oid] = private
                for method in v.METHODS:
                    for state, offset in enumerate(v._historical.OFFSETS):
                        tid, incumbent = f'{oid}s{state}', depth+offset
                        task = {'task_id': tid, 'base_task_id': tid, 'observation_id': oid, 'incumbent': v.encoded(incumbent)}
                        action = {'task_id': tid, 'method': method, 'unit_id': 'clip', 'reused': False,
                                  'status': 'KEEP' if method != 'joint' else 'MOVE',
                                  'output': v.encoded(incumbent if method != 'joint' else incumbent+10)}
                        rows.append(v._prior.make_row(task, private, action, checked))
                    branches = {} if method == 'keep' else {name: {'intervals': [], 'retained': []} for name in v.BRANCHES}
                    coverage.append(v._prior.make_coverage({'observation_id': oid, 'unit_id': 'clip'}, private,
                                                           method, checked, branches))
        claims = v.safety_claims(rows, coverage)
        self.assertEqual(claims['joint']['harm'], 162)
        self.assertEqual(claims['joint']['covered'], 0)
        self.assertEqual(claims['joint']['latent_bias_covered'], 0)
        pairs = v.paired_rows(rows)
        self.assertEqual(sum(row['worse'] for row in pairs), 162)
        aggregates = v.aggregate_rows(rows, truths)
        self.assertEqual(len(aggregates), 171)
        self.assertEqual(sum(row['grouping'] == 'by_depth' for row in aggregates), 27)

    def test_full_action_count_order_and_no_reuse(self):
        observations, tasks, decisions = public_fixture()
        self.assertEqual(len(v.verify_action_population(decisions, observations, tasks)), 486)
        mutations = [decisions[:-1], list(reversed(decisions))]
        duplicate = copy.deepcopy(decisions)
        duplicate[-1] = duplicate[-2]
        mutations.append(duplicate)
        reused = copy.deepcopy(decisions)
        reused[-1]['reused'] = True
        mutations.append(reused)
        boolean_number = copy.deepcopy(decisions)
        boolean_number[-1]['reused'] = 0
        mutations.append(boolean_number)
        for mutation in mutations:
            with self.subTest(kind=mutation[-1] if mutation else None), self.assertRaises(v.VerificationError):
                v.verify_action_population(mutation, observations, tasks)

    def test_budget_absolute_window_cannot_reset_or_expand(self):
        original = {'start_epoch': 100., 'deadline_epoch': 1900., 'limit_seconds': 1800,
                    'scope': 'one main run plus complete verification; do not reset'}
        with patch.object(v.time, 'time', return_value=1200.):
            self.assertEqual(v.verify_budget(original), 1900.)
            for field, value in [('deadline_epoch', 1901.), ('limit_seconds', 1801), ('scope', 'reset')]:
                with self.subTest(field=field), self.assertRaises(v.VerificationError):
                    v.verify_budget({**original, field: value})
        with patch.object(v.time, 'time', return_value=1900.):
            with self.assertRaises(TimeoutError):
                v.verify_budget(original)

    def test_source_gate_order_and_public_interface_tampering(self):
        self.assertEqual(v.audit_runner_source(source_fixture())['status'], 'PASS')
        mutations = [source_fixture().replace('complete_saved_actions(output, decisions)\n    truths = private_truths()',
                                              'truths = private_truths()\n    complete_saved_actions(output, decisions)'),
                     source_fixture().replace('!= 486', '!= 485', 1),
                     source_fixture().replace("item['clipped']", "truths[0]['clipped']"),
                     source_fixture().replace("DATA / 'private/truth.json'", "OTHER / 'private/truth.json'"),
                     source_fixture().replace('return frozen.execute_new(observation, radius, method, clipped)',
                                              'return frozen.execute_new(observation, radius, method, private_truths())')]
        for mutation in mutations:
            with self.assertRaises(v.VerificationError):
                v.audit_runner_source(mutation)

    def test_exact_contraction_certificate_tampering(self):
        rows = (('upper', (v.Q(1), v.Q(1), v.Q(0)), v.Q(1, 2)),)
        bounds = [[v.encoded(0), v.encoded(1)]] * 3
        expected = v._prior.independent_contraction(rows, bounds)
        v._prior.verify_contraction(rows, bounds, expected['certificate'])
        forged = copy.deepcopy(expected['certificate'])
        forged['steps'][0]['value'] = v.encoded(v.Q(3, 4))
        with self.assertRaises(v.VerificationError):
            v._prior.verify_contraction(rows, bounds, forged)

    def test_full_tree_and_retained_latent_bounds_tampering(self):
        image, geometry, calibration, outer = full_branch_fixture()
        checked = v._prior.verify_branch(outer, image, True, 'joint', 'none', geometry, calibration, 'synthetic')
        self.assertEqual((checked['evaluated'], len(checked['retained'])), (511, 256))
        mutations = []
        altered = copy.deepcopy(outer)
        altered['trace'][-1]['t_bounds'][0][1] = v.encoded(0)
        mutations.append(altered)
        omitted = copy.deepcopy(outer)
        omitted['trace'].pop()
        mutations.append(omitted)
        false_hull = copy.deepcopy(outer)
        false_hull['hull'][1] = v.encoded(659)
        mutations.append(false_hull)
        wrong_budget = copy.deepcopy(outer)
        wrong_budget['budget'] = 510
        mutations.append(wrong_budget)
        for mutation in mutations:
            with self.assertRaises(v.VerificationError):
                v._prior.verify_branch(mutation, image, True, 'joint', 'none', geometry, calibration, 'tampered')

    def test_action_arithmetic_tampering(self):
        task = {'task_id': 'synthetic', 'incumbent': v.encoded(600)}
        hull = (v.Q(601), v.Q(602))
        expected = v._arithmetic.hybrid_action(hull, task['incumbent'])
        decision = {'task_id': 'synthetic', 'method': 'joint',
                    **{key: v.encoded(value) if key in ('output', 'gain_lower') and value is not None else value
                       for key, value in expected.items()}}
        v._arithmetic.verify_decision(decision, task['incumbent'], hull, 'synthetic')
        decision['gain_lower'] = v.encoded(100)
        with self.assertRaises(v.VerificationError):
            v._arithmetic.verify_decision(decision, task['incumbent'], hull, 'tampered')

    def test_nested_sets_merge_adjacent_leaf_partitions(self):
        self.assertTrue(v.interval_subset([(v.Q(0), v.Q(2))], [(v.Q(0), v.Q(1)), (v.Q(1), v.Q(2))]))
        with self.assertRaises(v.VerificationError):
            v.nested_sets({'intervals': [(v.Q(0), v.Q(1))]}, {'intervals': [(v.Q(0), v.Q(2))]}, 'tampered')

    def test_csv_every_cell_tampering(self):
        rows = [{'task_id': 'synthetic', 'ae_mm': '1/2'}]
        for csv, passed in [('task_id,ae_mm\nsynthetic,1/2\n', True), ('task_id,ae_mm\nsynthetic,1/3\n', False)]:
            with patch.object(Path, 'open', return_value=io.StringIO(csv)):
                if passed:
                    v._prior.verify_csv(Path('not-created.csv'), rows)
                else:
                    with self.assertRaises(v.VerificationError):
                        v._prior.verify_csv(Path('not-created.csv'), rows)


if __name__ == '__main__':
    unittest.main()
