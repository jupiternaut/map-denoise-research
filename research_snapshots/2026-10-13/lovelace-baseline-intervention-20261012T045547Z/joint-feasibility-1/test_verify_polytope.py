"""Independent hand fixtures; never run the formal matrix or parse private."""
from copy import deepcopy
import ast
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location('independent_polytope_verifier_test', Path(__file__).with_name('verify_polytope.py'))
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)


def certificate(rows, bounds, weights):
    normalized, box = v.normalized_problem(rows, bounds)
    aggregate = [Q(0)] * len(box)
    bound = Q(0)
    for row, weight in weights:
        coefficients, b = normalized[row]
        aggregate = [a + weight * c for a, c in zip(aggregate, coefficients)]
        bound += weight * b
    minimum = v.minimum_over_box(aggregate, box)
    return {'sparse_weights': [{'row': row, 'weight': v.encoded(weight)} for row, weight in weights],
            'aggregate_coefficients': list(map(v.encoded, aggregate)), 'weighted_bound': v.encoded(bound),
            'box_min': v.encoded(minimum), 'margin': v.encoded(minimum - bound)}


def classification(rows, bounds, status='UNKNOWN', cert=None, witness=None):
    singles = v.single_row_contradictions(rows, bounds)
    return {'status': status, 'certificate': cert, 'witness': witness, 'lp_status': None,
            'seconds': 0.0, 'lp_seconds': 0.0, 'single_row_contradiction': bool(singles),
            'single_row_contradiction_rows': singles, 'row_count': len(rows), 'dimension': len(bounds), 'reason': 'hand_fixture'}


class ExactCertificateTests(unittest.TestCase):
    def setUp(self):
        # Neither individual row contradicts this box; only their sum does.
        self.rows = (('upper', (Q(1), Q(-1)), Q(-1, 2)),
                     ('lower', (Q(1), Q(-1)), Q(1, 2)))
        self.box = ((Q(0), Q(1)), (Q(0), Q(1)))
        self.cert = certificate(self.rows, self.box, [(0, Q(1)), (1, Q(1))])

    def test_signed_lower_normalization_and_joint_contradiction(self):
        self.assertEqual(v.normalized_problem(self.rows, self.box)[0], (((Q(1), Q(-1)), Q(-1, 2)), ((Q(-1), Q(1)), Q(-1, 2))))
        self.assertEqual(v.single_row_contradictions(self.rows, self.box), [])
        checked = v.verify_certificate(self.rows, self.box, self.cert)
        self.assertEqual(checked['aggregate_coefficients'], (Q(0), Q(0)))
        self.assertEqual(checked['weighted_bound'], Q(-1))
        self.assertEqual(checked['box_min'], Q(0))
        self.assertEqual(checked['margin'], Q(1))

    def test_signed_box_min_uses_upper_for_negative_aggregate(self):
        rows, box = (('upper', (Q(-2), Q(3)), Q(-20)),), ((Q(-1), Q(4)), (Q(2), Q(7)))
        cert = certificate(rows, box, [(0, Q(1, 3))])
        self.assertEqual(v.verify_certificate(rows, box, cert)['box_min'], Q(-2, 3))
        self.assertEqual(v.single_row_contradictions(rows, box), [0])

    def test_zero_margin_is_not_a_certificate(self):
        rows = (('upper', (Q(1), Q(-1)), Q(0)), ('lower', (Q(1), Q(-1)), Q(0)))
        cert = certificate(rows, self.box, [(0, Q(1)), (1, Q(1))])
        with self.assertRaises(v.VerificationError):
            v.verify_certificate(rows, self.box, cert)

    def test_strict_sub_float_precision_margin(self):
        tiny = Q(1, 10**100)
        rows = (('upper', (Q(1),), -tiny),)
        cert = certificate(rows, ((Q(0), Q(1)),), [(0, Q(1))])
        self.assertEqual(v.verify_certificate(rows, ((Q(0), Q(1)),), cert)['margin'], tiny)

    def test_every_certificate_field_and_weight_is_independently_checked(self):
        changes = []
        for key in self.cert:
            c = deepcopy(self.cert)
            del c[key]
            changes.append(c)
        for key in ('weighted_bound', 'box_min', 'margin'):
            c = deepcopy(self.cert)
            c[key] = v.encoded(100)
            changes.append(c)
        c = deepcopy(self.cert); c['aggregate_coefficients'][0] = v.encoded(1); changes.append(c)
        c = deepcopy(self.cert); c['sparse_weights'][0]['weight'] = v.encoded(-1); changes.append(c)
        c = deepcopy(self.cert); c['sparse_weights'].append(deepcopy(c['sparse_weights'][0])); changes.append(c)
        c = deepcopy(self.cert); c['sparse_weights'][0]['row'] = 2; changes.append(c)
        c = deepcopy(self.cert); c['sparse_weights'][0]['row'] = True; changes.append(c)
        c = deepcopy(self.cert); c['margin'] = 1.0; changes.append(c)
        c = deepcopy(self.cert); c['margin'] = {'numerator': 2, 'denominator': 2}; changes.append(c)
        for c in changes:
            with self.subTest(c=c), self.assertRaises(v.VerificationError):
                v.verify_certificate(self.rows, self.box, c)

    def test_missing_or_altered_certificate_cannot_authorize_deletion(self):
        result = classification(self.rows, self.box, 'CERTIFIED_INFEASIBLE', self.cert)
        self.assertFalse(v.verify_classification(self.rows, self.box, result))
        for cert in (None, {}, {**self.cert, 'margin': v.encoded(0)}):
            with self.assertRaises(v.VerificationError):
                v.verify_classification(self.rows, self.box, {**result, 'certificate': cert})


class WitnessAndUnknownTests(unittest.TestCase):
    def setUp(self):
        self.rows = (('upper', (Q(1), Q(-1)), Q(1, 2)), ('lower', (Q(1), Q(1)), Q(1, 2)))
        self.box = ((Q(0), Q(1)), (Q(0), Q(1)))
        self.point = [v.encoded(Q(1, 2)), v.encoded(Q(1, 2))]

    def test_exact_boundary_witness_checks_all_rows(self):
        self.assertEqual(v.verify_witness(self.rows, self.box, self.point), (Q(1, 2), Q(1, 2)))
        result = classification(self.rows, self.box, 'FEASIBLE_WITNESS', witness=self.point)
        self.assertTrue(v.verify_classification(self.rows, self.box, result))
        for point in ([v.encoded(-1), v.encoded(1)], [v.encoded(1), v.encoded(0)], [v.encoded(0), v.encoded(0)], self.point[:1], [0.5, 0.5]):
            with self.subTest(point=point), self.assertRaises(v.VerificationError):
                v.verify_witness(self.rows, self.box, point)

    def test_unknown_always_retains_even_if_single_row_contradiction_exists(self):
        rows = (('upper', (Q(1), Q(0)), Q(-1)),)
        self.assertTrue(v.verify_classification(rows, self.box, classification(rows, self.box)))
        result = classification(rows, self.box)
        result['single_row_contradiction_rows'] = []
        with self.assertRaises(v.VerificationError):
            v.verify_classification(rows, self.box, result)

    def test_no_implicit_floating_or_boolean_problem_values(self):
        for box in (((0.0, 1), (0, 1)), ((False, 1), (0, 1)), ((1, 0), (0, 1))):
            with self.assertRaises(v.VerificationError):
                v.normalized_problem(self.rows, box)


class FrozenLeafTests(unittest.TestCase):
    def fixture_filter(self):
        rows = (('upper', (Q(1), Q(-1)), Q(-1, 2)), ('lower', (Q(1), Q(-1)), Q(1, 2)))
        box = ((Q(0), Q(1)), (Q(0), Q(1)))
        trace = [{'id': index, 'parent': None, 'lo': v.encoded(600 + index), 'hi': v.encoded(601 + index),
                  'status': 'retained_tolerance', 'darkness_bounds': [[v.encoded(0), v.encoded(1)]] * 9,
                  't_bounds': [[v.encoded(0), v.encoded(Q(1, 50))]] * 3} for index in range(2)]
        branches = {name: {'trace': deepcopy(trace) if name == 'none' else [],
                           'retained_ids': [0, 1] if name == 'none' else [],
                           'hull': [v.encoded(600), v.encoded(602)] if name == 'none' else None} for name in v.BRANCHES}
        old = {'branches': branches, 'outer': {'source_leaves': [], 'retained_intervals': [], 'hull': None,
                                              'retained_branches': [], 'unresolved_budget_leaves': 0},
               'calibration_intervals': trace[0]['t_bounds'], 'frozen_field': 'same'}
        old['outer'] = v.independent_union(old['outer'], branches)
        arm = deepcopy(old)
        arm['branches']['none']['retained_ids'] = [1]
        arm['branches']['none']['hull'] = [v.encoded(601), v.encoded(602)]
        arm['outer'] = v.independent_union(old['outer'], arm['branches'])
        results = [classification(rows, box, 'CERTIFIED_INFEASIBLE', certificate(rows, box, [(0, Q(1)), (1, Q(1))])),
                   classification(rows, box)]
        arm['polytope_records'] = {name: [{'node_id': node['id'], 'lo': node['lo'], 'hi': node['hi'],
                                           'old_status': node['status'], 'classification': results[node['id']]}
                                          for node in value['trace']] for name, value in branches.items()}
        arm['polytope_stats'] = {'original_leaves': 2, 'status_counts': {'CERTIFIED_INFEASIBLE': 1, 'UNKNOWN': 1},
                                 'single_row_contradictions': 0, 'budget_leaves_rebuilt': 0,
                                 'lp_seconds': 0.0, 'filter_seconds': 0.0}
        return rows, box, old, arm

    def test_exact_certificate_removes_one_leaf_unknown_stays_unchanged(self):
        rows, box, old, arm = self.fixture_filter()
        with patch.object(v, 'leaf_problem', return_value=(rows, box)):
            checked = v.verify_filtered_arm(arm, old, {}, 'joint')
        self.assertEqual([node['id'] for node in checked['branches']['none']['retained']], [1])
        self.assertEqual(arm['branches']['none']['trace'], old['branches']['none']['trace'])

    def test_deleted_budget_leaf_keeps_original_complete_trace_metadata(self):
        rows, box, old, arm = self.fixture_filter()
        old['branches']['none']['trace'][0]['status'] = 'unresolved_budget'
        old['outer']['unresolved_budget_leaves'] = 1
        arm['branches']['none']['trace'][0]['status'] = 'unresolved_budget'
        arm['polytope_records']['none'][0]['old_status'] = 'unresolved_budget'
        arm['polytope_stats']['budget_leaves_rebuilt'] = 1
        arm['outer'] = v.independent_union(old['outer'], arm['branches'])
        with patch.object(v, 'leaf_problem', return_value=(rows, box)):
            checked = v.verify_filtered_arm(arm, old, {}, 'joint')
        self.assertEqual(arm['branches']['none']['retained_ids'], [1])
        self.assertEqual(arm['outer']['unresolved_budget_leaves'], 1)
        self.assertEqual(checked['counts']['unresolved_budget_leaf_checks'], 1)
        self.assertEqual(arm['branches']['none']['trace'], old['branches']['none']['trace'])
        changed = deepcopy(arm)
        changed['outer']['unresolved_budget_leaves'] = 0
        with patch.object(v, 'leaf_problem', return_value=(rows, box)), self.assertRaises(v.VerificationError):
            v.verify_filtered_arm(changed, old, {}, 'joint')

    def test_missing_unknown_leaf_missing_record_or_mutated_trace_is_rejected(self):
        rows, box, old, arm = self.fixture_filter()
        changes = []
        c = deepcopy(arm); c['branches']['none']['retained_ids'] = []; c['branches']['none']['hull'] = None; changes.append(c)
        c = deepcopy(arm); c['polytope_records']['none'].pop(); changes.append(c)
        c = deepcopy(arm); c['polytope_records']['none'][0]['classification']['certificate'] = None; changes.append(c)
        c = deepcopy(arm); c['branches']['none']['trace'][1]['darkness_bounds'][0][0] = v.encoded(Q(1, 10)); changes.append(c)
        c = deepcopy(arm); c['branches']['none']['trace'][1]['lo'] = v.encoded(602); changes.append(c)
        c = deepcopy(arm); c['outer']['hull'][0] = v.encoded(600); changes.append(c)
        for c in changes:
            with patch.object(v, 'leaf_problem', return_value=(rows, box)), self.assertRaises(v.VerificationError):
                v.verify_filtered_arm(c, old, {}, 'joint')

    def test_leaf_row_reconstruction_uses_exact_original_depth_and_box(self):
        node = {'id': 7, 'status': 'unresolved_budget', 'lo': v.encoded(600), 'hi': v.encoded(Q(1201, 2)),
                'darkness_bounds': [[v.encoded(0), v.encoded(1)]] * 9,
                't_bounds': [[v.encoded(0), v.encoded(Q(1, 50))]] * 3}
        public = {'radius': '1/5', 'baseline': '80', 'clipped': True, 'observation': ['1'] * 432}
        arm = {'calibration_intervals': node['t_bounds']}
        geometry = ((Q(0), Q(1), Q(0), Q(1), Q(0), Q(1)),) * 144
        for method, dim in (('calibrated', 9), ('joint', 12)):
            with patch.object(v._old, 'independent_geometry', return_value=geometry) as reconstructed:
                rows, box = v.leaf_problem(public, arm, method, 'partial', node)
            reconstructed.assert_called_once_with('partial', Q(1, 5), Q(600), Q(1201, 2), Q(80))
            self.assertEqual(len(box), dim)
            self.assertEqual(len(rows), 144 + 18)
            self.assertEqual(rows[-1][1][0], Q(0))

    def test_verifier_does_not_import_new_filter_or_runner(self):
        source = Path(v.__file__).read_text()
        self.assertNotIn('import polytope_filter', source)
        self.assertNotIn('import run_polytope', source)
        self.assertNotIn('data/private', source)


class PublicGateAndBudgetTests(unittest.TestCase):
    def test_actual_runner_and_executor_have_one_shared_post_readback_gate(self):
        self.assertEqual(v.audit_runner_source()['status'], 'PASS')
        self.assertEqual(v.audit_executor_source()['status'], 'PASS')

    def test_missing_readback_or_unique_matrix_gate_is_rejected(self):
        source = (v.ROOT / 'run_polytope.py').read_text()
        for old, new in (("check_actions(ds,ts);check_keep(ks,ts)", "check_keep(ks,ts)"),
                         ("ds=load(output/'decisions.json')", "ds=ds"),
                         ('len(set(keys))!=1296', 'len(set(keys))!=1295'),
                         ("if sum(c['original_leaves'] for c in timing)!=5103", "if sum(c['original_leaves'] for c in timing)!=5102")):
            self.assertIn(old, source)
            with self.assertRaises(v.VerificationError):
                v.audit_runner_source(source.replace(old, new))

    def test_budget_and_executor_cannot_reset_or_add_private_route(self):
        budget = {'start_epoch': 1000, 'deadline_epoch': 4600, 'limit_seconds': 3600, 'scope': v.BUDGET_SCOPE}
        with patch.object(v.time, 'time', return_value=1100):
            self.assertEqual(v.verify_budget(budget), 4600)
            for key, value in (('deadline_epoch', 4601), ('limit_seconds', 3601), ('scope', 'new_window')):
                with self.assertRaises(v.VerificationError):
                    v.verify_budget({**budget, key: value})
        with patch.object(v.time, 'time', return_value=4600), self.assertRaises(TimeoutError):
            v.verify_budget(budget)
        source = (v.ROOT / 'execute_polytope.py').read_text()
        with self.assertRaises(v.VerificationError):
            v.audit_executor_source(source.replace('remaining=deadline-time.time()', 'remaining=3600'))
        with self.assertRaises(v.VerificationError):
            v.audit_executor_source(source + "\nprivate_file = 'truth.json'\n")

    def test_independent_private_call_is_after_all_public_actions(self):
        tree = ast.parse(Path(v.__file__).read_text())
        verify = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'verify')
        private_calls = [node for node in ast.walk(verify) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                         and node.func.attr == 'verify_private_population']
        self.assertEqual(len(private_calls), 1)
        for name in ('verify_action_population', 'verify_filtered_arm', 'verify_action', 'verify_keep_population'):
            calls = [node for node in ast.walk(verify) if isinstance(node, ast.Call) and
                     ((isinstance(node.func, ast.Name) and node.func.id == name) or
                      (isinstance(node.func, ast.Attribute) and node.func.attr == name))]
            self.assertTrue(calls)
            self.assertTrue(all(node.lineno < private_calls[0].lineno for node in calls))

    def test_same_leaf_simultaneous_witness_required_even_if_hull_covers(self):
        node = {'id': 3, 'lo': v.encoded(600), 'hi': v.encoded(601), 'status': 'retained_tolerance',
                'darkness_bounds': [[v.encoded(0), v.encoded(1)]] * 9,
                't_bounds': [[v.encoded(0), v.encoded(Q(1, 50))]] * 3}
        before = {'branches': {'partial': {'retained': [node]}}}
        after = {'branches': {'partial': {'retained': []}}, 'union': {'hull': (Q(590), Q(610))}}
        fact = {'truth': '600', 'rear_branch': 'partial', 'view_colors': ['1/2'] * 9,
                'actual_view_bias': ['-1/100', '0', '1/100']}
        public = {'observation_id': 'manual_nonformal'}
        rows = (('upper', (Q(0),) * 12, Q(1)),)
        box = node['darkness_bounds'] + node['t_bounds']
        with patch.object(v, 'leaf_problem', return_value=(rows, box)), self.assertRaises(v.VerificationError):
            v.verify_same_leaf_witness(public, fact, 'joint', {}, before, after)
        after['branches']['partial']['retained'] = [node]
        with patch.object(v, 'leaf_problem', return_value=(rows, box)):
            result = v.verify_same_leaf_witness(public, fact, 'joint', {}, before, after)
        self.assertEqual(result['surviving_same_leaf_ids'], [3])


if __name__ == '__main__':
    unittest.main()
