"""A6 hand fixtures only: no formal batch or private facts are read."""
from copy import deepcopy
import ast
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location('independent_point_verifier_test', Path(__file__).with_name('verify_point.py'))
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)


def hand_arm():
    box = [[v.encoded(0), v.encoded(1)]] * 9
    ts = [[v.encoded(0), v.encoded(Q(1, 50))]] * 3
    trace = [{'id': 9, 'status': 'retained_tolerance', 'lo': v.encoded(599), 'hi': v.encoded(600), 'darkness_bounds': deepcopy(box), 't_bounds': deepcopy(ts)},
             {'id': 4, 'status': 'retained_tolerance', 'lo': v.encoded(600), 'hi': v.encoded(601), 'darkness_bounds': deepcopy(box), 't_bounds': deepcopy(ts)}]
    branches = {name: {'trace': deepcopy(trace), 'retained_ids': [9, 4]} for name in v.BRANCHES}
    return {'branches': branches, 'calibration_intervals': ts, 'appearance_model': 'bounded', 'drift_bound': v.encoded(Q(1, 10))}


def public_image():
    return {'radius': '1/5', 'baseline': '80', 'clipped': True, 'observation': ['1'] * 432}


def classification(rows, bounds, status='UNKNOWN', witness=None, certificate=None):
    singles = v._a5.single_row_contradictions(rows, bounds)
    return {'status': status, 'certificate': certificate, 'witness': witness, 'lp_status': None,
            'seconds': 0.0, 'lp_seconds': 0.0, 'single_row_contradiction': bool(singles),
            'single_row_contradiction_rows': singles, 'row_count': len(rows), 'dimension': len(bounds), 'reason': 'hand_fixture'}


def hand_record(method='joint', point_status='FEASIBLE_WITNESS'):
    public = {**public_image(), 'observation_id': 'manual_nonformal', 'unit_id': 'appearance_eta_0', 'variation_index': 0}
    arm = hand_arm()
    arm['outer'] = {'hull': [v.encoded(599), v.encoded(601)]}
    task = {'task_id': 'manual_nonformal_s0', 'incumbent': v.encoded(600)}
    local, original = [], {branch: [] for branch in v.BRANCHES}
    dimension = 12 if method == 'joint' else 9
    witness = [v.encoded(0)] * dimension
    for branch, node in v.containing_leaves(arm, '600'):
        rows, box = v._a5.leaf_problem(public, arm, method, branch, node)
        frozen = classification(rows, box, 'FEASIBLE_WITNESS', witness=witness)
        original[branch].append({'node_id': node['id'], 'classification': frozen})
        point_rows, point_box = v.point_problem(public, arm, method, branch, '600', node)
        local.append({'branch': branch, 'node_id': node['id'], 'lo': node['lo'], 'hi': node['hi'],
                      'box_classification': deepcopy(frozen), 'point_classification': classification(point_rows, point_box, point_status,
                          witness=witness if point_status == 'FEASIBLE_WITNESS' else None)})
    global_records = []
    for branch in v.BRANCHES:
        rows, box = v.point_problem(public, arm, method, branch, '600')
        global_records.append({'branch': branch, 'point_classification': classification(rows, box, point_status,
                               witness=witness if point_status == 'FEASIBLE_WITNESS' else None)})
    action = {'status': 'KEEP', 'output': v.encoded(600)}
    record = {'task_id': task['task_id'], 'observation_id': public['observation_id'], 'unit_id': public['unit_id'],
              'variation_index': public['variation_index'], 'method': method, 'incumbent': task['incumbent'],
              'source_hull_contains': True, 'source_union_contains': True, 'local': local, 'global': global_records,
              'box_status': 'FEASIBLE_WITNESS', 'local_status': point_status, 'global_status': point_status,
              'a4_action': deepcopy(action), 'a5_action': deepcopy(action), 'diagnostic_seconds': 0.0, 'lp_seconds': 0.0}
    return record, public, task, arm, {'polytope_records': original}, action, action


class PointProblemTests(unittest.TestCase):
    def test_closed_endpoint_contains_both_original_leaves_in_original_order(self):
        arm = hand_arm()
        contained = v.containing_leaves(arm, '600')
        self.assertEqual([(name, node['id']) for name, node in contained], [(name, ident) for name in v.BRANCHES for ident in (9, 4)])
        self.assertEqual(v.containing_leaves(arm, '602'), [])

    def test_global_and_local_priors_at_exact_point_depth(self):
        public, arm = public_image(), hand_arm()
        for method, dimension in (('calibrated', 9), ('joint', 12)):
            global_rows, global_box = v.point_problem(public, arm, method, 'partial', '600')
            node = arm['branches']['partial']['trace'][0]
            local_rows, local_box = v.point_problem(public, arm, method, 'partial', '600', node)
            self.assertEqual(global_rows, local_rows)
            self.assertEqual(global_box, local_box)
            self.assertEqual(len(global_rows), 162)
            self.assertEqual(len(global_box), dimension)
            v.verify_witness(global_rows, global_box, [v.encoded(0)] * dimension)

    def test_local_uses_frozen_final_box_and_budget_path(self):
        public, arm = public_image(), hand_arm()
        node = deepcopy(arm['branches']['none']['trace'][0])
        node['status'] = 'unresolved_budget'
        node['darkness_bounds'][0] = [v.encoded(Q(1, 10)), v.encoded(Q(9, 10))]
        _, local_box = v.point_problem(public, arm, 'joint', 'none', Q(600), node)
        self.assertEqual(local_box[0], node['darkness_bounds'][0])
        self.assertEqual(local_box[-3:], node['t_bounds'])
        self.assertEqual(node['darkness_bounds'][0], [v.encoded(Q(1, 10)), v.encoded(Q(9, 10))])

    def test_exact_point_area_bounds_nested_in_original_interval(self):
        public, arm = public_image(), hand_arm()
        for branch in v.BRANCHES:
            for depth in (Q(599), Q(1199, 2), Q(600)):
                v.point_geometry(public, branch, depth, arm['branches'][branch]['trace'][0])
        node = arm['branches']['none']['trace'][0]
        fixed = ((Q(1, 2), Q(1, 2), Q(0), Q(1), Q(0), Q(1)),) * 144
        wrong_interval = ((Q(3, 4), Q(1), Q(0), Q(1), Q(0), Q(1)),) * 144
        with patch.object(v._old, 'independent_geometry', side_effect=[fixed, wrong_interval]), self.assertRaises(v.VerificationError):
            v.point_geometry(public, 'none', '600', node)

    def test_wrong_depth_model_and_float_are_rejected(self):
        public, arm = public_image(), hand_arm()
        node = arm['branches']['none']['trace'][0]
        for depth in (Q(601), 600.0, True):
            with self.assertRaises(v.VerificationError):
                v.point_problem(public, arm, 'joint', 'none', depth, node)
        for model in ({**arm, 'drift_bound': '1/5'}, {**arm, 'appearance_model': 'independent'}):
            with self.assertRaises(v.VerificationError):
                v.point_problem(public, model, 'joint', 'none', '600')

    def test_clipped_zero_retains_only_lower_pixel_rows_and_signed_links(self):
        public, arm = public_image(), hand_arm()
        public['observation'] = ['0'] * 432
        rows, bounds = v.point_problem(public, arm, 'joint', 'none', '600')
        self.assertEqual(len(rows), 162)
        self.assertTrue(all(side == 'lower' for side, _coefficients, _bound in rows[:144]))
        self.assertEqual(rows[-18:], v._old.link_rows('joint', Q(1, 10)))
        with self.assertRaises(v.VerificationError):
            v.verify_witness(rows, bounds, [v.encoded(0)] * 12)


class AggregationAndLiftingTests(unittest.TestCase):
    def test_source_empty_or_witness_all_cert_and_unknown(self):
        C, F, U = ({'status': status} for status in ('CERTIFIED_INFEASIBLE', 'FEASIBLE_WITNESS', 'UNKNOWN'))
        for values, expected in (([], 'SOURCE_EXCLUDED'), ([C], 'CERTIFIED_INFEASIBLE'), ([C, C], 'CERTIFIED_INFEASIBLE'),
                                 ([C, U], 'UNKNOWN'), ([U], 'UNKNOWN'), ([C, F, U], 'FEASIBLE_WITNESS')):
            self.assertEqual(v.aggregate_status(values), expected)
        self.assertEqual(v.aggregate_status([], empty='UNKNOWN'), 'UNKNOWN')

    def test_global_witness_lifts_to_every_same_branch_closed_leaf(self):
        public, arm = public_image(), hand_arm()
        lifted = v.verify_global_lifting(public, arm, 'joint', '600', 'none', [v.encoded(0)] * 12)
        self.assertEqual(lifted, [9, 4])

    def test_global_witness_vs_local_box_conflict_is_an_invariant_failure(self):
        public, arm = public_image(), hand_arm()
        arm['branches']['none']['trace'][0]['darkness_bounds'][0] = [v.encoded(Q(1, 2)), v.encoded(1)]
        with self.assertRaises(v.VerificationError):
            v.verify_global_lifting(public, arm, 'joint', '600', 'none', [v.encoded(0)] * 12)

    def test_global_witness_without_containing_source_branch_fails(self):
        public, arm = public_image(), hand_arm()
        arm['branches']['none']['retained_ids'] = []
        with self.assertRaises(v.VerificationError):
            v.verify_global_lifting(public, arm, 'joint', '600', 'none', [v.encoded(0)] * 12)

    def test_no_new_solver_or_runner_business_imports(self):
        source = Path(v.__file__).read_text()
        self.assertNotIn('import point_solver', source)
        self.assertNotIn('import run_point', source)


class CompleteRecordTests(unittest.TestCase):
    def test_complete_manual_record_all_original_closed_leaves_all_three_branches(self):
        for method in v.METHODS:
            args = hand_record(method)
            counts = v.verify_record(*args)
            self.assertEqual(counts['local_checks'], 6)
            self.assertEqual(counts['global_checks'], 3)
            self.assertEqual(counts['local_FEASIBLE_WITNESS'], 6)

    def test_missing_duplicate_reordered_mismatched_or_mutated_evidence_fails(self):
        args = hand_record()
        record = args[0]
        changes = []
        c = deepcopy(record); c['local'].pop(); changes.append(c)
        c = deepcopy(record); c['local'][1] = deepcopy(c['local'][0]); changes.append(c)
        c = deepcopy(record); c['global'].reverse(); changes.append(c)
        c = deepcopy(record); c['global'].pop(); changes.append(c)
        c = deepcopy(record); c['local'][0]['box_classification']['seconds'] = 1.0; changes.append(c)
        c = deepcopy(record); c['local'][0]['hi'] = v.encoded(601); changes.append(c)
        c = deepcopy(record); c['global'][0]['point_classification']['witness'][0] = v.encoded(2); changes.append(c)
        c = deepcopy(record); c['source_union_contains'] = False; changes.append(c)
        c = deepcopy(record); c['local_status'] = 'UNKNOWN'; changes.append(c)
        c = deepcopy(record); c['a5_action']['output'] = v.encoded(601); changes.append(c)
        for c in changes:
            with self.subTest(record=c), self.assertRaises(v.VerificationError):
                v.verify_record(c, *args[1:])

    def test_source_hull_gap_is_source_excluded_with_complete_global_unknown(self):
        record, public, task, arm, a5_arm, a4_action, a5_action = hand_record(point_status='UNKNOWN')
        task['incumbent'] = record['incumbent'] = v.encoded(Q(1203, 2))
        for branch in v.BRANCHES:
            arm['branches'][branch]['trace'][1]['lo'] = v.encoded(602)
            arm['branches'][branch]['trace'][1]['hi'] = v.encoded(603)
        arm['outer']['hull'] = [v.encoded(599), v.encoded(603)]
        record['local'] = []
        record['source_union_contains'] = False
        record['box_status'] = record['local_status'] = 'SOURCE_EXCLUDED'
        for value in record['global']:
            rows, bounds = v.point_problem(public, arm, 'joint', value['branch'], task['incumbent'])
            value['point_classification'] = classification(rows, bounds)
        counts = v.verify_record(record, public, task, arm, a5_arm, a4_action, a5_action)
        self.assertEqual(counts['local_checks'], 0)
        self.assertEqual(counts['global_checks'], 3)
        self.assertTrue(record['source_hull_contains'])
        self.assertFalse(record['source_union_contains'])

    def test_valid_point_certificate_can_disagree_with_frozen_feasible_box(self):
        args = list(hand_record(point_status='UNKNOWN'))
        record = args[0]
        old_rows = (('upper', (Q(1), Q(1)), Q(1)),)
        point_rows = (('upper', (Q(1), Q(1)), Q(-1)),)
        box = ((Q(0), Q(1)), (Q(0), Q(1)))
        frozen = classification(old_rows, box, 'FEASIBLE_WITNESS', witness=[v.encoded(0)] * 2)
        cert = {'sparse_weights': [{'row': 0, 'weight': v.encoded(1)}], 'aggregate_coefficients': [v.encoded(1)] * 2,
                'weighted_bound': v.encoded(-1), 'box_min': v.encoded(0), 'margin': v.encoded(1)}
        point = classification(point_rows, box, 'CERTIFIED_INFEASIBLE', certificate=cert)
        for value in record['local']:
            value['box_classification'] = deepcopy(frozen)
            value['point_classification'] = deepcopy(point)
        for branch in v.BRANCHES:
            for value in args[4]['polytope_records'][branch]:
                value['classification'] = deepcopy(frozen)
        for value in record['global']:
            value['point_classification'] = deepcopy(point)
        record['local_status'] = record['global_status'] = 'CERTIFIED_INFEASIBLE'
        with patch.object(v._a5, 'leaf_problem', return_value=(old_rows, box)), patch.object(v, 'point_problem', return_value=(point_rows, box)):
            counts = v.verify_record(*args)
        self.assertEqual(counts['local_CERTIFIED_INFEASIBLE'], 6)
        changed = deepcopy(record)
        changed['local'][0]['point_classification']['certificate']['margin'] = v.encoded(0)
        with patch.object(v._a5, 'leaf_problem', return_value=(old_rows, box)), patch.object(v, 'point_problem', return_value=(point_rows, box)), self.assertRaises(v.VerificationError):
            v.verify_record(changed, *args[1:])


class GateBudgetAndEvaluationTests(unittest.TestCase):
    def test_actual_runner_executor_and_fixed_registration_pass(self):
        self.assertEqual(v.audit_runner_source()['status'], 'PASS')
        self.assertEqual(v.audit_executor_source()['status'], 'PASS')
        v.verify_registration(v.load(v.ROOT / 'REGISTRATION.json'))

    def test_missing_saved_readback_population_gate_or_extra_private_route_fails(self):
        source = (v.ROOT / 'run_point.py').read_text()
        for old, new in (("records=load(output/'diagnoses.json')", 'records=records'),
                         ('len(set(keys))!=648', 'len(set(keys))!=647'),
                         ('globals_count!=1944', 'globals_count!=1943')):
            with self.assertRaises(v.VerificationError):
                v.audit_runner_source(source.replace(old, new))
        with self.assertRaises(v.VerificationError):
            v.audit_runner_source(source + "\nprivate_path = 'truth.json'\n")

    def test_common_budget_is_absolute_and_never_reset(self):
        budget = {'start_epoch': 1000, 'deadline_epoch': 4600, 'limit_seconds': 3600, 'scope': v.BUDGET_SCOPE}
        with patch.object(v.time, 'time', return_value=1100):
            self.assertEqual(v.verify_budget(budget), 4600)
            with self.assertRaises(v.VerificationError):
                v.verify_budget({**budget, 'deadline_epoch': 4601})
        with patch.object(v.time, 'time', return_value=4600), self.assertRaises(TimeoutError):
            v.verify_budget(budget)
        source = (v.ROOT / 'execute_point.py').read_text()
        with self.assertRaises(v.VerificationError):
            v.audit_executor_source(source.replace('remaining=deadline-time.time()', 'remaining=3600'))

    def test_private_fact_parse_only_after_all_public_evidence_and_checkpoints(self):
        tree = ast.parse(Path(v.__file__).read_text())
        verify = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'verify')
        private = [node for node in ast.walk(verify) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == 'verify_private_population']
        self.assertEqual(len(private), 1)
        for name in ('verify_record_population', 'verify_record'):
            calls = [node for node in ast.walk(verify) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]
            self.assertTrue(calls)
            self.assertTrue(all(node.lineno < private[0].lineno for node in calls))
        checkpoint = [node for node in ast.walk(verify) if isinstance(node, ast.Constant) and node.value == 'checkpoint']
        self.assertTrue(checkpoint)
        self.assertTrue(all(node.lineno < private[0].lineno for node in checkpoint))

    def test_summary_preserves_fixed_primary_and_full_denominators(self):
        rows = []
        cases = ('uniform_isolated', 'textured_isolated', 'textured_partial', 'equal_partial', 'weak_partial', 'equal_full_negative')
        depths = ('6203/11', '7861/13', '9007/14')
        for unit in ('appearance_eta_0', 'appearance_eta_1_10'):
            for method in v.METHODS:
                for case in cases:
                    for depth in depths:
                        for state in range(9):
                            rows.append({'unit_id': unit, 'method': method, 'case': case, 'depth_mm': depth,
                                         'initial_ae_mm': str(state), 'box_status': 'UNKNOWN', 'local_status': 'UNKNOWN',
                                         'global_status': 'UNKNOWN', 'a5_status': 'KEEP'})
        summary = v.summary_rows(rows)
        v.verify_summary_denominators(summary)
        changed = deepcopy(summary)
        cell = next(value for value in changed if value['group'] == 'depth_informative' and value['value'] == '9007/14')
        cell['n'] = 39
        with self.assertRaises(v.VerificationError):
            v.verify_summary_denominators(changed)


if __name__ == '__main__':
    unittest.main()
