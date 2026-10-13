"""Small author tests and tampering tests; no confirmation is run."""
from collections import deque
import copy
from fractions import Fraction as Q
import unittest

from verification.arithmetic import (DOMAIN, EPSILON, TOLERANCE, MAX_BOXES, VerificationError,
                                     encoded, exact, merged_intervals, verify_trace, verify_union,
                                     hybrid_action, verify_decision)
from verification.evidence import (arm_contracts, reconstruct_aggregates, regenerate, make_row,
                                   registration, same, ROOT)
from verification.verify_results import check_enclosures, verify_strategy_boundary, primary_claims
from nuisance_model import ColoredModel


def branch_outer(empty=False, budget=False):
    if empty:
        trace = [{'id': 0, 'parent': None, 'lo': encoded(540), 'hi': encoded(660),
                  'status': 'excluded', 'witness_channel': 0, 'prediction_interval': [encoded(0), encoded(0)]}]
        retained, evaluated = [], 1
    else:
        trace, retained, evaluated, next_id = [], [], 0, 1
        queue = deque([(0, None, *DOMAIN)])
        while queue:
            ident, parent, low, high = queue.popleft()
            node = {'id': ident, 'parent': parent, 'lo': encoded(low), 'hi': encoded(high)}
            if budget and evaluated == MAX_BOXES:
                node['status'] = 'unresolved_budget'
                retained.append(ident)
            else:
                evaluated += 1
                if not budget and ident and ident % 2 == 1:
                    node.update(status='excluded', witness_channel=0, prediction_interval=[encoded(0), encoded(0)])
                elif high - low <= TOLERANCE:
                    node['status'] = 'retained_tolerance'
                    retained.append(ident)
                else:
                    mid = (low + high) / 2
                    node.update(status='split', children=[next_id, next_id + 1])
                    queue.extend([(next_id, ident, low, mid), (next_id + 1, ident, mid, high)])
                    next_id += 2
            trace.append(node)
    leaves = [node for node in trace if node['id'] in retained]
    hull = None if not leaves else [leaves[0]['lo'], leaves[-1]['hi']]
    return {'domain': list(map(encoded, DOMAIN)), 'epsilon': encoded(EPSILON), 'tolerance': encoded(TOLERANCE),
            'max_boxes': MAX_BOXES, 'evaluated_boxes': evaluated, 'trace': trace,
            'retained_ids': retained, 'hull': hull, 'scope': 'outer_set_for_continuous_depth_and_model_nuisance_contract'}


def combined(checked, prior=False):
    source = [{'branch': name, 'id': node['id'], 'lo': node['lo'], 'hi': node['hi']}
              for name, value in checked.items() for node in value['retained']]
    intervals = [DOMAIN] if prior else merged_intervals(interval for value in checked.values() for interval in value['intervals'])
    return {'domain': list(map(encoded, DOMAIN)), 'epsilon': encoded(EPSILON), 'tolerance': encoded(TOLERANCE),
            'max_boxes_per_branch': MAX_BOXES, 'evaluated_boxes': sum(c['evaluated'] for c in checked.values()),
            'unresolved_budget_leaves': sum(c['budget_leaves'] for c in checked.values()), 'source_leaves': source,
            'retained_intervals': [list(map(encoded, interval)) for interval in intervals],
            'hull': None if not intervals else list(map(encoded, (intervals[0][0], intervals[-1][1]))),
            'retained_branches': [name for name, c in checked.items() if c['hull'] is not None],
            'scope': 'prior_only_identity_no_image_solver' if prior else 'union_of_branch_depth_outer_leaves_then_convex_hull'}


class ArithmeticTests(unittest.TestCase):
    def test_registration_denominators(self):
        protocol = registration()
        self.assertEqual((protocol['expected_observations'], protocol['expected_tasks'], protocol['expected_decisions']), (54, 486, 2430))
        self.assertEqual(protocol['max_boxes_per_branch'], 255)

    def test_exact_rejects_float_bool_noncanonical(self):
        for value in (0.2, True, {'numerator': 2, 'denominator': 10}, {'numerator': True, 'denominator': 5}):
            with self.assertRaises(VerificationError):
                exact(value)

    def test_type_sensitive_comparison(self):
        with self.assertRaises(VerificationError):
            same({'count': True}, {'count': 1}, 'tamper')

    def test_midpoint_branch(self):
        result = hybrid_action((Q(599), Q(601)), Q(606))
        self.assertEqual((result['branch'], result['output'], result['gain_lower']), ('midpoint', Q(600), Q(24)))

    def test_projection_fallback_branch(self):
        result = hybrid_action((Q(599), Q(601)), Q(602))
        self.assertEqual((result['branch'], result['output'], result['gain_lower']), ('projection', Q(601), Q(1)))

    def test_inside_and_empty(self):
        self.assertEqual(hybrid_action((Q(599), Q(601)), Q(600))['branch'], 'keep')
        self.assertEqual(hybrid_action(None, Q(600))['branch'], 'incompatible')

    def test_keep_schema_and_branch_tamper(self):
        keep = {'task_id': 't', 'method': 'keep', 'status': 'KEEP', 'output': encoded(600), 'gain_lower': encoded(0), 'reason': 'identity'}
        self.assertEqual(verify_decision(keep, Q(600), DOMAIN, 'keep'), Q(600))
        decision = {'task_id': 't', 'method': 'calibrated_union', **hybrid_action((Q(599), Q(601)), Q(606))}
        decision['branch'] = 'projection'
        with self.assertRaises(VerificationError):
            verify_decision(decision, Q(606), (Q(599), Q(601)), 'branch tamper')

    def test_empty_wrong_model_is_not_execution_failure(self):
        checked = verify_trace(branch_outer(empty=True), (Q(1),), 'nominal')
        union = verify_union(combined({'none': checked}), {'none': checked}, 'nominal')
        self.assertIsNone(union['hull'])
        self.assertEqual(hybrid_action(union['hull'], Q(600))['status'], 'INCOMPATIBLE')

    def test_budget_retains_full_domain(self):
        outer = branch_outer(budget=True)
        checked = verify_trace(outer, (Q(1),), 'budget')
        self.assertEqual((checked['evaluated'], checked['budget_leaves'], checked['hull']), (255, 256, DOMAIN))
        class BroadKernel:
            def enclosure(self, low, high):
                return ((Q(0), Q(1)),)
        self.assertEqual(check_enclosures(outer, checked, (Q(1),), BroadKernel(), 'budget'), (255, 0))

    def test_trace_partition_tamper(self):
        outer = branch_outer()
        verify_trace(outer, (Q(1),), 'valid')
        outer['trace'][1]['hi'] = encoded(599)
        with self.assertRaises(VerificationError):
            verify_trace(outer, (Q(1),), 'gap tamper')

    def test_witness_nonconflict_tamper(self):
        outer = branch_outer(empty=True)
        outer['trace'][0]['prediction_interval'] = [encoded(1), encoded(1)]
        with self.assertRaises(VerificationError):
            verify_trace(outer, (Q(1),), 'witness tamper')

    def test_first_witness_kernel_tamper(self):
        outer = branch_outer(empty=True)
        checked = verify_trace(outer, (Q(1),), 'valid arithmetic')
        class DifferentKernel:
            def enclosure(self, low, high):
                return ((Q(1), Q(1)),)
        with self.assertRaises(VerificationError):
            check_enclosures(outer, checked, (Q(1),), DifferentKernel(), 'kernel witness tamper')

    def test_union_complete_scene_provenance_tamper(self):
        checked = {'none': verify_trace(branch_outer(empty=True), (Q(1),), 'empty'),
                   'full': verify_trace(branch_outer(budget=True), (Q(1),), 'full')}
        outer = combined(checked)
        self.assertEqual(verify_union(outer, checked, 'union')['intervals'], [DOMAIN])
        outer['source_leaves'][0]['branch'] = 'none'
        with self.assertRaises(VerificationError):
            verify_union(outer, checked, 'pixel/branch provenance tamper')

    def test_union_hull_tamper(self):
        checked = {'full': verify_trace(branch_outer(budget=True), (Q(1),), 'full')}
        outer = combined(checked)
        outer['hull'][0] = encoded(541)
        with self.assertRaises(VerificationError):
            verify_union(outer, checked, 'hull tamper')

    def test_samples_preserve_source_and_true_color(self):
        public, private = regenerate('unit_only', 'partial', (Q(1, 5), Q(3, 5), Q(2, 5)), Q(600), 99007, ColoredModel)
        anchored = arm_contracts(public['calibration'])['calibrated_union'][1]
        self.assertTrue(all(low <= exact(color) <= high for (low, high), color in zip(anchored, private['colors'])))
        self.assertEqual(len(public['observation']), 432)
        self.assertEqual(len(private['calibration_noise']), 3)
        tampered = copy.deepcopy(public['calibration'])
        tampered[0]['source'] = 'rear'
        with self.assertRaises(VerificationError):
            arm_contracts(tampered)

    def test_same_seed_regeneration_is_exact(self):
        args = ('unit_only', 'none', (Q(1, 5), Q(1, 5), Q(2, 5)), Q(600), 99001, ColoredModel)
        self.assertEqual(regenerate(*args), regenerate(*args))

    def test_rows_and_coverage_conditioned_metrics(self):
        task = {'task_id': 't', 'observation_id': 'o', 'incumbent': encoded(602)}
        truth = {'case': 'textured_isolated', 'truth': encoded(600)}
        action = {'method': 'calibrated_union', 'output': encoded(601), 'status': 'MOVE'}
        checked = {'hull': (Q(599), Q(601)), 'intervals': [(Q(599), Q(601))],
                   'retained_branches': ['none'], 'evaluated': 17, 'budget_leaves': 0}
        row = make_row(task, truth, action, checked)
        self.assertEqual((row['ae_mm'], row['actual_gain_mm2'], row['covered']), ('1', '3', True))
        aggregates = reconstruct_aggregates([row])
        self.assertEqual(len(aggregates), 6)
        self.assertTrue(all(r['improved'] == 1 and r['harm'] == 0 and r['auxiliary_samples_used'] == 3 for r in aggregates))

    def test_source_strategy_signatures(self):
        self.assertIn('not a complete', verify_strategy_boundary())

    def test_valid_contract_safety_and_negative_c2_acceptance(self):
        truths, unions, rows = {}, {}, []
        for i in range(54):
            oid = f'o{i:03d}'
            case = 'equal_full_negative' if i >= 45 else 'textured_isolated'
            truths[oid] = {'case': case, 'truth': encoded(600)}
            unions[oid] = {m: {'intervals': [DOMAIN]} for m in ('expanded_unanchored', 'calibrated_union')}
            for method in ('expanded_unanchored', 'calibrated_union'):
                for j in range(9):
                    rows.append({'method': method, 'task_id': f'{oid}-s{j}', 'case': case,
                                 'state_group': 'correct' if j == 0 else 'small',
                                 'status': 'KEEP', 'ae_mm': '0' if j == 0 else '1',
                                 'initial_ae_mm': '0' if j == 0 else '1'})
        rows.append({'method': 'calibrated_isolated', 'status': 'MOVE', 'ae_mm': '100', 'initial_ae_mm': '0'})
        claims = primary_claims(rows, unions, truths)
        self.assertEqual(claims['C2']['status'], 'NEGATIVE')
        self.assertEqual(claims['C1']['valid_contract_arms']['calibrated_union']['correct_initial_kept'], 54)
        unions['o000']['calibrated_union']['intervals'] = []
        with self.assertRaises(VerificationError):
            primary_claims(rows, unions, truths)

    def test_action_gain_tamper(self):
        decision = {'task_id': 't', 'method': 'calibrated_union', **hybrid_action((Q(599), Q(601)), Q(606))}
        decision['gain_lower'] = encoded(25)
        with self.assertRaises(VerificationError):
            verify_decision(decision, Q(606), (Q(599), Q(601)), 'gain tamper')


if __name__ == '__main__':
    unittest.main()
