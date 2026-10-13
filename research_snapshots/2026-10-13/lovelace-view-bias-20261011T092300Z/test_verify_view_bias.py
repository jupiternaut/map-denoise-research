"""Counterexamples for independent bias contracts, certificates and scoring."""
from collections import deque
from copy import deepcopy
from fractions import Fraction as Q
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_test_independent_view_bias', ROOT / 'verify_view_bias.py')
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)
encode = verify.encoded


def excluded_outer(epsilon=Q(3, 100), mode='channel'):
    return {'domain': [encode(value) for value in verify.DOMAIN], 'epsilon': encode(epsilon),
            'tolerance': encode(verify.TOLERANCE), 'max_boxes': 511, 'evaluated_boxes': 1,
            'trace': [{'id': 0, 'parent': None, 'lo': encode(Q(540)), 'hi': encode(Q(660)),
                       'status': 'excluded', 'witness_channel': 0,
                       'prediction_interval': [encode(Q(11, 20)), encode(Q(3, 5))]}],
            'retained_ids': [], 'hull': None, 'shared_tests': 0, 'propagation_rounds': 8,
            'mode': mode, 'rounds': 8, 'budget': 511,
            'scope': verify._old.CHANNEL_SCOPE if mode == 'channel' else verify._old.GROUP_SCOPE}


def diagnostic_rows():
    rows, coverage = [], []
    for beta in verify.BETAS:
        for observation in range(54):
            case = 'equal_full_negative' if observation >= 45 else 'informative'
            for method in verify.METHODS:
                strict = method == 'shared_strict'
                coverage.append({'beta': str(beta), 'method': method, 'covered': not strict, 'incompatible': strict})
                for state in range(9):
                    initial = Q(0) if state == 0 else Q(1)
                    error = Q(2) if strict else initial
                    rows.append({'task_id': f'{beta}-{observation}-{state}', 'beta': str(beta), 'method': method,
                                 'state_group': 'correct' if state == 0 else 'small', 'case': case,
                                 'initial_ae_mm': str(initial), 'ae_mm': str(error),
                                 'status': 'MOVE' if strict else 'KEEP'})
    return rows, coverage


def complete_tree(epsilon, mode):
    outer = excluded_outer(epsilon, mode)
    queue, records, retained, next_id = deque([(0, None, *verify.DOMAIN)]), [], [], 1
    while queue:
        ident, parent, low, high = queue.popleft()
        node = {'id': ident, 'parent': parent, 'lo': encode(low), 'hi': encode(high)}
        if high - low <= verify.TOLERANCE:
            node['status'] = 'retained_tolerance'
            retained.append(ident)
        else:
            middle = (low + high) / 2
            node.update(status='split', children=[next_id, next_id + 1])
            queue.extend(((next_id, ident, low, middle), (next_id + 1, ident, middle, high)))
            next_id += 2
        records.append(node)
    outer.update(trace=records, retained_ids=retained, evaluated_boxes=511,
                 shared_tests=0 if mode == 'channel' else 511, hull=[encode(value) for value in verify.DOMAIN])
    return outer


class BiasAndPublicBoundTests(unittest.TestCase):
    def test_bias_view_boundaries_and_rgb_channels(self):
        image = verify.biased_image((Q(1, 2),) * 432, Q(1, 50))
        self.assertEqual(image[:144], (Q(12, 25),) * 144)
        self.assertEqual(image[144:288], (Q(1, 2),) * 144)
        self.assertEqual(image[288:], (Q(13, 25),) * 144)
        self.assertEqual((image[143], image[144], image[287], image[288]),
                         (Q(12, 25), Q(1, 2), Q(1, 2), Q(13, 25)))

    def test_direct_clip_no_preclip_and_saturation(self):
        base = [Q(1, 2)] * 432
        base[0], base[1], base[144], base[288], base[289] = Q(101, 100), Q(0), Q(101, 100), Q(99, 100), Q(-1, 100)
        image = verify.biased_image(base, Q(1, 50))
        self.assertEqual((image[0], image[1], image[144], image[288], image[289]),
                         (Q(99, 100), Q(0), Q(1), Q(1), Q(1, 100)))
        self.assertNotEqual(image[0], min(Q(1), max(Q(0), min(Q(1), base[0]) - Q(1, 50))))

    def test_public_beta_and_image_layout_tampering(self):
        with self.assertRaises(verify.VerificationError):
            verify.biased_image((Q(1, 2),) * 432, Q(1, 1000))
        with self.assertRaises(verify.VerificationError):
            verify.biased_image((Q(1, 2),) * 431, Q(1, 50))
        with self.assertRaises(verify.VerificationError):
            verify.method_contract('channel_wide', Q(1, 49))
        for beta in verify.BETAS:
            self.assertEqual(verify.method_contract('channel_wide', beta)[1], Q(1, 100) + beta)
            self.assertEqual(verify.method_contract('global_shared_wide', beta)[1], Q(1, 100) + beta)
            self.assertEqual(verify.method_contract('shared_strict', beta)[1:], (Q(1, 100), False))

    def test_actual_truth_residual_checks_wrong_view_and_clip(self):
        class ConstantPrediction:
            def __init__(self, *_args, **_kwargs):
                pass
            def point_prediction(self, *_args, **_kwargs):
                return (Q(1),) * 432
        base = (Q(101, 100),) * 432
        private = {'rear_branch': 'none', 'truth': encode(Q(600)),
                   'actual_offsets': [encode(Q(0))] * 3, 'colors': [encode(Q(1, 5))] * 3}
        image = verify.biased_image(base, Q(1, 50))
        self.assertEqual(verify.validate_residual(base, image, private, Q(1, 50), ConstantPrediction, 'true residual'), Q(1, 100))
        changed = list(image)
        changed[0] = Q(98, 100)  # Incorrectly preclipped base.
        with self.assertRaises(verify.VerificationError):
            verify.validate_residual(base, changed, private, Q(1, 50), ConstantPrediction, 'wrong clip')
        changed = tuple(reversed(image))
        with self.assertRaises(verify.VerificationError):
            verify.validate_residual(base, changed, private, Q(1, 50), ConstantPrediction, 'wrong views')


class HistoricalBoundCoordinationTests(unittest.TestCase):
    def test_complete511_trees_and_budget_rejection(self):
        class Kernel:
            def enclosure(self, *_args):
                return ((Q(0), Q(1)),) * 432
        geometry = lambda *_args: ((Q(0), Q(1), Q(0), Q(1), Q(0), Q(1)),) * 144
        image = (Q(1, 2),) * 432
        for method in verify.METHODS[1:]:
            mode, epsilon, _valid = verify.method_contract(method, Q(1, 50))
            outer = complete_tree(epsilon, mode)
            checked, calls, _witnesses = verify.verify_branch(outer, image, Q(1, 50), method, 'none', geometry, Kernel(), 'complete')
            self.assertEqual((len(checked['retained']), checked['budget_leaves'], calls), (256, 0, 511))
            outer['trace'][-1]['status'] = 'unresolved_budget'
            with self.assertRaises(verify.VerificationError):
                verify.verify_branch(outer, image, Q(1, 50), method, 'none', geometry, Kernel(), 'incomplete')

    def test_all_historical_globals_restore_after_error(self):
        before = (verify._old.EPSILON, verify._historical.EPSILON, verify._old.verify_group_certificate)
        with self.assertRaises(RuntimeError):
            with verify.historical_contract(Q(3, 100)):
                self.assertEqual((verify._old.EPSILON, verify._historical.EPSILON), (Q(3, 100), Q(3, 100)))
                raise RuntimeError('fixture')
        self.assertEqual((verify._old.EPSILON, verify._historical.EPSILON, verify._old.verify_group_certificate), before)

    def test_wide_bound_must_not_default_to_old_epsilon(self):
        class Kernel:
            def enclosure(self, *_args):
                return ((Q(11, 20), Q(3, 5)),) * 432
        image = (Q(1, 2),) * 432
        outer = excluded_outer()
        checked, calls, witnesses = verify.verify_branch(outer, image, Q(1, 50), 'channel_wide', 'none', None, Kernel(), 'wide')
        self.assertEqual((checked['evaluated'], calls, witnesses), (1, 1, 1))
        changed = deepcopy(outer)
        changed['epsilon'] = encode(Q(1, 100))
        with self.assertRaises(verify.VerificationError):
            verify.verify_branch(changed, image, Q(1, 50), 'channel_wide', 'none', None, Kernel(), 'stolen epsilon')

    def test_kernel_witness_uses_wide_bound_too(self):
        class Kernel:
            def enclosure(self, *_args):
                return ((Q(13, 25), Q(3, 5)),) * 432
        image = (Q(1, 2),) * 432
        outer = excluded_outer()
        checked = {'boxes': {0: verify.DOMAIN}, 'evaluated': 1}
        with verify.historical_contract(Q(3, 100)), self.assertRaises(verify.VerificationError):
            verify._old.verify_kernel_boxes(outer, checked, image, Kernel(), 'nonconflicting under wide')

    def test_independent_rows_and_forged_certificate_bound(self):
        geometry = ((Q(1), Q(1), Q(0), Q(0), Q(0), Q(0)),) * 144
        image = [Q(1)] * 432
        image[1] = image[2] = Q(0)
        rows = verify.necessary_rows(image, geometry, Q(3, 100))
        self.assertEqual((rows[0][2], rows[1][2]), (Q(3, 100), Q(97, 100)))
        certificate = {'steps': [{'row': 0, 'variable': 0, 'side': 'upper', 'value': encode(Q(3, 100))}],
                       'contradiction': {'row': 1}}
        with verify.historical_contract(Q(3, 100)):
            checked = verify._old.verify_group_certificate(image, geometry, 'global_gray', 'global', certificate, 'proof')
            self.assertEqual(checked['updates'], 1)
            changed = deepcopy(certificate)
            changed['steps'][0]['value'] = encode(Q(1, 100))
            with self.assertRaises(verify.VerificationError):
                verify._old.verify_group_certificate(image, geometry, 'global_gray', 'global', changed, 'old bound forgery')
            with self.assertRaises(verify.VerificationError):
                verify._old.verify_group_certificate(image, geometry, 'global_gray', 'p0', certificate, 'forged group')

    def test_union_checks_its_explicit_wide_bound(self):
        epsilon = Q(3, 100)
        outer = {'domain': [encode(value) for value in verify.DOMAIN], 'epsilon': encode(epsilon),
                 'tolerance': encode(verify.TOLERANCE), 'max_boxes_per_branch': 511, 'evaluated_boxes': 1,
                 'unresolved_budget_leaves': 0, 'source_leaves': [], 'retained_intervals': [], 'hull': None,
                 'retained_branches': [], 'mode': 'channel', 'rounds': 8, 'budget': 511,
                 'scope': 'union_of_branch_depth_outer_leaves_then_convex_hull'}
        checked = {'none': {'retained': [], 'intervals': [], 'hull': None, 'evaluated': 1, 'budget_leaves': 0}}
        self.assertIsNone(verify.verify_union(outer, checked, Q(1, 50), 'channel_wide', 'empty')['hull'])
        outer['epsilon'] = encode(Q(1, 100))
        with self.assertRaises(verify.VerificationError):
            verify.verify_union(outer, checked, Q(1, 50), 'channel_wide', 'forged epsilon')


class ScientificDiagnosticTests(unittest.TestCase):
    def test_actual_truth_row_and_independent_pairs(self):
        task = {'task_id': 'b0-o000-s1', 'observation_id': 'b0-o000', 'incumbent': encode(Q(603))}
        private = {'truth': encode(Q(600)), 'case': 'textured_partial', 'seed': 61001,
                   'base_observation_id': 'o000', 'beta': '1/200'}
        decision = {'method': 'shared_strict', 'output': encode(Q(604)), 'status': 'MOVE'}
        checked = {'hull': (Q(603), Q(605)), 'intervals': [(Q(603), Q(605))],
                   'retained_branches': ['partial'], 'evaluated': 1, 'budget_leaves': 0}
        row = verify.make_row(task, private, decision, checked)
        self.assertEqual((row['ae_mm'], row['se_mm2'], row['actual_gain_mm2'], row['contract_valid']), ('4', '16', '-7', False))
        metadata = {'task_id': 'b0-o000-s1', 'observation_id': 'b0-o000', 'base_observation_id': 'o000',
                    'beta': '1/200', 'case': 'textured_partial', 'seed': 61001, 'state_group': 'small'}
        rows = [{**metadata, 'method': method, 'ae_mm': str(error)}
                for method, error in zip(verify.METHODS, (Q(1), Q(3, 4), Q(1, 2), Q(2)))]
        pairs = verify.paired_rows(rows)
        self.assertEqual(len(pairs), 4)
        self.assertEqual((pairs[-1]['baseline'], pairs[-1]['mae_gain_mm'], pairs[-1]['mse_gain_mm2']),
                         ('channel_wide', '1/4', '5/16'))
        self.assertTrue(pairs[2]['worse'])
        self.assertFalse(pairs[2]['contract_valid'])
        with self.assertRaises(verify.VerificationError):
            verify.paired_rows(rows[:-1])

    def test_beta_task_population_tampering(self):
        public = {'observation_id': 'o000', 'radius': '1/5', 'observation': [encode(Q(1, 2))] * 432}
        base_tasks = {'o000-s0': {'task_id': 'o000-s0', 'observation_id': 'o000', 'incumbent': encode(Q(600))}}
        private = {'observation_id': 'o000', 'truth': encode(Q(600)), 'rear_branch': 'none',
                   'actual_offsets': [encode(Q(0))] * 3, 'colors': [encode(Q(1, 2))] * 3}
        class Kernel:
            def __init__(self, *_args, **_kwargs):
                pass
            def point_prediction(self, *_args, **_kwargs):
                return (Q(1, 2),) * 432
        fixture = ({'o000': public}, base_tasks, {'o000': private})
        observations, tasks, truths = [], [], []
        for index, beta in enumerate(verify.BETAS):
            oid = f'b{index}-o000'
            observations.append({'observation_id': oid, 'base_observation_id': 'o000', 'beta': str(beta), 'radius': '1/5',
                                 'observation': list(map(encode, verify.biased_image(public['observation'], beta)))})
            tasks.append({'task_id': f'{oid}-s0', 'observation_id': oid, 'base_task_id': 'o000-s0', 'incumbent': encode(Q(600))})
            truths.append({**private, 'observation_id': oid, 'base_observation_id': 'o000', 'beta': str(beta),
                           'actual_view_bias': list(map(encode, (-beta, Q(0), beta)))})
        def fake_load(path):
            return {'observations.json': observations, 'tasks.json': tasks, 'truth.json': truths}[Path(path).name]
        with patch.object(verify._old, 'verify_frozen_population', return_value=fixture), patch.object(verify, 'load', side_effect=fake_load):
            self.assertEqual(len(verify.verify_population(Path('/unused'), Kernel)[0]), 3)
            tasks.pop()
            with self.assertRaises(verify.VerificationError):
                verify.verify_population(Path('/unused'), Kernel)
            tasks.append({'task_id': 'b2-o000-s0', 'observation_id': 'b2-o000', 'base_task_id': 'o000-s0', 'incumbent': encode(Q(600))})
            observations[0]['beta'] = '1/50'
            with self.assertRaises(verify.VerificationError):
                verify.verify_population(Path('/unused'), Kernel)

    def test_partial_output_fail_and_exclusive_verification(self):
        with tempfile.TemporaryDirectory(prefix='view-verifier-unit-', dir=ROOT) as temporary:
            output = Path(temporary)
            with patch.object(verify, 'verify', side_effect=verify.VerificationError('INCOMPLETE fixture')):
                self.assertEqual(verify.main(['--output', str(output)]), 1)
            report_path = output / 'verification/VERIFY.json'
            report = json.loads(report_path.read_text())
            self.assertEqual((report['status'], report['mechanical_verification']), ('FAIL', 'FAIL'))
            before = report_path.read_bytes()
            self.assertEqual(verify.main(['--output', str(output)]), 2)
            self.assertEqual(report_path.read_bytes(), before)

    def test_strict_negative_results_are_not_mechanical_failure(self):
        rows, coverage = diagnostic_rows()
        claims = verify.scientific_claims(rows, coverage)
        self.assertEqual(claims['C1']['status'], 'NEGATIVE')
        self.assertTrue(claims['scientific_negative_result_is_not_mechanical_verification_failure'])
        for beta in verify.BETAS:
            strict = claims['by_beta'][str(beta)]['safety']['shared_strict']
            self.assertEqual((strict['coverage_failures'], strict['correct_moved'], strict['harm']), (54, 54, 486))
            self.assertFalse(strict['contract_valid'])

    def test_removed_task_or_legal_failure_rejected(self):
        rows, coverage = diagnostic_rows()
        with self.assertRaises(verify.VerificationError):
            verify.scientific_claims(rows[1:], coverage)
        changed = deepcopy(coverage)
        changed[0]['covered'] = False
        with self.assertRaises(verify.VerificationError):
            verify.scientific_claims(rows, changed)
        changed = deepcopy(rows)
        changed[0]['ae_mm'] = '1'
        changed[0]['status'] = 'MOVE'
        with self.assertRaises(verify.VerificationError):
            verify.scientific_claims(changed, coverage)


if __name__ == '__main__':
    unittest.main()
