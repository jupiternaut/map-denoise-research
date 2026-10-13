"""Adversarial checks for independent511-budget evidence verification."""
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
_spec = importlib.util.spec_from_file_location('_test_independent_ablation_verifier', ROOT / 'verify_ablation.py')
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)
encode = verify.encoded


def full_tree(mode='channel'):
    """A fixture independent of either experimental search implementation."""
    queue = deque([(0, None, *verify.DOMAIN)])
    trace, retained, next_id = [], [], 1
    while queue:
        ident, parent, low, high = queue.popleft()
        node = {'id': ident, 'parent': parent, 'lo': encode(low), 'hi': encode(high)}
        if high - low <= verify.TOLERANCE:
            node['status'] = 'retained_tolerance'
            retained.append(ident)
        else:
            middle = (low + high) / 2
            node.update(status='split', children=[next_id, next_id + 1])
            queue.append((next_id, ident, low, middle))
            queue.append((next_id + 1, ident, middle, high))
            next_id += 2
        trace.append(node)
    groups = {'channel': 0, 'global_gray': 1, 'pixel_gray': 144, 'view_gray': 3, 'global_rgb': 3}[mode]
    return {'domain': [encode(value) for value in verify.DOMAIN], 'epsilon': encode(verify.EPSILON),
            'tolerance': encode(verify.TOLERANCE), 'max_boxes': 511, 'evaluated_boxes': 511,
            'trace': trace, 'retained_ids': retained, 'hull': [encode(value) for value in verify.DOMAIN],
            'shared_tests': groups * 511, 'propagation_rounds': 8, 'mode': mode, 'rounds': 8, 'budget': 511,
            'scope': verify.CHANNEL_SCOPE if mode == 'channel' else verify.GROUP_SCOPE}


def contradiction_fixture():
    geometry = ((Q(1), Q(1), Q(0), Q(0), Q(0), Q(0)),) * 144
    image = [Q(1)] * 432
    image[1] = image[2] = Q(0)
    certificate = {'steps': [{'row': 0, 'variable': 0, 'side': 'upper', 'value': encode(Q(1, 100))}],
                   'contradiction': {'row': 1}}
    return tuple(image), geometry, certificate


class IndependentGroupedRowsTests(unittest.TestCase):
    def test_exact_group_layout_and_rgb_relaxation(self):
        image, geometry, certificate = contradiction_fixture()
        global_gray = verify.necessary_groups(image, geometry, 'global_gray')
        pixel_gray = verify.necessary_groups(image, geometry, 'pixel_gray')
        view_gray = verify.necessary_groups(image, geometry, 'view_gray')
        rgb = verify.necessary_groups(image, geometry, 'global_rgb')
        self.assertEqual(list(global_gray), ['global'])
        self.assertEqual(list(pixel_gray), [f'p{i}' for i in range(144)])
        self.assertEqual(list(view_gray), ['v0', 'v1', 'v2'])
        self.assertEqual(list(rgb), ['c0', 'c1', 'c2'])
        self.assertEqual([len(rows) for rows in pixel_gray.values()], [2] * 144)
        self.assertEqual([len(rows) for rows in view_gray.values()], [96] * 3)
        self.assertEqual([len(rows) for rows in rgb.values()], [288] * 3)
        self.assertEqual(global_gray['global'][:2], pixel_gray['p0'])
        self.assertEqual(view_gray['v1'], global_gray['global'][96:192])
        self.assertEqual(rgb['c0'][1][2], Q(-1, 100))
        self.assertEqual(rgb['c1'][1][2], Q(99, 100))
        self.assertEqual(global_gray['global'][1][2], Q(99, 100))
        checked = verify.verify_group_certificate(image, geometry, 'global_gray', 'global', certificate, 'gray')
        self.assertEqual(checked, {'rounds': 1, 'updates': 1, 'row': 1})
        with self.assertRaises(verify.VerificationError):
            verify.verify_group_certificate(image, geometry, 'global_rgb', 'c0', certificate, 'rgb no intersection')

    def test_false_group_row_and_update_rejected(self):
        image, geometry, certificate = contradiction_fixture()
        for group_id in ('p1', 'p144', 'global', 0):
            with self.subTest(group_id=group_id), self.assertRaises(verify.VerificationError):
                verify.verify_group_certificate(image, geometry, 'pixel_gray', group_id, certificate, 'forged group')
        changed = deepcopy(certificate)
        changed['contradiction']['row'] = 0
        with self.assertRaises(verify.VerificationError):
            verify.verify_group_certificate(image, geometry, 'pixel_gray', 'p0', changed, 'forged row')
        changed = deepcopy(certificate)
        changed['steps'][0]['value'] = encode(Q(2, 100))
        with self.assertRaises(verify.VerificationError):
            verify.verify_group_certificate(image, geometry, 'pixel_gray', 'p0', changed, 'forged update')
        changed = deepcopy(certificate)
        changed['steps'] = []
        with self.assertRaises(verify.VerificationError):
            verify.verify_group_certificate(image, geometry, 'pixel_gray', 'p0', changed, 'omitted update')

    def test_local_row_numbers_are_not_global_pixel_rows(self):
        image, geometry, certificate = contradiction_fixture()
        image = list(image)
        image[:3] = [Q(1)] * 3
        image[3 * 50 + 1:3 * 50 + 3] = [Q(0), Q(0)]
        checked = verify.verify_group_certificate(image, geometry, 'pixel_gray', 'p50', certificate, 'pixel50')
        self.assertEqual(checked['row'], 1)
        with self.assertRaises(verify.VerificationError):
            verify.verify_group_certificate(image, geometry, 'pixel_gray', 'p0', certificate, 'wrong pixel')

    def test_invalid_layout_and_noncanonical_rational_rejected(self):
        image, geometry, _ = contradiction_fixture()
        with self.assertRaises(verify.VerificationError):
            verify.necessary_groups(image[:-1], geometry, 'global_gray')
        with self.assertRaises(verify.VerificationError):
            verify.necessary_groups(image, geometry[:-1], 'view_gray')
        with self.assertRaises(verify.VerificationError):
            verify.necessary_groups(image, geometry, 'unknown')
        changed = list(geometry)
        changed[0] = ({'numerator': 2, 'denominator': 2}, *geometry[0][1:])
        with self.assertRaises(verify.VerificationError):
            verify.necessary_groups(image, changed, 'global_rgb')


class IndependentTraceTests(unittest.TestCase):
    def setUp(self):
        self.image = (Q(1, 2),) * 432
        self.geometry = ((Q(0), Q(1), Q(0), Q(1), Q(0), Q(1)),) * 144
        self.bounds = lambda _branch, _radius, _low, _high: self.geometry

    def check(self, outer, mode='channel'):
        return verify.verify_trace(outer, self.image, 'fixture', mode=mode,
                                   branch='none', geometry_bounds=self.bounds)

    def test_full511_budget_tree_and_group_costs(self):
        for mode in ('channel', *verify.MODES):
            with self.subTest(mode=mode):
                checked = self.check(full_tree(mode), mode)
                self.assertEqual(checked['evaluated'], 511)
                self.assertEqual(len(checked['retained']), 256)
                self.assertEqual(checked['hull'], verify.DOMAIN)
                self.assertEqual(checked['budget_leaves'], 0)

    def test_malicious_counters_budgets_metadata_and_provenance(self):
        original = full_tree('pixel_gray')
        for field, value in (('evaluated_boxes', 510), ('max_boxes', 255), ('budget', 255),
                             ('rounds', 7), ('propagation_rounds', 9), ('shared_tests', 511),
                             ('mode', 'global_gray'), ('retained_ids', original['retained_ids'][:-1])):
            with self.subTest(field=field):
                changed = deepcopy(original)
                changed[field] = value
                with self.assertRaises(verify.VerificationError):
                    self.check(changed, 'pixel_gray')
        changed = deepcopy(original)
        changed['evaluated_boxes'] = True
        with self.assertRaises(verify.VerificationError):
            self.check(changed, 'pixel_gray')

    def test_partition_and_unresolved_budget_tampering(self):
        original = full_tree()
        for change in ('parent', 'endpoint', 'children', 'budget_leaf', 'orphan'):
            with self.subTest(change=change):
                outer = deepcopy(original)
                if change == 'parent':
                    outer['trace'][1]['parent'] = 1
                elif change == 'endpoint':
                    outer['trace'][1]['lo'] = encode(Q(541))
                elif change == 'children':
                    outer['trace'][0]['children'] = [2, 1]
                elif change == 'budget_leaf':
                    outer['trace'][-1]['status'] = 'unresolved_budget'
                else:
                    outer['trace'].append(deepcopy(outer['trace'][-1]))
                    outer['trace'][-1]['id'] = 511
                with self.assertRaises(verify.VerificationError):
                    self.check(outer)

    def test_group_certificate_in_trace_and_exact_group_counter(self):
        image, geometry, certificate = contradiction_fixture()
        outer = full_tree('pixel_gray')
        node = {'id': 0, 'parent': None, 'lo': encode(Q(540)), 'hi': encode(Q(660)),
                'status': 'excluded_shared', 'group_id': 'p0', 'certificate': certificate}
        outer.update(trace=[node], evaluated_boxes=1, shared_tests=1, retained_ids=[], hull=None)
        checked = verify.verify_trace(outer, image, 'exclude', mode='pixel_gray', branch='none',
                                      geometry_bounds=lambda *_args: geometry)
        self.assertEqual(checked['proof_updates'], 1)
        self.assertEqual(len(checked['shared_excluded']), 1)
        changed = deepcopy(outer)
        changed['shared_tests'] = 144
        with self.assertRaises(verify.VerificationError):
            verify.verify_trace(changed, image, 'false cost', mode='pixel_gray', branch='none',
                                geometry_bounds=lambda *_args: geometry)

    def test_kernel_witness_is_recomputed(self):
        node = {'id': 0, 'parent': None, 'lo': encode(Q(540)), 'hi': encode(Q(660)),
                'status': 'excluded', 'witness_channel': 0,
                'prediction_interval': [encode(Q(0)), encode(Q(1, 4))]}
        outer = full_tree()
        outer.update(trace=[node], evaluated_boxes=1, retained_ids=[], hull=None)
        checked = self.check(outer)
        class ActualKernel:
            def enclosure(self, _low, _high):
                return ((Q(0), Q(1, 5)),) * 432
        with self.assertRaises(verify.VerificationError):
            verify.verify_kernel_boxes(outer, checked, self.image, ActualKernel(), 'forged interval')


class IndependentActionAndAccountingTests(unittest.TestCase):
    def test_action_requires_exact_frozen_endpoint_arithmetic(self):
        action = verify._arithmetic.hybrid_action((Q(600), Q(601)), Q(603))
        decision = {'task_id': 'test', 'method': 'global_gray511',
                    **{key: encode(value) if isinstance(value, Q) else value for key, value in action.items()}}
        self.assertEqual(verify.verify_decision(decision, Q(603), (Q(600), Q(601)), 'action'), Q(1201, 2))
        for field, value in (('output', encode(Q(600))), ('gain_lower', encode(Q(999))), ('branch', 'projection')):
            changed = deepcopy(decision)
            changed[field] = value
            with self.assertRaises(verify.VerificationError):
                verify.verify_decision(changed, Q(603), (Q(600), Q(601)), 'forged action')

    def test_union_provenance_keeps_scene_branches_and_depth_gaps(self):
        branches = {}
        source = []
        for branch, interval in (('none', (Q(540), Q(550))), ('partial', (Q(650), Q(660))), ('full', None)):
            retained = [] if interval is None else [{'id': 1, 'lo': encode(interval[0]), 'hi': encode(interval[1])}]
            branches[branch] = {'retained': retained, 'intervals': [] if interval is None else [interval],
                                'hull': interval, 'evaluated': 3, 'budget_leaves': 0}
            source.extend({'branch': branch, **node} for node in retained)
        outer = {'domain': [encode(value) for value in verify.DOMAIN], 'epsilon': encode(verify.EPSILON),
                 'tolerance': encode(verify.TOLERANCE), 'max_boxes_per_branch': 511, 'evaluated_boxes': 9,
                 'unresolved_budget_leaves': 0, 'source_leaves': source,
                 'retained_intervals': [[encode(Q(540)), encode(Q(550))], [encode(Q(650)), encode(Q(660))]],
                 'hull': [encode(value) for value in verify.DOMAIN], 'retained_branches': ['none', 'partial'],
                 'mode': 'global_gray', 'rounds': 8, 'budget': 511,
                 'scope': 'union_of_branch_depth_outer_leaves_then_convex_hull'}
        checked = verify.verify_union(outer, branches, 'whole scene union', mode='global_gray')
        self.assertFalse(verify._historical.contains(checked['intervals'], Q(600)))
        for field, value in (('mode', 'pixel_gray'), ('budget', 255), ('evaluated_boxes', 8),
                             ('retained_intervals', [[encode(Q(540)), encode(Q(660))]])):
            changed = deepcopy(outer)
            changed[field] = value
            with self.assertRaises(verify.VerificationError):
                verify.verify_union(changed, branches, 'forged union', mode='global_gray')
        changed = deepcopy(outer)
        changed['source_leaves'][0]['branch'] = 'full'
        with self.assertRaises(verify.VerificationError):
            verify.verify_union(changed, branches, 'mixed scene provenance', mode='global_gray')

    def test_evaluation_uses_actual_truth_not_prediction(self):
        task = {'task_id': 'test', 'observation_id': 'obs', 'incumbent': encode(Q(603))}
        private = {'truth': encode(Q(600)), 'case': 'test', 'seed': 1}
        decision = {'method': 'global_rgb511', 'output': encode(Q(601)), 'status': 'MOVE'}
        checked = {'hull': (Q(600), Q(602)), 'intervals': [(Q(600), Q(602))],
                   'retained_branches': ['none'], 'evaluated': 1, 'budget_leaves': 0}
        row = verify.make_row(task, private, decision, checked)
        self.assertEqual((row['initial_ae_mm'], row['ae_mm'], row['se_mm2'], row['actual_gain_mm2']), ('3', '1', '1', '8'))
        self.assertEqual(row['auxiliary_samples_used'], 0)
        with tempfile.TemporaryDirectory(prefix='verifier-unit-', dir=ROOT) as temporary:
            output = Path(temporary)
            (output / 'evaluation').mkdir()
            changed = {**row, 'ae_mm': '0'}
            (output / 'evaluation/ROWS.json').write_text(json.dumps([changed]), encoding='utf-8')
            with self.assertRaises(verify.VerificationError):
                verify.verify_evaluation(output, [row], [], [], {}, {'elapsed_seconds': 1})

    def test_partial_output_fails_and_verification_directory_is_exclusive(self):
        with tempfile.TemporaryDirectory(prefix='verifier-unit-', dir=ROOT) as temporary:
            output = Path(temporary)
            with patch.object(verify, 'verify', side_effect=verify.VerificationError('INCOMPLETE fixture')):
                self.assertEqual(verify.main(['--output', str(output)]), 1)
            report = verify.load(output / 'verification/VERIFY.json')
            self.assertEqual(report['status'], 'FAIL')
            before = (output / 'verification/VERIFY.json').read_bytes()
            self.assertEqual(verify.main(['--output', str(output)]), 2)
            self.assertEqual((output / 'verification/VERIFY.json').read_bytes(), before)

    def test_json_duplicate_keys_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(prefix='verifier-unit-', dir=ROOT) as temporary:
            path = Path(temporary) / 'malicious.json'
            for value in ('{"count":54,"count":55}', '{"seconds":NaN}'):
                path.write_text(value, encoding='utf-8')
                with self.assertRaises(verify.VerificationError):
                    verify.load(path)


if __name__ == '__main__':
    unittest.main()
