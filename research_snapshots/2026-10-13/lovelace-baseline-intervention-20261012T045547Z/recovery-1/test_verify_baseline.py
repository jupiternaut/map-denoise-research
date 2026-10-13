"""Non-confirmation checks for independently reconstructed evidence.

These tests never draw the registered seed61005 or run a formal matrix.
"""
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('independent_baseline_verifier_test',
                                            Path(__file__).with_name('verify_baseline.py'))
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class IndependentGeometryTests(unittest.TestCase):
    def test_baseline_70_matches_frozen_bounds(self):
        _model, historical_bounds = v._historical.shared_kernels()
        for branch in v.BRANCHES:
            for left, right in ((Q(540), Q(660)), (Q(600), Q(600)), (Q(643), Q(644))):
                self.assertEqual(v.independent_geometry(branch, v.RADIUS, left, right, Q(70)),
                                 historical_bounds(branch, v.RADIUS, left, right))

    def test_baseline_80_bounds_cover_point_areas(self):
        for branch in v.BRANCHES:
            bounds = v.independent_geometry(branch, v.RADIUS, Q(639), Q(647), Q(80))
            for depth in (Q(639), Q(643), Q(647)):
                for offsets in ((-v.RADIUS, Q(0), v.RADIUS), (v.RADIUS, Q(0), -v.RADIUS),
                                (Q(1, 17), Q(0), Q(-1, 19))):
                    areas = v.independent_point_areas(branch, depth, offsets, Q(80))
                    for coefficients, actual in zip(bounds, areas):
                        self.assertTrue(all(lo <= value <= hi for lo, hi, value in
                                            zip(coefficients[::2], coefficients[1::2], actual)))

    def test_geometry_cache_includes_baseline(self):
        left = v.independent_geometry('none', v.RADIUS, Q(640), Q(644), Q(70))
        right = v.independent_geometry('none', v.RADIUS, Q(640), Q(644), Q(80))
        self.assertNotEqual(left, right)
        self.assertEqual(left, v.independent_geometry('none', v.RADIUS, Q(640), Q(644), Q(70)))

    def test_global_negative_control_and_depth_invariance(self):
        for baseline in v.BASELINES:
            proof = v.prove_full_negative_control(baseline)
            self.assertTrue(proof['full_domain_containment'])
            offsets = (Q(-1, 5), Q(0), Q(1, 5))
            images = [v.independent_rgb('full', depth, offsets, (Q(1, 5),) * 3, baseline)
                      for depth in (Q(540), Q(600), Q(660))]
            self.assertEqual(images[0], images[1])
            self.assertEqual(images[1], images[2])
        with self.assertRaises(v.VerificationError):
            v.prove_full_negative_control(Q(80), Q(200))

    def test_condition_specific_background_is_public(self):
        masks = [v._prior.public_background_mask(
            lambda branch, radius, lo, hi: v.independent_geometry(branch, radius, lo, hi, baseline))
            for baseline in v.BASELINES]
        self.assertTrue(all(len(mask) == 144 and any(mask) for mask in masks))
        image = tuple(Q(99, 100) for _ in range(432))
        for mask in masks:
            a = v._prior.calibration_from_public(image, True, mask)
            b = v._prior.calibration_from_public(image, True, mask)
            self.assertEqual(a, b)


class EvidenceBoundaryTests(unittest.TestCase):
    def test_nonconfirmation_tape_shared_by_conditions(self):
        tape, pose, noise = v.independent_tape('smoke_only', Q(600), 99156)
        self.assertEqual(tape, 'shared-appearance-v1:smoke_only:1/5:600:99156')
        self.assertEqual(len(noise), 432)
        self.assertEqual(pose[1], 0)
        self.assertTrue(all(abs(x) <= v.RADIUS for x in pose))
        self.assertTrue(all(abs(x) <= v.EPSILON for x in noise))
        second = v.independent_tape('smoke_only', Q(600), 99156)
        self.assertEqual((tape, pose, noise), second)
        predictions = [v.independent_rgb('partial', Q(600), pose,
                                        (Q(1, 5), Q(3, 5), Q(2, 5)), b) for b in v.BASELINES]
        self.assertNotEqual(predictions[0], predictions[1])

    def test_clip_ones_do_not_supply_lower_bound(self):
        bands = v._prior.pixel_bands((Q(1),) * 432, True)
        self.assertEqual(bands, ((Q(1, 50), None),) * 144)

    def test_contradiction_and_updates_are_independently_recomputed(self):
        rows = (('upper', (Q(1), Q(0), Q(0)), Q(1, 4)),
                ('lower', (Q(1), Q(0), Q(0)), Q(3, 4)))
        answer = v._prior.independent_contraction(rows, [[v.encoded(0), v.encoded(1)]] * 3)
        self.assertTrue(answer['excluded'])
        altered = {**answer['certificate'], 'steps': []}
        with self.assertRaises(v.VerificationError):
            v._prior.verify_contraction(rows, [[v.encoded(0), v.encoded(1)]] * 3, altered)

    def test_original_absolute_deadline_is_required(self):
        now = time.time()
        budget = {'start_epoch': now - 10, 'deadline_epoch': now - 10 + v.LIMIT,
                  'limit_seconds': v.LIMIT, 'scope': 'one main run plus complete verification; do not reset'}
        with patch.object(v, 'load', return_value=budget):
            self.assertEqual(v.verify_budget(budget), budget['deadline_epoch'])
            with self.assertRaises(v.VerificationError):
                v.verify_budget({**budget, 'deadline_epoch': budget['deadline_epoch'] + 1})
            # Even a mathematically valid new1800s window is a forbidden reset.
            reset = {**budget, 'start_epoch': budget['start_epoch'] + 1,
                     'deadline_epoch': budget['deadline_epoch'] + 1}
            with self.assertRaises(TimeoutError):
                with patch.object(v.time, 'time', return_value=budget['deadline_epoch']):
                    v.verify_budget(budget)
            with self.assertRaises(v.VerificationError):
                v.verify_budget(reset)

    def test_recovery_preserves_parent_source_graph_and_budget(self):
        self.assertEqual(v.DATA.resolve(), (v.PARENT / 'data').resolve())
        self.assertEqual(v.ROOT, v.EXPECTED_ROOT)
        sources = v.scientific_files()
        original = v.load(v.ORIGINAL_OUTPUT / 'SOURCE_LOCK.json')['files']
        self.assertTrue(set(original) < set(sources))
        self.assertEqual({key: sources[key] for key in original}, original)
        budget = v.load(v.ORIGINAL_OUTPUT / 'BUDGET.json')
        with patch.object(v.time, 'time', return_value=budget['start_epoch'] + 1):
            self.assertEqual(v.verify_budget(budget), budget['deadline_epoch'])

    def test_recovery_rejects_matching_child_hashes_for_changed_parent_data(self):
        original = v.load(v.ORIGINAL_OUTPUT / 'INPUTS.json')
        altered = dict(original['files'])
        target = next(iter(altered))
        altered[target] = '0' * 64
        child = {'source_root': str(v.DATA), 'files': altered}
        with patch.object(v, 'generated_inputs', return_value=altered), patch.object(v, 'load', return_value=original):
            with self.assertRaises(v.VerificationError):
                v.verify_inputs(child)

    def test_source_audit_and_early_private_parse_rejection(self):
        source = (v.ROOT / 'run_baseline.py').read_text()
        self.assertEqual(v.audit_runner_source(source)['status'], 'PASS')
        altered = source.replace('    complete_saved_actions(output,decisions)\n    truths = private_truths()',
                                 '    truths = private_truths()\n    complete_saved_actions(output,decisions)')
        self.assertNotEqual(source, altered)
        with self.assertRaises(v.VerificationError):
            v.audit_runner_source(altered)

    def test_csv_metric_corruption_is_rejected(self):
        import io
        path = v.ROOT / 'fixture.csv'
        rows = [{'task_id': 'fixture', 'mse': '9/4', 'condition': ['example', 99156]}]
        csv_text = 'task_id,mse,condition\nfixture,9/4,"[""example"", 99156]"\n'
        with patch.object(Path, 'open', return_value=io.StringIO(csv_text)):
            v.verify_csv(path, rows)
        with patch.object(Path, 'open', return_value=io.StringIO(csv_text.replace('9/4', '0'))):
            with self.assertRaises(v.VerificationError):
                v.verify_csv(path, rows)

    def test_missing_saved_action_is_rejected(self):
        with self.assertRaises(v.VerificationError):
            v.verify_action_population([], {}, {})


def negative_metric_fixture():
    """A deliberately bad scientific outcome, without rendering any seed."""
    rows, truths, coverage = [], {}, []
    population = [(case, truth) for case, _branch, _colors in v._historical.CASES
                  for truth in v._historical.TRUTHS]
    for index, (case, truth) in enumerate(population, 1):
        pid = f'p{index:03d}'
        for baseline, unit in zip(v.BASELINES, v.UNITS):
            oid = f'{pid}b{baseline}'
            truths[oid] = {'truth': v.encoded(truth)}
            for method in v.METHODS:
                coverage.append({'unit_id': unit, 'method': method, 'covered': False,
                                 'generating_branch_covered': False,
                                 'latent_material_covered': None if method == 'keep' else False,
                                 'latent_bias_covered': False if method == 'joint' else None})
                for state, offset in enumerate(v._historical.OFFSETS):
                    initial = abs(offset)
                    bad = unit == v.UNITS[1] and method == 'joint' and state != 0
                    error = initial + Q(1, 2) if bad else initial
                    rows.append({'task_id': f'{oid}s{state}', 'base_task_id': f'{pid}s{state}',
                        'observation_id': oid, 'unit_id': unit, 'pair_id': pid, 'baseline': str(baseline),
                        'method': method, 'case': case, 'seed': 99156,
                        'state_group': 'correct' if state == 0 else 'small', 'status': 'MOVE' if bad else 'KEEP',
                        'incumbent': str(truth + offset), 'output': str(truth + error),
                        'initial_ae_mm': str(initial), 'ae_mm': str(error), 'covered': False,
                        'hull_width_mm': '120', 'evaluated_boxes': 0, 'unresolved_budget_leaves': 0,
                        'auxiliary_samples_used': 0})
    return rows, truths, coverage


class NegativeScientificOutcomeTests(unittest.TestCase):
    def test_deep40_and_all144_regressions_are_preserved(self):
        rows, truths, _coverage = negative_metric_fixture()
        pairs = v.paired_baseline_rows(rows)
        summary = v.paired_baseline_summary(rows, pairs, truths)
        self.assertEqual(len(pairs), 486)
        self.assertEqual(summary['by_method']['joint']['worse'], 144)
        self.assertEqual(summary['deep_informative']['joint']['total'], 40)
        self.assertEqual(summary['deep_informative']['joint']['worse'], 40)
        self.assertTrue(any(Q(p['mse_gain_mm2']) < 0 for p in pairs))

    def test_method_regressions_and_scientific_harm_do_not_raise(self):
        rows, _truths, coverage = negative_metric_fixture()
        pairs = v.paired_method_rows(rows)
        self.assertEqual(len(pairs), 324)
        self.assertEqual(sum(p['worse'] for p in pairs), 144)
        outcomes = v.safety_claims(rows, coverage)
        self.assertEqual(outcomes[v.UNITS[1]]['joint']['harm'], 144)
        self.assertEqual(outcomes[v.UNITS[1]]['joint']['covered'], 0)
        self.assertEqual(outcomes[v.UNITS[1]]['joint']['negative_small_kept'], 0)

    def test_negative_mse_and_all_depth_rows_are_retained(self):
        rows, truths, _coverage = negative_metric_fixture()
        aggregates = v.aggregate_rows(rows, truths)
        selected = [r for r in aggregates if r['grouping'] == 'overall' and
                    r['state_group'] == 'small' and r['method'] == 'joint']
        self.assertEqual(len(selected), 2)
        baseline = {r['unit_id']: r for r in selected}
        self.assertGreater(baseline[v.UNITS[1]]['mse_mm2'], baseline[v.UNITS[0]]['mse_mm2'])
        self.assertEqual(baseline[v.UNITS[1]]['harm'], 144)
        deep = [r for r in aggregates if r['grouping'] == 'by_depth' and r['state_group'] == 'small']
        self.assertEqual(len(deep), 18)


if __name__ == '__main__':
    unittest.main()
