"""Independent verifier regressions; corruption fixtures use no runner/solver."""
import copy
from fractions import Fraction as Q
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import verify_structured as v


def image(value=1):
    return [Q(value)] * 432


def geometry(value=0):
    return tuple((Q(0), Q(value), Q(0), Q(0), Q(0), Q(0)) for _ in range(144))


def unmasked_calibration():
    return v.calibration_from_public(image(), False, [False] * 144)


def excluded_branch(method='joint', clipped=False):
    observation, calibration = image(0), unmasked_calibration()
    initial = [[v.encoded(0), v.encoded(1)] for _ in range(3)]
    if method == 'joint':
        initial.extend(calibration['intervals'])
    rows = v.necessary_rows(observation, geometry(), clipped, method, calibration['intervals'])
    contraction = v.independent_contraction(rows, initial)
    node = {'id': 0, 'parent': None, 'lo': v.encoded(540), 'hi': v.encoded(660),
            'status': 'excluded_shared', 'rounds': contraction['rounds'], 'updates': contraction['updates'],
            'group_id': 'global', 'certificate': contraction['certificate']}
    outer = {'domain': [v.encoded(540), v.encoded(660)], 'epsilon': v.encoded(v.EPSILON),
             'beta': v.encoded(v.BETA), 'clipped': clipped, 'tolerance': v.encoded(v.TOLERANCE),
             'max_boxes': 511, 'evaluated_boxes': 1, 'trace': [node], 'retained_ids': [], 'hull': None,
             'shared_tests': 1, 'propagation_rounds': 8, 'mode': method, 'rounds': 8, 'budget': 511,
             'channel_precheck': 'none', 'initial_bounds': initial, 'scope': v.GROUP_SCOPE}
    return outer, observation, calibration


class ObservationTests(unittest.TestCase):
    def test_clip_endpoints_and_unclipped_rgb_intersection(self):
        values = image()
        values[:3] = [Q(0)] * 3
        values[3:6] = [Q(1)] * 3
        values[6:9] = [Q(1, 2), Q(51, 100), Q(49, 100)]
        bands = v.pixel_bands(values, True)
        self.assertEqual(bands[0], (None, Q(1)))
        self.assertEqual(bands[1], (Q(1, 50), None))
        self.assertEqual(bands[2], (Q(51, 100), Q(51, 100)))
        self.assertEqual(v.pixel_bands(values, False)[0], (Q(51, 50), Q(1)))

    def test_clip_one_sided_preserves_feasible_saturation(self):
        self.assertTrue(v.pixel_bands(image(1), True)[0][1] is None)
        self.assertTrue(v.pixel_bands(image(0), True)[0][0] is None)
        self.assertNotEqual(v.pixel_bands(image(1), True), v.pixel_bands(image(1), False))

    def test_clip_rejects_outside_range_and_float(self):
        for values in (image(Q(101, 100)), image(-1), [1.0] * 432):
            with self.assertRaises(v.VerificationError):
                v.pixel_bands(values, True)

    def test_joint_view_variable_and_calibrated_marginalization(self):
        bounds = [[v.encoded(0), v.encoded(Q(1, 50))],
                  [v.encoded(Q(1, 100)), v.encoded(Q(1, 100))],
                  [v.encoded(0), v.encoded(0)]]
        areas = geometry(Q(1, 2))
        joint = v.necessary_rows(image(Q(1, 2)), areas, False, 'joint', bounds)
        calibrated = v.necessary_rows(image(Q(1, 2)), areas, False, 'calibrated', bounds)
        self.assertEqual(joint[0], ('upper', (Q(0), Q(0), Q(0), Q(1), Q(0), Q(0)), Q(13, 25)))
        self.assertEqual(joint[96][1][3:], (Q(0), Q(1), Q(0)))
        self.assertEqual(joint[192][1][3:], (Q(0), Q(0), Q(1)))
        self.assertEqual(calibrated[1], ('lower', (Q(1, 2), Q(0), Q(0)), Q(12, 25)))
        self.assertEqual(calibrated[96][2], Q(51, 100))

    def test_saturation_omits_only_absent_sides(self):
        ts = unmasked_calibration()['intervals']
        self.assertEqual(len(v.necessary_rows(image(1), geometry(), True, 'joint', ts)), 144)
        self.assertTrue(all(row[0] == 'upper' for row in v.necessary_rows(image(1), geometry(), True, 'joint', ts)))
        self.assertTrue(all(row[0] == 'lower' for row in v.necessary_rows(image(0), geometry(), True, 'joint', ts)))


class BackgroundTests(unittest.TestCase):
    def test_background_checks_every_branch_and_full_domain(self):
        calls = []
        def geo(branch, radius, left, right):
            calls.append((branch, radius, left, right))
            areas = list(geometry())
            if branch == 'full':
                areas[23] = (0, 0, 0, 0, 0, Q(1, 2))
            return areas
        mask = v.public_background_mask(geo)
        self.assertEqual(sum(mask), 143)
        self.assertFalse(mask[23])
        self.assertEqual(calls, [(branch, v.RADIUS, *v.DOMAIN) for branch in v.BRANCHES])

    def test_frozen_background_geometry_counts_are_derived(self):
        _model, geo = v._historical.shared_kernels()
        mask = v.public_background_mask(geo)
        self.assertEqual([sum(mask[48 * n:48 * (n + 1)]) for n in range(3)], [18, 12, 18])

    def test_positive_negative_raw_bias_and_clip_calibration(self):
        mask = [True] * 144
        values = image()
        values[:144] = [Q(99, 100)] * 144
        values[288:] = [Q(101, 100)] * 144
        raw = v.calibration_from_public(values, False, mask)
        self.assertEqual([tuple(map(v.exact, row)) for row in raw['intervals']], [(Q(1, 100), Q(1, 50)), (Q(0), Q(1, 50)), (Q(0), Q(1, 100))])
        clipped = [min(Q(1), y) for y in values]
        clip = v.calibration_from_public(clipped, True, mask)
        self.assertEqual(tuple(map(v.exact, clip['intervals'][2])), (Q(0), Q(1, 50)))

    def test_calibration_crossing_certificate_incompatible(self):
        values = image(0)
        mask = [True] + [False] * 143
        calibration = v.calibration_from_public(values, True, mask)
        self.assertTrue(calibration['excluded'])
        self.assertEqual(calibration['certificate']['contradiction'], {'kind': 'interval', 'view': 0, 'pixel': 0,
                         'lower': v.encoded(1), 'upper': v.encoded(Q(1, 50))})

    def test_false_background_or_false_clipping_calibration_rejected(self):
        geo = lambda *args: geometry()
        mask = v.public_background_mask(geo)
        arm = {'background_mask': mask, 'calibration': v.calibration_from_public(image(), True, mask)}
        arm['calibration_intervals'] = arm['calibration']['intervals']
        v.verify_arm_public(arm, image(), True, geo, 'good')
        corrupted = copy.deepcopy(arm)
        corrupted['background_mask'][0] = False
        with self.assertRaises(v.VerificationError):
            v.verify_arm_public(corrupted, image(), True, geo, 'false mask')
        corrupted = copy.deepcopy(arm)
        corrupted['calibration'] = v.calibration_from_public(image(Q(101, 100)), False, mask)
        with self.assertRaises(v.VerificationError):
            v.verify_arm_public(corrupted, image(), True, geo, 'false clip')


class CertificateTests(unittest.TestCase):
    def test_exact_updates_and_row_contradiction(self):
        rows = (('upper', (Q(1), Q(0), Q(0)), Q(2, 5)), ('lower', (Q(1), Q(0), Q(0)), Q(3, 5)))
        result = v.independent_contraction(rows, [(0, 1)] * 3)
        self.assertTrue(result['excluded'])
        self.assertEqual(result['certificate']['steps'], [{'round': 0, 'row': 0, 'variable': 0, 'side': 'upper', 'value': v.encoded(Q(2, 5))}])
        self.assertEqual(result['certificate']['contradiction'], {'kind': 'row', 'round': 0, 'row': 1,
                         'total': v.encoded(Q(2, 5)), 'bound': v.encoded(Q(3, 5))})
        v.verify_contraction(rows, [(0, 1)] * 3, result['certificate'])

    def test_wrong_row_update_missing_update_and_fake_crossing_fail(self):
        rows = (('upper', (Q(1), Q(0), Q(0)), Q(2, 5)), ('lower', (Q(1), Q(0), Q(0)), Q(3, 5)))
        certificate = v.independent_contraction(rows, [(0, 1)] * 3)['certificate']
        for field in ('row', 'value', 'missing', 'crossing', 'prior'):
            bad = copy.deepcopy(certificate)
            if field == 'row':
                bad['contradiction']['row'] = 0
            elif field == 'value':
                bad['steps'][0]['value'] = v.encoded(Q(1, 5))
            elif field == 'missing':
                bad['steps'] = []
            elif field == 'crossing':
                bad['contradiction'] = {'kind': 'interval', 'round': 0, 'row': 1, 'variable': 0,
                                        'lower': v.encoded(Q(3, 5)), 'upper': v.encoded(Q(2, 5))}
            else:
                bad['initial_bounds'][0][1] = v.encoded(Q(1, 2))
            with self.subTest(field=field), self.assertRaises(v.VerificationError):
                v.verify_contraction(rows, [(0, 1)] * 3, bad)

    def test_three_six_dimensional_and_shared_dependency(self):
        bounds = [(0, 1)] * 3 + [(0, Q(1, 50))] * 3
        rows = (('upper', (1, 0, 0, 1, 0, 0), Q(1, 2)), ('lower', (1, 0, 0, 1, 0, 0), Q(1, 2)))
        result = v.independent_contraction(rows, bounds)
        self.assertFalse(result['excluded'])
        self.assertEqual(tuple(map(v.exact, result['bounds'][0])), (Q(12, 25), Q(1, 2)))
        # A common d,t must satisfy both pixels, which separate pixel variables could evade.
        conflict = rows + (('lower', (1, 0, 0, 1, 0, 0), Q(3, 5)),)
        self.assertTrue(v.independent_contraction(conflict, bounds)['excluded'])

    def test_tree_certificate_budget_provenance_and_union(self):
        outer, observation, calibration = excluded_branch()
        checked = v.verify_branch(outer, observation, False, 'joint', 'none', lambda *args: geometry(), calibration, 'good tree')
        self.assertEqual(checked['evaluated'], 1)
        self.assertIsNone(checked['hull'])
        for change in ('parent', 'counter', 'row', 'mode', 'retained', 'orphan'):
            bad = copy.deepcopy(outer)
            if change == 'parent':
                bad['trace'][0]['parent'] = 0
            elif change == 'counter':
                bad['evaluated_boxes'] = 0
            elif change == 'row':
                bad['trace'][0]['certificate']['contradiction']['bound'] = v.encoded(0)
            elif change == 'mode':
                bad['mode'] = 'calibrated'
            elif change == 'retained':
                bad['retained_ids'] = [0]
            else:
                bad['trace'].append(copy.deepcopy(bad['trace'][0]))
                bad['trace'][-1]['id'] = 1
            with self.subTest(change=change), self.assertRaises(v.VerificationError):
                v.verify_branch(bad, observation, False, 'joint', 'none', lambda *args: geometry(), calibration, 'bad tree')

    def test_empty_calibration_requires_exclusion_root_and_no_solver_work(self):
        calibration = v.calibration_from_public(image(0), True, [True] * 144)
        outer, observation, _oldcal = excluded_branch('calibrated', True)
        outer['trace'] = [{'id': 0, 'parent': None, 'lo': v.encoded(540), 'hi': v.encoded(660),
                           'status': 'excluded_calibration', 'certificate': calibration['certificate']}]
        outer['evaluated_boxes'] = outer['shared_tests'] = 0
        result = v.verify_branch(outer, observation, True, 'calibrated', 'none', lambda *args: geometry(), calibration, 'incompatible')
        self.assertIsNone(result['hull'])


class IntegrityTests(unittest.TestCase):
    def test_source_parse_fence_and_public_arguments(self):
        source = (v.ROOT / 'run_structured.py').read_text()
        self.assertEqual(v.audit_runner_source(source)['status'], 'PASS')
        for bad in (source.replace('len(decisions) != 3888', 'len(decisions) != 3887'),
                    source.replace("execute_new(observation, radius, method, clipped)", "execute_new(observation, radius, method, clipped, truth)"),
                    source.replace("truths = private_truths(observations)", "truths = private_truths(observations)\n    private_truths(observations)")):
            with self.assertRaises(v.VerificationError):
                v.audit_runner_source(bad)

    def test_missing_action_gate(self):
        with self.assertRaises(v.VerificationError):
            v.verify_action_population([], {}, {})

    def test_false_historical_hash_provenance(self):
        with patch.object(v, 'historical_inputs', return_value={'/frozen/input': 'correct'}):
            with self.assertRaises(v.VerificationError):
                v.verify_inputs({'source_root': str(v.PREV_OUTPUT), 'files': {'/frozen/input': 'wrong'}})

    def test_shared_budget_uses_remaining_window(self):
        with patch.object(v.time, 'time', return_value=1050):
            budget = {'start_unix': 1000, 'deadline_unix': 1100, 'limit_seconds': 5400,
                      'scope': 'one main run plus complete verification; do not reset'}
            self.assertEqual(v.verify_budget(budget), 1100)
        with patch.object(v.time, 'time', return_value=1100), self.assertRaises(TimeoutError):
            v.verify_budget(budget)

    def test_truth_error_and_latent_witness_independently_reconstructed(self):
        task = {'task_id': 'raw-o000-s0', 'base_task_id': 'o000-s0', 'observation_id': 'raw-o000', 'incumbent': v.encoded(600)}
        private = {'truth': v.encoded(600), 'unit_id': 'raw', 'rear_branch': 'none', 'case': 'uniform_isolated', 'seed': 61001,
                   'colors': [v.encoded(Q(1, 5))] * 3, 'actual_view_bias': list(map(v.encoded, (-v.BETA, 0, v.BETA)))}
        checked = {'hull': (Q(590), Q(610)), 'intervals': [(Q(590), Q(610))], 'retained_branches': ['none'], 'evaluated': 1, 'budget_leaves': 0}
        row = v.make_row(task, private, {'method': 'joint', 'reused': False, 'output': v.encoded(601), 'status': 'MOVE'}, checked)
        self.assertEqual(row['ae_mm'], '1')
        self.assertEqual(row['actual_gain_mm2'], '-1')
        node = {'id': 0, 'lo': v.encoded(590), 'hi': v.encoded(610), 'darkness_bounds': [[v.encoded(0), v.encoded(1)]] * 3,
                't_bounds': [[v.encoded(0), v.encoded(Q(1, 50))]] * 3}
        branches = {'none': {**checked, 'retained': [node]}, 'partial': {'intervals': [], 'retained': []}, 'full': {'intervals': [], 'retained': []}}
        public = {'observation_id': task['observation_id'], 'unit_id': 'raw'}
        cov = v.make_coverage(public, private, 'joint', checked, branches)
        self.assertTrue(cov['latent_bias_covered'])
        node['t_bounds'][0][1] = v.encoded(Q(1, 100))
        self.assertFalse(v.make_coverage(public, private, 'joint', checked, branches)['latent_bias_covered'])


if __name__ == '__main__':
    unittest.main()
