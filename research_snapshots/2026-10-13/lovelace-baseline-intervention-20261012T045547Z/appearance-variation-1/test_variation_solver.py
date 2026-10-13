"""Hand-built public fixtures for signed rows and frozen endpoint bridges."""
from copy import deepcopy
from fractions import Fraction as Q
import inspect
import sys
import unittest

sys.dont_write_bytecode = True
import variation_solver as s

WHITE_TIMINGS = {}
_FIXTURES, _BOUNDED_RESULTS = {}, {}


def rgb(values):
    return tuple(channel for value in values for channel in (value,) * 3)


def fixture(eta=Q(0)):
    """Fixed point with legal same-material variation, unrelated to batch seeds."""
    if eta not in _FIXTURES:
        base = (Q(1, 5), Q(2, 3), Q(4, 5))
        directions = ((-1, 0, 1), (0, 1, -1), (1, -1, 0))
        materials = tuple(tuple(darkness + eta * direction * min(darkness, 1 - darkness)
                                for darkness, direction in zip(base, row)) for row in directions)
        clean = []
        for view, darks in enumerate(materials):
            point = s.geometry.point_rgb('partial', Q(617), (Q(-1, 10), Q(0), Q(1, 10)),
                                         tuple(1 - value for value in darks), Q(80))
            clean.extend(point[144 * view:144 * (view + 1)])
        raw = tuple(value + (-s.BETA, Q(0), s.BETA)[index // 144]
                    + (-s.EPSILON, Q(0), s.EPSILON)[index % 3]
                    for index, value in enumerate(clean))
        image = tuple(min(Q(1), max(Q(0), value)) for value in raw)
        _FIXTURES[eta] = (image, tuple(value for row in materials for value in row))
    return _FIXTURES[eta]


def bounded_result(method, eta):
    key = method, eta
    if key not in _BOUNDED_RESULTS:
        image, _materials = fixture(eta)
        s.clear_geometry_cache()
        _BOUNDED_RESULTS[key] = s.solve_public(list(map(s.encode, image)), s.encode(Q(1, 5)),
                                              s.encode(Q(80)), method, s.encode(Q(1, 10)))
    return _BOUNDED_RESULTS[key]


def without_new_metadata_and_timing(value):
    return {key: item for key, item in value.items()
            if key not in ('appearance_model', 'drift_bound', 'branch_seconds',
                           'outer_seconds', 'preparation_seconds')}


def signed_row(first, second, dimension=9):
    return tuple(Q(1) if index == first else Q(-1) if index == second else Q(0)
                 for index in range(dimension))


def replay_certificate(test, result, rows):
    """Check recorded signed candidates by summing the other variable extrema."""
    certificate = result['certificate']
    intervals = [list(map(s.decode, pair)) for pair in certificate['initial_bounds']]
    for step in certificate['steps']:
        kind, coefficients, bound = rows[step['row']]
        variable = step['variable']
        coefficient = coefficients[variable]
        others = Q(0)
        for index, other_coefficient in enumerate(coefficients):
            if index == variable:
                continue
            endpoint = (0 if other_coefficient >= 0 else 1) if kind == 'upper' else (1 if other_coefficient >= 0 else 0)
            others += other_coefficient * intervals[index][endpoint]
        candidate = (bound - others) / coefficient
        side = ('upper' if coefficient > 0 else 'lower') if kind == 'upper' else ('lower' if coefficient > 0 else 'upper')
        test.assertEqual(step['side'], side)
        test.assertEqual(s.decode(step['value']), candidate)
        endpoint = 1 if side == 'upper' else 0
        test.assertTrue(candidate < intervals[variable][1] if endpoint else candidate > intervals[variable][0])
        intervals[variable][endpoint] = candidate
    contradiction = certificate['contradiction']
    if contradiction['kind'] == 'initial_interval':
        variable = contradiction['variable']
        test.assertGreater(intervals[variable][0], intervals[variable][1])
    elif contradiction['kind'] == 'interval':
        variable = contradiction['variable']
        test.assertEqual(list(map(s.decode, (contradiction['lower'], contradiction['upper']))), intervals[variable])
        test.assertGreater(intervals[variable][0], intervals[variable][1])
    else:
        test.assertEqual(contradiction['kind'], 'row')
        kind, coefficients, bound = rows[contradiction['row']]
        total = Q(0)
        for index, coefficient in enumerate(coefficients):
            endpoint = (0 if coefficient >= 0 else 1) if kind == 'upper' else (1 if coefficient >= 0 else 0)
            total += coefficient * intervals[index][endpoint]
        test.assertEqual(s.decode(contradiction['total']), total)
        test.assertEqual(s.decode(contradiction['bound']), bound)
        test.assertTrue(total > bound if kind == 'upper' else total < bound)


class VariationSolverTests(unittest.TestCase):
    def test_registered_api_and_constants(self):
        signature = inspect.signature(s.solve_public)
        self.assertEqual(tuple(signature.parameters),
                         ('observation', 'radius', 'baseline', 'method', 'drift_bound', 'clipped'))
        self.assertIs(signature.parameters['clipped'].default, True)
        self.assertIs(signature.parameters['drift_bound'].default, inspect.Parameter.empty)
        self.assertEqual((s.ROUNDS, s.MAX_BOXES, s.TOLERANCE), (8, 511, Q(15, 32)))
        self.assertEqual((s.PIXEL_COUNT, s.RGB_COUNT), (144, 432))
        self.assertEqual(s.VIEW_PAIRS, ((0, 1), (0, 2), (1, 2)))
        self.assertEqual(s.APPEARANCE_MODELS, ('strict_shared', 'independent', 'bounded'))

    def test_difference_rows_have_exact_pair_material_direction_order(self):
        for method in s.METHODS:
            rows = s.difference_rows(method, s.encode(Q(1, 10)))
            dimension = 12 if method == 'joint' else 9
            self.assertEqual(len(rows), 18)
            expected = []
            for first, second in ((0, 1), (0, 2), (1, 2)):
                for material in range(3):
                    forward = signed_row(3 * first + material, 3 * second + material, dimension)
                    expected.extend((('upper', forward, Q(1, 10)),
                                     ('upper', tuple(-value for value in forward), Q(1, 10))))
            self.assertEqual(rows, tuple(expected))
            for kind, coefficients, bound in rows:
                self.assertEqual((kind, bound), ('upper', Q(1, 10)))
                self.assertEqual(sorted(value for value in coefficients if value), [Q(-1), Q(1)])
                self.assertEqual(coefficients[9:], (Q(0),) * (dimension - 9))

    def test_original_photometric_prefix_and_clipping_order_unchanged(self):
        choices = ((Q(0),) * 3, (Q(1),) * 3, (Q(1, 2),) * 3, (Q(0), Q(1), Q(1, 2)))
        image = tuple(channel for pixel in range(144) for channel in choices[pixel % 4])
        areas = tuple((Q(pixel + 1, 2000), Q(pixel + 201, 2000), Q(1, 7), Q(2, 7),
                       Q(0), Q(1, 8)) for pixel in range(144))
        ts = ((Q(0), Q(1, 50)),) * 3
        for clipped in (True, False):
            for method in s.METHODS:
                original = s.appearance_engine.necessary_rows(image, areas, method, clipped=clipped,
                                                              calibration_intervals=ts)
                current = s.necessary_rows(image, areas, method, clipped=clipped,
                                           drift_bound=Q(1, 10), calibration_intervals=ts)
                self.assertEqual(current[:-18], original)
                self.assertEqual(current[-18:], s.difference_rows(method, Q(1, 10)))
                self.assertEqual(len(original), 216 if clipped else 288)

    def test_nonnegative_contractor_full_identity_in_all_old_dimensions(self):
        for dimension in (3, 6, 9, 12):
            prior = ((Q(0), Q(1)),) * dimension
            padding = (Q(0),) * (dimension - 3)
            upper = ('upper', (Q(1), Q(1), Q(0)) + padding, Q(3, 4))
            retained = ('lower', (Q(1), Q(0), Q(0)) + padding, Q(1, 2))
            excluded = ('lower', (Q(1), Q(0), Q(0)) + padding, Q(4, 5))
            dense = (('upper', tuple(Q(index + 1, dimension + 1) for index in range(dimension)), Q(4, 5)),
                     ('lower', tuple(Q(dimension - index, dimension + 1) for index in range(dimension)), Q(3, 5)))
            cases = (((), prior), ((upper, retained), prior), ((upper, excluded), prior), (dense, prior),
                     ((('upper', (Q(0),) * dimension, Q(-1)),), prior),
                     ((), ((Q(1, 2), Q(1, 3)),) + prior[1:]))
            for rows, bounds in cases:
                for rounds in (0, 1, 8):
                    with self.subTest(dimension=dimension, rounds=rounds, rows=rows):
                        self.assertEqual(s.propagate(rows, bounds=bounds, rounds=rounds),
                                         s.appearance_engine.propagate(rows, bounds=bounds, rounds=rounds))

    def test_negative_coefficients_tighten_correct_opposite_endpoints(self):
        coefficients = signed_row(0, 3)
        prior = list(s.PER_VIEW_MATERIAL_BOUNDS)
        prior[0], prior[3] = (Q(1, 5), Q(1)), (Q(0), Q(2, 5))
        upper = s.propagate((('upper', coefficients, Q(1, 10)),), bounds=prior, rounds=1)
        self.assertFalse(upper['excluded'])
        self.assertEqual(tuple(map(s.decode, upper['bounds'][0])), (Q(1, 5), Q(1, 2)))
        self.assertEqual(tuple(map(s.decode, upper['bounds'][3])), (Q(1, 10), Q(2, 5)))
        self.assertEqual(upper['updates'], 2)
        prior[0], prior[3] = (Q(0), Q(2, 5)), (Q(1, 5), Q(1))
        lower = s.propagate((('lower', coefficients, Q(-1, 10)),), bounds=prior, rounds=1)
        self.assertFalse(lower['excluded'])
        self.assertEqual(tuple(map(s.decode, lower['bounds'][0])), (Q(1, 10), Q(2, 5)))
        self.assertEqual(tuple(map(s.decode, lower['bounds'][3])), (Q(1, 5), Q(1, 2)))
        self.assertEqual(lower['updates'], 2)

    def test_first_signed_contradiction_uses_minimum_or_maximum(self):
        coefficients = signed_row(0, 3)
        for kind in ('upper', 'lower'):
            prior = list(s.PER_VIEW_MATERIAL_BOUNDS)
            if kind == 'upper':
                prior[0], prior[3], bound, total = (Q(4, 5), Q(1)), (Q(0), Q(2, 5)), Q(1, 10), Q(2, 5)
            else:
                prior[0], prior[3], bound, total = (Q(0), Q(2, 5)), (Q(4, 5), Q(1)), Q(-1, 10), Q(-2, 5)
            rows = ((kind, coefficients, bound), ('upper', (Q(0),) * 9, Q(-1)))
            current = s.propagate(rows, bounds=prior)
            self.assertTrue(current['excluded'])
            self.assertEqual((current['rounds'], current['updates']), (1, 0))
            self.assertEqual(current['certificate']['contradiction'],
                             {'kind': 'row', 'round': 0, 'row': 0,
                              'total': s.encode(total), 'bound': s.encode(bound)})
            replay_certificate(self, current, rows)

    def test_signed_update_certificate_replay_and_tamper_detection(self):
        prior = list(s.PER_VIEW_MATERIAL_BOUNDS)
        prior[0], prior[3] = (Q(1, 5), Q(1)), (Q(0), Q(2, 5))
        rows = (('upper', signed_row(0, 3), Q(1, 10)),
                ('lower', tuple(Q(int(index == 0)) for index in range(9)), Q(3, 5)))
        current = s.propagate(rows, bounds=prior)
        self.assertTrue(current['excluded'])
        self.assertEqual([(step['variable'], step['side']) for step in current['certificate']['steps']],
                         [(0, 'upper'), (3, 'lower')])
        self.assertEqual(current['certificate']['contradiction']['row'], 1)
        replay_certificate(self, current, rows)
        tampered = deepcopy(current)
        tampered['certificate']['steps'][0]['value'] = s.encode(Q(2, 5))
        with self.assertRaises(AssertionError):
            replay_certificate(self, tampered, rows)
        tampered = deepcopy(current)
        tampered['certificate']['steps'][1]['side'] = 'upper'
        with self.assertRaises(AssertionError):
            replay_certificate(self, tampered, rows)

    def test_signed_dense_rows_keep_valid_witness(self):
        materials = (Q(1, 5), Q(3, 10), Q(2, 5), Q(1, 4), Q(7, 25), Q(9, 20),
                     Q(11, 50), Q(31, 100), Q(43, 100))
        for method in s.METHODS:
            witness = materials + (Q(1, 50), Q(1, 100), Q(0)) if method == 'joint' else materials
            prior = s.PER_VIEW_MATERIAL_BOUNDS + ((Q(0), Q(1, 50)),) * 3 if method == 'joint' else s.PER_VIEW_MATERIAL_BOUNDS
            rows = list(s.difference_rows(method, Q(1, 10)))
            for number in range(4):
                coefficients = tuple(Q((index + number) % 5 - 2, 3) for index in range(len(witness)))
                total = sum((coefficient * value for coefficient, value in zip(coefficients, witness)), Q(0))
                kind = 'upper' if number % 2 == 0 else 'lower'
                bound = total + Q(1, 10) if kind == 'upper' else total - Q(1, 10)
                rows.append((kind, coefficients, bound))
            for bounds in (prior, tuple((value, value) for value in witness)):
                current = s.propagate(rows, bounds=bounds)
                self.assertFalse(current['excluded'])
                self.assertTrue(all(s.decode(pair[0]) <= value <= s.decode(pair[1])
                                    for value, pair in zip(witness, current['bounds'])))

    def test_pairwise_spread_is_not_only_reference_view_distance(self):
        point = [Q(1, 2)] * 9
        point[0], point[3], point[6] = Q(0), Q(1, 10), Q(1, 5)
        rows = s.difference_rows('calibrated', Q(1, 10))
        current = s.propagate(rows, bounds=tuple((value, value) for value in point))
        self.assertTrue(current['excluded'])
        self.assertEqual(current['certificate']['contradiction']['row'], 7)
        self.assertEqual(s.decode(current['certificate']['contradiction']['total']), Q(1, 5))
        replay_certificate(self, current, rows)
        point[0], point[3], point[6] = Q(1, 4), Q(3, 10), Q(7, 20)
        self.assertFalse(s.propagate(rows, bounds=tuple((value, value) for value in point))['excluded'])

    def test_public_rationals_bounds_and_dimensions_rejected(self):
        image = [Q(1)] * 432
        invalid = (True, 0.1, Q(-1, 10), Q(11, 10),
                   {'numerator': True, 'denominator': 1}, {'numerator': 1, 'denominator': -10},
                   {'numerator': 1, 'denominator': 10, 'extra': 0})
        for delta in invalid:
            with self.subTest(delta=delta), self.assertRaises((TypeError, ValueError)):
                s.solve_public(image, Q(1, 5), Q(80), 'joint', delta)
        for delta in (Q(0), Q(1, 10), Q(1)):
            for bad in (True, 0.5, {'numerator': 1, 'denominator': 0}):
                with self.assertRaises((TypeError, ValueError)):
                    s.solve_public([bad] + image[1:], Q(1, 5), Q(80), 'joint', delta)
            with self.assertRaises(ValueError):
                s.solve_public(image[:-1], Q(1, 5), Q(80), 'joint', delta)
            with self.assertRaises(ValueError):
                s.solve_public(image, Q(1, 5), Q(75), 'joint', delta)
            with self.assertRaises(TypeError):
                s.solve_public(image, Q(1, 5), Q(80), 'joint', delta, clipped=1)
        for dimension in (0, 4, 5, 7, 8, 10, 11, 13):
            with self.assertRaises(ValueError):
                s.propagate((), bounds=((Q(0), Q(1)),) * dimension)
        with self.assertRaises(ValueError):
            s.propagate((('upper', (Q(0),) * 8, Q(0)),), bounds=s.PER_VIEW_MATERIAL_BOUNDS)
        with self.assertRaises(TypeError):
            s.propagate((('upper', (0.5,) + (Q(0),) * 8, Q(0)),), bounds=s.PER_VIEW_MATERIAL_BOUNDS)
        with self.assertRaises(ValueError):
            s.propagate((), bounds=((Q(-1), Q(1)),) * 9)
        crossed = ((Q(1, 2), Q(1, 3)),) + s.PER_VIEW_MATERIAL_BOUNDS[1:]
        current = s.propagate((), bounds=crossed, rounds=0)
        self.assertTrue(current['excluded'])
        replay_certificate(self, current, ())

    def test_delta_zero_and_one_full_frozen_identity(self):
        image, _materials = fixture()
        for method in s.METHODS:
            for delta in (Q(0), Q(1)):
                s.clear_geometry_cache()
                if delta == 0:
                    original = s.baseline_engine.solve_public(image, Q(1, 5), Q(80), method)
                else:
                    original = s.appearance_engine.solve_public(image, Q(1, 5), Q(80), method,
                                                                appearance_scope='per_view')
                s.clear_geometry_cache()
                current = s.solve_public(list(map(s.encode, image)), s.encode(Q(1, 5)),
                                         s.encode(Q(80)), method, s.encode(delta))
                self.assertEqual(without_new_metadata_and_timing(current), without_new_metadata_and_timing(original))
                self.assertEqual(current['appearance_model'], 'strict_shared' if delta == 0 else 'independent')
                self.assertEqual(current['drift_bound'], s.encode(delta))
                if delta == 0:
                    self.assertNotIn('appearance_scope', current)
                    self.assertNotIn('material_variable_names', current)
                    self.assertTrue(all('appearance_scope' not in branch for branch in current['branches'].values()))
                else:
                    self.assertEqual(current['appearance_scope'], 'per_view')

    def test_legal_pairwise_generated_point_satisfies_all_rows(self):
        biases, truth = (Q(1, 50), Q(1, 100), Q(0)), Q(617)
        for eta in (Q(0), Q(1, 10)):
            image, materials = fixture(eta)
            self.assertTrue(all(0 <= value <= 1 for value in materials))
            self.assertTrue(all(max(materials[material::3]) - min(materials[material::3]) <= Q(1, 10)
                                for material in range(3)))
            areas = s.geometry.geometry_bounds('partial', Q(1, 5), truth, truth, Q(80))
            background = s.baseline_engine.background_mask(Q(1, 5), Q(80))
            calibration = s.frozen.calibration_bounds(image, background, clipped=True)
            ts = tuple(tuple(map(s.decode, pair)) for pair in calibration['intervals'])
            for method in s.METHODS:
                rows = s.necessary_rows(image, areas, method, clipped=True, drift_bound=Q(1, 10),
                                        calibration_intervals=ts)
                witness = materials + biases if method == 'joint' else materials
                for kind, coefficients, bound in rows:
                    total = sum((coefficient * value for coefficient, value in zip(coefficients, witness)), Q(0))
                    self.assertTrue(total <= bound if kind == 'upper' else total >= bound)
                self.assertFalse(s.propagate(rows, bounds=tuple((value, value) for value in witness))['excluded'])

    def test_bounded_full_fixture_preserves_truth_calibration_and_schema(self):
        biases, truth = (Q(1, 50), Q(1, 100), Q(0)), Q(617)
        for eta in (Q(0), Q(1, 10)):
            image, materials = fixture(eta)
            background = s.baseline_engine.background_mask(Q(1, 5), Q(80))
            calibration = s.frozen.calibration_bounds(image, background, clipped=True)
            for method in s.METHODS:
                current = bounded_result(method, eta)
                self.assertEqual(current['appearance_model'], 'bounded')
                self.assertEqual(current['drift_bound'], s.encode(Q(1, 10)))
                self.assertEqual(current['appearance_scope'], 'per_view')
                self.assertEqual(current['material_variable_names'], list(s.appearance_engine.material_variable_names()))
                self.assertEqual(current['background_mask'], list(background))
                self.assertEqual(current['calibration'], calibration)
                self.assertTrue(any(s.decode(lo) <= truth <= s.decode(hi)
                                    for lo, hi in current['outer']['retained_intervals']))
                for name, branch in current['branches'].items():
                    self.assertEqual(branch['scope'], s.BOUNDED_SCOPE)
                    self.assertEqual(branch['appearance_scope'], 'per_view')
                    self.assertNotIn('drift_bound', branch)
                    self.assertNotIn('appearance_model', branch)
                    self.assertEqual(len(branch['initial_bounds']), 12 if method == 'joint' else 9)
                    self.assertEqual(branch['shared_tests'], branch['evaluated_boxes'])
                    self.assertLessEqual(branch['evaluated_boxes'], 511)
                    self.assertEqual([node['id'] for node in branch['trace']], list(range(len(branch['trace']))))
                    for node in branch['trace']:
                        if 'darkness_bounds' in node:
                            self.assertEqual((len(node['darkness_bounds']), len(node['t_bounds'])), (9, 3))
                        if name == 'partial' and s.decode(node['lo']) <= truth <= s.decode(node['hi']):
                            self.assertIn(node['status'], ('split', 'retained_tolerance', 'unresolved_budget'))
                            self.assertTrue(all(s.decode(pair[0]) <= value <= s.decode(pair[1])
                                                for value, pair in zip(materials, node['darkness_bounds'])))
                            self.assertTrue(all(s.decode(pair[0]) <= value <= s.decode(pair[1])
                                                for value, pair in zip(biases, node['t_bounds'])))

    def test_calibration_negative_control_all_models(self):
        for method in s.METHODS:
            for delta in (Q(0), Q(1, 10), Q(1)):
                current = s.solve_public([Q(0)] * 432, Q(1, 5), Q(80), method, delta)
                self.assertTrue(current['calibration']['excluded'])
                self.assertIsNone(current['outer']['hull'])
                self.assertEqual(current['outer']['retained_intervals'], [])
                for branch in current['branches'].values():
                    self.assertEqual(branch['evaluated_boxes'], 0)
                    self.assertEqual(len(branch['trace']), 1)
                    self.assertEqual(branch['trace'][0]['status'], 'excluded_calibration')
                    self.assertEqual(branch['trace'][0]['certificate'], current['calibration']['certificate'])

    def test_white_nonformal_fixture_finite_budget_and_cold_timing(self):
        for method in s.METHODS:
            s.clear_geometry_cache()
            current = s.solve_public([Q(1)] * 432, Q(1, 5), Q(80), method, Q(1, 10))
            WHITE_TIMINGS[method] = {'outer_seconds': current['outer_seconds'],
                                     'branch_seconds': current['branch_seconds'],
                                     'evaluated_boxes': [branch['evaluated_boxes']
                                                         for branch in current['branches'].values()]}
            self.assertEqual(current['outer']['hull'], [s.encode(Q(540)), s.encode(Q(660))])
            for branch in current['branches'].values():
                self.assertEqual((branch['evaluated_boxes'], branch['shared_tests']), (511, 511))
                self.assertEqual(len(branch['trace']), 511)
                self.assertEqual(len(branch['retained_ids']), 256)
                self.assertEqual(len(branch['initial_bounds']), 12 if method == 'joint' else 9)


if __name__ == '__main__':
    unittest.main()
