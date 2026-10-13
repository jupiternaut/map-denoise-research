"""Hand-built nonformal fixtures; no population seeds or private labels."""
from copy import deepcopy
from fractions import Fraction as Q
import inspect
import sys
import unittest

sys.dont_write_bytecode = True
import appearance_solver as s

WHITE_TIMINGS = {}
_RESULTS = {}


def rgb(values):
    return tuple(channel for value in values for channel in (value,) * 3)


def fixture():
    """One fixed exact point with shared material, bounded bias and RGB error."""
    image = s.geometry.point_rgb('partial', Q(617), (Q(-1, 10), Q(0), Q(1, 10)),
                                 (Q(0), Q(1, 3), Q(4, 5)), Q(80))
    raw = tuple(value + (-s.BETA, Q(0), s.BETA)[index // 144]
                + (-s.EPSILON, Q(0), s.EPSILON)[index % 3]
                for index, value in enumerate(image))
    return tuple(min(Q(1), max(Q(0), value)) for value in raw)


def result(method, appearance_scope):
    key = (method, appearance_scope)
    if key not in _RESULTS:
        s.clear_geometry_cache()
        _RESULTS[key] = s.solve_public(list(map(s.encode, fixture())), s.encode(Q(1, 5)),
                                      s.encode(Q(80)), method, appearance_scope)
    return _RESULTS[key]


def historical_semantics(value):
    """Only timing and explicitly added appearance metadata are omitted."""
    copied = deepcopy(value)
    for key in ('branch_seconds', 'outer_seconds', 'preparation_seconds',
                'appearance_scope', 'material_variable_names'):
        copied.pop(key, None)
    for branch in copied['branches'].values():
        branch.pop('appearance_scope', None)
        branch.pop('material_variable_names', None)
    return copied


def endpoint_image():
    choices = ((Q(0),) * 3, (Q(1),) * 3, (Q(1, 2),) * 3,
               (Q(0), Q(1), Q(1, 2)))
    return tuple(channel for pixel in range(144) for channel in choices[pixel % 4])


def independent_row_identities(image, clipped):
    records = []
    for pixel in range(len(image) // 3):
        channels = image[3 * pixel:3 * pixel + 3]
        for side, exists in (('upper', not clipped or any(value != 0 for value in channels)),
                             ('lower', not clipped or any(value != 1 for value in channels))):
            if exists:
                records.append((pixel, side))
    return records


class AppearanceSolverTests(unittest.TestCase):
    def test_registered_api_constants_and_material_order(self):
        signature = inspect.signature(s.solve_public)
        self.assertEqual(tuple(signature.parameters),
                         ('observation', 'radius', 'baseline', 'method', 'appearance_scope', 'clipped'))
        self.assertEqual(signature.parameters['appearance_scope'].default, 'per_view')
        self.assertIs(signature.parameters['clipped'].default, True)
        self.assertEqual((s.ROUNDS, s.MAX_BOXES, s.TOLERANCE), (8, 511, Q(15, 32)))
        self.assertEqual((s.PIXELS_PER_VIEW, s.PIXEL_COUNT, s.RGB_COUNT), (48, 144, 432))
        self.assertEqual(s.material_variable_names(),
                         ('d00', 'd01', 'd02', 'd10', 'd11', 'd12', 'd20', 'd21', 'd22'))
        self.assertEqual(s.material_variable_names('global_shared'), ('d0', 'd1', 'd2'))
        self.assertEqual(s.material_bounds(), s.frozen.MATERIAL_BOUNDS * 3)
        self.assertEqual(s.material_bounds('global_shared'), s.frozen.MATERIAL_BOUNDS)

    def test_strict_public_rational_encoding_boundary(self):
        self.assertEqual(s.decode({'numerator': 74, 'denominator': 202}), Q(37, 101))
        self.assertEqual(s.encode(Q(-37, 101)), {'numerator': -37, 'denominator': 101})
        self.assertEqual(s._exact({'numerator': 1, 'denominator': 5}, 'fixture'), Q(1, 5))
        for value in (True, False, 0.5, 80.0):
            with self.subTest(value=value), self.assertRaises(TypeError):
                s.encode(value)
            with self.subTest(value=value), self.assertRaises(TypeError):
                s._exact(value, 'fixture')
        invalid = ({'numerator': True, 'denominator': 1},
                   {'numerator': 1, 'denominator': False},
                   {'numerator': 0.5, 'denominator': 2},
                   {'numerator': 1, 'denominator': 0},
                   {'numerator': 1, 'denominator': -2},
                   {'numerator': 1}, {'numerator': 1, 'denominator': 2, 'extra': 0},
                   (1, 2), None)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                s.decode(value)

    def test_malformed_public_image_and_parameters_rejected(self):
        image = [Q(1)] * 432
        for size in (0, 3, 429, 431, 433, 435):
            observation = image[:size] if size <= 432 else image + [Q(1)] * (size - 432)
            with self.subTest(size=size), self.assertRaises(ValueError):
                s.solve_public(observation, Q(1, 5), Q(80), 'joint')
        cases = (((Q(-1), Q(80), 'joint', 'per_view', True), ValueError),
                 ((Q(1, 5), Q(75), 'joint', 'per_view', True), ValueError),
                 ((Q(1, 5), Q(80), 'cal', 'per_view', True), ValueError),
                 ((Q(1, 5), Q(80), 'joint', 'per_pixel', True), ValueError),
                 ((Q(1, 5), Q(80), 'joint', 'per_view', 1), TypeError),
                 ((0.2, Q(80), 'joint', 'per_view', True), TypeError),
                 ((Q(1, 5), 80.0, 'joint', 'per_view', True), TypeError))
        for arguments, error in cases:
            with self.subTest(arguments=arguments), self.assertRaises(error):
                s.solve_public(image, *arguments)
        for invalid in ({'numerator': True, 'denominator': 1},
                        {'numerator': 1, 'denominator': -2},
                        {'numerator': 1, 'denominator': 2, 'extra': 0}, True, 0.5):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                s.solve_public([invalid] + image[1:], Q(1, 5), Q(80), 'joint')
        with self.assertRaises(ValueError):
            s._image([Q(101, 100)] + image[1:], True)
        self.assertEqual(s._image([Q(101, 100)] + image[1:], False)[0], Q(101, 100))

    def test_view_major_sparse_rows_and_original_clipping_order(self):
        areas = tuple((Q(pixel + 1, 2000), Q(pixel + 201, 2000),
                       Q(pixel + 2, 2000), Q(pixel + 202, 2000),
                       Q(pixel + 3, 2000), Q(pixel + 203, 2000)) for pixel in range(144))
        image = endpoint_image()
        ts = ((Q(1, 200), Q(3, 200)), (Q(0), Q(1, 50)), (Q(1, 100), Q(1, 50)))
        for clipped in (True, False):
            identities = independent_row_identities(image, clipped)
            for method in s.METHODS:
                old = s.frozen.necessary_rows(image, areas, method, clipped=clipped,
                                              calibration_intervals=ts)
                new = s.necessary_rows(image, areas, method, clipped=clipped,
                                       calibration_intervals=ts)
                self.assertEqual(len(new), len(identities))
                self.assertEqual(len(new), len(old))
                for index, (pixel, side) in enumerate(identities):
                    view = pixel // 48
                    kind, coefficients, bound = new[index]
                    old_kind, old_coefficients, old_bound = old[index]
                    self.assertEqual((kind, bound), (side, old_bound))
                    self.assertEqual(kind, old_kind)
                    self.assertEqual(len(coefficients), 12 if method == 'joint' else 9)
                    self.assertEqual(coefficients[3 * view:3 * view + 3], old_coefficients[:3])
                    self.assertEqual([slot for slot, value in enumerate(coefficients[:9]) if value],
                                     [3 * view, 3 * view + 1, 3 * view + 2])
                    if method == 'joint':
                        self.assertEqual(coefficients[9:], old_coefficients[3:])
                        self.assertEqual([slot for slot, value in enumerate(coefficients) if value],
                                         [3 * view, 3 * view + 1, 3 * view + 2, 9 + view])
                self.assertEqual(identities[-1][0], 143)
        self.assertEqual(independent_row_identities(image, True)[:4],
                         [(0, 'lower'), (1, 'upper'), (2, 'upper'), (2, 'lower')])

    def test_global_rows_exact_historical_identity(self):
        image = endpoint_image()
        areas = ((Q(1, 7), Q(2, 7), Q(1, 9), Q(2, 9), Q(0), Q(1, 8)),) * 144
        ts = ((Q(0), Q(1, 50)),) * 3
        for clipped in (True, False):
            for method in s.METHODS:
                original = s.frozen.necessary_rows(image, areas, method, clipped=clipped,
                                                   calibration_intervals=ts)
                current = s.necessary_rows(list(map(s.encode, image)), areas, method,
                                          clipped=clipped, calibration_intervals=ts,
                                          appearance_scope='global_shared')
                self.assertEqual(current, original)

    def test_generic_three_six_contractor_full_historical_identity(self):
        for dimension in (3, 6):
            padding = (Q(0),) * (dimension - 3)
            prior = ((Q(0), Q(1)),) * dimension
            upper = ('upper', (Q(1), Q(1), Q(0)) + padding, Q(3, 4))
            lower = ('lower', (Q(1), Q(0), Q(0)) + padding, Q(1, 2))
            excluded_lower = ('lower', (Q(1), Q(0), Q(0)) + padding, Q(4, 5))
            multi = (('lower', tuple(Q(index + 1, dimension + 1) for index in range(dimension)), Q(3, 5)),
                     ('upper', tuple(Q(dimension - index, dimension + 1) for index in range(dimension)), Q(4, 5)),
                     ('upper', tuple(Q(int(index == dimension - 1)) for index in range(dimension)), Q(2, 7)))
            cases = (((), prior), ((upper, lower), prior), ((upper, excluded_lower), prior),
                     (multi, prior), ((('upper', (Q(0),) * dimension, Q(-1)),), prior),
                     ((), ((Q(1, 2), Q(1, 3)),) + prior[1:]))
            for rows, bounds in cases:
                for rounds in (0, 1, 8):
                    with self.subTest(dimension=dimension, rows=rows, rounds=rounds):
                        self.assertEqual(s.propagate(rows, bounds=bounds, rounds=rounds),
                                         s.frozen.propagate(rows, bounds=bounds, rounds=rounds))

    def test_generic_dimensions_and_exact_inputs_rejected(self):
        for dimension in (0, 1, 2, 4, 5, 7, 8, 10, 11, 13):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                s.propagate((), bounds=((Q(0), Q(1)),) * dimension)
        for dimension in (3, 6, 9, 12):
            prior = ((Q(0), Q(1)),) * dimension
            self.assertEqual(s.propagate((), bounds=prior, rounds=0),
                             {'excluded': False, 'rounds': 0, 'updates': 0,
                              'bounds': s._encoded_pairs(prior)})
            for rows in ((('upper', (Q(0),) * (dimension - 1), Q(0)),),
                         (('upper', (Q(-1),) + (Q(0),) * (dimension - 1), Q(0)),),
                         (('equal', (Q(0),) * dimension, Q(0)),)):
                with self.subTest(dimension=dimension, rows=rows), self.assertRaises(ValueError):
                    s.propagate(rows, bounds=prior)
        for rounds in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                s.propagate((), rounds=rounds)
        for bad in (((Q(-1), Q(1)),) * 9, ((),) * 9, ((Q(0),),) * 9):
            with self.assertRaises(ValueError):
                s.propagate((), bounds=bad)
        for value in (True, 0.5, {'numerator': 1, 'denominator': -2}):
            with self.assertRaises((TypeError, ValueError)):
                s.propagate((), bounds=((value, Q(1)),) * 9)

    def test_invalid_row_geometry_calibration_and_scope(self):
        image, area = rgb((Q(1, 2),)), ((Q(0),) * 6,)
        ts = ((Q(0), Q(1, 50)),) * 3
        malformed = ((), ((Q(0),) * 5,), ((Q(0), Q(-1), Q(0), Q(0), Q(0), Q(0)),),
                     ((Q(1), Q(0), Q(0), Q(0), Q(0), Q(0)),),
                     ((Q(0), Q(2), Q(0), Q(0), Q(0), Q(0)),))
        for areas in malformed:
            with self.subTest(areas=areas), self.assertRaises(ValueError):
                s.necessary_rows(image, areas, 'joint', clipped=True)
        with self.assertRaises(ValueError):
            s.necessary_rows(rgb((Q(1),) * 145), ((Q(0),) * 6,) * 145, 'joint', clipped=True)
        for calibration in (None, ts[:2], ((Q(1), Q(0)),) * 3, ((Q(-1), Q(1)),) * 3,
                            ((Q(0),),) * 3):
            with self.subTest(calibration=calibration), self.assertRaises(ValueError):
                s.necessary_rows(image, area, 'calibrated', clipped=True,
                                 calibration_intervals=calibration)
        with self.assertRaises(ValueError):
            s.necessary_rows(image, area, 'joint', clipped=True, appearance_scope='pixel')
        with self.assertRaises(ValueError):
            s.material_bounds('pixel')

    def test_true_shared_parameter_embeds_into_repeated_materials(self):
        image, truth = fixture(), Q(617)
        darkness, biases = (Q(1), Q(2, 3), Q(1, 5)), (Q(1, 50), Q(1, 100), Q(0))
        areas = s.geometry.geometry_bounds('partial', Q(1, 5), truth, truth, Q(80))
        background = s.baseline_engine.background_mask(Q(1, 5), Q(80))
        calibration = s.frozen.calibration_bounds(image, background, clipped=True)
        ts = tuple(tuple(map(s.decode, pair)) for pair in calibration['intervals'])
        for method in s.METHODS:
            old = s.necessary_rows(image, areas, method, clipped=True, calibration_intervals=ts,
                                   appearance_scope='global_shared')
            new = s.necessary_rows(image, areas, method, clipped=True, calibration_intervals=ts)
            old_point = darkness + biases if method == 'joint' else darkness
            new_point = darkness * 3 + biases if method == 'joint' else darkness * 3
            self.assertEqual(len(old), len(new))
            for (kind, a, bound), (new_kind, b, new_bound) in zip(old, new):
                old_value = sum((coefficient * value for coefficient, value in zip(a, old_point)), Q(0))
                new_value = sum((coefficient * value for coefficient, value in zip(b, new_point)), Q(0))
                self.assertEqual((new_kind, new_bound, new_value), (kind, bound, old_value))
                self.assertTrue(new_value <= bound if kind == 'upper' else new_value >= bound)
            self.assertFalse(s.propagate(new, bounds=tuple((value, value) for value in new_point))['excluded'])

    def test_different_view_materials_retained_while_global_excluded(self):
        areas = [(Q(0),) * 6 for _ in range(144)]
        grays = [Q(1)] * 144
        for view, darkness in enumerate((Q(1, 4), Q(1, 2), Q(3, 4))):
            for pixel in (48 * view, 48 * view + 1):
                areas[pixel] = (Q(1), Q(1), Q(0), Q(0), Q(0), Q(0))
                grays[pixel] = 1 - darkness
        ts = ((Q(0), Q(0)),) * 3
        for method in s.METHODS:
            outcomes = {}
            for scope in s.APPEARANCE_SCOPES:
                rows = s.necessary_rows(rgb(grays), areas, method, clipped=False,
                                        calibration_intervals=ts, appearance_scope=scope,
                                        beta=Q(0), epsilon=Q(0))
                bounds = s.material_bounds(scope) + ts if method == 'joint' else s.material_bounds(scope)
                outcomes[scope] = s.propagate(rows, bounds=bounds)
            self.assertFalse(outcomes['per_view']['excluded'])
            self.assertTrue(outcomes['global_shared']['excluded'])
            final = [tuple(map(s.decode, pair)) for pair in outcomes['per_view']['bounds']]
            self.assertEqual([final[3 * view] for view in range(3)],
                             [(Q(1, 4), Q(1, 4)), (Q(1, 2), Q(1, 2)), (Q(3, 4), Q(3, 4))])

    def test_same_view_pixels_still_share_material_and_bias(self):
        image = rgb((Q(3, 4), Q(1, 4)))
        areas = ((Q(1), Q(1), Q(0), Q(0), Q(0), Q(0)),) * 2
        ts = ((Q(0), Q(0)),) * 3
        for method in s.METHODS:
            rows = s.necessary_rows(image, areas, method, clipped=False,
                                    calibration_intervals=ts, beta=Q(0), epsilon=Q(0))
            bounds = s.material_bounds() + ts if method == 'joint' else s.material_bounds()
            contraction = s.propagate(rows, bounds=bounds)
            self.assertTrue(contraction['excluded'])
            self.assertEqual(contraction['certificate']['contradiction']['kind'], 'row')
            for pixel in range(2):
                one = s.necessary_rows(image[3 * pixel:3 * pixel + 3], areas[:1], method,
                                       clipped=False, calibration_intervals=ts,
                                       beta=Q(0), epsilon=Q(0))
                self.assertFalse(s.propagate(one, bounds=bounds)['excluded'])

    def test_global_solver_full_semantic_identity_except_new_metadata_and_timing(self):
        for method in s.METHODS:
            s.clear_geometry_cache()
            original = s.baseline_engine.solve_public(fixture(), Q(1, 5), Q(80), method)
            current = result(method, 'global_shared')
            self.assertEqual(historical_semantics(current), historical_semantics(original))
            self.assertEqual(current['appearance_scope'], 'global_shared')
            self.assertEqual(current['material_variable_names'], ['d0', 'd1', 'd2'])
            self.assertTrue(all(branch['appearance_scope'] == 'global_shared'
                                and branch['scope'] == s.GLOBAL_SCOPE
                                for branch in current['branches'].values()))

    def test_per_view_solver_preserves_calibration_and_repeated_truth(self):
        darkness = (Q(1), Q(2, 3), Q(1, 5)) * 3
        biases, truth = (Q(1, 50), Q(1, 100), Q(0)), Q(617)
        names = list(s.material_variable_names())
        for method in s.METHODS:
            old, current = result(method, 'global_shared'), result(method, 'per_view')
            self.assertEqual(current['appearance_scope'], 'per_view')
            self.assertEqual(current['material_variable_names'], names)
            self.assertEqual(current['calibration'], old['calibration'])
            self.assertEqual(current['calibration_intervals'], old['calibration_intervals'])
            self.assertEqual(current['background_mask'], old['background_mask'])
            self.assertTrue(any(s.decode(lo) <= truth <= s.decode(hi)
                                for lo, hi in current['outer']['retained_intervals']))
            for branch in current['branches'].values():
                self.assertEqual((branch['appearance_scope'], branch['scope']), ('per_view', s.PER_VIEW_SCOPE))
                self.assertEqual(branch['material_variable_names'], names)
                self.assertEqual(len(branch['initial_bounds']), 12 if method == 'joint' else 9)
                self.assertLessEqual(branch['evaluated_boxes'], 511)
                self.assertEqual(branch['shared_tests'], branch['evaluated_boxes'])
                self.assertEqual(branch['trace'][0]['id'], 0)
                self.assertEqual([node['id'] for node in branch['trace']], list(range(len(branch['trace']))))
                for node in branch['trace']:
                    if 'darkness_bounds' in node:
                        self.assertEqual((len(node['darkness_bounds']), len(node['t_bounds'])), (9, 3))
                    if node['status'] == 'split':
                        children = [branch['trace'][index] for index in node['children']]
                        middle = (s.decode(node['lo']) + s.decode(node['hi'])) / 2
                        self.assertEqual([child['parent'] for child in children], [node['id']] * 2)
                        self.assertEqual(children[0]['hi'], s.encode(middle))
                        self.assertEqual(children[1]['lo'], s.encode(middle))
                    elif node['status'] == 'retained_tolerance':
                        self.assertLessEqual(s.decode(node['hi']) - s.decode(node['lo']), Q(15, 32))
                if branch is current['branches']['partial']:
                    for node in branch['trace']:
                        if s.decode(node['lo']) <= truth <= s.decode(node['hi']):
                            self.assertIn(node['status'], ('split', 'retained_tolerance', 'unresolved_budget'))
                            self.assertTrue(all(s.decode(pair[0]) <= value <= s.decode(pair[1])
                                                for value, pair in zip(darkness, node['darkness_bounds'])))
                            self.assertTrue(all(s.decode(pair[0]) <= value <= s.decode(pair[1])
                                                for value, pair in zip(biases, node['t_bounds'])))

    def test_negative_background_control_explicitly_incompatible(self):
        for method in s.METHODS:
            current = s.solve_public([Q(0)] * 432, Q(1, 5), Q(80), method)
            self.assertTrue(current['calibration']['excluded'])
            self.assertIsNone(current['outer']['hull'])
            self.assertEqual(current['outer']['retained_intervals'], [])
            for branch in current['branches'].values():
                self.assertEqual(branch['evaluated_boxes'], 0)
                self.assertEqual(len(branch['trace']), 1)
                self.assertEqual(branch['trace'][0]['status'], 'excluded_calibration')
                self.assertEqual(branch['trace'][0]['certificate'], current['calibration']['certificate'])

    def test_white_nonformal_fixture_reaches_finite_budget_in_nine_twelve_dimensions(self):
        for method in s.METHODS:
            s.clear_geometry_cache()
            current = s.solve_public([Q(1)] * 432, Q(1, 5), Q(80), method)
            WHITE_TIMINGS[method] = {'outer_seconds': current['outer_seconds'],
                                     'branch_seconds': current['branch_seconds'],
                                     'evaluated_boxes': [branch['evaluated_boxes']
                                                         for branch in current['branches'].values()]}
            self.assertEqual(current['outer']['hull'], [s.encode(Q(540)), s.encode(Q(660))])
            for branch in current['branches'].values():
                self.assertEqual(branch['evaluated_boxes'], 511)
                self.assertEqual(branch['shared_tests'], 511)
                self.assertEqual(len(branch['trace']), 511)
                self.assertEqual(len(branch['retained_ids']), 256)
                self.assertTrue(all(node['status'] == 'retained_tolerance'
                                    for node in branch['trace'] if node['id'] in branch['retained_ids']))
                self.assertEqual(len(branch['initial_bounds']), 12 if method == 'joint' else 9)


if __name__ == '__main__':
    unittest.main()
