"""Finite nonformal public fixtures; no population generation or label access."""
from fractions import Fraction as Q
import inspect
import sys
import unittest

sys.dont_write_bytecode = True
import ablation_solver as s


def fixture(baseline, clipped=True):
    """Hand-fixed exact point with bounded error, independent of formal seeds."""
    image = s.geometry.point_rgb('partial', Q(617), (Q(-1, 10), Q(0), Q(1, 10)),
                                 (Q(0), Q(1, 3), Q(4, 5)), baseline)
    raw = tuple(value + (-s.BETA, Q(0), s.BETA)[i // 144]
                + (-s.EPSILON, Q(0), s.EPSILON)[i % 3]
                for i, value in enumerate(image))
    return tuple(min(Q(1), max(Q(0), value)) for value in raw) if clipped else raw


def without_timing(result):
    return {key: value for key, value in result.items()
            if key not in ('branch_seconds', 'outer_seconds', 'preparation_seconds',
                           'inference_mask')}


def endpoint_image():
    choices = ((Q(0),) * 3, (Q(1),) * 3, (Q(1, 2),) * 3,
               (Q(0), Q(1), Q(1, 2)))
    return tuple(channel for pixel in range(144) for channel in choices[pixel % 4])


def independent_metadata(image, clipped):
    result = []
    for pixel in range(144):
        channels = image[3 * pixel:3 * pixel + 3]
        for side, exists in (('upper', not clipped or any(y != 0 for y in channels)),
                             ('lower', not clipped or any(y != 1 for y in channels))):
            if exists:
                result.append({'row_id': len(result), 'pixel': pixel, 'view': pixel // 48,
                               'row': (pixel % 48) // 8, 'column': pixel % 8, 'side': side})
    return tuple(result)


class AblationSolverTests(unittest.TestCase):
    def test_registered_interface_and_constants(self):
        self.assertEqual(tuple(inspect.signature(s.solve_public).parameters),
                         ('observation', 'radius', 'baseline', 'method', 'mask_name', 'clipped'))
        self.assertEqual((s.ROUNDS, s.MAX_BOXES, s.TOLERANCE), (8, 511, Q(15, 32)))
        self.assertEqual((s.PIXEL_COUNT, s.RGB_COUNT), (144, 432))
        self.assertEqual(s.MASK_NAMES, ('drop_F', 'matched_drop'))

    def test_exact_fixed_mirror_masks(self):
        expected = {'drop_F': (7, 15, 23, 31, 39, 47, 96, 104, 112, 120, 128, 136),
                    'matched_drop': (6, 14, 22, 30, 38, 46, 97, 105, 113, 121, 129, 137)}
        for mask, pixels in expected.items():
            self.assertEqual(s.dropped_pixels(mask), pixels)
            self.assertEqual(len(set(pixels)), 12)
            self.assertEqual([sum(p // 48 == view for p in pixels) for view in range(3)],
                             [6, 0, 6])
        self.assertEqual(s.dropped_pixels('full'), ())
        self.assertFalse(set(expected['drop_F']) & set(expected['matched_drop']))

    def test_endpoint_metadata_is_original_row_order(self):
        image = endpoint_image()
        for clipped in (True, False):
            metadata = s.row_metadata(list(map(s.encode, image)), clipped)
            self.assertEqual(metadata, independent_metadata(image, clipped))
            self.assertEqual([record['row_id'] for record in metadata], list(range(len(metadata))))
            self.assertEqual(metadata[-1]['pixel'], 143)
            self.assertEqual(metadata[-1]['view'], 2)
            self.assertEqual(metadata[-1]['column'], 7)
        self.assertEqual([record['side'] for record in s.row_metadata(image) if record['pixel'] == 0], ['lower'])
        self.assertEqual([record['side'] for record in s.row_metadata(image) if record['pixel'] == 1], ['upper'])
        self.assertNotEqual(next(record['row_id'] for record in s.row_metadata(image)
                                 if record['pixel'] == 48), 2 * 48)

    def test_filter_preserves_original_pixels_sides_and_view_coefficients(self):
        # Fixed distinct areas make an accidental image/geometry deletion visible.
        areas = tuple((Q(pixel, 200), Q(pixel, 200), Q(0), Q(0), Q(0), Q(0))
                      for pixel in range(144))
        image = endpoint_image()
        ts = ((Q(0), Q(1, 50)),) * 3
        for clipped in (True, False):
            metadata = independent_metadata(image, clipped)
            for method in ('calibrated', 'joint'):
                rows = s.frozen.necessary_rows(image, areas, method, clipped=clipped,
                                               calibration_intervals=ts)
                self.assertEqual(len(rows), len(metadata))
                for mask in s.MASK_NAMES:
                    dropped = set(s.dropped_pixels(mask))
                    kept = tuple(record['row_id'] for record in metadata if record['pixel'] not in dropped)
                    filtered = s._filter_rows(rows, kept, len(metadata))
                    self.assertEqual(filtered, tuple(rows[index] for index in kept))
                    for local_index, original_id in enumerate(kept):
                        record = metadata[original_id]
                        kind, coefficients, _bound = filtered[local_index]
                        self.assertEqual(kind, record['side'])
                        self.assertEqual(coefficients[0], Q(record['pixel'], 200))
                        if method == 'joint':
                            self.assertEqual(coefficients[3:], tuple(Q(int(view == record['view']))
                                                                    for view in range(3)))
                    self.assertEqual(metadata[kept[-1]]['pixel'], 143)
        with self.assertRaises(ValueError):
            s._filter_rows(rows[:-1], (0,), len(metadata))

    def test_encoded_rational_boundary_and_rejection(self):
        self.assertEqual(s._exact({'numerator': 37, 'denominator': 101}, 'fixture'), Q(37, 101))
        self.assertEqual(s.row_metadata(list(map(s.encode, endpoint_image()))),
                         s.row_metadata(endpoint_image()))
        invalid = ({'numerator': True, 'denominator': 1},
                   {'numerator': 1, 'denominator': False},
                   {'numerator': 0.5, 'denominator': 2},
                   {'numerator': 1, 'denominator': 0},
                   {'numerator': 1, 'denominator': -2},
                   {'numerator': 1}, {'numerator': 1, 'denominator': 2, 'extra': 0})
        for value in invalid:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                s.row_metadata([value] + [Q(1)] * 431)
        for value in (0.5, True):
            with self.assertRaises(TypeError):
                s.row_metadata([value] + [Q(1)] * 431)

    def test_malformed_public_inputs_fail_before_solving(self):
        image = [Q(1)] * 432
        for size in (0, 3, 429, 431, 433, 435):
            with self.assertRaises(ValueError):
                s.solve_public(image[:size] if size <= 432 else image + [Q(1)] * (size - 432),
                               Q(1, 5), Q(80), 'joint', 'drop_F')
        for arguments, error in (((Q(-1), Q(80), 'joint', 'drop_F', True), ValueError),
                                 ((Q(1, 5), Q(75), 'joint', 'drop_F', True), ValueError),
                                 ((Q(1, 5), Q(80), 'cal', 'drop_F', True), ValueError),
                                 ((Q(1, 5), Q(80), 'joint', 'adaptive', True), ValueError),
                                 ((Q(1, 5), Q(80), 'joint', 'drop_F', 1), TypeError),
                                 ((0.2, Q(80), 'joint', 'drop_F', True), TypeError),
                                 ((Q(1, 5), 80.0, 'joint', 'drop_F', True), TypeError)):
            with self.assertRaises(error):
                s.solve_public(image, *arguments)
        with self.assertRaises(ValueError):
            s.row_metadata([Q(101, 100)] + image[1:], clipped=True)
        self.assertEqual(len(s.row_metadata([Q(101, 100)] + image[1:], clipped=False)), 288)

    def test_full_identity_against_frozen_engine_except_wall_times(self):
        # One registered camera for each method; both use hand-fixed point inputs.
        for baseline, method in ((Q(70), 'calibrated'), (Q(80), 'joint')):
            image = fixture(baseline)
            s.clear_geometry_cache()
            original = s.baseline_engine.solve_public(image, Q(1, 5), baseline, method)
            s.clear_geometry_cache()
            current = s.solve_public(list(map(s.encode, image)), {'numerator': 1, 'denominator': 5},
                                     s.encode(baseline), method, 'full')
            self.assertEqual(without_timing(current), without_timing(original))
            self.assertEqual(current['inference_mask'], {'name': 'full', 'dropped_pixels': [],
                              'dropped_rows': [], 'kept_row_count': len(s.row_metadata(image))})

    def test_masks_keep_full_calibration_and_latent_point(self):
        baseline, truth = Q(80), Q(617)
        image = fixture(baseline)
        background = s.baseline_engine.background_mask(Q(1, 5), baseline)
        full_calibration = s.frozen.calibration_bounds(image, background, clipped=True)
        metadata = independent_metadata(image, True)
        for mask in s.MASK_NAMES:
            dropped = set(s.dropped_pixels(mask))
            removed = [record for record in metadata if record['pixel'] in dropped]
            for method in ('calibrated', 'joint'):
                s.clear_geometry_cache()
                result = s.solve_public(image, Q(1, 5), baseline, method, mask)
                self.assertEqual(result['calibration'], full_calibration)
                self.assertEqual(result['background_mask'], list(background))
                self.assertEqual(result['calibration_intervals'], full_calibration['intervals'])
                self.assertEqual(result['inference_mask'], {'name': mask, 'dropped_pixels': sorted(dropped),
                                  'dropped_rows': removed, 'kept_row_count': len(metadata) - len(removed)})
                self.assertTrue(any(s.decode(lo) <= truth <= s.decode(hi)
                                    for lo, hi in result['outer']['retained_intervals']))
                for branch in result['branches'].values():
                    self.assertEqual(branch['scope'], 'necessary_shared_bias_halfspaces_with_historical_continuous_geometry_bounds')
                    self.assertLessEqual(branch['evaluated_boxes'], 511)
                    self.assertEqual(branch['shared_tests'], branch['evaluated_boxes'])

    def test_weak_fixture_reaches_finite_box_limit(self):
        # A white point uses no generating seed and has no depth information.
        image = s.geometry.point_rgb('none', Q(611), (Q(0),) * 3, (Q(1),) * 3, Q(80))
        s.clear_geometry_cache()
        result = s.solve_public(image, Q(1, 5), Q(80), 'calibrated', 'drop_F')
        self.assertEqual(result['outer']['hull'], [s.encode(Q(540)), s.encode(Q(660))])
        self.assertEqual([branch['evaluated_boxes'] for branch in result['branches'].values()],
                         [511, 511, 511])
        self.assertEqual(result['inference_mask']['kept_row_count'], 132)
        self.assertTrue(all(branch['evaluated_boxes'] <= branch['max_boxes']
                            for branch in result['branches'].values()))


if __name__ == '__main__':
    unittest.main()
