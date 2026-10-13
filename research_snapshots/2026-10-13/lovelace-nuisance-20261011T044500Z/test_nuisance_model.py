"""Exact regression tests for the declared geometry/material enclosure contract."""
from fractions import Fraction as Q
import unittest

from nuisance_model import (ColoredModel, BRANCHES, PUBLIC_COLORS, LEFT_RECT, RIGHT_RECT,
                            FRONT_RECT, GRID, _project_rect, _geometry_bounds,
                            rectangle_area)
from continuous_world.interval_solver import solve_outer, decode
from layered_model import rectangle_intersection


def singleton(colors):
    return tuple((value, value) for value in colors)


def gray_at(image, camera, row, column):
    return image[((camera * GRID.height + row) * GRID.width + column) * 3]


class ColoredModelTests(unittest.TestCase):
    def test_two_patches_partition_front_without_area_overlap(self):
        for depth in (Q(540), Q(6199, 11), Q(660)):
            for camera in (Q(-351, 5), Q(0), Q(351, 5)):
                left, right, front = (_project_rect(rect, depth, camera)
                                      for rect in (LEFT_RECT, RIGHT_RECT, FRONT_RECT))
                self.assertIsNone(rectangle_intersection(left, right))
                self.assertEqual(rectangle_area(left) + rectangle_area(right), rectangle_area(front))
                for row in range(GRID.height):
                    for column in range(GRID.width):
                        pixel = GRID.pixel_bounds(column, row)
                        self.assertEqual(rectangle_area(left, pixel) + rectangle_area(right, pixel),
                                         rectangle_area(front, pixel))

    def test_hand_computed_split_pixel_integrals(self):
        colors = (Q(1, 5), Q(3, 5), Q(2, 5))
        image = ColoredModel('none', 0, singleton(colors)).point_prediction(600)
        # Each reference patch occupies 5/8 of its adjacent middle pixel.
        self.assertEqual(gray_at(image, 1, 2, 3), Q(1, 2))
        self.assertEqual(gray_at(image, 1, 2, 4), Q(3, 4))
        self.assertEqual(gray_at(image, 1, 0, 0), 1)
        for index in range(0, len(image), 3):
            self.assertEqual(image[index:index + 3], (image[index],) * 3)

    def test_rear_minus_whole_front_avoids_double_counting(self):
        colors = (Q(1, 5), Q(3, 5), Q(2, 5))
        image = ColoredModel('partial', 0, singleton(colors)).point_prediction(600)
        # Front-right area=5/8, visible rear area=85/288, so gray=55/96.
        self.assertEqual(gray_at(image, 1, 2, 4), Q(55, 96))
        self.assertTrue(all(0 <= value <= 1 for value in image))

    def test_white_materials_remove_all_shapes(self):
        for branch in BRANCHES:
            image = ColoredModel(branch, Q(1, 5)).point_prediction(600, colors=(1, 1, 1))
            self.assertEqual(image, (Q(1),) * 432)

    def test_none_branch_does_not_depend_on_rear_color(self):
        model = ColoredModel('none')
        first = model.point_prediction(600, colors=(Q(1, 5), Q(3, 5), 0))
        second = model.point_prediction(600, colors=(Q(1, 5), Q(3, 5), 1))
        self.assertEqual(first, second)
        narrow = ((Q(1, 5), Q(2, 5)), (Q(1, 2), Q(3, 5)))
        self.assertEqual(ColoredModel('none', 0, (*narrow, (0, 0))).enclosure(590, 610),
                         ColoredModel('none', 0, (*narrow, (1, 1))).enclosure(590, 610))

    def test_singleton_geometry_and_colors_give_exact_singleton_bounds(self):
        colors = (Q(1, 5), Q(3, 5), Q(2, 5))
        for branch in BRANCHES:
            model = ColoredModel(branch, 0, singleton(colors))
            depth = Q(6199, 11)
            image = model.point_prediction(depth)
            self.assertEqual(model.enclosure(depth, depth), tuple((value, value) for value in image))

    def test_singleton_geometry_color_extrema_are_exact_without_equal_constraint(self):
        intervals = ((Q(1, 10), Q(1, 5)), (Q(3, 5), Q(7, 10)), (Q(3, 10), Q(2, 5)))
        for branch in BRANCHES:
            model = ColoredModel(branch, 0, intervals)
            lower_image = model.point_prediction(600, colors=tuple(low for low, _high in intervals))
            upper_image = model.point_prediction(600, colors=tuple(high for _low, high in intervals))
            self.assertEqual(model.enclosure(600, 600), tuple(zip(lower_image, upper_image)))

    def test_continuous_boxes_cover_sampled_depth_pose_and_independent_colors(self):
        intervals = ((Q(1, 10), Q(3, 10)), (Q(1, 2), Q(7, 10)), (Q(1, 5), Q(2, 5)))
        point_colors = [tuple(pair[(mask >> index) & 1] for index, pair in enumerate(intervals))
                        for mask in range(8)]
        point_colors.append(tuple((low + high) / 2 for low, high in intervals))
        offsets = ((Q(-1, 5), 0, Q(1, 5)), (Q(1, 5), 0, Q(-1, 5)),
                   (Q(1, 13), 0, Q(-2, 17)))
        for branch in BRANCHES:
            model = ColoredModel(branch, Q(1, 5), intervals)
            for low, high in ((Q(540), Q(660)), (Q(579), Q(1201, 2))):
                bounds = model.enclosure(low, high)
                for depth in (low, (2 * low + high) / 3, high):
                    for errors in offsets:
                        for colors in point_colors:
                            image = model.point_prediction(depth, errors, colors)
                            for value, (minimum, maximum) in zip(image, bounds):
                                self.assertLessEqual(minimum, value)
                                self.assertGreaterEqual(maximum, value)

    def test_full_same_material_is_exactly_depth_independent(self):
        colors = (Q(1, 5),) * 3
        errors = (Q(1, 13), Q(0), Q(-2, 17))
        model = ColoredModel('full', Q(1, 5), singleton(colors))
        first = model.point_prediction(540, errors)
        for depth in (Q(6199, 11), Q(7853, 13), Q(8999, 14), Q(660)):
            self.assertEqual(model.point_prediction(depth, errors), first)

    def test_independent_color_ranges_admit_unequal_materials(self):
        model = ColoredModel('full', colors=((0, 1),) * 3)
        equal = model.point_prediction(600, colors=(Q(1, 5),) * 3)
        unequal = model.point_prediction(600, colors=(Q(1, 5), Q(3, 5), Q(2, 5)))
        self.assertNotEqual(equal, unequal)
        bounds = model.enclosure(600, 600)
        for image in (equal, unequal):
            self.assertTrue(all(low <= value <= high for value, (low, high) in zip(image, bounds)))

    def test_fresh_outer_contains_truth_under_valid_unknown_material_contract(self):
        depth, errors = Q(6199, 11), (Q(1, 13), 0, Q(-2, 17))
        for branch in BRANCHES:
            model = ColoredModel(branch)
            observation = model.point_prediction(depth, errors, (Q(1, 5), Q(3, 5), Q(2, 5)))
            outer = solve_outer(observation, model.enclosure, tolerance=Q(15, 32), max_boxes=31)
            self.assertIsNotNone(outer['hull'])
            low, high = map(decode, outer['hull'])
            self.assertLessEqual(low, depth)
            self.assertGreaterEqual(high, depth)
            retained = set(outer['retained_ids'])
            self.assertTrue(any(entry['id'] in retained and decode(entry['lo']) <= depth <= decode(entry['hi'])
                                for entry in outer['trace']))

    def test_equal_full_compatible_color_ranges_preserve_entire_prior(self):
        public = ColoredModel('full')
        colors = (Q(1, 5),) * 3
        observation = public.point_prediction(Q(6199, 11), (Q(1, 10), 0, Q(-1, 5)), colors)
        # Independent intervals each contain the equal-world witness, without equality constraints.
        intervals = ((Q(9, 50), Q(11, 50)), (Q(19, 100), Q(21, 100)), (Q(1, 5), Q(11, 50)))
        model = ColoredModel('full', Q(1, 5), intervals)
        outer = solve_outer(observation, model.enclosure, tolerance=Q(15, 32), max_boxes=31)
        self.assertEqual(tuple(map(decode, outer['hull'])), (Q(540), Q(660)))
        self.assertFalse(any(entry['status'] == 'excluded' for entry in outer['trace']))

    def test_geometry_cache_is_shared_across_material_instances(self):
        depth = Q(620001, 1000)
        first = ColoredModel('partial', 0, PUBLIC_COLORS)
        second = ColoredModel('partial', 0, ((Q(1, 5), Q(2, 5)),) * 3)
        first.enclosure(depth, depth)
        before = _geometry_bounds.cache_info()
        second.enclosure(depth, depth)
        after = _geometry_bounds.cache_info()
        self.assertEqual(after.misses, before.misses)
        self.assertGreater(after.hits, before.hits)

    def test_point_values_are_explicit_and_within_public_intervals(self):
        with self.assertRaises(ValueError):
            ColoredModel('none').point_prediction(600)
        model = ColoredModel('partial', colors=((Q(1, 5), Q(2, 5)),) * 3)
        for values in ((0, Q(1, 3), Q(1, 3)), (Q(1, 3), Q(1, 3)), (Q(1, 3),) * 4):
            with self.assertRaises(ValueError):
                model.point_prediction(600, colors=values)

    def test_invalid_exactness_ranges_and_gauge_are_rejected(self):
        for branch in ('unknown', '', None):
            with self.assertRaises(ValueError):
                ColoredModel(branch)
        for radius in (Q(-1, 5), -1):
            with self.assertRaises(ValueError):
                ColoredModel('none', radius)
        for colors in (((-1, 1),) * 3, ((0, 2),) * 3, ((1, 0),) * 3, ((0, 1),) * 2):
            with self.assertRaises(ValueError):
                ColoredModel('partial', colors=colors)
        model = ColoredModel('partial')
        invalid_type_calls = (lambda: ColoredModel('none', 0.2),
                              lambda: ColoredModel('none', True),
                              lambda: ColoredModel('none', colors=((0.0, 1),) * 3),
                              lambda: model.enclosure(540.0, 660),
                              lambda: model.enclosure(True, 660),
                              lambda: model.point_prediction(600.0, colors=(0, 0, 0)),
                              lambda: model.point_prediction(600, (0.0, 0, 0), (0, 0, 0)),
                              lambda: model.point_prediction(600, colors=(True, 0, 0)))
        for call in invalid_type_calls:
            with self.assertRaises(TypeError):
                call()
        for low, high in ((539, 600), (600, 661), (610, 590), (0, 600)):
            with self.assertRaises(ValueError):
                model.enclosure(low, high)
        for errors in ((0, Q(1, 100), 0), (Q(21, 100), 0, 0), (0, 0)):
            with self.assertRaises(ValueError):
                model.point_prediction(600, errors, (0, 0, 0))

    def test_public_metadata_discloses_conservative_material_contract(self):
        model = ColoredModel('partial')
        info = model.to_dict()
        self.assertEqual(info['color_labels'], ['front-left', 'front-right', 'rear'])
        self.assertEqual(info['color_intervals'], [['0', '1']] * 3)
        self.assertIn('conservative', info['enclosure_scope'])
        self.assertEqual(model.observation_size, 432)
        self.assertEqual(model.depth_range, (Q(540), Q(660)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
