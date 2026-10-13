"""Focused exact geometry checks; rational samples are regression witnesses only."""

from fractions import Fraction as Q
from random import Random
import unittest

from layered_model import (
    FRONT_RECT, GRID, HIDDEN_REAR_RECT, KINDS, NOMINAL_CX, REAR_DEPTH,
    REAR_RECT, Interval, LayeredModel, _front_first_gray, _inner_outer,
    _project_rect, _project_rect_boxes, rectangle_area,
    rectangle_difference_area, rectangle_intersection, rectangle_union_area,
)
from finite_world.renderer import Camera, Material, Surface, World, evaluation_target_z, render


class RectangleAreaTests(unittest.TestCase):
    def test_hand_calculable_clipping_overlap_union_and_difference(self):
        pixel = (Q(0), Q(2), Q(0), Q(2))
        front = (Q(-1), Q(1), Q(0), Q(2))
        rear = (Q(0), Q(2), Q(1), Q(3))
        self.assertEqual(rectangle_area(front, pixel), Q(2))
        self.assertEqual(rectangle_area(rear, pixel), Q(2))
        self.assertEqual(rectangle_intersection(front, rear), (Q(0), Q(1), Q(1), Q(2)))
        self.assertEqual(rectangle_area(rectangle_intersection(front, rear), pixel), Q(1))
        self.assertEqual(rectangle_union_area(front, rear, pixel), Q(3))
        self.assertEqual(rectangle_difference_area(rear, front, pixel), Q(1))
        self.assertEqual(_front_first_gray(front, rear, Q(1, 3), pixel), Q(1, 3))
        self.assertEqual(_front_first_gray(front, rear, Q(0), pixel), Q(1, 4))

    def test_empty_touching_and_degenerate_sets_have_zero_area(self):
        pixel = (Q(0), Q(1), Q(0), Q(1))
        touching = (Q(1), Q(2), Q(0), Q(1))
        self.assertIsNone(rectangle_intersection(pixel, touching))
        self.assertEqual(rectangle_area(touching, pixel), Q(0))
        self.assertEqual(rectangle_union_area(None, pixel, pixel), Q(1))
        self.assertEqual(rectangle_difference_area(None, pixel, pixel), Q(0))
        inner, outer = _inner_outer((Interval(0, 2), Interval(1, 3), Interval(0), Interval(1)))
        self.assertIsNone(inner)
        self.assertEqual(outer, (Q(0), Q(3), Q(0), Q(1)))

    def test_opaque_front_owns_overlap_without_double_counting(self):
        pixel = (Q(0), Q(2), Q(0), Q(2))
        front = (Q(0), Q(1), Q(0), Q(2))
        self.assertEqual(_front_first_gray(front, pixel, Q(1, 3), pixel), Q(1, 6))
        self.assertEqual(_front_first_gray(pixel, pixel, Q(1, 3), pixel), Q(0))


class LayeredModelTests(unittest.TestCase):
    def test_layout_exact_rationals_and_reference_target(self):
        for kind in KINDS:
            model = LayeredModel(kind)
            prediction = model.point_prediction(Q(600))
            self.assertEqual(model.observation_size, 432)
            self.assertEqual(len(prediction), 432)
            self.assertTrue(all(isinstance(x, Q) and 0 <= x <= 1 for x in prediction))
            world = self._world(model, Q(600))
            self.assertEqual(evaluation_target_z(world), Q(600))

    @staticmethod
    def _world(model, depth):
        black = Material("solid", (Q(0),) * 3)
        gray = Material("solid", (model.rear_color,) * 3)
        return World("independent_exact_compositor", (
            Surface(depth, *FRONT_RECT, black, "foreground"),
            Surface(REAR_DEPTH, *model.rear_rect, gray, "rear")))

    def test_point_intervals_exact_when_camera_radius_is_zero(self):
        for kind in KINDS:
            model = LayeredModel(kind)
            for depth in (Q(540), Q(600), Q(11541, 19), Q(660)):
                prediction = model.point_prediction(depth)
                self.assertEqual(model.enclosure(depth, depth), tuple((x, x) for x in prediction))

    def test_budget_gauge_domain_type_and_projection_pole_rejection(self):
        for kind in KINDS:
            model = LayeredModel(kind, Q(1, 5))
            for offsets in ((0, Q(1, 100), 0), (Q(1, 4), 0, 0), (0, 0), (0, 0, -1)):
                with self.assertRaises(ValueError):
                    model.point_prediction(600, offsets)
            for low, high in ((539, 600), (600, 661), (601, 600), (0, 600)):
                with self.assertRaises(ValueError):
                    model.enclosure(low, high)
            for depth in (0, -1):
                with self.assertRaises(ValueError):
                    model.point_prediction(depth)
            with self.assertRaises(TypeError):
                model.point_prediction(600.0)
            with self.assertRaises(TypeError):
                model.point_prediction(True)
            with self.assertRaises(TypeError):
                model.point_prediction(600, (0.0, 0, 0))
            with self.assertRaises(ValueError):
                LayeredModel(kind, -1)
            with self.assertRaises(TypeError):
                LayeredModel(kind, 0.2)
        with self.assertRaises(ValueError):
            LayeredModel("unknown")
        with self.assertRaises(ValueError):
            LayeredModel("layered_hidden", Q(150001, 1000))
        with self.assertRaises(ValueError):
            _project_rect_boxes(FRONT_RECT, Interval(0, 1), Interval(0))

    def test_random_rational_witnesses_in_continuous_bounds(self):
        random = Random(90103)
        for kind in KINDS:
            for radius in (Q(0), Q(1, 5), Q(1)):
                model = LayeredModel(kind, radius)
                for low, high in ((Q(540), Q(660)), (Q(596), Q(604)), (Q(600), Q(600))):
                    bounds = model.enclosure(low, high)
                    self.assertTrue(all(isinstance(a, Q) and isinstance(b, Q) and 0 <= a <= b <= 1
                                        for a, b in bounds))
                    depths = (low, high) + tuple(low + (high - low) * Q(random.randrange(1, 97), 97)
                                               for _ in range(8))
                    for depth in depths:
                        offsets = (radius * Q(random.randrange(-101, 102), 101), Q(0),
                                   radius * Q(random.randrange(-103, 104), 103))
                        point = model.point_prediction(depth, offsets)
                        self.assertTrue(all(a <= value <= b for value, (a, b) in zip(point, bounds)),
                                        (kind, radius, low, high, depth, offsets))

    def test_exact_pixel_integral_against_independent_cell_renderer_and_depth_order(self):
        for kind in KINDS:
            model = LayeredModel(kind, Q(1))
            offsets = (Q(2, 7), Q(0), Q(-3, 11))
            cameras = tuple(Camera(c + e) for c, e in zip(NOMINAL_CX, offsets))
            # The independent renderer partitions pixels at projected layer
            # edges and ray-tests each cell. z=800 also checks reversed order.
            for depth in (Q(500), Q(600), Q(660), Q(800)):
                self.assertEqual(model.point_prediction(depth, offsets),
                                 render(self._world(model, depth), cameras, GRID))
            self.assertEqual(len(model.point_prediction(Q(720), offsets)), 432)

    def test_hidden_depth_independence_for_shared_camera_pose(self):
        for radius in (Q(0), Q(1, 5), Q(1), Q(150)):
            model = LayeredModel("layered_hidden", radius)
            for offsets in ((-radius, 0, radius), (radius, 0, -radius), (radius / 7, 0, -radius / 3)):
                first = model.point_prediction(540, offsets)
                self.assertEqual(model.point_prediction(Q(11541, 19), offsets), first)
                self.assertEqual(model.point_prediction(660, offsets), first)
                for c, e in zip(NOMINAL_CX, offsets):
                    rear = _project_rect(HIDDEN_REAR_RECT, REAR_DEPTH, c + e)
                    for z in (Q(540), Q(660)):
                        front = _project_rect(FRONT_RECT, z, c + e)
                        self.assertEqual(rectangle_intersection(front, rear), front)
            whole = model.enclosure(540, 660)
            self.assertEqual(model.enclosure(600, 600), whole)
            self.assertEqual(model.enclosure(596, 604), whole)
        # Same-color alone is not an unconditional depth ambiguity: finite
        # partially overlapping surfaces leave informative silhouettes.
        same = LayeredModel("layered_same")
        self.assertNotEqual(same.point_prediction(540), same.point_prediction(660))

    def test_normalized_queries_cache_and_audit_metadata(self):
        model = LayeredModel("layered_contrast", Q(1, 5))
        box = model.enclosure(596, 604)
        self.assertIs(model.enclosure(Q(596), "604"), box)
        self.assertIs(model.point_prediction(600), model.point_prediction("600", [0, 0, 0]))
        details = model.enclosure_details(596, 604)
        self.assertEqual(details["camera_rectangles"][1]["camera_cx_box_mm"], ["0", "0"])
        self.assertEqual(model.to_dict()["rear_depth_mm"], "720")


if __name__ == "__main__":
    unittest.main(verbosity=2)
