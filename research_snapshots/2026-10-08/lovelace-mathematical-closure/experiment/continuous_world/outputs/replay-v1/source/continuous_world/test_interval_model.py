"""Hand-calculable interval/polygon checks and independent exact point witnesses.

Sample containment is a regression check only. Continuous coverage follows from
interval projection and convex dilation/erosion inclusions, not these samples.
Run from the experiment root: python -B -m continuous_world.test_interval_model
"""

from fractions import Fraction as Q
import unittest

from continuous_world.interval_model import (GRID, Interval, Model, _convex_hull,
                                             _dilate_erode_triangle, _overlap_interval)
from finite_world.mesh_patch import polygon_area


class IntervalArithmeticTests(unittest.TestCase):
    def test_signed_operations(self):
        self.assertEqual((Interval(-2, 3) * Interval(-4, -1)).pair(), (Q(-12), Q(8)))
        self.assertEqual((Interval(-1, 2) / Interval(2, 4)).pair(), (Q(-1, 2), Q(1)))
        self.assertEqual(Interval(-4, -2).reciprocal().pair(), (Q(-1, 2), Q(-1, 4)))
        self.assertEqual((3 - Interval(-2, 4)).pair(), (Q(-1), Q(5)))
        self.assertEqual((Interval(1, 2) + Q(1, 3)).pair(), (Q(4, 3), Q(7, 3)))

    def test_invalid_arithmetic(self):
        for interval in (Interval(-1, 1), Interval(0, 1), Interval(0)):
            with self.assertRaises(ZeroDivisionError):
                interval.reciprocal()
        with self.assertRaises(ValueError):
            Interval(2, 1)
        with self.assertRaises(TypeError):
            Interval(1.0)
        with self.assertRaises(TypeError):
            Interval(True)

    def test_rectangle_overlap_hand_result(self):
        self.assertEqual(_overlap_interval(Interval(Q(-1, 2), 0), Interval(1, Q(3, 2)),
                                          Q(0), Q(1)).pair(), (Q(1), Q(1)))
        self.assertEqual(_overlap_interval(Interval(Q(1, 2), Q(3, 4)), Interval(2, 3),
                                          Q(0), Q(1)).pair(), (Q(1, 4), Q(1, 2)))
        self.assertEqual(_overlap_interval(Interval(2, 3), Interval(4, 5),
                                          Q(0), Q(1)).pair(), (Q(0), Q(0)))


class ConvexBoundsTests(unittest.TestCase):
    def test_hull_order_and_duplicates(self):
        points = ((Q(0), Q(0)), (Q(2), Q(0)), (Q(2), Q(2)), (Q(0), Q(2)),
                  (Q(1), Q(1)), (Q(0), Q(0)), (Q(1), Q(0)))
        self.assertEqual(_convex_hull(points), points[:4])
        self.assertEqual(polygon_area(_convex_hull(reversed(points))), Q(4))

    def test_triangle_dilation_erosion_hand_areas(self):
        triangle = ((Q(0), Q(0)), (Q(2), Q(0)), (Q(0), Q(2)))
        outer, inner = _dilate_erode_triangle(triangle, Q(1, 4))
        # inner constraints x>=1/4, y>=1/4, x+y<=3/2 leave legs of length 1.
        self.assertEqual(polygon_area(inner), Q(1, 2))
        self.assertEqual(polygon_area(outer), Q(17, 4))
        self.assertEqual(_dilate_erode_triangle(tuple(reversed(triangle)), Q(1, 4)), (outer, inner))
        same_outer, same_inner = _dilate_erode_triangle(triangle, Q(0))
        self.assertEqual(polygon_area(same_outer), Q(2))
        self.assertEqual(polygon_area(same_inner), Q(2))
        self.assertEqual(polygon_area(_dilate_erode_triangle(triangle, Q(1))[1]), Q(0))

    def test_degenerate_nominal_safe(self):
        line = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(2), Q(0)))
        outer, inner = _dilate_erode_triangle(line, Q(1, 4))
        self.assertEqual(inner, ())
        self.assertEqual(polygon_area(outer), Q(5, 4))


class ContinuousModelTests(unittest.TestCase):
    def test_declared_grid_and_shape(self):
        self.assertEqual((GRID.width, GRID.height), (8, 6))
        self.assertEqual((GRID.u_min, GRID.u_max, GRID.v_min, GRID.v_max),
                         (Q(-1, 5), Q(1, 5), Q(-2, 25), Q(2, 25)))
        for kind in ("lovelace_triangle", "rectangle"):
            model = Model(kind)
            self.assertEqual(model.observation_size, 432)
            prediction = model.point_prediction(Q(600))
            self.assertEqual(len(prediction), 432)
            self.assertTrue(all(isinstance(c, Q) and 0 <= c <= 1 for c in prediction))

    def test_point_boxes_are_exact_at_zero_camera_radius(self):
        for kind in ("lovelace_triangle", "rectangle"):
            model = Model(kind)
            for depth in (Q(540), Q(600), Q(660)):
                exact = model.point_prediction(depth)
                self.assertEqual(model.enclosure(depth, depth), tuple((x, x) for x in exact))

    def test_domain_camera_gauge_and_float_rejection(self):
        for kind in ("lovelace_triangle", "rectangle"):
            model = Model(kind, Q(1, 5))
            for depths in ((539, 600), (600, 661), (610, 600)):
                with self.assertRaises(ValueError):
                    model.enclosure(*depths)
            for offsets in ((0, Q(1, 100), 0), (Q(1, 4), 0, 0), (0, 0)):
                with self.assertRaises(ValueError):
                    model.point_prediction(Q(600), offsets)
            with self.assertRaises(TypeError):
                model.point_prediction(600.0)
        with self.assertRaises(TypeError):
            Model("rectangle", 0.2)
        with self.assertRaises(ValueError):
            Model("rectangle", -1)

    def test_outside_domain_baseline_points_remain_separate(self):
        for kind in ("lovelace_triangle", "rectangle"):
            model = Model(kind, Q(1))
            self.assertEqual(len(model.point_prediction(Q(500), (Q(1, 7), 0, Q(-2, 9)))), 432)
            with self.assertRaises(ValueError):
                model.enclosure(Q(500), Q(600))
            with self.assertRaises(ValueError):
                model.point_prediction(Q(-1))

    def test_normalized_queries_share_cached_enclosures(self):
        model = Model("lovelace_triangle", Q(1, 5))
        first = model.enclosure(590, 610)
        self.assertIs(model.enclosure(Q(590), "610"), first)
        prediction = model.point_prediction(600)
        self.assertIs(model.point_prediction(Q(600), [0, 0, 0]), prediction)

    def test_independent_interior_and_camera_corner_witnesses(self):
        checked = 0
        for kind in ("lovelace_triangle", "rectangle"):
            for eta in (Q(0), Q(1, 5), Q(1)):
                model = Model(kind, eta)
                errors = tuple(set(((-eta, 0, -eta), (-eta, 0, eta),
                                    (eta, 0, -eta), (eta, 0, eta),
                                    (eta / 3, 0, -eta / 5))))
                for low, high in ((Q(540), Q(660)), (Q(596), Q(604)), (Q(600), Q(600))):
                    bounds = model.enclosure(low, high)
                    self.assertEqual(len(bounds), 432)
                    self.assertTrue(all(isinstance(a, Q) and isinstance(b, Q) and 0 <= a <= b <= 1
                                        for a, b in bounds))
                    # Includes endpoints and a non-midpoint rational interior witness.
                    for depth in set((low, high, low + (high - low) * Q(7, 19))):
                        for offsets in errors:
                            prediction = model.point_prediction(depth, offsets)
                            self.assertTrue(all(a <= x <= b for x, (a, b) in zip(prediction, bounds)),
                                            (kind, str(eta), str(low), str(high), str(depth), offsets))
                            checked += 1
        self.assertGreater(checked, 100)

    def test_auditable_polygon_metadata(self):
        model = Model("lovelace_triangle", Q(1, 5))
        details = model.enclosure_details(Q(598), Q(602))
        self.assertIn("erosion", details["method"])
        self.assertEqual(len(details["triangle_cameras"]), 3)
        for camera in details["triangle_cameras"]:
            self.assertGreaterEqual(Q(camera["rho_linf"]), 0)
            self.assertTrue(camera["outer_polygon"])
            self.assertTrue(camera["inner_polygon"])
            self.assertEqual(len(camera["projected_vertex_intervals"]), 3)
        self.assertEqual(details["triangle_cameras"][1]["camera_cx_box_mm"], ["0", "0"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
