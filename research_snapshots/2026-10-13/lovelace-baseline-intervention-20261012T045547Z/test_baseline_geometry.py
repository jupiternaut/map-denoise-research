"""Nonconfirmation identity, continuous inclusion and acquisition invariants."""
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import baseline_geometry as geometry
import baseline_solver as solver
from layered_model import _project_rect

HYBRID_ROOT = ROOT.parent / 'lovelace-hybrid-20261010T214217Z'
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))
from hybrid import certificate_hybrid


def fixture(baseline):
    """No RNG or confirmation labels: fixed 600 mm synthetic public fixture."""
    image = geometry.point_rgb('partial', Q(600), (Q(-1, 10), Q(0), Q(1, 10)),
                               (Q(1, 4), Q(1, 2), Q(3, 4)), baseline)
    weights = (Q(-1), Q(0), Q(1))
    biases = (-solver.BETA, Q(0), solver.BETA)
    return tuple(min(Q(1), max(Q(0), value + solver.EPSILON * weights[i % 3]
                    + biases[i // (3 * geometry.PIXELS_PER_VIEW)])) for i, value in enumerate(image))


class BaselineGeometryTests(unittest.TestCase):
    def test_b70_geometry_identity(self):
        for branch, radius, (lo, hi) in product(geometry.BRANCHES, (Q(0), Q(1, 5)),
                ((Q(540), Q(660)), (Q(599), Q(601)), (Q(645), Q(645)))):
            self.assertEqual(geometry.geometry_bounds(branch, radius, lo, hi, Q(70)),
                             geometry.historical._geometry_bounds(branch, radius, lo, hi))

    def test_b70_point_identity(self):
        colors = (Q(1, 4), Q(1, 2), Q(3, 4))
        errors = (Q(-1, 5), Q(0), Q(1, 10))
        for branch, depth in product(geometry.BRANCHES, (Q(540), Q(600), Q(660))):
            model = geometry.historical.ColoredModel(branch, camera_radius=Q(1, 5))
            self.assertEqual(geometry.point_rgb(branch, depth, errors, colors, Q(70)),
                             model.point_prediction(depth, errors, colors=colors))

    def test_b80_continuous_area_inclusion(self):
        radius = Q(1, 5)
        for branch, (lo, hi) in product(geometry.BRANCHES,
                ((Q(590), Q(615)), (Q(640), Q(648)))):
            bounds = geometry.geometry_bounds(branch, radius, lo, hi, Q(80))
            for depth, left_error, right_error in product((lo, (lo + hi) / 2, hi),
                    (-radius, Q(0), radius), (-radius, Q(0), radius)):
                areas = geometry._point_areas(branch, depth,
                    (left_error, Q(0), right_error), Q(80))
                for bound, point in zip(bounds, areas):
                    self.assertTrue(all(a <= p <= b for a, b, p in zip(bound[::2], bound[1::2], point)))

    def test_equal_full_negative_all_depths(self):
        # The worst horizontal side bound occurs at minimum z and maximum |c|.
        # Clearing denominators proves containment for every z in [540,660]
        # and every registered side pose, beyond the finite point checks below.
        max_camera = Q(80) + Q(1, 5)
        self.assertGreaterEqual(geometry.DEPTH_MIN * (100 + max_camera),
                                geometry.REAR_DEPTH * (20 + max_camera))
        self.assertGreaterEqual(geometry.DEPTH_MIN * 80, geometry.REAR_DEPTH * 15)
        color = Q(1, 5)
        for baseline, errors in product(geometry.BASELINES,
                ((Q(-1, 5), Q(0), Q(1, 5)), (Q(1, 5), Q(0), Q(-1, 5)))):
            reference = geometry.point_rgb('full', Q(540), errors, (color,) * 3, baseline)
            for depth in (Q(600), Q(660)):
                self.assertEqual(geometry.point_rgb('full', depth, errors, (color,) * 3, baseline), reference)
            # Also check the declared rectangle inclusion at extremal cameras.
            for center, error in zip(geometry.camera_centers(baseline), errors):
                front = _project_rect(geometry.FRONT_RECT, Q(540), center + error)
                rear = _project_rect(geometry.HIDDEN_REAR_RECT, geometry.REAR_DEPTH, center + error)
                self.assertTrue(rear[0] <= front[0] <= front[1] <= rear[1])
                self.assertTrue(rear[2] <= front[2] <= front[3] <= rear[3])

    def test_b80_adds_unsaturated_rear_free_deep_fringe(self):
        # Right camera, top central row, outer negative-u pixel [-.2,-.15].
        pixel = 2 * geometry.PIXELS_PER_VIEW + 3 * geometry.GRID.width
        depth = Q(643)
        old = geometry._point_areas('partial', depth, (Q(0),) * 3, Q(70))[pixel]
        new = geometry._point_areas('partial', depth, (Q(0),) * 3, Q(80))[pixel]
        self.assertEqual(old, (Q(0),) * 3)
        self.assertGreater(new[0], 0)
        self.assertEqual(new[1:], (Q(0), Q(0)))
        self.assertEqual(new[0], Q(11250) * (Q(100) / depth ** 2 - Q(3, 20) / depth))
        self.assertGreater(Q(7, 10) * new[0], solver.BETA + solver.EPSILON)

    def test_cache_contains_baseline_and_clear(self):
        geometry.clear_geometry_cache()
        first = geometry.geometry_bounds('none', Q(1, 5), Q(640), Q(648), Q(70))
        second = geometry.geometry_bounds('none', Q(1, 5), Q(640), Q(648), Q(80))
        self.assertNotEqual(first, second)
        self.assertEqual(geometry._geometry_bounds.cache_info().currsize, 2)
        self.assertEqual(first, geometry.geometry_bounds('none', Q(1, 5), Q(640), Q(648), Q(70)))
        self.assertGreater(geometry._geometry_bounds.cache_info().hits, 0)
        solver.clear_geometry_cache()
        for cached in (geometry._front_boxes, geometry._front_area_bounds, geometry._rear_boxes,
                geometry._geometry_bounds, geometry._point_areas, geometry.historical._geometry_bounds):
            self.assertEqual(cached.cache_info().currsize, 0)

    def test_b70_cold_solver_and_actions_identity(self):
        image = fixture(Q(70))
        for method in ('calibrated', 'joint'):
            solver.clear_geometry_cache()
            original = solver.frozen.solve_structured(image, Q(1, 5), method, clipped=True)
            solver.clear_geometry_cache()
            current = solver.solve_public(image, Q(1, 5), Q(70), method, clipped=True)
            timed = {'branch_seconds', 'outer_seconds', 'preparation_seconds'}
            self.assertEqual({k: v for k, v in current.items() if k not in timed},
                             {k: v for k, v in original.items() if k not in timed})
            for offset in (Q(0), Q(-1, 4), Q(1, 4), Q(-3, 4), Q(3, 4), Q(-2), Q(2), Q(-3), Q(3)):
                incumbent = Q(600) + offset
                self.assertEqual(certificate_hybrid(current['outer'], incumbent),
                                 certificate_hybrid(original['outer'], incumbent))

    def test_invalid_inputs_rejected(self):
        for baseline in (Q(75), 80.0, True):
            with self.assertRaises((TypeError, ValueError)):
                geometry.geometry_bounds('none', Q(1, 5), Q(540), Q(660), baseline)
        with self.assertRaises(ValueError):
            geometry.point_rgb('none', Q(600), (Q(0), Q(1), Q(0)), (Q(1, 5),) * 3, Q(80))


if __name__ == '__main__':
    unittest.main()
