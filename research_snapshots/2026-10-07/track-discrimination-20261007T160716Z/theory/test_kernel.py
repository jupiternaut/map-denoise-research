import unittest
from fractions import Fraction
import numpy as np
from kernel import gain_bounds, gain_bounds_world, strictly_improves


class GainKernelTests(unittest.TestCase):
    def test_empty_is_unknown(self):
        self.assertIsNone(gain_bounds(2, 4, []))
        self.assertFalse(strictly_improves(gain_bounds(2, 4, [])))
        self.assertIsNone(gain_bounds_world([0, 0, 2], [0, 0, 4], [0]*3, [0, 0, 1], []))

    def test_right_and_left(self):
        self.assertEqual(gain_bounds(2, 4, [(4, 5)]), (4, 8))
        self.assertEqual(gain_bounds(4, 2, [(1, 2)]), (4, 8))

    def test_crossing_midpoint_nonconvex(self):
        self.assertEqual(gain_bounds(2, 4, [(1, 2), (4, 5)]), (-8, 8))
        self.assertFalse(strictly_improves(gain_bounds(2, 4, [(1, 2), (4, 5)])))

    def test_equal_and_boundary(self):
        self.assertEqual(gain_bounds(3, 3, [(1, 2), (5, 7)]), (0, 0))
        self.assertEqual(gain_bounds(2, 4, [(3, 5)]), (0, 8))
        self.assertFalse(strictly_improves(gain_bounds(2, 4, [(3, 5)])))

    def test_nonconvex_on_one_side(self):
        self.assertEqual(gain_bounds(2, 4, [(4, 5), (8, 9)]), (4, 24))

    def test_world_reduces_to_ray(self):
        intervals = [(1, 2), (4, 5)]
        self.assertEqual(gain_bounds_world([0, 0, 2], [0, 0, 4], [0]*3, [0, 0, 1], intervals), gain_bounds(2, 4, intervals))

    def test_world_off_ray_constant_gain(self):
        self.assertEqual(gain_bounds_world([3, 0, 4], [1, 0, 4], [0]*3, [0, 0, 1], [(1, 6)]), (8, 8))

    def test_world_nonunit_and_offset(self):
        rng = np.random.default_rng(91)
        for _ in range(100):
            a, b, C, r = rng.normal(size=(4, 3))
            intervals = [(0.1, 2.3), (3.7, 8.1)]
            bounds = gain_bounds_world(a, b, C, r, intervals)
            zs = np.r_[np.linspace(.1, 2.3, 30), np.linspace(3.7, 8.1, 30)]
            gains = ((a - C - zs[:, None]*r)**2).sum(1) - ((b - C - zs[:, None]*r)**2).sum(1)
            np.testing.assert_allclose(bounds, [gains.min(), gains.max()], rtol=1e-11, atol=1e-11)

    def test_shared_bias_does_not_imply_coverage(self):
        wrong_set = [(3.2, 3.4)]
        self.assertTrue(strictly_improves(gain_bounds(4.4, 3.2, wrong_set)))
        self.assertLess((4.4-6)**2 - (3.2-6)**2, 0)

    def test_wrong_layer_not_covered(self):
        self.assertTrue(strictly_improves(gain_bounds(4.4, 3.4, [(3.55, 3.65)])))
        self.assertLess((4.4-6)**2 - (3.4-6)**2, 0)

    def test_invalid(self):
        for intervals in [[(2, 1)], [(0, float('nan'))], [(0, float('inf'))]]:
            with self.assertRaises(ValueError):
                gain_bounds(2, 4, intervals)
        with self.assertRaises(ValueError):
            gain_bounds_world([0]*3, [1]*3, [0]*3, [0]*3, [(1, 2)])

    def test_outward_bounds_for_represented_floats(self):
        a, b, left, right = 1.1, 3.7, 2.4000000000000004, 8.9
        lo, hi = gain_bounds(a, b, [(left, right)])
        exact = [(Fraction(a)-Fraction(z))**2 - (Fraction(b)-Fraction(z))**2 for z in [left, right]]
        self.assertLessEqual(Fraction(lo), min(exact))
        self.assertGreaterEqual(Fraction(hi), max(exact))

    def test_cancellation_does_not_invent_gain(self):
        self.assertEqual(gain_bounds(1e16-2, 1e16+2, [(1e16, 1e16)]), (0, 0))
        self.assertFalse(strictly_improves(gain_bounds(1e16-2, 1e16+2, [(1e16, 1e16)])))


if __name__ == '__main__':
    unittest.main()
