import os
import sys
import unittest

import numpy as np

os.chdir("/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration")
sys.path.insert(0, os.getcwd())

from evaluation.metrics import EMPTY_PENALTY_MM, score_arrays  # noqa: E402


class EvalTests(unittest.TestCase):
    def test_empty_output_is_penalized(self):
        laser = np.random.default_rng(0).normal(size=(200, 3))
        aabb_min = np.array([-10.0, -10.0, -10.0])
        aabb_max = np.array([10.0, 10.0, 10.0])
        result = score_arrays(np.zeros((0, 3)), laser, aabb_min, aabb_max, None, 0.8)
        self.assertEqual(result["status"], "EMPTY_OUTPUT")
        self.assertEqual(result["E_sym_mm"], EMPTY_PENALTY_MM)
        self.assertNotEqual(result["E_sym_mm"], 0)

    def test_identical_points_near_zero(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = score_arrays(pts, pts, pts.min(0) - 1, pts.max(0) + 1, None, 0.1)
        self.assertEqual(result["status"], "SCORED")
        self.assertLess(result["E_sym_mm"], 1e-9)

    def test_duplicate_points_do_not_win_after_voxel(self):
        rng = np.random.default_rng(1)
        pts = rng.normal(scale=2.0, size=(80, 3))
        doubled = np.vstack([pts, pts])
        lo, hi = pts.min(0) - 1, pts.max(0) + 1
        a = score_arrays(pts, pts, lo, hi, None, 0.5)
        b = score_arrays(doubled, pts, lo, hi, None, 0.5)
        self.assertAlmostEqual(a["E_sym_mm"], b["E_sym_mm"], places=6)
        self.assertEqual(a["n_output_eval"], b["n_output_eval"])


if __name__ == "__main__":
    unittest.main()
