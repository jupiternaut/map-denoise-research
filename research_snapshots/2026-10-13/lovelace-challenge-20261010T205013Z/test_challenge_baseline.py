"""Small, independent checks for the same-information finite-grid baseline."""
from fractions import Fraction as Q
from itertools import product
import unittest

import numpy as np

from challenge import ProfiledResidual, model_for
from continuous_world.interval_solver import decode


class ProfiledResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = ProfiledResidual("rectangle", (Q(590), Q(600), Q(610)))

    def test_profile_equals_explicit_joint_pose_search(self):
        for radius in (Q(0), Q(1, 5), Q(1)):
            obs = model_for("rectangle", radius).point_prediction(
                Q(601), (radius / 2, Q(0), -radius))
            observed = np.asarray(list(map(float, obs))).reshape(3, -1)
            allowed = self.baseline.allowed_indices(radius)
            brute = []
            for forecasts in self.baseline.forecasts:
                brute.append(min(max(np.max(np.abs(forecasts[l, 0] - observed[0])),
                                     np.max(np.abs(forecasts[allowed[0], 1] - observed[1])),
                                     np.max(np.abs(forecasts[r, 2] - observed[2])))
                                 for l, r in product(allowed, repeat=2)))
            np.testing.assert_array_equal(self.baseline.grid_losses(obs, radius), brute)

    def test_true_exact_incumbent_is_kept(self):
        obs = model_for("rectangle", Q(1)).point_prediction(Q(600), (Q(1), Q(0), Q(-1, 2)))
        losses = self.baseline.grid_losses(obs, Q(1))
        answer = self.baseline.select(obs, Q(1), Q(600), losses)
        self.assertEqual(answer["status"], "KEEP")
        self.assertEqual(decode(answer["output"]), Q(600))
        self.assertEqual(answer["incumbent_residual"], 0)

    def test_non_grid_incumbent_is_a_real_candidate(self):
        a = Q(4205, 7)
        obs = model_for("rectangle", Q(0)).point_prediction(a)
        losses = self.baseline.grid_losses(obs, Q(0))
        self.assertGreater(float(min(losses)), 0)
        answer = self.baseline.select(obs, Q(0), a, losses)
        self.assertEqual(answer["status"], "KEEP")
        self.assertEqual(decode(answer["output"]), a)

    def test_zero_radius_has_only_nominal_pose(self):
        self.assertEqual(len(self.baseline.allowed_indices(Q(0))), 1)
        self.assertEqual(len(self.baseline.allowed_indices(Q(1, 5))), 5)
        self.assertEqual(len(self.baseline.allowed_indices(Q(1))), 5)

    def test_no_depth_information_tie_keeps_incumbent(self):
        baseline = ProfiledResidual("layered_hidden", (Q(540), Q(600), Q(660)))
        obs = model_for("layered_hidden", Q(1)).point_prediction(Q(610), (Q(1), Q(0), Q(-1)))
        a = Q(3965, 7)
        result = baseline.select(obs, Q(1), a, baseline.grid_losses(obs, Q(1)))
        self.assertEqual(result["status"], "KEEP")
        self.assertEqual(decode(result["output"]), a)


if __name__ == "__main__":
    unittest.main()
