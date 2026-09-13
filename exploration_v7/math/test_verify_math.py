import unittest

import numpy as np

from verify_math import (decision_toy, duplicate_representation_toy,
                         free_energy, monotonic_toy, posterior)


class AlgebraTests(unittest.TestCase):
    def test_variational_identity_and_forbidden_candidate(self):
        cost = np.array([[.2, 1., 3.], [2., .1, .4]])
        prior = np.array([[.4, .6, 0.], [.2, .3, .5]])
        r = np.array([[.7, .3, 0.], [.3, .3, .4]])
        weights = np.array([.4, .6])
        q, log_z = posterior(cost, prior)
        mask = r > 0
        kl = np.zeros_like(r)
        kl[mask] = r[mask] * np.log(r[mask] / q[mask])
        expected = weights @ (kl.sum(axis=1) - log_z)
        self.assertAlmostEqual(free_energy(r, cost, prior, weights), expected, places=13)
        self.assertAlmostEqual(free_energy(q, cost, prior, weights), -weights @ log_z, places=13)
        self.assertEqual(q[0, 2], 0.)
        np.testing.assert_allclose(q.sum(axis=1), 1., atol=1e-14, rtol=0.)

    def test_full_rank_half_steps(self):
        result = monotonic_toy()
        self.assertLessEqual(result["maximum_objective_increase"], result["numerical_tolerance"])
        self.assertTrue(all(r == [3, 3] for r in result["rank_by_iteration"]))

    def test_rank_deficiency_does_not_prevent_in_sample_minimization(self):
        result = monotonic_toy(True)
        self.assertLessEqual(result["maximum_objective_increase"], result["numerical_tolerance"])
        self.assertTrue(all(r == [2, 2] for r in result["rank_by_iteration"]))

    def test_prior_refinement_not_uniform_renormalization_is_invariant(self):
        result = duplicate_representation_toy()
        np.testing.assert_allclose(result["original_aggregate"], result["cloned_split_prior_aggregate"])
        self.assertNotEqual(result["original_aggregate"], result["cloned_uniform_aggregate"])

    def test_representation_refinement_with_nonzero_costs(self):
        old, _ = posterior(np.array([[1., 3.]]), np.array([[.4, .6]]))
        refined, _ = posterior(np.array([[1., 1., 3.]]), np.array([[.1, .3, .6]]))
        np.testing.assert_allclose(old[0], [refined[0, :2].sum(), refined[0, 2]], atol=1e-14)

    def test_mean_and_hard_actions_optimize_different_losses(self):
        result = decision_toy()
        self.assertEqual(result["mean_distance_to_plane_union_mm"], 2.)
        self.assertEqual(result["hard_distance_to_plane_union_mm"], 0.)
        self.assertEqual(result["mean_expected_correspondence_mse_mm2"], 4.)
        self.assertEqual(result["hard_expected_correspondence_mse_mm2"], 8.)


if __name__ == "__main__":
    unittest.main()
