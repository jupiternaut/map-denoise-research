"""Synthetic evaluation contracts; no files, GT references, or models are read."""

import unittest

import numpy as np

from evaluate_replay import matched_random, measure


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        # Deliberately put very large improvements/harm outside fixed support.
        # Zero-displacement rows have equal before/after distances.
        self.d0 = np.array([2.0, 1.0, 3.0, 0.5, 7.0, 4.0, 9.0, 6.0])
        self.d1 = np.array([1.0, 2.0, 3.0, 0.2, 100.0, 0.1, 1.0, 6.0])
        self.support = np.array([True, True, True, True, False, True, False, False])
        self.move2 = np.array([1.0, 1.0, 0.0, 0.3, 93.0, 3.9, 8.0, 0.0]) ** 2

    def test_keep_has_zero_gain_and_no_motion(self):
        keep = np.zeros(len(self.d0), dtype=bool)
        result = measure(self.d0, self.d1, self.support, self.move2, keep)
        self.assertAlmostEqual(result["source_MSE_mm2"], np.mean(self.d0[self.support] ** 2))
        self.assertAlmostEqual(result["source_MAE_mm"], np.mean(self.d0[self.support]))
        for key in (
            "benefit_sum_mm2", "harm_sum_mm2", "improved_fraction", "harmed_fraction",
            "accepted_fraction", "accepted_support_fraction", "moved_fraction", "move_RMS_mm",
        ):
            with self.subTest(metric=key):
                self.assertEqual(result[key], 0.0)

    def test_all_benefit_minus_harm_equals_mse_reduction_mass(self):
        accept = np.ones(len(self.d0), dtype=bool)
        result = measure(self.d0, self.d1, self.support, self.move2, accept)
        identity_mse = float(np.mean(self.d0[self.support] ** 2))
        after_mse = float(np.mean(self.d1[self.support] ** 2))
        self.assertAlmostEqual(result["source_MSE_mm2"], after_mse)
        self.assertAlmostEqual(result["benefit_sum_mm2"], 19.2)
        self.assertAlmostEqual(result["harm_sum_mm2"], 3.0)
        self.assertAlmostEqual(
            result["benefit_sum_mm2"] - result["harm_sum_mm2"],
            (identity_mse - after_mse) * int(self.support.sum()),
        )
        self.assertAlmostEqual(result["improved_fraction"], 3 / 5)
        self.assertAlmostEqual(result["harmed_fraction"], 1 / 5)
        self.assertEqual(result["accepted_fraction"], 1.0)
        self.assertEqual(result["accepted_support_fraction"], 1.0)

    def test_arbitrary_mask_obeys_gain_mass_identity(self):
        rng = np.random.default_rng(17)
        before = float(np.mean(self.d0[self.support] ** 2))
        for iteration in range(20):
            accept = rng.random(len(self.d0)) < 0.5
            with self.subTest(iteration=iteration):
                result = measure(self.d0, self.d1, self.support, self.move2, accept)
                self.assertAlmostEqual(
                    result["benefit_sum_mm2"] - result["harm_sum_mm2"],
                    (before - result["source_MSE_mm2"]) * int(self.support.sum()),
                )

    def test_accepted_zero_displacement_does_not_count_as_movement(self):
        only_zero = self.move2 == 0
        result = measure(self.d0, self.d1, self.support, self.move2, only_zero)
        self.assertGreater(result["accepted_fraction"], 0)
        self.assertEqual(result["moved_fraction"], 0.0)
        self.assertEqual(result["move_RMS_mm"], 0.0)
        all_result = measure(
            self.d0, self.d1, self.support, self.move2, np.ones(len(self.d0), dtype=bool)
        )
        self.assertEqual(all_result["moved_fraction"], 6 / 8)
        self.assertAlmostEqual(all_result["move_RMS_mm"], np.sqrt(np.mean(self.move2)))

    def test_measure_preserves_every_input_array(self):
        accept = np.array([True, False, True, False, True, False, True, False])
        arrays = (self.d0, self.d1, self.support, self.move2, accept)
        snapshots = tuple(array.copy() for array in arrays)
        # Read-only arrays make accidental mutation fail immediately.
        for array in arrays:
            array.flags.writeable = False
        result = measure(*arrays)
        self.assertGreater(result["accepted_fraction"], 0)
        for array, snapshot in zip(arrays, snapshots):
            np.testing.assert_array_equal(array, snapshot)

    def test_random_controls_match_each_support_and_displacement_stratum(self):
        levels = np.array([
            0.0, 0.8e-7, 1e-7, 0.1, 0.25, 0.3, 0.5, 0.7, 1.0, 1.6,
            2.0, 2.5, 3.0, 3.5, 4.0, 5.5, 6.0, 6.000001, 7.0,
        ])
        movement = np.repeat(levels, 12)
        support = np.tile(np.array([False] * 6 + [True] * 6), len(levels))
        accept = np.arange(len(movement)) % 5 < 2
        # The declared displacement intervals are [lo, hi), including exact
        # boundary values. Build them explicitly, independent of digitize.
        edges = np.array([0.0, 1e-7, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.000001, np.inf])
        accept[(movement < 1e-7) & ~support] = False
        accept[(movement >= 6.000001) & support] = True
        snapshots = [value.copy() for value in (accept, support, movement)]
        for value in (accept, support, movement):
            value.flags.writeable = False
        for mode in ("count", "bin"):
            intervals = [(0.0, np.inf)] if mode == "count" else list(zip(edges[:-1], edges[1:]))
            for seed in (0, 17, 20260926):
                with self.subTest(mode=mode, seed=seed):
                    actual = matched_random(accept, support, movement, mode, seed)
                    self.assertEqual(actual.dtype, np.dtype(bool))
                    self.assertEqual(actual.shape, accept.shape)
                    np.testing.assert_array_equal(
                        actual, matched_random(accept, support, movement, mode, seed)
                    )
                    self.assertEqual(int(actual.sum()), int(accept.sum()))
                    for inside_support in (False, True):
                        for low, high in intervals:
                            stratum = (support == inside_support) & (movement >= low) & (movement < high)
                            self.assertEqual(int(actual[stratum].sum()), int(accept[stratum].sum()))
        for value, snapshot in zip((accept, support, movement), snapshots):
            np.testing.assert_array_equal(value, snapshot)


if __name__ == "__main__":
    unittest.main()
