"""Synthetic tests only: no datasets, geometry labels, or model fitting."""

import unittest
import warnings

import numpy as np

from paired_features import FEATURE_NAMES, SCORE_TOLERANCE, gate_pairs, summarize_pairs


class PairedFeaturesTest(unittest.TestCase):
    def test_feature_schema(self):
        expected = [
            f"{kind}_{view}"
            for kind in ("old_cost", "new_cost", "paired_margin", "pair_valid",
                         "old_valid", "new_valid")
            for view in range(4)
        ] + ["paired_mean_margin", "paired_min_margin", "paired_max_margin",
             "paired_std_margin", "paired_win_fraction", "paired_valid_fraction",
             "paired_mean_old_cost", "paired_mean_new_cost"]
        self.assertEqual(tuple(expected), FEATURE_NAMES)
        self.assertEqual(len(set(FEATURE_NAMES)), 32)

    def test_exact_aggregation_and_gate(self):
        scores = np.array([[[-1, 1]], [[0, 0.5]], [[0.5, 0]], [[1, 0]]])
        features = summarize_pairs(scores)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(features.shape, (1, 32))
        np.testing.assert_allclose(features[0, :4], [2, 1, 0.5, 0])
        np.testing.assert_allclose(features[0, 4:8], [0, 0.5, 1, 1])
        margins = np.array([2, 0.5, -0.5, -1])
        np.testing.assert_array_equal(features[0, 8:12], margins)
        np.testing.assert_array_equal(features[0, 12:24], 1)
        np.testing.assert_allclose(features[0, 24:], [
            margins.mean(), margins.min(), margins.max(), margins.std(),
            0.5, 1, 0.875, 0.625,
        ])
        np.testing.assert_array_equal(gate_pairs(features), [True])

    def test_zero_is_valid_but_ties_abstain(self):
        features = summarize_pairs(np.zeros((4, 3, 2), dtype=np.float32))
        np.testing.assert_array_equal(features[:, :8], 1)
        np.testing.assert_array_equal(features[:, 8:12], 0)
        np.testing.assert_array_equal(features[:, 12:24], 1)
        np.testing.assert_array_equal(features[:, 24:29], 0)
        np.testing.assert_array_equal(features[:, 29], 1)
        np.testing.assert_array_equal(features[:, 30:], 1)
        np.testing.assert_array_equal(gate_pairs(features), False)

    def test_missing_support_is_not_favorable_evidence(self):
        scores = np.full((4, 3, 2), np.nan)
        scores[:, 1, :] = [[0, 0.5], [0.5, np.nan], [np.nan, 1], [np.nan, np.nan]]
        scores[:, 2, :] = [[-0.5, 0.5], [0.5, 1], [1, np.nan], [np.nan, -1]]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            features = summarize_pairs(scores)
        np.testing.assert_array_equal(features[0, :8], 1)
        np.testing.assert_array_equal(features[0, 8:30], 0)
        np.testing.assert_array_equal(features[0, 30:], 1)
        np.testing.assert_array_equal(features[1, 12:16], [1, 0, 0, 0])
        np.testing.assert_array_equal(features[1, 16:20], [1, 1, 0, 0])
        np.testing.assert_array_equal(features[1, 20:24], [1, 0, 1, 0])
        np.testing.assert_allclose(features[1, 24:], [0.5, 0.5, 0.5, 0, 1, 0.25, 1, 0.5])
        np.testing.assert_allclose(features[2, 24:], [0.75, 0.5, 1, 0.25, 1, 0.5, 1, 0.25])
        np.testing.assert_array_equal(gate_pairs(features), [False, False, True])

    def test_disjoint_one_sided_validity_has_empty_paired_support(self):
        scores = np.array([[[1, np.nan]], [[np.nan, 1]],
                           [[-1, np.nan]], [[np.nan, -1]]])
        features = summarize_pairs(scores)
        np.testing.assert_array_equal(features[:, 12:16], 0)
        np.testing.assert_array_equal(features[:, 24:30], 0)
        np.testing.assert_array_equal(features[:, 30:], 1)
        np.testing.assert_array_equal(gate_pairs(features), [False])

    def test_candidate_swap(self):
        rng = np.random.default_rng(941)
        scores = rng.uniform(-1, 1, (4, 17, 2))
        scores[1, 2, 0] = np.nan
        scores[2, 3, 1] = np.nan
        forward, reverse = summarize_pairs(scores), summarize_pairs(scores[:, :, ::-1])
        np.testing.assert_array_equal(forward[:, :4], reverse[:, 4:8])
        np.testing.assert_array_equal(forward[:, 4:8], reverse[:, :4])
        np.testing.assert_array_equal(forward[:, 8:12], -reverse[:, 8:12])
        np.testing.assert_array_equal(forward[:, 12:16], reverse[:, 12:16])
        np.testing.assert_array_equal(forward[:, 16:20], reverse[:, 20:24])
        np.testing.assert_array_equal(forward[:, 24], -reverse[:, 24])
        np.testing.assert_array_equal(forward[:, 25], -reverse[:, 26])
        np.testing.assert_array_equal(forward[:, 27], reverse[:, 27])
        np.testing.assert_allclose(forward[:, 28] + reverse[:, 28], 1)
        np.testing.assert_array_equal(forward[:, 30], reverse[:, 31])
        self.assertFalse(np.any(gate_pairs(forward) & gate_pairs(reverse)))

    def test_row_and_view_permutation(self):
        rng = np.random.default_rng(320)
        scores = rng.uniform(-1, 1, (4, 8, 2))
        scores[0, 0, 0] = np.nan
        features = summarize_pairs(scores)
        order = np.array([3, 0, 2, 1])
        permuted = summarize_pairs(scores[order, ::-1])
        for start in range(0, 24, 4):
            np.testing.assert_array_equal(permuted[:, start:start + 4],
                                          features[::-1, start:start + 4][:, order])
        np.testing.assert_allclose(permuted[:, 24:], features[::-1, 24:])

    def test_no_mutation_alias_or_edit_for_identical_candidates(self):
        rng = np.random.default_rng(19)
        original = rng.uniform(-1, 1, (4, 9))
        scores = np.stack((original, original), axis=2)
        scores[1, 1, :] = np.nan
        snapshot = scores.copy()
        scores.flags.writeable = False
        features = summarize_pairs(scores)
        np.testing.assert_array_equal(scores, snapshot)
        self.assertFalse(np.shares_memory(features, scores))
        features.flags.writeable = False
        selected = gate_pairs(features)
        self.assertEqual(selected.dtype, np.bool_)
        self.assertFalse(np.shares_memory(selected, features))
        self.assertFalse(selected.any())
        np.testing.assert_array_equal(features[:, 8:12], 0)

    def test_empty_points(self):
        features = summarize_pairs(np.empty((4, 0, 2)))
        self.assertEqual(features.shape, (0, 32))
        self.assertEqual(gate_pairs(features).shape, (0,))

    def test_range_tolerance_clips_without_mutation(self):
        scores = np.zeros((4, 1, 2))
        scores[0, 0] = [-1 - SCORE_TOLERANCE / 2, 1 + SCORE_TOLERANCE / 2]
        original = scores.copy()
        features = summarize_pairs(scores)
        np.testing.assert_array_equal(scores, original)
        self.assertEqual(features[0, 0], 2)
        self.assertEqual(features[0, 4], 0)
        self.assertEqual(features[0, 8], 2)

    def test_invalid_scores_rejected(self):
        for shape in ((4, 2), (3, 5, 2), (4, 5, 3), (4, 5, 2, 1), ()):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                summarize_pairs(np.zeros(shape))
        for value in (np.inf, -np.inf, 1.001, -1.001):
            scores = np.zeros((4, 1, 2))
            scores[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize_pairs(scores)
        for dtype in (complex, object, bool, str):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                summarize_pairs(np.zeros((4, 1, 2)).astype(dtype))

    def test_invalid_gate_inputs_rejected(self):
        for shape in ((32,), (2, 31), (2, 32, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                gate_pairs(np.zeros(shape))
        for value in (np.inf, np.nan):
            features = np.zeros((1, 32))
            features[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                gate_pairs(features)
        features = np.zeros((1, 32))
        features[0, 12] = 0.5
        with self.assertRaises(ValueError):
            gate_pairs(features)


if __name__ == "__main__":
    unittest.main()
