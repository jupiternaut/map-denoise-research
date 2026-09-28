"""Focused tests for F-only support selection and paired R identity."""
import unittest
import numpy as np
from common import CLOSEOUT  # Set only the frozen import paths.
from support_features import choose_support, gather_reserved
from paired_features import summarize_pairs
from v28_closeout.surfacelet import _aggregate


class SupportTests(unittest.TestCase):
    def test_historical_top_three_min_two_and_tie(self):
        fit = np.full((4, 3, 20), np.nan)
        fit[:, 0, 4] = [.9, .8, .7, -.9]
        fit[:, 0, 5] = [.79, .79, .79, .79]
        fit[:2, 1, 8] = [.5, .5]
        fit[:2, 1, 9] = [.5, .5]
        fit[0, 2, 1] = 1
        np.testing.assert_array_equal(choose_support(fit), [4, 8, -1])
        cost, _, _ = _aggregate(fit[..., None], 3)
        np.testing.assert_array_equal(choose_support(fit)[:2], cost[:2, :, 0].argmin(axis=1))

    def test_choice_does_not_read_reserved_or_incumbent(self):
        rng = np.random.default_rng(9)
        scores = rng.uniform(-1, 1, (8, 13, 20, 2)).astype(np.float32)
        h = choose_support(scores[:4, :, :, 1])
        changed = scores.copy()
        changed[4:] = np.nan
        changed[:4, :, :, 0] = 1
        np.testing.assert_array_equal(h, choose_support(changed[:4, :, :, 1]))

    def test_same_h_for_both_positions_and_source_identity(self):
        scores = np.arange(4 * 3 * 20 * 2, dtype=float).reshape(4, 3, 20, 2)
        h = np.array([5, 19, 0], dtype=np.int8)
        pair = gather_reserved(scores, h)
        for row in range(3):
            np.testing.assert_array_equal(pair[:, row], scores[:, row, h[row]])

    def test_missing_f_has_neutral_features_and_zero_validity(self):
        h = choose_support(np.full((4, 2, 20), np.nan))
        pair = gather_reserved(np.ones((4, 2, 20, 2)), h)
        self.assertTrue(np.isnan(pair).all())
        feature = summarize_pairs(pair)
        np.testing.assert_array_equal(feature[:, :8], 1)
        np.testing.assert_array_equal(feature[:, 8:30], 0)
        np.testing.assert_array_equal(feature[:, 30:], 1)

    def test_h5_retains_old_full_pair(self):
        rng = np.random.default_rng(4)
        scores = rng.uniform(-1, 1, (4, 11, 20, 2)).astype(np.float32)
        h = np.full(11, 5, dtype=np.int8)
        np.testing.assert_array_equal(summarize_pairs(gather_reserved(scores, h)),
                                      summarize_pairs(scores[:, :, 5]))


if __name__ == '__main__':
    unittest.main()
