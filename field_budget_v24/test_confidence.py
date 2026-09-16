"""Tests for signed fixed-action confidence diagnostics."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confidence import diagnose


class ConfidenceTests(unittest.TestCase):
    def test_positive_and_negative_signed_rank(self):
        alpha = np.linspace(0, 1, 9)
        result = diagnose(alpha, np.ones(9), {'help': 1-alpha, 'harm': 1+alpha})
        self.assertAlmostEqual(result['probes']['help']['spearman_rank_correlation'], 1.)
        self.assertAlmostEqual(result['probes']['harm']['spearman_rank_correlation'], -1.)
        self.assertGreater(result['probes']['help']['high_low_comparison']['high_minus_low_mean_signed_gain'], 0)
        self.assertLess(result['probes']['harm']['high_low_comparison']['high_minus_low_mean_signed_gain'], 0)
        json.dumps(result, allow_nan=False)

    def test_large_harm_not_misread_as_good_absolute_changes(self):
        result = diagnose([0., .2, .7, 1.], [10.] * 4, {'probe': [9., 10., 12., 20.]})
        probe = result['probes']['probe']
        self.assertAlmostEqual(probe['spearman_rank_correlation'], -1.)
        self.assertEqual(probe['overall']['beneficial_fraction'], .25)
        self.assertEqual(probe['overall']['harmful_fraction'], .5)
        self.assertLess(probe['overall']['mean_signed_gain'], 0)

    def test_constant_alpha_has_one_bin_and_no_tail_comparison(self):
        result = diagnose([.5] * 8, [2.] * 8, {'probe': np.arange(8) / 4})
        probe = result['probes']['probe']
        self.assertIsNone(probe['spearman_rank_correlation'])
        self.assertIsNone(probe['high_low_comparison'])
        self.assertEqual(len(probe['bins']), 1)
        self.assertEqual(probe['bins'][0]['n'], 8)
        self.assertEqual(result['high_low_counts']['overlapping_ties_excluded'], 8)
        json.dumps(result, allow_nan=False)

    def test_constant_gain_and_single_point(self):
        constant = diagnose([0, .5, 1], [2, 3, 4], {'probe': [1, 2, 3]})
        self.assertIsNone(constant['probes']['probe']['spearman_rank_correlation'])
        singleton = diagnose([.5], [1.], {'probe': [1.]})
        self.assertIsNone(singleton['probes']['probe']['high_low_comparison'])
        self.assertEqual(singleton['probes']['probe']['overall']['unchanged_fraction'], 1.)

    def test_ties_not_split_and_order_does_not_change_results(self):
        alpha = np.array([0.] * 5 + [.5] * 6 + [1.] * 5)
        baseline = np.full(len(alpha), 2.)
        errors = baseline-alpha
        first = diagnose(alpha, baseline, {'probe': errors})
        permutation = np.random.default_rng(932414).permutation(len(alpha))
        second = diagnose(alpha[permutation], baseline[permutation], {'probe': errors[permutation]})
        self.assertEqual(first, second)
        bins = first['probes']['probe']['bins']
        self.assertEqual(sum(row['n'] for row in bins), len(alpha))
        for value in np.unique(alpha):
            self.assertEqual(sum(row['min_alpha'] <= value <= row['max_alpha'] for row in bins), 1)

    def test_overlapping_tail_ties_removed_or_missing(self):
        alpha = np.array([0., .1] + [.5] * 16 + [.9, 1.])
        result = diagnose(alpha, np.ones(20), {'probe': 1-alpha})
        self.assertEqual(result['high_low_counts'], dict(low=2, high=2, overlapping_ties_excluded=16))
        self.assertIsNotNone(result['probes']['probe']['high_low_comparison'])
        result = diagnose([0.] * 9 + [1.], [2.] * 10, {'probe': [1.] * 10})
        self.assertIsNone(result['probes']['probe']['high_low_comparison'])
        self.assertEqual(len(result['probes']['probe']['bins']), 2)

    def test_invalid_inputs(self):
        cases = [([], [], {'p': []}), ([[.5]], [1], {'p': [1]}),
                 ([.5, .6], [1], {'p': [1, 1]}),
                 ([np.nan], [1], {'p': [1]}), ([np.inf], [1], {'p': [1]}),
                 ([-.01], [1], {'p': [1]}), ([1.01], [1], {'p': [1]}),
                 ([.5], [np.nan], {'p': [1]}), ([.5], [1], {'p': [np.inf]}),
                 ([.5], [1], {'p': [[1]]}), ([.5], [1], {'p': [1, 2]}),
                 ([.5], [1], {}), ([.5], [1], []), ([.5], [1], {1: [1]}),
                 ([.5], [1], {'': [1]}), ([.5], [1e308], {'p': [-1e308]})]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                diagnose(*args)

    def test_inputs_remain_unchanged(self):
        alpha = np.array([0., .5, 1.])
        baseline, probe = np.array([1., 2., 3.]), np.array([1.1, 2.1, 3.1])
        original = [array.copy() for array in (alpha, baseline, probe)]
        diagnose(alpha, baseline, {'probe': probe})
        for actual, expected in zip((alpha, baseline, probe), original):
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
