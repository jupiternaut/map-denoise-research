"""Small exhaustive checks of the evaluation-only continuous oracle."""

import unittest

import numpy as np
from scipy.spatial import cKDTree

from segment_oracle import segment_nearest


def exhaustive(start, end, reference):
    direction = end - start
    length_sq = np.einsum("ij,ij->i", direction, direction)
    relative = reference[None, :, :] - start[:, None, :]
    dot = np.einsum("nkj,nj->nk", relative, direction)
    lam = np.divide(dot, length_sq[:, None], out=np.zeros_like(dot),
                    where=length_sq[:, None] > 0)
    lam = np.clip(lam, 0, 1)
    residual = relative - lam[:, :, None] * direction[:, None, :]
    squared = np.einsum("nkj,nkj->nk", residual, residual)
    minimum = squared.min(axis=1)
    min_lambda = np.where(squared == minimum[:, None], lam, np.inf).min(axis=1)
    index = np.where((squared == minimum[:, None]) & (lam == min_lambda[:, None]),
                     np.arange(len(reference)), len(reference)).min(axis=1)
    return minimum, min_lambda, index


class SegmentOracleTests(unittest.TestCase):
    def test_random_matches_exhaustive(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            reference = rng.normal(size=(137, 3))
            start = rng.normal(size=(79, 3))
            end = start + rng.normal(size=(79, 3)) * (0.01 if seed % 2 else 5)
            end[::9] = start[::9]
            expected = exhaustive(start, end, reference)
            actual = segment_nearest(start, end, reference, chunk_size=7)
            np.testing.assert_allclose(actual.squared_distance, expected[0], rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(actual.lambda_, expected[1], rtol=1e-13, atol=1e-14)
            np.testing.assert_array_equal(actual.reference_index, expected[2])

    def test_continuous_interior_improvement(self):
        a = np.array([[0., 0., 0.]])
        b = np.array([[1., 0., 0.]])
        result = segment_nearest(a, b, np.array([[0.4, 0., 0.]]))
        self.assertEqual(result.lambda_[0], 0.4)
        self.assertEqual(result.squared_distance[0], 0.)

    def test_endpoints_monotonicity(self):
        rng = np.random.default_rng(120)
        ref = rng.normal(size=(80, 3))
        a, b = rng.normal(size=(2, 160, 3))
        tree = cKDTree(ref)
        result = segment_nearest(a, b, tree=tree)
        endpoint_sq = np.minimum(tree.query(a)[0] ** 2, tree.query(b)[0] ** 2)
        self.assertTrue(np.all(result.squared_distance <= endpoint_sq + 1e-13))
        self.assertTrue(np.all((result.lambda_ >= 0) & (result.lambda_ <= 1)))

    def test_ties_favor_keep_then_lowest_reference_index(self):
        a = np.array([[0., 0., 0.], [0., 0., 0.]])
        b = np.array([[1., 0., 0.], [0., 0., 0.]])
        reference = np.array([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        result = segment_nearest(a, b, reference)
        np.testing.assert_array_equal(result.lambda_, [0., 0.])
        np.testing.assert_array_equal(result.reference_index, [1, 1])

    def test_zero_length_and_single_reference(self):
        a = np.array([[2., 3., 4.], [1., 1., 1.]])
        result = segment_nearest(a, a.copy(), np.array([[0., 0., 0.]]))
        np.testing.assert_array_equal(result.lambda_, [0., 0.])
        np.testing.assert_array_equal(result.squared_distance, [29., 3.])

    def test_empty_segments(self):
        result = segment_nearest(np.empty((0, 3)), np.empty((0, 3)), np.zeros((1, 3)))
        self.assertEqual(result.squared_distance.size, 0)
        self.assertEqual(result.candidate_evaluations, 0)

    def test_chunk_and_tree_interfaces_do_not_change_results(self):
        rng = np.random.default_rng(44)
        a, b = rng.normal(size=(2, 100, 3))
        ref = rng.normal(size=(400, 3))
        copies = [x.copy() for x in (a, b, ref)]
        first = segment_nearest(a, b, ref, chunk_size=1)
        second = segment_nearest(a, b, cKDTree(ref), chunk_size=1000)
        np.testing.assert_array_equal(first.squared_distance, second.squared_distance)
        np.testing.assert_array_equal(first.lambda_, second.lambda_)
        for actual, copy in zip((a, b, ref), copies):
            np.testing.assert_array_equal(actual, copy)

    def test_large_origin_small_segments(self):
        rng = np.random.default_rng(52)
        origin = np.array([1e8, -1e8, 1e8])
        a = origin + rng.normal(size=(20, 3)) * 1e-3
        b = a + rng.normal(size=(20, 3)) * 1e-3
        ref = origin + rng.normal(size=(70, 3)) * 1e-3
        expected = exhaustive(a, b, ref)
        result = segment_nearest(a, b, ref, chunk_size=3)
        np.testing.assert_allclose(result.squared_distance, expected[0], rtol=1e-13, atol=1e-20)
        np.testing.assert_allclose(result.lambda_, expected[1], rtol=1e-13, atol=1e-14)

    def test_invalid_reference_rejected(self):
        points = np.zeros((1, 3))
        with self.assertRaises(ValueError):
            segment_nearest(points, points, np.empty((0, 3)))
        with self.assertRaises(ValueError):
            segment_nearest(points, points, np.full((1, 3), np.nan))


if __name__ == "__main__":
    unittest.main()
