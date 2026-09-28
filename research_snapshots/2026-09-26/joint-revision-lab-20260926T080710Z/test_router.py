"""Independent pure tests of the joint KEEP/A/B method contract."""
import unittest
from unittest.mock import patch

import numpy as np
import router
from train import calibration_row


class RecordingRegressor:
    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, x, y, sample_weight):
        self.x = np.array(x, copy=True)
        self.y = np.array(y, copy=True)
        self.weights = np.array(sample_weight, copy=True)
        return self

    def predict(self, x):
        return x[:, 0] + 2 * x[:, 2] - x[:, -1]


class JointContractTests(unittest.TestCase):
    def setUp(self):
        self.features = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        self.geometry = np.array([
            [[0, 0, 0], [1, 0, 0], [0, 2, 0]],
            [[1, 1, 1], [2, 1, 1], [0, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
            [[1, 2, 3], [2, 2, 3], [2, 2, 3]],
        ], dtype=float)

    def test_context_has_own_competitor_and_geometric_units(self):
        context = router.contextual_features(self.features, self.geometry)
        self.assertEqual(context.shape, (4, 2, 10))
        np.testing.assert_array_equal(context[:, 0, :3], self.features[:, 0])
        np.testing.assert_array_equal(context[:, 0, 3:6], self.features[:, 1])
        np.testing.assert_array_equal(context[:, 1, :3], self.features[:, 1])
        np.testing.assert_array_equal(context[:, 1, 3:6], self.features[:, 0])
        np.testing.assert_allclose(context[0, 0, -4:], [1, 2, np.sqrt(5), 0], rtol=1e-7)
        np.testing.assert_allclose(context[0, 1, -4:], [2, 1, np.sqrt(5), 0], rtol=1e-7)
        self.assertEqual(context[1, 0, -1], -1)
        self.assertEqual(context[2, 0, -1], 0)

    def test_context_and_same_shared_scorer_swap_equivariance(self):
        context = router.contextual_features(self.features, self.geometry)
        swapped = router.contextual_features(self.features[:, ::-1], self.geometry[:, [0, 2, 1]])
        np.testing.assert_array_equal(swapped, context[:, ::-1])
        model = RecordingRegressor()
        scores = router.predict_shared(model, context)
        swapped_scores = router.predict_shared(model, swapped)
        np.testing.assert_array_equal(swapped_scores, scores[:, ::-1])

    def test_geometry_context_rigid_motion_invariance(self):
        a = router.contextual_features(self.features, self.geometry)
        moved = self.geometry[:, :, [2, 0, 1]] * np.array([-1, 1, -1]) + [4, 5, 6]
        b = router.contextual_features(self.features, moved)
        np.testing.assert_array_equal(a, b)

    def test_shared_fit_interleaves_candidates_targets_and_weights(self):
        context = router.contextual_features(self.features, self.geometry)
        e0 = np.array([10., 20., 30., 40.])
        losses = np.array([[8, 11], [12, 10], [35, 31], [0, 60]], dtype=float)
        gains = e0[:, None] - losses
        weights = np.array([1., 2., 3., 4.])
        with patch.object(router, 'HistGradientBoostingRegressor', RecordingRegressor):
            model = router.fit_shared(context, gains, weights)
        np.testing.assert_array_equal(model.x, context.reshape(8, 10))
        np.testing.assert_array_equal(model.y, [2, -1, 8, 10, -5, -1, 40, -20])
        np.testing.assert_array_equal(model.weights, [1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual(model.parameters['loss'], 'squared_error')

    def test_independent_fit_uses_candidate_own_features_and_gains(self):
        gains = np.arange(8).reshape(4, 2) - 3
        weights = np.arange(4) + 1
        with patch.object(router, 'HistGradientBoostingRegressor', RecordingRegressor):
            models = router.fit_independent(self.features, gains, weights)
        self.assertIsNot(models[0], models[1])
        for candidate in range(2):
            np.testing.assert_array_equal(models[candidate].x, self.features[:, candidate])
            np.testing.assert_array_equal(models[candidate].y, gains[:, candidate])
            np.testing.assert_array_equal(models[candidate].weights, weights)

    def test_case_weights_match_case_and_baseline_mse_objective(self):
        ids = np.array([0, 0, 1, 1, 1])
        e0 = np.array([1, 3, 2, 4, 6], dtype=float)
        raw = np.array([1 / 4, 1 / 4, 1 / 12, 1 / 12, 1 / 12])
        weights = router.case_weights(ids, e0)
        np.testing.assert_allclose(weights, raw / raw.mean(), rtol=1e-15)
        self.assertAlmostEqual(weights.mean(), 1.)
        np.testing.assert_allclose(router.case_weights(ids, e0 * 100), weights, rtol=1e-15)
        zero_case = router.case_weights(np.array([0, 1]), np.array([0., 1.]))
        self.assertTrue(np.isfinite(zero_case).all())

    def test_identity_is_keep_even_with_large_predicted_gain(self):
        geometry = np.repeat(self.geometry[:, :1], 3, axis=1)
        scores = np.array([[99, 2], [0, -1], [-np.inf, 5], [8, -np.inf]])
        decision = router.routes(scores, geometry, -1e6)
        np.testing.assert_array_equal(decision, 0)
        np.testing.assert_array_equal(router.materialize(geometry, decision), geometry[:, 0])

    def test_duplicate_geometry_has_equal_canonical_utility(self):
        geometry = self.geometry[[3, 3, 3, 3]]
        scores = np.array([[2., 6.], [2., -np.inf], [-np.inf, 6.], [-np.inf, -np.inf]])
        canonical = router.canonical_scores(scores, geometry)
        np.testing.assert_array_equal(canonical[:, 0], canonical[:, 1])
        self.assertEqual(canonical[0, 0], 4.)
        self.assertTrue(np.isneginf(canonical[-1]).all())
        np.testing.assert_array_equal(scores[0], [2, 6])

    def test_strict_threshold_and_keep_zero_utility(self):
        geometry = self.geometry[[0, 0, 0, 0]]
        scores = np.array([[0, -1], [2, 1], [1, 2], [-2, -1]], dtype=float)
        np.testing.assert_array_equal(router.routes(scores, geometry), [0, 1, 2, 0])
        np.testing.assert_array_equal(router.routes(scores, geometry, 2), 0)
        np.testing.assert_array_equal(router.routes(scores, geometry, -100, 'keep'), 0)

    def test_tie_routes_output_is_swap_equivariant(self):
        # Includes unequal displacement, symmetric equal-length proposals and duplicates.
        scores = np.ones((4, 2))
        for threshold in (0., 1., -1.):
            original = router.materialize(self.geometry, router.routes(scores, self.geometry, threshold))
            swapped = self.geometry[:, [0, 2, 1]]
            other = router.materialize(swapped, router.routes(scores[:, ::-1], swapped, threshold))
            np.testing.assert_array_equal(original, other)
        route = router.routes(scores, self.geometry)
        np.testing.assert_array_equal(route, [1, 2, 2, 1])

    def test_random_route_output_is_swap_equivariant(self):
        rng = np.random.default_rng(195)
        geometry = rng.normal(size=(200, 3, 3))
        scores = rng.normal(size=(200, 2))
        for threshold in (-1, 0, 1):
            original = router.materialize(geometry, router.routes(scores, geometry, threshold))
            swapped = geometry[:, [0, 2, 1]]
            other = router.materialize(swapped, router.routes(scores[:, ::-1], swapped, threshold))
            np.testing.assert_array_equal(original, other)

    def test_missing_support_never_produces_positive_evidence(self):
        features = np.zeros((4, 2, 128), np.float32)
        features[:, :, 120] = 100
        scores = router.support_scores(features)
        self.assertTrue(np.isneginf(scores).all())
        np.testing.assert_array_equal(router.routes(scores, self.geometry, -1e6), 0)
        features[0, 1, 125] = .5
        self.assertEqual(router.support_scores(features)[0, 1], 100)

    def test_calibration_uses_fixed_supported_loss_rows_and_case_means(self):
        geometry = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]] * 5, dtype=float)
        data = dict(case_id=np.array([0, 0, 1, 1, 1]),
                    calibration_support=np.array([True, False, True, True, False]),
                    condition=np.array(['native', 'native', 'plus3', 'plus3', 'plus3']),
                    losses=np.array([[4., 2., 10.], [1e9, 0., 0.],
                                     [10., 2., 30.], [10., 6., 30.], [1e9, 0., 0.]]))
        scores = np.ones((5, 2))
        scores[:, 1] = -1
        row = calibration_row(data, geometry, scores)
        self.assertAlmostEqual(row['objective_relative_MSE'], .45)
        self.assertAlmostEqual(row['native_MSE'], 2.)
        self.assertAlmostEqual(row['native_identity_MSE'], 4.)
        self.assertAlmostEqual(row['injected_relative_MSE'], .4)

    def test_materialize_returns_exact_selected_candidate_coordinates(self):
        decisions = np.array([0, 1, 2, 0], dtype=np.uint8)
        result = router.materialize(self.geometry, decisions)
        np.testing.assert_array_equal(result, np.stack([self.geometry[i, c] for i, c in enumerate(decisions)]))


if __name__ == '__main__':
    unittest.main()
