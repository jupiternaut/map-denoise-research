"""Synthetic checks; never reads an archived dataset or replay reference."""

import io
import unittest

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from innovation_models import HGB_PARAMS, NORMALIZATION_EPSILON_MM2, train_policies


def errors_for_gain(gain):
    gain = np.asarray(gain, dtype=np.float64)
    return 0.25 + np.maximum(gain, 0.0), 0.25 + np.maximum(-gain, 0.0)


class ConstructionTests(unittest.TestCase):
    def test_small_weighted_objectives_and_zero_weight_exclusion(self):
        X = np.zeros((4, 64))
        gain = np.array([2.0, -4.0, 0.0, 100.0])
        e0, e1 = errors_for_gain(gain)
        weight = np.array([1.0, 2.0, 3.0, 0.0])
        policies = train_policies(X, gain, e0, e1, weight, 7)
        expected_gain = np.average(gain, weights=weight)
        expected_relative = np.average(
            gain / (e0 + e1 + NORMALIZATION_EPSILON_MM2), weights=weight
        )
        self.assertEqual(set(policies), {"direct_gain", "normalized_gain", "benefit_harm", "hurdle_gain"})
        for name, policy in policies.items():
            expected = expected_relative if name == "normalized_gain" else expected_gain
            np.testing.assert_allclose(policy.score(X), expected, rtol=1e-12, atol=1e-12)
            self.assertEqual(policy.default_threshold, 0.0)
            self.assertEqual(policy.score(X[:0]).shape, (0,))

    def test_single_class_and_zero_cases(self):
        for gain in (np.ones(5), -np.ones(5), np.zeros(5), np.array([0.0, 0.0, 0.0, 0.0, 1.0])):
            with self.subTest(gain=gain.tolist()):
                X = np.zeros((len(gain), 64))
                e0, e1 = errors_for_gain(gain)
                policies = train_policies(X, gain, e0, e1)
                for name, policy in policies.items():
                    expected = np.mean(gain / (e0 + e1 + 0.01)) if name == "normalized_gain" else np.mean(gain)
                    np.testing.assert_allclose(policy.score(X), expected, atol=1e-12)
        gain = np.array([1.0, -2.0])
        e0, e1 = errors_for_gain(gain)
        for weight in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
            policies = train_policies(np.zeros((2, 64)), gain, e0, e1, weight)
            np.testing.assert_allclose(policies["hurdle_gain"].score(np.zeros((1, 64))), np.average(gain, weights=weight))

    def test_hgb_learning_fixed_parameters_and_serialization(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(640, 64))
        X[::19, 8] = np.nan
        gain = np.where(X[:, 0] > 0.0, 1.0 + 0.2 * X[:, 1], -1.0 + 0.2 * X[:, 1])
        e0, e1 = errors_for_gain(gain)
        weight = rng.uniform(0.5, 2.0, size=len(X))
        policies = train_policies(X, gain, e0, e1, weight, seed=23)
        repeat = train_policies(X, gain, e0, e1, weight, seed=23)
        for name, policy in policies.items():
            scores = policy.score(X)
            self.assertTrue(np.isfinite(scores).all())
            self.assertGreater(np.corrcoef(scores, gain)[0, 1], 0.9)
            np.testing.assert_array_equal(scores, repeat[name].score(X))
            buffer = io.BytesIO()
            joblib.dump(policy, buffer)
            buffer.seek(0)
            np.testing.assert_array_equal(scores, joblib.load(buffer).score(X))
            for value in vars(policy).values():
                if isinstance(value, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
                    for parameter, expected in HGB_PARAMS.items():
                        self.assertEqual(value.get_params()[parameter], expected)

    def test_validation(self):
        X = np.zeros((3, 64))
        gain = np.array([1.0, 0.0, -1.0])
        e0, e1 = errors_for_gain(gain)
        with self.assertRaises(ValueError):
            train_policies(X, gain, e0, e1, np.zeros(3))
        with self.assertRaises(ValueError):
            train_policies(X, gain, e0, e1, np.array([-1.0, 1.0, 1.0]))
        with self.assertRaises(ValueError):
            train_policies(X, gain, -e0, e1)
        with self.assertRaises(ValueError):
            train_policies(X, gain[:, None], e0, e1)
        with self.assertRaises(ValueError):
            train_policies(X[:0], gain[:0], e0[:0], e1[:0])
        policy = train_policies(X, gain, e0, e1)["direct_gain"]
        with self.assertRaises(ValueError):
            policy.score(np.zeros((2, 63)))
        invalid = X.copy()
        invalid[0, 0] = np.inf
        with self.assertRaises(ValueError):
            policy.score(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
