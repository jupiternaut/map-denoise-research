"""Synthetic baseline API tests: no archived data or replay labels are read."""

from io import BytesIO
import unittest

import joblib
import numpy as np

from baseline_models import (
    CandidateErrorPolicy,
    ClassifierPolicy,
    ConstantPolicy,
    HGB_PARAMS,
    METHOD_NAMES,
    train_policies,
)


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(17)
        cls.X = rng.normal(size=(480, 64)).astype(np.float32)
        cls.e1 = np.where(cls.X[:, 0] > 0, 0.2, 2.2)
        cls.e0 = np.where(cls.X[:, 1] > 0, 3.0, 0.1)
        cls.gain = cls.e0 - cls.e1
        cls.weights = np.linspace(0.5, 1.5, len(cls.X))
        cls.policies = train_policies(
            cls.X, cls.gain, cls.e0, cls.e1, cls.weights, 29
        )

    def test_interface_serialization_and_empty_batch(self):
        self.assertEqual(tuple(self.policies), METHOD_NAMES)
        for name, policy in self.policies.items():
            with self.subTest(name=name):
                score = policy.score(self.X)
                self.assertEqual(score.shape, (len(self.X),))
                self.assertTrue(np.isfinite(score).all())
                self.assertEqual(policy.score(self.X[:0]).shape, (0,))
                buffer = BytesIO()
                joblib.dump(policy, buffer)
                buffer.seek(0)
                restored = joblib.load(buffer)
                np.testing.assert_array_equal(score, restored.score(self.X))

    def test_exact_scalar_definitions(self):
        formulas = {
            "photo_cost": -self.X[:, 12],
            "mode_gap": self.X[:, 14],
            "source_agreement": self.X[:, 20],
            "small_displacement": -np.abs(self.X[:, 8]),
        }
        for name, expected in formulas.items():
            np.testing.assert_array_equal(self.policies[name].score(self.X), expected)
            self.assertIsNone(self.policies[name].default_threshold)

    def test_learned_direction_and_budget(self):
        confidence = self.policies["absolute_confidence_hgb"]
        quality = self.policies["candidate_error_hgb"]
        benefit = self.policies["gain_sign_hgb"]
        self.assertIsInstance(confidence, ClassifierPolicy)
        self.assertIsInstance(quality, CandidateErrorPolicy)
        self.assertIsInstance(benefit, ClassifierPolicy)
        for model in (confidence, quality, benefit):
            params = model.estimator.get_params()
            for key, value in HGB_PARAMS.items():
                self.assertEqual(params[key], value)
        for model, target in (
            (confidence, self.e1 <= 1),
            (quality, self.e1 <= 1),
            (benefit, self.gain > 0),
        ):
            score = model.score(self.X)
            self.assertGreater(score[target].mean(), score[~target].mean())
        np.testing.assert_array_equal(quality.score(self.X), -quality.estimator.predict(self.X))
        self.assertEqual(confidence.default_threshold, 0.5)
        self.assertEqual(quality.default_threshold, -1.0)
        self.assertEqual(benefit.default_threshold, 0.5)

    def test_boundary_targets_and_zero_weight_class(self):
        X = np.zeros((160, 64))
        e0 = np.ones(160)
        e1 = np.ones(160)
        e1[-1] = 2
        w = np.ones(160)
        w[-1] = 0
        policies = train_policies(X, e0 - e1, e0, e1, w, 2)
        absolute = policies["absolute_confidence_hgb"]
        gain = policies["gain_sign_hgb"]
        self.assertIsInstance(absolute, ConstantPolicy)
        self.assertIsInstance(gain, ConstantPolicy)
        np.testing.assert_array_equal(absolute.score(X), np.ones(160))
        np.testing.assert_array_equal(gain.score(X), np.zeros(160))

    def test_weights_are_used_by_models(self):
        X = np.zeros((200, 64))
        e1 = np.concatenate([np.zeros(100), np.full(100, 2.0)])
        e0 = np.ones(200)
        w = np.concatenate([np.ones(100), np.full(100, 3.0)])
        policies = train_policies(X, e0 - e1, e0, e1, w, 2)
        np.testing.assert_allclose(policies["absolute_confidence_hgb"].score(X), 0.25, atol=1e-7)
        np.testing.assert_allclose(policies["gain_sign_hgb"].score(X), 0.25, atol=1e-7)
        np.testing.assert_allclose(policies["candidate_error_hgb"].score(X), -1.5, atol=1e-7)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.policies["photo_cost"].score(np.zeros((2, 63)))
        invalid = self.X[:2].copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.policies["photo_cost"].score(invalid)
        with self.assertRaises(ValueError):
            train_policies(self.X, self.gain + 1, self.e0, self.e1, None, 2)
        with self.assertRaises(ValueError):
            train_policies(self.X, self.gain, self.e0, self.e1, np.zeros(len(self.X)), 2)


if __name__ == "__main__":
    unittest.main()
