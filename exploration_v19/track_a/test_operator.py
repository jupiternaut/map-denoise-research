import unittest
from unittest.mock import patch
import numpy as np
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("track_a_operator", Path(__file__).with_name("operator.py"))
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)


class Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(919)
        self.x = np.c_[np.ones(60), rng.uniform(-1, 1, (60, 2))]
        self.y = rng.normal(0, .5, 60) + np.where(self.x[:, 1] > 0, 2., -2.)
        self.baseline = dict(k=2, family="L", means=np.array([-2., 2.]), slope=np.zeros(2))

    def test_fixed_prior_gradient(self):
        prior = np.c_[np.full(60, .3), np.full(60, .7)]
        t = np.array([.1, 2., .2, -.1])
        _, gradient = op.fixed_prior_objective(t, self.x[:, 1:], self.y, np.log(prior))
        numeric = np.zeros(4)
        for j in range(4):
            h = np.eye(4)[j] * 1e-6
            numeric[j] = (op.fixed_prior_objective(t+h, self.x[:, 1:], self.y, np.log(prior))[0]
                          - op.fixed_prior_objective(t-h, self.x[:, 1:], self.y, np.log(prior))[0]) / 2e-6
        np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-6)

    def test_gate_rows_exclude_held_out_heights(self):
        def fake_train(x, y, sigma, family):
            return dict(family="C", means=np.array([-1., 1.]), slope=np.zeros(2),
                        feature_center=np.zeros(2), feature_scale=np.ones(2),
                        gate=np.array([y.mean()]), nll=0., starts=1,
                        solver_diagnostics=[dict(success=True, gradient_inf=0.)])
        with patch.object(op, "train_gate", side_effect=fake_train):
            prior, _, diag = op.estimate_priors(self.x, self.y, 1., "C", True)
            fold = int(diag["folds"][0])
            y = self.y.copy()
            y[diag["folds"] == fold] += 1000
            changed, _, _ = op.estimate_priors(self.x, y, 1., "C", True)
        np.testing.assert_array_equal(prior[diag["folds"] == fold], changed[diag["folds"] == fold])
        for record in diag["training"]:
            self.assertEqual(len(np.intersect1d(record["train_indices"], record["validation_indices"])), 0)

    def test_geometry_valid_map_and_no_trimming(self):
        prior = np.c_[np.where(self.x[:, 1] > 0, .05, .95), np.where(self.x[:, 1] > 0, .95, .05)]
        model = op.fit_geometry(self.x, self.y, .5, prior, self.baseline, [self.baseline])
        self.assertEqual(model["posterior"].shape, (60, 2))
        np.testing.assert_allclose(model["posterior"].sum(1), 1.)
        self.assertLessEqual(model["means"][0], model["means"][1])
        levels = model["means"][None, :] + (self.x[:, 1:] @ model["slope"])[:, None]
        np.testing.assert_array_equal(model["prediction"], levels[np.arange(60), model["groups"]])
        self.assertEqual(model["geometry_starts"], 6)

    def test_single_surface_is_exact_passthrough(self):
        plane = np.array([1., .3, -.1])
        baseline = dict(k=1, family="S", means=plane[:1], slope=plane[1:],
                        prediction=self.x@plane, posterior=np.ones((60, 1)), groups=np.zeros(60, int))
        with patch.object(op, "train_gate", side_effect=AssertionError("must not train S")):
            outputs = op.run(self.x, self.y, 1., baseline)
        for model in outputs.values():
            np.testing.assert_array_equal(model["prediction"], baseline["prediction"])
            self.assertEqual(model["metadata"]["truth_fields_used"], [])

    def test_validation_rejects_bad_sigma(self):
        with self.assertRaises(ValueError):
            op.run(self.x, self.y, 0., self.baseline)


if __name__ == "__main__":
    unittest.main(verbosity=2)
