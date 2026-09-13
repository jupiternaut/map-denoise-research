import copy
import importlib.util
import itertools
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("v19_track_b_operator", Path(__file__).with_name("operator.py"))
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)


class Tests(unittest.TestCase):
    def setUp(self):
        self.x = np.c_[np.ones(6), np.linspace(-1, 1, 6), np.zeros(6)]
        self.y = np.arange(6.)
        self.q = np.array([.03, .15, .31, .48, .6, .9])
        self.model = dict(k=2, means=np.array([-1., 1.]), slope=np.array([.2, .1]),
                          posterior=np.c_[1-self.q, self.q], gate=np.zeros(1), bic=25.)

    def test_unit_mass_and_surface_support(self):
        outputs = op.run(self.x, self.y, 2., self.model)
        w = outputs["weighted_measure"]
        np.testing.assert_allclose(w["support_weights"].sum(1), 1.)
        self.assertAlmostEqual(float(w["support_weights"].sum()/len(self.y)), 1.)
        surfaces = self.model["means"][None, :] + (self.x[:, 1:] @ self.model["slope"])[:, None]
        np.testing.assert_array_equal(w["support_heights"], surfaces)
        self.assertIsNone(w["prediction"])
        for name in ("map", "quota_map", "posterior_draw"):
            self.assertTrue(np.all(outputs[name]["output_distance_to_fitted_surface_mm"] == 0))

    def test_quota_matches_enumerated_optimum(self):
        a, m, expected = op.quota_assignment(self.q)
        self.assertEqual(a.sum(), m)
        self.assertEqual(m, int(np.floor(self.q.sum()+.5)))
        self.assertAlmostEqual(expected, self.q.sum())
        risk = lambda b: float(np.sum(np.where(b, 1-self.q, self.q)))
        possibilities = [np.array(t) for t in itertools.product((0, 1), repeat=6) if sum(t) == m]
        self.assertAlmostEqual(risk(a), min(map(risk, possibilities)))

    def test_tie_and_rounding_are_explicit(self):
        a, m, _ = op.quota_assignment(np.full(5, .5))
        self.assertEqual(m, 3)
        np.testing.assert_array_equal(a, [1, 1, 1, 0, 0])

    def test_one_surface_agreement(self):
        single = dict(k=1, means=np.array([.3]), slope=np.array([.2, .1]), posterior=np.ones((6, 1)))
        out = op.run(self.x, self.y, 2., single)
        for name in ("mean", "quota_map", "posterior_draw"):
            np.testing.assert_array_equal(out[name]["prediction"], out["map"]["prediction"])
        np.testing.assert_array_equal(out["weighted_measure"]["support_heights"][:, 0], out["map"]["prediction"])

    def test_input_model_unmodified_and_truth_free(self):
        before = copy.deepcopy(self.model)
        out = op.run(self.x, self.y, 2., self.model)
        for key, value in before.items():
            np.testing.assert_array_equal(self.model[key], value)
        for model in out.values():
            self.assertFalse(model["fit_changed"])
            self.assertEqual(model["truth_fields_used"], [])
            np.testing.assert_array_equal(model["posterior"], before["posterior"])
            np.testing.assert_array_equal(model["means"], before["means"])
        second = op.run(self.x, self.y+123., 100., self.model)
        for name in ("map", "mean", "quota_map", "posterior_draw"):
            np.testing.assert_array_equal(second[name]["prediction"], out[name]["prediction"])

    def test_invalid_probabilities_rejected_not_silently_changed(self):
        bad = dict(self.model, posterior=self.model["posterior"] * .9)
        with self.assertRaises(ValueError):
            op.run(self.x, self.y, 2., bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
