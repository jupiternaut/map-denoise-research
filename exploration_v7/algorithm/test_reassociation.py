"""Small legal-input, EM-objective and frozen-output checks; no data GT."""
import inspect
import importlib.util
import unittest

import numpy as np

import reassociation as R


def measured_cloud():
    rng = np.random.default_rng(711)
    scan = np.repeat(np.arange(4), 128)
    xy = rng.uniform(-.08, .08, (len(scan), 2))
    # Input-only fixture generation, never labels passed to the estimator.
    z = ((np.arange(len(scan)) % 2)*.008 + (scan-1.5)*.001 + rng.normal(0, .0005, len(scan)))
    return np.c_[xy, z], scan


class ReassociationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world, cls.scan = measured_cloud()
        cls.state = R.freeze(cls.world, cls.scan, 1.)

    def test_public_api_has_no_evaluator_fields(self):
        self.assertEqual(tuple(inspect.signature(R.freeze).parameters), ("xyz_world_m", "scan_id", "sigma_mm"))
        self.assertEqual(tuple(inspect.signature(R.estimate).parameters),
                         ("xyz_world_m", "scan_id", "sigma_mm", "variant", "budget", "sharing"))
        self.assertEqual(tuple(inspect.signature(R.fit_frozen).parameters), ("state", "variant", "budget", "sharing"))

    def test_frozen_owned_input_and_support(self):
        original = self.world.copy()
        for variant in R.VARIANTS:
            out, info, arrays = R.fit_frozen(self.state, variant, 3)
            np.testing.assert_array_equal(self.world, original)
            np.testing.assert_array_equal(out[~arrays["support_mask"]], original[~arrays["support_mask"]])
            self.assertFalse(self.state["world"].flags.writeable)
            self.assertEqual(out.shape, self.world.shape)

    def test_mass_and_accessibility(self):
        for variant in R.VARIANTS:
            _, _, a = R.fit_frozen(self.state, variant, 3)
            r, active = a["responsibility"], a["active_mask"]
            np.testing.assert_allclose(r[active].sum(axis=1), 1., atol=1e-12, rtol=0)
            self.assertTrue(np.all(r[~a["candidate_mask"]] == 0))
            self.assertTrue(np.all(r[~active] == 0))

    def test_half_step_objective_nonincrease(self):
        _, info, arrays = R.fit_frozen(self.state, "reassociate", 6)
        trace = arrays["objective_trace"]
        self.assertGreater(len(trace), 2)
        self.assertTrue(np.all(np.diff(trace) <= 1e-8*np.maximum(1., abs(trace[:-1]))))
        self.assertEqual(info["objective_steps"][1]["step"], "M")

    def test_deterministic(self):
        for variant in R.VARIANTS:
            a, _, aa = R.fit_frozen(self.state, variant, 3)
            b, _, bb = R.fit_frozen(self.state, variant, 3)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(aa["group_ids"], bb["group_ids"])
            np.testing.assert_array_equal(aa["responsibility"], bb["responsibility"])

    def test_original_same_wls_and_same_shared_as_v5(self):
        for sharing in R.SHARING:
            out, _, _ = R.fit_frozen(self.state, "original", 0, sharing)
            groups = self.state["original_groups"]
            active = self.state["active"]
            predicted, _ = R.V5._fit_model(self.state["design"][active, 1:], self.state["corrected"][active],
                self.state["weights"][active], groups, np.arange(groups.max()+1),
                "shared_group_slope" if sharing == "shared" else "node_intercepts")
            ordered = self.state["ordered_world"].copy()
            all_prediction = self.state["corrected"].copy(); all_prediction[active] = predicted
            use = self.state["support"]
            ordered[use] += ((all_prediction[use]-self.state["local"][use, 2])/1000.)[:, None]*self.state["normal"]
            expected = self.world.copy(); expected[self.state["order"]] = ordered
            np.testing.assert_array_equal(out, expected)

    def test_original_matches_independent_v6_replay(self):
        path = R.PROJECT / "exploration_v6/association/association_counterfactual.py"
        spec = importlib.util.spec_from_file_location("_test_only_v6_original", path)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        state = old.freeze(self.world, self.scan, 1.)
        for sharing in R.SHARING:
            expected, _, expected_groups = old.fit_original(state, sharing)
            output, _, arrays = R.fit_frozen(self.state, "original", 0, sharing)
            np.testing.assert_array_equal(output, expected)
            np.testing.assert_array_equal(arrays["group_ids"], expected_groups)

    def test_unsupported_is_identity(self):
        world = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.], [.01, .01, .001]])
        state = R.freeze(world, np.array([0, 0, 1, 1]), 1.)
        for variant in R.VARIANTS:
            out, info, arrays = R.fit_frozen(state, variant, 3)
            self.assertEqual(info["status"], "UNSUPPORTED")
            self.assertFalse(arrays["support_mask"].any())
            np.testing.assert_array_equal(out, world)

    def test_mass_preserving_clone_invariant_uniform_clone_not_invariant(self):
        x = np.array([[1., 0., 0.]])
        y = np.array([0.])
        beta = np.array([[-2., 0., 0.], [2., 0., 0.]])
        r, _, nll = R._e_step(x, y, beta, np.ones((1, 2), bool), 1., R._counter(), prior=np.array([.5, .5]))
        cloned = beta[[0, 0, 1]]
        rr, _, nn = R._e_step(x, y, cloned, np.ones((1, 3), bool), 1., R._counter(), prior=np.array([.25, .25, .5]))
        np.testing.assert_allclose([rr[0, :2].sum(), rr[0, 2]], r[0], atol=1e-15)
        np.testing.assert_allclose(nll, nn, atol=1e-15)
        uniform, _, _ = R._e_step(x, y, cloned, np.ones((1, 3), bool), 1., R._counter())
        self.assertGreater(uniform[0, :2].sum(), r[0, 0])

    def test_hard_action_does_not_average_into_gap(self):
        r = np.array([[.5, .5]])
        group, _, margin = R._hard_assignment(r, np.ones_like(r, bool), np.array([1]))
        self.assertEqual(group[0], 1)
        self.assertEqual(margin[0], 0.)
        predictions = np.array([-2., 2.])
        self.assertEqual(predictions[group[0]], 2.)
        self.assertEqual(r @ predictions, 0.)


if __name__ == "__main__":
    unittest.main()
