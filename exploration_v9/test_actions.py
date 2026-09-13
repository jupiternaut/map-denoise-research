"""Algebraic action checks, not a geometric performance benchmark."""
import importlib.util
import inspect
from pathlib import Path
import unittest

import numpy as np

from action_ablation import action_frozen, V8, PROJECT


class ActionTests(unittest.TestCase):
    @staticmethod
    def fixture(rank_deficient=False):
        rng = np.random.default_rng(913913)
        n = 180
        order = rng.permutation(n)
        xy = rng.uniform(-10, 10, (n, 2))
        if rank_deficient:
            xy[:, 1] = 2.*xy[:, 0] + 3.
        local = np.c_[xy, .15*xy[:, 0] + rng.normal(size=n)]
        world = np.empty((n, 3)); world[order] = local/1000.
        active = np.flatnonzero(np.arange(n) % 11 != 0)
        support = np.zeros(n, bool); support[active] = True; support[::7] = False
        groups = (local[:, 2] >= .15*xy[:, 0]).astype(int)
        world_groups = np.full(n, -1, int); world_groups[order[active]] = groups[active]
        state = dict(world=world, ordered_world=world[order].copy(), order=order,
            active=active, support=support, normal=np.array([0., 0., 1.]), local=local,
            design=np.c_[np.ones(n), xy/10.], corrected=local[:, 2].copy(),
            weights=rng.uniform(.2, 2., n))
        artifact = dict(group_ids=world_groups, coefficients=np.array([[-.2, 1., .3], [.2, 2., -.2]]),
            candidate_mask=np.ones((n, 2), bool))
        return state, artifact

    def test_full_matches_v8(self):
        s, a = self.fixture()
        old, _, old_a = V8.compensate_frozen(s, a, 1., "conditional")
        new, _, new_a = action_frozen(s, a, 1., "full")
        np.testing.assert_allclose(new, old, atol=1e-15, rtol=0)
        np.testing.assert_allclose(new_a["subtracted_noise_mm"], old_a["subtracted_noise_mm"], atol=0, rtol=0)

    def test_none_matches_v7_hard_fit(self):
        path = PROJECT / "exploration_v7/algorithm/reassociation.py"
        spec = importlib.util.spec_from_file_location("_v9_test_v7", path)
        v7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v7)
        s, a = self.fixture()
        groups = a["group_ids"][s["order"][s["active"]]]
        old, _ = v7._final_fit(s, groups, "independent")
        new, _, _ = action_frozen(s, a, 1., "none")
        np.testing.assert_allclose(new, old, atol=1e-14, rtol=0)

    def test_constant_leaves_base_slopes_exact(self):
        for deficient in (False, True):
            s, a = self.fixture(deficient)
            _, _, none = action_frozen(s, a, 1., "none")
            _, _, const = action_frozen(s, a, 1., "constant")
            np.testing.assert_array_equal(const["coefficients"][:, 1:], none["coefficients"][:, 1:])
            np.testing.assert_allclose(const["coefficients"][:, 0],
                none["coefficients"][:, 0]-const["group_constant_values_mm"], atol=0, rtol=0)

    def test_displacement_additivity(self):
        for deficient in (False, True):
            s, a = self.fixture(deficient)
            outputs = {m: action_frozen(s, a, 1., m)[0] for m in ("none", "constant", "slope", "full")}
            np.testing.assert_allclose(outputs["full"]-outputs["none"],
                outputs["constant"]+outputs["slope"]-2.*outputs["none"], atol=1e-14, rtol=0)

    def test_slope_correction_is_zero_weighted_mean(self):
        s, a = self.fixture()
        _, _, z = action_frozen(s, a, 1., "slope")
        rows = s["order"][s["active"]]; g = a["group_ids"][rows]
        for group in np.unique(g):
            use = g == group
            self.assertAlmostEqual(np.dot(s["weights"][s["active"]][use], z["subtracted_noise_mm"][rows][use]), 0., places=12)

    def test_slope_preserves_fitted_group_weighted_mean(self):
        s, a = self.fixture()
        _, _, none = action_frozen(s, a, 1., "none")
        _, _, slope = action_frozen(s, a, 1., "slope")
        groups = a["group_ids"][s["order"][s["active"]]]
        for index, group in enumerate(none["coefficient_ids"]):
            take = groups == group
            x = s["design"][s["active"]][take]; w = s["weights"][s["active"]][take]
            difference = x @ (slope["coefficients"][index]-none["coefficients"][index])
            self.assertAlmostEqual(np.dot(w, difference), 0., places=11)

    def test_inputs_labels_support_unchanged(self):
        s, a = self.fixture(); bs = {k: v.copy() for k, v in s.items()}; ba = {k: v.copy() for k, v in a.items()}
        for mode in ("none", "constant", "slope", "full"):
            out, info, z = action_frozen(s, a, 1., mode)
            np.testing.assert_array_equal(z["group_ids"], a["group_ids"])
            np.testing.assert_array_equal(out[~z["support_mask"]], s["world"][~z["support_mask"]])
            self.assertEqual(info["truth_fields_used"], [])
        for k in s: np.testing.assert_array_equal(s[k], bs[k])
        for k in a: np.testing.assert_array_equal(a[k], ba[k])

    def test_determinism(self):
        s, a = self.fixture()
        for mode in ("none", "constant", "slope", "full"):
            one, _, _ = action_frozen(s, a, 1., mode); two, _, _ = action_frozen(s, a, 1., mode)
            np.testing.assert_array_equal(one, two)

    def test_unsupported_identity(self):
        s, a = self.fixture(); s["active"] = np.empty(0, int); s["support"][:] = False
        out, info, artifact = action_frozen(s, a, 1.)
        np.testing.assert_array_equal(out, s["world"])
        self.assertEqual(info["status"], "UNSUPPORTED")
        self.assertEqual(artifact["coefficients"].shape, (0, 3))

    def test_invalid_and_public_interface(self):
        s, a = self.fixture()
        with self.assertRaises(ValueError): action_frozen(s, a, 0.)
        with self.assertRaises(ValueError): action_frozen(s, a, 1., "winner")
        self.assertEqual(list(inspect.signature(action_frozen).parameters), ["state", "artifacts", "sigma_mm", "mode"])


if __name__ == "__main__": unittest.main()
