"""Contract tests on already exposed inputs; no new samples or fitted gates."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import association_counterfactual as model
from run_association import CASES, INPUTS, PRIMARY, RUNS


def load(case):
    with np.load(INPUTS/(case+".npz"), allow_pickle=False) as data:
        world, scans = data["xyz_world"].copy(), data["scan_id"].copy()
    with np.load(INPUTS/"evaluation"/(case+".eval.npz"), allow_pickle=False) as data:
        labels = data["gt_layer"].copy()
    return world, scans, labels


class AssociationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = (model.V5_PATH, model.V5._V4_PATH, model.V5._V4._V3_PATH)
        cls.hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.sources}
        cls.cases = {}
        for case in PRIMARY:
            world, scans, labels = load(case)
            cls.cases[case] = (world, scans, labels, model.freeze(world, scans, 1.))

    @classmethod
    def tearDownClass(cls):
        assert cls.hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.sources}

    def test_01_bounded_cases_and_no_gt_original_api(self):
        self.assertEqual(len(CASES), 24)
        self.assertEqual(len(set(CASES)), 24)
        self.assertTrue(set(PRIMARY).issubset(CASES))
        self.assertEqual(list(inspect.signature(model.freeze).parameters), ["xyz_world_m", "scan_id", "sigma_mm"])
        self.assertEqual(list(inspect.signature(model.fit_original).parameters), ["state", "sharing"])
        for _, _, _, state in self.cases.values():
            with patch.object(np, "load", side_effect=AssertionError("estimator must not read files")):
                for sharing in model.SHARING:
                    _, info, _ = model.fit_original(state, sharing)
                    self.assertEqual(info["truth_fields_used"], [])

    def test_02_original_replays_v4_v5_saved_outputs(self):
        for case, (_, _, _, state) in self.cases.items():
            for sharing, method, run in (("independent", "pool_compatible", "exploration-v4-dev-e0c_n4hk"),
                                         ("shared", "shared_group_slope", "exploration-v5-development-vdbkdouk")):
                output, _, _ = model.fit_original(state, sharing)
                with np.load(RUNS/run/"outputs"/(case+"__"+method+".npz")) as data:
                    self.assertLessEqual(float(np.max(abs(output-data["xyz_world"]))), 1e-12)

    def test_03_fixed_state_weights_active_support_and_readonly(self):
        for world, scans, labels, state in self.cases.values():
            before = model.common_fingerprints(state)
            expected_weight = float(state["weights"][state["active"]].sum())
            support = np.empty(len(world), bool)
            support[state["order"]] = state["support"]
            for sharing in model.SHARING:
                results = (model.fit_original(state, sharing), model.fit_oracle_split(state, labels, sharing))
                for output, info, assignment in results:
                    self.assertEqual(info["common_fingerprints"], before)
                    self.assertEqual(info["total_fit_weight"], expected_weight)
                    self.assertEqual(info["active_point_count"], len(state["active"]))
                    np.testing.assert_array_equal(output[~support], world[~support])
                    np.testing.assert_array_equal(np.flatnonzero(assignment >= 0), np.sort(state["order"][state["active"]]))
                    self.assertEqual(output.shape, world.shape)
                    self.assertTrue(np.isfinite(output).all())
                    json.dumps(info, allow_nan=False)
            self.assertEqual(before, model.common_fingerprints(state))
            np.testing.assert_array_equal(world, state["world"])
            np.testing.assert_array_equal(scans, state["scan_id"])
            self.assertFalse(state["weights"].flags.writeable)

    def test_04_split_only_and_label_renaming(self):
        for _, _, labels, state in self.cases.values():
            original = model.fit_original(state)[2]
            for sharing in model.SHARING:
                out, info, groups = model.fit_oracle_split(state, labels, sharing)
                renamed = model.fit_oracle_split(state, 11-7*labels, sharing)[0]
                np.testing.assert_allclose(out, renamed, atol=1e-12, rtol=0)
                for group in range(info["group_count"]):
                    take = groups == group
                    self.assertEqual(len(np.unique(original[take])), 1)
                    self.assertEqual(len(np.unique(labels[take])), 1)
                g = info["group_count"]
                self.assertEqual(info["fit_parameter_count"], 3*g if sharing == "independent" else g+2)

    def test_05_single_label_oracle_is_noop(self):
        for world, _, _, state in self.cases.values():
            for sharing in model.SHARING:
                original = model.fit_original(state, sharing)[0]
                oracle = model.fit_oracle_split(state, np.zeros(len(world), dtype=int), sharing)[0]
                np.testing.assert_array_equal(original, oracle)

    def test_06_order_rigid_coordinates_and_units(self):
        world, scans, labels, state = self.cases[PRIMARY[0]]
        angle = .43
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        translation = np.array([3.5, -1.2, .6])
        order = np.arange(len(world))[::-1]
        changed = model.freeze(world[order]@rotation.T+translation, scans[order], 1.)
        scaled = model.freeze(world*2, scans, 2.)
        for sharing in model.SHARING:
            for oracle in (False, True):
                base = model.fit_oracle_split(state, labels, sharing)[0] if oracle else model.fit_original(state, sharing)[0]
                moved = model.fit_oracle_split(changed, labels[order], sharing)[0] if oracle else model.fit_original(changed, sharing)[0]
                doubled = model.fit_oracle_split(scaled, labels, sharing)[0] if oracle else model.fit_original(scaled, sharing)[0]
                np.testing.assert_allclose(moved, base[order]@rotation.T+translation, atol=1e-9, rtol=0)
                np.testing.assert_allclose(doubled, 2*base, atol=1e-12, rtol=0)

    def test_07_oracle_does_not_mutate_state_or_original_results(self):
        _, _, labels, state = self.cases[PRIMARY[0]]
        original = model.fit_original(state, "shared")[0]
        model.fit_oracle_split(state, 1-labels, "shared")
        np.testing.assert_array_equal(original, model.fit_original(state, "shared")[0])
        with self.assertRaises(ValueError):
            model.fit_oracle_split(state, labels[:-1])
        with self.assertRaises(ValueError):
            model.fit_oracle_split(state, labels.astype(float))
        with self.assertRaises(ValueError):
            model.fit_original(state, "unknown")

    def test_08_known_weighted_parallel_model(self):
        xy = np.array([[-2., -1.], [-2., 1.], [1., -1.], [1., 1.], [-1., -2.], [1., -2.], [-1., 2.], [1., 2.]])
        groups = np.repeat([0, 1], 4)
        z = np.array([-2., 3.])[groups]+xy@np.array([.1, -.2])
        weights = np.arange(1., 9.)
        for variant in ("node_intercepts", "shared_group_slope"):
            prediction, fit = model.V5._fit_model(xy, z, weights, groups, np.arange(2), variant)
            np.testing.assert_allclose(prediction, z, atol=1e-12, rtol=0)
            self.assertTrue(fit["fit_full_rank"])

    def test_09_projection_poison_not_an_observation(self):
        _, _, labels, state = self.cases[PRIMARY[0]]
        poisoned = dict(state)
        poisoned["baseline"] = np.full_like(state["baseline"], 12345.)
        for sharing in model.SHARING:
            np.testing.assert_array_equal(model.fit_original(state, sharing)[0], model.fit_original(poisoned, sharing)[0])
            np.testing.assert_array_equal(model.fit_oracle_split(state, labels, sharing)[0], model.fit_oracle_split(poisoned, labels, sharing)[0])

    def test_10_rank_deficient_design_is_explicit(self):
        xy = np.column_stack((np.arange(6.), np.zeros(6)))
        values = 2.+.1*xy[:, 0]
        for variant in ("node_intercepts", "shared_group_slope"):
            prediction, fit = model.V5._fit_model(xy, values, np.ones(6), np.zeros(6, dtype=int), np.arange(1), variant)
            self.assertEqual(fit["fit_rank"], 2)
            self.assertEqual(fit["fit_parameter_count"], 3)
            self.assertFalse(fit["fit_full_rank"])
            self.assertIsNone(fit["fit_condition_number"])
            np.testing.assert_allclose(prediction, values, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
