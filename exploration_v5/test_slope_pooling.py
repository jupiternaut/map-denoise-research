"""V5 bounded development/unit tests, not new confirmation data.

Conditional-model risk fixtures use known labels only to test the algebraic
constraint itself. Public estimate() never receives any of those labels or
truth coordinates. End-to-end fixtures use the already exposed seed 912101.

Public development smoke on liekkas (2026-09-12); sigma=1 mm, all points scored:
                              V4 / shared_group_slope / node_intercepts MAE mm
ghost_s912101_b0               .042599 / .042599 / .089791
dual_g2_s912101_b4             .197566 / .191002 / .215549
dual_g4_s912101_b4             .424661 / .616582 / .441342
dual_g8_s912101_b4             .325660 / .976779 / .331978
dual_g2_s912127_b4             .199979 / .234687 / .226669
The corresponding g2/g4/g8 seed912101 fitted same-XY gaps changed from
1.336464/3.666621/7.851612 to .596687/1.761917/2.746261 mm in the shared-slope
proposal. These are negative development results, not a claimed improvement.
All three fingerprints and supports agreed, all conditional fits were full
rank, and V4 output exactly matched the existing saved V4 development arrays.
One evaluator-only check found a g8 frozen group containing 38/92 points from
the two truth layers. Its steep fitted slope can contaminate the parallelism
constraint. Truth was used only to explain this failure, never by the operator.
Weighted group-centering independently reproduced the joint SVD predictions
to 7.11e-15 mm. No threshold, association or model was changed after this smoke.
Measured single-case cost was .15--.23 s including the V4 call/reconstruction.
"""
import hashlib
import inspect
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

import slope_pooling as method


def scene(kind="plane", gap=8., bias=True):
    rng = np.random.default_rng(912101)
    frames = np.repeat(np.arange(8), 96)
    xy = rng.uniform([-90., -65.], [90., 65.], (len(frames), 2))
    if kind == "dual":
        layer = rng.integers(0, 2, len(frames))
    elif kind == "step":
        layer = (xy[:, 0] > 0).astype(int)
    else:
        layer = np.zeros(len(frames), dtype=int)
    truth = np.c_[xy, gap * layer] / 1000.
    shifts = np.array([-5., 4., 2., -3., 5., -4., -2., 3.]) if bias else np.zeros(8)
    observed = truth.copy()
    observed[:, 2] += (shifts[frames] + rng.normal(0., 1., len(frames))) / 1000.
    return observed, frames, truth, layer


class SlopePoolingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in method.FROZEN_SOURCE_SHA256}

    @classmethod
    def tearDownClass(cls):
        for path, expected in cls.old_hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise AssertionError("frozen V4/V3 source modified: " + path)

    def test_api_readonly_and_frozen_state(self):
        self.assertEqual(list(inspect.signature(method.estimate).parameters),
                         ["xyz_world_m", "scan_id", "sigma_mm", "variant"])
        world, frames, _, _ = scene("dual", 8.)
        before, frame_before = world.copy(), frames.copy()
        results = [method.estimate(world, frames, 1., variant) for variant in method.VARIANTS]
        for out, info in results:
            json.dumps(info, allow_nan=False)
            self.assertEqual(out.shape, world.shape)
            self.assertTrue(np.isfinite(out).all())
            self.assertTrue(info["fit_full_rank"])
            self.assertGreater(info["fit_condition_number"], 0.)
            self.assertEqual(info["frozen_state_sha256"], results[0][1]["frozen_state_sha256"])
            self.assertEqual(info["unsupported_point_indices"], results[0][1]["unsupported_point_indices"])
            rejected = info["unsupported_point_indices"]
            np.testing.assert_array_equal(out[rejected], world[rejected])
        np.testing.assert_array_equal(world, before)
        np.testing.assert_array_equal(frames, frame_before)
        with self.assertRaises(ValueError):
            method.estimate(world, frames, 0.)

    def test_exact_v4_replication(self):
        world, frames, _, _ = scene("step", 8.)
        expected, _ = method._V4.estimate(world, frames, 1., variant="compatible")
        actual, _ = method.estimate(world, frames, 1., variant="v4_compatible")
        np.testing.assert_array_equal(actual, expected)

    def test_true_single_plane(self):
        world, frames, truth, _ = scene(bias=False)
        for variant in method.VARIANTS:
            out, info = method.estimate(world, frames, 1., variant)
            self.assertLess(np.mean(abs(out[:, 2] - truth[:, 2])) * 1000., .2)
            self.assertGreater(info["supported_fraction"], .9)

    def test_parallel_dual_and_step_outputs(self):
        for kind in ("dual", "step"):
            for gap in (2., 8.):
                world, frames, _, layer = scene(kind, gap)
                out, info = method.estimate(world, frames, 1., variant="shared_group_slope")
                slopes = np.asarray(info["group_slopes_common_xy"])
                np.testing.assert_array_equal(slopes, np.repeat(slopes[:1], len(slopes), axis=0))
                recovered = (out[layer == 1, 2].mean() - out[layer == 0, 2].mean()) * 1000.
                if gap == 8.:
                    self.assertGreater(recovered, 6.)
                self.assertTrue(np.isfinite(recovered))

    def test_rigid_origin_order_and_units(self):
        world, frames, _, _ = scene("step", 8.)
        base, info = method.estimate(world, frames, 1.)
        rotation = Rotation.from_euler("xyz", [23., -31., 57.], degrees=True).as_matrix()
        translation = np.array([2.37, -5.11, .79])
        moved, _ = method.estimate(world @ rotation.T + translation, frames, 1.)
        np.testing.assert_allclose((moved-translation) @ rotation, base, atol=1e-8, rtol=0.)
        order = np.random.default_rng(11).permutation(len(world))
        shuffled, _ = method.estimate(world[order], frames[order], 1.)
        np.testing.assert_allclose(shuffled[np.argsort(order)], base, atol=1e-10, rtol=0.)
        scaled, _ = method.estimate(world * 10., frames, 10.)
        np.testing.assert_allclose(scaled / 10., base, atol=2e-8, rtol=0.)
        normal = np.asarray(info["normal_world"])
        delta = base - world
        np.testing.assert_allclose(delta - np.outer(delta @ normal, normal), 0., atol=1e-12)

    def test_projected_cloud_is_not_an_observation(self):
        world, frames, _, _ = scene("step", 8.)
        baseline, info = method.estimate(world, frames, 1.)
        with mock.patch.object(method._V4, "estimate", return_value=(np.full_like(world, 1e6), info["v4_info"])):
            altered, changed_info = method.estimate(world, frames, 1.)
        np.testing.assert_array_equal(altered, baseline)
        self.assertEqual(info["frozen_state_sha256"], changed_info["frozen_state_sha256"])

    def test_nonparallel_and_wrong_group_risks_are_not_hidden(self):
        # Conditional model check, not an estimator GT/oracle option.
        xy = np.tile(np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]), (2, 1))
        nodes = np.repeat([0, 1], 4)
        values = np.r_[2.*xy[:4, 0], 8.-2.*xy[4:, 0]]
        mapping = np.array([0, 1])
        independent, _ = method._fit_model(xy, values, np.ones(8), nodes, mapping, "node_intercepts")
        forced, _ = method._fit_model(xy, values, np.ones(8), nodes, mapping, "shared_group_slope")
        self.assertLess(np.linalg.norm(independent-values), 1e-10)
        self.assertGreater(np.sqrt(np.mean((forced-values)**2)), 1.)
        # A mistaken SAME group forces one intercept in the main proposal;
        # preserving node intercepts can distinguish this from slope sharing.
        levels = np.repeat([0., 8.], 4)
        wrong_group = np.array([0, 0])
        collapsed, _ = method._fit_model(xy, levels, np.ones(8), nodes, wrong_group, "shared_group_slope")
        node_control, _ = method._fit_model(xy, levels, np.ones(8), nodes, wrong_group, "node_intercepts")
        self.assertLess(abs(collapsed[:4].mean()-collapsed[4:].mean()), 1e-10)
        np.testing.assert_allclose(node_control, levels, atol=1e-10)

    def test_shared_slope_matches_independent_intercept_elimination(self):
        rng = np.random.default_rng(912101)
        xy = rng.normal(size=(60, 2))
        groups = np.repeat(np.arange(3), 20)
        xy += np.c_[2.*groups, -.3*groups]
        values = .4*xy[:, 0] + .2*xy[:, 1] + 3.*groups + rng.normal(0., .2, 60)
        weights = rng.uniform(.4, 1., 60)
        predicted, _ = method._fit_model(xy, values, weights, groups, np.arange(3), "shared_group_slope")
        mx = np.array([np.average(xy[groups == j], axis=0, weights=weights[groups == j]) for j in range(3)])
        mz = np.array([np.average(values[groups == j], weights=weights[groups == j]) for j in range(3)])
        dx, dz = xy-mx[groups], values-mz[groups]
        slope = np.linalg.solve(dx.T @ (weights[:, None]*dx), dx.T @ (weights*dz))
        intercept = mz-mx@slope
        np.testing.assert_allclose(predicted, intercept[groups]+xy@slope, atol=1e-11, rtol=0.)

    def test_rank_deficiency_and_unsupported(self):
        _, info = method._fit_model(np.zeros((8, 2)), np.arange(8.), np.ones(8),
                                    np.repeat([0, 1], 4), np.array([0, 1]), "shared_group_slope")
        self.assertFalse(info["fit_full_rank"])
        self.assertIsNone(info["fit_condition_number"])
        json.dumps(info, allow_nan=False)
        world = np.arange(18).reshape(6, 3) / 1000.
        infos = []
        for variant in method.VARIANTS:
            out, meta = method.estimate(world, np.arange(6), 1., variant)
            np.testing.assert_array_equal(out, world)
            self.assertEqual(meta["unsupported_point_indices"], list(range(6)))
            self.assertEqual(meta["status"], "UNSUPPORTED")
            infos.append(meta)
        self.assertEqual(len({i["frozen_state_sha256"] for i in infos}), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
