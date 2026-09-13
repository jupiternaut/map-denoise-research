"""Bounded synthetic mechanism tests; no evaluation enters estimate().

Exposed development diagnostic, liekkas, 2026-09-12, fixed sigma=1 mm:
The original repair-v2-oyuie4pl/identifiable inputs below were read without
modification. No threshold was changed after inspecting this small diagnostic.
All point outputs were scored; support was identical across the three arms.
                    independent/global/compatible surface MAE (mm)
ghost_s912101_b0     0.162394 / 0.042599 / 0.042599
ghost_s912101_b4     0.162575 / 0.042575 / 0.042575
dual_g2_s912101_b4   0.257021 / 0.468886 / 0.197566
dual_g4_s912101_b4   0.494837 / 0.987843 / 0.424661
dual_g8_s912101_b4   0.357473 / 1.747099 / 0.325660
Compatible same-XY affine-fit gaps were 1.336464, 3.666621, 7.851612 mm;
these do not establish complete structure recovery. Single-case runtimes were
0.12--0.20 s. This is development evidence, not an independent confirmation.
V4 independent is a matched V4 arm, NOT an exact reproduction of V3 output:
it uses canonical input ordering and frozen-assignment raw-point plane refits.
"""
import inspect
import json
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

import surface_pooling as pooling
from surface_pooling import estimate, _pool


def scene(kind="plane", gap=8., bias=True, seed=2309):
    rng = np.random.default_rng(seed)
    frames = np.repeat(np.arange(8), 128)
    xy = rng.uniform([-90., -65.], [90., 65.], (len(frames), 2))
    if kind == "dual":
        layer = rng.integers(0, 2, len(frames))
    elif kind == "step":
        layer = (xy[:, 0] > 0).astype(int)
    else:
        layer = np.zeros(len(frames), dtype=int)
    truth = np.c_[xy, gap * layer] / 1000.
    shifts = np.array([-5., 4., 2., -3., 5., -4., -2., 3.]) if bias else np.zeros(8)
    world = truth.copy()
    world[:, 2] += (shifts[frames] + rng.normal(0., 1., len(frames))) / 1000.
    return world, frames, truth, layer


class SurfacePoolingTests(unittest.TestCase):
    def test_contract_no_truth_inputs(self):
        self.assertEqual(list(inspect.signature(estimate).parameters), ["xyz_world_m", "scan_id", "sigma_mm", "variant"])
        world, frames, _, _ = scene()
        original, frame_copy = world.copy(), frames.copy()
        infos = []
        for variant in ("independent", "global", "compatible"):
            out, info = estimate(world, frames, 1., variant)
            self.assertEqual(out.shape, world.shape)
            self.assertTrue(np.isfinite(out).all())
            json.dumps(info, allow_nan=False)
            infos.append(info)
        np.testing.assert_array_equal(world, original)
        np.testing.assert_array_equal(frames, frame_copy)
        for info in infos[1:]:
            np.testing.assert_array_equal(info["bias_mm"], infos[0]["bias_mm"])
            np.testing.assert_array_equal(info["normal_world"], infos[0]["normal_world"])
            self.assertEqual(info["supported_fraction"], infos[0]["supported_fraction"])
        with self.assertRaises(ValueError):
            estimate(world, frames, 0.)

    def test_single_plane_pooling(self):
        for bias in (False, True):
            world, frames, truth, _ = scene(bias=bias)
            independent, _ = estimate(world, frames, 1., "independent")
            pooled, info = estimate(world, frames, 1., "compatible")
            self.assertEqual(info["pooled_surface_count"], 1)
            self.assertTrue(info["complete_link_verified"])
            self.assertLess(np.mean(abs(pooled[:, 2] - truth[:, 2])), np.mean(abs(independent[:, 2] - truth[:, 2])))
            self.assertLess(np.mean(abs(pooled[:, 2] - truth[:, 2])) * 1000., .2)

    def test_dual_and_step(self):
        for kind in ("dual", "step"):
            for gap in (2., 8.):
                world, frames, truth, layer = scene(kind, gap)
                pooled, info = estimate(world, frames, 1., "compatible")
                global_fit, _ = estimate(world, frames, 1., "global")
                z = pooled[:, 2] * 1000.
                recovered_gap = z[layer == 1].mean() - z[layer == 0].mean()
                self.assertTrue(info["complete_link_verified"])
                for group in info["group_node_ids"]:
                    cells = [info["node_cells"][j] for j in group]
                    self.assertEqual(len(cells), len(set(cells)))
                self.assertGreater(recovered_gap, .45 * gap)
                self.assertGreaterEqual(info["pooled_surface_count"], 2)
                if kind == "dual" and gap == 8.:
                    self.assertLess(np.mean(abs(pooled[:, 2] - truth[:, 2])), np.mean(abs(global_fit[:, 2] - truth[:, 2])))

    def test_projected_xyz_are_not_observations(self):
        world, frames, _, _ = scene("step", 8.)
        baseline, info = estimate(world, frames, 1.)
        # Holding legal initial diagnostics fixed, arbitrarily corrupting the
        # discarded V3 projection must not alter the pooling reconstruction.
        with mock.patch.object(pooling._V3, "estimate", return_value=(np.full_like(world, 1e6), info["initial_info"])):
            altered, _ = estimate(world, frames, 1.)
        np.testing.assert_allclose(altered, baseline, atol=1e-12, rtol=0.)

    def test_rigid_origin_and_point_order(self):
        world, frames, _, _ = scene("step", 8.)
        base, base_info = estimate(world, frames, 1.)
        rotation = Rotation.from_euler("xyz", [23., -31., 57.], degrees=True).as_matrix()
        translation = np.array([2.37, -5.11, .79])
        moved, info = estimate(world @ rotation.T + translation, frames, 1.)
        np.testing.assert_allclose((moved - translation) @ rotation, base, atol=1e-8, rtol=0.)
        self.assertEqual(base_info["pooled_surface_count"], info["pooled_surface_count"])
        order = np.random.default_rng(451).permutation(len(world))
        permuted, _ = estimate(world[order], frames[order], 1.)
        np.testing.assert_allclose(permuted[np.argsort(order)], base, atol=1e-10, rtol=0.)
        # Every displacement is normal-only, and output rows correspond to input.
        normal = np.asarray(base_info["normal_world"])
        delta = base - world
        tangent = delta - np.outer(delta @ normal, normal)
        np.testing.assert_allclose(tangent, 0., atol=1e-12)

    def test_no_chain_merging(self):
        nodes = []
        for j in range(3):
            nodes.append({"indices": np.array([j]), "lhs": np.eye(3), "rhs": np.zeros(3),
                          "yy": 0., "beta": np.zeros(3), "cov": np.eye(3), "sse": 0., "rank": 3})
        # A~B and B~C, but A is incompatible with C: no transitive collapse.
        allowed = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)
        groups, _ = _pool(nodes, allowed, 1., 100)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(allowed[np.ix_(g["members"], g["members"])].all() for g in groups))

    def test_unsupported_exact_identity(self):
        world = np.arange(18).reshape(6, 3) / 1000.
        for variant in ("independent", "global", "compatible"):
            out, info = estimate(world, np.arange(6), 1., variant)
            np.testing.assert_array_equal(out, world)
            self.assertEqual(info["status"], "UNSUPPORTED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
