"""Small exposed mechanism/contract checks, not a performance benchmark suite."""
import json
import unittest

import numpy as np

from graph_surface import estimate


def scene(kind='plane', seed=1307):
    rng = np.random.default_rng(seed)
    nf, per_frame = 8, 128
    scan = np.repeat(np.arange(nf), per_frame)
    xy = rng.uniform(-80., 80., (len(scan), 2))
    levels = np.zeros(len(scan))
    if kind == 'step':
        levels = 10. * (xy[:, 0] > 0)
    if kind == 'dual':
        levels = 10. * (np.arange(len(scan)) % 2)
    clean = np.c_[xy, levels]
    epsilon = rng.normal(0., .6, len(scan))
    bias = np.array([-5., 4., 2., -3., 5., -4., -2., 3.])
    observed = clean.copy()
    observed[:, 2] += epsilon + bias[scan]
    return observed / 1000., scan, clean / 1000., bias


class GraphSurfaceTests(unittest.TestCase):
    def test_contract_no_mutation_and_json(self):
        xyz, scan, _, _ = scene()
        before = xyz.copy()
        labels = scan.copy()
        for variant in ('graph', 'local_only'):
            out, info = estimate(xyz, scan, .6, variant)
            self.assertEqual(out.shape, xyz.shape)
            self.assertTrue(np.isfinite(out).all())
            self.assertEqual(info['point_count'], len(xyz))
            np.testing.assert_array_equal(xyz, before)
            np.testing.assert_array_equal(scan, labels)
            json.dumps(info, allow_nan=False)
        with self.assertRaises(ValueError):
            estimate(xyz, scan, 0.)

    def test_translation_equivariance_and_order(self):
        xyz, scan, _, _ = scene()
        translation = np.array([1.37, -2.89, .731])
        out, _ = estimate(xyz, scan, .6)
        shifted, _ = estimate(xyz + translation, scan, .6)
        np.testing.assert_allclose(shifted, out + translation, atol=2e-8, rtol=0.)
        # The two tangent components are untouched, which directly checks that
        # arbitrary row order was not replaced by a sorted point cloud.
        perm = np.random.default_rng(87).permutation(len(scan))
        shuffled, _ = estimate(xyz[perm], scan[perm], .6)
        np.testing.assert_allclose(shuffled, out[perm], atol=2e-5, rtol=0.)

    def test_connected_bias_loop_and_ablation(self):
        xyz, scan, clean, bias = scene()
        graph, info = estimate(xyz, scan, .6)
        local, local_info = estimate(xyz, scan, .6, 'local_only')
        estimated = np.asarray(info['bias_mm'])
        self.assertAlmostEqual(float(estimated.mean()), 0., places=7)
        self.assertLess(float(np.sqrt(np.mean((estimated - bias) ** 2))), .4)
        graph_error = float(np.mean(abs(graph[:, 2] - clean[:, 2])) * 1000)
        local_error = float(np.mean(abs(local[:, 2] - clean[:, 2])) * 1000)
        self.assertLess(graph_error, .5)
        self.assertLess(graph_error, local_error)
        self.assertEqual(info['grid_shape'], local_info['grid_shape'])
        np.testing.assert_array_equal(info['normal_world'], local_info['normal_world'])

    def test_step_and_dual_are_not_replaced_by_global_ramp(self):
        for kind in ('step', 'dual'):
            xyz, scan, clean, _ = scene(kind)
            out, info = estimate(xyz, scan, .6)
            low, high = clean[:, 2] == 0., clean[:, 2] > 0.
            gap = float((out[high, 2].mean() - out[low, 2].mean()) * 1000)
            error = float(np.mean(abs(out[:, 2] - clean[:, 2])) * 1000)
            self.assertGreater(gap, 8.)
            self.assertLess(error, 1.5)
            self.assertGreater(info['supported_fraction'], .5)

    def test_unsupported_small_input_is_exact_identity(self):
        xyz = np.arange(18, dtype=float).reshape(6, 3) / 1000.
        output, info = estimate(xyz, np.zeros(6, dtype=int), 1.)
        np.testing.assert_array_equal(output, xyz)
        self.assertEqual(info['status'], 'UNSUPPORTED')

    def test_unequal_scan_counts_keep_explicit_weighted_gauge(self):
        xyz, scan, _, _ = scene()
        take = ~((scan == 0) & (np.arange(len(scan)) % 2 == 0))
        out, info = estimate(xyz[take], scan[take], .6)
        self.assertTrue(np.isfinite(out).all())
        self.assertAlmostEqual(float(np.average(info['bias_mm'], weights=info['gauge_point_counts'])), 0., places=7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
