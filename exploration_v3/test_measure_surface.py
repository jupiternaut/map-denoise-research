"""Small construction checks, not reserved-seed denoising benchmarks."""
import json
import unittest

import numpy as np

from measure_surface import PARAMETERS, _transport, estimate


class SurfaceMeasureTests(unittest.TestCase):
    def test_mass_relaxation_is_real_and_balanced_marginals_are_checked(self):
        cost = np.array([[0., 0.], [20., 20.]])
        mass = np.full(2, .5)
        balanced, binfo = _transport(cost, mass, mass, 'balanced')
        relaxed, uinfo = _transport(cost, mass, mass, 'unbalanced')
        np.testing.assert_allclose(balanced.sum(1), mass, atol=1e-12)
        np.testing.assert_allclose(balanced.sum(0), mass, atol=1e-12)
        self.assertLess(relaxed[1].sum(), relaxed[0].sum()*.01)
        self.assertLess(uinfo['transport_mass'], binfo['transport_mass'])

    def test_common_source_correction_keeps_points_order_and_layer_step(self):
        # Every scan observes the same two finite parallel sheets. The truth
        # below is used by this unit check only, never passed into estimate.
        rng = np.random.default_rng(5021)
        xy = rng.uniform(-.06, .06, (48, 2))
        labels = np.arange(48) % 2
        base = np.column_stack((xy, labels*.012))
        offsets = np.array([-1.5, -.5, .5, 1.5])/1000.
        xyz = np.concatenate([base+np.array([0., 0., d]) for d in offsets])
        scan = np.repeat([11, 24, 39, 56], len(base))
        before = xyz.copy()
        before_scan = scan.copy()
        after, info = estimate(xyz, scan, .8, 'unbalanced')
        self.assertEqual(after.shape, xyz.shape)
        self.assertTrue(np.isfinite(after).all())
        np.testing.assert_array_equal(xyz, before)
        np.testing.assert_array_equal(scan, before_scan)
        for fid in np.unique(scan):
            delta = after[scan == fid]-xyz[scan == fid]
            np.testing.assert_allclose(delta, np.broadcast_to(delta[0], delta.shape), atol=1e-14)
            out_gap = after[scan == fid][labels == 1].mean(0)-after[scan == fid][labels == 0].mean(0)
            base_gap = base[labels == 1].mean(0)-base[labels == 0].mean(0)
            np.testing.assert_allclose(out_gap, base_gap, atol=1e-14)
        before_spread = np.std([xyz[scan == fid, 2].mean() for fid in np.unique(scan)])
        after_spread = np.std([after[scan == fid, 2].mean() for fid in np.unique(scan)])
        self.assertLess(after_spread, before_spread)
        self.assertGreater(info['moved_point_fraction'], 0.)
        json.dumps(info, allow_nan=False)
        _, balanced = estimate(xyz, scan, .8, 'balanced')
        self.assertEqual(info['parameters'], balanced['parameters'])
        self.assertEqual(info['surface_element_counts'], balanced['surface_element_counts'])
        np.testing.assert_allclose(info['normal_world'], balanced['normal_world'], atol=0.)
        self.assertEqual(len(info['outer_history']), PARAMETERS['outer_iterations'])

    def test_unmatched_scan_stays_whole_and_unchanged(self):
        rng = np.random.default_rng(5023)
        base = np.column_stack((rng.uniform(-.05, .05, (40, 2)), np.zeros(40)))
        xyz = np.concatenate((base+[0., 0., -.001], base+[0., 0., .001], base+[10., 0., .4]))
        scan = np.repeat([0, 1, 2], 40)
        after, info = estimate(xyz, scan, 1., 'unbalanced')
        np.testing.assert_array_equal(after[scan == 2], xyz[scan == 2])
        self.assertFalse(info['supported_scans'][2])
        self.assertTrue(info['supported_scans'][0])

    def test_insufficient_sources_and_degenerate_directions_return_identity(self):
        xyz = np.arange(15., dtype=float).reshape(5, 3)/1000.
        for scan in (np.zeros(5), np.arange(5)):
            output, info = estimate(xyz, scan, 1.)
            np.testing.assert_array_equal(output, xyz)
            self.assertEqual(info['status'], 'UNCHANGED')
            json.dumps(info, allow_nan=False)
        with self.assertRaises(ValueError):
            estimate(xyz, np.arange(5), 0.)


if __name__ == '__main__':
    unittest.main()
