"""Focused correctness tests; no parameter search or reserved-seed evaluation."""
from pathlib import Path
import json
import unittest
from unittest.mock import patch

import numpy as np

import warm_transport as method


class WarmTransportTests(unittest.TestCase):
    @staticmethod
    def sample():
        rng = np.random.default_rng(701)
        xy = rng.uniform(-.06, .06, (48, 2))
        base = np.c_[xy, np.zeros(len(xy))]
        cloud = np.vstack([base+np.array([0., 0., z]) for z in (-.002, 0., .002)])
        return cloud, np.repeat([7, 19, 42], len(base))

    def test_source_points_fixed_and_each_prepost_scan_move_is_rigid(self):
        xyz, scans = self.sample(); x0, s0 = xyz.copy(), scans.copy()
        parameters = dict(method._measure.PARAMETERS)
        raw, info = method._correct(xyz, scans, 1., 'warm_balanced_2')
        np.testing.assert_array_equal(xyz, x0); np.testing.assert_array_equal(scans, s0)
        for sid in np.unique(scans):
            delta = raw[scans == sid]-xyz[scans == sid]
            np.testing.assert_allclose(delta, np.broadcast_to(delta[0], delta.shape), atol=2e-17, rtol=0)
        self.assertEqual(method._measure.PARAMETERS, parameters)
        self.assertFalse(info['graph_projected_cloud_used'])
        self.assertEqual(len(info['outer_history']), 2)
        json.dumps(info, allow_nan=False)

    def test_original_measure_target_is_identical_for_zero_warm_and_controls(self):
        xyz, scans = self.sample()
        fingerprints = []
        for variant in ('zero_balanced_6', 'warm_balanced_2', 'warm_unbalanced_2', 'graph_projected_bias_only'):
            raw, info = method._correct(xyz, scans, 1., variant)
            fingerprints.append(info['target_representation_sha256'])
            self.assertTrue(np.isfinite(raw).all())
        self.assertIsNotNone(fingerprints[0])
        self.assertEqual(len(set(fingerprints)), 1)

    def test_graph_projected_cloud_is_ignored_and_final_local_runs_once(self):
        xyz, scans = self.sample()
        original_estimate = method._graph.estimate
        def poisoned_projection(p, f, sigma, variant='graph'):
            result, info = original_estimate(p, f, sigma, variant=variant)
            if variant == 'graph': result = np.full_like(result, 123456.)
            return result, info
        normal_raw, normal_info = method._correct(xyz, scans, 1., 'warm_balanced_2')
        with patch.object(method._graph, 'estimate', side_effect=poisoned_projection) as mocked:
            output, info = method.estimate(xyz, scans, 1., 'warm_balanced_2')
        expected, _ = original_estimate(normal_raw, scans, 1., variant='local_only')
        np.testing.assert_array_equal(output, expected)
        self.assertEqual([c.kwargs['variant'] for c in mocked.call_args_list], ['graph', 'local_only'])
        self.assertEqual(info['initial_bias_mm'], normal_info['initial_bias_mm'])
        json.dumps(info, allow_nan=False)

    def test_zero6_exactly_reproduces_saved_v3_budget6(self):
        run = Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/transport-budget-v3-mxi1kprj')
        if not run.is_dir(): self.skipTest('exact prior development budget run unavailable')
        case = 'ghost_s912101_b4'
        with np.load(run/'inputs'/f'{case}.npz', allow_pickle=False) as a:
            xyz, scans = a['xyz_world'], a['scan_id']
        for variant in ('balanced', 'unbalanced'):
            out, info = method.estimate(xyz, scans, 1., f'zero_{variant}_6')
            raw = xyz-np.asarray(info['refined_bias_mm'])[scans, None]*np.asarray(info['normal_world'])[None, :]/1000.
            for stage, actual in (('raw', raw), ('local', out)):
                with np.load(run/'outputs'/f'{case}__{variant}_o6__{stage}.npz', allow_pickle=False) as a:
                    np.testing.assert_array_equal(actual, a['xyz_world'])

    def test_projected_control_is_exact_warm_initial_state(self):
        xyz, scans = self.sample()
        raw, control = method._correct(xyz, scans, 1., 'graph_projected_bias_only')
        _, warm = method._correct(xyz, scans, 1., 'warm_balanced_2')
        self.assertEqual(control['refined_bias_mm'], warm['initial_bias_mm'])
        np.testing.assert_allclose(np.average(warm['initial_bias_mm'], weights=warm['points_per_scan']), 0., atol=1e-14)
        _, inverse = np.unique(scans, return_inverse=True)
        reconstruction = xyz-np.asarray(warm['initial_bias_mm'])[inverse, None]*np.asarray(warm['normal_world'])[None, :]/1000.
        np.testing.assert_array_equal(raw, reconstruction)

    def test_graph_bias_control_uses_translation_not_graph_surface_output(self):
        xyz, scans = self.sample()
        _, graph_info = method._graph.estimate(xyz, scans, 1., variant='graph')
        raw, info = method._correct(xyz, scans, 1., 'graph_bias_only')
        _, inverse = np.unique(scans, return_inverse=True)
        expected = xyz-np.asarray(graph_info['bias_mm'])[inverse, None]*np.asarray(graph_info['normal_world'])[None, :]/1000.
        np.testing.assert_array_equal(raw, expected)
        self.assertEqual(info['outer_iterations'], 0)

    def test_projected_initialization_uses_target_components_and_limit(self):
        graph_info = {'normal_world': [0., 0., 1.], 'scan_ids': [7, 19, 42], 'bias_mm': [9., 4., -9.]}
        bias, info = method._project_graph_bias(graph_info, np.array([7, 19, 42]),
                                              np.ones(3), np.array([0., 0., 1.]), [(0, 2)], 1.)
        np.testing.assert_allclose(bias, [8., 0., -8.], rtol=0, atol=1e-14)
        self.assertEqual(info['initial_gauge_components_scan_ids'], [[7, 42], [19]])


if __name__ == '__main__':
    unittest.main()
