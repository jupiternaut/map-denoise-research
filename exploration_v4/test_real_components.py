"""Small mathematical contract tests; no real input or output files are written."""
import unittest
import numpy as np
from real_components import (construct_input, decomposed_mm, evaluate_output,
                             reachability_diagnostic, scan_coefficients)


class ComponentContracts(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(72591)
        self.points = rng.normal(size=(55, 3)) * np.array([.3, .2, .04]) + [7., -3., 5.]
        self.scans = np.repeat([17, 29], [13, 42])
        self.basis = np.eye(3)

    def test_weighted_gauge_and_units(self):
        ids, frame, counts, coeff = scan_coefficients(self.scans, 12)
        self.assertEqual(ids.tolist(), [17, 29])
        self.assertAlmostEqual(float(np.average(coeff, weights=counts)), 0.)
        self.assertAlmostEqual(float(np.average(coeff**2, weights=counts)), 1.)
        for mode, zero_component in [('normal_translation', 'tangent_rms_mm'),
                                     ('tangent_translation', 'normal_rms_mm')]:
            out, info = construct_input(self.points, self.scans, self.basis, mode, seed=12)
            self.assertAlmostEqual(info['actual']['xyz_rms_mm'], 3.5, places=9)
            self.assertLess(info['actual'][zero_component], 1e-9)
            np.testing.assert_allclose((out-self.points).mean(axis=0), 0., atol=1e-14)
            for f in range(2):
                delta = (out-self.points)[frame == f]
                np.testing.assert_allclose(delta, np.repeat(delta[:1], len(delta), axis=0), atol=1e-14)

    def test_rotation_rigidity_and_zero_scan_mean(self):
        out, info = construct_input(self.points, self.scans, self.basis, 'small_rotation')
        self.assertAlmostEqual(info['actual']['xyz_rms_mm'], 3.5, places=8)
        self.assertLessEqual(info['max_abs_rotation_deg'], 5.)
        for f, sid in enumerate(info['ids']):
            mask = self.scans == sid
            np.testing.assert_allclose((out-self.points)[mask].mean(axis=0), 0., atol=1e-14)
            p, q = self.points[mask], out[mask]
            np.testing.assert_allclose(np.linalg.norm(p-p[:1], axis=1), np.linalg.norm(q-q[:1], axis=1), atol=1e-13)
            t = info['extra_world_transforms'][f]
            np.testing.assert_allclose(p @ t[:3, :3].T + t[:3, 3], q, atol=1e-13)
        diagnostic = reachability_diagnostic(out-self.points, self.scans, [0., 0., 1.])
        self.assertLess(diagnostic['best_scan_constant_axis_removable_rms_mm'], 1e-9)
        self.assertAlmostEqual(diagnostic['best_scan_constant_axis_remaining_rms_mm'], 3.5, places=8)

    def test_identity_score_decomposition(self):
        for mode in ('zero', 'normal_translation', 'tangent_translation', 'small_rotation'):
            current, _ = construct_input(self.points, self.scans, self.basis, mode)
            row = evaluate_output(current, current, self.points, self.points, self.basis[:, 2])
            expected = 0. if mode == 'zero' else 3.5
            self.assertAlmostEqual(row['recovery_xyz_rms_mm'], expected, places=8)
            self.assertAlmostEqual(row['recovery_xyz_rms_mm']**2,
                                   row['recovery_normal_rms_mm']**2+row['recovery_tangent_rms_mm']**2, places=9)
            self.assertEqual(row['zero_edit_xyz_rms_mm'], 0.)
            self.assertEqual(row['input_edit_xyz_rms_mm'], 0.)
            self.assertAlmostEqual(row['self_response_xyz_rms_mm'], expected, places=8)
            self.assertTrue(row['unchanged_exact'])

    def test_candidate_axis_projection_is_not_pointwise_rotation_oracle(self):
        current, _ = construct_input(self.points, self.scans, self.basis, 'normal_translation')
        diag = reachability_diagnostic(current-self.points, self.scans, [0., 0., 1.])
        self.assertAlmostEqual(diag['best_scan_constant_axis_removable_rms_mm'], 3.5, places=8)
        self.assertLess(diag['best_scan_constant_axis_remaining_rms_mm'], 1e-9)
        tilted = np.array([1., 0., 1.])/np.sqrt(2.)
        diag = reachability_diagnostic(current-self.points, self.scans, tilted)
        self.assertAlmostEqual(diag['best_scan_constant_axis_removable_rms_mm'], 3.5/np.sqrt(2.), places=8)
        self.assertAlmostEqual(diag['best_scan_constant_axis_remaining_rms_mm'], 3.5/np.sqrt(2.), places=8)
        normal_scores = decomposed_mm(current-self.points, tilted)
        self.assertAlmostEqual(normal_scores['normal_rms_mm'], normal_scores['tangent_rms_mm'], places=8)


if __name__ == '__main__':
    unittest.main()
