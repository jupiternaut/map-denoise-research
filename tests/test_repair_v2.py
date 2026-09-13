import copy
import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapt import build_adapter, to_operator_mm, from_operator_mm
from evaluate_v2 import geometry_metrics, synthetic_geometry, source_correspondence_metrics
from generate_synthetics import make_ghost, make_dual, make_ambiguous
from schema import validate_pose_consistency, write_patch, fill_rays, pose_table
from perturb import make_perturbation
from hashutil import sha256_file
from repair_pilot_v2 import compute_current, real_metrics


class EvaluationRepairTests(unittest.TestCase):
    def test_exact_truth_scores_zero(self):
        for gap in (2., 4., 8.):
            _, _, ev = make_dual(912101, gap, 4.)
            score = synthetic_geometry(ev['gt_clean_xyz_world'], {}, ev)
            self.assertLess(score['surface_accuracy_mean_mm'], 1e-9)
            self.assertLess(score['source_group_gap_error_mm'], 1e-9)
            self.assertEqual(score['reference_sample_coverage_1mm'], 1.)

    def test_wrong_plane_100mm_not_zero(self):
        _, _, ev = make_ghost(912101, 0.)
        out = ev['gt_clean_xyz_world'].copy(); out[:, 2] += .1
        s = geometry_metrics(out, ev)
        self.assertAlmostEqual(s['surface_accuracy_mean_mm'], 100.)
        self.assertEqual(s['reference_sample_coverage_2mm'], 0.)

    def test_metadata_cannot_change_geometry(self):
        _, _, ev = make_dual(912101, 4., 0.)
        out = ev['gt_clean_xyz_world'].copy(); out[:, 2] = .002
        a = synthetic_geometry(out, {'k': 1}, ev)
        b = synthetic_geometry(out, {'k': 2, 'mu_mm': [0., 4.]}, ev)
        a.pop('reported_model_k'); b.pop('reported_model_k')
        self.assertEqual(a, b)

    def test_geometry_invariant_to_point_order(self):
        p, _, ev = make_dual(912101, 4., 4.)
        a = geometry_metrics(p['xyz_world'], ev)
        b = geometry_metrics(p['xyz_world'][::-1], ev)
        for k in a:
            if isinstance(a[k], float): self.assertAlmostEqual(a[k], b[k], places=10)
            else: self.assertEqual(a[k], b[k])

    def test_collapsed_layers_not_rewarded_as_exact(self):
        _, _, ev = make_dual(912101, 4., 0.)
        out = ev['gt_clean_xyz_world'].copy(); out[:, 2] = .002
        s = synthetic_geometry(out, {'k': 2, 'mu_mm': [0., 4.]}, ev)
        self.assertAlmostEqual(s['surface_accuracy_mean_mm'], 2.)
        self.assertAlmostEqual(s['source_group_gap_error_mm'], 4.)
        self.assertEqual(s['reference_sample_coverage_1mm'], 0.)

    def test_world_score_after_local_roundtrip(self):
        p, _, ev = make_dual(912101, 2., 4.)
        ad = build_adapter(p['xyz_world'])
        truth = ev['gt_clean_xyz_world']
        roundtrip = from_operator_mm(to_operator_mm(truth, ad), ad)
        self.assertLess(geometry_metrics(roundtrip, ev)['surface_accuracy_mean_mm'], 1e-9)

    def test_single_noise_cloud_not_auto_classified_two(self):
        p, _, ev = make_ghost(912101, 0.)
        s = synthetic_geometry(p['xyz_world'], {}, ev)
        self.assertIsNone(s['reported_model_k'])
        self.assertNotIn('false_split', s)
        self.assertNotIn('k_hat', s)

    def test_finite_surface_boundary_counts(self):
        _, _, ev = make_ghost(912101, 0.)
        out = ev['gt_clean_xyz_world'].copy(); out[:, 0] = .16
        self.assertAlmostEqual(geometry_metrics(out, ev)['surface_accuracy_mean_mm'], 100.)

    def test_ambiguous_not_scored(self):
        p, _, ev = make_ambiguous(912101)
        s = synthetic_geometry(p['xyz_world'], {}, ev)
        self.assertFalse(s['geometry_scored'])
        self.assertNotIn('surface_accuracy_mean_mm', s)


class DataRepairTests(unittest.TestCase):
    def test_current_adapter_uses_only_visible_xyz(self):
        class IdentityOps:
            @staticmethod
            def estimate(method, xyz, frame, sigma):
                return xyz.copy(), {'method': method}
        p, _, _ = make_ghost(912101, 0.)
        p['xyz_world'] = p['xyz_world'] + [.13, -.2, .03]
        # Deliberately inconsistent unused fields must not select the adapter.
        p['xyz_local'] = p['xyz_local'] * 100
        results, adapter = compute_current(IdentityOps, p, 2.)
        np.testing.assert_allclose(adapter['origin_m'], p['xyz_world'].mean(axis=0), atol=1e-14)
        for _, world, _, _, error in results:
            self.assertIsNone(error)
            np.testing.assert_allclose(world, p['xyz_world'], atol=1e-14)

    def test_baseline_change_decomposition(self):
        rng = np.random.default_rng(912501)
        ref = rng.normal(size=(100, 3))
        base = ref + rng.normal(size=(100, 3)) * .04
        out = base + rng.normal(size=(100, 3)) * .004
        metrics = real_metrics(out, ref, base)
        self.assertLess(metrics['decomposition_error_mm2'], 1e-9)

    def test_generated_poses_and_rays_consistent(self):
        for p, m, _ in (make_ghost(912101, 4.), make_dual(912101, 4., 4.), make_ambiguous(912101)):
            validate_pose_consistency(p, m)

    def test_old_local_equals_world_bug_rejected(self):
        p, m, _ = make_ghost(912101, 4.)
        p['xyz_local'] = p['xyz_world'].copy()
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            validate_pose_consistency(p, m)

    def test_perturbed_pose_origin_ray_consistency(self):
        p, m, _ = make_ghost(912101, 4.)
        change = make_perturbation(p['xyz_world'], p['scan_id'], pose_table(m),
                                   912401, .005, np.deg2rad(.05), 0)
        p['xyz_world'] = change['xyz_world_perturbed']
        m['T_world_from_scan_input'] = change['current_T_world_from_scan']
        for sid, T in pose_table(m).items():
            p['scanner_origin_world'][p['scan_id'] == sid] = T[:3, 3]
        fill_rays(p)
        validate_pose_consistency(p, m)

    def test_patch_overwrite_rejected_without_changes(self):
        p, m, ev = make_ghost(912101, 0.)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            write_patch(dest, 'case', p, m, ev)
            before = {str(f): sha256_file(f) for f in dest.rglob('*') if f.is_file()}
            with self.assertRaises(FileExistsError): write_patch(dest, 'case', p, m, ev)
            after = {str(f): sha256_file(f) for f in dest.rglob('*') if f.is_file()}
            self.assertEqual(before, after)

    def test_ray_corruption_rejected(self):
        p, m, _ = make_ghost(912101, 0.)
        p['ray_direction'][0] *= -1
        with self.assertRaisesRegex(ValueError, 'ray directions'):
            validate_pose_consistency(p, m)


if __name__ == '__main__': unittest.main()
