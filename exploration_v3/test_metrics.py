import unittest
import numpy as np
from metrics import structure_metrics, sampling_indices, subset_evaluation


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(77)
        xy = rng.uniform(-40,40,(200,2))
        labels = (xy[:,0] > 0).astype(int)
        ref = np.c_[xy, 4*labels]/1000
        self.ev = {'identifiable':True, 'gt_clean_xyz_world':ref, 'gt_layer':labels, 'true_gap_mm':4.}

    def test_ramp_is_not_two_planes(self):
        ref = self.ev['gt_clean_xyz_world']
        a = structure_metrics(ref,self.ev)
        ramp = ref.copy(); ramp[:,2]=2/1000 + .06*ramp[:,0]
        b = structure_metrics(ramp,self.ev)
        self.assertLess(a['fitted_gap_at_same_xy_error_mm'],1e-9)
        self.assertGreater(b['fitted_gap_at_same_xy_error_mm'],3.99)
        self.assertGreater(b['source_surface_tilt_mean_deg'],3.)

    def test_single_global_alignment_only(self):
        ref = self.ev['gt_clean_xyz_world']
        a = structure_metrics(ref+np.array([.1,.2,.3]), self.ev)
        self.assertLess(a['global_translation_only_aligned_rms_mm'],1e-8)
        collapsed=ref.copy(); collapsed[:,2]=.002
        b=structure_metrics(collapsed,self.ev)
        self.assertGreater(b['global_translation_only_aligned_rms_mm'],1.9)

    def test_sampling_keeps_provenance(self):
        ref=self.ev['gt_clean_xyz_world']; frame=np.arange(len(ref))%4
        for variant in ('density','partial_overlap'):
            idx=sampling_indices(ref,frame,variant)
            self.assertEqual(len(idx),len(np.unique(idx)))
            self.assertLess(len(idx),len(ref))
            self.assertEqual(len(np.unique(frame[idx])),4)
            ev=subset_evaluation(self.ev,idx,len(ref))
            np.testing.assert_array_equal(ev['gt_clean_xyz_world'],ref[idx])


if __name__=='__main__': unittest.main()
