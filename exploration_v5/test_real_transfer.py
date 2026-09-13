"""Bounded read-only contracts for the real transfer harness."""
import inspect
import unittest
from types import SimpleNamespace

import numpy as np

import real_transfer as probe


class RealTransferTests(unittest.TestCase):
    def test_exact_24_inputs_and_source_identity(self):
        cases=probe.load_cases()
        self.assertEqual(len(cases),24)
        self.assertEqual({c['patch'] for c in cases},set(probe.PATCH_NAMES))
        for c in cases:
            self.assertEqual(len(c['xyz_world']),len(c['source_point_index']))
            if c['mode']=='zero':np.testing.assert_array_equal(c['xyz_world'],c['reference'])

    def test_candidate_boundary_contains_only_copied_legal_arrays(self):
        seen=[]
        def estimate(p,scan,sigma,variant):
            self.assertEqual(sigma,2.)
            seen.append((variant,p.copy(),scan.copy()))
            p[:] += .001;scan[:] = 99
            return p,{'status':'APPLY'}
        modules={name:SimpleNamespace(estimate=estimate) for name in ('pool','slope','graph','measure','gicp')}
        xyz=np.arange(36,dtype=float).reshape(12,3)/1000.;scans=np.repeat([5,9],6)
        original,ids=xyz.copy(),scans.copy()
        self.assertEqual(list(inspect.signature(probe.invoke).parameters),['method','current_xyz','scan_id','sigma_mm','modules'])
        for method in probe.METHODS:probe.invoke(method,xyz,scans,2.,modules)
        np.testing.assert_array_equal(xyz,original);np.testing.assert_array_equal(scans,ids)
        self.assertEqual(len(seen),7)
        for _,p,f in seen:np.testing.assert_array_equal(p,original);np.testing.assert_array_equal(f,ids)

    def test_support_mask_not_replaced_by_movement(self):
        scans=np.array([7,7,19,19])
        mask,info=probe.applicability({'supported_fraction':.5,'unsupported_point_indices':[1,3]},scans)
        np.testing.assert_array_equal(mask,[True,False,True,False])
        self.assertTrue(info['support_mask_available'])
        mask,info=probe.applicability({'supported_fraction':.75},scans)
        self.assertIsNone(mask);self.assertFalse(info['support_mask_available'])
        with self.assertRaises(ValueError):probe.applicability({'supported_fraction':1.,'unsupported_point_indices':[1]},scans)

    def test_axes_and_component_metrics_are_evaluator_only(self):
        rng=np.random.default_rng(9051)
        reference=rng.normal(size=(24,3))*[.1,.2,.01]
        scans=np.repeat([1,2,3],8)
        delta=np.zeros_like(reference);delta[:,2]=np.repeat([-.0035,0.,.0035],8)
        current=reference+delta
        case=dict(xyz_world=current,reference=reference,scan_id=scans,
                  construction={'construction_normal_world':[0.,0.,1.]})
        row=probe.score(current,case,reference,np.array([1.,0.,0.]),np.array([1.,0.,0.]))
        self.assertAlmostEqual(row['recovery_xyz_rms_mm']**2,
                               row['recovery_normal_rms_mm']**2+row['recovery_tangent_rms_mm']**2)
        self.assertEqual(row['candidate_axis_abs_cos'],0.)
        self.assertEqual(row['zero_edit_xyz_rms_mm'],0.)
        self.assertEqual(row['input_edit_xyz_rms_mm'],0.)
        self.assertTrue(row['independent_real_geometry']=='not_measured')


if __name__=='__main__':unittest.main(verbosity=2)
