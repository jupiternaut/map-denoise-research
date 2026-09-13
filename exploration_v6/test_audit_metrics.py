import unittest
import numpy as np
from audit_metrics import axis_bounds,decompose,synthetic_scores


class AuditMetricsTests(unittest.TestCase):
    def test_perpendicular_bound(self):
        ref=np.zeros((8,3));cur=ref+np.array([.0035,0,0])
        out=cur+np.array([0,0,.001])
        r=axis_bounds(out,cur,ref,[0,0,1],np.ones(8,bool))
        self.assertAlmostEqual(r['axis_only_floor_mm'],3.5)
        self.assertGreater(r['actual_recovery_xyz_rms_mm'],3.5)

    def test_support_tightens_bound(self):
        ref=np.zeros((2,3));cur=ref+np.array([0,0,.004])
        out=cur.copy();out[0]=ref[0]
        r=axis_bounds(out,cur,ref,[0,0,1],np.array([True,False]))
        self.assertAlmostEqual(r['axis_only_floor_mm'],0)
        self.assertAlmostEqual(r['axis_and_support_floor_mm'],4/np.sqrt(2))

    def test_illegal_tangential_edit_detected(self):
        a=np.zeros((5,3));b=a+np.array([.001,0,0])
        with self.assertRaises(AssertionError):
            axis_bounds(b,a,a,[0,0,1],np.ones(5,bool))

    def test_decomposition(self):
        d=np.array([[.003,.004,0],[.003,.004,0]])
        r=decompose(d,[1,0,0])
        self.assertAlmostEqual(r['xyz_rms_mm'],5)
        self.assertAlmostEqual(r['normal_rms_mm'],3)
        self.assertAlmostEqual(r['tangent_rms_mm'],4)

    def test_finite_surface_and_gap(self):
        xy=np.array([[x,y] for x in [-1,0,1] for y in [-1,1]],float)
        q=np.vstack([np.c_[xy,np.full(6,z)] for z in [0,4]])/1000
        ref=q.copy();q[:,2]+=.001
        r=synthetic_scores(q,ref,[[0,-1,1,-1,1],[4,-1,1,-1,1]],
                           np.repeat([0,1],6),4)
        self.assertAlmostEqual(r['surface_accuracy_mean_mm'],1)
        self.assertAlmostEqual(r['matched_point_rms_mm'],1)
        self.assertAlmostEqual(r['fitted_gap_at_same_xy_error_mm'],0)


if __name__=='__main__':unittest.main()
