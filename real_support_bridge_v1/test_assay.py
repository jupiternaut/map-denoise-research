"""Unit checks for descriptive geometry, not tests of research success."""
import unittest
import numpy as np
from assay import project, rectangle, assess


class AssayTests(unittest.TestCase):
    def test_pixel_projection(self):
        P=np.array([[100.,0,20,0],[0,200,30,0],[0,0,1,0]])
        uv,z=project(np.array([[2.,3.,10.]]),P)
        np.testing.assert_allclose(uv,[[40,90]])
        np.testing.assert_allclose(z,[10])

    def test_rectangle_exclusive_edges_and_negative_depth(self):
        uv=np.array([[10,10],[20,10],[10,20],[11,11],[11,11]])
        z=np.array([1,1,1,-1,1])
        np.testing.assert_array_equal(rectangle(uv,z,(10,10,20,20)),[True,False,False,False,True])

    def test_camera_scale_composition(self):
        S=np.diag([12.,12.,12.,1.]);S[:3,3]=[3,4,20]
        P=np.array([[100.,0,20,0],[0,100,30,0],[0,0,1,0]])
        normalized=np.array([[1.,2.,5.],[-1,2,8]])
        physical=normalized@S[:3,:3].T+S[:3,3]
        np.testing.assert_allclose(project(normalized,P)[0],project(physical,P@np.linalg.inv(S))[0])

    def test_plane_sign_invariance(self):
        a=dict(normal=np.array([0.,0,1]),offset=0.,fraction=.6)
        b=dict(normal=np.array([0.,0,-1]),offset=3.,fraction=.25)
        result=assess([a,b],np.array([2.,3,17.]))
        self.assertTrue(result['provisional_parallel_pair'])
        self.assertAlmostEqual(result['line_separation_at_center_mm'],3.)
        self.assertAlmostEqual(result['angle_deg'],0.)

    def test_crossing_planes_rejected_without_gap_infinity(self):
        a=dict(normal=np.array([0.,0,1]),offset=0.,fraction=.6)
        b=dict(normal=np.array([1.,0,0]),offset=3.,fraction=.25)
        result=assess([a,b],np.array([0.,0,0.]))
        self.assertFalse(result['provisional_parallel_pair'])
        self.assertIsNone(result['line_separation_at_center_mm'])

    def test_single_plane_and_small_remainder_not_pair(self):
        a=dict(normal=np.array([0.,0,1]),offset=0.,fraction=.9)
        b=dict(normal=np.array([0.,0,1]),offset=3.,fraction=.1)
        self.assertFalse(assess([a],np.zeros(3))['provisional_parallel_pair'])
        self.assertFalse(assess([a,b],np.zeros(3))['provisional_parallel_pair'])


if __name__=='__main__':unittest.main()
