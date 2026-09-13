import unittest,tempfile
from pathlib import Path
import numpy as np
from test_outputs import triangle_metrics,measures,write_points,read,solve

class Checks(unittest.TestCase):
    def test_identity(self):
        q=np.random.default_rng(1).normal(size=(30,3));m=measures(q,q,1.)
        self.assertEqual(m['displacement_max_spacing'],0)
        self.assertEqual(m['edge_ratio_max'],1)
    def test_plane_solver(self):
        xy=np.random.default_rng(2).normal(size=(100,2));x=np.column_stack((np.ones(100),xy));y=x@np.array([2.,.3,-.2])
        m=solve(x,y,1.,iterations=36)
        self.assertEqual(m['k'],1);np.testing.assert_allclose(m['prediction'],y,atol=1e-10)
    def test_cross_boundary_flip_detected(self):
        q=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        m=triangle_metrics(q,np.array([2]),np.array([[0.,-1,0]]),np.array([[0,1,2]]))
        self.assertEqual(m['orientation_reversals'],1)
    def test_identity_triangles(self):
        q=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        m=triangle_metrics(q,np.array([2]),q[[2]],np.array([[0,1,2]]))
        self.assertEqual(m['orientation_reversals'],0);self.assertEqual(m['area_ratio_min'],1.)
    def test_ply_roundtrip(self):
        q=np.random.default_rng(3).normal(size=(10,3))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'q.ply';write_points(p,q);v=read(p)['vertex'].data
            for j,k in enumerate(('x','y','z')):np.testing.assert_array_equal(v[k],q[:,j])

if __name__=='__main__':unittest.main()
