"""Pre-GT tests for the replay contract."""
import unittest
import numpy as np
import replay
a=replay.a

class Contract(unittest.TestCase):
    def test_resize_centers(self):
        P=np.array([[200.,0,100.,-20.],[0,210.,80.,-40.],[0,0,1.,-3.]])
        rgb=np.zeros((162,202,3),np.uint8)
        camera=a.half_resolution_camera(P,rgb)
        for uv in ([0,0],[10,90],[200,160]):
            ray=np.linalg.solve(P[:,:3],[*uv,1.]); center=-np.linalg.solve(P[:,:3],P[:,3]);point=center+10*ray
            h=camera['P']@np.r_[point,1.]
            np.testing.assert_allclose(h[:2]/h[2],(np.array(uv)+.5)*.5-.5,atol=1e-10)
    def test_strict_veto(self):
        scores=np.zeros((4,4,2));scores[:,:,1]=1
        scores[1,2:,1]=-1;scores[2,2:,:]=np.nan;scores[3,:,1]=0
        accept=a.witness_accept(scores)['accepted']
        np.testing.assert_array_equal(accept,[True,False,False,False])
        np.testing.assert_array_equal(a.veto_routes(np.array([2,1,2,0]),accept),[2,0,0,0])
    def test_heldout_gate(self):
        log=a.ReadLog()
        with self.assertRaises(PermissionError):log.bind('/does/not/exist.png',group='H')
    def test_reject_reference(self):
        self.assertTrue(a.forbidden_path('/data/ground_truth/a.ply'))
        self.assertTrue(a.forbidden_path('/data/sparse/0/points3D.bin'))

if __name__=='__main__':unittest.main()
