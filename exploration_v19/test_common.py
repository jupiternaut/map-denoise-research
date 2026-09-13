import unittest
import numpy as np
from common import make,world_support

class CommonTests(unittest.TestCase):
    def test_generator_pairing(self):
        a,ta=make(19,2,1,.2,0);b,tb=make(19,2,1,.2,1)
        np.testing.assert_array_equal(a['design'],b['design'])
        np.testing.assert_array_equal(ta['eps_mm'],tb['eps_mm'])
        self.assertEqual(ta['gt_layer'].sum(),tb['gt_layer'].sum())
        for f in (0,1):self.assertEqual(ta['gt_layer'][a['scan_id']==f].sum(),77)
    def test_bridge_order_and_mass(self):
        inp={'xyz_world':np.array([[.001,0,.010],[.002,0,.020],[.003,0,.030]])}
        state=dict(order=[2,0,1],support=[True,False,True],local=[[3,0,30],[1,0,10],[2,0,20]],normal=[0,0,1])
        m=dict(representation='weighted',support_heights=[[29,31],[9,11],[19,21]],support_weights=[[.2,.8],[.3,.7],[.4,.6]])
        p,w,_=world_support(inp,{'state':state},m)
        np.testing.assert_allclose(p[:,:,2],[[10,10],[19,21],[29,31]])
        np.testing.assert_allclose(w,[[1,0],[.4,.6],[.2,.8]])
        np.testing.assert_allclose(w.sum(1),1)
    def test_identity_no_upstream_move(self):
        inp={'xyz_world':np.array([[1.,2.,3.]])}
        p,w,_=world_support(inp,{},dict(identity=True))
        np.testing.assert_array_equal(p[0,0],[1000,2000,3000])
    def test_weighted_surface_does_not_average_position(self):
        inp={'xyz_world':np.zeros((1,3))}
        p,w,_=world_support(inp,{},dict(representation='weighted',support_heights=[[-1,1]],support_weights=[[.5,.5]]))
        np.testing.assert_array_equal(p[:,:,2],[[-1,1]])
        self.assertEqual(float(w.sum()),1.)

if __name__=='__main__':unittest.main()
