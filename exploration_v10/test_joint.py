import unittest
import numpy as np
from joint_decoder import solve

class JointTests(unittest.TestCase):
    def test_single_frame_bias_and_gauge(self):
        rng=np.random.default_rng(91);s=np.repeat(np.arange(4),100)
        x=np.column_stack((np.ones(400),rng.normal(size=(400,2))))
        b=np.array([-3.,-1.,1.,3.]);y=2+x[:,1:]@np.array([.3,-.2])+b[s]
        m=solve(x,y,s,1.,force_k=1)
        np.testing.assert_allclose(m['prediction'],y-b[s],atol=1e-12)
        np.testing.assert_allclose(m['bias'],b,atol=1e-12)
    def test_dual_fixed_gap(self):
        rng=np.random.default_rng(93);s=np.repeat(np.arange(4),100)
        x=np.column_stack((np.ones(400),rng.normal(size=(400,2))))
        g=np.arange(400)%2;b=np.array([-3.,-1.,1.,3.])
        y=2+8*(g-.5)+b[s]+rng.normal(0,.05,400)
        m=solve(x,y,s,.1,gap=8.,force_k=2,iterations=12,initial_y=y-b[s])
        self.assertEqual(m['gap'],8.)
        self.assertLess(np.max(np.abs(m['bias']-b)),.02)

if __name__=='__main__':unittest.main()
