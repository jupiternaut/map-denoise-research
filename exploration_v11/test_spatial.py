import unittest
import numpy as np
from spatial_filter import features,logistic,solve

class Tests(unittest.TestCase):
    def test_spatial_gate(self):
        rng=np.random.default_rng(331);xy=rng.uniform(-1,1,(400,2));z=features(xy)
        b=logistic(z,(xy[:,0]>0).astype(float),np.zeros(z.shape[1]))
        self.assertGreater(np.mean((z@b>0)==(xy[:,0]>0)),.95)
    def test_fixed_gap(self):
        rng=np.random.default_rng(332);xy=rng.uniform(-1,1,(400,2));x=np.c_[np.ones(400),xy]
        y=4*(xy[:,0]>0)+rng.normal(0,.15,400)
        m=solve(x,y,.15,gap=4.,force_k=2,iterations=12)
        self.assertAlmostEqual(abs(np.diff(m['means'])[0]),4.)
        self.assertLess(np.mean(abs(m['prediction']-4*(xy[:,0]>0))),.08)
    def test_single_and_input_unchanged(self):
        rng=np.random.default_rng(333);x=np.c_[np.ones(400),rng.normal(size=(400,2))];y=x@np.array([3.,.2,-.1]);old=y.copy()
        m=solve(x,y,1.,force_k=1);np.testing.assert_allclose(m['prediction'],y,atol=1e-12)
        np.testing.assert_array_equal(y,old)

if __name__=='__main__':unittest.main()
