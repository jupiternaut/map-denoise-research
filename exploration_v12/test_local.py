import unittest
import numpy as np
from local_filter import windows
from external import project

class Tests(unittest.TestCase):
    def test_window_partition(self):
        x=np.random.default_rng(881).normal(size=(200,3));jobs=windows(x,48)
        np.testing.assert_array_equal(np.sort(np.concatenate([c for c,_ in jobs])),np.arange(200))
        for core,fit in jobs:self.assertTrue(np.isin(core,fit).all())
    def test_plane_external(self):
        rng=np.random.default_rng(882);q=np.c_[rng.uniform(-.1,.1,(300,2)),np.zeros(300)]
        for method in ('apss','rimls'):
            out=project(q,method,2.)
            self.assertLess(np.max(abs(out[:,2])),1e-6)
            self.assertEqual(out.shape,q.shape)
if __name__=='__main__':unittest.main()
