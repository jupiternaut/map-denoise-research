import sys,unittest
import numpy as np
sys.path.insert(0,'/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2')
from experiment import generate
from joint_framewise import METHODS,estimate,_box_gauge

class CandidateTests(unittest.TestCase):
    def test_outputs_and_no_mutation(self):
        inp,_=generate('crossed_imbalanced',211);original=inp.xyz_mm.copy()
        for method in METHODS:
            out,b,info=estimate(inp,method)
            self.assertEqual(out.shape,original.shape);self.assertTrue(np.isfinite(out).all())
            self.assertTrue(np.isfinite(b).all());self.assertTrue(np.array_equal(out[:,:2],original[:,:2]))
            self.assertLessEqual(abs(b).max(),inp.bias_bound_mm+1e-10)
            self.assertFalse(info['requires_ground_truth'])
        self.assertTrue(np.array_equal(inp.xyz_mm,original))
    def test_ambiguous_pair_identical(self):
        a,_=generate('single_ghost_no_anchor',213);b,_=generate('thin_segregated_no_anchor',213)
        self.assertTrue(np.array_equal(a.xyz_mm,b.xyz_mm))
        oa,ba,ia=estimate(a,METHODS[0]);ob,bb,ib=estimate(b,METHODS[0])
        self.assertTrue(np.array_equal(oa,ob));self.assertEqual(ia['status'],'WAIT')
    def test_box_gauge(self):
        b=_box_gauge(np.array([-20.,4.,15.]),np.array([2.,3.,5.]),8.)
        self.assertLessEqual(abs(b).max(),8.)
        self.assertLess(abs(np.dot(b,[2.,3.,5.])),1e-10)
if __name__=='__main__':unittest.main()
