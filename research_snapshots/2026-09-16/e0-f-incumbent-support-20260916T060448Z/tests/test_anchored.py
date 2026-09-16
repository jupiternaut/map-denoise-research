import inspect
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_experiment import OLD, ZeroNoise, curve_features, anchored_prominence, construct


class Tests(unittest.TestCase):
    def curve(self,secondary):
        g=OLD.TGRID
        c=1-np.maximum(np.exp(-.5*(g-6.)**2),.6*np.exp(-.5*(g-secondary)**2))
        return c[None,:]

    def test_near_incumbent(self):
        c=self.curve(0.);f=curve_features(c,OLD.TGRID)
        p,loc=anchored_prominence(c,OLD.TGRID,f.width_mm,1.)
        self.assertGreater(p[0],.5);self.assertAlmostEqual(loc[0],0.)

    def test_far_irrelevant(self):
        c=self.curve(-5.);f=curve_features(c,OLD.TGRID)
        p,_=anchored_prominence(c,OLD.TGRID,f.width_mm,1.)
        self.assertGreater(f.secondary_prominence[0],.5);self.assertEqual(p[0],0.)

    def test_radius_sensitivity(self):
        c=self.curve(1.5);f=curve_features(c,OLD.TGRID)
        p1,_=anchored_prominence(c,OLD.TGRID,f.width_mm,1.)
        p2,_=anchored_prominence(c,OLD.TGRID,f.width_mm,2.)
        self.assertEqual(p1[0],0.);self.assertGreater(p2[0],.4)

    def test_flat(self):
        c=np.ones((2,len(OLD.TGRID)));f=curve_features(c,OLD.TGRID)
        p,_=anchored_prominence(c,OLD.TGRID,f.width_mm)
        np.testing.assert_array_equal(p,np.zeros(2))

    def test_single(self):
        c=(1-np.exp(-.5*(OLD.TGRID-6.)**2))[None,:];f=curve_features(c,OLD.TGRID)
        p,_=anchored_prominence(c,OLD.TGRID,f.width_mm)
        self.assertEqual(p[0],0.)

    def test_global_subset_and_offset_invariance(self):
        c=np.vstack([self.curve(0.),self.curve(-5.),self.curve(1.5)])
        f=curve_features(c,OLD.TGRID)
        p,_=anchored_prominence(c,OLD.TGRID,f.width_mm)
        p2,_=anchored_prominence(c+47,OLD.TGRID,f.width_mm)
        np.testing.assert_allclose(p,p2,atol=1e-13)
        self.assertTrue(np.all(p<=f.secondary_prominence+1e-12))

    def test_invalid(self):
        with self.assertRaises(ValueError):anchored_prominence([[np.nan]*5],np.arange(5),[1.])
        with self.assertRaises(ValueError):anchored_prominence(np.zeros((1,5)),np.arange(5),[1.],-1.)

    def test_zero_noise_is_reproducible(self):
        self.assertEqual(OLD.smooth_noise(ZeroNoise(),3).max(),0.)

    def test_no_truth_api_and_mask_endpoints(self):
        self.assertEqual(list(inspect.signature(construct).parameters),['curves','grid','features','diagnostics','existing'])
        c=np.vstack([self.curve(0.),self.curve(-5.)]);f=curve_features(c,OLD.TGRID)
        obs={'threshold':np.full(2,.25),'test':np.ones(2,bool),'veto':np.ones(2,bool)}
        old={'identity':np.zeros(2),'test_projection':np.full(2,4.),
             'test_veto_projection':np.zeros(2),'argmin':np.full(2,6.)}
        out,_,_=construct(c,OLD.TGRID,f,obs,old)
        np.testing.assert_array_equal(out['anchor1_projection'],[0.,4.])
        np.testing.assert_array_equal(out['anchor1_argmin'],[0.,6.])
        # External evaluation worlds differ but do not enter the method call.
        repeat,_,_=construct(c.copy(),OLD.TGRID,f,obs,old)
        for k in out:np.testing.assert_array_equal(out[k],repeat[k])


if __name__=='__main__':unittest.main()
