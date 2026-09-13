import unittest
import numpy as np
from spectral_filter import frame_power,fit_power,FREQUENCY,fit_surfaces,filter_frozen

class SpectralTests(unittest.TestCase):
    def test_frame_translation_invariance(self):
        rng=np.random.default_rng(9131001);scan=np.repeat(np.arange(4),50);y=rng.normal(size=200)
        a,_,_=frame_power(y,scan,1.);b,_,_=frame_power(y+np.array([4.,-2.,12.,0.])[scan],scan,1.)
        np.testing.assert_allclose(a,b,atol=1e-14)
    def test_matches_explicit_differences(self):
        y=np.array([-.7,.2,1.4,2.]);q,_,_=frame_power(y,np.zeros(4,int),1.)
        pairs=y[:,None]-y[None,:];off=~np.eye(4,dtype=bool)
        expected=np.cos(pairs[off,None]*FREQUENCY).mean(0)
        np.testing.assert_allclose(q[0],expected,atol=1e-14)
    def test_scale_invariance(self):
        y=np.arange(10.)*.3;s=np.repeat([0,1],5)
        a,_,_=frame_power(y,s,1.);b,_,_=frame_power(y*3,s,3.)
        np.testing.assert_allclose(a,b,atol=1e-14)
    def test_population_gap(self):
        for gap in (2.,4.,8.):
            q=np.exp(-FREQUENCY**2)*(1-.8*np.sin(FREQUENCY*gap/2)**2)
            r=fit_power(np.array([q]),[96])
            self.assertLess(abs(r['gap_over_sigma']-gap),.03)
    def test_negative_power_not_clipped(self):
        q,_,_=frame_power(np.array([-2.,0.,2.]),np.zeros(3,int),1.)
        self.assertTrue(np.any(q<0))
    def test_fixed_gap_is_fixed(self):
        rng=np.random.default_rng(9131003);n=300;x=np.c_[np.ones(n),rng.normal(size=(n,2))]
        y=3*(np.arange(n)%2)+rng.normal(scale=.2,size=n)
        fit=fit_surfaces(x,y,np.zeros(n,int),.2,gap=3.,force_k=2,iterations=12)
        self.assertAlmostEqual(abs(np.diff(fit['means'])[0]),3.)
    def test_input_support_preserved(self):
        rng=np.random.default_rng(9131005);n=100;world=rng.normal(size=(n,3))*.001
        state=dict(world=world,order=np.arange(n),scan_id=np.repeat([0,1],50),local=world*1000,
            corrected=world[:,2]*1000,design=np.c_[np.ones(n),world[:,:2]*1000],
            support=np.arange(n)%3!=0,normal=np.array([0.,0.,1.]))
        before=world.copy();out,info,a=filter_frozen(state,1.,method='pooled_em',iterations=4)
        np.testing.assert_array_equal(out[~state['support']],before[~state['support']]);np.testing.assert_array_equal(world,before)
        self.assertTrue(np.isfinite(out).all());self.assertEqual(info['truth_fields_used'],[])
if __name__=='__main__':unittest.main()
