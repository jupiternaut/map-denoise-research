import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_support import OLD, evaluate, linear_projection, arrays_hash
from selective_operator import curve_features, veto_mask, choose_outputs, calibration_roughness


def calibration():
    return {name:dict(sorted_s=np.linspace(0,.2,2000),q_tgt=.2,q_abs=.5,q_test=.15)
            for name in OLD.STRATA}


class SelectiveTests(unittest.TestCase):
    def curve(self,a=0):
        t=OLD.TGRID
        return (1-np.maximum(np.exp(-.5*(t-6)**2),a*np.exp(-.5*t**2)))[None,:]

    def select(self,c):
        return choose_outputs(OLD,linear_projection,c,np.zeros(len(c),int),calibration(),np.zeros(3))

    def test_strong_secondary_detected(self):
        self.assertTrue(veto_mask(curve_features(self.curve(.3),OLD.TGRID),0)[0][0])

    def test_weak_secondary_below_rho(self):
        self.assertFalse(veto_mask(curve_features(self.curve(.15),OLD.TGRID),0)[0][0])

    def test_single_valley_not_vetoed(self):
        self.assertFalse(veto_mask(curve_features(self.curve(),OLD.TGRID),0)[0][0])

    def test_flat_identity(self):
        out,_=self.select(np.zeros((2,len(OLD.TGRID))))
        np.testing.assert_array_equal(out['test_veto_projection'],0)

    def test_projection_is_continuous(self):
        t=OLD.TGRID
        s=(t-.8)**2
        dec=OLD.decide(s[None,:],np.zeros(1),np.zeros(1,int),calibration())
        got=linear_projection(OLD,s[None,:],dec,np.ones(1,bool))[0]
        left=dec['a_near'][0]
        expected=t[left-1]+.05*(.2-s[left-1])/(s[left]-s[left-1])
        self.assertAlmostEqual(got,expected,12)
        self.assertLess(got,t[left])

    def test_projection_improves_known_single(self):
        out,_=self.select(self.curve())
        self.assertLess(abs(out['test_projection'][0]-6),6)

    def test_veto_keeps_current(self):
        out,obs=self.select(self.curve(.3))
        self.assertTrue(obs['test'][0])
        self.assertTrue(obs['veto'][0])
        self.assertGreater(out['test_projection'][0],0)
        self.assertEqual(out['test_veto_projection'][0],0)

    def test_acquire_without_move(self):
        out,obs=self.select(self.curve(.3))
        self.assertTrue(obs['research_acquire'][0])
        self.assertEqual(out['test_veto_projection'][0],0)

    def test_truth_not_selector_argument(self):
        import inspect
        names=inspect.signature(choose_outputs).parameters
        self.assertFalse(any(k in names for k in ('truth','tstar','subset','mm')))

    def test_target_truth_intervention(self):
        curves=np.repeat(self.curve(.3),4,axis=0)
        outputs,_=self.select(curves)
        original_hash=arrays_hash(curves,*outputs.values())
        for truth in (np.zeros(4),np.full(4,6.)):
            evaluate(outputs['test_veto_projection'],np.zeros((4,3)),truth,
                     np.full(4,'base'),np.zeros(4,bool))
        rerun,_=self.select(curves)
        self.assertEqual(original_hash,arrays_hash(curves,*rerun.values()))

    def test_coordinate_veto_harm_monotone_not_mae(self):
        truth=np.array([10.,1.])
        candidate=np.array([10.,3.])
        guarded=np.array([0.,3.])
        harm=lambda th:np.maximum(np.abs(th-truth)-np.abs(truth),0)
        self.assertTrue(np.all(harm(guarded)<=harm(candidate)))
        self.assertGreater(np.abs(guarded-truth).mean(),np.abs(candidate-truth).mean())

    def test_accounting_conservation(self):
        m=evaluate(np.array([1.,3.]),np.zeros((2,3)),np.array([2.,0.]),
                   np.array(['big','base']),np.zeros(2,bool))['groups']
        self.assertAlmostEqual(m['big']['weighted_delta']+m['base']['weighted_delta'],m['ALL']['delta_mae'])

    def test_roughness_positive_finite(self):
        rng=np.random.default_rng(9)
        c=rng.normal(size=(80,len(OLD.TGRID)))
        sigma=calibration_roughness(c,np.zeros(80,int))
        self.assertTrue(np.isfinite(sigma).all())
        self.assertTrue((sigma>0).all())

    def test_curve_offset_invariance(self):
        f1=curve_features(self.curve(.3),OLD.TGRID)
        f2=curve_features(self.curve(.3)+12.,OLD.TGRID)
        np.testing.assert_allclose(f1.secondary_prominence,f2.secondary_prominence,atol=1e-12)
        np.testing.assert_allclose(f1.width_mm,f2.width_mm,atol=1e-12)

    def test_invalid_curves_rejected(self):
        with self.assertRaises(ValueError):
            curve_features(np.full((1,481),np.nan),OLD.TGRID)


if __name__=='__main__':unittest.main()
