import unittest
import numpy as np
from scipy.optimize import approx_fprime
from constant_solver import objective, solve_constant
from run import structural, v14, choose


class ChallengeChecks(unittest.TestCase):
    def test_likelihood_gradient(self):
        rng=np.random.default_rng(10)
        x=rng.normal(size=(53,2));y=rng.normal(size=53);t=np.array([-.8,.7,.2,-.3,.4])
        value,grad=objective(t,x,y)
        fd=approx_fprime(t,lambda q:objective(q,x,y)[0],1e-7)
        np.testing.assert_allclose(grad,fd,atol=5e-6,rtol=1e-5)

    def test_likelihood_label_symmetry(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(40,2));y=rng.normal(size=40)
        t=np.array([-1.,1.,.3,-.2,.8]);q=t.copy();q[:2]=t[:2][::-1];q[4]=-t[4]
        self.assertAlmostEqual(objective(t,x,y)[0],objective(q,x,y)[0],places=12)

    def test_constant_solver_retains_incumbent(self):
        rng=np.random.default_rng(12);x=np.c_[np.ones(96),rng.normal(size=(96,2))]
        y=np.repeat([-2.,2.],48)+x[:,1:]@np.array([.2,-.1])+rng.normal(0,.2,96)
        model=solve_constant(x,y,.2,starts=4,maxiter=80)
        self.assertLessEqual(model['final_two_plane_nll'],model['initial_two_plane_nll']+1e-9)
        self.assertEqual(model['k'],2)
        self.assertLess(abs(abs(np.diff(model['means'])[0])-4),.15)

    def test_structural_metric_detects_collapse(self):
        _,_,e=v14.make(9150003,4,0,1.,1.)
        q=e['gt_clean_xyz_world'].copy()
        self.assertLess(structural(q,e)['gap_absolute_relative_error'],1e-10)
        q[:,2]=np.mean([p[0] for p in e['surface_rectangles_mm']])/1000
        m=structural(q,e)
        self.assertAlmostEqual(m['gap_absolute_relative_error'],1.)
        self.assertEqual(m['severe_gap_contraction'],1.)

    def test_structural_does_not_use_method_k(self):
        _,_,e=v14.make(9150003,4,0,1.,.25)
        a=structural(e['gt_clean_xyz_world'],e)
        e['reported_model_k']=1
        self.assertEqual(a,structural(e['gt_clean_xyz_world'],e))

    def test_structural_selection_can_be_empty(self):
        agg={k:dict(failed=0,surface_accuracy_mean_mm=.1,matched_point_rms_mm=.2,
                    gap_absolute_relative_error=.2,severe_gap_contraction=0)
             for k in ['constant_36','spatial_36','rimls_8.0']}
        self.assertTrue(all(v is None for v in choose(agg)['structural'].values()))


if __name__ == '__main__':
    unittest.main()
