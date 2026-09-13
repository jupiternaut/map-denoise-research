import unittest
import numpy as np
from scipy.optimize import approx_fprime
from models import (fit, fixed_objective, fixed_geometry, fixed_constant,
                    original_spatial, solve_constant, attach_posterior)
from generator import make
from metrics import score_new, dependence_diagnostics, error_decomposition


class MechanismTests(unittest.TestCase):
    def test_paired_marginals_and_noise(self):
        pairs = [make(9160003, 4., 1., lam, n=96) for lam in (0., .5, 1.)]
        for inp, e in pairs:
            np.testing.assert_array_equal(inp['design'], pairs[0][0]['design'])
            np.testing.assert_array_equal(e['eps_mm'], pairs[0][1]['eps_mm'])
            for f in (0,1):
                self.assertEqual(int(e['gt_layer'][inp['scan_id']==f].sum()), 24)
        self.assertFalse(np.array_equal(pairs[0][1]['gt_layer'],pairs[2][1]['gt_layer']))

    def test_finite_common_support(self):
        for lam in (0., .5, 1.):
            inp, e = make(9160004, 8., 2., lam, n=96)
            q = e['gt_clean_xyz_world']*1000.
            self.assertTrue(np.all(abs(q[:,:2])<=np.array([60.,50.])))
            np.testing.assert_allclose(q[:,2], e['gt_layer']*8.)
            self.assertNotIn('gt_layer', inp)

    def test_fixed_gradient(self):
        y = np.random.default_rng(7).normal(size=55)
        t = np.array([-.8,.9,.4])
        fd = approx_fprime(t, lambda q:fixed_objective(q,y)[0], 1e-7)
        np.testing.assert_allclose(fd, fixed_objective(t,y)[1], atol=5e-6, rtol=1e-5)

    def test_fixed_label_symmetry(self):
        y = np.random.default_rng(8).normal(size=55)
        self.assertAlmostEqual(fixed_objective(np.array([-.8,.9,.4]),y)[0],
                               fixed_objective(np.array([.9,-.8,-.4]),y)[0], places=12)

    def test_fixed_geometry_stationary(self):
        rng = np.random.default_rng(9)
        r = rng.uniform(size=(30,2));r /= r.sum(1)[:,None]
        y = rng.normal(size=30)
        means = fixed_geometry(y,r)
        np.testing.assert_allclose((r*(means-y[:,None])).sum(0),0.,atol=1e-12)

    def test_free_versions_exact(self):
        inp, _ = make(9160005, 4., 1., .5, n=96)
        x,y = inp['design'],inp['height_mm']
        for method,old in [('spatial_free',original_spatial(x,y,1.,iterations=36)),
                           ('constant_free',solve_constant(x,y,1.))]:
            model = fit(x,y,1.,method)
            np.testing.assert_array_equal(model['prediction'], old['prediction'])
            self.assertEqual(model['score'],old['score'])
            self.assertEqual(model['k'],old['k'])

    def test_oracle_contract(self):
        inp, _ = make(9160006, 4.,1.,1.,n=96)
        with self.assertRaises(ValueError):
            fit(inp['design'],inp['height_mm'],1.,'spatial_free',oracle_slope=np.zeros(2))
        with self.assertRaises(ValueError):
            fit(inp['design'],inp['height_mm'],1.,'spatial_oracle_slope')

    def test_oracle_nonzero_slope(self):
        inp,_ = make(9160007,8.,.2,1.,n=96)
        x,y = inp['design'],inp['height_mm']
        slope=np.array([.3,-.5])
        for method in ('constant_oracle_slope','spatial_oracle_slope'):
            m=fit(x,y+x[:,1:]@slope,.2,method,oracle_slope=slope)
            self.assertEqual(m['k'],2)
            np.testing.assert_array_equal(m['slope'],slope)
            self.assertLess(abs(np.ptp(m['means'])-8.),.2)
            np.testing.assert_allclose(m['posterior'].sum(1),1.,atol=1e-12)

    def test_oracle_incumbent(self):
        inp,_=make(9160008,2.,1.,.5,n=96)
        m=fixed_constant(inp['design'],inp['height_mm'],1.,starts=4,maxiter=40)
        self.assertLessEqual(m['final_two_plane_nll'],m['initial_two_plane_nll']+1e-10)

    def test_ols_identity(self):
        for lam in (0.,.5,1.):
            inp,e=make(9160009,8.,2.,lam,n=96)
            self.assertLess(dependence_diagnostics(inp,e)['ols_identity_error'],1e-12)

    def test_gap_identity_and_error_decomposition(self):
        inp,e=make(9160010,4.,1.,.5,n=96)
        m=fit(inp['design'],inp['height_mm'],1.,'spatial_free')
        q=inp['xyz_world'].copy();q[:,2]=m['prediction']/1000.
        metrics=score_new(inp,e,dict(**m,xyz_world=q))
        self.assertLess(metrics['gap_identity_error_mm'],1e-12)
        self.assertLess(metrics['decomposition_error_mm2'],1e-12)

    def test_wrong_labels_shrink_source_gap_without_moving_planes(self):
        inp,e=make(9160011,4.,1.,1.,n=96)
        labels=e['gt_layer'];g=labels.copy()
        g[np.flatnonzero(labels==0)[:12]]=1
        g[np.flatnonzero(labels==1)[:12]]=0
        q=inp['xyz_world'].copy();q[:,2]=4*g/1000.
        art=dict(xyz_world=q,k=2,means=np.array([0.,4.]),slope=np.zeros(2),groups=g)
        s=score_new(inp,e,art)
        self.assertAlmostEqual(s['fitted_gap_error_mm'],0.)
        self.assertAlmostEqual(s['source_gap_mm'],2.)
        self.assertAlmostEqual(s['assignment_error'],.25)
        self.assertAlmostEqual(s['counterfactual_geometry_mse_mm2'],0.)

    def test_decomposition_cross_term_is_not_omitted(self):
        reference=np.zeros((4,3));cf=np.ones((4,3))*.001;out=cf*2
        d=error_decomposition(out,cf,reference)
        self.assertAlmostEqual(d['total_mse_mm2'],12.)
        self.assertAlmostEqual(d['geometry_assignment_cross_mm2'],6.)
        self.assertLess(d['decomposition_error_mm2'],1e-12)

    def test_single_split_is_not_assignment_accuracy(self):
        inp,e=make(9160012,0.,1.,0.,n=96)
        q=inp['xyz_world'].copy();q[:,2]=0.
        art=dict(xyz_world=q,k=1,means=np.zeros(1),slope=np.zeros(2),groups=np.zeros(96,int))
        s=score_new(inp,e,art)
        self.assertEqual(s['single_false_split'],0)
        self.assertNotIn('assignment_error',s)


if __name__=='__main__':
    unittest.main()
