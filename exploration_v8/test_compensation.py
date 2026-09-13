import inspect
import unittest
import numpy as np
from scipy.integrate import quad
from compensation import interval_noise_mean, selection_bias, compensate_frozen, estimate

class CompensationTests(unittest.TestCase):
    def test_half_normal(self):
        means=np.zeros((2,1)); prior=np.ones((2,1))
        b,p,_=interval_noise_mean(means,prior,np.array([-np.inf,0.]),np.array([0.,np.inf]),1.)
        np.testing.assert_allclose(b,[-np.sqrt(2/np.pi),np.sqrt(2/np.pi)],atol=1e-14)
        np.testing.assert_allclose(p,.5,atol=1e-14)

    def test_full_interval_zero(self):
        b,p,_=interval_noise_mean(np.array([[-2.,1.]]),np.array([[.2,.8]]),
                                  np.array([-np.inf]),np.array([np.inf]),1.)
        np.testing.assert_allclose(b,0.,atol=1e-14);np.testing.assert_allclose(p,1.)

    def test_independent_quadrature(self):
        means=np.array([[-1.1,2.2,.3]]); prior=np.array([[.15,.55,.3]])
        sigma=.8; lower=-.2; upper=1.4
        b,p,_=interval_noise_mean(means,prior,np.array([lower]),np.array([upper]),sigma)
        density=lambda e: np.exp(-.5*(e/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
        mass=first=0.
        for m,q in zip(means[0],prior[0]):
            mass+=q*quad(density,lower-m,upper-m)[0]
            first+=q*quad(lambda e:e*density(e),lower-m,upper-m)[0]
        self.assertAlmostEqual(p[0],mass,places=12);self.assertAlmostEqual(b[0],first/mass,places=12)

    def test_mixture_not_single_plane_collapse(self):
        x=np.array([[1.,0,0],[1.,0,0]])
        beta=np.array([[-4.,0,0],[4.,0,0]])
        b,_=selection_bias(x,beta,np.ones((2,2),bool),np.array([0,1]),1.)
        self.assertLess(max(abs(b)),.0003)
        np.testing.assert_allclose(b[0],-b[1],atol=1e-14)

    def test_coordinate_and_sigma_scaling(self):
        x=np.array([[1.,.2,.1],[1.,.4,-.2]])
        beta=np.array([[-1.,.1,.2],[1.,.2,.1]])
        mask=np.ones((2,2),bool);g=np.array([0,1])
        b,_=selection_bias(x,beta,mask,g,1.)
        c,_=selection_bias(x,beta*5,mask,g,5.)
        np.testing.assert_allclose(c,5*b,atol=1e-12)

    def test_translation_invariance(self):
        x=np.array([[1.,.2,.1],[1.,.4,-.2]])
        beta=np.array([[-1.,.1,.2],[1.,.2,.1]]); shifted=beta.copy();shifted[:,0]+=100.
        mask=np.ones((2,2),bool);g=np.array([0,1])
        b,_=selection_bias(x,beta,mask,g,1.)
        c,_=selection_bias(x,shifted,mask,g,1.)
        np.testing.assert_allclose(c,b,atol=1e-12)

    def test_tail_probability_stability(self):
        b,p,_=interval_noise_mean(np.zeros((1,1)),np.ones((1,1)),np.array([5.]),np.array([6.]),1.)
        self.assertGreater(p[0],0.);self.assertTrue(5.<b[0]<6.)

    def test_duplicate_means_do_not_divide_zero(self):
        b,diag=selection_bias(np.array([[1.,0,0]]),np.zeros((2,3)),np.ones((1,2),bool),np.array([1]),1.)
        np.testing.assert_allclose(b,0.);np.testing.assert_allclose(diag['modeled_selection_probability'],1.)

    def test_independent_noise_subtraction_variance(self):
        rng=np.random.default_rng(913811)
        a=rng.normal(size=200000);b=rng.normal(size=200000)
        self.assertLess(abs(np.var(a-b)/np.var(a)-2.),.03)

    @staticmethod
    def fixture():
        rng=np.random.default_rng(913809);n=120
        order=rng.permutation(n); local=np.c_[rng.uniform(-10,10,(n,2)),rng.normal(size=n)]
        world=np.empty((n,3));world[order]=local/1000.
        support=np.ones(n,bool);support[::7]=False
        groups=(local[:,2]>=0).astype(int);world_groups=np.empty(n,int);world_groups[order]=groups
        state=dict(world=world,order=order,active=np.arange(n),support=support,normal=np.array([0.,0.,1.]),
                   local=local,design=np.c_[np.ones(n),local[:,:2]/10.],corrected=local[:,2].copy(),weights=np.ones(n))
        artifacts=dict(group_ids=world_groups,coefficients=np.array([[-.1,0,0],[.1,0,0]]),candidate_mask=np.ones((n,2),bool))
        return state,artifacts

    def test_output_support_and_immutable(self):
        state,a=self.fixture();before={k:v.copy() for k,v in state.items()}
        out,info,art=compensate_frozen(state,a,1.)
        for k in state:np.testing.assert_array_equal(state[k],before[k])
        np.testing.assert_array_equal(out[~art['support_mask']],state['world'][~art['support_mask']])
        self.assertTrue(np.isfinite(out).all());self.assertEqual(info['truth_fields_used'],[])

    def test_deterministic(self):
        state,a=self.fixture()
        x,_,_=compensate_frozen(state,a,1.);y,_,_=compensate_frozen(state,a,1.)
        np.testing.assert_array_equal(x,y)

    def test_none_matches_direct_wls(self):
        state,a=self.fixture();out,_,_=compensate_frozen(state,a,1.,mode='none')
        groups=a['group_ids'][state['order']]
        expected=state['world'].copy()
        for g in (0,1):
            rows=groups==g;x=state['design'][rows]
            beta=np.linalg.lstsq(x,state['corrected'][rows],rcond=1e-12)[0]
            use=rows&state['support'];expected[state['order'][use],2]=(state['design'][use]@beta)/1000.
        np.testing.assert_allclose(out,expected,atol=1e-14)

    def test_public_signature(self):
        self.assertEqual(list(inspect.signature(estimate).parameters),
                         ['xyz_world_m','scan_id','sigma_mm','budget','mode'])

    def test_invalid_sigma_and_candidate(self):
        with self.assertRaises(ValueError):
            interval_noise_mean(np.zeros((1,1)),np.ones((1,1)),np.array([-1.]),np.array([1.]),0.)
        with self.assertRaises(ValueError):
            selection_bias(np.array([[1.,0,0]]),np.zeros((1,3)),np.zeros((1,1),bool),np.array([0]),1.)

if __name__=='__main__':unittest.main()
