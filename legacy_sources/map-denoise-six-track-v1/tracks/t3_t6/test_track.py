import sys, unittest
import numpy as np
sys.path.insert(0,'/home/grf/Documents/Codex/2026-09-10/map-denoise-v0')
from geometry import FAMILIES,sample,distances,rotation,evaluate
from local_operator import denoise,oracle_association,oracle_normal
from oracle_ablation import matched
from parallel_geometry_v5.association.operator import denoise as old_a

class TrackTests(unittest.TestCase):
    def test_exact_surfaces(self):
        for family in FAMILIES:
            p,_,labels=sample(family,2000,.007,.65,np.random.default_rng(25))
            d=distances(family,p,.007)
            self.assertLess(float(np.max(d[np.arange(len(p)),labels])),1e-12)
    def test_rotation(self):
        r=rotation(789);self.assertTrue(np.allclose(r.T@r,np.eye(3)))
        self.assertGreater(np.linalg.det(r),.99)
    def test_metrics_detect_collapsed_layers(self):
        p,_,labels=sample('parallel_plates',3000,.009,.5,np.random.default_rng(14))
        collapsed=p.copy();collapsed[:,2]=0
        m=evaluate('parallel_plates',collapsed,.009,p,labels,len(p))
        self.assertAlmostEqual(m['gap_intrusion_fraction'],1.)
        self.assertAlmostEqual(m['surface_mean_mm'],4.5)
        self.assertAlmostEqual(m['min_component_recall'],0.)
    def test_operator_legal_input(self):
        p,_,_=sample('parallel_plates',180,.009,.5,np.random.default_rng(9))
        p+=np.random.default_rng(19).normal(0,.001,p.shape);before=p.copy()
        out,diag=denoise(p,config={'iterations':1,'em_iterations':4})
        self.assertTrue(np.array_equal(p,before));self.assertEqual(out.shape,p.shape)
        self.assertTrue(np.isfinite(out).all());self.assertFalse(diag['requires_ground_truth'])
    def test_oracle_no_shape_changes(self):
        p,n,l=sample('rod_plate',180,.007,.5,np.random.default_rng(9))
        for fn,arg in ((oracle_normal,n),(oracle_association,l)):
            out,d=fn(p,arg,iterations=1)
            self.assertEqual(out.shape,p.shape);self.assertFalse(d['deployable'])
    def test_matched_oracle_baseline(self):
        p,_,_=sample('concentric_shells',180,.007,.5,np.random.default_rng(41))
        p+=np.random.default_rng(44).normal(0,.001,p.shape)
        expected,_=old_a(p,{'fallback':'bilateral','bic_gain':1e6})
        self.assertTrue(np.allclose(matched(p),expected,atol=1e-12,rtol=0))
    def test_rigid_rotation_equivariance(self):
        p,_,_=sample('parallel_plates',240,.009,.5,np.random.default_rng(58))
        p+=np.random.default_rng(64).normal(0,.001,p.shape)
        r=rotation(35);cfg={'iterations':1,'em_iterations':4}
        a,_=denoise(p,config=cfg);b,_=denoise(p@r.T,config=cfg)
        self.assertTrue(np.allclose(a,b@r,atol=1e-9,rtol=0))

if __name__=='__main__':unittest.main()
