import unittest
import numpy as np
from common import Config, generate, evaluate, suite


class Tests(unittest.TestCase):
    def test_ambiguity_exact(self):
        a,ga=generate(Config(family='ambiguous_single',gap=8.),3)
        b,gb=generate(Config(family='ambiguous_dual',gap=8.),3)
        self.assertTrue(np.array_equal(a.xyz_mm,b.xyz_mm))
        self.assertTrue(np.array_equal(a.frame,b.frame))
        self.assertFalse(np.array_equal(ga['xyz_mm'],gb['xyz_mm']))
    def test_truth_metric(self):
        a,g=generate(Config(normal_degrees=3.),7)
        score=evaluate(g['xyz_mm']@g['rotation'].T,g['bias_mm'],g,a)
        self.assertLess(score['normal_mae_mm'],1e-12)
        self.assertLess(score['gap_abs_error_mm'],1e-12)
    def test_rays_first_hit(self):
        a,g=generate(Config(family='raycast'),9)
        p=g['xyz_mm'];o=g['origins_mm'];back=g['labels']==0
        # A rear hit must not have intersected the front rectangle first.
        t=(6.-o[back,2])/(p[back,2]-o[back,2]); front=o[back]+t[:,None]*(p[back]-o[back])
        occluded=(front[:,0]>=-15)&(front[:,0]<=60)&(abs(front[:,1])<=50)
        self.assertFalse(occluded.any())
    def test_labels_nonempty(self):
        for _,c in suite('development'):
            a,g=generate(c,19)
            self.assertTrue(np.isfinite(a.xyz_mm).all())
            if g['gap'] is not None:self.assertEqual(len(np.unique(g['labels'])),2)


if __name__=='__main__': unittest.main()
