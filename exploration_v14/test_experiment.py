import unittest
import numpy as np
from experiment import make,layer_scores
from schema import validate_points
from evaluate_v2 import synthetic_geometry

class Checks(unittest.TestCase):
    def test_generation(self):
        for gap in (0,4):
            p,m,e=make(9140001,gap,4,2.,.25);validate_points(p)
            delta=(p['xyz_world']-e['gt_clean_xyz_world'])*1000
            np.testing.assert_allclose(delta[:,2],e['gt_eps_mm']+e['gt_bias_mm'][p['scan_id']],atol=1e-12)
            np.testing.assert_allclose(delta[:,:2],0,atol=1e-12)
            self.assertEqual(len(np.unique(p['source_point_index'])),len(delta))
    def test_clean_metric(self):
        _,_,e=make(9140001,4,4,1.,.25)
        self.assertLess(layer_scores(e['gt_clean_xyz_world'],e)['worst_layer_source_surface_mae_mm'],1e-10)
    def test_independent_metric(self):
        p,_,e=make(9140001,2,4,2.,1.)
        a=synthetic_geometry(p['xyz_world'],{},e);b=layer_scores(p['xyz_world'],e)
        self.assertAlmostEqual(a['surface_accuracy_mean_mm'],b['independent_surface_mae_mm'],places=12)
    def test_dropout_is_nested(self):
        p,_,e=make(9140001,4,4,1.,1.);q,_,f=make(9140001,4,4,1.,.25)
        ids=q['source_point_index'];np.testing.assert_array_equal(q['xyz_world'],p['xyz_world'][ids])
        self.assertEqual(np.sum(f['gt_layer']==0),np.sum(e['gt_layer']==0))

if __name__=='__main__':unittest.main()
