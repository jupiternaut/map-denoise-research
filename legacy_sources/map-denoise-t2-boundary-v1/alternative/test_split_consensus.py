import importlib.util
import sys
import unittest
import numpy as np
from split_consensus import METHODS,estimate

p='/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2/experiment.py'
s=importlib.util.spec_from_file_location('frozen_t2',p);old=importlib.util.module_from_spec(s);sys.modules[s.name]=old;s.loader.exec_module(old)

class Tests(unittest.TestCase):
    def test_imbalance_improves(self):
        inp,truth=old.generate('crossed_imbalanced',0)
        out,b,info=estimate(inp,METHODS[0]);metrics=old.evaluate(inp,truth,out,b)
        self.assertLess(metrics['normal_mae_mm'],.3)
        self.assertLess(metrics['layer_gap_abs_error_mm'],.3)
    def test_frame_translation_changes_centers_not_gap(self):
        inp,_=old.generate('crossed_balanced',0)
        out,b,info=estimate(inp,METHODS[0]);changed=old.Input(inp.xyz_mm.copy(),inp.frame,inp.roi,inp.sigma_mm,inp.bias_bound_mm)
        shifts=np.linspace(-3,3,10);changed.xyz_mm[:,2]+=shifts[inp.frame]
        new,bb,ii=estimate(changed,METHODS[0])
        self.assertAlmostEqual(info['gap_mm'],ii['gap_mm'],places=7)
        np.testing.assert_allclose(new,out,atol=1e-7)
    def test_identical_worlds_same_output(self):
        a,_=old.generate('single_ghost_no_anchor',0);b,_=old.generate('thin_segregated_no_anchor',0)
        x,_,_=estimate(a,METHODS[0]);y,_,_=estimate(b,METHODS[0]);np.testing.assert_array_equal(x,y)
    def test_no_input_mutation(self):
        inp,_=old.generate('crossed_balanced',0);before=inp.xyz_mm.copy()
        for method in METHODS:
            out,b,info=estimate(inp,method);self.assertTrue(np.isfinite(out).all());self.assertTrue(np.isfinite(b).all())
        np.testing.assert_array_equal(before,inp.xyz_mm)
    def test_missing_layer_local_filter_preserves_pair(self):
        inp,truth=old.generate('thin_segregated_no_anchor',0);out,b,info=estimate(inp,METHODS[0])
        self.assertLess(old.evaluate(inp,truth,out,b)['layer_gap_abs_error_mm'],.3)
        self.assertEqual(info['route'],'local_only_insufficient_crossed_support')

if __name__=='__main__':unittest.main()
