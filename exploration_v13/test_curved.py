import unittest
import numpy as np
from scipy.optimize import minimize_scalar
from curved_data import make,scores

class CurvedTests(unittest.TestCase):
    def test_clean_surface_zero(self):
        _,_,ev=make(8111,2.,4.,8.)
        self.assertLess(scores(ev['gt_clean_xyz_world'],ev)['surface_accuracy_mean_mm'],1e-10)
    def test_distance_against_scalar_optimizer(self):
        rng=np.random.default_rng(8112);q=rng.uniform([-80,-60,-10],[80,60,20],(25,3));a=8/3600
        ev=dict(parabola_a_per_mm=a,surface_rectangles_mm=[[2,-60,60,-50,50]],gt_clean_xyz_world=q/1000)
        reference=[]
        for x,y,z in q:
            opt=minimize_scalar(lambda u:(u-x)**2+(a*u*u+2-z)**2,bounds=(-60,60),method='bounded',options={'xatol':1e-12})
            value=min(opt.fun,(-60-x)**2+(a*3600+2-z)**2,(60-x)**2+(a*3600+2-z)**2)
            reference.append(np.sqrt(value+(np.clip(y,-50,50)-y)**2))
        self.assertAlmostEqual(scores(q/1000,ev)['surface_accuracy_mean_mm'],float(np.mean(reference)),places=8)
if __name__=='__main__':unittest.main()
