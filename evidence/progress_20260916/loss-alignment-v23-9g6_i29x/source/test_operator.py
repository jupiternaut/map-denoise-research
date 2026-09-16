import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('lossop',Path(__file__).with_name('loss_operator.py'));op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
class Tests(unittest.TestCase):
    def test_project(self):
        uv,z=op.project(np.array([[2.,4.,2.]]),np.eye(4));np.testing.assert_allclose(uv,[[1,2]]);self.assertEqual(z[0],2)
    def test_bilinear(self):
        a=np.zeros((3,3,3),dtype=np.uint8);a[1,:,0]=100;a[2,:,0]=200
        c,v=op.sample(a,np.array([[.5,.5],[3.,1.]]));self.assertAlmostEqual(c[0,0],50/255);self.assertEqual(v.tolist(),[True,False])
    def test_fixed_support(self):
        images=[np.zeros((4,4,3),dtype=np.uint8)]*2;p=np.array([[1.,1.,1.]])
        support=np.ones((2,1),bool)
        a,_=op.photo_loss(p,images,[np.eye(4)]*2,support);b,_=op.photo_loss(p+np.array([10.,0,0]),images,[np.eye(4)]*2,support)
        self.assertEqual(a,0);self.assertEqual(b,1)
    def test_tie_and_missing(self):
        self.assertEqual(op.select({'x':0.,'identity':0.}),'identity');self.assertEqual(op.select({'x':None}),'identity')
    def test_actions(self):
        u,v=np.meshgrid(np.linspace(-1,1,9),np.linspace(-1,1,9));q=np.column_stack([u.ravel(),v.ravel(),np.zeros(u.size)])
        saved=q.copy();out,fit,n=op.actions(q);np.testing.assert_array_equal(q,saved);self.assertEqual(len(out),12)
        for a in out.values():self.assertEqual(a.shape,q.shape);self.assertTrue(np.isfinite(a).all())
        self.assertAlmostEqual(fit['quadratic64'],0,places=20)
if __name__=='__main__':unittest.main()
