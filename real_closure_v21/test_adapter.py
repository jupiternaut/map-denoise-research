import unittest
import numpy as np
from run import local, transform, v18, project

class Tests(unittest.TestCase):
    def test_frame_roundtrip(self):
        q=np.random.default_rng(1).normal(size=(100,3))*[10,20,1]+[200,30,700]
        c,b,l,h,e=local(q)
        np.testing.assert_allclose(l@b.T+c,q,atol=1e-10)
        np.testing.assert_allclose(b.T@b,np.eye(3),atol=1e-12)
        self.assertGreater(h,0)

    def test_metric_transform(self):
        q=np.array([[1,2,3.],[-1,0,0]])
        m=np.eye(4);m[:3,:3]*=300;m[:3,3]=[5,6,7]
        np.testing.assert_array_equal(transform(q,m),q*300+[5,6,7])

    def test_v18_tangent_unchanged_and_finite(self):
        rng=np.random.default_rng(2);q=np.c_[rng.uniform(-20,20,(96,2)),rng.normal(0,.2,96)]
        out,m,d=v18.filter_local(q,.2)
        np.testing.assert_array_equal(out[:,:2],q[:,:2])
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out.shape,q.shape)

    def test_external_output_identity_order(self):
        rng=np.random.default_rng(3);q=np.c_[rng.uniform(-.02,.02,(96,2)),rng.normal(0,.0002,96)]
        for kind in ('apss','rimls'):
            out=project(q,kind,4.)
            self.assertEqual(out.shape,q.shape)
            self.assertTrue(np.isfinite(out).all())

if __name__=='__main__':unittest.main()
