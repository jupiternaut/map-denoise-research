import unittest
import numpy as np
from registration_baselines import estimate


class RegistrationTests(unittest.TestCase):
    def cloud(self):
        rng=np.random.default_rng(74)
        q=rng.normal(size=(100,3))*.03
        p=np.r_[q+[0,0,.0007],q-[0,0,.0007]]
        return p,np.repeat([3,7],len(q)),np.r_[q,q]

    def test_invalid_inputs(self):
        for p,f,s in [(np.zeros((3,2)),np.ones(3,int),1),
                      (np.zeros((3,3)),np.ones(3,float),1),
                      (np.zeros((3,3)),np.ones(3,int),0)]:
            with self.assertRaises(ValueError): estimate(p,f,s)

    def test_single_scan_identity(self):
        p,f,_=self.cloud()
        q,info=estimate(p,np.ones(len(p),int),1)
        np.testing.assert_array_equal(q,p)
        self.assertEqual(info['status'],'UNSUPPORTED')

    def test_rigid_and_input_order_preserved(self):
        p,f,ref=self.cloud(); initial=p.copy()
        out,info=estimate(p,f,1)
        np.testing.assert_array_equal(p,initial)
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(np.sqrt(np.mean((out-ref)**2)),1e-6)
        origin=np.array(info['origin_world_m']); shift=np.array(info['gauge_common_translation_m'])
        for sid,t in zip(info['scan_ids'],np.array(info['applied_raw_transforms_centered'])):
            q=p[f==sid]
            expected=(q-origin)@t[:3,:3].T+t[:3,3]+origin+shift
            np.testing.assert_allclose(out[f==sid],expected,atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(np.diff(q,axis=0),axis=1),
                                       np.linalg.norm(np.diff(out[f==sid],axis=0),axis=1),atol=1e-12)
        np.testing.assert_allclose(out.mean(axis=0),p.mean(axis=0),atol=1e-12)

    def test_global_origin_and_point_permutation(self):
        p,f,_=self.cloud(); a,_=estimate(p,f,1)
        shift=np.array([11.,-4.,7.]); b,_=estimate(p+shift,f,1)
        np.testing.assert_allclose(a,b-shift,atol=1e-7)
        order=np.random.default_rng(10).permutation(len(p)); c,_=estimate(p[order],f[order],1)
        np.testing.assert_allclose(a[order],c,atol=1e-7)

    def test_disconnected_clouds_not_silently_translated(self):
        p,f,_=self.cloud(); p[f==7]+=[100,0,0]
        a,info=estimate(p,f,1)
        np.testing.assert_array_equal(a,p)
        self.assertEqual(info['status'],'UNSUPPORTED')


if __name__=='__main__': unittest.main()
