import unittest
import numpy as np
from visibility_features import *


def cam(center=(0,0,0)):
    center=np.asarray(center,float);K=np.array([[100.,0,32],[0,100.,32],[0,0,1.]])
    return dict(P=K@np.column_stack([np.eye(3),-center]),center=center,width=64,height=64)


def plane(tilt=False):
    u,v=np.meshgrid(np.arange(28,37),np.arange(28,37));u=u.ravel()+.3;v=v.ravel()+.4
    z=1/(.01+.00004*(u-32)+.00003*(v-32)) if tilt else np.full(len(u),100.)
    return np.column_stack([(u-32)*z/100,(v-32)*z/100,z])


class VisibilityTests(unittest.TestCase):
    def test_front_and_back(self):
        p=plane();r=rasterize(p,cam());idx=np.array([40]);q=p[idx].copy();q[:,2]+=5
        result=query_support(q,idx,r)
        self.assertTrue(result['fit'][0]);self.assertAlmostEqual(result['planegap'][0],5.,places=8)
        q[:,2]-=10;result=query_support(q,idx,r)
        self.assertAlmostEqual(result['planegap'][0],-5.,places=8)

    def test_tilted_plane_detrending(self):
        p=plane(True);idx=np.arange(20,61);r=rasterize(p,cam());result=query_support(p[idx],idx,r)
        self.assertTrue(result['fit'].all());self.assertLess(abs(result['planegap']).max(),1e-10)
        self.assertLess(result['residual'].max(),1e-10)
        self.assertGreater(np.ptp(p[:,2]),1.)

    def test_behind_and_empty_unknown(self):
        p=plane();r=rasterize(p,cam());q=np.array([[0,0,-100.],[20,20,100.]])
        result=query_support(q,np.array([0,1]),r)
        self.assertFalse(result['valid'][0]);self.assertFalse(result['known'].any());self.assertFalse(result['fit'].any())
        self.assertEqual(result['planegap'].sum(),0.)

    def test_self_anchor_and_depth_units(self):
        p=plane();idx=np.array([40]);c=cam();c['P']*=3
        r=rasterize(p,c);result=query_support(p[idx],idx,r)
        self.assertTrue(result['self_front'][0]);self.assertAlmostEqual(r['z'][40],100.)

    def test_exchange_nonmutation_and_missing(self):
        p=plane();ids=np.arange(len(p));candidates=np.stack([p+[0,0,2],p-[0,0,3]],axis=1)
        score=np.full((2,4,len(p),2),.5);score[:,:,:,1]=.8
        before=p.copy();cb=candidates.copy();sb=score.copy();cams=[cam((x,0,0)) for x in (0,1,2,3)]
        a=visibility_features(p,candidates,ids,cams,np.array([0,0,0]),score)
        b=visibility_features(p,candidates[:,::-1],ids,cams,np.array([0,0,0]),score[::-1])
        np.testing.assert_array_equal(a[:,::-1],b)
        np.testing.assert_array_equal(p,before);np.testing.assert_array_equal(candidates,cb);np.testing.assert_array_equal(score,sb)
        missing=visibility_features(p,candidates,ids,cams,np.array([0,0,0]),score*np.nan)
        self.assertFalse(missing[:,:,GEOMETRY_WIDTH:].any());self.assertTrue(np.isfinite(missing).all())


if __name__=='__main__':unittest.main()
