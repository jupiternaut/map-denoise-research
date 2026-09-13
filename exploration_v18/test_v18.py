import unittest
import numpy as np
import v18_operator as op

class Tests(unittest.TestCase):
    def setUp(self):
        self.x=np.c_[np.ones(6),np.linspace(-1,1,6),np.zeros(6)]
        self.m=dict(k=2,means=np.array([0.,4.]),slope=np.array([.2,0.]),gate=np.zeros(12),family='R',
                    feature_center=np.zeros(2),feature_scale=np.ones(2))
        self.y=np.array([0.,.5,1.,2.,3.,4.])
        self.m.update(op.base.evaluate(self.m,self.x,self.y,1.))

    def test_projection_and_conditional_risk(self):
        hard=op.project(self.m,self.x,1);mean=op.project(self.m,self.x,0);half=op.project(self.m,self.x,.5)
        np.testing.assert_allclose(half['prediction'],(hard['prediction']+mean['prediction'])/2)
        levels=self.m['means'][None,:]+(self.x[:,1:]@self.m['slope'])[:,None]
        for m in (hard,mean,half):
            risk=np.sum(self.m['posterior']*(m['prediction'][:,None]-levels)**2,axis=1)
            np.testing.assert_allclose(risk,mean['posterior_model_variance_mm2']+(m['prediction']-mean['prediction'])**2)
        self.assertTrue(np.all(hard['output_distance_to_fitted_surface_mm']==0))

    def test_incumbent_preserved(self):
        s=dict(k=1,means=np.zeros(1),slope=np.zeros(2),bic=1e7)
        pool={f:dict(s,bic=1e7) for f in ('S','C','L')};pool['R']=dict(self.m,bic=1e7)
        m,d=op.retain(pool,self.m,self.x,self.y,1.)
        self.assertTrue(d['replaced_r']);np.testing.assert_array_equal(m['means'],self.m['means'])
        self.assertLess(d['r_bic_after'],d['r_bic_before'])

    def test_single_and_no_degradation(self):
        s=dict(k=1,means=np.array([1.]),slope=np.zeros(2),posterior=np.ones((6,1)),groups=np.zeros(6,int),bic=-100.)
        pool={f:dict(s,bic=-100.) for f in ('S','C','L')};pool['R']=dict(self.m,bic=-100.)
        m,d=op.retain(pool,s,self.x,self.y,1.)
        self.assertFalse(d['replaced_r']);self.assertEqual(d['selected_family'],'S')
        np.testing.assert_array_equal(op.project(m,self.x,0)['prediction'],op.project(m,self.x,1)['prediction'])

if __name__=='__main__':unittest.main(verbosity=2)
