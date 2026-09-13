import unittest
import numpy as np
from atlas import plane,project,decode

class Tests(unittest.TestCase):
    def test_rigid_plane_projection(self):
        rng=np.random.default_rng(192);q=np.c_[rng.normal(size=(100,2)),np.full(100,3.)]
        np.testing.assert_allclose(project(q,plane(q)),q,atol=1e-12)
    def test_preserve_two_planes(self):
        rng=np.random.default_rng(193);g=np.repeat([0,1],100);q=np.c_[rng.normal(size=(200,2))*10,4*g]
        mask=np.ones(200,bool);mask[0]=False
        p=dict(mm=q,mask=mask,labels=g,k=2,prior=np.eye(2)[g]*.96+.02,reference=[plane(q[g==i]) for i in (0,1)],seconds=0.)
        state=dict(world=q/1000)
        for mode in ('global_hard','atlas_hard','atlas_tied','atlas_relax'):
            # In the well-separated limit, even soft assignments are exact.
            # At sigma=1, nonzero soft cross-mass need not fix noiseless points.
            out,_,_=decode(state,p,.1,mode,48)
            np.testing.assert_allclose(out,state['world'],atol=1e-12)
            np.testing.assert_array_equal(out[~mask],state['world'][~mask])
if __name__=='__main__':unittest.main()
