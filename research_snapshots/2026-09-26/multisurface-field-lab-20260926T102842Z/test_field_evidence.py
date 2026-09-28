import unittest
import numpy as np
from field_evidence import (candidate_offsets,common_cost,score_positions,
                            weighted_common_cost,visibility_cost)


class FieldEvidenceTests(unittest.TestCase):
    def test_nine_physical_offsets(self):
        p=np.array([[0.,0.,10.],[0.,0.,11.]])
        got=candidate_offsets(p,p+[0,0,2],p-[0,0,1],np.zeros(3))
        np.testing.assert_allclose(got,[[0,2,-1,1,-.5,-6,-3,3,6]]*2)

    def test_common_source_set_across_all_candidates(self):
        s=np.array([[[.2,.4]],[[.6,.8]],[[np.nan,1.]]])
        r=common_cost(s)
        np.testing.assert_allclose(r['cost'],[[.6,.4]])
        self.assertEqual(r['common_count'][0],2)

    def test_missing_support_not_zero_cost(self):
        s=np.full((4,3,9),np.nan);s[0]=.9
        r=common_cost(s)
        self.assertFalse(r['supported'].any())
        np.testing.assert_array_equal(r['cost'],np.ones((3,9)))

    def test_zero_gaps_equal_matched_mean(self):
        rng=np.random.default_rng(17);s=rng.uniform(-1,1,(4,7,9));s[2,2,1]=np.nan
        np.testing.assert_allclose(weighted_common_cost(s,np.zeros_like(s))['cost'],common_cost(s)['cost'])

    def test_occluded_view_is_downweighted_not_deleted(self):
        s=np.array([[[.9]],[[.1]]]);gap=np.array([[[0.]],[[.75]]])
        r=weighted_common_cost(s,gap)
        expected=1.-(.9+np.exp(-1)*.1)/(1+np.exp(-1))
        self.assertAlmostEqual(r['cost'][0,0],expected)
        self.assertEqual(r['common_count'][0],2)
        self.assertGreater(r['weights'][1,0,0],0.)

    def test_large_gap_stable_normalization(self):
        s=np.array([[[.9]],[[.1]]]);gap=np.array([[[1000.]],[[1000.75]]])
        r=weighted_common_cost(s,gap)
        self.assertTrue(np.isfinite(r['cost']).all())
        self.assertAlmostEqual(r['cost'][0,0],1.-(.9+np.exp(-1)*.1)/(1+np.exp(-1)))

    def test_common_mask_cannot_be_changed_by_visibility(self):
        s=np.ones((3,2,2))
        with self.assertRaises(ValueError):
            weighted_common_cost(s,np.zeros_like(s),np.zeros((3,2),bool))

    def test_positions_are_directly_rescored(self):
        p=np.array([[0.,0.,10.]])
        offsets=np.array([[0.,.3,2.]])
        def mock(p,n,o,r,v,**kw):
            self.assertEqual(kw['patch_radius'],3)
            self.assertEqual(kw['mode'],'tangent')
            np.testing.assert_array_equal(o,offsets)
            cand=p[:,None]+o[:,:,None]*np.array([0.,0.,1.])
            return dict(scores=np.ones((4,1,3))*.5,candidates=cand)
        r=score_positions(p,offsets,np.array([[0.,0.,1.]]),{},[{}]*4,mock)
        np.testing.assert_allclose(r['candidates'][0,:,2],[10,10.3,12])

    def test_self_rendered_gap_and_unknown_are_explicit(self):
        # One point per row: second point sits behind first in the same pixel.
        cam=dict(P=np.array([[1.,0,0,0],[0,1.,0,0],[0,0,1.,0]]),
                 center=np.zeros(3),image=np.zeros((10,10)))
        render=np.array([[5.,5.,10.],[6.,6.,12.]])
        candidates=np.stack([render,render+np.array([300.,0,0])],axis=1)
        s=np.ones((2,2,2))*.5
        r=visibility_cost(s,candidates,render,[cam,cam])
        self.assertTrue(r['self_front'][0,0,0])
        self.assertFalse(r['self_front'][0,1,0])
        self.assertAlmostEqual(float(r['gap_mm'][0,1,0]),2.)
        self.assertTrue(r['unknown'][0,0,1])
        self.assertEqual(r['weights'][0,0,1],1.)


if __name__=='__main__':
    unittest.main()
