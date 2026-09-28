import unittest
import numpy as np
from step_observation import GRID, common_view_choice, parabolic_vertex, revise_path


class StepObservationTests(unittest.TestCase):
    def test_common_views_not_per_step_best(self):
        s = np.array([[[.4,.5,.6,.7,.8]], [[.4,.3,.2,.1,0]], [[np.nan,1,1,1,1]]])
        r = common_view_choice(s)
        self.assertEqual(r['count'][0],2)
        self.assertFalse(r['common'][2,0])
        self.assertEqual(r['step'][0],0.)

    def test_insufficient_common_retains_old_endpoint(self):
        s = np.array([[[.1,.2,.3,.4,.9]],[[np.nan,.2,.3,.4,.9]]])
        self.assertEqual(common_view_choice(s)['step'][0],1.)

    def test_all_missing_retains_old_endpoint(self):
        r = common_view_choice(np.full((4,3,5),np.nan))
        np.testing.assert_array_equal(r['step'],np.ones(3))

    def test_order_invariant_and_no_mutation(self):
        rng=np.random.default_rng(4)
        s=rng.uniform(-1,1,(4,7,5));s[2,3,1]=np.nan;before=s.copy()
        r=common_view_choice(s);other=common_view_choice(s[::-1])
        np.testing.assert_array_equal(r['step'],other['step'])
        np.testing.assert_array_equal(s,before)

    def test_exact_parabolic_vertex(self):
        c=(GRID[None,:]-.4)**2
        t,valid=parabolic_vertex(c,np.array([2]),np.array([True]))
        self.assertTrue(valid[0]);self.assertAlmostEqual(t[0],.4)

    def test_boundary_or_flat_no_interpolation(self):
        c=np.array([[0,1,2,3,4],[0,0,0,0,0]],float)
        _,valid=parabolic_vertex(c,np.array([0,2]),np.ones(2,bool))
        self.assertFalse(valid.any())

    def test_actual_coordinate_rescore(self):
        p=np.array([[0.,0.,10.]])
        q=np.array([[0.,0.,12.]])
        calls=[]
        def score(points,normals,offsets,ref,sources,**kwargs):
            calls.append(offsets.copy())
            rays=points-ref['center'];rays/=np.linalg.norm(rays,axis=1,keepdims=True)
            cand=points[:,None]+offsets[:,:,None]*rays[:,None]
            scores=1.-(offsets/2.-.4)**2
            return dict(scores=np.stack([scores]*len(sources)),candidates=cand)
        r=revise_path(p,q,np.array([[0.,0.,1.]]),dict(center=np.zeros(3)),[{},{}],scorer=score)
        self.assertEqual(len(calls),2)
        np.testing.assert_allclose(calls[1],[[.8]])
        np.testing.assert_allclose(r['step_grid'],[[0,0,11.]])
        np.testing.assert_allclose(r['step_quadratic'],[[0,0,10.8]])
        self.assertTrue(r['quadratic_accepted'][0])

    def test_quad_loses_common_support_keeps_grid(self):
        p=np.array([[0.,0.,10.]])
        def score(points,normals,offsets,ref,sources,**kwargs):
            rays=points-ref['center'];rays/=np.linalg.norm(rays,axis=1,keepdims=True)
            scores=np.stack([1.-(offsets/2.-.4)**2]*len(sources))
            if offsets.shape[1]==1:scores[0,:,:]=np.nan
            return dict(scores=scores,candidates=points[:,None]+offsets[:,:,None]*rays[:,None])
        r=revise_path(p,p+[0,0,2],np.array([[0.,0.,1.]]),dict(center=np.zeros(3)),[{},{}],scorer=score)
        self.assertFalse(r['quadratic_accepted'][0])
        np.testing.assert_array_equal(r['step_grid'],r['step_quadratic'])

    def test_nonray_candidate_rejected(self):
        with self.assertRaises(ValueError):
            revise_path(np.array([[0.,0.,10.]]),np.array([[1.,0.,10.]]),np.array([[0.,0.,1.]]),
                        dict(center=np.zeros(3)),[{},{}],scorer=lambda *a,**k:None)


if __name__=='__main__':
    unittest.main()
