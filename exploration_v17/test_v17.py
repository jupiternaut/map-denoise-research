import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import check_grad
import v17_operator as op

class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(17)
        self.x=np.c_[np.ones(48),self.rng.uniform(-1,1,(48,2))]
        self.y=self.x@np.array([2.,.6,-.4])+self.rng.normal(0,.1,48)

    def test_gradients(self):
        for family in ('C','L','R'):
            f=op.feature_matrix(self.x[:,1:],family,np.zeros(2),np.ones(2))
            t=self.rng.normal(0,.2,4+f.shape[1])
            err=check_grad(lambda t:op.objective(t,self.x[:,1:],self.y,f)[0],
                           lambda t:op.objective(t,self.x[:,1:],self.y,f)[1],t)
            self.assertLess(err,1e-4,family)

    def test_nesting(self):
        matrices=[op.feature_matrix(self.x[:,1:],f,np.zeros(2),np.ones(2)) for f in ('C','L','R')]
        for a,b in zip(matrices,matrices[1:]):np.testing.assert_array_equal(a,b[:,:a.shape[1]])

    def test_folds_and_isolation(self):
        fold=op.folds_from_x(self.x)
        self.assertEqual(set(fold),{0,1,2})
        seen=[]
        def fake(x,y,sigma):
            seen.append((x.copy(),y.copy()))
            p=np.linalg.lstsq(x,y,rcond=None)[0]
            return {f:dict(k=1,family=f,plane=p,means=p[:1],slope=p[1:],gate=np.zeros(1),
                feature_center=np.zeros(2),feature_scale=np.ones(2),starts=1,solver_diagnostics=[],nll=0.,bic=0.) for f in op.FAMILIES}
        with patch.object(op,'fit_pool',fake):cv=op.cross_validation(self.x,self.y,.1)
        for j,(x,y) in enumerate(seen):
            np.testing.assert_array_equal(x,self.x[fold!=j]);np.testing.assert_array_equal(y,self.y[fold!=j])
        for f in op.FAMILIES:self.assertAlmostEqual(cv['scores'][f],cv['per_point'][f].sum())

    def test_pool_nonzero_slope_and_retained_candidates(self):
        pool=op.fit_pool(self.x,self.y,.1)
        np.testing.assert_allclose(pool['S']['slope'],[.6,-.4],atol=.08)
        for f in ('C','L','R'):
            m=pool[f];self.assertLessEqual(m['final_objective'],m['minimum_start_objective']+1e-8)
            self.assertTrue(np.isfinite(m['prediction']).all())
            np.testing.assert_allclose(m['posterior'].sum(1),1.,atol=1e-10)
        self.assertLessEqual(pool['R']['final_objective'],pool['L']['final_objective']+1e-7)
        self.assertLessEqual(pool['L']['final_objective'],pool['C']['final_objective']+1e-7)
        self.assertEqual([pool[f]['starts'] for f in op.FAMILIES],[1,64,13,14])

    def test_complexity_tie(self):
        self.assertEqual(op.choose(dict.fromkeys(op.FAMILIES,0.)),'S')

    def test_local_wrapper(self):
        from v17_apply import filter_local
        points=np.c_[self.x[:,1:]*50.,self.y]
        out,m,d=filter_local(points,.1)
        np.testing.assert_array_equal(out[:,:2],points[:,:2])
        np.testing.assert_array_equal(out[:,2],m['prediction'])
        self.assertEqual(m['selected_family'],op.choose(d['scores']))
        self.assertIsNone(d['cv'])

    def test_bridge_scoring_dictionary_merge(self):
        import tempfile,json
        from pathlib import Path
        import v17_run as run
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);points=np.zeros((4,3))
            inp=dest/'input.npz';ev=dest/'truth.npz';out=dest/'output.npz'
            run.npz(inp,dict(xyz_world=points))
            meta=dict(surface_rectangles_mm=[[0,-1,1,-1,1]],n_true_layers=1,true_gap_mm=0)
            run.npz(ev,dict(gt_clean_xyz_world=points,gt_layer=np.zeros(4,int),json=np.frombuffer(json.dumps(meta).encode(),np.uint8)))
            run.npz(out,dict(xyz_world=points,k=1))
            record=dict(phase='bridge',case='unit',seed=0,gap=0,sigma=1,method='unit',status='OK',shared_case_seconds=0,
                input=str(inp),evaluation=str(ev),output=str(out),output_sha256=run.v14.sha(out))
            with patch.object(run.metrics,'score_replay',return_value=({},{})):
                rows=run.score(dest,'bridge',[record])
            self.assertEqual(rows[0]['surface_mae_mm'],0.)
            self.assertEqual(rows[0]['balanced_source_mae_mm'],0.)

if __name__=='__main__':unittest.main(verbosity=2)
