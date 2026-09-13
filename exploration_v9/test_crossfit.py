import unittest
import numpy as np
from crossfit import split_rows,candidate_search,project_candidate

class CrossfitTests(unittest.TestCase):
    @staticmethod
    def fixture():
        rng=np.random.default_rng(913829);n=120
        p=np.c_[rng.uniform(-1,1,(n,2)),rng.normal(0,.001,n)]
        x=np.c_[np.ones(n),p[:,:2]];support=np.ones(n,bool);support[::9]=False
        state=dict(active=np.arange(n),order=np.arange(n),world=p,design=x,corrected=p[:,2]*1000,
                   weights=np.ones(n),scan_id=np.repeat(np.arange(4),30),support=support,
                   local=p*1000,normal=np.array([0.,0.,1.]))
        a=dict(coefficients=np.zeros((1,3)),candidate_mask=np.ones((n,1),bool),group_ids=np.zeros(n,int))
        return state,a
    def test_split_balanced_reproducible(self):
        x=np.repeat([1,3,8],[11,17,20]);f=split_rows(x)
        np.testing.assert_array_equal(f,split_rows(x))
        for sid in np.unique(x):self.assertLessEqual(abs(np.sum(f[x==sid]==0)-np.sum(f[x==sid]==1)),1)
    def test_disjoint_training_and_query(self):
        s,a=self.fixture();art,info=candidate_search(s,a,1.,budget=3,folds=2)
        self.assertFalse(np.any(art['candidate_training_mask']&art['candidate_query_mask']))
        np.testing.assert_array_equal(art['candidate_query_mask'].sum(0),1)
        self.assertEqual(info['fallback_count'],0)
    def test_training_coefficients_ignore_query_values_conditional_on_upstream(self):
        s,a=self.fixture();art,_=candidate_search(s,a,1.,budget=3,folds=2)
        altered={k:v.copy() for k,v in s.items()};q=art['candidate_query_mask'][0]
        altered['corrected'][q]+=100
        other,_=candidate_search(altered,a,1.,budget=3,folds=2)
        np.testing.assert_array_equal(art['coefficients'][0],other['coefficients'][0])
    def test_full_data_control_matches_single_wls(self):
        s,a=self.fixture();art,_=candidate_search(s,a,1.,budget=3,folds=1)
        expected=np.linalg.lstsq(s['design'],s['corrected'],rcond=1e-12)[0]
        np.testing.assert_allclose(art['coefficients'][0],expected,atol=1e-12)
    def test_candidate_projection_support(self):
        s,a=self.fixture();before=s['world'].copy();art,_=candidate_search(s,a,1.,folds=2)
        out,info,_=project_candidate(s,art)
        np.testing.assert_array_equal(out[~s['support']],before[~s['support']])
        np.testing.assert_array_equal(s['world'],before);self.assertTrue(np.isfinite(out).all())
        self.assertEqual(info['truth_fields_used'],[])
    def test_rank_deficient_seed_fallback_is_disclosed(self):
        s,a=self.fixture();s['design'][:,2]=0.;a['coefficients'][0,0]=.75
        art,info=candidate_search(s,a,1.,budget=0,folds=2)
        self.assertEqual(info['fallback_count'],2)
        np.testing.assert_array_equal(art['coefficients'][:,0],.75)
        self.assertTrue(all(d['initial_seed_fallback_groups']==[0] for d in info['fold_diagnostics']))
if __name__=='__main__':unittest.main()
