"""Field evaluation contracts on small explicit finite geometries."""
import unittest
import numpy as np
from scipy.spatial import cKDTree
from evaluate import metrics,pointwise_oracle,proposal_candidates,summarize,ARMS,NUMERIC,SCENES,CONDS,PROPOSAL_NAMES


class FieldEvaluationTests(unittest.TestCase):
    def test_pool_oracle_is_pointwise_and_keeps_unsupported_rows(self):
        p=np.array([[0.,0,0],[2.,0,0],[100.,0,0]])
        a=np.array([[1.,0,0],[4.,0,0],[1.,0,0]])
        b=np.array([[3.,0,0],[1.,0,0],[1.,0,0]])
        support=np.array([True,True,False]);tree=cKDTree([[1.,0,0]])
        q,d,choice,names=pointwise_oracle(p,support,tree,[('A',a),('B',b)])
        np.testing.assert_array_equal(q,np.array([[1.,0,0],[1.,0,0],[100.,0,0]]))
        np.testing.assert_array_equal(choice,[1,2]);self.assertEqual(names,['identity','A','B'])
        np.testing.assert_array_equal(d,[0.,0.])

    def test_oracle_exact_ties_keep_identity(self):
        p=np.array([[0.,0,0]]);q,d,choice,names=pointwise_oracle(p,np.ones(1,bool),cKDTree([[1.,0,0]]),[('other',np.array([[2.,0,0]]))])
        np.testing.assert_array_equal(q,p);self.assertEqual(choice[0],0)

    def test_pool_geometry_contract_rejects_bad_candidate(self):
        p=np.zeros((2,3));support=np.ones(2,bool);tree=cKDTree([[0.,0,0]])
        for bad in (np.zeros((1,3)),np.full((2,3),np.nan)):
            with self.assertRaises(AssertionError):pointwise_oracle(p,support,tree,[('bad',bad)])

    def test_proposal_oracle_excludes_fitted_field_positions(self):
        names=list(PROPOSAL_NAMES)+['field_K1','field_K2a','field_K2b']
        pool=np.zeros((1,12,3));pool[0,9:]=[1.,0.,0.]
        p=np.zeros((1,3));support=np.ones(1,bool);tree=cKDTree([[1.,0,0]])
        proposals=proposal_candidates(pool,names)
        self.assertEqual(len(proposals),9)
        _,d,_,_=pointwise_oracle(p,support,tree,proposals)
        self.assertEqual(d[0],1.)
        _,d,_,_=pointwise_oracle(p,support,tree,proposals+[('field',pool[:,9])])
        self.assertEqual(d[0],0.)
        with self.assertRaises(AssertionError):proposal_candidates(pool,names[::-1])

    def test_metrics_support_and_reverse_are_explicit(self):
        p=np.array([[0.,0,0],[2.,0,0],[100.,0,0]]);q=np.array([[1.,0,0],[2.,0,0],[0.,0,0]])
        r=np.array([[1.,0,0],[20.,0,0]]);support=np.array([True,True,False]);tree=cKDTree(r)
        d0=tree.query(p[support])[0];m=metrics(q,p,r,support,tree,d0)
        self.assertEqual(m['source_MSE_mm2'],.5)
        self.assertEqual(m['reverse_MAE_mm'],9.)
        self.assertEqual(m['moved_support_fraction'],.5)
        self.assertAlmostEqual(m['move_RMS_mm'],np.sqrt(.5))
        self.assertEqual(m['improved_fraction'],.5)

    def test_aggregation_equal_scene_ROI_and_paired_differences(self):
        rows=[]
        for j,s in enumerate(SCENES,1):
            for r in range(4):
                for c in CONDS:
                    for arm in ARMS:
                        m=dict.fromkeys(NUMERIC,0.)
                        m['source_MSE_mm2']=j*(10. if arm=='identity' else 9.)
                        rows.append(dict(scene=s,roi=f'{s}r{r}',condition=c,arm=arm,n_source=100**r,**m))
        result=summarize(rows)['native'];self.assertEqual(result['single_field']['source_MSE_mm2'],18.)
        self.assertAlmostEqual(result['single_field']['MSE_gain_percent'],10.)
        self.assertEqual(result['single_field']['wins'],12)
        self.assertEqual(result['matched_comparisons']['multi_field_minus_single_field']['ties'],12)


if __name__=='__main__':unittest.main()
