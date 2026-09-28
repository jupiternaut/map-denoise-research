import unittest
import numpy as np
from scipy.spatial import cKDTree
from evaluate import metrics, summarize, ARMS, NUMERIC
from common import CONDS, materialize_step

class EvaluationTests(unittest.TestCase):
    def test_fixed_support(self):
        p=np.array([[0.,0,0],[100.,0,0]])
        q=np.array([[.4,0,0],[-100.,0,0]])
        ref=np.array([[.4,0,0]])
        result,d=metrics(q,p,ref,np.array([True,False]),cKDTree(ref),np.array([.4]))
        self.assertEqual(result['source_MSE_mm2'],0.)
        self.assertEqual(result['reverse_MAE_mm'],0.)
        self.assertEqual(result['improved_fraction'],1.)

    def test_keep_and_step(self):
        g=np.array([[[0.,0,0],[1.,0,0],[2.,0,0]],[[4.,0,0],[5.,0,0],[6.,0,0]]])
        q=materialize_step(g,np.array([0,2],dtype=np.uint8),np.array([.5,.25]))
        np.testing.assert_array_equal(q,[[0,0,0],[4.5,0,0]])

    def test_interior_counts_only_moved_points(self):
        p=np.array([[0.,0,0],[2.,0,0]])
        q=np.array([[0.,0,0],[2.5,0,0]])
        ref=p.copy()
        m,d=metrics(q,p,ref,np.ones(2,dtype=bool),cKDTree(ref),np.zeros(2),np.full(2,.5))
        self.assertEqual(m['interior_step_support_fraction'],.5)

    def test_aggregate_ratio_not_mean_gain(self):
        rows=[]
        for cond in CONDS:
            for scene in (55,65,69):
                for roi in range(4):
                    for arm in ARMS:
                        v=float(1+roi) if arm=='identity' else .5
                        row={key:v for key in NUMERIC}
                        row.update(condition=cond,scene=scene,roi=str(roi),arm=arm)
                        rows.append(row)
        result=summarize(rows)
        self.assertAlmostEqual(result['native']['grid']['MSE_gain_percent'],80.)

if __name__=='__main__': unittest.main()
