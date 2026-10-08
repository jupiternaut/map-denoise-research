import unittest
import numpy as np
from run import combine_curves,clean
from evaluate import stats,compare

class EvaluationTests(unittest.TestCase):
    def test_pool_denominator(self):
        loss,c,v=combine_curves([dict(loss=np.array([2.,4]),pixel_count=10,valid=True),dict(loss=np.array([8.,10]),pixel_count=20,valid=True)])
        np.testing.assert_allclose(loss,[6.,8.]);self.assertTrue(v)
    def test_reject_in_denominator(self):
        r=dict(id='a',initial_depth=540,selected_depth=540,initial_error=60,error=60,intervals=[],arm='K')
        s=stats([r]);self.assertEqual(s['mae'],60);self.assertEqual(s['empty_support'],1)
        b={**r,'arm':'B','selected_depth':600,'error':0,'intervals':[[599,601]]}
        self.assertEqual(compare([r,b],'B','K')['mae_gain'],60)
    def test_json_nonfinite(self):
        self.assertEqual(clean(dict(x=np.array([1.,np.inf,np.nan]))),dict(x=[1.,None,None]))

if __name__=='__main__':unittest.main()
