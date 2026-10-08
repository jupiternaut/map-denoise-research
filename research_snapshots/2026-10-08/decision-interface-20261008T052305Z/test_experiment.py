"""Runner integration properties not dependent on a winning scientific result."""
import unittest
import numpy as np
import experiment as e

class RunnerTests(unittest.TestCase):
    def test_summary_identity_and_damage(self):
        rows=[]
        for change in (-5.,0.,3.):
            rows.append(dict(arm='x',initial_kind='minus',error=10+change,change=change,move=change!=0,reason='ok_move' if change else 'keep_incumbent',regret=0.))
        s=e.summary(rows)[0]
        self.assertEqual((s['n'],s['improved'],s['harmed'],s['unchanged']),(3,1,1,1))
        self.assertAlmostEqual(s['mae'],28/3)
    def test_old_predictor_unchanged_api(self):
        self.assertEqual(e.predictor.METHOD_VERSION,'image-rectangle-linear-v1')
        self.assertEqual(e.predictor._OFFSET.shape,(9,2))
    def test_json_cleanup(self):
        self.assertEqual(e.clean(dict(a=np.float64(1),b=np.array([2,np.nan]))),dict(a=1.,b=[2.,None]))

if __name__=='__main__':unittest.main()
