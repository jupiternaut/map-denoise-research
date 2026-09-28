"""Independent contract checks: candidate identity and paired-feature meaning."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock
import numpy as np

from common import cached_features, apply_threshold
from paired_features import summarize_pairs, gate_pairs


class CandidateContractTests(unittest.TestCase):
    def test_default_and_explicit_candidate_features_are_b(self):
        # A deliberately differs from B; matching shape alone cannot pass this.
        with TemporaryDirectory(prefix='boundary-cross-contract-') as tmp:
            path = Path(tmp)/'real_results'/'case'
            path.mkdir(parents=True)
            a = np.arange(128, dtype=np.float32).reshape(2, 64)
            b = -a-7
            np.savez(path/'features.npz', A_all_post=a, B_all_post=b)
            np.testing.assert_array_equal(cached_features(path), b)
            np.testing.assert_array_equal(cached_features(path, 'B'), b)
            np.testing.assert_array_equal(cached_features(path, 'A'), a)

    def test_replay_feature_filename_keeps_candidate_identity(self):
        with TemporaryDirectory(prefix='boundary-cross-contract-') as tmp:
            path = Path(tmp)/'scan55'/'case'
            path.mkdir(parents=True)
            a = np.full((3,64), 2, dtype=np.float32)
            b = np.full((3,64), 9, dtype=np.float32)
            np.savez(path/'FEATURES.npz', A_all_post=a, B_all_post=b)
            np.testing.assert_array_equal(cached_features(path,'B'), b)

    def test_pair_margin_is_proposal_minus_incumbent_correlation(self):
        old = np.array([.2,.3,.4,.5])
        new_a = old-.1
        new_b = old+.1
        fa = summarize_pairs(np.stack([old,new_a],axis=-1)[:,None,:])
        fb = summarize_pairs(np.stack([old,new_b],axis=-1)[:,None,:])
        self.assertLess(float(fa[0,24]),0)
        self.assertGreater(float(fb[0,24]),0)
        self.assertFalse(gate_pairs(fa)[0])
        self.assertTrue(gate_pairs(fb)[0])

    def test_missing_proposal_does_not_become_positive_evidence(self):
        values = np.array([[[.2,np.nan]],[[.3,.8]],[[.4,np.nan]],[[.5,np.nan]]])
        f = summarize_pairs(values)
        self.assertEqual(float(f[0,29]),.25)
        self.assertFalse(gate_pairs(f)[0])

    def test_threshold_is_strict_and_actions_are_exact(self):
        scores = np.array([-.5,0.,.5])
        np.testing.assert_array_equal(apply_threshold(scores,{'action':'threshold','threshold':0}),[False,False,True])
        np.testing.assert_array_equal(apply_threshold(scores,{'action':'keep'}),[False]*3)
        np.testing.assert_array_equal(apply_threshold(scores,{'action':'all'}),[True]*3)

    def test_fit_passes_candidate_specific_normalized_gain_and_rows(self):
        import train_cross
        x=np.arange(288,dtype=float).reshape(3,96)
        d=dict(e0=np.array([1.,9.,4.]),e1=np.array([4.,1.,2.]),
               gain=np.array([-3.,8.,2.]),case_id=np.array([0,0,1]))
        selected=np.array([True,False,True])
        model=MagicMock()
        with patch.object(train_cross,'HistGradientBoostingRegressor',return_value=model):
            self.assertIs(train_cross.fit(x,d,selected),model)
        args,kwargs=model.fit.call_args
        np.testing.assert_array_equal(args[0],x[selected])
        np.testing.assert_array_equal(args[1],np.array([-3/5.01,2/6.01]))
        self.assertEqual(kwargs['sample_weight'].shape,(2,))


if __name__=='__main__':
    unittest.main()
