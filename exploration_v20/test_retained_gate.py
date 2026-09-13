import unittest
from unittest.mock import patch
import numpy as np
import retained_gate as op


class Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(920)
        self.x = np.c_[np.ones(48), rng.uniform(-1, 1, (48, 2))]
        self.y = np.where(self.x[:, 1] > 0, 2., -2.) + rng.normal(0, .2, 48)

    def test_fold_training_excludes_heldout_heights(self):
        def fake(x, y, sigma, family):
            m = dict(family='C', k=2, means=np.array([-1.,1.]), slope=np.zeros(2),
                     feature_center=np.zeros(2), feature_scale=np.ones(2), gate=np.array([y.mean()]))
            return m, dict(training_rows=len(y))
        with patch.object(op, 'training_family', side_effect=fake):
            prior, _, diag = op.crossfit_priors(self.x, self.y, 1., 'C')
            fold = diag['folds'][0]
            y = self.y.copy()
            y[diag['folds']==fold] += 100
            changed, _, _ = op.crossfit_priors(self.x, y, 1., 'C')
        np.testing.assert_array_equal(prior[diag['folds']==fold], changed[diag['folds']==fold])
        for r in diag['training']:
            self.assertEqual(len(set(r['train_indices']) & set(r['validation_indices'])), 0)

    def test_single_surface_passthrough(self):
        model = dict(k=1, prediction=self.y.copy())
        with patch.object(op, 'crossfit_priors', side_effect=AssertionError('must bypass')):
            output, diag = op.run(self.x, self.y, 1., model, {})
        self.assertTrue(diag['single_passthrough'])
        for m in output.values():
            np.testing.assert_array_equal(m['prediction'], self.y)

    def test_full_pool_matches_retained_default(self):
        _, d = op.v18.construct(self.x, self.y, .2)
        pool = op.family_pool(d, self.x, self.y, .2)
        family = op.base.choose({f:m['bic'] for f,m in pool.items()})
        np.testing.assert_allclose(pool[family]['prediction'], d['retained']['prediction'], atol=1e-10)
        self.assertLessEqual(pool['R']['bic'], d['pool']['R']['bic']+1e-9)

    def test_map_projects_to_one_fitted_surface(self):
        model = dict(k=2, means=np.array([-1., 1.]), slope=np.zeros(2),
                     posterior=np.tile([.4,.6],(48,1)), groups=np.ones(48,int))
        projected = op.v18.project(model, self.x, 1.)
        np.testing.assert_array_equal(projected['prediction'], 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
