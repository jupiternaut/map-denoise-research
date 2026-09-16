import unittest
import numpy as np
from field_ops import construct, rms, to_budget, BUDGETS


class BudgetTests(unittest.TestCase):
    def sample(self):
        rng = np.random.default_rng(92424)
        x = rng.normal(size=(30, 3))
        fields = {name: x+scale*rng.normal(size=x.shape) for name, scale in
                  [('local_plane64', .2), ('quadratic64', .1), ('multiscale_full', .15)]}
        return x, fields, rng.uniform(.05, 1, len(x))

    def test_budget_and_shape(self):
        x, cached, alpha = self.sample()
        records, _ = construct(x, cached, alpha)
        self.assertEqual(len(records), 29)
        for r in records:
            self.assertEqual(r['output'].shape, x.shape)
            if r['kind'] == 'budget':
                self.assertAlmostEqual(rms(r['output']-x), r['actual_budget_mm'], places=12)
                if r['allocation'] == 'uniform':
                    self.assertLessEqual(r['normalization_factor'], 1.+1e-12)

    def test_repeatable_and_input_unchanged(self):
        x, cached, alpha = self.sample(); before = x.copy(); a = alpha.copy()
        first, _ = construct(x, cached, alpha); second, _ = construct(x, cached, alpha)
        for p, q in zip(first, second): np.testing.assert_array_equal(p['output'], q['output'])
        np.testing.assert_array_equal(before, x); np.testing.assert_array_equal(a, alpha)

    def test_zero_field_not_amplified(self):
        self.assertEqual(to_budget(np.zeros((3, 3)), .1), (None, None))
        x, cached, alpha = self.sample(); cached['local_plane64'] = x.copy()
        records, _ = construct(x, cached, alpha)
        for r in records:
            if r['kind'] == 'budget':
                np.testing.assert_array_equal(r['output'], x)
                self.assertFalse(r['direction_informative'])

    def test_zero_alpha_infeasible(self):
        x, cached, alpha = self.sample()
        records, _ = construct(x, cached, np.zeros_like(alpha))
        for r in records:
            if r.get('allocation') in ('alpha', 'shuffle'):
                self.assertEqual(r['status'], 'INFEASIBLE')
                self.assertIsNone(r['output'])

    def test_common_cap_duplicate_and_scaling(self):
        x, cached, alpha = self.sample()
        cached['local_plane64'] = x+0.001
        records, _ = construct(x, cached, alpha)
        for r in records:
            if r['kind'] == 'budget':
                self.assertAlmostEqual(r['actual_budget_mm'], np.sqrt(3)*.001)
                self.assertEqual(r['duplicate_budget'], r['requested_budget_mm'] != BUDGETS[0])

    def test_rigid_equivariance(self):
        x, cached, alpha = self.sample()
        rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        t = np.array([1., 2., 3.])
        original, _ = construct(x, cached, alpha)
        transformed, _ = construct(x@rot+t, {k: v@rot+t for k, v in cached.items()}, alpha)
        for a, b in zip(original, transformed):
            np.testing.assert_allclose(a['output']@rot+t, b['output'], atol=1e-12)

    def test_invalid(self):
        with self.assertRaises(ValueError): to_budget(np.ones((3, 3)), -1.)
        x, cached, alpha = self.sample()
        with self.assertRaises(ValueError): construct(x, cached, alpha*0+np.nan)
        with self.assertRaises(ValueError): construct(x, cached, alpha*0+2)


if __name__ == '__main__': unittest.main()
