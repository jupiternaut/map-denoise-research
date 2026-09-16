import inspect
import unittest

import numpy as np

from common import load_old
from projection import candidates
from projection_boundary import linear_projection


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load_old()

    def run_curve(self, center=3., threshold=4., calibration_score=0.):
        old = self.old
        s = ((old.TGRID - center) ** 2)[None, :]
        cal = {name: dict(sorted_s=np.full(999, calibration_score), q_tgt=threshold,
                          q_abs=1., q_test=threshold) for name in old.STRATA}
        return candidates(old, s, np.zeros(1), np.zeros(1, int), cal)

    def test_actual_projection_helps_covered_truth(self):
        _, arms = self.run_curve()
        a = arms['eligible_no_bh']
        self.assertTrue(a['mask'][0])
        self.assertAlmostEqual(a['projection'][0], 1.)
        self.assertAlmostEqual(a['argmin'][0], 3.)
        self.assertLess(abs(a['projection'][0] - 1), 1.)
        self.assertGreater(abs(a['argmin'][0] - 1), 1.)

    def test_argmin_can_repair_more(self):
        _, arms = self.run_curve()
        a = arms['eligible_no_bh']
        self.assertLess(abs(a['argmin'][0] - 3), abs(a['projection'][0] - 3))

    def test_wrong_confidence_set_both_can_harm(self):
        _, arms = self.run_curve()
        a = arms['eligible_no_bh']
        self.assertGreater(abs(a['projection'][0]), 0.)
        self.assertGreater(abs(a['argmin'][0]), 0.)

    def test_keep_if_incumbent_in_set(self):
        _, arms = self.run_curve(center=0.)
        self.assertFalse(arms['eligible_no_bh']['mask'][0])
        self.assertEqual(arms['eligible_no_bh']['projection'][0], 0.)

    def test_bh_bypass_is_explicit(self):
        _, arms = self.run_curve(calibration_score=100.)
        self.assertFalse(arms['original_bh']['mask'][0])
        self.assertTrue(arms['eligible_no_bh']['mask'][0])

    def test_no_evaluation_truth_argument(self):
        self.assertEqual(list(inspect.signature(candidates).parameters),
                         ['old', 's', 'm', 'strata', 'calibration'])
        _, a = self.run_curve()
        before = a['eligible_no_bh']['projection'].copy()
        # Change evaluation answers without regenerating observation curves.
        evaluations = [abs(before[0] - truth) for truth in (1., 3., 0.)]
        self.assertEqual(len(set(evaluations)), 3)
        _, b = self.run_curve()
        np.testing.assert_array_equal(before, b['eligible_no_bh']['projection'])

    def test_grid_endpoint_is_not_continuous_projection(self):
        old = self.old
        s = (np.abs(old.TGRID - 1.01) - .01)[None, :]
        cal = {name: dict(sorted_s=np.zeros(999), q_tgt=.99, q_abs=1., q_test=.99)
               for name in old.STRATA}
        dec, arms = candidates(old, s, np.zeros(1), np.zeros(1, int), cal)
        mask = arms['eligible_no_bh']['mask']
        grid = arms['eligible_no_bh']['projection'][0]
        corrected = linear_projection(old, s, dec, mask)[0]
        truth = .015
        self.assertLessEqual(old.interp_rows(s, np.array([truth]))[0], .99)
        self.assertGreater(abs(grid-truth), truth)
        self.assertLess(abs(corrected-truth), truth)
        self.assertAlmostEqual(corrected,.01)


if __name__ == '__main__':
    unittest.main()
