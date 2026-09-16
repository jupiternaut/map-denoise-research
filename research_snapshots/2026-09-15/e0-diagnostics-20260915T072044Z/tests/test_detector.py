"""Deterministic controls and selector/calibration isolation checks for A."""
import inspect
import unittest

import numpy as np

import detector


class DetectorTests(unittest.TestCase):
    def test_small_p_positive_and_negative_controls(self):
        controls = detector.deterministic_controls()
        self.assertEqual(controls['small_p_positive']['observed']['n_rejected'], 20)
        self.assertEqual(controls['negative']['observed']['n_rejected'], 0)

    def test_later_rank_rejects_without_many_floor_hits(self):
        p = np.r_[0.005, np.full(9, .010), np.ones(90)]
        rejected, summary, ranks = detector.bh_rank_diagnostics(p)
        self.assertEqual(int((p == .005).sum()), 1)
        self.assertLess(ranks['bh_margin'][0], 0)
        self.assertEqual(summary['first_passing_rank'], 10)
        self.assertEqual(int(rejected.sum()), 10)
        np.testing.assert_array_equal(rejected, detector.E0.bh_reject(p, .1))

    def test_calibration_ties_and_pool_fallback_match_old(self):
        score = np.array([.1, .2, .2, .4])
        strata = np.array([0, 0, 1, 2])
        cal = detector.calibrate_scores(score, strata)
        old = detector.E0.calibrate(score, np.zeros(4), strata, .2)
        for k, name in enumerate(detector.E0.STRATA):
            np.testing.assert_array_equal(cal[k]['scores'], old[name]['sorted_s'])
            self.assertTrue(cal[k]['fallback'])
        p, floor = detector.test_pvalues(np.array([.2, .5, .0]), np.array([0, 1, 2]), cal)
        np.testing.assert_allclose(p, [.8, .2, 1])
        np.testing.assert_allclose(floor, [.2, .2, .2])

    def test_detector_reaches_small_p_positive_control(self):
        # Enough calibration resolution plus discriminating observations.
        score = np.zeros(9999)
        cal = detector.calibrate_scores(score, np.arange(len(score)) % 3)
        observed = np.r_[np.ones(20), np.zeros(80)]
        selected = detector.select_points(observed, np.zeros(100, dtype=int), cal)
        self.assertEqual(int(selected['bh'].sum()), 20)
        self.assertEqual(int(selected['pointwise'].sum()), 20)

    def test_selector_api_and_evaluation_labels_are_separate(self):
        self.assertEqual(list(inspect.signature(detector.select_points).parameters),
                         ['s0', 'observed_stratum', 'calibration', 'q', 'point_alpha'])
        cal = detector.calibrate_scores(np.zeros(9999), np.arange(9999) % 3)
        s0 = np.r_[np.ones(20), np.zeros(80)]
        strata = np.zeros(100, dtype=int)
        before = detector.select_points(s0, strata, cal)
        labels = np.full(100, 'base')
        eval_a = detector.evaluate_rejections(before['bh'], np.zeros(100), labels, np.zeros(100, bool))
        eval_b = detector.evaluate_rejections(before['bh'], np.full(100, 5.), labels, np.zeros(100, bool))
        self.assertNotEqual(eval_a['within_tolerance']['n'], eval_b['within_tolerance']['n'])
        after = detector.select_points(s0, strata, cal)
        np.testing.assert_array_equal(before['p'], after['p'])
        np.testing.assert_array_equal(before['bh'], after['bh'])

    def test_hash_preserves_prefix_identity_and_detects_changes(self):
        a = np.arange(18).reshape(9, 2)
        nested = np.concatenate([a[:3], a[3:6], a[6:]])
        self.assertEqual(detector.arrays_hash(a=a[:6]), detector.arrays_hash(a=nested[:6]))
        changed = nested.copy()
        changed[0, 0] = 99
        self.assertNotEqual(detector.arrays_hash(a=a[:6]), detector.arrays_hash(a=changed[:6]))

    def test_no_qualifying_rank_returns_zero(self):
        _, summary, ranks = detector.bh_rank_diagnostics(np.full(20, .101))
        self.assertEqual(summary['n_rejected'], 0)
        self.assertTrue(np.all(ranks['bh_margin'] < 0))


if __name__ == '__main__':
    unittest.main()
