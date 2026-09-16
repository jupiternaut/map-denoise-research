"""Mechanism controls for the observed-gap guard; no statistical guarantees."""

import inspect
import unittest

import numpy as np

import guards
from common import load_old


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load_old()

    @staticmethod
    def apply(fixture):
        return guards.gap_contraction_guard(
            fixture["theta"], fixture["x"], fixture["y"], fixture["z0"], fixture["h"]
        )

    def test_normal_controls_no_alarm_and_full_repair_retained(self):
        for h in guards.H_VALUES:
            for seed in guards.SEEDS:
                for fixture in guards.make_fixtures(h, seed)[2:5]:
                    with self.subTest(h=h, seed=seed, scenario=fixture["scenario"]):
                        theta, flagged, _ = self.apply(fixture)
                        self.assertFalse(flagged.any())
                        np.testing.assert_array_equal(theta, fixture["theta"])
                        metrics = guards.evaluate(fixture, theta, flagged, fixture["theta"])
                        self.assertLess(metrics["source_sheet_mae_after_mm"], 1e-12)
                        self.assertAlmostEqual(metrics["repair_retained_fraction"], 1.0)

    def test_wrong_merge_flags_all_moved_endpoints(self):
        for h in guards.H_VALUES:
            for seed in guards.SEEDS:
                fixture = guards.make_fixtures(h, seed)[0]
                with self.subTest(h=h, seed=seed):
                    theta, flagged, details = self.apply(fixture)
                    self.assertTrue(flagged.all())
                    self.assertTrue(details["flagged_pair"].any())
                    np.testing.assert_array_equal(theta, np.zeros_like(theta))
                    metrics = guards.evaluate(fixture, theta, flagged, fixture["theta"])
                    self.assertEqual(metrics["harmful_proposals_blocked_count"], len(theta))

    def test_identical_worlds_intentionally_false_veto_valid_ghost_repair(self):
        for h in guards.H_VALUES:
            for seed in guards.SEEDS:
                wrong, ghost = guards.make_fixtures(h, seed)[:2]
                with self.subTest(h=h, seed=seed):
                    for name in ("x", "y", "z0", "theta"):
                        np.testing.assert_array_equal(wrong[name], ghost[name])
                    wrong_theta, wrong_flags, _ = self.apply(wrong)
                    ghost_theta, ghost_flags, _ = self.apply(ghost)
                    np.testing.assert_array_equal(wrong_flags, ghost_flags)
                    np.testing.assert_array_equal(wrong_theta, ghost_theta)
                    wrong_metrics = guards.evaluate(wrong, wrong_theta, wrong_flags, wrong["theta"])
                    ghost_metrics = guards.evaluate(ghost, ghost_theta, ghost_flags, ghost["theta"])
                    self.assertEqual(ghost_metrics["beneficial_proposals_blocked_count"], len(ghost_theta))
                    self.assertEqual(ghost_metrics["repair_retained_fraction"], 0.0)
                    self.assertGreater(ghost_metrics["source_sheet_mae_after_mm"], 2.9)
                    self.assertLess(wrong_metrics["source_sheet_mae_after_mm"], 0.1)

    def test_old_grid_bound_for_arbitrary_z_motion(self):
        # The smallest finite-precision XY gap is still far above .4 at h=.8.
        for h in (0.8, 1.6):
            for seed in guards.SEEDS:
                fixture = guards.make_fixtures(h, seed)[0]
                rng = np.random.default_rng(seed)
                for candidate_z in (np.zeros(len(fixture["x"])), rng.normal(0.0, 100.0, len(fixture["x"]))):
                    theta = candidate_z - fixture["z0"]
                    kept, flagged, nn = self.old.apply_guard(theta, fixture["x"], fixture["y"], fixture["z0"])
                    self.assertFalse(flagged.any())
                    self.assertGreaterEqual(float(nn.min()), h - 1e-12)
                    np.testing.assert_array_equal(theta, kept)

    def test_same_xy_positive_control_old_and_new_guard_fire(self):
        fixture = guards.make_fixtures(0.8, 101)[-1]
        kept, flagged, nn = self.old.apply_guard(fixture["theta"], fixture["x"], fixture["y"], fixture["z0"])
        new_kept, new_flags, _ = self.apply(fixture)
        self.assertLess(float(nn.min()), self.old.D_GUARD)
        np.testing.assert_array_equal(flagged, [False, True])
        np.testing.assert_array_equal(new_flags, [False, True])
        np.testing.assert_array_equal(kept, [0.0, 0.0])
        np.testing.assert_array_equal(new_kept, [0.0, 0.0])

    def test_initial_gap_and_contraction_thresholds_are_exact(self):
        xy = np.array([0.0, 0.1])
        zeros = np.zeros(2)
        # Initial gap exactly 2 is eligible, contraction exactly half is not.
        kept, flags, info = guards.gap_contraction_guard([0.0, -1.0], xy, zeros, [0.0, 2.0], 0.8)
        self.assertTrue(info["eligible_pair"][0])
        self.assertFalse(flags.any())
        kept, flags, _ = guards.gap_contraction_guard([0.0, -1.01], xy, zeros, [0.0, 2.0], 0.8)
        np.testing.assert_array_equal(flags, [False, True])
        kept, flags, info = guards.gap_contraction_guard([0.0, -1.9], xy, zeros, [0.0, 1.9], 0.8)
        self.assertFalse(info["eligible_pair"][0])
        self.assertFalse(flags.any())

    def test_xy_radius_limits_pairs(self):
        _, flags, info = guards.gap_contraction_guard([0.0, -6.0], [0.0, 1.201], [0.0, 0.0], [0.0, 6.0], 0.8)
        self.assertEqual(len(info["pairs"]), 0)
        self.assertFalse(flags.any())

    def test_single_pass_does_not_recompute_after_veto(self):
        # AB contracts, vetoing B. The proposal BC does not halve its gap, but
        # after B is restored it does. One pass must retain C in that new state.
        theta = np.array([0.0, -3.0, -6.5])
        kept, flags, info = guards.gap_contraction_guard(theta, [0.0, 1.0, 2.0], [0.0] * 3, [0.0, 4.0, 12.0], 0.8)
        np.testing.assert_array_equal(flags, [False, True, False])
        np.testing.assert_array_equal(kept, [0.0, 0.0, -6.5])
        np.testing.assert_array_equal(theta, [0.0, -3.0, -6.5])
        np.testing.assert_array_equal(info["flagged_pair"], [True, False])

    def test_selector_signature_has_no_truth_channel(self):
        self.assertEqual(list(inspect.signature(guards.gap_contraction_guard).parameters), ["theta", "x", "y", "z0", "h"])

    def test_rejects_nonfinite_or_misaligned_input(self):
        with self.assertRaises(ValueError):
            guards.gap_contraction_guard([0.0], [0.0, 1.0], [0.0], [0.0], 0.8)
        with self.assertRaises(ValueError):
            guards.gap_contraction_guard([np.nan], [0.0], [0.0], [0.0], 0.8)


if __name__ == "__main__":
    unittest.main()
