"""B0 deterministic mathematical/implementation checks, not B1/B2/B3 runs.

The sole real artifact read is the already-public frozen w002 score curve.
No scores are regenerated, no calibration is fit, and no files are written.
"""
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from decisions import (ABS_TOL, REL_TOL, acceptance, combine_scale, crps,
                       decide, distribution)


OLD = Path(__file__).resolve().parent.parent / "mixed-pixel-20261008T041249Z"
W002 = OLD / "ordinary_stage/w002_ED_600.npz"


class DecisionChecks(unittest.TestCase):
    def setUp(self):
        self.grid = np.arange(5.0)
        self.loss = (self.grid - 2.0) ** 2
        self.candidates = [0.0, 1.0, 2.0, 3.0, 4.0]

    def choose(self, rule, **kwargs):
        options = dict(rule=rule, sigma2=1.0)
        options.update(kwargs)
        return decide(self.grid, self.loss, self.candidates, 0.0, **options)

    def test_actual_w002_curve_and_three_states(self):
        before = hashlib.sha256(W002.read_bytes()).hexdigest()
        with np.load(W002, allow_pickle=False) as data:
            grid, loss = data["grid"], data["loss"]
            sigma2, sources = combine_scale(data["sigma_values"], data["pixel_counts"])
        self.assertEqual(sources, ["local", "local"])
        candidates = [450.0, 540.0, 600.0, 660.0, 900.0]
        old = decide(grid, loss, candidates, 600, rule="P", sigma2=sigma2)
        self.assertEqual(old["intervals"], [[486.0, 782.0]])
        self.assertEqual(old["support_mean"], 634.0)
        self.assertEqual(old["selected_depth"], 660.0)
        for incumbent in (540, 600, 660):
            new = decide(grid, loss, candidates, incumbent, rule="M", sigma2=sigma2)
            self.assertEqual(new["selected_depth"], 600.0)
        replay = decide(grid, loss, candidates, 600, rule="P", sigma2=sigma2,
                        historical_intervals=[[486, 782]])
        self.assertEqual(replay["selected_depth"], old["selected_depth"])
        self.assertEqual(hashlib.sha256(W002.read_bytes()).hexdigest(), before)

    def test_legacy_tie_preserves_order(self):
        for candidates, expected in (([1.0, 3.0], 1.0), ([3.0, 1.0], 3.0)):
            result = decide(self.grid, self.loss, candidates, 0, rule="P", sigma2=1,
                            historical_intervals=[[1.0, 3.0]])
            self.assertEqual(result["selected_depth"], expected)

    def test_historical_p_is_independent_of_nan_flat_and_raw_flags(self):
        for loss in (np.full(5, np.nan), np.zeros(5)):
            result = decide(self.grid, loss, self.candidates, 0, rule="P", sigma2=None,
                            raw_valid=False, historical_intervals=[[1, 3]])
            self.assertEqual(result["selected_depth"], 2)
            result = decide(self.grid, loss, self.candidates, 0, rule="P", sigma2=None,
                            raw_valid=False, historical_intervals=[])
            self.assertEqual(result["selected_depth"], 0)

    def test_legacy_uniform_interval_geometry(self):
        grid = np.arange(300.0, 322.0, 2.0)
        accepted = np.zeros(len(grid), dtype=bool)
        accepted[[2, 3, 7]] = True
        result = decide(grid, grid, grid, 300, rule="P", sigma2=1, accepted=accepted)
        self.assertEqual(result["intervals"], [[302.0, 308.0], [312.0, 316.0]])

    def test_nonuniform_physical_intervals(self):
        grid = np.array([0.0, 2.0, 8.0, 10.0])
        result = decide(grid, grid, grid, 0, rule="P", sigma2=1,
                        accepted=np.array([False, True, False, False]))
        self.assertEqual(result["intervals"], [[0.0, 6.0]])

    def test_new_rules_candidate_order_and_duplicates(self):
        for rule in ("M", "Mraw", "R", "U"):
            expected = self.choose(rule)
            shuffled = decide(self.grid, self.loss, [4, 2, 0, 3, 2, 1, 0], 0,
                              rule=rule, sigma2=1)
            self.assertEqual(shuffled["selected_depth"], expected["selected_depth"])
            self.assertEqual(shuffled["reason"], expected["reason"])
            self.assertEqual(shuffled["candidate_count"], 5)

    def test_new_exact_best_tie_keeps_incumbent(self):
        loss = np.array([3.0, 0.0, 2.0, 0.0, 3.0])
        for rule in ("M", "Mraw"):
            result = decide(self.grid, loss, self.candidates, 4, rule=rule, sigma2=1)
            self.assertEqual(result["reason"], "tied_best")
            self.assertEqual(result["selected_depth"], 4)

    def test_new_near_tie_uses_registered_tolerance(self):
        loss = np.array([2.0, 0.0, ABS_TOL / 2, 1.0, 2.0])
        result = decide(self.grid, loss, self.candidates, 4, rule="M", sigma2=1)
        self.assertEqual(result["reason"], "tied_best")

    def test_absolute_tolerance_scale_sensitivity_is_explicit(self):
        # The fixed absolute tie tolerance forbids universal rescaling invariance
        # right at its boundary. This is recorded, not silently normalized away.
        loss = np.array([2.0, 0.0, ABS_TOL / 2, 1.0, 2.0])
        a = decide(self.grid, loss, self.candidates, 4, rule="M", sigma2=1)
        b = decide(self.grid, 100 * loss, self.candidates, 4, rule="M", sigma2=100)
        self.assertEqual(a["reason"], "tied_best")
        self.assertEqual(b["selected_depth"], 1)
        self.assertEqual(REL_TOL, 1e-10)

    def test_flat_curves_keep(self):
        for rule in ("P", "M", "Mraw", "R", "U"):
            result = decide(self.grid, np.ones(5), self.candidates, 0, rule=rule, sigma2=1)
            self.assertTrue(result["flat"])
            self.assertFalse(result["move"])
            self.assertEqual(result["reason"], "flat_curve")

    def test_new_and_legacy_flat_tolerances_are_distinct(self):
        loss = np.array([0, 2e-11, 4e-11, 6e-11, 8e-11])
        old = decide(self.grid, loss, self.candidates, 0, rule="P", sigma2=1)
        new = decide(self.grid, loss, self.candidates, 0, rule="M", sigma2=1)
        self.assertFalse(old["flat"])
        self.assertTrue(new["flat"])

    def test_nonfinite_and_raw_invalid_keep_json_safe(self):
        for rule in ("P", "M", "Mraw", "R", "U"):
            for invalid in (float("nan"), float("inf")):
                loss = self.loss.copy()
                loss[0] = invalid
                result = decide(self.grid, loss, self.candidates, 0, rule=rule, sigma2=1)
                self.assertEqual(result["reason"], "nonfinite_curve")
                self.assertFalse(result["move"])
                json.dumps(result, allow_nan=False)
            self.assertEqual(self.choose(rule, raw_valid=False)["reason"], "raw_invalid")

    def test_empty_acceptance_gates_and_raw_bypass(self):
        for rule in ("P", "M", "R", "U"):
            result = self.choose(rule, accepted=np.zeros(5, dtype=bool))
            self.assertEqual(result["reason"], "threshold_empty")
        raw = self.choose("Mraw", sigma2=float("nan"), accepted=np.zeros(5, dtype=bool))
        self.assertEqual(raw["selected_depth"], 2)
        self.assertFalse(raw["scale_valid"])
        self.assertEqual(raw["accepted_count"], 0)

    def test_scale_unavailable_does_not_drop_folds(self):
        scale, sources = combine_scale([2.0, float("nan")], [1, 3])
        self.assertTrue(math.isnan(scale))
        self.assertEqual(sources, ["local", "unavailable"])
        self.assertEqual(self.choose("M", sigma2=scale)["reason"], "scale_unavailable")
        fallback, sources = combine_scale([2.0, float("nan")], [1, 3], sigma_cal2=16)
        self.assertEqual(fallback, 13.0)
        self.assertEqual(sources, ["local", "calibration"])
        self.assertEqual(self.choose("M", sigma2=fallback)["selected_depth"], 2)

    def test_fallback_preserves_all_valid_local_values(self):
        expected, _ = combine_scale([2.0, 3.0], [3, 7])
        found, sources = combine_scale([2.0, 3.0], [3, 7], sigma_cal2=1000)
        self.assertEqual(expected, found)
        self.assertEqual(sources, ["local", "local"])
        scale, sources = combine_scale([0.9, -1.0], [1, 1], sigma_cal2=4)
        self.assertEqual(scale, 4)
        self.assertEqual(sources, ["calibration", "calibration"])
        with self.assertRaises(ValueError):
            combine_scale([1.0, 1.0], [1, 0], sigma_cal2=4)

    def test_external_full9_mask_overrides_gray_threshold(self):
        result = decide(self.grid, self.loss + 100, self.candidates, 0, rule="M",
                        sigma2=None, accepted=np.ones(5, dtype=bool))
        self.assertEqual(result["selected_depth"], 2)
        self.assertFalse(result["scale_valid"])
        risk = decide(self.grid, self.loss, self.candidates, 0, rule="R",
                      sigma2=None, accepted=np.ones(5, dtype=bool))
        self.assertEqual(risk["reason"], "scale_unavailable")

    def test_external_full9_finite_domain_is_preserved(self):
        loss = np.array([np.nan, 2.0, 0.0, 1.0, np.nan])
        mask = np.array([False, True, True, False, False])
        for rule in ("M", "Mraw", "R", "U"):
            result = decide(self.grid, loss, self.candidates, 4, rule=rule,
                            sigma2=1, accepted=mask)
            self.assertNotEqual(result["reason"], "nonfinite_curve")
            self.assertEqual(result["finite_score_count"], 3)
            if rule in ("M", "Mraw"):
                self.assertEqual(result["selected_depth"], 2)
                self.assertEqual(result["finite_candidate_count"], 3)
        raw = decide(self.grid, loss, self.candidates, 4, rule="Mraw", sigma2=None,
                     accepted=np.zeros(5, dtype=bool))
        self.assertEqual(raw["selected_depth"], 2)
        q = distribution(self.grid, loss, 1, 1, mask, uniform=True)
        np.testing.assert_array_equal(q, [0, 0.5, 0.5, 0, 0])

    def test_nan_nodes_do_not_redefine_physical_cells(self):
        grid = np.array([0.0, 1.0, 2.0, 10.0])
        loss = np.array([np.nan, 0.0, 0.0, np.nan])
        q = distribution(grid, loss, 1, 1, np.array([False, True, True, False]), uniform=True)
        # Original-grid widths at nodes 1 and 2 are 1 and 4.5, not equal.
        self.assertAlmostEqual(q[1], 1 / 5.5)
        self.assertAlmostEqual(q[2], 4.5 / 5.5)

    def test_all_invalid_full9_or_flat_finite_domain_keep(self):
        for rule in ("M", "Mraw", "R", "U"):
            mask = np.ones(5, dtype=bool)
            invalid = decide(self.grid, np.full(5, np.nan), self.candidates, 0,
                             rule=rule, sigma2=1, accepted=mask)
            self.assertEqual(invalid["reason"], "nonfinite_curve")
            flat = decide(self.grid, [np.nan, 1, 1, np.nan, 1], self.candidates, 0,
                          rule=rule, sigma2=1, accepted=mask)
            self.assertEqual(flat["reason"], "flat_curve")

    def test_legacy_full9_p_does_not_gain_new_flat_gate(self):
        result = decide(self.grid, np.zeros(5), self.candidates, 0, rule="P",
                        sigma2=None, accepted=np.ones(5, dtype=bool))
        self.assertTrue(result["flat"])
        self.assertEqual(result["selected_depth"], 2)

    def test_separated_curve_rescaling_invariant(self):
        for rule in ("P", "M", "Mraw", "R", "U"):
            reference = self.choose(rule)
            for factor in (0.01, 0.5, 3.0, 100.0):
                found = decide(self.grid, self.loss * factor, self.candidates, 0,
                               rule=rule, sigma2=factor)
                self.assertEqual(found["selected_depth"], reference["selected_depth"])
                self.assertEqual(found["accepted_count"], reference["accepted_count"])

    def test_actions_need_exact_scores_not_nearest_grid(self):
        for rule in ("M", "Mraw"):
            with self.assertRaisesRegex(ValueError, "exact score-grid"):
                decide(self.grid, self.loss, [0, 2.1, 4], 0, rule=rule, sigma2=1)

    def test_input_grid_and_mask_contract(self):
        with self.assertRaises(ValueError):
            acceptance([0, 1, 1], [2, 1, 0], 1)
        with self.assertRaises(ValueError):
            self.choose("M", accepted=[1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            self.choose("M", historical_intervals=[[0, 4]])

    def test_r_and_u_share_support_and_physical_cells(self):
        grid = np.array([0.0, 1.0, 4.0, 10.0])
        loss = np.array([1.0, 0.0, 2.0, 3.0])
        mask = np.array([False, True, True, False])
        q_r = distribution(grid, loss, 1, 1, mask)
        q_u = distribution(grid, loss, 1, 1, mask, uniform=True)
        np.testing.assert_array_equal(q_r > 0, q_u > 0)
        self.assertAlmostEqual(q_u[1], 2.0 / 6.5)
        self.assertAlmostEqual(q_u[2], 4.5 / 6.5)
        self.assertGreater(q_r[1], q_u[1])

    def test_extreme_external_scores_avoid_underflow(self):
        grid = np.array([0.0, 1.0, 2.0])
        q = distribution(grid, [0, 1e8, 1e8 + 1], 1, 0.001,
                         np.array([False, True, True]))
        self.assertEqual(q[0], 0)
        self.assertEqual(q[1], 1)
        self.assertTrue(np.isfinite(q).all())

    def test_uniform_depth_measure_on_inverse_depth_grid(self):
        inverse = np.linspace(1 / 5, 1.0, 1001)
        grid = np.sort(1 / inverse)
        q = distribution(grid, np.zeros(len(grid)), 1, 1,
                         np.ones(len(grid), dtype=bool), uniform=True)
        self.assertAlmostEqual(float(q @ grid), 3.0, places=12)
        self.assertGreater(abs(float(grid.mean()) - 3), 0.5)
        self.assertAlmostEqual(float(q.sum()), 1.0)

    def test_grid_integral_converges_2_1_half_mm(self):
        expected = (1 - 5 * math.exp(-4)) / (1 - math.exp(-4))
        errors = []
        for step in (2.0, 1.0, 0.5):
            grid = np.arange(0, 4.0 + step / 2, step)
            q = distribution(grid, grid, 1, 1, np.ones(len(grid), dtype=bool))
            errors.append(abs(float(q @ grid) - expected))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[-1], 0.05)

    def test_crps_matches_direct_pairwise_definition(self):
        grid = np.array([0.0, 1.0, 3.0, 10.0])
        q = np.array([0.1, 0.2, 0.3, 0.4])
        truth = 2.7
        expected = float(q @ np.abs(grid - truth)
                         - 0.5 * (q[:, None] * q[None, :] * np.abs(grid[:, None] - grid[None, :])).sum())
        self.assertAlmostEqual(crps(grid, q, truth), expected, places=13)
        self.assertEqual(crps([0, 2], [0, 1], 2), 0)
        with self.assertRaises(ValueError):
            crps(grid, q / 2, truth)

    def test_risk_can_select_unsupported_middle_of_two_valleys(self):
        grid = np.round(np.linspace(-6, 10, 321), 8)
        loss = (np.abs(grid) - 3) ** 2
        result = decide(grid, loss, [-3, 0, 3, 9], 9, rule="R", sigma2=1)
        self.assertEqual(result["selected_depth"], 0)
        self.assertFalse(acceptance(grid, loss, 1)[np.flatnonzero(grid == 0)[0]])
        raw = decide(grid, loss, [-3, 0, 3, 9], 9, rule="Mraw", sigma2=1)
        self.assertEqual(raw["reason"], "tied_best")
        self.assertEqual(raw["selected_depth"], 9)

    def test_risk_tie_keeps_even_when_all_tied_moves_improve(self):
        grid = np.array([-1.0, 0.0, 1.0, 3.0])
        # Equal mass on -1/+1; every action between them minimizes L1 risk.
        # The nonuniform cells have widths .5 and 1.5, countered by score ln(3).
        loss = np.array([0.0, 9.0, math.log(3), 9.0])
        result = decide(grid, loss, [-1, 0, 1, 3], 3, rule="R", sigma2=1,
                        accepted=np.array([True, False, True, False]))
        self.assertEqual(result["reason"], "tied_best")
        self.assertEqual(result["selected_depth"], 3)

    def test_asymmetric_harm_cost_can_change_move_to_keep(self):
        grid = np.array([0.0, 3.0, 4.0])
        loss = np.array([0.0, 9.0, 0.0])
        mask = np.array([True, False, True])
        symmetric = decide(grid, loss, grid, 3, rule="U", sigma2=1, accepted=mask)
        asymmetric = decide(grid, loss, grid, 3, rule="U", sigma2=1, accepted=mask, kappa=5)
        self.assertEqual(symmetric["selected_depth"], 0)
        self.assertEqual(asymmetric["selected_depth"], 3)

    def test_conditional_two_eta_separation_suffices_for_argmin(self):
        candidates = np.array([0.0, 1.0, 2.0])
        predictions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        eta = 0.2
        for truth_index in range(3):
            separation = np.sqrt(np.mean((predictions - predictions[truth_index]) ** 2, axis=1))
            self.assertGreater(np.delete(separation, truth_index).min(), 2 * eta)
            for error in (np.array([eta, -eta]), np.array([-eta, -eta]), np.zeros(2)):
                self.assertLessEqual(float(np.sqrt(np.mean(error ** 2))), eta)
                observed = predictions[truth_index] + error
                loss = np.mean((predictions - observed) ** 2, axis=1)
                found = decide(candidates, loss, candidates, 0, rule="Mraw", sigma2=None)
                self.assertEqual(found["selected_depth"], float(truth_index))

    def test_two_eta_equality_does_not_guarantee_unique_recovery(self):
        grid = np.array([0.0, 1.0, 2.0])
        observed = 0.5  # eta=.5 and candidate separation=1=2eta.
        result = decide(grid, (grid - observed) ** 2, grid, 2, rule="Mraw", sigma2=None)
        self.assertEqual(result["reason"], "tied_best")
        self.assertEqual(result["selected_depth"], 2)

    def test_convex_hull_projection_keeps_all_w002_initial_states(self):
        low, high = 486.0, 782.0
        actions = np.array([450, 540, 600, 660, 900], dtype=float)
        for incumbent in (540.0, 600.0, 660.0):
            projection = min(high, max(low, incumbent))
            self.assertEqual(projection, incumbent)
            worst_gains = np.minimum((incumbent - low) ** 2 - (actions - low) ** 2,
                                     (incumbent - high) ** 2 - (actions - high) ** 2)
            self.assertEqual(actions[int(worst_gains.argmax())], incumbent)
            self.assertEqual(float(worst_gains.max()), 0)

    def test_functions_do_not_mutate_input_arrays(self):
        grid, loss, candidates = self.grid.copy(), self.loss.copy(), np.array(self.candidates)
        before = [array.copy() for array in (grid, loss, candidates)]
        for rule in ("P", "M", "Mraw", "R", "U"):
            result = decide(grid, loss, candidates, 0, rule=rule, sigma2=1)
            json.dumps(result, allow_nan=False)
        for actual, expected in zip((grid, loss, candidates), before):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
