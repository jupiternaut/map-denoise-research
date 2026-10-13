"""Exact acceptance cases for the new scalar projection decision policy.

These tests intentionally avoid the experiment run and image rendering. The
source snapshot is read as the historical midpoint-policy comparator. Reference
truth is used only for evaluation after the policy has returned its result.
Run from this directory with: python -m unittest test_projection_policy -v
"""

from fractions import Fraction as Q
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source_snapshot"))

from continuous_world.interval_solver import decode, decide, encode, solve_outer
from challenge import certificate_projection


def recorded_interval(left, right):
    return {"hull": [encode(left), encode(right)]}


def endpoint_gain(left, right, incumbent, proposal):
    return min((incumbent - left) ** 2 - (proposal - left) ** 2,
               (incumbent - right) ** 2 - (proposal - right) ** 2)


class ProjectionPolicyTests(unittest.TestCase):
    def assert_result(self, result, status, output, gain, reason):
        self.assertEqual(set(result), {"status", "output", "gain_lower", "reason"})
        self.assertEqual(result["status"], status)
        self.assertEqual(result["output"], encode(output))
        self.assertEqual(result["gain_lower"], None if gain is None else encode(gain))
        self.assertEqual(result["reason"], reason)

    def test_empty_outer_set_is_incompatible_not_vacuously_safe(self):
        self.assert_result(
            certificate_projection({"hull": None}, Q(23, 7)),
            "INCOMPATIBLE", Q(23, 7), None, "empty_certified_outer_set")

    def test_projection_above_and_below_interval_has_exact_distance_gain(self):
        outer = recorded_interval(2, 5)
        self.assert_result(
            certificate_projection(outer, 9), "MOVE", 5, 16,
            "strict_gain_over_continuous_outer_hull")
        self.assert_result(
            certificate_projection(outer, -3), "MOVE", 2, 25,
            "strict_gain_over_continuous_outer_hull")

    def test_inside_and_closed_boundary_incumbents_are_kept_exactly(self):
        for incumbent in (Q(2), Q(5), Q(7, 2), Q(2000001, 1000000)):
            with self.subTest(incumbent=incumbent):
                self.assert_result(
                    certificate_projection(recorded_interval(2, 5), incumbent),
                    "KEEP", incumbent, 0, "no_strict_safe_gain_for_proposal")

    def test_singleton_hull_moves_only_if_incumbent_differs(self):
        truth = Q(23, 7)
        self.assert_result(
            certificate_projection(recorded_interval(truth, truth), truth),
            "KEEP", truth, 0, "no_strict_safe_gain_for_proposal")
        for offset in (Q(-2), Q(1, 4), Q(3, 4)):
            with self.subTest(offset=offset):
                self.assert_result(
                    certificate_projection(recorded_interval(truth, truth), truth + offset),
                    "MOVE", truth, offset ** 2, "strict_gain_over_continuous_outer_hull")

    def test_near_edge_projection_repairs_when_default_midpoint_keeps(self):
        outer = recorded_interval(0, 10)
        for incumbent, projected in ((Q(-1, 4), Q(0)), (Q(41, 4), Q(10))):
            with self.subTest(incumbent=incumbent):
                self.assertEqual(endpoint_gain(0, 10, incumbent, Q(5)), Q(-399, 16))
                self.assertEqual(decide(outer, incumbent)["status"], "KEEP")
                self.assert_result(
                    certificate_projection(outer, incumbent), "MOVE", projected, Q(1, 16),
                    "strict_gain_over_continuous_outer_hull")

    def test_midpoint_zero_gain_is_rejected_but_projection_is_strict(self):
        outer = recorded_interval(0, 10)
        self.assertEqual(endpoint_gain(0, 10, Q(15), Q(5)), 0)
        self.assertEqual(decide(outer, 15)["status"], "KEEP")
        self.assert_result(
            certificate_projection(outer, 15), "MOVE", 10, 25,
            "strict_gain_over_continuous_outer_hull")

    def test_midpoint_can_be_more_accurate_despite_smaller_robust_gain(self):
        outer, incumbent, truth = recorded_interval(590, 610), Q(640), Q(600)
        projected = certificate_projection(outer, incumbent)
        midpoint = decide(outer, incumbent)
        self.assertEqual((projected["status"], midpoint["status"]), ("MOVE", "MOVE"))
        self.assertEqual((decode(projected["gain_lower"]), decode(midpoint["gain_lower"])),
                         (Q(900), Q(800)))
        self.assertEqual(abs(decode(projected["output"]) - truth), 10)
        self.assertEqual(abs(decode(midpoint["output"]) - truth), 0)

    def test_projection_uniform_gain_and_non_regression_when_truth_is_contained(self):
        left, right = Q(-3, 2), Q(11, 3)
        outer = recorded_interval(left, right)
        incumbents = (left - 7, left - Q(1, 4), left, Q(0), right,
                      right + Q(1, 4), right + 7)
        truths = tuple(left + (right - left) * Q(index, 40) for index in range(41))
        for incumbent in incumbents:
            decision = certificate_projection(outer, incumbent)
            output, lower = decode(decision["output"]), decode(decision["gain_lower"])
            for truth in truths:
                with self.subTest(incumbent=incumbent, truth=truth):
                    gain = (incumbent - truth) ** 2 - (output - truth) ** 2
                    self.assertGreaterEqual(gain, lower)
                    self.assertLessEqual(abs(output - truth), abs(incumbent - truth))
                    if decision["status"] == "MOVE":
                        self.assertGreater(gain, 0)

    def test_projection_is_best_among_arbitrary_scalar_proposals(self):
        left, right = Q(-3, 2), Q(11, 3)
        outer = recorded_interval(left, right)
        proposals = tuple(Q(index, 12) for index in range(-120, 145)) + (left, right)
        for incumbent in (left - 2, left - Q(1, 100), Q(0), right + Q(1, 100), right + 2):
            decision = certificate_projection(outer, incumbent)
            output, lower = decode(decision["output"]), decode(decision["gain_lower"])
            for proposal in proposals:
                with self.subTest(incumbent=incumbent, proposal=proposal):
                    gain = endpoint_gain(left, right, incumbent, proposal)
                    self.assertLessEqual(gain, lower)
                    if proposal != output:
                        self.assertLess(gain, lower)

    def test_no_proposal_is_strictly_safe_for_an_inside_incumbent(self):
        left, right = Q(2), Q(5)
        outer = recorded_interval(left, right)
        for incumbent in (left, Q(7, 2), right):
            for proposal in (Q(-100), left, Q(13, 4), Q(7, 2), right, Q(100)):
                with self.subTest(incumbent=incumbent, proposal=proposal):
                    self.assertLessEqual(endpoint_gain(left, right, incumbent, proposal), 0)
                    self.assertEqual(decide(outer, incumbent, proposal)["status"], "KEEP")

    def test_tiny_rational_outside_distance_has_positive_unrounded_certificate(self):
        left, right, offset = Q(3965, 7), Q(4238, 7), Q(1, 10 ** 18 + 3)
        self.assert_result(
            certificate_projection(recorded_interval(left, right), right + offset),
            "MOVE", right, offset ** 2, "strict_gain_over_continuous_outer_hull")

    def test_exact_incumbent_inputs_reject_float_and_boolean(self):
        for incumbent in (600.0, True):
            with self.subTest(incumbent=incumbent):
                with self.assertRaises(TypeError):
                    certificate_projection(recorded_interval(540, 660), incumbent)

    def test_full_prior_alone_repairs_large_outside_error_without_model_evaluation(self):
        def must_not_run(_left, _right):
            raise AssertionError("zero work budget must never evaluate the model")

        outer = solve_outer((Q(-999),), must_not_run, lo=540, hi=660, max_boxes=0)
        self.assertEqual(outer["evaluated_boxes"], 0)
        self.assertEqual(outer["hull"], [encode(540), encode(660)])
        self.assert_result(
            certificate_projection(outer, 840), "MOVE", 660, 32400,
            "strict_gain_over_continuous_outer_hull")
        for incumbent in (Q(540), Q(600), Q(660)):
            with self.subTest(incumbent=incumbent):
                self.assert_result(
                    certificate_projection(outer, incumbent), "KEEP", incumbent, 0,
                    "no_strict_safe_gain_for_proposal")

    def test_small_offsets_inside_the_prior_need_image_narrowing_before_repair(self):
        truth, prior = Q(4238, 7), recorded_interval(540, 660)
        for offset in (Q(-2), Q(-3, 4), Q(-1, 4), Q(0), Q(1, 4), Q(3, 4), Q(2)):
            with self.subTest(offset=offset):
                self.assertEqual(certificate_projection(prior, truth + offset)["status"], "KEEP")
        narrower = recorded_interval(truth - Q(1, 8), truth + Q(1, 8))
        self.assertEqual(certificate_projection(narrower, truth)["status"], "KEEP")
        for offset in (Q(-1, 4), Q(1, 4)):
            with self.subTest(offset=offset):
                self.assertEqual(certificate_projection(narrower, truth + offset)["status"], "MOVE")

    def test_nominal_pairwise_separation_does_not_remove_nuisance_noise_ambiguity(self):
        epsilon, eta = Q(1, 100), Q(1, 100)
        first_depth, second_depth = Q(0), 3 * epsilon
        observation = Q(3, 2) * epsilon
        self.assertGreater(abs(second_depth - first_depth), 2 * epsilon)
        for depth, camera in ((first_depth, eta), (second_depth, -eta)):
            with self.subTest(depth=depth):
                self.assertLessEqual(abs(camera), eta)
                self.assertLessEqual(abs(observation - (depth + camera)), epsilon)
        # The same observation covers both depths; an inside incumbent is not
        # eligible for a strict safe move even though nominal predictions differ.
        self.assertEqual(
            certificate_projection(recorded_interval(first_depth, second_depth), first_depth)["status"],
            "KEEP")

    def test_missing_truth_coverage_can_make_projection_regress_actual_error(self):
        actual_truth = Q(600)
        wrong_outer = recorded_interval(610, 620)
        decision = certificate_projection(wrong_outer, actual_truth)
        self.assertEqual(decision["status"], "MOVE")
        self.assertEqual(decode(decision["gain_lower"]), 100)
        self.assertGreater(abs(decode(decision["output"]) - actual_truth), 0)
        # This is a deliberately invalid containing-truth contract, not a safe
        # experimental success or a contradiction of the conditional theorem.


if __name__ == "__main__":
    unittest.main()
