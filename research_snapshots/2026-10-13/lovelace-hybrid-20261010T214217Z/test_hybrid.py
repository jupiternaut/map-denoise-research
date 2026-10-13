"""Exact policy tests; no confirmation data or image experiment is executed.

Selectors receive only a fresh outer set or a hand-recorded hull and incumbent.
Reference depths are used for post-decision assertions, never policy arguments.
Run from this directory: python -m unittest test_hybrid -v
"""
from fractions import Fraction as Q
import inspect
import unittest
from unittest.mock import patch

import hybrid
from hybrid import certificate_hybrid, certificate_projection, decode, decide, encode, solve_outer


def interval(left, right):
    return {"hull": [encode(left), encode(right)]}


def endpoint_gain(left, right, incumbent, proposal):
    return min((incumbent - left) ** 2 - (proposal - left) ** 2,
               (incumbent - right) ** 2 - (proposal - right) ** 2)


def identity_enclosure(left, right):
    return ((left, right),)


class HybridPolicyTests(unittest.TestCase):
    def assert_result(self, result, status, output, gain, branch):
        self.assertEqual(set(result), {"status", "output", "gain_lower", "reason", "branch"})
        self.assertEqual(result["status"], status)
        self.assertEqual(result["output"], encode(output))
        self.assertEqual(result["gain_lower"], None if gain is None else encode(gain))
        self.assertEqual(result["branch"], branch)
        expected_reason = {"MOVE": "strict_gain_over_continuous_outer_hull",
                           "KEEP": "no_strict_safe_gain_for_proposal",
                           "INCOMPATIBLE": "empty_certified_outer_set"}[status]
        self.assertEqual(result["reason"], expected_reason)

    def test_empty_outer_is_incompatible_with_no_vacuous_gain(self):
        self.assert_result(certificate_hybrid({"hull": None}, Q(23, 7)),
                           "INCOMPATIBLE", Q(23, 7), None, "incompatible")

    def test_singleton_moves_to_its_point_unless_already_correct(self):
        center = Q(23, 7)
        outer = interval(center, center)
        self.assert_result(certificate_hybrid(outer, center), "KEEP", center, 0, "keep")
        for offset in (Q(-2), Q(-1, 4), Q(1, 10**18 + 3), Q(3, 4)):
            with self.subTest(offset=offset):
                self.assert_result(certificate_hybrid(outer, center + offset),
                                   "MOVE", center, offset**2, "midpoint")

    def test_closed_boundaries_and_interior_are_kept_exactly(self):
        outer = interval(Q(2, 3), Q(7, 3))
        for incumbent in (Q(2, 3), Q(7, 3), Q(3, 2), Q(1000001, 1000000)):
            with self.subTest(incumbent=incumbent):
                self.assert_result(certificate_hybrid(outer, incumbent),
                                   "KEEP", incumbent, 0, "keep")

    def test_near_edges_use_projection_on_both_sides(self):
        outer = interval(0, 10)
        for incumbent, output in ((Q(-1, 4), Q(0)), (Q(41, 4), Q(10))):
            with self.subTest(incumbent=incumbent):
                self.assertEqual(endpoint_gain(0, 10, incumbent, Q(5)), Q(-399, 16))
                self.assertEqual(decide(outer, incumbent)["status"], "KEEP")
                self.assert_result(certificate_hybrid(outer, incumbent),
                                   "MOVE", output, Q(1, 16), "projection")

    def test_strict_zero_midpoint_gain_falls_back_to_positive_projection(self):
        outer = interval(0, 10)
        for incumbent, output in ((Q(-5), Q(0)), (Q(15), Q(10))):
            with self.subTest(incumbent=incumbent):
                self.assertEqual(endpoint_gain(0, 10, incumbent, Q(5)), 0)
                self.assertEqual(decide(outer, incumbent)["status"], "KEEP")
                self.assert_result(certificate_hybrid(outer, incumbent),
                                   "MOVE", output, 25, "projection")

    def test_exactly_above_midpoint_threshold_preserves_original_fields(self):
        outer = interval(0, 10)
        tiny = Q(1, 10**18 + 3)
        for incumbent in (Q(-5) - tiny, Q(15) + tiny):
            with self.subTest(incumbent=incumbent):
                midpoint = decide(outer, incumbent)
                self.assertEqual(midpoint["status"], "MOVE")
                expected_gain = (Q(5) + tiny)**2 - 25
                self.assert_result(certificate_hybrid(outer, incumbent),
                                   "MOVE", 5, expected_gain, "midpoint")
                result = certificate_hybrid(outer, incumbent)
                self.assertEqual({k: v for k, v in result.items() if k != "branch"}, midpoint)

    def test_midpoint_success_does_not_invoke_projection(self):
        outer = interval(Q(-1, 3), Q(2, 3))
        accepted = decide(outer, Q(3))
        with patch.object(hybrid, "certificate_projection", side_effect=AssertionError("fallback called")):
            self.assertEqual(certificate_hybrid(outer, Q(3)), {**accepted, "branch": "midpoint"})

    def test_midpoint_failure_invokes_projection_once_with_same_inputs(self):
        outer, incumbent = interval(0, 2), Q(9, 4)
        projected = certificate_projection(outer, incumbent)
        with patch.object(hybrid, "certificate_projection", wraps=certificate_projection) as fallback:
            self.assertEqual(certificate_hybrid(outer, incumbent), {**projected, "branch": "projection"})
            fallback.assert_called_once_with(outer, incumbent)

    def test_exact_incumbents_reject_float_and_boolean_even_for_empty_outer(self):
        for outer in (interval(0, 2), {"hull": None}):
            for incumbent in (0.0, 4.0, True, False):
                with self.subTest(outer=outer, incumbent=incumbent):
                    with self.assertRaises(TypeError):
                        certificate_hybrid(outer, incumbent)
        for invalid in (1.0, True, False):
            with self.subTest(encoded_input=invalid):
                with self.assertRaises(TypeError):
                    encode(invalid)

    def test_supported_exact_rational_incumbents_have_identical_results(self):
        outer = interval(0, 2)
        for incumbent in (Q(9, 4), "9/4"):
            with self.subTest(incumbent=incumbent):
                self.assert_result(certificate_hybrid(outer, incumbent),
                                   "MOVE", 2, Q(1, 16), "projection")
        self.assert_result(certificate_hybrid(outer, 4), "MOVE", 1, 3, "midpoint")

    def test_exhaustive_small_rational_intervals_decisions_and_dominance(self):
        grid = tuple(sorted({Q(n, d) for d in (1, 2, 3) for n in range(-3, 4)}))
        decision_cases = truth_cases = 0
        for index, left in enumerate(grid):
            for right in grid[index:]:
                outer = interval(left, right)
                center, width = (left + right) / 2, (right - left) / 2
                incumbents = sorted(set(grid) | {left - Q(1, 7), right + Q(1, 7), center,
                    center - 2 * width, center + 2 * width,
                    center - 2 * width - Q(1, 7), center + 2 * width + Q(1, 7)})
                references = sorted({q for q in grid if left <= q <= right} |
                                    {left + (right - left) * Q(j, 4) for j in range(5)})
                for incumbent in incumbents:
                    with self.subTest(left=left, right=right, incumbent=incumbent):
                        midpoint = decide(outer, incumbent)
                        projected = certificate_projection(outer, incumbent)
                        result = certificate_hybrid(outer, incumbent)
                        distance_center = abs(incumbent - center)
                        midpoint_moves = distance_center > 2 * width
                        outside = incumbent < left or incumbent > right
                        projection = min(max(incumbent, left), right)
                        expected_output = center if midpoint_moves else projection
                        expected_status = "MOVE" if outside else "KEEP"
                        expected_branch = "midpoint" if midpoint_moves else "projection" if outside else "keep"
                        expected_gain = (distance_center * (distance_center - 2 * width)
                                         if midpoint_moves else (incumbent - projection)**2)
                        self.assert_result(result, expected_status, expected_output, expected_gain, expected_branch)
                        self.assertEqual(midpoint["status"] == "MOVE", midpoint_moves)
                        self.assertEqual(result["status"] == "MOVE", projected["status"] == "MOVE")
                        if midpoint_moves:
                            self.assertEqual({k: v for k, v in result.items() if k != "branch"}, midpoint)
                        output = decode(result["output"])
                        midpoint_output = decode(midpoint["output"])
                        lower = decode(result["gain_lower"])
                        decision_cases += 1
                        for reference in references:
                            self.assertLessEqual(abs(output - reference), abs(midpoint_output - reference))
                            self.assertLessEqual((output - reference)**2, (midpoint_output - reference)**2)
                            actual_gain = (incumbent - reference)**2 - (output - reference)**2
                            self.assertGreaterEqual(actual_gain, lower)
                            if outside:
                                self.assertGreater(actual_gain, 0)
                            if incumbent == reference:
                                self.assertEqual((result["status"], output), ("KEEP", incumbent))
                            truth_cases += 1
        self.assertGreater(decision_cases, 2000)
        self.assertGreater(truth_cases, 15000)

    def test_hybrid_and_projection_can_each_have_smaller_actual_error(self):
        outer, incumbent = interval(0, 2), Q(4)
        result = certificate_hybrid(outer, incumbent)
        projected = certificate_projection(outer, incumbent)
        self.assert_result(result, "MOVE", 1, 3, "midpoint")
        self.assertEqual(decode(projected["output"]), 2)
        for reference, expected_hybrid, expected_projection in ((Q(2), Q(1), Q(0)), (Q(1), Q(0), Q(1))):
            with self.subTest(reference=reference):
                self.assertEqual(abs(decode(result["output"]) - reference), expected_hybrid)
                self.assertEqual(abs(decode(projected["output"]) - reference), expected_projection)
                self.assertEqual((decode(result["output"]) - reference)**2, expected_hybrid**2)
                self.assertEqual((decode(projected["output"]) - reference)**2, expected_projection**2)

    def test_fresh_outer_is_used_without_any_truth_argument(self):
        self.assertEqual(tuple(inspect.signature(certificate_hybrid).parameters), ("outer", "incumbent"))
        observation = (Q(3, 2),)
        outer = solve_outer(observation, identity_enclosure, lo=0, hi=3,
                            epsilon=Q(1, 8), tolerance=Q(1, 64), max_boxes=511)
        self.assertGreater(outer["evaluated_boxes"], 0)
        self.assertIsNotNone(outer["hull"])
        left, right = map(decode, outer["hull"])
        self.assertLessEqual(left, Q(3, 2))
        self.assertGreaterEqual(right, Q(3, 2))
        self.assert_result(certificate_hybrid(outer, Q(3, 2)), "KEEP", Q(3, 2), 0, "keep")
        for incumbent in (left - Q(1, 100), right + Q(1, 100), right + (right - left)):
            result = certificate_hybrid(outer, incumbent)
            midpoint = decide(outer, incumbent)
            self.assertEqual(result["status"], "MOVE")
            output = decode(result["output"])
            for reference in (Q(11, 8), Q(3, 2), Q(13, 8)):
                self.assertLessEqual(left, reference)
                self.assertGreaterEqual(right, reference)
                self.assertLessEqual(abs(output - reference), abs(decode(midpoint["output"]) - reference))

    def test_zero_budget_prior_keeps_inside_and_marks_prior_only_outside_actions(self):
        def must_not_run(_left, _right):
            raise AssertionError("zero work budget must not evaluate model")
        outer = solve_outer((Q(-999),), must_not_run, lo=540, hi=660, max_boxes=0)
        self.assertEqual(outer["evaluated_boxes"], 0)
        for incumbent in (Q(540), Q(600), Q(660)):
            self.assert_result(certificate_hybrid(outer, incumbent), "KEEP", incumbent, 0, "keep")
        self.assert_result(certificate_hybrid(outer, Q(661)), "MOVE", 660, 1, "projection")
        self.assert_result(certificate_hybrid(outer, Q(840)), "MOVE", 600, 28800, "midpoint")

    def test_missing_truth_coverage_is_a_counterexample_to_unconditional_safety(self):
        result = certificate_hybrid(interval(0, 2), Q(4))
        reference_outside_hull = Q(4)
        self.assertEqual(result["status"], "MOVE")
        self.assertGreater(abs(decode(result["output"]) - reference_outside_hull), 0)
        # The conditional theorem does not apply when the actual depth is outside H.


if __name__ == "__main__":
    unittest.main(verbosity=2)
