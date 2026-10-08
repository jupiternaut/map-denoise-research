"""Exact hand-computable acceptance tests for continuous outer-set decisions.

These tests exercise the generic solver independently of image rendering. The
callbacks enclose WHOLE continuous parameter sets; they are not finite samples.
Actual parameters enter only observation generation and subsequent evaluation.
"""

from fractions import Fraction as Q
import unittest

from .interval_solver import solve_outer, decide


def q(raw):
    return Q(raw["numerator"], raw["denominator"])


def hull(outer):
    return None if outer["hull"] is None else tuple(q(value) for value in outer["hull"])


def identity(lo, hi):
    return ((lo, hi),)


def square_about(point):
    def enclosure(lo, hi):
        endpoint_values = ((lo - point) ** 2, (hi - point) ** 2)
        minimum = Q(0) if lo <= point <= hi else min(endpoint_values)
        return ((minimum, max(endpoint_values)),)
    return enclosure


def contains(outer, value):
    by_id = {entry["id"]: entry for entry in outer["trace"]}
    return any(q(by_id[node]["lo"]) <= value <= q(by_id[node]["hi"])
               for node in outer["retained_ids"])


def recorded_interval(lo, hi):
    return {"hull": [{"numerator": Q(lo).numerator, "denominator": Q(lo).denominator},
                     {"numerator": Q(hi).numerator, "denominator": Q(hi).denominator}]}


class IntervalSolverTests(unittest.TestCase):
    def test_interior_quadratic_solution_not_excluded_by_endpoint_samples(self):
        truth = Q(3, 2)
        # Both endpoint samples equal 1/4 while the true interior prediction is 0.
        self.assertEqual((Q(1) - truth) ** 2, Q(1, 4))
        self.assertEqual((Q(2) - truth) ** 2, Q(1, 4))
        outer = solve_outer((Q(0),), square_about(truth), lo=1, hi=2,
                            epsilon=0, tolerance=Q(1, 64), max_boxes=255)
        self.assertEqual(outer["trace"][0]["status"], "split")
        self.assertTrue(contains(outer, truth))
        self.assertIsNotNone(hull(outer))

    def test_non_grid_true_depth_is_covered(self):
        truth = Q(1201, 2)  # 600.5 mm, absent from the stage-1 nine-depth grid.
        outer = solve_outer((truth,), identity, lo=540, hi=660, epsilon=0,
                            tolerance=Q(15, 64), max_boxes=511)
        self.assertTrue(contains(outer, truth))
        self.assertLessEqual(hull(outer)[0], truth)
        self.assertGreaterEqual(hull(outer)[1], truth)

    def test_error_budget_boundary_is_closed(self):
        # Only depth 2 can meet |prediction-3| <= 1 in this domain.
        outer = solve_outer((Q(3),), identity, lo=1, hi=2, epsilon=1,
                            tolerance=Q(1, 32), max_boxes=127)
        self.assertTrue(contains(outer, Q(2)))
        self.assertEqual(hull(outer)[1], 2)
        self.assertNotEqual(outer["trace"][0]["status"], "excluded")

    def test_strictly_separated_box_is_proved_empty(self):
        outer = solve_outer((Q(3),), identity, lo=1, hi=2, epsilon=Q(1, 2))
        self.assertIsNone(hull(outer))
        self.assertEqual(outer["retained_ids"], [])
        self.assertEqual(outer["trace"][0]["status"], "excluded")
        decision = decide(outer, Q(3, 2))
        self.assertEqual(decision["status"], "INCOMPATIBLE")
        self.assertEqual(q(decision["output"]), Q(3, 2))
        self.assertIsNone(decision["gain_lower"])

    def test_zero_work_budget_keeps_root_without_evaluating_model(self):
        def must_not_run(_lo, _hi):
            raise AssertionError("budget=0 must not invoke enclosure")
        outer = solve_outer((Q(100),), must_not_run, lo=1, hi=2,
                            epsilon=0, max_boxes=0)
        self.assertEqual(hull(outer), (Q(1), Q(2)))
        self.assertEqual(outer["evaluated_boxes"], 0)
        self.assertEqual(outer["trace"][0]["status"], "unresolved_budget")
        self.assertEqual(outer["retained_ids"], [0])

    def test_unprocessed_children_remain_in_complete_outer_hull(self):
        outer = solve_outer((Q(3, 2),), identity, lo=1, hi=2,
                            epsilon=0, tolerance=Q(1, 64), max_boxes=1)
        self.assertEqual(outer["evaluated_boxes"], 1)
        self.assertEqual(hull(outer), (Q(1), Q(2)))
        self.assertEqual(len(outer["retained_ids"]), 2)
        self.assertEqual({entry["status"] for entry in outer["trace"][1:]},
                         {"unresolved_budget"})
        self.assertTrue(contains(outer, Q(3, 2)))

    def test_exhausted_budget_cannot_fabricate_empty_set(self):
        # Observation is actually incompatible, but no work means no proof of it.
        outer = solve_outer((Q(100),), identity, lo=1, hi=2, epsilon=0, max_boxes=0)
        self.assertIsNotNone(hull(outer))
        self.assertEqual(decide(outer, Q(3, 2))["status"], "KEEP")

    def test_bisection_children_exactly_cover_parent_with_shared_endpoint(self):
        outer = solve_outer((Q(5, 3),), identity, lo=1, hi=2, epsilon=Q(1, 8),
                            tolerance=Q(1, 32), max_boxes=127)
        by_id = {entry["id"]: entry for entry in outer["trace"]}
        self.assertEqual(len(by_id), len(outer["trace"]))
        for entry in outer["trace"]:
            if entry["status"] != "split":
                continue
            first, second = (by_id[node] for node in entry["children"])
            lo, hi = q(entry["lo"]), q(entry["hi"])
            mid = (lo + hi) / 2
            self.assertEqual((q(first["lo"]), q(first["hi"])), (lo, mid))
            self.assertEqual((q(second["lo"]), q(second["hi"])), (mid, hi))
            self.assertEqual(first["parent"], entry["id"])
            self.assertEqual(second["parent"], entry["id"])

    def test_larger_budget_preserves_previously_true_depth(self):
        truth, observation = Q(3), Q(17, 5)
        small = solve_outer((observation,), identity, lo=1, hi=5,
                            epsilon=Q(2, 5), tolerance=Q(1, 32), max_boxes=255)
        large = solve_outer((observation,), identity, lo=1, hi=5,
                            epsilon=Q(3, 5), tolerance=Q(1, 32), max_boxes=255)
        self.assertTrue(contains(small, truth))
        self.assertTrue(contains(large, truth))

    def test_two_continuous_camera_nuisances_include_interior_values(self):
        # Analytic proxy F(z,left,right)=(z+left,z+right), not an image claim.
        # Neither nuisance is represented by endpoint-only camera hypotheses.
        truth = Q(23, 7)
        left_actual, right_actual = Q(1, 3), Q(-2, 7)
        left_range, right_range = (Q(-1, 2), Q(1, 2)), (Q(-1, 3), Q(2, 3))
        observation = (truth + left_actual, truth + right_actual)
        def all_camera_enclosure(lo, hi):
            return ((lo + left_range[0], hi + left_range[1]),
                    (lo + right_range[0], hi + right_range[1]))
        # No endpoint-camera pair can match BOTH observations at one depth.
        self.assertTrue(all(observation[0] - left != observation[1] - right
                            for left in left_range for right in right_range))
        outer = solve_outer(observation, all_camera_enclosure, lo=1, hi=5,
                            epsilon=0, tolerance=Q(1, 64), max_boxes=255)
        self.assertTrue(contains(outer, truth))
        self.assertIsNotNone(hull(outer))
        self.assertEqual(decide(outer, truth)["status"], "KEEP")

    def test_continuous_nuisance_budget_enlargement_still_covers_truth(self):
        truth, left, right = Q(1201, 2), Q(1, 3), Q(-2, 5)
        observation = (truth + left, truth + right)
        def enclosure(lo, hi):
            return ((lo - Q(1, 2), hi + Q(1, 2)),
                    (lo - Q(1, 2), hi + Q(1, 2)))
        for epsilon in (Q(0), Q(1, 100), Q(1, 20)):
            outer = solve_outer(observation, enclosure, lo=540, hi=660,
                                epsilon=epsilon, tolerance=Q(15, 64), max_boxes=511)
            self.assertTrue(contains(outer, truth))

    def test_endpoint_gain_is_exact_and_strict(self):
        outer = recorded_interval(1, 2)
        positive = decide(outer, 0, proposal=1)
        self.assertEqual(positive["status"], "MOVE")
        self.assertEqual(q(positive["gain_lower"]), 1)
        self.assertEqual(q(positive["output"]), 1)
        touching = decide(outer, 0, proposal=2)
        self.assertEqual(touching["status"], "KEEP")
        self.assertEqual(q(touching["output"]), 0)
        negative = decide(outer, 0, proposal=3)
        self.assertEqual(negative["status"], "KEEP")

    def test_default_midpoint_has_rational_endpoint_certificate(self):
        decision = decide(recorded_interval(1, 2), 0)
        self.assertEqual(decision["status"], "MOVE")
        self.assertEqual(q(decision["output"]), Q(3, 2))
        self.assertEqual(q(decision["gain_lower"]), Q(3, 4))

    def test_correct_non_grid_point_is_always_kept_when_covered(self):
        truth = Q(1201, 2)
        observation = truth + Q(1, 200)
        outer = solve_outer((observation,), identity, lo=540, hi=660,
                            epsilon=Q(1, 100), tolerance=Q(15, 64), max_boxes=511)
        self.assertTrue(contains(outer, truth))
        decision = decide(outer, truth)
        self.assertEqual(decision["status"], "KEEP")
        self.assertEqual(q(decision["output"]), truth)

    def test_correct_point_is_kept_even_with_wide_unfinished_outer_set(self):
        truth = Q(23, 7)
        outer = solve_outer((truth,), identity, lo=1, hi=5, epsilon=0, max_boxes=0)
        self.assertEqual(decide(outer, truth)["status"], "KEEP")
        self.assertEqual(q(decide(outer, truth)["output"]), truth)

    def test_singleton_domain_and_singleton_certificate(self):
        truth = Q(23, 7)
        outer = solve_outer((truth,), identity, lo=truth, hi=truth,
                            epsilon=0, tolerance=1, max_boxes=1)
        self.assertEqual(hull(outer), (truth, truth))
        decision = decide(outer, truth + 2)
        self.assertEqual(decision["status"], "MOVE")
        self.assertEqual(q(decision["output"]), truth)
        self.assertEqual(q(decision["gain_lower"]), 4)

    def test_covered_offset_move_improves_every_feasible_depth(self):
        outer = solve_outer((Q(7, 4),), identity, lo=1, hi=2,
                            epsilon=Q(1, 20), tolerance=Q(1, 64), max_boxes=255)
        decision = decide(outer, Q(1))
        self.assertEqual(decision["status"], "MOVE")
        a, b, gain = Q(1), q(decision["output"]), q(decision["gain_lower"])
        for truth in (Q(17, 10), Q(7, 4), Q(9, 5)):
            self.assertTrue(contains(outer, truth))
            self.assertGreaterEqual((a - truth) ** 2 - (b - truth) ** 2, gain)

    def test_budget_violation_is_a_separate_counterexample_not_safe_success(self):
        actual_truth, observation = Q(3, 2), Q(5, 2)
        declared_epsilon = Q(0)
        self.assertGreater(abs(observation - actual_truth), declared_epsilon)
        outer = solve_outer((observation,), identity, lo=1, hi=3,
                            epsilon=declared_epsilon, tolerance=Q(1, 64), max_boxes=255)
        self.assertFalse(contains(outer, actual_truth))
        decision = decide(outer, actual_truth)
        self.assertEqual(decision["status"], "MOVE")
        self.assertLess(-(q(decision["output"]) - actual_truth) ** 2, 0)
        # This explicitly demonstrates the missing-coverage failure outside contract.

    def test_invalid_exactness_and_parameters_are_rejected(self):
        invalid_calls = (
            lambda: solve_outer((1.0,), identity),
            lambda: solve_outer((1,), identity, epsilon=0.01),
            lambda: solve_outer((1,), identity, lo=True),
            lambda: decide(recorded_interval(1, 2), 0.0),
        )
        for call in invalid_calls:
            with self.assertRaises(TypeError):
                call()
        for kwargs in ({"lo":2, "hi":1}, {"epsilon":-1}, {"tolerance":0},
                       {"max_boxes":-1}, {"max_boxes":True}):
            with self.assertRaises(ValueError):
                solve_outer((1,), identity, **kwargs)
        with self.assertRaises(ValueError):
            solve_outer((), identity)

    def test_invalid_model_bounds_cannot_be_used_for_certification(self):
        with self.assertRaises(ValueError):
            solve_outer((1,), lambda _lo, _hi: ((Q(2), Q(1)),))
        with self.assertRaises(ValueError):
            solve_outer((1, 2), lambda _lo, _hi: ((Q(1), Q(2)),))
        with self.assertRaises(TypeError):
            solve_outer((1,), lambda _lo, _hi: ((1.0, 2),))


if __name__ == "__main__":
    unittest.main(verbosity=2)
