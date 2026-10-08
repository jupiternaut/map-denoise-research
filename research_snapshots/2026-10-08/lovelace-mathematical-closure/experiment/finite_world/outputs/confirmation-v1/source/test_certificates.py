"""Independent hand-calculated checks of the finite-world contract.

These tests check a supplied finite family, not a continuous scene class or
full9.  In particular, the two deliberate failures of truth coverage below
must remain counterexamples to an unconditional safety claim.
"""
from fractions import Fraction
import unittest

from certificates import (
    Hypothesis,
    certify,
    decode,
    encode,
    feasible_hypotheses,
    linf,
    minimum_residual,
)


Q = Fraction


def world(name, depth, *prediction):
    return Hypothesis(name, Q(depth), tuple(Q(x) for x in prediction))


class HandCalculatedCertificates(unittest.TestCase):
    def setUp(self):
        # Identical observations leave both target depths possible.
        self.two_targets = (world("near", 2, 0, 0), world("far", 3, 0, 0))
        self.observation = (Q(0), Q(0))

    def test_maximum_safe_gain_has_hand_calculated_value(self):
        # At targets 2 and 3, b=1 gives gains 3 and 5; b=2 gives 4
        # and 8; b=4 gives 0 and 8.  Thus the unique best lower bound is 4.
        result = certify(self.observation, self.two_targets, 0, 0, (0, 1, 2, 4))
        self.assertEqual(result.status, "MOVE")
        self.assertEqual(result.output, Q(2))
        self.assertEqual(result.certified_gain, Q(4))
        self.assertEqual(set(result.feasible_ids), {"near", "far"})

    def test_margin_equality_is_rejected(self):
        result = certify(self.observation, self.two_targets, 0, 0, (1, 2, 4), margin=4)
        self.assertEqual((result.status, result.output), ("KEEP", Q(0)))

    def test_margin_just_below_exact_gain_permits_move(self):
        result = certify(self.observation, self.two_targets, 0, 0, (2,), margin=Q(7, 2))
        self.assertEqual((result.status, result.output, result.certified_gain),
                         ("MOVE", Q(2), Q(4)))

    def test_zero_gain_is_not_a_strict_certificate(self):
        # b=4 does not change squared error when the true target is 2.
        result = certify(self.observation, self.two_targets, 0, 0, (4,))
        self.assertEqual((result.status, result.output), ("KEEP", Q(0)))

    def test_correct_incumbent_must_be_preserved(self):
        # A move away from 2 necessarily damages the permitted world "near".
        result = certify(self.observation, self.two_targets, 0, 2, (0, 1, 3, 4))
        self.assertEqual((result.status, result.output), ("KEEP", Q(2)))

    def test_same_observation_different_depths_prevents_universal_repair(self):
        hypotheses = (world("correct", 0, 7), world("wrong_old_point", 2, 7))
        result = certify((Q(7),), hypotheses, 0, 0, (2,))
        self.assertEqual(result.feasible_ids, ("correct", "wrong_old_point"))
        self.assertEqual((result.status, result.output), ("KEEP", Q(0)))
        self.assertEqual(minimum_residual((Q(7),), hypotheses, 0, (2,)), Q(0))

    def test_truth_is_not_inserted_into_the_allowed_action_pool(self):
        # The known finite worlds have depth 2; the caller permits only 1, 3
        # and KEEP=0.  Both 1 and 3 gain 3, and 1 is closer to KEEP.
        hypotheses = (world("fixed_target", 2, 9),)
        result = certify((Q(9),), hypotheses, 0, 0, (1, 3))
        self.assertEqual((result.status, result.output, result.certified_gain),
                         ("MOVE", Q(1), Q(3)))
        self.assertNotEqual(result.output, hypotheses[0].depth)

    def test_safe_gain_tie_favors_smaller_displacement(self):
        hypotheses = (world("fixed_target", 2, 9),)
        result = certify((Q(9),), hypotheses, 0, 4, (1, 3))
        self.assertEqual((result.output, result.certified_gain), (Q(3), Q(3)))

    def test_action_and_world_order_and_duplicates_do_not_change_action(self):
        first = certify(self.observation, self.two_targets, 0, 0, (0, 1, 2, 4))
        second = certify(self.observation, self.two_targets[::-1], 0, 0, (4, 2, 1, 2, 0, 4))
        self.assertEqual((first.status, first.output, first.certified_gain),
                         (second.status, second.output, second.certified_gain))
        self.assertEqual(set(first.feasible_ids), set(second.feasible_ids))

    def test_keep_is_available_even_with_empty_allowed_actions(self):
        result = certify(self.observation, self.two_targets, 0, 7, ())
        self.assertEqual((result.status, result.output), ("KEEP", Q(7)))

    def test_empty_consistency_set_is_incompatible_and_keeps_incumbent(self):
        hypotheses = (world("left", 1, -1), world("right", 2, 1))
        result = certify((Q(0),), hypotheses, Q(1, 2), 8, (1, 2))
        self.assertEqual((result.status, result.output), ("INCOMPATIBLE", Q(8)))
        self.assertEqual(result.feasible_ids, ())
        self.assertIsNone(result.certified_gain)
        self.assertEqual(result.reason, "no_consistent_world")

    def test_positive_gain_smaller_than_float_coordinate_resolution_is_kept_exact(self):
        tiny = Q(1, 10**18)
        target = Q(1) + tiny
        self.assertEqual(float(target), float(Q(1)))  # Not a solver input.
        result = certify((Q(0),), (world("sub_float_target", target, 0),),
                         0, 1, (target,))
        self.assertEqual((result.status, result.output), ("MOVE", target))
        self.assertEqual(result.certified_gain, Q(1, 10**36))
        self.assertIsInstance(result.output, Fraction)
        self.assertIsInstance(result.certified_gain, Fraction)

    def test_budget_violation_can_damage_an_actually_correct_point(self):
        # Actual world is "truth", but |y-F(truth)|=2 exceeds epsilon=0.
        # The finite certificate therefore applies to a different world only.
        truth = world("truth", 0, 0)
        other = world("wrong_explanation", 2, 2)
        result = certify((Q(2),), (truth, other), 0, 0, (2,))
        self.assertEqual((result.status, result.output, result.certified_gain),
                         ("MOVE", Q(2), Q(4)))
        self.assertNotIn(truth.world_id, result.feasible_ids)
        actual_gain = (Q(0) - truth.depth)**2 - (result.output - truth.depth)**2
        self.assertEqual(actual_gain, Q(-4))

    def test_finite_family_certificate_cannot_certify_omitted_continuous_worlds(self):
        # Continuous class: t in [1,2], F(t)=0.  The supplied finite pool
        # contains only t=2, omitting the actual t=1 with identical observation.
        result = certify((Q(0),), (world("sampled_depth_2", 2, 0),), 0, 1, (2,))
        self.assertEqual((result.status, result.output, result.certified_gain),
                         ("MOVE", Q(2), Q(1)))
        omitted_true_depth = Q(1)
        actual_gain = (Q(1) - omitted_true_depth)**2 - (result.output - omitted_true_depth)**2
        self.assertEqual(actual_gain, Q(-1))
        # This witness rejects a continuous or natural-world safety claim,
        # rather than requiring the finite solver to infer a missing world.

    def test_covered_truth_implies_no_harm_in_fixed_small_case_table(self):
        cases = (
            ((world("a", 2, 0), world("b", 3, 0)), 0, (1, 2, 4)),
            ((world("a", -3, 0), world("b", -2, 0)), 0, (-1, -2, -4)),
            ((world("a", -1, 0), world("b", 1, 0)), 0, (-2, 2)),
            ((world("a", 2, 0), world("b", 3, 0)), 2, (1, 3)),
            ((world("a", Q(1, 3), 0),), 1, (Q(1, 3), Q(1, 2))),
        )
        for hypotheses, incumbent, actions in cases:
            result = certify((Q(0),), hypotheses, 0, incumbent, actions)
            for true_world in hypotheses:
                with self.subTest(incumbent=incumbent, truth=true_world.depth):
                    old_error = (Q(incumbent) - true_world.depth)**2
                    new_error = (result.output - true_world.depth)**2
                    self.assertLessEqual(new_error, old_error)
                    if result.status == "MOVE":
                        self.assertGreater(old_error - new_error, 0)
                        self.assertLessEqual(result.certified_gain, old_error - new_error)
                    if true_world.depth == incumbent:
                        self.assertEqual(result.output, Q(incumbent))


class ExactObservationContract(unittest.TestCase):
    def test_epsilon_boundary_is_closed_and_exact(self):
        epsilon = Q(1, 3)
        tiny = Q(1, 10**18)
        hypotheses = (
            world("boundary", 1, epsilon, -epsilon),
            world("outside", 2, epsilon + tiny, 0),
            world("interior", 3, 0, epsilon / 2),
        )
        feasible = feasible_hypotheses((Q(0), Q(0)), hypotheses, epsilon)
        self.assertEqual(tuple(h.world_id for h in feasible), ("boundary", "interior"))

    def test_linf_bound_rejects_a_large_coordinate_even_if_mean_is_small(self):
        hypotheses = (world("large_second_coordinate", 1, 0, 2),)
        self.assertEqual(linf((Q(0), Q(2)), (Q(0), Q(0))), Q(2))
        self.assertEqual(feasible_hypotheses((Q(0), Q(0)), hypotheses, 1), ())

    def test_nonempty_common_observation_dimension_is_required(self):
        with self.assertRaises(ValueError):
            feasible_hypotheses((Q(0), Q(0)), (world("one_pixel", 1, 0),), 0)
        with self.assertRaises(ValueError):
            feasible_hypotheses((), (world("one_pixel", 1, 0),), 0)

    def test_negative_error_budget_or_margin_is_rejected(self):
        hypotheses = (world("w", 1, 0),)
        with self.assertRaises(ValueError):
            feasible_hypotheses((Q(0),), hypotheses, -1)
        with self.assertRaises(ValueError):
            certify((Q(0),), hypotheses, 0, 0, (1,), margin=-1)

    def test_empty_family_and_duplicate_world_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            feasible_hypotheses((Q(0),), (), 0)
        with self.assertRaises(ValueError):
            feasible_hypotheses((Q(0),), (world("same", 1, 0), world("same", 2, 0)), 0)

    def test_binary_float_inputs_are_rejected_instead_of_silently_rationalized(self):
        with self.assertRaises(TypeError):
            Hypothesis("float_depth", 0.1, (Q(0),))
        with self.assertRaises(TypeError):
            Hypothesis("float_pixel", Q(1), (0.1,))
        hypotheses = (world("w", 1, 0),)
        with self.assertRaises(TypeError):
            feasible_hypotheses((0.0,), hypotheses, 0)
        with self.assertRaises(TypeError):
            certify((Q(0),), hypotheses, 0.0, 0, (1,))
        with self.assertRaises(TypeError):
            certify((Q(0),), hypotheses, 0, 0.0, (1,))
        with self.assertRaises(TypeError):
            certify((Q(0),), hypotheses, 0, 0, (1.0,))

    def test_rational_json_representation_is_lossless(self):
        value = Q(-7, 13)
        self.assertEqual(encode(value), {"numerator": -7, "denominator": 13})
        self.assertEqual(decode(encode(value)), value)
        self.assertEqual(decode(encode(Q(1, 10**36))), Q(1, 10**36))


class SameFinitePoolBaselines(unittest.TestCase):
    def test_low_residual_is_not_geometric_truth_even_with_valid_budget(self):
        # Truth is depth 3, whose observation error 1/2 is allowed.
        # Depth 2 has a smaller residual, but moving 3 -> 2 damages truth.
        hypotheses = (world("d2", 2, 0), world("d3", 3, Q(1, 2)))
        baseline = minimum_residual((Q(0),), hypotheses, 3, (2, 3))
        certificate = certify((Q(0),), hypotheses, Q(1, 2), 3, (2, 3))
        self.assertEqual(baseline, Q(2))
        self.assertEqual((baseline - Q(3))**2, Q(1))
        self.assertEqual((certificate.status, certificate.output), ("KEEP", Q(3)))
        self.assertEqual(set(certificate.feasible_ids), {"d2", "d3"})

    def test_incumbent_wins_a_residual_score_tie(self):
        hypotheses = (world("first", 1, 0), world("old", 2, 0), world("last", 3, 0))
        self.assertEqual(minimum_residual((Q(0),), hypotheses, 2, (1, 3)), Q(2))

    def test_residual_baseline_scores_each_target_class_by_its_best_world(self):
        hypotheses = (world("bad_d2", 2, 10), world("good_d2", 2, 0), world("d3", 3, 1))
        self.assertEqual(minimum_residual((Q(0),), hypotheses, 3, (2, 3)), Q(2))

    def test_missing_incumbent_prediction_is_not_given_a_fictitious_score(self):
        hypotheses = (world("only_modeled_target", 2, 0),)
        self.assertEqual(minimum_residual((Q(0),), hypotheses, 9, (2,)), Q(2))

    def test_baseline_keeps_when_all_allowed_actions_have_no_prediction(self):
        hypotheses = (world("not_allowed", 2, 0),)
        self.assertEqual(minimum_residual((Q(0),), hypotheses, 9, (1,)), Q(9))

    def test_residual_tie_without_incumbent_uses_distance_then_coordinate(self):
        hypotheses = (world("positive", 1, 0), world("negative", -1, 0))
        self.assertEqual(minimum_residual((Q(0),), hypotheses, 0, (-1, 1)), Q(-1))


if __name__ == "__main__":
    unittest.main()
