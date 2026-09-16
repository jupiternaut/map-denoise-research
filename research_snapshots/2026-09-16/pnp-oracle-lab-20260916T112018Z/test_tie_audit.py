"""Exhaustive independent prediction-function checks of the post hoc envelope."""

import random
import unittest

import planning
from tie_audit import tie_envelope


def enumerate_functions(worlds, bits, budget):
    functions = {0, sum(1 << world for world in worlds)}
    if not budget:
        return functions
    for bit in bits:
        low = tuple(world for world in worlds if (world >> bit) & 1 == 0)
        high = tuple(world for world in worlds if (world >> bit) & 1 == 1)
        if not low or not high:
            continue
        rest = tuple(b for b in bits if b != bit)
        for left in enumerate_functions(low, rest, budget - 1):
            for right in enumerate_functions(high, rest, budget - 1):
                functions.add(left | right)
    return functions


class TieEnvelopeTests(unittest.TestCase):
    def test_exhaustive_two_bit_models_targets_and_budgets(self):
        worlds, bits = (0, 1, 2, 3), (0, 1)
        for budget in range(3):
            possible = enumerate_functions(worlds, bits, budget)
            for model in range(16):
                optimum = min((model ^ action).bit_count() for action in possible)
                optimal_actions = [action for action in possible if (model ^ action).bit_count() == optimum]
                for truth in range(16):
                    errors = [(truth ^ action).bit_count() for action in optimal_actions]
                    result = tie_envelope(model, truth, budget, worlds=worlds, bits=bits)
                    self.assertEqual(result["model_error_count"], optimum)
                    self.assertEqual(result["min_true_error_count"], min(errors))
                    self.assertEqual(result["max_true_error_count"], max(errors))

    def test_both_terminal_guesses_retained_on_tie(self):
        result = tie_envelope(planning.BIT_WORLDS[0], 0, 0)
        self.assertEqual(result["model_error"], 0.5)
        self.assertEqual(result["min_true_error"], 0)
        self.assertEqual(result["max_true_error"], 1)

    def test_model_equals_truth_removes_error_ambiguity(self):
        rng = random.Random(872)
        for _ in range(20):
            mask = rng.randrange(65536)
            for budget in range(5):
                result = tie_envelope(mask, mask, budget)
                self.assertEqual(result["min_true_error"], result["model_error"])
                self.assertEqual(result["max_true_error"], result["model_error"])

    def test_original_policy_inside_envelope_and_primary_objective_unchanged(self):
        rng = random.Random(873)
        for _ in range(30):
            model, truth = rng.randrange(65536), rng.randrange(65536)
            for budget in range(5):
                original = planning.solve_adaptive(model, budget)
                actual = sum(planning.replay(original["policy"], world) != ((truth >> world) & 1)
                             for world in range(16)) / 16
                result = tie_envelope(model, truth, budget)
                self.assertEqual(original["error"], result["model_error"])
                self.assertLessEqual(result["min_true_error"], actual)
                self.assertLessEqual(actual, result["max_true_error"])

    def test_empty_support_rejected(self):
        with self.assertRaises(ValueError):
            tie_envelope(0, 0, 0, worlds=())


if __name__ == "__main__":
    unittest.main()
