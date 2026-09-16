"""Independent small-case enumeration and behavioral checks for planning.py."""

import inspect
import itertools
import random
import unittest

import planning


def brute_prediction_masks(worlds, bits, budget, costs):
    """Enumerate all achievable adaptive prediction functions, not risk DP.

    Each returned integer encodes predictions on the listed worlds.  All
    combinations of both branch functions are explicitly enumerated, including
    functions that are suboptimal for any particular goal under test.
    """
    support_mask = sum(1 << world for world in worlds)
    predictions = {0, support_mask}
    for bit in bits:
        if costs[bit] > budget:
            continue
        zero = tuple(w for w in worlds if not (w >> bit) & 1)
        one = tuple(w for w in worlds if (w >> bit) & 1)
        if not zero or not one:
            continue
        rest = tuple(b for b in bits if b != bit)
        left = brute_prediction_masks(zero, rest, budget - costs[bit], costs)
        right = brute_prediction_masks(one, rest, budget - costs[bit], costs)
        predictions.update(a | b for a in left for b in right)
    return predictions


def brute_fixed_error(goal_mask, worlds, bits, budget, costs):
    """Enumerate subsets and terminal assignments on observed signatures."""
    best = len(worlds)
    for flags in itertools.product((0, 1), repeat=len(bits)):
        subset = tuple(bit for bit, flag in zip(bits, flags) if flag)
        if sum(costs[b] for b in subset) > budget:
            continue
        signatures = sorted({tuple((w >> b) & 1 for b in subset) for w in worlds})
        for guesses in itertools.product((0, 1), repeat=len(signatures)):
            actions = dict(zip(signatures, guesses))
            error = sum(actions[tuple((w >> b) & 1 for b in subset)] != ((goal_mask >> w) & 1)
                        for w in worlds)
            best = min(best, error)
    return best


class ExactPlanningTests(unittest.TestCase):
    def test_exhaustive_three_bit_adaptive_functions(self):
        worlds, bits = tuple(range(8)), (0, 1, 2)
        for budget in range(4):
            possible = brute_prediction_masks(worlds, bits, budget, (1, 1, 1, 1))
            for target in range(256):
                best = min((target ^ prediction).bit_count() for prediction in possible)
                found = planning.solve_adaptive(target, budget, bits, support=worlds)
                self.assertEqual(found["error_count"], best, (budget, target))

    def test_bruteforce_two_bit_all_goals_nonunit_costs(self):
        worlds, bits, costs = (0, 1, 2, 3), (0, 1), (1, 2, 3, 4)
        for budget in range(4):
            predictions = brute_prediction_masks(worlds, bits, budget, costs)
            for target in range(16):
                adaptive = planning.solve_adaptive(target, budget, bits, costs, support=worlds)
                fixed = planning.solve_open_loop(target, budget, bits, costs, support=worlds)
                self.assertEqual(adaptive["error_count"], min(
                    (target ^ prediction).bit_count() for prediction in predictions))
                self.assertEqual(fixed["error_count"], brute_fixed_error(
                    target, worlds, bits, budget, costs))

    def test_all_goal_dominance_replay_and_budget(self):
        configurations = [((0, 1, 2, 3), (1, 1, 1, 1)),
                          ((0, 1, 2, 3), (1, 2, 3, 4)),
                          ((1, 2, 3), (1, 1, 1, 1))]
        for goal in planning.goal_catalog():
            mask = goal["goal_mask"]
            for bits, costs in configurations:
                for budget in range(5):
                    results = [solver(mask, budget, bits, costs) for solver in planning.SOLVERS.values()]
                    self.assertLessEqual(results[0]["error_count"], results[1]["error_count"])
                    self.assertLessEqual(results[1]["error_count"], results[2]["error_count"])
                    for result in results:
                        errors = 0
                        for world in planning.ALL_WORLDS:
                            guess, trace = planning.replay_trace(result["policy"], world)
                            errors += guess != ((mask >> world) & 1)
                            self.assertTrue(all(step["bit"] in bits for step in trace))
                            self.assertEqual(len({step["bit"] for step in trace}), len(trace))
                            self.assertLessEqual(sum(costs[step["bit"]] for step in trace), budget)
                            self.assertTrue(all(step["cost"] == costs[step["bit"]] for step in trace))
                        self.assertEqual(errors, result["error_count"])

    def test_multiplexer_adaptivity_and_missing_selector(self):
        mask = planning.truth_mask(lambda w: (w >> (2 if w & 1 else 1)) & 1)
        self.assertEqual(planning.solve_adaptive(mask, 2)["error"], 0)
        self.assertEqual(planning.solve_open_loop(mask, 2)["error"], 0.25)
        self.assertEqual(planning.solve_fixed_prefix(mask, 2)["error"], 0.25)
        for solver in planning.SOLVERS.values():
            self.assertEqual(solver(mask, 4, (1, 2, 3))["error"], 0.25)

    def test_parity_and_no_sensor(self):
        parity = planning.truth_mask(lambda w: w.bit_count() % 2)
        for solver in planning.SOLVERS.values():
            for budget in range(4):
                self.assertEqual(solver(parity, budget)["error"], 0.5)
            self.assertEqual(solver(parity, 4)["error"], 0)
            self.assertEqual(solver(parity, 99, ())["error"], 0.5)

    def test_terminal_guess_responds_to_observations(self):
        for solver in (planning.solve_open_loop, planning.solve_fixed_prefix):
            result = solver(planning.BIT_WORLDS[0], 1)
            self.assertEqual(result["error"], 0)
            self.assertEqual(planning.replay(result["policy"], 0), 0)
            self.assertEqual(planning.replay(result["policy"], 1), 1)

    def test_cost_budget_and_prefix_no_skip(self):
        mask = planning.BIT_WORLDS[1]
        costs = (3, 1, 1, 1)
        self.assertEqual(planning.solve_fixed_prefix(mask, 2, costs=costs)["error"], 0.5)
        self.assertEqual(planning.solve_fixed_prefix(mask, 2, costs=costs)["query_subset"], [])
        self.assertEqual(planning.solve_open_loop(mask, 2, costs=costs)["error"], 0)
        self.assertEqual(planning.solve_adaptive(mask, 2, costs=costs)["error"], 0)
        self.assertEqual(planning.solve_fixed_prefix(mask, 4, costs=costs)["query_subset"], [0, 1])

    def test_empty_support_and_validation(self):
        for solver in planning.SOLVERS.values():
            with self.assertRaisesRegex(ValueError, "empty support"):
                solver(0, 0, support=())
            for kwargs in ({"support": (0, 0)}, {"support": (16,)},
                           {"costs": (0, 1, 1, 1)}, {"allowed_bits": (0, 0)}):
                with self.assertRaises(ValueError):
                    solver(0, 0, **kwargs)
            with self.assertRaises(ValueError):
                solver(65536, 0)
            with self.assertRaises(ValueError):
                solver(0, -1)

    def test_unreachable_support_branch_rejected(self):
        result = planning.solve_fixed_prefix(0, 1, support=(0, 2))
        with self.assertRaisesRegex(ValueError, "support"):
            planning.replay(result["policy"], 1)

    def test_no_world_argument_or_unqueried_sensor_leakage(self):
        for solver in planning.SOLVERS.values():
            self.assertNotIn("world", inspect.signature(solver).parameters)
        self.assertNotIn("goal_mask", inspect.signature(planning.replay_with_sensor).parameters)
        mask = planning.truth_mask(lambda w: (w >> (2 if w & 1 else 1)) & 1)
        policy = planning.solve_adaptive(mask, 2)["policy"]
        for world in planning.ALL_WORLDS:
            calls = []
            def sensor(bit):
                calls.append(bit)
                return (world >> bit) & 1
            guess, trace = planning.replay_with_sensor(policy, sensor)
            self.assertEqual(calls, [step["bit"] for step in trace])
            for other in planning.ALL_WORLDS:
                if all(((world ^ other) >> bit) & 1 == 0 for bit in calls):
                    self.assertEqual(planning.replay_trace(policy, other), (guess, trace))

    def test_goal_population_and_seed(self):
        goals = planning.goal_catalog()
        expected = {"multiplexer": 48, "affine": 30, "arbitrary": 32}
        for family, count in expected.items():
            selected = [g for g in goals if g["family"] == family]
            self.assertEqual(len(selected), count)
            if family != "arbitrary":
                self.assertEqual(len({g["goal_mask"] for g in selected}), count)
        rng = random.Random(2026091602)
        self.assertEqual([g["goal_mask"] for g in goals if g["family"] == "arbitrary"],
                         [rng.randrange(65536) for _ in range(32)])


if __name__ == "__main__":
    unittest.main()
