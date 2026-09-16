"""Independent correctness/leakage audit. No production module writes or patches.

Run: python3 -m unittest -v audit_checks
The reference implementations intentionally avoid production truth-table,
loss-table, replay, and Bellman helper functions.
"""
from __future__ import annotations

import csv
import itertools
import json
import random
import re
import sys
import time
import unittest
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import composition
import learning
import planning

ROOT = Path(__file__).resolve().parent


def parse_expression(expression):
    tokens = re.findall(r"!?x[0-3]|[01()&|^]", expression)
    if "".join(tokens) != expression:
        raise AssertionError("Unrecognized expression token")
    iterator = iter(tokens)

    def parse():
        token = next(iterator)
        if token in ("0", "1"):
            return [int(token)] * 16, 0
        if "x" in token:
            bit = int(token[-1])
            return [((x >> bit) % 2) ^ token.startswith("!") for x in range(16)], 0
        if token != "(":
            raise AssertionError(token)
        left, left_gates = parse()
        op = next(iterator)
        right, right_gates = parse()
        if next(iterator) != ")":
            raise AssertionError("Unbalanced expression")
        operations = {"&": lambda a, b: int(bool(a) and bool(b)),
                      "|": lambda a, b: int(bool(a) or bool(b)),
                      "^": lambda a, b: int(a != b)}
        return [operations[op](a, b) for a, b in zip(left, right)], 1 + left_gates + right_gates

    values, gates = parse()
    if next(iterator, None) is not None:
        raise AssertionError("Unconsumed tokens")
    return sum(value << x for x, value in enumerate(values)), gates


def reference_grammar():
    # Exact-size semantic levels include redundant expressions; unlike production
    # this does not discard semantics represented at a smaller size.
    level = {0: "0", 65535: "1"}
    for bit in range(4):
        mask = sum(((x >> bit) % 2) << x for x in range(16))
        level[mask] = f"x{bit}"
        level[65535 - mask] = f"!x{bit}"
    levels, best = [level], {mask: (0, expr) for mask, expr in level.items()}
    for gates in range(1, 4):
        current = {}
        for left_size in range(gates):
            right_size = gates - left_size - 1
            for left_mask, left_expr in levels[left_size].items():
                for right_mask, right_expr in levels[right_size].items():
                    a, b = sorted((left_expr, right_expr))
                    for op, result in (("&", left_mask & right_mask),
                                       ("|", left_mask | right_mask),
                                       ("^", left_mask ^ right_mask)):
                        expression = f"({a}{op}{b})"
                        current[result] = min(current.get(result, expression), expression)
        levels.append(current)
        for mask, expression in current.items():
            best[mask] = min(best.get(mask, (gates, expression)), (gates, expression))
    return best


def reference_fixed_error(goal, sensors, costs, budget, worlds):
    optimum = len(worlds)
    for size in range(len(sensors) + 1):
        for subset in itertools.combinations(sensors, size):
            if sum(costs[bit] for bit in subset) > budget:
                continue
            groups = defaultdict(list)
            for world in worlds:
                groups[tuple((world >> bit) % 2 for bit in subset)].append((goal >> world) % 2)
            loss = sum(min(sum(labels), len(labels) - sum(labels)) for labels in groups.values())
            optimum = min(optimum, loss)
    return optimum


def reference_adaptive_error(goal, sensors, costs, budget, worlds):
    @lru_cache(None)
    def solve(possible, available, remaining):
        labels = [(goal >> world) % 2 for world in possible]
        best = min(sum(labels), len(labels) - sum(labels))
        for bit in available:
            if costs[bit] > remaining:
                continue
            branches = [tuple(world for world in possible if (world >> bit) % 2 == value)
                        for value in (0, 1)]
            if not all(branches):
                continue
            rest = tuple(other for other in available if other != bit)
            loss = sum(solve(branch, rest, remaining - costs[bit]) for branch in branches)
            best = min(best, loss)
        return best
    return solve(tuple(worlds), tuple(sensors), budget)


def independent_replay(policy, world, sensors=(0, 1, 2, 3), costs=(1, 1, 1, 1), budget=4):
    seen, spent = [], 0
    while policy["kind"] == "query":
        bit = policy["bit"]
        assert bit in sensors and bit not in seen
        assert policy["cost"] == costs[bit]
        seen.append(bit)
        spent += costs[bit]
        assert spent <= budget
        policy = policy["one"] if (world >> bit) % 2 else policy["zero"]
    assert policy["kind"] == "guess" and policy["value"] in (0, 1)
    return policy["value"], tuple(seen), spent


class IndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = learning.build_library()

    def test_complete_three_gate_grammar_and_expression_parser(self):
        expected = reference_grammar()
        actual = {p.mask: (p.gates, p.expression) for p in self.library}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(self.library))
        for program in self.library:
            self.assertEqual(parse_expression(program.expression), (program.mask, program.gates))

    def test_naive_erm_with_duplicate_and_conflicting_labels(self):
        rng = random.Random(716900)
        candidate_sets = (self.library, self.library[:32],
                          tuple(p for p in self.library if p.mask in learning.affine_masks()))
        for _ in range(16):
            samples = tuple((rng.randrange(16), rng.randrange(2)) for _ in range(25))
            for candidates in candidate_sets:
                losses = {p.mask: sum(((p.mask >> x) % 2) != y for x, y in samples)
                          for p in candidates}
                expected = min(candidates, key=lambda p: (losses[p.mask], p.gates, p.expression))
                result = learning.fit(samples, reversed(candidates))
                self.assertEqual(result.program, expected)
                self.assertEqual(result.train_errors, losses[expected.mask])
                self.assertEqual(result.consistent, losses[expected.mask] == 0)
                self.assertEqual(result.candidate_checks, len(candidates))
        with self.assertRaises(ValueError):
            learning.fit([], [])

    def test_hidden_labels_and_deliberate_cheating_positive_control(self):
        samples = ((0, 0), (2, 1), (4, 1), (6, 0), (6, 0))
        observed = {x for x, y in samples}
        first = sum(y << x for x, y in dict(samples).items())
        second = first ^ sum(1 << x for x in range(16) if x not in observed)
        self.assertTrue(all((first >> x) % 2 == (second >> x) % 2 == y for x, y in samples))

        def passes_invariance(learner):
            return learner(samples, first) == learner(samples, second)

        self.assertTrue(passes_invariance(lambda train, hidden: learning.fit(train, self.library).program.mask))
        self.assertFalse(passes_invariance(lambda train, hidden: hidden))
        model = learning.fit(samples, self.library).program.mask
        base = dict(family="audit", target_id="audit", regime="shift_x3", m=len(samples),
                    rep=0, method="exact", pred_mask=model)
        for budget in (0, 1, 2, 3, 4):
            for planner in ("adaptive", "batch"):
                row_a, detail_a = composition.evaluate_one(dict(base, target_mask=first), budget, planner)
                row_b, detail_b = composition.evaluate_one(dict(base, target_mask=second), budget, planner)
                self.assertEqual(detail_a["deployed_policy"], detail_b["deployed_policy"])
                self.assertEqual(row_a["action_mask"], row_b["action_mask"])
                self.assertEqual(row_a["predicted_error"], row_b["predicted_error"])

    def test_analytic_information_floors_and_adaptivity(self):
        mask = lambda fn: sum(int(fn(world)) << world for world in range(16))
        parity = mask(lambda w: w.bit_count() % 2)
        mux = mask(lambda w: ((w >> 1) % 2) if w % 2 else ((w >> 2) % 2))
        x3 = mask(lambda w: (w >> 3) % 2)
        fixtures = [(0, (0, 1, 2, 3), [0] * 5, [0] * 5),
                    (parity, (0, 1, 2, 3), [.5] * 4 + [0], [.5] * 4 + [0]),
                    (mux, (0, 1, 2, 3), [.5, .25, 0, 0, 0], [.5, .25, .25, 0, 0]),
                    (x3, (0, 1, 2), [.5] * 5, [.5] * 5)]
        for goal, sensors, adaptive, fixed in fixtures:
            for budget in range(5):
                for solver, errors in ((planning.solve_adaptive, adaptive),
                                       (planning.solve_open_loop, fixed)):
                    result = solver(goal, budget, sensors)
                    self.assertEqual(result["error"], errors[budget])
                    guesses = [independent_replay(result["policy"], world, sensors, budget=budget)[0]
                               for world in range(16)]
                    self.assertEqual(sum(guess != (goal >> world) % 2 for world, guess in enumerate(guesses)),
                                     result["error_count"])

    def test_random_planners_against_independent_reference(self):
        rng = random.Random(716901)
        for case in range(48):
            goal = rng.randrange(65536)
            sensors = tuple(bit for bit in range(4) if rng.random() < .8)
            costs = tuple(rng.randrange(1, 4) for _ in range(4))
            support = tuple(sorted(rng.sample(range(16), rng.randrange(1, 17))))
            for budget in range(6):
                results = []
                for solver, reference in ((planning.solve_adaptive, reference_adaptive_error),
                                          (planning.solve_open_loop, reference_fixed_error)):
                    result = solver(goal, budget, sensors, costs, support=support)
                    expected = reference(goal, sensors, costs, budget, support)
                    self.assertEqual(result["error_count"], expected, (case, budget, solver.__name__))
                    self.assertEqual(result["support_size"], len(support))
                    guesses = [independent_replay(result["policy"], world, sensors, costs, budget)[0]
                               for world in support]
                    self.assertEqual(sum(guess != (goal >> world) % 2 for world, guess in zip(support, guesses)), expected)
                    results.append(expected)
                self.assertLessEqual(results[0], results[1])
        for solver in (planning.solve_adaptive, planning.solve_open_loop, planning.solve_fixed_prefix):
            with self.assertRaises(ValueError):
                solver(0, 1, support=())

    def test_sensor_callable_reads_only_selected_bits(self):
        goal = sum((((w >> 1) % 2) if w % 2 else ((w >> 2) % 2)) << w for w in range(16))
        policy = planning.solve_adaptive(goal, 2)["policy"]
        for world in range(16):
            queries = []
            def read_bit(bit):
                queries.append(bit)
                return (world >> bit) % 2
            guess, trace = planning.replay_with_sensor(policy, read_bit)
            expected, bits, _ = independent_replay(policy, world, budget=2)
            self.assertEqual(guess, expected)
            self.assertEqual(queries, list(bits))
            self.assertEqual([entry["bit"] for entry in trace], list(bits))

    def test_composition_risk_and_oracle_bound_independently(self):
        rng = random.Random(716902)
        for case in range(24):
            target, model = rng.randrange(65536), rng.randrange(65536)
            base = dict(family="audit", target_id=str(case), regime="uniform", m=8,
                        rep=0, method="audit", pred_mask=model, target_mask=target)
            for budget in (2, 3, 4):
                for name, reference in (("adaptive", reference_adaptive_error),
                                        ("batch", reference_fixed_error)):
                    row, detail = composition.evaluate_one(base, budget, name)
                    values = [independent_replay(detail["deployed_policy"], world, budget=budget)[0]
                              for world in range(16)]
                    actual = sum(value != ((target >> world) % 2) for world, value in enumerate(values)) / 16
                    model_risk = sum(value != ((model >> world) % 2) for world, value in enumerate(values)) / 16
                    oracle = reference(target, (0, 1, 2, 3), (1, 1, 1, 1), budget, tuple(range(16))) / 16
                    self.assertEqual(row["actual_error"], actual)
                    self.assertEqual(row["predicted_error"], model_risk)
                    self.assertEqual(row["oracle_error"], oracle)
                    self.assertGreaterEqual(actual, oracle)
                    if budget == 4:
                        self.assertEqual(actual, (model ^ target).bit_count() / 16)


class RawArtifactAudit(unittest.TestCase):
    """Every saved row is checked, in addition to the solver fixtures above."""

    def test_all_learning_rows_and_sampling_integrity(self):
        catalog = json.loads((ROOT / "results/learning_catalog.json").read_text())
        sample_data = json.loads((ROOT / "results/learning_samples.json").read_text())
        samples = {(r["target_id"], r["regime"], r["rep"]): r for r in sample_data["samples"]}
        targets = {target["target_id"]: target for target in catalog["targets"]}
        programs = {program["mask"]: program for program in catalog["programs"]}
        rows = list(csv.DictReader((ROOT / "results/learning_rows.csv").open()))
        self.assertEqual(len(rows), 144 * 2 * 8 * 4 * 3)
        exact_consistency = {(r["target_id"], r["regime"], r["rep"], r["m"]): int(r["consistent"])
                             for r in rows if r["method"] == "exact"}
        seen_keys = set()
        for stream in samples.values():
            support = tuple(range(16)) if stream["regime"] == "uniform" else tuple(range(8))
            rng = random.Random(stream["derived_seed"])
            self.assertEqual([x for x, y in stream["samples"]], [rng.choice(support) for _ in range(32)])
            target = targets[stream["target_id"]]["target_mask"]
            self.assertTrue(all(y == (target >> x) % 2 for x, y in stream["samples"]))
        for row in rows:
            key = (row["target_id"], row["regime"], row["rep"], row["m"], row["method"])
            self.assertNotIn(key, seen_keys)
            seen_keys.add(key)
            train = samples[(row["target_id"], row["regime"], int(row["rep"]))]["samples"][:int(row["m"])]
            target, model = int(row["target_mask"]), int(row["pred_mask"])
            self.assertEqual(target, targets[row["target_id"]]["target_mask"])
            self.assertEqual(row["expression"], programs[model]["expression"])
            self.assertEqual(int(row["candidate_checks"]), catalog["method_candidate_counts"][row["method"]])
            seen = {x for x, y in train}
            unseen = set(range(16)) - seen
            wrong = {x for x in range(16) if (target >> x) % 2 != (model >> x) % 2}
            train_errors = sum((model >> x) % 2 != y for x, y in train)
            self.assertEqual(float(row["train_error"]), train_errors / len(train))
            self.assertEqual(int(row["consistent"]), int(train_errors == 0))
            self.assertEqual(int(row["candidate_set_support_failure"]), int(train_errors != 0))
            self.assertEqual(float(row["full_domain_error"]), len(wrong) / 16)
            self.assertEqual(float(row["seen_error"]), len(wrong & seen) / len(seen))
            self.assertEqual(int(row["seen_unique"]), len(seen))
            self.assertEqual(int(row["unseen_count"]), len(unseen))
            if unseen:
                self.assertEqual(float(row["unseen_error"]), len(wrong & unseen) / len(unseen))
            else:
                self.assertEqual(row["unseen_error"], "")
            deploy = set(range(16)) if row["regime"] == "uniform" else set(range(8, 16))
            self.assertEqual(float(row["test_error"]), len(wrong & deploy) / len(deploy))
            prefix = row["method"] == "prefix32"
            exact_ok = exact_consistency[key[:4]]
            representation_failure = not exact_ok if prefix else train_errors != 0
            search_failure = prefix and train_errors != 0 and exact_ok
            self.assertEqual(int(row["representation_support_failure"]), int(representation_failure))
            self.assertEqual(int(row["search_budget_failure"]), int(search_failure))

    def test_all_planning_policies_and_saved_traces(self):
        goals = {row["goal_id"]: row["goal_mask"] for row in json.loads((ROOT / "results/planning_goals.json").read_text())}
        key_fields = ("case", "goal_id", "method", "budget")
        key = lambda row: tuple(str(row[field]) for field in key_fields)
        rows = {key(row): row for row in csv.DictReader((ROOT / "results/planning_rows.csv").open())}
        policies = [json.loads(line) for line in (ROOT / "results/planning_policies.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), len(policies))
        seen = set()
        expected_traces = {}
        for item in policies:
            ident = key(item)
            self.assertNotIn(ident, seen)
            seen.add(ident)
            row = rows[ident]
            sensors = tuple(int(value) for value in row["allowed_bits"].split("|") if value)
            costs = tuple(int(value) for value in row["costs"].split("|"))
            goal, budget = goals[row["goal_id"]], int(row["budget"])
            errors, spent, all_queries = [], [], []
            for world in range(16):
                guess, queries, total = independent_replay(item["policy"], world, sensors, costs, budget)
                truth = (goal >> world) % 2
                errors.append(guess != truth)
                spent.append(total)
                all_queries.append(queries)
                expected_traces[ident + (str(world),)] = dict(
                    truth=str(truth), guess=str(guess), error=str(int(guess != truth)),
                    queried_bits="|".join(map(str, queries)),
                    observations="|".join(str((world >> bit) % 2) for bit in queries), total_cost=str(total))
            self.assertEqual(sum(errors), int(row["error_count"]))
            self.assertEqual(sum(errors) / 16, float(row["error"]))
            self.assertEqual(sum(spent) / 16, float(row["mean_query_cost"]))
            self.assertEqual(max(spent), int(row["max_query_cost"]))
            if row["method"] in ("open_loop", "fixed_prefix"):
                self.assertEqual(len(set(all_queries)), 1)
            if row["case"] == "missing_sensor":
                self.assertNotIn(int(row["removed_bit"]), sensors)
        traces = list(csv.DictReader((ROOT / "results/planning_traces.csv").open()))
        self.assertEqual(len(traces), len(expected_traces))
        seen_traces = set()
        for trace in traces:
            ident = key(trace) + (trace["world"],)
            self.assertNotIn(ident, seen_traces)
            seen_traces.add(ident)
            for field, expected in expected_traces[ident].items():
                self.assertEqual(trace[field], expected)

    def test_all_composition_rows_and_adaptive_reversal_attribution(self):
        rows = list(csv.DictReader((ROOT / "results/composition_rows.csv").open()))
        self.assertEqual(len(rows), 144 * 2 * 8 * 2 * 3 * 3 * 2)
        pairs = defaultdict(dict)
        keys = ("target_id", "regime", "m", "rep", "method", "budget")
        for row in rows:
            model, target, action = (int(row[field]) for field in ("pred_mask", "target_mask", "action_mask"))
            self.assertEqual(float(row["actual_error"]), (action ^ target).bit_count() / 16)
            self.assertEqual(float(row["predicted_error"]), (action ^ model).bit_count() / 16)
            self.assertEqual(float(row["program_error"]), (model ^ target).bit_count() / 16)
            self.assertEqual(float(row["high_bit_error"]), ((action ^ target) & 65280).bit_count() / 8)
            self.assertEqual(float(row["excess_error"]), float(row["actual_error"]) - float(row["oracle_error"]))
            self.assertGreaterEqual(float(row["excess_error"]), 0)
            if int(row["budget"]) == 4:
                self.assertEqual(action, model)
            pair_key = tuple(row[key] for key in keys)
            self.assertNotIn(row["planner"], pairs[pair_key])
            pairs[pair_key][row["planner"]] = row
        reversals, strict_model_reversals = 0, 0
        for pair in pairs.values():
            self.assertEqual(set(pair), {"adaptive", "batch"})
            adaptive, batch = pair["adaptive"], pair["batch"]
            self.assertLessEqual(float(adaptive["predicted_error"]), float(batch["predicted_error"]))
            if float(adaptive["actual_error"]) > float(batch["actual_error"]):
                reversals += 1
                strict_model_reversals += float(adaptive["predicted_error"]) < float(batch["predicted_error"])
        # These are descriptive findings from the frozen suite, not prespecified
        # thresholds used to choose a population or declare scientific success.
        self.assertEqual(reversals, 131)
        self.assertEqual(strict_model_reversals, 0)


if __name__ == "__main__":
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    artifact = {
        "tests_run": result.testsRun,
        "failures": [{"test": str(test), "traceback": traceback} for test, traceback in result.failures],
        "errors": [{"test": str(test), "traceback": traceback} for test, traceback in result.errors],
        "skipped": [(str(test), reason) for test, reason in result.skipped],
        "passed": result.wasSuccessful(),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Finite fixtures and raw artifacts; behavioral noninterference is not process isolation.",
        "verified_rows": {"learning": 27648, "planning": 3630, "planning_traces": 58080, "composition": 82944}
            if result.wasSuccessful() else {},
    }
    (ROOT / "results/audit_results.json").write_text(json.dumps(artifact, indent=2) + "\n")
    sys.exit(not result.wasSuccessful())
