"""Finite exact sensing policies for Experiment B (Python standard library only).

World w is a four-bit integer; bit i is (w >> i) & 1.  Bit w of
goal_mask is the desired terminal action in world w.  The public support
is uniform, and is not the identity of the actual world.  Each solver starts
fresh at each call/budget.  See PLANNING_NOTES.md for diagnostic definitions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

ALL_WORLDS = tuple(range(16))
ALL_BITS = (0, 1, 2, 3)
ARBITRARY_SEED = 2026091602
BIT_WORLDS = tuple(sum(1 << w for w in ALL_WORLDS if (w >> bit) & 1)
                   for bit in ALL_BITS)


def _validate(goal_mask, budget, allowed_bits, costs, support):
    if type(goal_mask) is not int or not 0 <= goal_mask < (1 << 16):
        raise ValueError("goal_mask must be an integer in [0, 65535]")
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    allowed_bits, costs, support = tuple(allowed_bits), tuple(costs), tuple(support)
    if (len(set(allowed_bits)) != len(allowed_bits)
            or any(type(b) is not int or b not in ALL_BITS for b in allowed_bits)):
        raise ValueError("allowed_bits must be distinct bit indices in [0, 3]")
    if len(costs) != 4 or any(type(c) is not int or c <= 0 for c in costs):
        raise ValueError("costs must contain four positive integers")
    if not support:
        raise ValueError("empty support is a support failure, not zero risk")
    if (len(set(support)) != len(support)
            or any(type(w) is not int or w not in ALL_WORLDS for w in support)):
        raise ValueError("support must contain distinct four-bit worlds")
    belief = sum(1 << w for w in support)
    return allowed_bits, costs, support, belief


def _terminal(goal_mask, belief):
    if not belief:
        raise ValueError("cannot optimize a terminal guess on empty support")
    positives = (goal_mask & belief).bit_count()
    negatives = belief.bit_count() - positives
    guess = int(positives > negatives)  # Ties deterministically favor zero.
    return min(positives, negatives), {"kind": "guess", "value": guess}


def _finish(error_count, policy, support, started, counts):
    return {
        "error": error_count / len(support),
        "error_count": error_count,
        "support_size": len(support),
        "policy": policy,
        "counts": counts,
        "runtime_seconds": time.perf_counter() - started,
    }


def solve_adaptive(goal_mask, budget, allowed_bits=ALL_BITS,
                   costs=(1, 1, 1, 1), *, support=ALL_WORLDS):
    """Exact adaptive decision-tree DP, minimizing terminal error.

    The budget is a worst-case sum of query costs on a path.  Stopping wins
    ties against querying; query ties follow allowed_bits order.  Queries
    whose response is already known in a belief are omitted.
    """
    started = time.perf_counter()
    allowed_bits, costs, support, initial = _validate(
        goal_mask, budget, allowed_bits, costs, support)
    counts = {"states": 0, "query_candidates": 0, "terminal_evaluations": 0}

    @lru_cache(maxsize=None)
    def visit(belief, remaining_budget, remaining_bits):
        counts["states"] += 1
        counts["terminal_evaluations"] += 1
        best_error, best_policy = _terminal(goal_mask, belief)
        for bit in remaining_bits:
            if costs[bit] > remaining_budget:
                continue
            one = belief & BIT_WORLDS[bit]
            zero = belief & ~BIT_WORLDS[bit]
            if not one or not zero:
                continue
            counts["query_candidates"] += 1
            next_bits = tuple(b for b in remaining_bits if b != bit)
            zero_error, zero_policy = visit(zero, remaining_budget - costs[bit], next_bits)
            one_error, one_policy = visit(one, remaining_budget - costs[bit], next_bits)
            error = zero_error + one_error
            if error < best_error:
                best_error = error
                best_policy = {"kind": "query", "bit": bit, "cost": costs[bit],
                               "zero": zero_policy, "one": one_policy}
        return best_error, best_policy

    error, policy = visit(initial, budget, allowed_bits)
    counts["cache_hits"] = visit.cache_info().hits
    return _finish(error, policy, support, started, counts)


def _fixed_tree(goal_mask, belief, bits, costs, counts):
    counts["tree_nodes"] += 1
    if not belief:
        # This node is unreachable from the declared public support.  Replay
        # must not turn this support failure into a successful certificate.
        return 0, {"kind": "unsupported"}
    if not bits:
        counts["terminal_evaluations"] += 1
        return _terminal(goal_mask, belief)
    bit, rest = bits[0], bits[1:]
    zero_error, zero_policy = _fixed_tree(
        goal_mask, belief & ~BIT_WORLDS[bit], rest, costs, counts)
    one_error, one_policy = _fixed_tree(
        goal_mask, belief & BIT_WORLDS[bit], rest, costs, counts)
    return zero_error + one_error, {
        "kind": "query", "bit": bit, "cost": costs[bit],
        "zero": zero_policy, "one": one_policy,
    }


def solve_open_loop(goal_mask, budget, allowed_bits=ALL_BITS,
                    costs=(1, 1, 1, 1), *, support=ALL_WORLDS):
    """Exact fixed query subset, then response-dependent majority guess.

    Every feasible subset is evaluated.  Ties favor fewer queries, then the
    combination order induced by allowed_bits.  All chosen sensors are
    queried on every supported path, even when a response makes the goal clear.
    """
    started = time.perf_counter()
    allowed_bits, costs, support, initial = _validate(
        goal_mask, budget, allowed_bits, costs, support)
    counts = {"subsets_evaluated": 0, "tree_nodes": 0, "terminal_evaluations": 0}
    best_error, best_policy, best_bits = len(support) + 1, None, ()
    for size in range(len(allowed_bits) + 1):
        for bits in itertools.combinations(allowed_bits, size):
            if sum(costs[b] for b in bits) > budget:
                continue
            counts["subsets_evaluated"] += 1
            error, policy = _fixed_tree(goal_mask, initial, bits, costs, counts)
            if error < best_error:
                best_error, best_policy, best_bits = error, policy, bits
    result = _finish(best_error, best_policy, support, started, counts)
    result["query_subset"] = list(best_bits)
    return result


def solve_fixed_prefix(goal_mask, budget, allowed_bits=ALL_BITS,
                       costs=(1, 1, 1, 1), *, support=ALL_WORLDS):
    """Query the longest affordable literal prefix, then majority guess.

    At nonunit costs, stop before the first unaffordable next sensor; do not
    skip it.  The allowed_bits order therefore defines this baseline.
    """
    started = time.perf_counter()
    allowed_bits, costs, support, initial = _validate(
        goal_mask, budget, allowed_bits, costs, support)
    bits, spent = [], 0
    for bit in allowed_bits:
        if spent + costs[bit] > budget:
            break
        bits.append(bit)
        spent += costs[bit]
    counts = {"subsets_evaluated": 1, "tree_nodes": 0, "terminal_evaluations": 0}
    error, policy = _fixed_tree(goal_mask, initial, tuple(bits), costs, counts)
    result = _finish(error, policy, support, started, counts)
    result["query_subset"] = bits
    return result


def replay_with_sensor(policy, read_bit):
    """Execute using only read_bit(bit); return guess and observed query trace.

    The callable boundary allows independent audit of exactly what is sensed.
    A deployment controller need not receive an integer encoding its world.
    """
    trace = []
    while policy["kind"] == "query":
        bit = policy["bit"]
        response = read_bit(bit)
        if type(response) is not int or response not in (0, 1):
            raise ValueError("sensor response must be integer zero or one")
        trace.append({"bit": bit, "observation": response, "cost": policy["cost"]})
        policy = policy["one" if response else "zero"]
    if policy["kind"] == "unsupported":
        raise ValueError("observations left the declared support")
    if policy["kind"] != "guess" or policy["value"] not in (0, 1):
        raise ValueError("invalid terminal policy")
    return policy["value"], trace


def replay_trace(policy, world):
    """Replay on one evaluation world; access it only through selected bits."""
    if type(world) is not int or world not in ALL_WORLDS:
        raise ValueError("world must be an integer in [0, 15]")
    return replay_with_sensor(policy, lambda bit: (world >> bit) & 1)


def replay(policy, world):
    """Return terminal guess; policies never read the goal or unqueried bits."""
    return replay_trace(policy, world)[0]


def truth_mask(function):
    return sum(int(bool(function(w))) << w for w in ALL_WORLDS)


def goal_catalog():
    """Protocol-frozen families; arbitrary draws retain duplicate entries."""
    goals, mux_seen = [], set()
    for selector in ALL_BITS:
        for low, high in itertools.permutations([b for b in ALL_BITS if b != selector], 2):
            for complement in (0, 1):
                mask = truth_mask(lambda w: (((w >> (high if (w >> selector) & 1 else low)) & 1)
                                             ^ complement))
                if mask in mux_seen:
                    continue
                mux_seen.add(mask)
                goals.append({"goal_id": f"mux_{len(mux_seen)-1:02d}", "family": "multiplexer",
                              "goal_mask": mask, "selector": selector, "low": low,
                              "high": high, "complement": complement})
    for subset in range(1, 16):
        for complement in (0, 1):
            mask = truth_mask(lambda w: ((w & subset).bit_count() % 2) ^ complement)
            goals.append({"goal_id": f"affine_{subset:02d}_{complement}", "family": "affine",
                          "goal_mask": mask, "subset": subset, "complement": complement})
    rng = random.Random(ARBITRARY_SEED)
    for draw in range(32):
        goals.append({"goal_id": f"arbitrary_{draw:02d}", "family": "arbitrary",
                      "goal_mask": rng.randrange(1 << 16), "draw": draw,
                      "seed": ARBITRARY_SEED})
    return goals


def informative_bits(goal_mask):
    return tuple(bit for bit in ALL_BITS if any(
        ((goal_mask >> w) & 1) != ((goal_mask >> (w ^ (1 << bit))) & 1)
        for w in ALL_WORLDS))


SOLVERS = {"adaptive": solve_adaptive, "open_loop": solve_open_loop,
           "fixed_prefix": solve_fixed_prefix}


def run_experiment(output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    goals = goal_catalog()
    (output / "planning_goals.json").write_text(json.dumps(goals, indent=2) + "\n")
    started, rows, solve_seconds, total_counts = time.perf_counter(), [], Counter(), Counter()
    trace_count = 0
    row_fields = ["case", "goal_id", "family", "goal_mask", "method", "budget",
                  "allowed_bits", "costs", "removed_bit", "error", "error_count",
                  "mean_query_cost", "max_query_cost", "mean_query_count",
                  "runtime_seconds", "counts"]
    trace_fields = ["case", "goal_id", "method", "budget", "world", "truth", "guess",
                    "error", "queried_bits", "observations", "total_cost"]
    with (output / "planning_rows.csv").open("w", newline="") as row_file, \
         (output / "planning_traces.csv").open("w", newline="") as trace_file, \
         (output / "planning_policies.jsonl").open("w") as policy_file:
        row_writer = csv.DictWriter(row_file, fieldnames=row_fields)
        trace_writer = csv.DictWriter(trace_file, fieldnames=trace_fields)
        row_writer.writeheader()
        trace_writer.writeheader()
        for goal in goals:
            mask = goal["goal_mask"]
            informative = informative_bits(mask)
            removed = informative[0] if informative else None
            cases = [("primary", ALL_BITS, (1, 1, 1, 1), range(5), None),
                     ("nonunit_costs", ALL_BITS, (1, 2, 3, 4), range(5), None)]
            if removed is not None:
                cases.append(("missing_sensor", tuple(b for b in ALL_BITS if b != removed),
                              (1, 1, 1, 1), (4,), removed))
            for case, allowed_bits, costs, budgets, removed_bit in cases:
                for budget in budgets:
                    errors = {}
                    for method, solver in SOLVERS.items():
                        result = solver(mask, budget, allowed_bits, costs)
                        errors[method] = result["error_count"]
                        costs_seen, counts_seen, error_count = [], [], 0
                        for world in ALL_WORLDS:
                            guess, trace = replay_trace(result["policy"], world)
                            truth = (mask >> world) & 1
                            error_count += int(guess != truth)
                            spent = sum(t["cost"] for t in trace)
                            assert spent <= budget
                            assert all(t["bit"] in allowed_bits for t in trace)
                            costs_seen.append(spent)
                            counts_seen.append(len(trace))
                            trace_writer.writerow({"case": case, "goal_id": goal["goal_id"],
                                "method": method, "budget": budget, "world": world,
                                "truth": truth, "guess": guess, "error": int(guess != truth),
                                "queried_bits": "|".join(str(t["bit"]) for t in trace),
                                "observations": "|".join(str(t["observation"]) for t in trace),
                                "total_cost": spent})
                            trace_count += 1
                        assert error_count == result["error_count"]
                        row = {"case": case, "goal_id": goal["goal_id"], "family": goal["family"],
                            "goal_mask": mask, "method": method, "budget": budget,
                            "allowed_bits": "|".join(map(str, allowed_bits)),
                            "costs": "|".join(map(str, costs)), "removed_bit": removed_bit,
                            "error": result["error"], "error_count": error_count,
                            "mean_query_cost": sum(costs_seen) / 16, "max_query_cost": max(costs_seen),
                            "mean_query_count": sum(counts_seen) / 16,
                            "runtime_seconds": result["runtime_seconds"],
                            "counts": json.dumps(result["counts"], sort_keys=True)}
                        rows.append(row)
                        row_writer.writerow(row)
                        policy_file.write(json.dumps({"case": case, "goal_id": goal["goal_id"],
                            "method": method, "budget": budget, "policy": result["policy"]}) + "\n")
                        solve_seconds[method] += result["runtime_seconds"]
                        for key, value in result["counts"].items():
                            total_counts[f"{method}.{key}"] += value
                    assert errors["adaptive"] <= errors["open_loop"] <= errors["fixed_prefix"]
    grouped = {}
    for row in rows:
        key = (row["case"], row["family"], row["budget"], row["method"])
        grouped.setdefault(key, []).append(row["error"])
    summary = {
        "world_count": 16, "prior": "uniform over all 16 worlds",
        "arbitrary_seed": ARBITRARY_SEED, "goal_count": len(goals),
        "families": {family: {"draws": sum(g["family"] == family for g in goals),
            "unique_masks": len({g["goal_mask"] for g in goals if g["family"] == family})}
            for family in sorted({g["family"] for g in goals})},
        "row_count": len(rows), "trace_count": trace_count,
        "solver_seconds": dict(solve_seconds), "total_counts": dict(total_counts),
        "elapsed_seconds": time.perf_counter() - started,
        "group_means": [{"case": key[0], "family": key[1], "budget": key[2],
                         "method": key[3], "mean_error": sum(values) / len(values),
                         "n": len(values)} for key, values in sorted(grouped.items())],
    }
    (output / "planning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    summary = run_experiment(args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != "group_means"}, indent=2))
