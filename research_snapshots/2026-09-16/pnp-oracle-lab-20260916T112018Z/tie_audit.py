"""POST HOC, EVALUATION-ONLY oracle envelope of model-optimal sensing policies.

True labels select among policies only in this diagnostic, never in the original
deployed planner.  The lexicographic primary objective is learned-model error;
secondary true error is minimized or maximized over all primary optima.
"""

import argparse
import csv
import json
import time
from functools import lru_cache
from pathlib import Path

import planning


def _lexicographic_solve(model_mask, true_mask, budget, worlds, bits, direction):
    initial = sum(1 << w for w in worlds)
    count = 0

    @lru_cache(None)
    def visit(belief, remaining, available):
        nonlocal count
        count += 1
        size = belief.bit_count()
        model_ones = (belief & model_mask).bit_count()
        true_ones = (belief & true_mask).bit_count()
        best_key, best_true, best_policy = None, None, None
        # Both terminal actions remain available even when model-risk ties.
        for guess in (0, 1):
            model_errors = model_ones if guess == 0 else size - model_ones
            true_errors = true_ones if guess == 0 else size - true_ones
            key = (model_errors, direction * true_errors)
            if best_key is None or key < best_key:
                best_key, best_true = key, true_errors
                best_policy = {"kind": "guess", "value": guess}
        if remaining:
            for bit in available:
                one = belief & planning.BIT_WORLDS[bit]
                zero = belief & ~planning.BIT_WORLDS[bit]
                if not one or not zero:
                    continue
                rest = tuple(b for b in available if b != bit)
                left = visit(zero, remaining - 1, rest)
                right = visit(one, remaining - 1, rest)
                model_errors, true_errors = left[0] + right[0], left[1] + right[1]
                key = (model_errors, direction * true_errors)
                if key < best_key:
                    best_key, best_true = key, true_errors
                    best_policy = {"kind": "query", "bit": bit, "cost": 1,
                                   "zero": left[2], "one": right[2]}
        return best_key[0], best_true, best_policy

    result = visit(initial, budget, bits)
    return result, count


def tie_envelope(model_mask, true_mask, budget, *, worlds=tuple(range(16)), bits=(0, 1, 2, 3)):
    """Evaluation-only extrema of true error among model-optimal policies.

    Public API uses full four-bit worlds and unit-cost queries by default.
    Smaller supports/sensor sets exist solely for exhaustive unit tests.
    """
    started = time.perf_counter()
    worlds, bits = tuple(worlds), tuple(bits)
    if not worlds or len(set(worlds)) != len(worlds):
        raise ValueError("world support must be nonempty and unique")
    if any(type(w) is not int or not 0 <= w < 16 for w in worlds):
        raise ValueError("invalid four-bit world")
    if len(set(bits)) != len(bits) or any(type(b) is not int or b not in range(4) for b in bits):
        raise ValueError("invalid sensor set")
    if type(budget) is not int or budget < 0:
        raise ValueError("invalid budget")
    if any(type(mask) is not int or not 0 <= mask < 65536 for mask in (model_mask, true_mask)):
        raise ValueError("invalid goal mask")
    minimum, min_states = _lexicographic_solve(model_mask, true_mask, budget, worlds, bits, 1)
    maximum, max_states = _lexicographic_solve(model_mask, true_mask, budget, worlds, bits, -1)
    assert minimum[0] == maximum[0]
    assert minimum[1] <= maximum[1]
    for solution in (minimum, maximum):
        actual, model = 0, 0
        for world in worlds:
            guess, trace = planning.replay_trace(solution[2], world)
            actual += guess != ((true_mask >> world) & 1)
            model += guess != ((model_mask >> world) & 1)
            assert len(trace) <= budget
        assert (model, actual) == solution[:2]
    return {
        "model_error_count": minimum[0], "model_error": minimum[0] / len(worlds),
        "min_true_error_count": minimum[1], "min_true_error": minimum[1] / len(worlds),
        "max_true_error_count": maximum[1], "max_true_error": maximum[1] / len(worlds),
        "min_policy": minimum[2], "max_policy": maximum[2],
        "states": min_states + max_states,
        "runtime_seconds": time.perf_counter() - started,
    }


PAIR_FIELDS = ("family", "target_id", "target_mask", "regime", "m", "rep", "method", "pred_mask", "budget")


def run_audit(source, output_directory):
    started = time.perf_counter()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    pairs = {}
    with Path(source).open() as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[field] for field in PAIR_FIELDS)
            current = pairs.setdefault(key, {})
            if row["planner"] in current:
                raise ValueError("duplicate composition pairing")
            current[row["planner"]] = row
    reversals = []
    for key, pair in pairs.items():
        if set(pair) != {"adaptive", "batch"}:
            raise ValueError("composition pair missing adaptive/batch result")
        adaptive, batch = pair["adaptive"], pair["batch"]
        if float(adaptive["actual_error"]) > float(batch["actual_error"]):
            reversals.append((key, adaptive, batch))
    results, diagnostic_rows, policies = {}, [], []
    for key, adaptive, batch in reversals:
        model_mask, true_mask, budget = int(adaptive["pred_mask"]), int(adaptive["target_mask"]), int(adaptive["budget"])
        unique = (model_mask, true_mask, budget)
        if unique not in results:
            result = tie_envelope(model_mask, true_mask, budget)
            original = planning.solve_adaptive(model_mask, budget)
            assert result["model_error"] == original["error"]
            result["multiplicity"] = 0
            results[unique] = result
            policies.append({"pred_mask": model_mask, "target_mask": true_mask, "budget": budget,
                             "min_policy": result["min_policy"], "max_policy": result["max_policy"]})
        result = results[unique]
        result["multiplicity"] += 1
        adaptive_error, batch_error = float(adaptive["actual_error"]), float(batch["actual_error"])
        assert result["min_true_error"] <= adaptive_error <= result["max_true_error"]
        # These inequalities follow only because the retained C reversals are
        # tied in the model objective.  A fixed policy is feasible adaptively.
        model_tied = float(adaptive["predicted_error"]) == float(batch["predicted_error"])
        if model_tied:
            assert result["min_true_error"] <= batch_error <= result["max_true_error"]
        for row in (adaptive, batch):
            actions = int(row["action_mask"])
            assert (actions ^ true_mask).bit_count() / 16 == float(row["actual_error"])
            assert (actions ^ model_mask).bit_count() / 16 == float(row["predicted_error"])
        diagnostic_rows.append({**dict(zip(PAIR_FIELDS, key)),
            "canonical_adaptive_error": adaptive_error, "canonical_batch_error": batch_error,
            "model_error": result["model_error"], "model_objective_tied": int(model_tied),
            "min_true_error": result["min_true_error"], "max_true_error": result["max_true_error"],
            "better_or_equal_batch_optimum_exists": int(result["min_true_error"] <= batch_error)})
    with (output / "tie_audit_pairs.csv").open("w", newline="") as stream:
        fields = list(PAIR_FIELDS) + ["canonical_adaptive_error", "canonical_batch_error", "model_error",
            "model_objective_tied", "min_true_error", "max_true_error", "better_or_equal_batch_optimum_exists"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    with (output / "tie_audit_unique.csv").open("w", newline="") as stream:
        fields = ["pred_mask", "target_mask", "budget", "multiplicity", "model_error",
                  "min_true_error", "max_true_error", "states", "runtime_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (model, truth, budget), result in sorted(results.items()):
            writer.writerow({"pred_mask": model, "target_mask": truth, "budget": budget,
                             **{field: result[field] for field in fields[3:]}})
    (output / "tie_audit_policies.json").write_text(json.dumps(policies, indent=2) + "\n")
    n = len(diagnostic_rows)
    summary = {
        "posthoc": True, "evaluation_only_true_label_tie_breaking": True,
        "source": str(source), "composition_pairs": len(pairs), "reversal_pairs": n,
        "unique_model_target_budget_cases": len(results),
        "model_tied_reversal_pairs": sum(row["model_objective_tied"] for row in diagnostic_rows),
        "strict_model_improvement_reversal_pairs": sum(
            float(a["predicted_error"]) < float(b["predicted_error"]) for _, a, b in reversals),
        "batch_or_better_model_optimum_exists_pairs": sum(
            row["better_or_equal_batch_optimum_exists"] for row in diagnostic_rows),
        "pair_weighted_means": {field: sum(row[field] for row in diagnostic_rows) / n if n else None
            for field in ("canonical_adaptive_error", "canonical_batch_error", "min_true_error", "max_true_error")},
        "states": sum(result["states"] for result in results.values()),
        "solver_seconds": sum(result["runtime_seconds"] for result in results.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "Observed canonical reversals do not establish an intrinsic harm from adaptivity; "
                          "true labels are used only to characterize ambiguity among model-optimal policies.",
    }
    (output / "tie_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/composition_rows.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    print(json.dumps(run_audit(args.source, args.output), indent=2))
