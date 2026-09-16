"""Compose fitted Boolean goal programs with exact sensing planners.

True goal labels occur only in evaluator code and explicitly named oracle calls.
No selected action is scored against truth until its entire policy is constructed.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import planning

ROOT = Path(__file__).resolve().parent
PLANNERS = {
    "adaptive": planning.solve_adaptive,
    "batch": planning.solve_open_loop,
}


@lru_cache(maxsize=None)
def compile_policy(model_mask: int, budget: int, planner_name: str) -> dict:
    """Only a fitted model or explicitly selected oracle enters this interface."""
    solved = PLANNERS[planner_name](model_mask, budget)
    pred_mask = sum(planning.replay(solved["policy"], x) << x for x in range(16))
    model_error = (pred_mask ^ model_mask).bit_count() / 16
    if abs(model_error - solved["error"]) > 1e-12:
        raise AssertionError("Planner-reported risk differs from independent replay")
    return {"policy": solved["policy"], "action_mask": pred_mask,
            "model_error": model_error}


def risk(action_mask: int, target_mask: int, region_mask: int = 65535) -> float:
    return ((action_mask ^ target_mask) & region_mask).bit_count() / region_mask.bit_count()


def evaluate_one(row: dict, budget: int, planner_name: str) -> tuple[dict, dict]:
    model_mask = int(row["pred_mask"])
    target_mask = int(row["target_mask"])
    deployed = compile_policy(model_mask, budget, planner_name)
    # This explicit true-goal oracle is never fed back into the deployed compiler.
    oracle = compile_policy(target_mask, budget, planner_name)
    actual_error = risk(deployed["action_mask"], target_mask)
    oracle_error = risk(oracle["action_mask"], target_mask)
    if actual_error + 1e-12 < oracle_error:
        raise AssertionError("Candidate beat exhaustive same-class true-goal oracle")
    if budget == 4 and deployed["action_mask"] != model_mask:
        raise AssertionError("Full sensing budget did not realize fitted function")
    result = {k: row[k] for k in ("family", "target_id", "target_mask", "regime", "m", "rep", "method", "pred_mask")}
    result.update({"budget": budget, "planner": planner_name,
                   "actual_error": actual_error,
                   "predicted_error": deployed["model_error"],
                   "oracle_error": oracle_error,
                   "excess_error": actual_error - oracle_error,
                   "high_bit_error": risk(deployed["action_mask"], target_mask, 65280),
                   "program_error": risk(model_mask, target_mask),
                   "action_mask": deployed["action_mask"]})
    example = {"row": result, "fitted_expression": row.get("expression", ""),
               "deployed_policy": deployed["policy"], "oracle_policy": oracle["policy"]}
    return result, example


def run(input_path: Path, output_dir: Path, expected_rows: int = 82944) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates = defaultdict(lambda: defaultdict(float))
    worst_example = None
    worst_reversal = None
    reversal_counts = {"adaptive_worse": 0, "adaptive_better": 0, "equal": 0}
    count = 0
    with input_path.open(newline="") as source, (output_dir / "composition_rows.csv").open("w", newline="") as dest:
        writer = None
        for row in csv.DictReader(source):
            if int(row["m"]) not in (8, 32):
                continue
            for budget in (2, 3, 4):
                pair = {}
                for name in PLANNERS:
                    result, example = evaluate_one(row, budget, name)
                    if writer is None:
                        writer = csv.DictWriter(dest, fieldnames=list(result))
                        writer.writeheader()
                    writer.writerow(result)
                    count += 1
                    key = tuple(result[k] for k in ("family", "regime", "m", "method", "budget", "planner"))
                    agg = aggregates[key]
                    agg["n"] += 1
                    for field in ("actual_error", "predicted_error", "oracle_error", "excess_error", "high_bit_error", "program_error"):
                        agg[field] += result[field]
                    if worst_example is None or result["excess_error"] > worst_example["row"]["excess_error"]:
                        worst_example = example
                    pair[name] = example
                diff = pair["adaptive"]["row"]["actual_error"] - pair["batch"]["row"]["actual_error"]
                reversal_counts["adaptive_worse" if diff > 0 else "adaptive_better" if diff < 0 else "equal"] += 1
                if diff > 0 and (worst_reversal is None or diff > worst_reversal["difference"]):
                    worst_reversal = {"difference": diff, **pair}
    if count != expected_rows:
        raise AssertionError(f"Expected {expected_rows} primary rows, got {count}; no success receipt emitted")
    summaries = []
    for key, agg in sorted(aggregates.items()):
        item = dict(zip(("family", "regime", "m", "method", "budget", "planner"), key))
        item["n"] = int(agg["n"])
        item.update({k: v / agg["n"] for k, v in agg.items() if k != "n"})
        summaries.append(item)
    summary = {"rows": count, "runtime_seconds": time.perf_counter() - started,
               "policy_cache": compile_policy.cache_info()._asdict(),
               "distribution": "uniform over all 16 worlds for both training regimes",
               "reversal_counts": reversal_counts,
               "aggregates": summaries,
               "invariants": {"independent_replay_matches_model_risk": True,
                              "actual_risk_ge_matching_oracle": True,
                              "full_budget_realizes_fitted_function": True},
               "inference_scope": "finite correlated suite, no rowwise iid significance claims"}
    (output_dir / "composition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "composition_examples.json").write_text(json.dumps({"largest_oracle_gap": worst_example,
                                                                      "largest_adaptive_reversal": worst_reversal}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "learning_rows.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "aggregates"}, indent=2))
