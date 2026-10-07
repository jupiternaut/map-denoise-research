"""Verify sealed decisions, then independently load synthetic truth."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    seal = json.loads((ROOT / "decision_seal.json").read_text())
    for name, expected in seal["sealed_files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert seal["truth_files_read"] == []
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    obs = {s["id"]: s for s in json.loads((ROOT / "observations.json").read_text())["scenes"]}
    decisions = {s["observation_id"]: s for s in json.loads((ROOT / "decisions.json").read_text())["scenes"]}
    # Truth is first accessed only after the completed selector's hashes pass.
    worlds = json.loads((ROOT / "evaluation_truth.json").read_text())["worlds"]
    rows, neighbor_rows, condition_rows = [], [], []
    for world in worlds:
        scene, result = obs[world["observation_id"]], decisions[world["observation_id"]]
        target, bad, coverage, finite_widths = world["target_depth"], 0, 0, []
        for actual, measured, support in zip(world["true_neighbors"], scene["neighbors"], result["neighbor_supports"]):
            assert actual["id"] == measured["id"]
            error_n = math.dist(actual["normal"], measured["normal"])
            error_d = abs(actual["offset"] - measured["offset"])
            bounded = error_n <= measured["eps_normal"] + 1e-12 and error_d <= measured["eps_offset"] + 1e-12
            same_layer = abs(actual["layer_depth"] - target) < 1e-12
            good = bounded and same_layer
            bad += not good
            covered = not support["finite"] or abs(support["depth"] - target) <= support["radius"] + 1e-12
            coverage += covered
            if support["finite"]:
                finite_widths.append(2 * support["radius"])
            if good:
                assert covered, (world["world_id"], actual["id"])
            neighbor_rows.append({
                "world_id": world["world_id"], "neighbor_id": actual["id"], "mechanism": scene["mechanism"],
                "normal_error": error_n, "offset_error_m": error_d, "error_budget_holds": bounded,
                "same_target_layer": same_layer, "good_neighbor": good, "target_covered": covered,
                "finite_interval": support["finite"], "projected_depth_m": support["depth"],
                "radius_m": support["radius"],
            })
        truth_in_pool = any(abs(c["depth"] - target) < 1e-12 for c in scene["candidates"])
        conditional_assumptions = truth_in_pool and bad <= scene["declared_max_bad_neighbors"]
        condition_rows.append({
            "world_id": world["world_id"], "mechanism": scene["mechanism"], "observation_id": scene["id"],
            "target_depth_m": target, "actual_bad_neighbors": bad, "declared_bad_bound": scene["declared_max_bad_neighbors"],
            "theorem_assumptions_hold": conditional_assumptions, "true_depth_covered_count": coverage,
            "finite_interval_count": len(finite_widths), "max_finite_width_m": max(finite_widths, default=None),
            "qualified_candidate_ids": ";".join(result["qualified_candidates"]),
            "parameters": json.dumps(scene["parameters"], sort_keys=True),
        })
        for decision in result["decisions"]:
            error = abs(decision["selected_depth"] - target)
            accepted = not decision["abstained"]
            if decision["method"] == "unique_threshold" and accepted and conditional_assumptions:
                assert error < 1e-12, (world["world_id"], decision)
            rows.append({
                "world_id": world["world_id"], "observation_id": scene["id"], "mechanism": scene["mechanism"],
                "method": decision["method"], "target_depth_m": target, "selected_id": decision["selected_id"],
                "selected_depth_m": decision["selected_depth"], "error_mm": error * 1000,
                "selected_true_candidate": error < 1e-12, "abstained": decision["abstained"],
                "theorem_assumptions_hold": conditional_assumptions, "actual_bad_neighbors": bad,
            })
    aggregate = []
    for mechanism in sorted({r["mechanism"] for r in rows}):
        for method in protocol["methods"]:
            part = [r for r in rows if r["mechanism"] == mechanism and r["method"] == method]
            aggregate.append({"mechanism": mechanism, "method": method, "worlds": len(part),
                              "correct": sum(r["selected_true_candidate"] for r in part),
                              "abstained": sum(r["abstained"] for r in part),
                              "mean_absolute_error_mm": sum(r["error_mm"] for r in part) / len(part),
                              "accepted_wrong": sum(not r["selected_true_candidate"] and not r["abstained"] for r in part)})
    csv_write(ROOT / "results.csv", rows)
    csv_write(ROOT / "neighbor_diagnostics.csv", neighbor_rows)
    csv_write(ROOT / "conditions.csv", condition_rows)
    csv_write(ROOT / "aggregate.csv", aggregate)
    summary = {
        "seed": protocol["seed"], "scene_count": len(obs), "world_count": len(worlds), "decision_rows": len(rows),
        "all_good_neighbor_intervals_cover_target": all(r["target_covered"] for r in neighbor_rows if r["good_neighbor"]),
        "conditional_accepted_wrong_count": sum(not r["selected_true_candidate"] and not r["abstained"] for r in rows if r["method"] == "unique_threshold" and r["theorem_assumptions_hold"]),
        "aggregate": aggregate,
        "same_observation_two_truths": {
            "observation_id": "common_shift_shared", "truth_depths_m": [2.0, 2.04],
            "separation_mm": 40, "lower_bound_worst_case_absolute_error_mm_any_single_estimate": 20,
            "candidate_pool_excludes_midpoint": True,
            "lower_bound_worst_case_error_mm_for_this_fixed_candidate_pool": 40,
            "meaning": "A truth-blind function receives precisely one shared observation object; it cannot choose distinct answers in the two worlds.",
        },
        "conditions": condition_rows,
        "input_decision_seal_verified": True,
        "limits": ["Synthetic constructed examples; frequencies are not real-world rates.", "The error budgets are known environment assumptions, not estimated real-data calibration.", "Unique-threshold correctness assumes the exact true depth is present in the fixed pool.", "Abstention retains a potentially wrong incumbent and is not counted as a repair."],
    }
    dump(ROOT / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("scene_count", "world_count", "decision_rows", "all_good_neighbor_intervals_cover_target", "conditional_accepted_wrong_count", "input_decision_seal_verified")}))


if __name__ == "__main__":
    main()
