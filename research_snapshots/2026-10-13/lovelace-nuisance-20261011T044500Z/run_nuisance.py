"""Registered CPU-only unknown-appearance and occlusion-contract experiment."""
import os
import sys

sys.dont_write_bytecode = True
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import random
import signal
import time

ROOT = Path(__file__).resolve().parent
HYBRID_ROOT = ROOT.parent / "lovelace-hybrid-20261010T214217Z"
DEPENDENCY_ROOT = ROOT.parent / "lovelace-challenge-20261010T205013Z"
sys.path.insert(0, str(HYBRID_ROOT))
from hybrid import certificate_hybrid
from run_challenge import write_json, finish_record_before_alarm, save_progress, deadline_reached
from continuous_world.interval_solver import solve_outer, encode, decode
from nuisance_model import ColoredModel

# Frozen dependencies adjust sys.path for their own modules. Restore this run's
# root so the same-named verification package resolves to the new experiment.
sys.path.insert(0, str(ROOT))

CASES = (
    ("uniform_isolated", "none", (Q(1, 5), Q(1, 5), Q(2, 5))),
    ("textured_isolated", "none", (Q(1, 5), Q(3, 5), Q(2, 5))),
    ("textured_partial", "partial", (Q(1, 5), Q(3, 5), Q(2, 5))),
    ("equal_partial", "partial", (Q(1, 5), Q(1, 5), Q(1, 5))),
    ("weak_partial", "partial", (Q(3, 10), Q(3, 10), Q(31, 100))),
    ("equal_full_negative", "full", (Q(1, 5), Q(1, 5), Q(1, 5))),
)
TRUTHS = (Q(6199, 11), Q(7853, 13), Q(8999, 14))
SEEDS = (51001, 51002, 51003)
OFFSETS = (Q(0), Q(-1, 4), Q(1, 4), Q(-3, 4), Q(3, 4), Q(-2), Q(2), Q(-3), Q(3))
WEIGHTS = (Q(-1), Q(-1, 2), Q(0), Q(1, 2), Q(1))
RADIUS, EPSILON, TOLERANCE, MAX_BOXES = Q(1, 5), Q(1, 100), Q(15, 32), 255
METHODS = ("keep", "frozen_nominal", "expanded_unanchored", "calibrated_isolated", "calibrated_union")
BRANCHES = ("none", "partial", "full")
SAMPLE_SOURCES = ("front-left", "front-right", "rear")
RANDOM_TAPE_PREFIX = "nuisance-v1"
WALL_TIME_SECONDS = 1800
SOURCE_COMMIT = "bd8d05aad65e2c55cd9cd288beae4ef949606984"


def input_sources():
    files = set(ROOT.glob("*.py"))
    files.add(ROOT / "PROTOCOL.md")
    if (ROOT / "THEORY.md").exists():
        files.add(ROOT / "THEORY.md")
    for root in (ROOT, HYBRID_ROOT, DEPENDENCY_ROOT):
        files.update(root.glob("*.py"))
        for suffix in ("*.py", "*.json"):
            files.update((root / "verification").rglob(suffix))
        if root != ROOT:
            for suffix in ("*.py", "*.json"):
                files.update((root / "source_snapshot").rglob(suffix))
            files.add(root / "PROTOCOL.md")
            if (root / "THEORY.md").exists():
                files.add(root / "THEORY.md")
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def configuration():
    n = len(CASES) * len(TRUTHS) * len(SEEDS)
    return {"cases_evaluation_only": [{"case": case, "rear_branch": branch, "colors": list(map(str, colors))}
            for case, branch, colors in CASES], "truths_evaluation_only": list(map(str, TRUTHS)),
            "seeds_evaluation_only": SEEDS, "offsets_evaluation_only": list(map(str, OFFSETS)),
            "camera_radius_mm": str(RADIUS), "epsilon": str(EPSILON), "tolerance_mm": str(TOLERANCE),
            "max_boxes_per_branch": MAX_BOXES, "depth_domain_mm": ["540", "660"],
            "expected_observations": n, "expected_tasks": n * len(OFFSETS),
            "expected_decisions": n * len(OFFSETS) * len(METHODS), "methods": METHODS,
            "allowed_occlusion_branches": BRANCHES, "calibration_sources": SAMPLE_SOURCES,
            "calibration_samples_per_observation": 3,
            "calibration_samples_used_per_method": {m: 3 if m.startswith("calibrated_") else 0 for m in METHODS},
            "random_tape_prefix": RANDOM_TAPE_PREFIX,
            "random_draw_order": "left pose; right pose; each image RGB channel; front-left, front-right, rear calibration noise",
            "random_weights": list(map(str, WEIGHTS)), "cpu_threads": 1, "gpu_count": 0,
            "wall_time_limit_seconds": WALL_TIME_SECONDS,
            "scope": "shared front depth; two continuous piecewise-constant front grays and one rear gray; finite branch competition",
            "information_note": "calibrated arms use the same three additional sourced material observations; other arms do not use their values"}


def validate_output(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT.resolve()) or output == ROOT.resolve():
        raise ValueError("output must be a fresh directory within this new experiment root")
    if output.exists():
        raise FileExistsError("choose a fresh output directory; existing results are never overwritten")
    return output


def generate_observation(case, rear_branch, colors, truth, seed, radius=RADIUS, prefix=RANDOM_TAPE_PREFIX):
    """Generator-only inputs are returned exclusively in a private truth record."""
    tape = f"{prefix}:{case}:{radius}:{truth}:{seed}"
    rng = random.Random(tape)
    offsets = (radius * rng.choice(WEIGHTS), Q(0), radius * rng.choice(WEIGHTS))
    model = ColoredModel(rear_branch, camera_radius=radius, colors=((Q(0), Q(1)),) * 3)
    prediction = model.point_prediction(truth, offsets, colors=colors)
    image_noise = tuple(EPSILON * rng.choice(WEIGHTS) for _ in prediction)
    observation = tuple(value + noise for value, noise in zip(prediction, image_noise))
    calibration_noise = tuple(EPSILON * rng.choice(WEIGHTS) for _ in SAMPLE_SOURCES)
    calibration = [{"source": source, "value": encode(color + noise)}
                   for source, color, noise in zip(SAMPLE_SOURCES, colors, calibration_noise)]
    public = {"radius": str(radius), "observation": list(map(encode, observation)), "calibration": calibration}
    private = {"case": case, "rear_branch": rear_branch, "truth": encode(truth), "colors": list(map(encode, colors)),
               "seed": seed, "random_tape": tape, "actual_offsets": list(map(encode, offsets)),
               "image_noise": list(map(encode, image_noise)), "calibration_noise": list(map(encode, calibration_noise)),
               "noise_linf": encode(max(map(abs, image_noise))), "calibration_noise_linf": encode(max(map(abs, calibration_noise)))}
    return public, private


def calibration_intervals(calibration):
    if len(calibration) != 3 or tuple(item["source"] for item in calibration) != SAMPLE_SOURCES:
        raise ValueError("three material observations with registered source labels are required")
    intervals = tuple((max(Q(0), decode(item["value"]) - EPSILON),
                       min(Q(1), decode(item["value"]) + EPSILON)) for item in calibration)
    if any(lo > hi for lo, hi in intervals):
        raise ValueError("calibration observation contradicts the declared physical range/noise bound")
    return intervals


def arm_specifications(calibration):
    anchored = calibration_intervals(calibration)
    return {"frozen_nominal": (("none",), ((Q(0), Q(0)), (Q(0), Q(0)), (Q(0), Q(1)))),
            "expanded_unanchored": (BRANCHES, ((Q(0), Q(1)),) * 3),
            "calibrated_isolated": (("none",), anchored), "calibrated_union": (BRANCHES, anchored)}


def prior_outer():
    interval = [encode(Q(540)), encode(Q(660))]
    return {"domain": interval, "epsilon": encode(EPSILON), "tolerance": encode(TOLERANCE),
            "max_boxes_per_branch": MAX_BOXES, "evaluated_boxes": 0, "unresolved_budget_leaves": 0,
            "source_leaves": [], "retained_intervals": [interval], "hull": interval,
            "retained_branches": [], "scope": "prior_only_identity_no_image_solver"}


def union_outer(branch_outers):
    """Merge depth leaves of complete branch solves; no pixelwise branch mixing."""
    leaves = []
    for branch, outer in branch_outers.items():
        retained = set(outer["retained_ids"])
        for node in outer["trace"]:
            if node["id"] in retained:
                leaves.append({"branch": branch, "id": node["id"], "lo": node["lo"], "hi": node["hi"]})
    merged = []
    for left, right in sorted((decode(n["lo"]), decode(n["hi"])) for n in leaves):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    intervals = [[encode(lo), encode(hi)] for lo, hi in merged]
    hull = None if not merged else [encode(merged[0][0]), encode(merged[-1][1])]
    return {"domain": [encode(Q(540)), encode(Q(660))], "epsilon": encode(EPSILON), "tolerance": encode(TOLERANCE),
            "max_boxes_per_branch": MAX_BOXES,
            "evaluated_boxes": sum(o["evaluated_boxes"] for o in branch_outers.values()),
            "unresolved_budget_leaves": sum(n["status"] == "unresolved_budget" for o in branch_outers.values() for n in o["trace"]),
            "source_leaves": leaves, "retained_intervals": intervals, "hull": hull,
            "retained_branches": [b for b, o in branch_outers.items() if o["hull"] is not None],
            "scope": "union_of_branch_depth_outer_leaves_then_convex_hull"}


def solve_arm(observation, radius, allowed_branches, color_intervals):
    """Only public image, pose bound, branch set and gray intervals enter."""
    tick = time.perf_counter()
    branches, seconds = {}, {}
    for branch in allowed_branches:
        start = time.perf_counter()
        model = ColoredModel(branch, camera_radius=radius, colors=color_intervals)
        branches[branch] = solve_outer(observation, model.enclosure, epsilon=EPSILON,
                                      tolerance=TOLERANCE, max_boxes=MAX_BOXES)
        seconds[branch] = time.perf_counter() - start
    return {"branches": branches, "outer": union_outer(branches), "branch_seconds": seconds,
            "outer_seconds": time.perf_counter() - tick}


def select_action(outer, incumbent, keep=False):
    """The original hybrid receives only its public outer and own incumbent."""
    if keep:
        return {"status": "KEEP", "output": encode(incumbent), "gain_lower": encode(Q(0)), "reason": "identity"}
    return certificate_hybrid(outer, incumbent)


def contains_truth(outer, truth):
    return any(decode(lo) <= truth <= decode(hi) for lo, hi in outer["retained_intervals"])


def aggregates(rows):
    grouped = defaultdict(list)
    for row in rows:
        scopes = (("overall", "all"), ("by_case", row["case"]),
                  ("by_information", "negative_control" if row["case"] == "equal_full_negative" else "informative"))
        for grouping, condition in scopes:
            for state in ("all", row["state_group"]):
                grouped[(grouping, condition, state, row["method"])].append(row)
    output = []
    for (grouping, condition, state, method), values in sorted(grouped.items()):
        n = len(values)
        errors = [Q(r["ae_mm"]) for r in values]
        initial = [Q(r["initial_ae_mm"]) for r in values]
        widths = [Q(r["hull_width_mm"]) for r in values if r["hull_width_mm"] is not None]
        observations = {r["observation_id"]: r for r in values}
        output.append({"grouping": grouping, "condition": condition, "state_group": state, "method": method,
                       "n": n, "observations": len(observations),
                       "covered_observations": sum(r["covered"] for r in observations.values()),
                       "empty_observations": sum(r["hull_width_mm"] is None for r in observations.values()),
                       "wrongly_excluded_observations": sum(not r["covered"] for r in observations.values()),
                       "improved": sum(a < b for a, b in zip(errors, initial)), "harm": sum(a > b for a, b in zip(errors, initial)),
                       "same": sum(a == b for a, b in zip(errors, initial)),
                       "moved": sum(r["status"] == "MOVE" for r in values),
                       "kept": sum(r["status"] == "KEEP" for r in values),
                       "incompatible": sum(r["status"] == "INCOMPATIBLE" for r in values),
                       "correct_initial_preserved": sum(r["state_group"] == "correct" and Q(r["ae_mm"]) == 0 for r in values),
                       "correct_initial_kept": sum(r["state_group"] == "correct" and r["status"] == "KEEP" for r in values),
                       "mae_mm": float(sum(errors) / n), "mse_mm2": float(sum(a * a for a in errors) / n),
                       "initial_mae_mm": float(sum(initial) / n), "initial_mse_mm2": float(sum(a * a for a in initial) / n),
                       "max_error_mm": float(max(errors)),
                       "mean_hull_width_mm": None if not widths else float(sum(widths) / len(widths)),
                       "evaluated_boxes": sum(r["evaluated_boxes"] for r in observations.values()),
                       "unresolved_budget_leaves": sum(r["unresolved_budget_leaves"] for r in observations.values()),
                       "auxiliary_samples_used": sum(r["auxiliary_samples_used"] for r in observations.values())})
    return output


def evaluate(output, observations, tasks, truths, decisions, outers, timing, generation_seconds, started):
    """Evaluation is the only place that associates actions with case and truth."""
    truth_map = {t["observation_id"]: t for t in truths}
    task_map = {t["task_id"]: t for t in tasks}
    outer_map = {o["observation_id"]: o["arms"] for o in outers}
    rows = []
    for decision in decisions:
        task = task_map[decision["task_id"]]
        oid, method = task["observation_id"], decision["method"]
        private = truth_map[oid]
        truth, incumbent, result = decode(private["truth"]), decode(task["incumbent"]), decode(decision["output"])
        outer = outer_map[oid][method]["outer"]
        error, initial = abs(result - truth), abs(incumbent - truth)
        width = None if outer["hull"] is None else str(decode(outer["hull"][1]) - decode(outer["hull"][0]))
        rows.append({"task_id": task["task_id"], "observation_id": oid, "case": private["case"], "method": method,
                     "state_group": "correct" if initial == 0 else "small", "prior_position": "inside_prior" if 540 <= incumbent <= 660 else "outside_prior",
                     "incumbent": str(incumbent), "output": str(result), "status": decision["status"],
                     "initial_ae_mm": str(initial), "ae_mm": str(error), "se_mm2": str(error * error),
                     "actual_gain_mm2": str(initial * initial - error * error), "covered": contains_truth(outer, truth),
                     "hull_width_mm": width, "retained_branch_count": len(outer["retained_branches"]),
                     "evaluated_boxes": outer["evaluated_boxes"], "unresolved_budget_leaves": outer["unresolved_budget_leaves"],
                     "auxiliary_samples_used": 3 if method.startswith("calibrated_") else 0})
    write_json(output / "evaluation/ROWS.json", rows)
    import csv
    with (output / "evaluation/ROWS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    coverage = []
    for observation in outers:
        private = truth_map[observation["observation_id"]]
        truth = decode(private["truth"])
        for method, entry in observation["arms"].items():
            outer = entry["outer"]
            coverage.append({"observation_id": observation["observation_id"], "case": private["case"], "method": method,
                             "covered": contains_truth(outer, truth), "incompatible": outer["hull"] is None,
                             "hull_width_mm": None if outer["hull"] is None else str(decode(outer["hull"][1]) - decode(outer["hull"][0])),
                             "retained_branches": outer["retained_branches"],
                             "branch_coverage": {b: contains_truth(union_outer({b: o}), truth) for b, o in entry["branches"].items()}})
    write_json(output / "evaluation/COVERAGE.json", coverage)
    with (output / "evaluation/COVERAGE.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows({**r, "retained_branches": json.dumps(r["retained_branches"]),
                         "branch_coverage": json.dumps(r["branch_coverage"])} for r in coverage)
    per_method = {}
    for method in METHODS:
        method_outer = sum(t["outer_seconds_by_method"][method] for t in timing)
        method_action = sum(t["selector_seconds_by_method"][method] for t in timing)
        per_method[method] = {"outer_seconds": method_outer, "selector_seconds": method_action,
                              "total_seconds": method_outer + method_action,
                              "evaluated_boxes": sum(o["arms"][method]["outer"]["evaluated_boxes"] for o in outers),
                              "unresolved_budget_leaves": sum(o["arms"][method]["outer"]["unresolved_budget_leaves"] for o in outers)}
    summary = {"observation_configurations": len(observations), "template_count": len(CASES),
               "task_count": len(tasks), "decision_count": len(decisions), "rows": aggregates(rows),
               "sampling_structure": {"case_templates": len(CASES), "depth_settings": len(TRUTHS),
                   "noise_replicates_per_case_depth": len(SEEDS), "shared_incumbents_per_observation": len(OFFSETS)},
               "sampling_note": "configurations are synthetic case/depth settings with repeated noise seeds; initial states and arms share observations; these counts are not independent real scenes",
               "hybrid_branch_counts": {m: dict(sorted(Counter(d.get("branch", "identity") for d in decisions if d["method"] == m).items())) for m in METHODS},
               "sample_counts": {"image_configurations": len(observations), "image_channels_per_observation": len(observations[0]["observation"]),
                   "auxiliary_samples_per_observation": 3, "auxiliary_samples_total": 3 * len(observations),
                   "auxiliary_samples_used_by_method": {m: 3 * len(observations) if m.startswith("calibrated_") else 0 for m in METHODS},
                   "shared_samples_note": "both calibrated arms reuse the same samples; they are not independent extra observations"},
               "per_method_timing": per_method, "generation_seconds": generation_seconds,
               "elapsed_seconds": time.perf_counter() - started,
               "timing_scope": "observed shared-geometry-cache execution order; includes interval solves and hybrid actions, excludes a standalone cold-cache extrapolation"}
    write_json(output / "evaluation/SUMMARY.json", summary)
    with (output / "evaluation/SUMMARY.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    timing_rows = [{"method": method, **values} for method, values in per_method.items()]
    with (output / "evaluation/TIMING.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)
    return summary


def run(output):
    output = validate_output(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    save_progress(output, [], [], [], started)
    source_lock = input_sources()
    write_json(output / "SOURCE_LOCK.json", {"utc": datetime.now(timezone.utc).isoformat(), "source_commit": SOURCE_COMMIT,
               "new_experiment_root": str(ROOT), "read_only_dependency_roots": [str(HYBRID_ROOT), str(DEPENDENCY_ROOT)],
               "path_format": "absolute", "historical_outputs_used": False, "files": source_lock})
    protocol = configuration()
    write_json(output / "PROTOCOL.json", protocol)
    observations, truths, tasks = [], [], []
    tick = time.perf_counter()
    for case, rear_branch, colors in CASES:
        for truth in TRUTHS:
            for seed in SEEDS:
                oid = f"o{len(observations):03d}"
                public, private = generate_observation(case, rear_branch, colors, truth, seed)
                observations.append({"observation_id": oid, **public})
                truths.append({"observation_id": oid, **private})
                for j, offset in enumerate(OFFSETS):
                    tasks.append({"task_id": f"{oid}-s{j}", "observation_id": oid, "incumbent": encode(truth + offset)})
    generation_seconds = time.perf_counter() - tick
    write_json(output / "observations.json", observations)
    write_json(output / "tasks.json", tasks)
    write_json(output / "evaluation/truth.json", truths)
    tasks_by_observation = defaultdict(list)
    for task in tasks:
        tasks_by_observation[task["observation_id"]].append(task)
    decisions, outers, timing = [], [], []
    for entry in observations:
        oid, radius = entry["observation_id"], Q(entry["radius"])
        image = tuple(map(decode, entry["observation"]))
        specs = arm_specifications(entry["calibration"])
        arm_outers = {"keep": {"branches": {}, "outer": prior_outer(), "branch_seconds": {}, "outer_seconds": 0.0}}
        for method in METHODS[1:]:
            allowed_branches, gray_intervals = specs[method]
            arm_outers[method] = solve_arm(image, radius, allowed_branches, gray_intervals)
        seconds = {m: 0.0 for m in METHODS}
        completed_decisions = []
        for task in tasks_by_observation[oid]:
            incumbent = decode(task["incumbent"])
            for method in METHODS:
                tick = time.perf_counter()
                action = select_action(arm_outers[method]["outer"], incumbent, keep=method == "keep")
                seconds[method] += time.perf_counter() - tick
                completed_decisions.append({"task_id": task["task_id"], "method": method, **action})
        with finish_record_before_alarm():
            decisions.extend(completed_decisions)
            outers.append({"observation_id": oid, "arms": arm_outers})
            timing.append({"observation_id": oid, "outer_seconds_by_method": {m: arm_outers[m]["outer_seconds"] for m in METHODS},
                           "selector_seconds_by_method": seconds})
            save_progress(output, decisions, outers, timing, started)
        print(json.dumps({"stage": "confirmation", "observations_complete": len(outers),
                          "decisions_complete": len(decisions), "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    recorded = json.loads((output / "decisions.json").read_text(encoding="utf-8"))
    if recorded != decisions or len(recorded) != protocol["expected_decisions"]:
        raise RuntimeError("all registered decisions must be written before evaluation")
    write_json(output / "TIMING.json", {"generation_seconds": generation_seconds, "observations": timing,
               "shared_cost_note": "geometry cache is shared across models; each arm records its actually observed solve cost, including all allowed branches"})
    summary = evaluate(output, observations, tasks, truths, recorded, outers, timing, generation_seconds, started)
    if input_sources() != source_lock:
        raise RuntimeError("scientific sources changed during confirmation; outputs cannot be accepted")
    write_json(output / "RUN_STATUS.json", {"status": "COMPLETE", "observations_complete": len(observations),
               "decisions_complete": len(decisions), "elapsed_seconds": time.perf_counter() - started})
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    return summary


def run_with_deadline(output):
    output = validate_output(output)
    old_handler = signal.signal(signal.SIGALRM, deadline_reached)
    signal.alarm(WALL_TIME_SECONDS)
    try:
        return run(output)
    except (Exception, KeyboardInterrupt) as exc:
        if output.exists():
            status_path = output / "RUN_STATUS.json"
            previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            write_json(status_path, {**previous, "status": "INCOMPLETE", "reason": str(exc),
                       "evaluation_claim": "partial checkpoint only; not a completed registered confirmation"})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def smoke():
    """One explicitly nonconfirmation scene; no experiment output directory."""
    started = time.perf_counter()
    truth = Q(600)
    public, _ = generate_observation("smoke_only", "partial", (Q(1, 4), Q(1, 2), Q(3, 4)),
                                    truth, 99001, prefix="nuisance-smoke-v1")
    image = tuple(map(decode, public["observation"]))
    intervals = calibration_intervals(public["calibration"])
    solved = solve_arm(image, RADIUS, BRANCHES, intervals)
    assert contains_truth(solved["outer"], truth)
    for incumbent in (truth, truth - Q(1, 4), truth + Q(3, 4)):
        action = select_action(solved["outer"], incumbent)
        assert abs(decode(action["output"]) - truth) <= abs(incumbent - truth)
        if incumbent == truth:
            assert action["status"] == "KEEP"
    print(json.dumps({"smoke": "passed", "seconds": time.perf_counter() - started,
                      "channels": len(image), "calibration_samples": len(public["calibration"]),
                      "evaluated_boxes": solved["outer"]["evaluated_boxes"],
                      "unresolved_budget_leaves": solved["outer"]["unresolved_budget_leaves"],
                      "retained_branches": solved["outer"]["retained_branches"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the one registered CPU nuisance-contract confirmation in a fresh output directory.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/confirmation-v1",
                        help="fresh output directory inside this new experiment (default: outputs/confirmation-v1)")
    parser.add_argument("--smoke", action="store_true", help="diagnostic 600 mm scene/99001 seed only; writes no confirmation outputs")
    args = parser.parse_args()
    if args.smoke:
        smoke()
    else:
        run_with_deadline(args.output)
