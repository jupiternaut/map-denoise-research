"""One registered hybrid confirmation using a frozen, read-only dependency."""
import os
import sys

# Importing historical scientific code must not create historical cache files.
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

import numpy as np

from hybrid import (ROOT, DEPENDENCY_ROOT, ProfiledResidual, model_for,
                    certificate_projection, certificate_hybrid, solve_outer,
                    decide, encode, decode)
from run_challenge import (aggregates, write_json, finish_record_before_alarm,
                           save_progress, deadline_reached)

KINDS = ("lovelace_triangle", "rectangle", "layered_contrast", "layered_same", "layered_hidden")
RADII = (Q(0), Q(1, 5), Q(1))
TRUTHS = (Q(6217, 11), Q(7864, 13), Q(8441, 13))
SEEDS = (41001, 41002, 41003)
OFFSETS = (Q(0), Q(-1, 4), Q(1, 4), Q(-3, 4), Q(3, 4), Q(-2), Q(2), Q(-60), Q(60))
EPSILON, TOLERANCE, MAX_BOXES = Q(1, 100), Q(15, 32), 511
WEIGHTS = (Q(-1), Q(-1, 2), Q(0), Q(1, 2), Q(1))
METHODS = ("keep", "certificate_midpoint", "certificate_projection", "certificate_hybrid", "pose_profiled_residual")
RANDOM_TAPE_PREFIX = "hybrid-v1"
WALL_TIME_SECONDS = 1800
SOURCE_COMMIT = "bd8d05aad65e2c55cd9cd288beae4ef949606984"


def input_sources():
    """Declare absolute hashes of new sources and the entire imported old tree.

    Historical outputs and forecast caches are not read. The frozen dependency
    superset includes every imported historical Python module and JSON asset.
    New verifier sources are covered before the source freeze, too.
    """
    files = set(DEPENDENCY_ROOT.glob("*.py"))
    for suffix in ("*.py", "*.json"):
        files.update((DEPENDENCY_ROOT / "source_snapshot").rglob(suffix))
        files.update((DEPENDENCY_ROOT / "verification").rglob(suffix))
        files.update((ROOT / "verification").rglob(suffix))
    files.update(ROOT.glob("*.py"))
    files.add(DEPENDENCY_ROOT / "PROTOCOL.md")
    files.add(ROOT / "PROTOCOL.md")
    if (ROOT / "THEORY.md").exists():
        files.add(ROOT / "THEORY.md")
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def configuration():
    observations = len(KINDS) * len(RADII) * len(TRUTHS) * len(SEEDS)
    tasks = observations * len(OFFSETS)
    return {"kinds": KINDS, "camera_radii_mm": list(map(str, RADII)),
            "truths_evaluation_only": list(map(str, TRUTHS)), "seeds": SEEDS,
            "offsets_evaluation_only": list(map(str, OFFSETS)), "epsilon": str(EPSILON),
            "tolerance_mm": str(TOLERANCE), "max_boxes": MAX_BOXES,
            "expected_observations": observations, "expected_tasks": tasks,
            "expected_decisions": tasks * len(METHODS), "methods": METHODS,
            "random_tape_prefix": RANDOM_TAPE_PREFIX, "depth_grid_step_mm": "1/2",
            "pose_grid_multipliers": list(map(str, WEIGHTS)),
            "tie_threshold": 1e-12, "depth_domain_mm": ["540", "660"],
            "cpu_threads": 1, "gpu_count": 0, "wall_time_limit_seconds": WALL_TIME_SECONDS,
            "scope": "known-shape known-radiance scalar depth; lateral camera error; partial opaque-layer visibility"}


def validate_output(output):
    """Keep all writes in this new experiment and never reuse any output."""
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT.resolve()) or output == ROOT.resolve():
        raise ValueError("output must be a fresh directory within the new experiment root")
    if output.exists():
        raise FileExistsError("choose a fresh output directory; archived results are never overwritten")
    return output


def observation_for(kind, radius, truth, seed):
    model = model_for(kind, radius)
    rng = random.Random(f"{RANDOM_TAPE_PREFIX}:{kind}:{radius}:{truth}:{seed}")
    offsets = (radius * rng.choice(WEIGHTS), Q(0), radius * rng.choice(WEIGHTS))
    image = model.point_prediction(truth, offsets)
    observation = tuple(x + EPSILON * rng.choice(WEIGHTS) for x in image)
    return model, offsets, image, observation


def select_task(outer, incumbent, observation, radius, baseline, grid_losses):
    """Selection signature contains no truth, offsets, seed or state group."""
    actions = (("keep", lambda: {"status": "KEEP", "output": encode(incumbent),
                "gain_lower": encode(Q(0)), "reason": "identity"}),
               ("certificate_midpoint", lambda: decide(outer, incumbent)),
               ("certificate_projection", lambda: certificate_projection(outer, incumbent)),
               ("certificate_hybrid", lambda: certificate_hybrid(outer, incumbent)),
               ("pose_profiled_residual", lambda: baseline.select(observation, radius, incumbent, grid_losses)))
    decisions, seconds = {}, {}
    for method, action in actions:
        tick = time.perf_counter()
        decisions[method] = action()
        seconds[method] = time.perf_counter() - tick
    return decisions, seconds


def paired_statistics(rows, decisions):
    """Evaluation-only exact comparisons, separated by information and prior."""
    by_task = defaultdict(dict)
    for row in rows:
        by_task[row["task_id"]][row["method"]] = row
    decision_map = {(d["task_id"], d["method"]): d for d in decisions}
    groups = defaultdict(list)
    for task_id, arms in by_task.items():
        row = arms["certificate_hybrid"]
        information = "hidden" if row["kind"] == "layered_hidden" else "informative"
        for scope in ("overall", information):
            for prior in ("all", row["prior_position"]):
                for state in ("all", row["state_group"]):
                    groups[(scope, prior, state)].append((task_id, arms))
    result = []
    for (scope, prior, state), tasks in sorted(groups.items()):
        count = Counter()
        branches = Counter()
        for task_id, arms in tasks:
            hybrid, midpoint, projection = (arms[m] for m in
                ("certificate_hybrid", "certificate_midpoint", "certificate_projection"))
            h_error, m_error = Q(hybrid["ae_mm"]), Q(midpoint["ae_mm"])
            count["ae_no_worse_than_midpoint"] += h_error <= m_error
            count["mse_no_worse_than_midpoint"] += h_error * h_error <= m_error * m_error
            count["strictly_lower_ae_than_midpoint"] += h_error < m_error
            count["projection_move_membership_equal"] += ((hybrid["status"] == "MOVE") == (projection["status"] == "MOVE"))
            if midpoint["status"] == "MOVE":
                count["midpoint_moves"] += 1
                h_decision = decision_map[(task_id, "certificate_hybrid")]
                m_decision = decision_map[(task_id, "certificate_midpoint")]
                count["midpoint_move_decision_preserved"] += ({k: v for k, v in h_decision.items() if k not in ("method", "branch")} ==
                                                            {k: v for k, v in m_decision.items() if k != "method"})
            branches[decision_map[(task_id, "certificate_hybrid")]["branch"]] += 1
        result.append({"information": scope, "prior_position": prior, "state_group": state,
                       "n": len(tasks), **{name: count[name] for name in
                       ("ae_no_worse_than_midpoint", "mse_no_worse_than_midpoint", "strictly_lower_ae_than_midpoint",
                        "projection_move_membership_equal", "midpoint_moves", "midpoint_move_decision_preserved")},
                       "hybrid_branch_counts": dict(sorted(branches.items()))})
    return result


def evaluate(output, observations, tasks, truth_records, decisions, outers, started):
    """Called only after all registered decisions have been written and reread."""
    observation_map = {r["observation_id"]: r for r in observations}
    task_map = {r["task_id"]: r for r in tasks}
    truth_map = {r["observation_id"]: r for r in truth_records}
    rows = []
    for decision in decisions:
        task = task_map[decision["task_id"]]
        observed = observation_map[task["observation_id"]]
        truth = decode(truth_map[task["observation_id"]]["truth"])
        a, b = decode(task["incumbent"]), decode(decision["output"])
        off = a - truth
        # Match the historical row schema exactly; branches live in decisions.
        rows.append({"task_id": decision["task_id"], "observation_id": task["observation_id"],
                     "kind": observed["kind"], "radius": observed["radius"], "method": decision["method"],
                     "offset": str(off), "state_group": "correct" if off == 0 else "large" if abs(off) == 60 else "small",
                     "prior_position": "inside_prior" if 540 <= a <= 660 else "outside_prior",
                     "incumbent": str(a), "output": str(b), "truth": str(truth), "status": decision["status"],
                     "initial_ae_mm": str(abs(off)), "ae_mm": str(abs(b - truth)),
                     "actual_gain_mm2": str((a - truth) ** 2 - (b - truth) ** 2)})
    write_json(output / "evaluation" / "ROWS.json", rows)
    import csv
    with (output / "evaluation" / "ROWS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"observation_configurations": len(observations), "template_count": len(KINDS),
               "task_count": len(tasks), "decision_count": len(decisions), "rows": aggregates(rows),
               "information_strata": {name: aggregates([r for r in rows if
                   (r["kind"] == "layered_hidden") == (name == "hidden")])
                   for name in ("informative", "hidden")},
               "hybrid_branch_counts": dict(sorted(Counter(d["branch"] for d in decisions
                   if d["method"] == "certificate_hybrid").items())),
               "paired_relations": paired_statistics(rows, decisions),
               "elapsed_seconds": time.perf_counter() - started,
               "evaluated_boxes": sum(o["outer"]["evaluated_boxes"] for o in outers),
               "unresolved_budget_leaves": sum(sum(n["status"] == "unresolved_budget" for n in o["outer"]["trace"]) for o in outers),
               "incompatible": sum(o["outer"]["hull"] is None for o in outers)}
    write_json(output / "evaluation" / "SUMMARY.json", summary)
    return summary


def run(output):
    output = validate_output(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    write_json(output / "RUN_STATUS.json", {"status": "RUNNING", "observations_complete": 0,
               "decisions_complete": 0, "elapsed_seconds": 0})
    source_lock = input_sources()
    write_json(output / "SOURCE_LOCK.json", {"utc": datetime.now(timezone.utc).isoformat(),
               "source_commit": SOURCE_COMMIT, "new_experiment_root": str(ROOT.resolve()),
               "read_only_dependency_root": str(DEPENDENCY_ROOT.resolve()),
               "path_format": "absolute", "historical_outputs_used": False, "files": source_lock})
    protocol = configuration()
    write_json(output / "PROTOCOL.json", protocol)
    # Public forecasts are freshly compiled before any confirmation observation.
    baseline = {}
    for kind in KINDS:
        baseline[kind] = ProfiledResidual(kind)
        np.savez_compressed(output / f"forecasts-{kind}.npz", forecasts=baseline[kind].forecasts)
        print(json.dumps({"stage": "public_forecast_compile", "kind": kind,
                          "seconds": round(baseline[kind].compile_seconds, 3)}), flush=True)
    write_json(output / "FORECAST_INFO.json", {k: {"compile_seconds": b.compile_seconds,
               "depths": list(map(str, b.depth_grid)), "shape": list(b.forecasts.shape)} for k, b in baseline.items()})
    observations, truth_records, tasks = [], [], []
    specs = [(k, r, t, s) for k in KINDS for r in RADII for t in TRUTHS for s in SEEDS]
    for i, (kind, radius, truth, seed) in enumerate(specs):
        _, offsets, prediction, obs = observation_for(kind, radius, truth, seed)
        oid = f"o{i:03d}"
        observations.append({"observation_id": oid, "kind": kind, "radius": str(radius),
                             "observation": [encode(v) for v in obs]})
        truth_records.append({"observation_id": oid, "truth": encode(truth), "seed": seed,
                              "actual_offsets": [encode(v) for v in offsets],
                              "noise_linf": encode(max(abs(a - b) for a, b in zip(prediction, obs)))})
        for j, offset in enumerate(OFFSETS):
            tasks.append({"task_id": f"{oid}-s{j}", "observation_id": oid, "incumbent": encode(truth + offset)})
    write_json(output / "observations.json", observations)
    write_json(output / "tasks.json", tasks)
    write_json(output / "evaluation" / "truth.json", truth_records)
    tasks_by_observation = defaultdict(list)
    for task in tasks:
        tasks_by_observation[task["observation_id"]].append(task)
    decisions, outers, timing = [], [], []
    for entry in observations:
        oid, kind, radius = entry["observation_id"], entry["kind"], Q(entry["radius"])
        obs = tuple(map(decode, entry["observation"]))
        model = model_for(kind, radius)
        tick = time.perf_counter()
        outer = solve_outer(obs, model.enclosure, epsilon=EPSILON, tolerance=TOLERANCE, max_boxes=MAX_BOXES)
        solve_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        grid_losses = baseline[kind].grid_losses(obs, radius)
        losses_seconds = time.perf_counter() - tick
        selector_seconds = {method: 0.0 for method in METHODS}
        completed_decisions = []
        for task in tasks_by_observation[oid]:
            actions, seconds = select_task(outer, decode(task["incumbent"]), obs, radius, baseline[kind], grid_losses)
            for method in METHODS:
                selector_seconds[method] += seconds[method]
                completed_decisions.append({"task_id": task["task_id"], "method": method, **actions[method]})
        # Commit one entire observation; pending alarms cannot split this record.
        with finish_record_before_alarm():
            decisions.extend(completed_decisions)
            outers.append({"observation_id": oid, "outer": outer})
            timing.append({"observation_id": oid, "solve_seconds": solve_seconds,
                           "grid_score_seconds": losses_seconds, "selector_seconds_by_method": selector_seconds})
            save_progress(output, decisions, outers, timing, started)
        if len(outers) % 9 == 0:
            print(json.dumps({"stage": "confirmation", "observations_complete": len(outers),
                              "kind": kind, "radius": str(radius), "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    write_json(output / "TIMING.json", {"forecast_compile_seconds": {k: b.compile_seconds for k, b in baseline.items()},
               "shared_cost_note": "All three certificate arms reuse one solve_outer per observation; each standalone arm must count that solve. Residual includes fresh forecast compilation, grid scoring and per-incumbent rendering/selecting.",
               "observations": timing})
    # Evaluation consumes the recorded decisions, never an unrecorded action.
    recorded = json.loads((output / "decisions.json").read_text(encoding="utf-8"))
    if recorded != decisions or len(recorded) != protocol["expected_decisions"]:
        raise RuntimeError("complete registered decisions must be saved before evaluation")
    summary = evaluate(output, observations, tasks, truth_records, recorded, outers, started)
    if input_sources() != source_lock:
        raise RuntimeError("scientific sources changed during confirmation; keep outputs but do not accept")
    write_json(output / "RUN_STATUS.json", {"status": "COMPLETE", "observations_complete": len(observations),
               "decisions_complete": len(decisions), "elapsed_seconds": time.perf_counter() - started})
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "information_strata", "paired_relations")}), flush=True)
    return summary


def run_with_deadline(output):
    output = validate_output(output)
    previous_handler = signal.signal(signal.SIGALRM, deadline_reached)
    signal.alarm(WALL_TIME_SECONDS)
    try:
        return run(output)
    except (TimeoutError, RuntimeError) as exc:
        status_path = output / "RUN_STATUS.json"
        previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        write_json(status_path, {**previous, "status": "INCOMPLETE", "reason": str(exc),
                   "evaluation_claim": "partial checkpoint only; not an accepted confirmation"})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def smoke():
    """A diagnostic depth/seed outside confirmation; no confirmation files."""
    tick = time.perf_counter()
    baseline = ProfiledResidual("rectangle")
    model, _, _, obs = observation_for("rectangle", Q(1, 5), Q(600), 91001)
    outer = solve_outer(obs, model.enclosure, epsilon=EPSILON, tolerance=TOLERANCE, max_boxes=MAX_BOXES)
    lo, hi = map(decode, outer["hull"])
    assert lo <= 600 <= hi
    losses = baseline.grid_losses(obs, Q(1, 5))
    for incumbent in (Q(600), Q(59975, 100), Q(60075, 100)):
        actions, _ = select_task(outer, incumbent, obs, Q(1, 5), baseline, losses)
        midpoint, hybrid, projection = (actions[m] for m in
            ("certificate_midpoint", "certificate_hybrid", "certificate_projection"))
        assert abs(decode(hybrid["output"]) - 600) <= abs(decode(midpoint["output"]) - 600)
        assert (hybrid["status"] == "MOVE") == (projection["status"] == "MOVE")
        if midpoint["status"] == "MOVE":
            assert {k: v for k, v in hybrid.items() if k != "branch"} == midpoint
    print(json.dumps({"smoke": "passed", "seconds": time.perf_counter() - tick,
                      "outer": [str(lo), str(hi)], "forecasts": list(baseline.forecasts.shape)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "confirmation-v1")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        smoke()
    else:
        run_with_deadline(arguments.output)
