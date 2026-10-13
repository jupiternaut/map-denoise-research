"""Fresh paired confirmation of shared versus channelwise appearance bounds."""
import os
import sys

sys.dont_write_bytecode = True
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent
NUISANCE_ROOT = ROOT.parent / "lovelace-nuisance-20261011T044500Z"
HYBRID_ROOT = ROOT.parent / "lovelace-hybrid-20261010T214217Z"
DEPENDENCY_ROOT = ROOT.parent / "lovelace-challenge-20261010T205013Z"
sys.path.insert(0, str(NUISANCE_ROOT))
import run_nuisance as old

# Historical modules restore their own paths; all new imports target this run.
sys.path.insert(0, str(ROOT))
from shared_solver import solve_shared

generate_observation = old.generate_observation
calibration_intervals = old.calibration_intervals
prior_outer = old.prior_outer
union_outer = old.union_outer
select_action = old.select_action
solve_channelwise = old.solve_arm
encode, decode = old.encode, old.decode
write_json, save_progress = old.write_json, old.save_progress
finish_record_before_alarm, deadline_reached = old.finish_record_before_alarm, old.deadline_reached

CASES, BRANCHES = old.CASES, old.BRANCHES
OFFSETS, WEIGHTS, SAMPLE_SOURCES = old.OFFSETS, old.WEIGHTS, old.SAMPLE_SOURCES
RADIUS, EPSILON, TOLERANCE, MAX_BOXES = Q(1, 5), Q(1, 100), Q(15, 32), 255
PUBLIC_COLORS = ((Q(0), Q(1)),) * 3
TRUTHS = (Q(6203, 11), Q(7861, 13), Q(9007, 14))
SEEDS = (61001, 61002, 61003)
METHODS = ("keep", "channelwise_union", "shared_union", "calibrated_union")
RANDOM_TAPE_PREFIX = "shared-appearance-v1"
WALL_TIME_SECONDS = 1800
SOURCE_COMMIT = "bd8d05aad65e2c55cd9cd288beae4ef949606984"


def input_sources():
    """Freeze scientific inputs while allowing subsequent reports/analysis."""
    files = {ROOT / "run_shared.py", ROOT / "shared_solver.py", ROOT / "PROTOCOL.md"}
    if (ROOT / "shared_verification.py").exists():
        files.add(ROOT / "shared_verification.py")
    files.update(ROOT.glob("test_*.py"))
    if (ROOT / "THEORY.md").exists():
        files.add(ROOT / "THEORY.md")
    for root in (ROOT, NUISANCE_ROOT, HYBRID_ROOT, DEPENDENCY_ROOT):
        for suffix in ("*.py", "*.json"):
            files.update((root / "verification").rglob(suffix))
        if root != ROOT:
            files.update(p for p in root.glob("*.py") if not p.name.startswith(("analy", "report")))
            for suffix in ("*.py", "*.json"):
                files.update((root / "source_snapshot").rglob(suffix))
            files.add(root / "PROTOCOL.md")
            if (root / "THEORY.md").exists():
                files.add(root / "THEORY.md")
    # Include any imported experiment module even if a future filename differs.
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.name.startswith(("analy", "report")):
                continue
            if path.suffix == ".py" and any(path.is_relative_to(r) for r in (ROOT, NUISANCE_ROOT, HYBRID_ROOT, DEPENDENCY_ROOT)):
                files.add(path)
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def configuration():
    n = len(CASES) * len(TRUTHS) * len(SEEDS)
    return {"cases_evaluation_only": [{"case": c, "rear_branch": b, "colors": list(map(str, colors))} for c, b, colors in CASES],
            "truths_evaluation_only": list(map(str, TRUTHS)), "seeds_evaluation_only": SEEDS,
            "offsets_evaluation_only": list(map(str, OFFSETS)), "camera_radius_mm": str(RADIUS),
            "epsilon": str(EPSILON), "tolerance_mm": str(TOLERANCE), "max_boxes_per_branch": MAX_BOXES,
            "depth_domain_mm": ["540", "660"], "expected_observations": n, "expected_tasks": n * len(OFFSETS),
            "expected_decisions": n * len(OFFSETS) * len(METHODS), "methods": METHODS,
            "expected_correct_tasks": n, "expected_small_tasks": n * (len(OFFSETS) - 1),
            "allowed_occlusion_branches": BRANCHES, "calibration_sources": SAMPLE_SOURCES,
            "calibration_samples_per_observation": 3,
            "calibration_samples_used_per_method": {m: 3 if m == "calibrated_union" else 0 for m in METHODS},
            "random_tape_prefix": RANDOM_TAPE_PREFIX,
            "random_draw_order": "left pose; right pose; each image RGB channel; front-left, front-right, rear calibration noise",
            "random_weights": list(map(str, WEIGHTS)), "shared_constraint_max_rounds": 8,
            "cpu_threads": 1, "gpu_count": 0, "wall_time_limit_seconds": WALL_TIME_SECONDS,
            "information_note": "channelwise_union/shared_union use identical images, [0,1] material priors and branches; calibrated_union alone uses three additional sourced samples"}


def validate_output(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT.resolve()) or output == ROOT.resolve():
        raise ValueError("output must be a fresh directory inside the new shared-appearance experiment")
    if output.exists():
        raise FileExistsError("choose a fresh output directory; existing results are never overwritten")
    return output


def public_arm_outers(observation, radius, calibration):
    """Raw shared constraints never receive calibration or generator metadata."""
    entries = {"keep": {"branches": {}, "outer": prior_outer(), "branch_seconds": {}, "outer_seconds": 0.0}}
    entries["channelwise_union"] = solve_channelwise(observation, radius, BRANCHES, PUBLIC_COLORS)
    entries["shared_union"] = solve_shared(observation, radius, BRANCHES, colors=PUBLIC_COLORS)
    entries["calibrated_union"] = solve_channelwise(observation, radius, BRANCHES, calibration_intervals(calibration))
    return entries


def contains_truth(outer, truth):
    return any(decode(lo) <= truth <= decode(hi) for lo, hi in outer["retained_intervals"])


def aggregate_rows(rows):
    result = old.aggregates(rows)
    for seed in sorted({r["seed"] for r in rows}):
        values = [r for r in rows if r["seed"] == seed]
        for entry in old.aggregates(values):
            if entry["grouping"] == "overall":
                result.append({**entry, "grouping": "by_seed", "condition": str(seed)})
        for case in sorted({r["case"] for r in values}):
            for entry in old.aggregates([r for r in values if r["case"] == case]):
                if entry["grouping"] == "overall":
                    result.append({**entry, "grouping": "by_case_seed", "condition": [case, seed]})
    return result


def paired_statistics(rows):
    by_task = defaultdict(dict)
    for row in rows:
        by_task[row["task_id"]][row["method"]] = row
    grouped = defaultdict(list)
    for arms in by_task.values():
        shared = arms["shared_union"]
        for scope in ("overall", "negative_control" if shared["case"] == "equal_full_negative" else "informative"):
            for state in ("all", shared["state_group"]):
                grouped[(scope, state)].append(arms)
    results = []
    for (scope, state), tasks in sorted(grouped.items()):
        shared = [Q(a["shared_union"]["ae_mm"]) for a in tasks]
        channel = [Q(a["channelwise_union"]["ae_mm"]) for a in tasks]
        results.append({"information": scope, "state_group": state, "n": len(tasks),
                        "shared_lower_ae": sum(a < b for a, b in zip(shared, channel)),
                        "shared_same_ae": sum(a == b for a, b in zip(shared, channel)),
                        "shared_higher_ae": sum(a > b for a, b in zip(shared, channel)),
                        "shared_mae_mm": str(sum(shared) / len(shared)), "channelwise_mae_mm": str(sum(channel) / len(channel)),
                        "shared_mse_mm2": str(sum(a * a for a in shared) / len(shared)),
                        "channelwise_mse_mm2": str(sum(a * a for a in channel) / len(channel))})
    return results


def evaluate(output, observations, tasks, truths, decisions, outers, timing, generation_seconds, started):
    """After recording, associate cases/seeds/truth only for endpoint scoring."""
    truth_map = {t["observation_id"]: t for t in truths}
    task_map = {t["task_id"]: t for t in tasks}
    outer_map = {o["observation_id"]: o["arms"] for o in outers}
    rows = []
    for decision in decisions:
        task = task_map[decision["task_id"]]
        oid, method = task["observation_id"], decision["method"]
        private, outer = truth_map[oid], outer_map[oid][method]["outer"]
        truth, incumbent, result = decode(private["truth"]), decode(task["incumbent"]), decode(decision["output"])
        error, initial = abs(result - truth), abs(incumbent - truth)
        width = None if outer["hull"] is None else str(decode(outer["hull"][1]) - decode(outer["hull"][0]))
        rows.append({"task_id": task["task_id"], "observation_id": oid, "case": private["case"], "seed": private["seed"],
                     "method": method, "state_group": "correct" if initial == 0 else "small",
                     "prior_position": "inside_prior" if 540 <= incumbent <= 660 else "outside_prior",
                     "incumbent": str(incumbent), "output": str(result), "status": decision["status"],
                     "initial_ae_mm": str(initial), "ae_mm": str(error), "se_mm2": str(error * error),
                     "actual_gain_mm2": str(initial * initial - error * error), "covered": contains_truth(outer, truth),
                     "hull_width_mm": width, "retained_branch_count": len(outer["retained_branches"]),
                     "evaluated_boxes": outer["evaluated_boxes"], "unresolved_budget_leaves": outer["unresolved_budget_leaves"],
                     "auxiliary_samples_used": 3 if method == "calibrated_union" else 0})
    write_json(output / "evaluation/ROWS.json", rows)
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
            coverage.append({"observation_id": observation["observation_id"], "case": private["case"], "seed": private["seed"], "method": method,
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
        solve_seconds = sum(t["outer_seconds_by_method"][method] for t in timing)
        action_seconds = sum(t["selector_seconds_by_method"][method] for t in timing)
        per_method[method] = {"outer_seconds": solve_seconds, "selector_seconds": action_seconds, "total_seconds": solve_seconds + action_seconds,
                              "evaluated_boxes": sum(o["arms"][method]["outer"]["evaluated_boxes"] for o in outers),
                              "unresolved_budget_leaves": sum(o["arms"][method]["outer"]["unresolved_budget_leaves"] for o in outers)}
    summary = {"observation_configurations": len(observations), "template_count": len(CASES), "task_count": len(tasks),
               "decision_count": len(decisions), "rows": aggregate_rows(rows), "paired_relations": paired_statistics(rows),
               "sampling_structure": {"case_templates": len(CASES), "depth_settings": len(TRUTHS), "noise_replicates_per_case_depth": len(SEEDS),
                                      "shared_incumbents_per_observation": len(OFFSETS)},
               "sampling_note": "synthetic configurations with repeated noise seeds; initial states and arms share observations; not independent real scenes",
               "hybrid_branch_counts": {m: dict(sorted(Counter(d.get("branch", "identity") for d in decisions if d["method"] == m).items())) for m in METHODS},
               "sample_counts": {"image_configurations": len(observations), "image_channels_per_observation": len(observations[0]["observation"]),
                                 "auxiliary_samples_per_observation": 3, "auxiliary_samples_total": 3 * len(observations),
                                 "auxiliary_samples_used_by_method": {m: 3 * len(observations) if m == "calibrated_union" else 0 for m in METHODS}},
               "information_note": "only channelwise_union versus shared_union is a same-information algorithm comparison; calibrated_union consumes additional sourced samples",
               "per_method_timing": per_method, "generation_seconds": generation_seconds, "elapsed_seconds": time.perf_counter() - started,
               "timing_scope": "observed shared-geometry-cache execution order; every arm includes all branch solves and actions; not standalone cold-cache timing"}
    write_json(output / "evaluation/SUMMARY.json", summary)
    for filename, exported in (("SUMMARY.csv", summary["rows"]), ("TIMING.csv", [{"method": m, **v} for m, v in per_method.items()]),
                               ("PAIRED.csv", summary["paired_relations"])):
        with (output / "evaluation" / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
            writer.writeheader()
            writer.writerows({**r, "condition": json.dumps(r["condition"])} if isinstance(r.get("condition"), list) else r for r in exported)
    return summary


def run(output):
    output = validate_output(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    save_progress(output, [], [], [], started)
    source_lock = input_sources()
    write_json(output / "SOURCE_LOCK.json", {"utc": datetime.now(timezone.utc).isoformat(), "source_commit": SOURCE_COMMIT,
               "new_experiment_root": str(ROOT), "read_only_dependency_roots": [str(NUISANCE_ROOT), str(HYBRID_ROOT), str(DEPENDENCY_ROOT)],
               "path_format": "absolute", "historical_outputs_used": False, "files": source_lock,
               "lock_scope": "scientific implementation/tests/verifier/protocol; post-confirmation reports and analysis excluded"})
    protocol = configuration()
    write_json(output / "PROTOCOL.json", protocol)
    observations, truths, tasks = [], [], []
    tick = time.perf_counter()
    for case, branch, colors in CASES:
        for truth in TRUTHS:
            for seed in SEEDS:
                oid = f"o{len(observations):03d}"
                public, private = generate_observation(case, branch, colors, truth, seed, radius=RADIUS, prefix=RANDOM_TAPE_PREFIX)
                observations.append({"observation_id": oid, **public})
                truths.append({"observation_id": oid, **private})
                for j, offset in enumerate(OFFSETS):
                    tasks.append({"task_id": f"{oid}-s{j}", "observation_id": oid, "incumbent": encode(truth + offset)})
    generation_seconds = time.perf_counter() - tick
    write_json(output / "observations.json", observations)
    write_json(output / "tasks.json", tasks)
    write_json(output / "evaluation/truth.json", truths)
    by_observation = defaultdict(list)
    for task in tasks:
        by_observation[task["observation_id"]].append(task)
    decisions, outers, timing = [], [], []
    for entry in observations:
        oid = entry["observation_id"]
        arms = public_arm_outers(tuple(map(decode, entry["observation"])), Q(entry["radius"]), entry["calibration"])
        completed, seconds = [], {m: 0.0 for m in METHODS}
        for task in by_observation[oid]:
            for method in METHODS:
                tick = time.perf_counter()
                action = select_action(arms[method]["outer"], decode(task["incumbent"]), keep=method == "keep")
                seconds[method] += time.perf_counter() - tick
                completed.append({"task_id": task["task_id"], "method": method, **action})
        with finish_record_before_alarm():
            decisions.extend(completed)
            outers.append({"observation_id": oid, "arms": arms})
            timing.append({"observation_id": oid, "outer_seconds_by_method": {m: arms[m]["outer_seconds"] for m in METHODS},
                           "selector_seconds_by_method": seconds})
            save_progress(output, decisions, outers, timing, started)
        print(json.dumps({"stage": "confirmation", "observations_complete": len(outers), "decisions_complete": len(decisions),
                          "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    recorded = json.loads((output / "decisions.json").read_text(encoding="utf-8"))
    if recorded != decisions or len(recorded) != protocol["expected_decisions"]:
        raise RuntimeError("all registered decisions must be saved before evaluation")
    write_json(output / "TIMING.json", {"generation_seconds": generation_seconds, "observations": timing,
               "shared_cost_note": "fixed execution order and shared geometry cache; recorded arm timing counts all branches and final hybrid actions"})
    summary = evaluate(output, observations, tasks, truths, recorded, outers, timing, generation_seconds, started)
    if input_sources() != source_lock:
        raise RuntimeError("scientific sources changed during confirmation; keep outputs but do not accept")
    write_json(output / "RUN_STATUS.json", {"status": "COMPLETE", "observations_complete": len(observations),
               "decisions_complete": len(decisions), "elapsed_seconds": time.perf_counter() - started})
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "paired_relations")}), flush=True)
    return summary


def run_with_deadline(output):
    output = validate_output(output)
    old_handler = signal.signal(signal.SIGALRM, deadline_reached)
    signal.alarm(WALL_TIME_SECONDS)
    try:
        return run(output)
    except (Exception, KeyboardInterrupt) as exc:
        if output.exists():
            status = output / "RUN_STATUS.json"
            previous = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
            write_json(status, {**previous, "status": "INCOMPLETE", "reason": str(exc),
                       "evaluation_claim": "partial checkpoint only; not a completed confirmation"})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def smoke():
    """Two raw arms at nonconfirmation 600 mm, 99011, distinct material grays."""
    tick = time.perf_counter()
    truth = Q(600)
    public, _ = generate_observation("smoke_only", "partial", (Q(1, 4), Q(1, 2), Q(3, 4)), truth, 99011,
                                    radius=RADIUS, prefix="shared-appearance-smoke-v1")
    image = tuple(map(decode, public["observation"]))
    arms = {"channelwise_union": solve_channelwise(image, RADIUS, BRANCHES, PUBLIC_COLORS),
            "shared_union": solve_shared(image, RADIUS, BRANCHES, colors=PUBLIC_COLORS)}
    for method, entry in arms.items():
        assert contains_truth(entry["outer"], truth)
        for incumbent in (truth, truth - Q(1, 4), truth + Q(3, 4)):
            action = select_action(entry["outer"], incumbent)
            assert abs(decode(action["output"]) - truth) <= abs(incumbent - truth)
            if incumbent == truth:
                assert action["status"] == "KEEP"
    print(json.dumps({"smoke": "passed", "seconds": time.perf_counter() - tick,
                      "arms": {m: {"outer_seconds": e["outer_seconds"], "evaluated_boxes": e["outer"]["evaluated_boxes"],
                          "retained_branches": e["outer"]["retained_branches"], "hull": e["outer"]["hull"]} for m, e in arms.items()}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One registered shared-appearance confirmation; CPU only, fresh output, 1800-second limit.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/confirmation-v1", help="fresh output directory inside this new experiment")
    parser.add_argument("--smoke", action="store_true", help="nonconfirmation 600 mm/99011 scene with both raw arms; no confirmation files")
    args = parser.parse_args()
    if args.smoke:
        smoke()
    else:
        run_with_deadline(args.output)
