"""A9 public-only interval inference. No private parsing, task-grid seeds or GT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parent
EXPECTED = Path("/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/continuous-anchors-1")
P = ROOT.parent
A8 = P / "independent-anchors-1"
OLD = A8 / "outputs/anchor-v1"
OUT = ROOT / "outputs/continuous-v1"
PYTHON = Path("/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python")
import continuous_solver as solver
SCIENCE_NAMES = ("AGENTS.md", "PROTOCOL.md", "REGISTRATION.json", "continuous_solver.py",
                 "test_continuous_solver.py", "verify_continuous.py", "test_verify_continuous.py",
                 "run_continuous.py", "audit_continuous.py", "execute_continuous.py",
                 "test_continuous_pipeline.py")
def load(path):
    return json.loads(Path(path).read_text())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value, replace=False):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT):
        raise ValueError("write outside exact A9 formal outputs")
    if replace and path.name != "RUN_STATUS.json":
        raise ValueError("only live status may replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        target = path.with_name(path.name + ".pending") if replace else path
        with target.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        if replace:
            target.replace(path)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
def sources():
    files = load(OLD / "SOURCE_LOCK.json")["files"].copy()
    for name in SCIENCE_NAMES:
        path = ROOT / name
        files[str(path)] = sha(path)
    return dict(sorted(files.items()))
def input_paths():
    return (A8 / "ARTIFACTS.json", A8 / "FINAL_SEAL.json", OLD / "observations.json",
            OLD / "tasks.json", OLD / "STAGE_STATUS.json", OLD / "verification/VERIFY.json",
            A8 / "data/anchors.json", A8 / "data/private/measurements.json",
            P / "appearance-variation-1/data/private/truth.json",
            P / "joint-feasibility-1/outputs/polytope-v1/outers.json")
def inputs():
    return {str(path): sha(path) for path in input_paths()}
def validate_old():
    index = load(A8 / "ARTIFACTS.json")
    for rel, digest in index["files"].items():
        if sha(A8 / rel) != digest:
            raise RuntimeError("old A8 artifact drift: " + rel)
    if load(OLD / "STAGE_STATUS.json")["status"] != "COMPLETE":
        raise RuntimeError("A8 formal status must remain COMPLETE")
    if load(OLD / "verification/VERIFY.json")["status"] != "PASS":
        raise RuntimeError("A8 independent check not PASS")
    for path, digest in load(OLD / "SOURCE_LOCK.json")["files"].items():
        if sha(path) != digest:
            raise RuntimeError("old transitive source changed: " + path)
def photo_payload(public):
    return {key: public[key] for key in ("observation", "radius", "baseline", "clipped")}
def calibration_payload(oldarm):
    return {key: oldarm[key] for key in ("calibration_intervals", "appearance_model", "drift_bound")}
def outer_for_observation(public, oldarm, anchor, arm, main_end):
    # No task or incumbent collection can enter this method.
    solver.clear_geometry_cache()
    branches = []
    photo = photo_payload(public)
    calibration = calibration_payload(oldarm)
    for branch in solver.BRANCHES:
        end = time.monotonic() + max(0.0, min(solver.BRANCH_SECONDS, main_end - time.time()))
        record = solver.solve_interval_branch(photo, calibration, anchor, arm, branch, deadline=end)
        branches.append({"branch": branch, **record})
    union = solver.merge_intervals(pair for b in branches for pair in b["outer_intervals"])
    return {"arm": arm, "observation_id": public["observation_id"], "branches": branches,
            "outer_intervals": [[solver.encode(lo), solver.encode(hi)] for lo, hi in union],
            "hull": None if not union else [solver.encode(union[0][0]), solver.encode(union[-1][1])]}
def selections(tasks, sets):
    lookup = {(s["observation_id"], s["arm"]): s for s in sets}
    records = []
    for task in tasks:
        for arm in solver.ARM_NAMES:
            outer = lookup[task["observation_id"], arm]["outer_intervals"]
            for policy in ("HULL", "UNION"):
                result = solver.select_depth(outer, task["incumbent"], policy)
                records.append({"task_id": task["task_id"], "observation_id": task["observation_id"],
                                "arm": arm, **result})
    return records
def main(output):
    if socket.gethostname() != "liekkas" or ROOT != EXPECTED or output != OUT or Path(sys.executable) != PYTHON:
        raise RuntimeError("wrong exact target/runtime")
    budget = load(OUT / "BUDGET.json")
    deadline, main_end = budget["deadline_epoch"], budget["main_deadline_epoch"]
    frozen = load(OUT / "SOURCE_LOCK.json")["files"]
    if sources() != frozen:
        raise RuntimeError("source lock drift")
    validate_old()
    input_hashes = inputs()
    save(OUT / "INPUTS.json", {"files": input_hashes})
    public = load(OLD / "observations.json")
    anchors = {a["observation_id"]: a for a in load(A8 / "data/anchors.json")}
    frozen_arms = load(P / "joint-feasibility-1/outputs/polytope-v1/outers.json")
    if len(public) != 36 or len({p["observation_id"] for p in public}) != 36 or set(anchors) != {p["observation_id"] for p in public}:
        raise RuntimeError("fixed population identity mismatch")
    save(OUT / "observations.json", public)
    save(OUT / "REGISTRATION.json", load(ROOT / "REGISTRATION.json"))
    sets = []
    def alarm(*args):
        raise TimeoutError("A9 shared formal deadline")
    old_alarm = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()))
    try:
        for index, photo in enumerate(public):
            oid = photo["observation_id"]
            arm_records = [outer_for_observation(photo, frozen_arms[oid]["box"]["joint"], anchors[oid], arm, main_end)
                           for arm in solver.ARM_NAMES]
            sets.extend(arm_records)
            save(OUT / ("checkpoint/" + oid + ".json"), arm_records)
            summary = {"status": "RUNNING", "observations_complete": index + 1, "sets_complete": len(sets),
                       "seconds": time.time() - budget["start_epoch"],
                       "classified_nodes": sum(b["classified_nodes"] for r in sets for b in r["branches"]),
                       "main_cutoff_reached": time.time() >= main_end}
            save(OUT / "RUN_STATUS.json", summary, replace=True)
            print(json.dumps(summary), flush=True)
        if len(sets) != 144 or sum(len(r["branches"]) for r in sets) != 432:
            raise RuntimeError("complete outer-set matrix required")
        save(OUT / "sets.json", sets)
        if load(OUT / "sets.json") != sets:
            raise RuntimeError("outer-set readback failed")
        # Task file is first parsed AFTER every outer set is fixed and saved.
        tasks = load(OLD / "tasks.json")
        if len(tasks) != 324 or len({t["task_id"] for t in tasks}) != 324:
            raise RuntimeError("324 unique tasks required")
        counts = {p["observation_id"]: 0 for p in public}
        for task in tasks:
            if set(task) != {"task_id", "base_task_id", "observation_id", "incumbent"} or task["observation_id"] not in counts:
                raise RuntimeError("frozen task schema/association")
            counts[task["observation_id"]] += 1
        if set(counts.values()) != {9}:
            raise RuntimeError("nine separately evaluated incumbents per observation")
        predictions = selections(tasks, sets)
        if len(predictions) != 2592:
            raise RuntimeError("complete prediction matrix required")
        save(OUT / "tasks.json", tasks)
        save(OUT / "predictions.json", predictions)
        if load(OUT / "predictions.json") != predictions:
            raise RuntimeError("prediction readback failed")
        if sources() != frozen or inputs() != input_hashes:
            raise RuntimeError("source/input bytes changed")
        save(OUT / "PREDICTION_RECEIPT.json", {"sets_sha256": sha(OUT / "sets.json"),
             "predictions_sha256": sha(OUT / "predictions.json"), "persisted_epoch": time.time(),
             "source_lock_sha256": sha(OUT / "SOURCE_LOCK.json"), "private_parsed": False})
        save(OUT / "RUN_STATUS.json", {"status": "COMPLETE", "observations_complete": 36,
             "sets_complete": 144, "trees_complete": 432, "predictions_complete": 2592,
             "seconds": time.time() - budget["start_epoch"], "main_cutoff_reached": time.time() >= main_end}, replace=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    main(parser.parse_args().output.resolve())

