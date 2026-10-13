"""A one-shot registered controlled experiment, never a true-depth selector."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import random
import signal
import time

import numpy as np

from challenge import ROOT, ProfiledResidual, model_for, certificate_projection
from continuous_world.interval_solver import solve_outer, decide, encode, decode

KINDS = ("lovelace_triangle", "rectangle", "layered_contrast", "layered_same", "layered_hidden")
RADII = (Q(0), Q(1, 5), Q(1))
TRUTHS = (Q(3965, 7), Q(4238, 7), Q(4541, 7))
SEEDS = (31001, 31002, 31003)
OFFSETS = (Q(0), Q(-1, 4), Q(1, 4), Q(-3, 4), Q(3, 4), Q(-2), Q(2), Q(-60), Q(60))
EPSILON, TOLERANCE, MAX_BOXES = Q(1, 100), Q(15, 32), 511
WEIGHTS = (Q(-1), Q(-1, 2), Q(0), Q(1, 2), Q(1))
METHODS = ("keep", "certificate_midpoint", "certificate_projection", "pose_profiled_residual")


@contextmanager
def finish_record_before_alarm():
    """Deliver a pending deadline immediately after a short record write."""
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def write_json(path, value):
    with finish_record_before_alarm():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_progress(output, decisions, outers, timing, started):
    """Save only completed observations; interrupted work is not a result."""
    with finish_record_before_alarm():
        write_json(output / "decisions.json", decisions)
        write_json(output / "outers.json", outers)
        write_json(output / "TIMING_PARTIAL.json", timing)
        write_json(output / "RUN_STATUS.json", {"status": "RUNNING", "observations_complete": len(outers),
                   "decisions_complete": len(decisions), "elapsed_seconds": time.perf_counter() - started})


def deadline_reached(signum, frame):
    raise TimeoutError("registered 1800-second CPU wall-time limit reached")


def input_sources():
    files = list(ROOT.glob("*.py")) + list((ROOT / "source_snapshot").rglob("*.py"))
    files += list((ROOT / "verification").rglob("*.py"))
    files += list((ROOT / "source_snapshot").rglob("*.json"))
    files += [ROOT / "PROTOCOL.md"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def observation_for(kind, radius, truth, seed):
    model = model_for(kind, radius)
    rng = random.Random(f"challenge-v1:{kind}:{radius}:{truth}:{seed}")
    offsets = (radius * rng.choice(WEIGHTS), Q(0), radius * rng.choice(WEIGHTS))
    image = model.point_prediction(truth, offsets)
    observation = tuple(x + EPSILON * rng.choice(WEIGHTS) for x in image)
    return model, offsets, image, observation


def aggregates(rows):
    grouped = defaultdict(list)
    for r in rows:
        for grouping in ("overall", "by_kind", "by_radius", "by_offset", "by_prior", "by_kind_radius"):
            extra = {"overall": (), "by_kind": (r["kind"],), "by_radius": (r["radius"],),
                     "by_offset": (r["offset"],), "by_prior": (r["prior_position"],),
                     "by_kind_radius": (r["kind"], r["radius"])}[grouping]
            grouped[(grouping, r["state_group"], r["method"], *extra)].append(r)
    out = []
    for key, values in sorted(grouped.items()):
        n = len(values)
        ae = [Q(v["ae_mm"]) for v in values]
        be = [Q(v["initial_ae_mm"]) for v in values]
        mae = sum(ae) / n
        mse = sum(x*x for x in ae) / n
        baseline_mae = sum(be) / n
        baseline_mse = sum(x*x for x in be) / n
        out.append({"grouping": key[0], "state_group": key[1], "method": key[2], "condition": list(key[3:]),
                    "n": n, "improved": sum(a < b for a,b in zip(ae,be)),
                    "harm": sum(a > b for a,b in zip(ae,be)), "same": sum(a == b for a,b in zip(ae,be)),
                    "moved": sum(v["status"] == "MOVE" for v in values),
                    "mae_mm": float(mae), "mse_mm2": float(mse),
                    "initial_mae_mm": float(baseline_mae), "initial_mse_mm2": float(baseline_mse),
                    "mae_reduction_percent": None if baseline_mae == 0 else float(100*(1-mae/baseline_mae)),
                    "mse_reduction_percent": None if baseline_mse == 0 else float(100*(1-mse/baseline_mse)),
                    "max_error_mm": float(max(ae))})
    return out


def run(output):
    if output.exists():
        raise FileExistsError("choose a fresh output directory; archived results are never overwritten")
    output.mkdir(parents=True)
    started = time.perf_counter()
    write_json(output / "RUN_STATUS.json", {"status": "RUNNING", "observations_complete": 0,
               "decisions_complete": 0, "elapsed_seconds": 0})
    source_lock = input_sources()
    write_json(output / "SOURCE_LOCK.json", {"utc": datetime.now(timezone.utc).isoformat(),
               "source_commit": "bd8d05aad65e2c55cd9cd288beae4ef949606984", "files": source_lock})
    specs = [(k, r, t, s) for k in KINDS for r in RADII for t in TRUTHS for s in SEEDS]
    protocol = {"kinds": KINDS, "camera_radii_mm": list(map(str, RADII)), "truths_evaluation_only": list(map(str, TRUTHS)),
                "seeds": SEEDS, "offsets_evaluation_only": list(map(str, OFFSETS)), "epsilon": str(EPSILON),
                "tolerance_mm": str(TOLERANCE), "max_boxes": MAX_BOXES, "expected_observations": len(specs),
                "expected_tasks": len(specs)*len(OFFSETS), "methods": METHODS, "depth_grid_step_mm": "1/2",
                "pose_grid_multipliers": ["-1", "-1/2", "0", "1/2", "1"],
                "scope": "known-shape known-radiance scalar depth; lateral camera error; partial opaque-layer visibility"}
    write_json(output / "PROTOCOL.json", protocol)
    # Models/forecasts compiled before producing any confirmation observation.
    baseline = {}
    for kind in KINDS:
        baseline[kind] = ProfiledResidual(kind)
        np.savez_compressed(output / f"forecasts-{kind}.npz", forecasts=baseline[kind].forecasts)
        print(json.dumps({"stage": "public_forecast_compile", "kind": kind,
                          "seconds": round(baseline[kind].compile_seconds, 3)}), flush=True)
    write_json(output / "FORECAST_INFO.json", {k: {"compile_seconds": b.compile_seconds,
               "depths": list(map(str, b.depth_grid)), "shape": list(b.forecasts.shape)} for k,b in baseline.items()})
    observations, truth_records, tasks = [], [], []
    for i, (kind, radius, truth, seed) in enumerate(specs):
        model, offsets, prediction, obs = observation_for(kind, radius, truth, seed)
        oid = f"o{i:03d}"
        observations.append({"observation_id": oid, "kind": kind, "radius": str(radius),
                             "observation": [encode(v) for v in obs]})
        truth_records.append({"observation_id": oid, "truth": encode(truth), "seed": seed,
                              "actual_offsets": [encode(v) for v in offsets],
                              "noise_linf": encode(max(abs(a-b) for a,b in zip(prediction,obs)))})
        for j, off in enumerate(OFFSETS):
            tasks.append({"task_id": f"{oid}-s{j}", "observation_id": oid, "incumbent": encode(truth+off)})
    write_json(output / "observations.json", observations)
    write_json(output / "tasks.json", tasks)
    write_json(output / "evaluation" / "truth.json", truth_records)
    observation_map = {r["observation_id"]: r for r in observations}
    task_map = {r["task_id"]: r for r in tasks}
    truth_map = {r["observation_id"]: r for r in truth_records}
    decisions, outers, timing = [], [], []
    # Selection only sees observation/model, its own incumbent and public grid.
    for entry in observations:
        oid, kind, radius = entry["observation_id"], entry["kind"], Q(entry["radius"])
        obs = tuple(map(decode, entry["observation"]))
        model = model_for(kind, radius)
        tick = time.perf_counter()
        outer = solve_outer(obs, model.enclosure, epsilon=EPSILON, tolerance=TOLERANCE, max_boxes=MAX_BOXES)
        solve_seconds = time.perf_counter()-tick
        outers.append({"observation_id": oid, "outer": outer})
        tick = time.perf_counter()
        grid_losses = baseline[kind].grid_losses(obs, radius)
        losses_seconds = time.perf_counter()-tick
        selector_seconds = {method: 0.0 for method in METHODS}
        for task in tasks:
            if task["observation_id"] != oid:
                continue
            a = decode(task["incumbent"])
            actions = {"keep": lambda: {"status": "KEEP", "output": encode(a), "gain_lower": encode(Q(0)), "reason": "identity"},
                       "certificate_midpoint": lambda: decide(outer, a),
                       "certificate_projection": lambda: certificate_projection(outer, a),
                       "pose_profiled_residual": lambda: baseline[kind].select(obs, radius, a, grid_losses)}
            for method, action in actions.items():
                tick = time.perf_counter()
                decision = action()
                selector_seconds[method] += time.perf_counter() - tick
                decisions.append({"task_id": task["task_id"], "method": method, **decision})
        timing.append({"observation_id": oid, "solve_seconds": solve_seconds, "grid_score_seconds": losses_seconds,
                       "selector_seconds_by_method": selector_seconds})
        if len(outers) % 9 == 0:
            save_progress(output, decisions, outers, timing, started)
            print(json.dumps({"stage": "confirmation", "observations_complete": len(outers),
                              "kind": kind, "radius": str(radius), "elapsed_seconds": round(time.perf_counter()-started, 2)}), flush=True)
    write_json(output / "decisions.json", decisions)
    write_json(output / "outers.json", outers)
    write_json(output / "TIMING.json", {"forecast_compile_seconds": {k:b.compile_seconds for k,b in baseline.items()},
               "shared_cost_note": "Both certificates reuse one solve_outer per observation; each standalone arm must count that solve. Residual includes forecast compilation, grid scoring and per-incumbent rendering/selecting.",
               "observations": timing})
    # Evaluation starts only after every decision has been saved.
    rows = []
    for d in decisions:
        task = task_map[d["task_id"]]
        observed = observation_map[task["observation_id"]]
        truth = decode(truth_map[task["observation_id"]]["truth"])
        a, b = decode(task["incumbent"]), decode(d["output"])
        off = a-truth
        rows.append({"task_id": d["task_id"], "observation_id": task["observation_id"],
                     "kind": observed["kind"], "radius": observed["radius"], "method": d["method"],
                     "offset": str(off), "state_group": "correct" if off == 0 else "large" if abs(off) == 60 else "small",
                     "prior_position": "inside_prior" if 540 <= a <= 660 else "outside_prior",
                     "incumbent": str(a), "output": str(b), "truth": str(truth), "status": d["status"],
                     "initial_ae_mm": str(abs(off)), "ae_mm": str(abs(b-truth)),
                     "actual_gain_mm2": str((a-truth)**2-(b-truth)**2)})
    write_json(output / "evaluation" / "ROWS.json", rows)
    import csv
    with (output / "evaluation" / "ROWS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"observation_configurations": len(observations), "template_count": len(KINDS),
               "task_count": len(tasks), "decision_count": len(decisions), "rows": aggregates(rows),
               "elapsed_seconds": time.perf_counter()-started,
               "evaluated_boxes": sum(o["outer"]["evaluated_boxes"] for o in outers),
               "unresolved_budget_leaves": sum(sum(n["status"] == "unresolved_budget" for n in o["outer"]["trace"]) for o in outers),
               "incompatible": sum(o["outer"]["hull"] is None for o in outers)}
    write_json(output / "evaluation" / "SUMMARY.json", summary)
    if input_sources() != source_lock:
        raise RuntimeError("scientific sources changed during confirmation; keep outputs but do not accept")
    write_json(output / "RUN_STATUS.json", {"status": "COMPLETE", "observations_complete": len(observations),
               "decisions_complete": len(decisions), "elapsed_seconds": time.perf_counter() - started})
    print(json.dumps({k:v for k,v in summary.items() if k != "rows"}), flush=True)


def smoke():
    tick = time.perf_counter()
    model, _, _, obs = observation_for("rectangle", Q(1,5), Q(600), 91001)
    outer = solve_outer(obs, model.enclosure, epsilon=EPSILON, tolerance=TOLERANCE, max_boxes=MAX_BOXES)
    lo, hi = map(decode, outer["hull"])
    assert lo <= 600 <= hi
    baseline = ProfiledResidual("rectangle")
    losses = baseline.grid_losses(obs, Q(1,5))
    for a in (Q(600), Q(59975,100), Q(60075,100)):
        result = certificate_projection(outer,a)
        b = decode(result["output"])
        assert (b-600)**2 <= (a-600)**2
        baseline.select(obs,Q(1,5),a,losses)
    print(json.dumps({"smoke": "passed", "seconds": time.perf_counter()-tick,
                      "outer": [str(lo),str(hi)], "forecasts": list(baseline.forecasts.shape)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "confirmation-v1")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        smoke()
    else:
        output = args.output.resolve()
        signal.signal(signal.SIGALRM, deadline_reached)
        signal.alarm(1800)
        try:
            run(output)
        except TimeoutError as exc:
            status_path = output / "RUN_STATUS.json"
            previous = json.loads(status_path.read_text()) if status_path.exists() else {}
            write_json(status_path, {**previous, "status": "INCOMPLETE", "reason": str(exc),
                       "evaluation_claim": "partial checkpoint only; not a completed confirmation"})
            raise
        finally:
            signal.alarm(0)
