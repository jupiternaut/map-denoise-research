#!/usr/bin/env python3
"""Deterministic independent arithmetic audit of the registered confirmation.

No challenge/run_challenge or solver policy/aggregation function is imported.
Exact arithmetic, labels, partition structure and every recorded exclusion
witness are checked here. The disclosed shared image model is used only for
135 truth-image/noise checks and a bounded deterministic sample of exclusions.
This is not an independent proof of that model's physical/enclosure contract,
and it does not rerun the finite residual search or the continuous solver.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import socket
import sys
import time

sys.dont_write_bytecode = True
Q = Fraction
ROOT = Path(__file__).resolve().parents[1]
KINDS = ("lovelace_triangle", "rectangle", "layered_contrast", "layered_same", "layered_hidden")
RADII = (Q(0), Q(1, 5), Q(1))
TRUTHS = (Q(3965, 7), Q(4238, 7), Q(4541, 7))
SEEDS = (31001, 31002, 31003)
OFFSETS = (Q(0), Q(-1, 4), Q(1, 4), Q(-3, 4), Q(3, 4), Q(-2), Q(2), Q(-60), Q(60))
METHODS = ("keep", "certificate_midpoint", "certificate_projection", "pose_profiled_residual")
CERTIFICATES = METHODS[1:3]
DOMAIN = (Q(540), Q(660))
EPSILON, TOLERANCE, MAX_BOXES = Q(1, 100), Q(15, 32), 511
WEIGHTS = (Q(-1), Q(-1, 2), Q(0), Q(1, 2), Q(1))
CHANNELS = 3 * 8 * 6 * 3
REQUIRED = {
    "observations": "observations.json", "tasks": "tasks.json",
    "truth": "evaluation/truth.json", "decisions": "decisions.json",
    "outers": "outers.json", "rows": "evaluation/ROWS.json",
    "summary": "evaluation/SUMMARY.json",
}


class VerificationError(ValueError):
    """A recorded result violates a reproducible check."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def exact(value, where="rational"):
    """Decode the solver's rational schema without importing its decoder."""
    if isinstance(value, dict):
        require(set(value) == {"numerator", "denominator"}, f"{where}: invalid rational keys")
        n, d = value["numerator"], value["denominator"]
        require(type(n) is int and type(d) is int and d > 0, f"{where}: invalid integer rational")
        result = Q(n, d)
        require(result.numerator == n and result.denominator == d, f"{where}: noncanonical rational")
        return result
    require(isinstance(value, (Q, str, int)) and not isinstance(value, bool),
            f"{where}: exact rational required; floats are forbidden")
    try:
        return Q(value)
    except (ValueError, ZeroDivisionError) as error:
        raise VerificationError(f"{where}: invalid rational {value!r}") from error


def pair(value, where):
    require(isinstance(value, (list, tuple)) and len(value) == 2, f"{where}: expected endpoint pair")
    result = tuple(exact(x, where) for x in value)
    require(result[0] <= result[1], f"{where}: reversed endpoints")
    return result


def encoded(value):
    value = exact(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def indexed(records, key, where):
    require(isinstance(records, list) and records, f"{where}: empty or invalid list")
    out = {}
    for record in records:
        require(isinstance(record, dict), f"{where}: record must be object")
        try:
            ident = key(record)
            require(ident not in out, f"{where}: duplicate key {ident!r}")
            out[ident] = record
        except (KeyError, TypeError) as error:
            raise VerificationError(f"{where}: invalid record identity") from error
    return out


def verify_partition(intervals, domain=DOMAIN):
    """A closed partition may share boundaries but cannot have gaps/overlaps."""
    require(bool(intervals), "partition: no terminal leaves")
    intervals = sorted(intervals)
    require(intervals[0][0] == domain[0] and intervals[-1][1] == domain[1],
            "partition: terminal leaves do not span the domain")
    for low, high in intervals:
        require(domain[0] <= low < high <= domain[1], "partition: invalid leaf interval")
    for first, second in zip(intervals, intervals[1:]):
        require(first[1] == second[0], "partition: gap or interior overlap between terminal leaves")


def verify_outer(outer, observation, truth, where):
    require(isinstance(outer, dict), f"{where}: outer must be object")
    domain = pair(outer["domain"], f"{where}.domain")
    require(domain == DOMAIN, f"{where}: incorrect registered initial domain")
    require(exact(outer["epsilon"]) == EPSILON and exact(outer["tolerance"]) == TOLERANCE,
            f"{where}: changed noise/tolerance budget")
    require(type(outer["max_boxes"]) is int and outer["max_boxes"] == MAX_BOXES,
            f"{where}: changed max_boxes")
    require(outer.get("scope") == "outer_set_for_continuous_depth_and_model_nuisance_contract",
            f"{where}: changed outer scope")
    nodes = indexed(outer["trace"], lambda n: n["id"], f"{where}.trace")
    trace = outer["trace"]
    require(all(type(n["id"]) is int for n in trace), f"{where}: node IDs must be integers")
    require([n["id"] for n in trace] == list(range(len(trace))), f"{where}: noncontiguous/BFS trace IDs")
    require(nodes[0]["parent"] is None, f"{where}: root has parent")
    boxes = {ident: (exact(n["lo"]), exact(n["hi"])) for ident, n in nodes.items()}
    require(boxes[0] == domain, f"{where}: root does not equal full initial domain")
    retained, leaves, excluded, referenced = [], [], [], set()
    evaluated = 0
    for n in trace:
        ident, status = n["id"], n["status"]
        low, high = boxes[ident]
        require(domain[0] <= low < high <= domain[1], f"{where}: invalid box {ident}")
        require(status in {"split", "excluded", "retained_tolerance", "unresolved_budget"},
                f"{where}: unknown node status {status!r}")
        base = {"id", "parent", "lo", "hi", "status"}
        extras = {"split": {"children"}, "excluded": {"witness_channel", "prediction_interval"},
                  "retained_tolerance": set(), "unresolved_budget": set()}[status]
        require(set(n) == base | extras, f"{where}: incorrect node schema {ident}")
        if ident:
            require(type(n["parent"]) is int and n["parent"] < ident and n["parent"] in nodes,
                    f"{where}: missing/invalid parent for {ident}")
        if status == "unresolved_budget":
            require(evaluated == MAX_BOXES, f"{where}: unresolved before work budget exhausted")
        else:
            require(evaluated < MAX_BOXES, f"{where}: evaluated beyond registered budget")
            evaluated += 1
        if status == "split":
            require(high - low > TOLERANCE, f"{where}: split below tolerance")
            children = n["children"]
            require(isinstance(children, list) and len(children) == 2 and
                    all(type(child) is int and child in nodes for child in children) and
                    children[0] != children[1], f"{where}: invalid split children")
            middle = (low + high) / 2
            for child, expected in zip(children, ((low, middle), (middle, high))):
                require(child > ident and child not in referenced and nodes[child]["parent"] == ident,
                        f"{where}: duplicate/mismatched child {child}")
                require(boxes[child] == expected, f"{where}: child is not exact midpoint bisection")
                referenced.add(child)
        else:
            leaves.append((low, high))
            if status == "excluded":
                channel = n["witness_channel"]
                require(type(channel) is int and 0 <= channel < len(observation),
                        f"{where}: invalid exclusion channel")
                a, b = pair(n["prediction_interval"], f"{where}.witness")
                require(Q(0) <= a <= b <= Q(1), f"{where}: witness outside model radiance range")
                require(a > observation[channel] + EPSILON or b < observation[channel] - EPSILON,
                        f"{where}: exclusion witness does not conflict with observation noise bound")
                excluded.append(n)
            else:
                if status == "retained_tolerance":
                    require(high - low <= TOLERANCE, f"{where}: retained_tolerance leaf too wide")
                retained.append(n)
    require(referenced == set(nodes) - {0}, f"{where}: orphan/unreachable trace nodes")
    verify_partition(leaves, domain)
    require(type(outer["evaluated_boxes"]) is int and outer["evaluated_boxes"] == evaluated,
            f"{where}: false evaluated_boxes")
    retained_ids = outer["retained_ids"]
    require(isinstance(retained_ids, list) and all(type(i) is int for i in retained_ids) and
            retained_ids == [n["id"] for n in retained], f"{where}: incorrect retained_ids")
    require(retained, f"{where}: empty certified outer cannot cover this registered truth")
    expected_hull = (min(boxes[n["id"]][0] for n in retained), max(boxes[n["id"]][1] for n in retained))
    require(outer["hull"] is not None and pair(outer["hull"], f"{where}.hull") == expected_hull,
            f"{where}: hull is not the hull of all retained leaves")
    require(any(boxes[n["id"]][0] <= truth <= boxes[n["id"]][1] for n in retained),
            f"{where}: truth is outside every retained leaf")
    require(expected_hull[0] <= truth <= expected_hull[1], f"{where}: truth outside hull")
    return {"hull": expected_hull, "excluded": excluded, "boxes": boxes,
            "evaluated": evaluated, "budget_leaves": sum(n["status"] == "unresolved_budget" for n in trace)}


def certificate_action(hull, incumbent, method):
    """Independent endpoint calculation, including strict gain and KEEP fallback."""
    a = exact(incumbent)
    if hull is None:
        return {"status": "INCOMPATIBLE", "output": a, "gain_lower": None,
                "reason": "empty_certified_outer_set"}
    left, right = hull
    proposal = (left + right) / 2 if method == "certificate_midpoint" else min(max(a, left), right)
    lower_gain = min((a - left) ** 2 - (proposal - left) ** 2,
                     (a - right) ** 2 - (proposal - right) ** 2)
    move = proposal != a and lower_gain > 0
    return {"status": "MOVE" if move else "KEEP", "output": proposal if move else a,
            "gain_lower": lower_gain if move else Q(0),
            "reason": "strict_gain_over_continuous_outer_hull" if move else "no_strict_safe_gain_for_proposal"}


def finite_number(value, where):
    require(type(value) in (int, float) and math.isfinite(value), f"{where}: finite number required")
    return value


def verify_decision(decision, incumbent, hull, where):
    method = decision["method"]
    b = exact(decision["output"], f"{where}.output")
    require(method in METHODS, f"{where}: unregistered method")
    fields = {"task_id", "method", "status", "output", "gain_lower", "reason"}
    if method == "keep":
        expected = {"status": "KEEP", "output": incumbent, "gain_lower": Q(0), "reason": "identity"}
    elif method in CERTIFICATES:
        expected = certificate_action(hull, incumbent, method)
    else:
        fields |= {"incumbent_residual", "minimum_residual"}
        require(b == incumbent or (DOMAIN[0] <= b <= DOMAIN[1] and (b - DOMAIN[0]) * 2 == int((b - DOMAIN[0]) * 2)),
                f"{where}: residual output is neither incumbent nor registered half-mm depth grid")
        expected = {"status": "KEEP" if b == incumbent else "MOVE", "output": b,
                    "gain_lower": None, "reason": "finite_pose_profiled_residual"}
        initial = finite_number(decision["incumbent_residual"], where)
        minimum = finite_number(decision["minimum_residual"], where)
        require(0 <= minimum <= initial, f"{where}: invalid residual minimum")
        require((initial <= minimum + 1e-12) == (b == incumbent), f"{where}: residual tie/KEEP inconsistent")
    require(set(decision) == fields, f"{where}: incorrect decision schema")
    for field, value in expected.items():
        actual = decision[field]
        if field in {"output", "gain_lower"} and actual is not None:
            actual = exact(actual, f"{where}.{field}")
        require(actual == value, f"{where}: incorrect {field}; got {actual!r}, expected {value!r}")
    return b


def expected_row(task, observation, truth, decision, output):
    a = exact(task["incumbent"])
    off = a - truth
    return {"task_id": task["task_id"], "observation_id": task["observation_id"],
            "kind": observation["kind"], "radius": observation["radius"], "method": decision["method"],
            "offset": str(off), "state_group": "correct" if off == 0 else "large" if abs(off) == 60 else "small",
            "prior_position": "inside_prior" if DOMAIN[0] <= a <= DOMAIN[1] else "outside_prior",
            "incumbent": str(a), "output": str(output), "truth": str(truth), "status": decision["status"],
            "initial_ae_mm": str(abs(off)), "ae_mm": str(abs(output - truth)),
            "actual_gain_mm2": str((a - truth) ** 2 - (output - truth) ** 2)}


def verify_row(row, expected, where):
    require(isinstance(row, dict) and set(row) == set(expected), f"{where}: incorrect evaluation row schema")
    for field, value in expected.items():
        require(row[field] == value, f"{where}: incorrect {field}; got {row[field]!r}, expected {value!r}")


def recompute_summary(rows):
    """Build all registered groups from independently derived rows only."""
    groups = defaultdict(list)
    for row in rows:
        conditions = {"overall": (), "by_kind": (row["kind"],), "by_radius": (row["radius"],),
                      "by_offset": (row["offset"],), "by_prior": (row["prior_position"],),
                      "by_kind_radius": (row["kind"], row["radius"])}
        for grouping, condition in conditions.items():
            groups[(grouping, row["state_group"], row["method"], condition)].append(row)
    out = {}
    for key, values in groups.items():
        n = len(values)
        errors = [exact(row["ae_mm"]) for row in values]
        initial = [exact(row["initial_ae_mm"]) for row in values]
        mae, mse = sum(errors, Q(0)) / n, sum((e * e for e in errors), Q(0)) / n
        initial_mae = sum(initial, Q(0)) / n
        initial_mse = sum((e * e for e in initial), Q(0)) / n
        out[key] = {"n": n, "improved": sum(e < a for e, a in zip(errors, initial)),
                    "harm": sum(e > a for e, a in zip(errors, initial)), "same": sum(e == a for e, a in zip(errors, initial)),
                    "moved": sum(row["status"] == "MOVE" for row in values),
                    "mae_mm": float(mae), "mse_mm2": float(mse), "initial_mae_mm": float(initial_mae),
                    "initial_mse_mm2": float(initial_mse), "max_error_mm": float(max(errors)),
                    "mae_reduction_percent": None if initial_mae == 0 else float(100 * (1 - mae / initial_mae)),
                    "mse_reduction_percent": None if initial_mse == 0 else float(100 * (1 - mse / initial_mse))}
    return out


def verify_summary_rows(actual_rows, derived_rows):
    expected = recompute_summary(derived_rows)
    actual = indexed(actual_rows, lambda r: (r["grouping"], r["state_group"], r["method"], tuple(r["condition"])),
                     "SUMMARY.rows")
    require(set(actual) == set(expected), "SUMMARY.rows: missing/extra registered groups")
    integer_fields = {"n", "improved", "harm", "same", "moved"}
    for key, values in expected.items():
        row = actual[key]
        require(type(row["condition"]) is list, f"SUMMARY {key}: condition must be list")
        require(set(row) == set(values) | {"grouping", "state_group", "method", "condition"},
                f"SUMMARY {key}: invalid summary schema")
        for field, value in values.items():
            if field in integer_fields:
                require(type(row[field]) is int, f"SUMMARY {key}.{field}: integer required")
            elif row[field] is not None:
                finite_number(row[field], f"SUMMARY {key}.{field}")
            require(row[field] == value, f"SUMMARY {key}: incorrect {field}; got {row[field]!r}, expected {value!r}")
    return expected


def shared_models():
    """Import only disclosed geometry kernels, never decision/aggregation code."""
    sys.path.insert(0, str(ROOT / "source_snapshot"))
    sys.path.insert(0, str(ROOT))
    from continuous_world.interval_model import Model
    from layered_model import LayeredModel
    return {(kind, radius): (Model(kind, radius) if kind in KINDS[:2] else LayeredModel(kind, radius))
            for kind in KINDS for radius in RADII}


def model_spot_checks(model, observation, truth_record, verified_outer, where, witness_samples):
    truth = exact(truth_record["truth"])
    radius = exact(observation["radius"])
    seed = truth_record["seed"]
    rng = random.Random(f"challenge-v1:{observation['kind']}:{radius}:{truth}:{seed}")
    expected_offsets = (radius * rng.choice(WEIGHTS), Q(0), radius * rng.choice(WEIGHTS))
    offsets = tuple(exact(v) for v in truth_record["actual_offsets"])
    require(offsets == expected_offsets, f"{where}: camera offsets do not match registered seed")
    prediction = model.point_prediction(truth, offsets)
    obs = tuple(exact(v) for v in observation["observation"])
    require(len(prediction) == len(obs), f"{where}: shared prediction dimension mismatch")
    noise = tuple(value - predicted for value, predicted in zip(obs, prediction))
    require(all(abs(value) <= EPSILON for value in noise), f"{where}: actual observation noise exceeds bound")
    require(noise == tuple(EPSILON * rng.choice(WEIGHTS) for _ in obs),
            f"{where}: observation noise does not match registered seed")
    require(exact(truth_record["noise_linf"]) == max(map(abs, noise)), f"{where}: false noise_linf")
    excluded = verified_outer["excluded"]
    # Small fixed sample; every recorded witness was separately checked above.
    selected = sorted(excluded, key=lambda n: hashlib.sha256(f"{where}:{n['id']}".encode()).digest())[:witness_samples]
    for node in selected:
        low, high = verified_outer["boxes"][node["id"]]
        enclosure = model.enclosure(low, high)
        require(len(enclosure) == len(obs), f"{where}: shared enclosure dimension mismatch")
        channel = node["witness_channel"]
        claimed = pair(node["prediction_interval"], where)
        recomputed = tuple(exact(v) for v in enclosure[channel])
        require(claimed == recomputed, f"{where}: forged exclusion interval at node {node['id']}")
    return len(selected)


def verify_bundle(bundle, *, models=None, witness_samples=2):
    require(type(witness_samples) is int and 1 <= witness_samples <= 8, "witness sample count must be 1..8")
    counts = {"observations": 135, "truth": 135, "outers": 135, "tasks": 1215, "decisions": 4860, "rows": 4860}
    for name, expected in counts.items():
        records = bundle[name]
        require(isinstance(records, list) and len(records) == expected,
                f"{name}: expected {expected} nonempty registered records, got {len(records) if isinstance(records, list) else 'invalid'}")
    observations = indexed(bundle["observations"], lambda r: r["observation_id"], "observations")
    truths = indexed(bundle["truth"], lambda r: r["observation_id"], "truth")
    outers = indexed(bundle["outers"], lambda r: r["observation_id"], "outers")
    tasks = indexed(bundle["tasks"], lambda r: r["task_id"], "tasks")
    decisions = indexed(bundle["decisions"], lambda r: (r["task_id"], r["method"]), "decisions")
    rows = indexed(bundle["rows"], lambda r: (r["task_id"], r["method"]), "ROWS")
    require(set(observations) == set(truths) == set(outers), "observation/truth/outer IDs differ")
    specs = [(kind, radius, truth, seed) for kind in KINDS for radius in RADII for truth in TRUTHS for seed in SEEDS]
    expected_task_ids = set()
    verified_outers = {}
    for index, (kind, radius, truth, seed) in enumerate(specs):
        oid = f"o{index:03d}"
        require(oid in observations, f"missing registered observation {oid}")
        obs, record = observations[oid], truths[oid]
        require(set(obs) == {"observation_id", "kind", "radius", "observation"}, f"{oid}: invalid observation schema")
        require(obs["kind"] == kind and obs["radius"] == str(radius), f"{oid}: wrong template/radius label")
        require(set(record) == {"observation_id", "truth", "seed", "actual_offsets", "noise_linf"}, f"{oid}: invalid truth schema")
        require(exact(record["truth"]) == truth and type(record["seed"]) is int and record["seed"] == seed,
                f"{oid}: wrong registered truth/seed")
        require(isinstance(obs["observation"], list) and len(obs["observation"]) == CHANNELS,
                f"{oid}: wrong observation channel count")
        observation = tuple(exact(v, oid) for v in obs["observation"])
        require(all(-EPSILON <= v <= 1 + EPSILON for v in observation), f"{oid}: observation outside noise-expanded radiance range")
        require(isinstance(record["actual_offsets"], list) and len(record["actual_offsets"]) == 3,
                f"{oid}: invalid actual offsets")
        offsets = tuple(exact(v) for v in record["actual_offsets"])
        require(offsets[1] == 0 and all(offsets[j] in {radius * v for v in WEIGHTS} for j in (0, 2)),
                f"{oid}: actual camera pose violates registered radius/gauge/grid")
        require(0 <= exact(record["noise_linf"]) <= EPSILON, f"{oid}: invalid noise_linf")
        require(set(outers[oid]) == {"observation_id", "outer"}, f"{oid}: invalid outer record schema")
        verified_outers[oid] = verify_outer(outers[oid]["outer"], observation, truth, oid)
        for j, off in enumerate(OFFSETS):
            tid = f"{oid}-s{j}"
            expected_task_ids.add(tid)
            require(tid in tasks and set(tasks[tid]) == {"task_id", "observation_id", "incumbent"}, f"missing/invalid task {tid}")
            require(tasks[tid]["observation_id"] == oid and exact(tasks[tid]["incumbent"]) == truth + off,
                    f"{tid}: wrong observation association/registered initial offset")
    require(set(tasks) == expected_task_ids, "tasks: extra/missing task IDs")
    expected_keys = {(tid, method) for tid in tasks for method in METHODS}
    require(set(decisions) == set(rows) == expected_keys, "decisions/ROWS: missing/extra task-method pairs")
    derived_rows = []
    for key in sorted(expected_keys):
        tid, method = key
        task, decision = tasks[tid], decisions[key]
        oid = task["observation_id"]
        truth = exact(truths[oid]["truth"])
        a = exact(task["incumbent"])
        b = verify_decision(decision, a, verified_outers[oid]["hull"], f"{tid}/{method}")
        expected = expected_row(task, observations[oid], truth, decision, b)
        verify_row(rows[key], expected, f"ROWS {tid}/{method}")
        if method in CERTIFICATES:
            actual_gain = (a - truth) ** 2 - (b - truth) ** 2
            require(actual_gain >= 0, f"{tid}/{method}: observed certificate harm")
            require(actual_gain >= exact(decision["gain_lower"]), f"{tid}/{method}: actual gain below certificate")
            if observations[oid]["kind"] == "layered_hidden" and DOMAIN[0] <= a <= DOMAIN[1]:
                require(decision["status"] == "KEEP", f"{tid}/{method}: hidden inside-prior certificate must KEEP")
        derived_rows.append(expected)
    midpoint_moves = sum(d["status"] == "MOVE" for (tid, method), d in decisions.items() if method == CERTIFICATES[0])
    projection_moves = sum(d["status"] == "MOVE" for (tid, method), d in decisions.items() if method == CERTIFICATES[1])
    require(projection_moves >= midpoint_moves, "projection accepts fewer moves than midpoint")
    for tid in tasks:
        require(decisions[(tid, CERTIFICATES[0])]["status"] != "MOVE" or decisions[(tid, CERTIFICATES[1])]["status"] == "MOVE",
                f"{tid}: projection fails to accept a midpoint-safe task")
    summary = bundle["summary"]
    require(isinstance(summary, dict), "SUMMARY must be object")
    top_expected = {"observation_configurations": 135, "template_count": 5, "task_count": 1215, "decision_count": 4860,
                    "evaluated_boxes": sum(v["evaluated"] for v in verified_outers.values()),
                    "unresolved_budget_leaves": sum(v["budget_leaves"] for v in verified_outers.values()), "incompatible": 0}
    require(set(summary) == set(top_expected) | {"rows", "elapsed_seconds"}, "SUMMARY: incorrect top-level schema")
    for field, value in top_expected.items():
        require(type(summary[field]) is int and summary[field] == value, f"SUMMARY: incorrect {field}")
    require(finite_number(summary["elapsed_seconds"], "SUMMARY.elapsed_seconds") >= 0, "SUMMARY: negative elapsed time")
    aggregates = verify_summary_rows(summary["rows"], derived_rows)
    if models is None:
        models = shared_models()
    sampled = sum(model_spot_checks(models[(obs["kind"], exact(obs["radius"]))], obs, truths[oid],
                                   verified_outers[oid], oid, witness_samples) for oid, obs in sorted(observations.items()))
    totals = {}
    for method in METHODS:
        selected = [row for row in derived_rows if row["method"] == method]
        errors = [exact(row["ae_mm"]) for row in selected]
        initial = [exact(row["initial_ae_mm"]) for row in selected]
        totals[method] = {"n": len(selected), "improved": sum(e < a for e, a in zip(errors, initial)),
                          "harm": sum(e > a for e, a in zip(errors, initial)), "same": sum(e == a for e, a in zip(errors, initial)),
                          "moved": sum(row["status"] == "MOVE" for row in selected),
                          "mae_mm_exact": str(sum(errors, Q(0)) / len(selected)),
                          "mse_mm2_exact": str(sum((e * e for e in errors), Q(0)) / len(selected)),
                          "max_error_mm_exact": str(max(errors))}
    partial_repairs = {kind: sum(row["kind"] == kind and row["method"] == CERTIFICATES[1] and
                                row["prior_position"] == "inside_prior" and exact(row["ae_mm"]) < exact(row["initial_ae_mm"])
                                for row in derived_rows) for kind in KINDS[2:4]}
    return {"counts": top_expected, "summary_groups_checked": len(aggregates), "method_totals": totals,
            "endpoint_observations": {"projection_accepts_at_least_midpoint": True,
                                      "projection_inside_prior_repairs_partial": partial_repairs},
            "model_checks": {"truth_images_recomputed": 135, "recorded_exclusion_witnesses_checked":
                             sum(len(v["excluded"]) for v in verified_outers.values()),
                             "shared_enclosure_witnesses_recomputed": sampled,
                             "witnesses_per_observation_cap": witness_samples}}


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, f"JSON object contains duplicate key {key!r}")
        out[key] = value
    return out


def load_bundle(output):
    bundle, hashes = {}, {}
    for name, relative in REQUIRED.items():
        raw = (output / relative).read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        bundle[name] = json.loads(raw, object_pairs_hook=unique_object,
                                  parse_constant=lambda value: (_ for _ in ()).throw(VerificationError(f"nonfinite JSON {value}")))
    return bundle, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Existing experiment output directory to audit")
    parser.add_argument("--witnesses-per-observation", type=int, default=2, help="Shared-model exclusion checks per observation (1..8; default 2)")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_dir():
        parser.error(f"existing output directory required: {output}")
    report_path = output / "verification" / "VERIFY.json"
    if report_path.exists():
        parser.error(f"refusing to overwrite existing verification: {report_path}")
    started = time.perf_counter()
    report = {"schema": "independent-challenge-verification-v1", "status": "FAIL", "output_directory": str(output),
              "host": socket.gethostname(), "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_source_root": str(ROOT), "independent_arithmetic": True,
              "limitations": ["Image predictions and sampled enclosures reuse the disclosed source geometry kernel; its mathematical contract is not independently reproved.",
                              "Every recorded exclusion witness is checked for noise conflict, but only a bounded sample is recomputed from model.enclosure.",
                              "Residual outputs, grid membership, tie consistency and all geometric outcomes are checked; residual grid optimality is not rerun.",
                              "Recorded timing is validated as a finite nonnegative value, not independently benchmarked."]}
    try:
        bundle, hashes = load_bundle(output)
        report["input_sha256"] = hashes
        report.update(verify_bundle(bundle, witness_samples=args.witnesses_per_observation))
        report["status"] = "PASS"
        report["checks"] = ["complete registered population and unique task-method associations",
                            "all exact per-decision errors, endpoint gains, midpoint and clamp actions",
                            "all state, offset, prior, template and camera-radius labels",
                            "all trace trees, exact bisections and complete terminal-domain partitions",
                            "truth in retained leaves and reconstructed hull for all 135 observations",
                            "all exclusion witness/noise conflicts and sampled shared-model witness integrity",
                            "all seeded actual poses and all 135 actual bounded-noise truth images",
                            "all six summary grouping families and exact-derived float metrics",
                            "both certificate arms have zero observed harm; projection acceptance contains midpoint",
                            "hidden inside-prior certificate decisions all KEEP"]
    except (VerificationError, OSError, KeyError, TypeError, ValueError, ImportError, ArithmeticError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["verification_seconds"] = time.perf_counter() - started
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation handles a concurrent run without overwriting its report.
    try:
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        parser.error(f"refusing to overwrite concurrent verification: {report_path}")
    print(json.dumps({"status": report["status"], "report": str(report_path), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
