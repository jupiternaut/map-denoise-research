"""Deterministic, read-only artifact rechecker; not a scientific audit.

The author also implemented new_fixtures.py. This checker independently
recomputes recorded metrics but does not claim independent scientific authorship.
It imports neither experiment.py nor decisions.py, generates no observations,
and writes only an explicitly requested, previously nonexistent report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import socket
import sys

import numpy as np


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/srv/slam-research/grf/map-denoise/runs/decision-interface-20261008T052305Z")
OLD = ROOT.parent / "mixed-pixel-20261008T041249Z"
MECHANISMS = {"flat_contrast", "flat_equal", "textured_boundary", "textured_single"}
STAGE_SEEDS = {"calibration": {31001, 31002, 31003}, "confirmation": {41001, 41002, 41003}}
REJECT_REASONS = {"raw_invalid", "nonfinite_curve", "flat_curve", "scale_unavailable", "threshold_empty"}
STRICT_TOL = 1e-9


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_file(value, base):
    """Resolve the declared target only; never search for substitute files."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(base)/path).resolve()


def numeric(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite metric input")
    return value


def identity(row):
    return str(row["id"]), str(row["arm"]), numeric(row["initial_depth"])


def index_unique(rows, key):
    result = {}
    for row in rows:
        label = key(row)
        if label in result:
            raise ValueError(f"duplicate record key: {label}")
        result[label] = row
    return result


def compare_values(actual, expected, location="root"):
    """Structural comparison; counts exact, aggregates only roundoff tolerant."""
    differences = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{location}: expected dictionary"]
        if set(actual) != set(expected):
            differences.append(f"{location}: field mismatch actual={sorted(actual)} expected={sorted(expected)}")
        for key in sorted(set(actual) & set(expected)):
            differences.extend(compare_values(actual[key], expected[key], f"{location}.{key}"))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [f"{location}: list length/type mismatch"]
        for i, (left, right) in enumerate(zip(actual, expected)):
            differences.extend(compare_values(left, right, f"{location}[{i}]"))
    elif isinstance(expected, bool) or expected is None or isinstance(expected, (str, int)):
        if actual != expected or (isinstance(expected, bool) and not isinstance(actual, bool)):
            differences.append(f"{location}: {actual!r} != {expected!r}")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)) or not math.isclose(
                float(actual), expected, rel_tol=1e-12, abs_tol=1e-10):
            differences.append(f"{location}: {actual!r} != {expected!r}")
    else:
        raise TypeError(f"unsupported expected value {type(expected)}")
    return differences


def aggregate(rows):
    """Recalculate counts/means with Python math; no experiment implementation."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["initial_kind"])].append(row)
    result = []
    for (arm, initial_kind), group in sorted(groups.items()):
        count = len(group)
        result.append({
            "arm": arm, "initial_kind": initial_kind, "n": count,
            "mae": math.fsum(row["error"] for row in group)/count,
            "mse": math.fsum(row["error"]**2 for row in group)/count,
            "improved": sum(row["change"] < -STRICT_TOL for row in group),
            "harmed": sum(row["change"] > STRICT_TOL for row in group),
            "unchanged": sum(abs(row["change"]) <= STRICT_TOL for row in group),
            "practical_harm": sum(row["change"] > 7.5 for row in group),
            "move": sum(row["move"] for row in group),
            "rejected": sum(row["reason"] in REJECT_REASONS for row in group),
            "regret": math.fsum(row["regret"] for row in group)/count,
        })
    return result


class Rechecker:
    def __init__(self):
        self.checks = []
        self.pending = []
        self.counts = {}
        self._hash_cache = {}
        self.datasets = {}

    def check(self, name, passed, details=None):
        record = {"check": name, "passed": bool(passed)}
        if details is not None:
            record["details"] = details
        self.checks.append(record)
        return bool(passed)

    def section(self, name, function):
        try:
            function()
        except Exception as exc:
            self.check(name, False, {"exception": type(exc).__name__, "message": str(exc)})

    def hash(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
        if cache_key not in self._hash_cache:
            self._hash_cache[cache_key] = digest(path)
        return self._hash_cache[cache_key]

    def hash_manifest(self, name, records, base):
        bad = []
        for target, expected in sorted(records.items()):
            path = resolve_file(target, base)
            try:
                actual = self.hash(path)
                if actual != expected:
                    bad.append({"path": str(path), "expected": expected, "actual": actual})
            except OSError as exc:
                bad.append({"path": str(path), "error": str(exc)})
        self.check(name, not bad, {"file_count": len(records), "mismatches": bad})

    def integrity(self):
        path = ROOT/"SOURCE_LOCK.json"
        if not path.is_file():
            self.pending.append("SOURCE_LOCK.json is not available")
            return
        lock = load_json(path)
        self.check("source_lock_nonempty", bool(lock.get("files")) and bool(lock.get("history")))
        self.hash_manifest("source_lock_sources_unchanged", lock["files"], ROOT)
        self.hash_manifest("source_lock_historical_files_unchanged", lock["history"], ROOT)
        seals = set(ROOT.rglob("SEAL.json"))
        seals.update(resolve_file(item, ROOT) for item in lock["history"] if Path(item).name == "SEAL.json")
        self.counts["seal_count"] = len(seals)
        for seal in sorted(seals):
            self.section(f"seal_exception:{seal}", lambda seal=seal: self.verify_seal(seal))

    def verify_seal(self, seal):
        payload = load_json(seal)
        if isinstance(payload.get("files"), dict):
            schema = "file_map"
            records = dict(payload["files"])
        elif "observations_sha256" in payload and isinstance(payload.get("curves"), dict):
            # footprint-support stage producer writes OBSERVATIONS.json plus
            # its curve map under these fields, not under a `files` key.
            schema = "observations_sha256_and_curves"
            records = dict(payload["curves"])
            records[str(seal.parent/"OBSERVATIONS.json")] = payload["observations_sha256"]
        elif "decisions_sha256" in payload and isinstance(payload.get("observation_seals"), dict):
            schema = "decisions_sha256_and_observation_seals"
            records = {str(seal.parent/"DECISIONS.json"): payload["decisions_sha256"]}
        elif "sha256" in payload and isinstance(payload.get("observation_seals"), dict):
            schema = "predictions_sha256_and_observation_seals"
            records = {str(seal.parent/"PREDICTIONS.json"): payload["sha256"]}
        else:
            # An unknown schema means this check is incomplete, not that the
            # bytes were corrupted. SOURCE_LOCK.history is checked separately.
            self.pending.append(f"unsupported seal content schema: {seal}")
            return
        if "observation_seals" in payload:
            for stage, expected in payload["observation_seals"].items():
                if Path(stage).name != stage:
                    raise ValueError(f"unexpected observation-stage reference: {stage}")
                records[str(seal.parent.parent/stage/"SEAL.json")] = expected
        self.check(f"seal_schema_supported:{seal}", bool(records),
                   {"schema": schema, "referenced_files": len(records)})
        self.hash_manifest(f"seal_integrity:{seal}", records, seal.parent)

    def dataset(self, stage, data_root, expected_count, is_new):
        observations = load_json(data_root/"observed/inputs.json")
        truth = load_json(data_root/"truth/metadata.json")
        om = index_unique(observations, lambda row: str(row["id"]))
        tm = index_unique(truth, lambda row: str(row["id"]))
        self.check(f"{stage}:object_cardinality", len(om) == expected_count and len(tm) == expected_count,
                   {"observations": len(om), "truth": len(tm), "expected": expected_count})
        self.check(f"{stage}:observation_truth_ids", set(om) == set(tm))
        image_errors, tensor_hashes = [], {}
        for identifier, row in om.items():
            if is_new and set(row) != {"id", "cameras", "image_file", "sha256"}:
                image_errors.append({"id": identifier, "reason": "unexpected observation fields"})
            path = resolve_file(row["image_file"], data_root)
            if self.hash(path) != row["sha256"]:
                image_errors.append({"id": identifier, "reason": "image sha256 differs"})
            with np.load(path, allow_pickle=False) as archive:
                if is_new and archive.files != ["images"]:
                    image_errors.append({"id": identifier, "reason": "NPZ contains fields besides images"})
                image = archive["images"]
                if image.shape != (3, 128, 128) or image.dtype != np.float64 or not np.isfinite(image).all():
                    image_errors.append({"id": identifier, "reason": "image array shape/dtype/finiteness"})
                tensor_hashes[identifier] = hashlib.sha256(image.tobytes()).hexdigest()
                if is_new and tm[identifier]["mechanism"] == "flat_equal" and not np.all(image == image.flat[0]):
                    image_errors.append({"id": identifier, "reason": "equal-color control is not exactly constant"})
        self.check(f"{stage}:observations_and_image_integrity", not image_errors, image_errors)
        self.datasets[stage] = {"observations": om, "truth": tm, "tensor_hashes": tensor_hashes}
        if not is_new:
            return
        counts = Counter(row["mechanism"] for row in truth)
        self.check(f"{stage}:mechanism_cardinality", counts == Counter({name: 12 for name in MECHANISMS}), dict(counts))
        seed_counts = Counter(row["producer_parameters"]["batch_seed"] for row in truth)
        self.check(f"{stage}:registered_batch_seeds", seed_counts == Counter({seed: 16 for seed in STAGE_SEEDS[stage]}),
                   {str(key): count for key, count in sorted(seed_counts.items())})
        grid_counts, errors = Counter(), []
        expected_addresses = {(seed, mechanism, j) for seed in STAGE_SEEDS[stage] for mechanism in MECHANISMS for j in range(4)}
        addresses = []
        equal_levels = []
        for row in truth:
            params, depth = row["producer_parameters"], numeric(row["true_depth"])
            address = (params["batch_seed"], row["mechanism"], params["within_batch_index"])
            addresses.append(address)
            on_grid = abs((depth-450.)/15.-round((depth-450.)/15.)) <= 1e-9
            grid_counts[(row["mechanism"], on_grid)] += 1
            if not (520. <= depth <= 690.) or on_grid != bool(params["on_action_grid"]):
                errors.append({"id": row["id"], "reason": "depth range or grid classification"})
            if params["stage"] != stage or params["mechanism"] != row["mechanism"] or numeric(params["true_depth"]) != depth:
                errors.append({"id": row["id"], "reason": "producer metadata disagreement"})
            if row["mechanism"] in {"flat_contrast", "textured_boundary"}:
                if not (1.2 <= params["half_width"] <= 3.2 and params["background_depth"]-depth >= 150.):
                    errors.append({"id": row["id"], "reason": "width/background-gap contract"})
            if row["mechanism"] == "flat_equal":
                equal_levels.append(numeric(params["equal_level"]))
        self.check(f"{stage}:producer_addresses", len(addresses) == len(set(addresses)) and set(addresses) == expected_addresses)
        self.check(f"{stage}:depth_geometry_metadata", not errors, errors)
        self.check(f"{stage}:half_grid_half_offgrid", all(grid_counts[(name, status)] == 6 for name in MECHANISMS for status in (False, True)),
                   {f"{name}:{'on' if status else 'off'}": count for (name, status), count in sorted(grid_counts.items())})
        self.check(f"{stage}:distinct_equal_color_objects", len(equal_levels) == 12 and len(set(equal_levels)) == 12)
        self.counts[stage] = {"objects": len(om), "mechanisms": dict(counts), "batch_seeds": sorted(seed_counts)}

    def stages(self):
        for stage in ("calibration", "confirmation"):
            data = ROOT/stage/"data"
            if (data/"observed/inputs.json").is_file():
                self.section(f"{stage}:dataset_exception", lambda stage=stage, data=data: self.dataset(stage, data, 48, True))
            elif stage == "calibration":
                self.pending.append("calibration observations are not available")
        if "calibration" in self.datasets and "confirmation" in self.datasets:
            seeds = {stage: {row["producer_parameters"]["batch_seed"] for row in self.datasets[stage]["truth"].values()}
                     for stage in ("calibration", "confirmation")}
            self.check("calibration_confirmation_seed_disjointness", seeds["calibration"].isdisjoint(seeds["confirmation"]))
            tensor_sets = {stage: set(self.datasets[stage]["tensor_hashes"].values()) for stage in seeds}
            self.check("calibration_confirmation_observation_disjointness", tensor_sets["calibration"].isdisjoint(tensor_sets["confirmation"]))
        gate_file = ROOT/"replay/evaluation/GATE.json"
        if gate_file.is_file():
            gate = load_json(gate_file)
            if not gate["passed"]:
                self.check("failed_B2_forbids_confirmation_directory", not (ROOT/"confirmation").exists())
            elif not (ROOT/"confirmation/evaluation/SUMMARY.json").is_file():
                self.pending.append("B2 passed but confirmation evaluation is not available")
        else:
            self.check("confirmation_requires_recorded_B2_gate", not (ROOT/"confirmation").exists())
            self.pending.append("replay B2 gate is not available")

    def derive_rows(self, stage, decisions):
        truth = self.datasets[stage]["truth"]
        tensors = self.datasets[stage]["tensor_hashes"]
        result, errors = [], []
        for row in decisions:
            depth = numeric(truth[row["id"]]["true_depth"])
            initial, selected = numeric(row["initial_depth"]), numeric(row["selected_depth"])
            initial_error, error = abs(initial-depth), abs(selected-depth)
            move = selected != initial
            if bool(row["move"]) != move:
                errors.append({"key": identity(row), "reason": "move flag differs from selected!=initial"})
            if selected not in [numeric(value) for value in row["candidates"]] and selected != initial:
                errors.append({"key": identity(row), "reason": "selection outside candidate actions and KEEP"})
            derived = dict(row)
            derived.update({"true_depth": depth, "initial_error": initial_error, "error": error,
                            "change": error-initial_error, "move": move,
                            "mechanism": truth[row["id"]]["mechanism"], "tensor": tensors[row["id"]],
                            "initial_kind": "correct" if initial_error < STRICT_TOL else ("minus" if initial < depth else "plus"),
                            "regret": error-min(abs(numeric(value)-depth) for value in row["candidates"])})
            result.append(derived)
        self.check(f"{stage}:decision_action_consistency", not errors, errors)
        return result

    def resolve_prediction_inputs(self, stage, decisions):
        observation_path = OLD/"ordinary_stage/OBSERVATIONS.json" if stage == "replay" else ROOT/"confirmation/scores/OBSERVATIONS.json"
        observations = load_json(observation_path)["rows"]
        errors = []
        if stage == "confirmation":
            tasks = index_unique(load_json(ROOT/"confirmation/TASKS.json"), lambda row: str(row["task_id"]))
        else:
            tasks = {}
        for decision in decisions:
            arm = decision["arm"].split("_")[0]
            if arm == "KEEP":
                arm = "N"
            matches = [row for row in observations if row["id"] == decision["id"] and row["arm"] == arm and
                       (stage != "confirmation" or row.get("task_id") == decision.get("task_id")) and
                       (row.get("incumbent") is None or numeric(row["incumbent"]) == numeric(decision["initial_depth"]))]
            if len(matches) != 1:
                errors.append({"key": identity(decision), "reason": "nonunique source observation", "matches": len(matches)})
                continue
            source = matches[0]
            if decision["arm"] != "KEEP" and resolve_file(decision["curve_file"], ROOT) != resolve_file(source["curve_file"], observation_path.parent):
                errors.append({"key": identity(decision), "reason": "curve target differs from observation"})
            if stage == "confirmation":
                task = tasks[str(decision["task_id"])]
                if task["id"] != decision["id"] or numeric(task["incumbent"]) != numeric(decision["initial_depth"]) or task["candidates"] != decision["candidates"]:
                    errors.append({"key": identity(decision), "reason": "task identity/candidates differ"})
        self.check(f"{stage}:prediction_inputs_resolved", not errors, {"decision_count": len(decisions), "errors": errors})

    def evaluate(self, stage):
        prediction_path = ROOT/stage/"decisions/PREDICTIONS.json"
        summary_path = ROOT/stage/"evaluation/SUMMARY.json"
        if not prediction_path.is_file() or not summary_path.is_file():
            if stage == "replay":
                self.pending.append("replay predictions/summary are not available")
            return
        if stage == "replay":
            self.dataset("replay", OLD/"data", 36, False)
        decisions = load_json(prediction_path)
        dm = index_unique(decisions, identity)
        self.resolve_prediction_inputs(stage, decisions)
        rows = self.derive_rows(stage, decisions)
        expected_rows = 108 if stage == "replay" else 144
        counts = Counter(row["arm"] for row in rows)
        expected_arms = {"KEEP", "N_P", "N_M", "N_Mraw", "N_R"}
        if stage == "replay":
            expected_arms |= {f"{arm}_{scale}_{rule}" for arm in ("ED", "EF") for scale in ("S0", "S1") for rule in ("P", "M", "R")}
            expected_arms |= {f"{arm}_{rule}" for arm in ("ED", "EF") for rule in ("S1_U", "Mraw")}
        else:
            expected_arms |= {"ED_S0_P", "ED_S1_M", "ED_S1_R", "ED_Mraw", "EF_S1_M", "EF_S1_R"}
        self.check(f"{stage}:arm_and_decision_cardinality", set(counts) == expected_arms and all(count == expected_rows for count in counts.values()), dict(counts))
        seen, unique = set(), []
        for row in rows:
            key = (row["tensor"], row["arm"], row["initial_depth"])
            if key not in seen:
                unique.append(row)
                seen.add(key)
        expected = {"overall": aggregate(rows),
                    "by_mechanism": {name: aggregate([row for row in rows if row["mechanism"] == name]) for name in sorted({row["mechanism"] for row in rows})},
                    "unique_tensors": aggregate(unique),
                    "distinct_tensors": len(set(self.datasets[stage]["tensor_hashes"].values()))}
        differences = compare_values(load_json(summary_path), expected)
        self.check(f"{stage}:summary_metrics_recomputed", not differences, differences)
        stored_rows = index_unique(load_json(ROOT/stage/"evaluation/ROWS.json"), identity)
        self.check(f"{stage}:evaluated_row_keys", set(stored_rows) == set(dm))
        row_differences = []
        columns = ("true_depth", "error", "initial_error", "change", "mechanism", "tensor", "initial_kind", "regret", "selected_depth", "move")
        for row in rows:
            recorded = stored_rows.get(identity(row), {})
            row_differences.extend(compare_values({key: recorded.get(key) for key in columns},
                                                  {key: row[key] for key in columns}, str(identity(row))))
        self.check(f"{stage}:per_row_errors_recomputed", not row_differences, row_differences)
        self.recompute_gate(stage, rows)
        if stage == "replay":
            self.old_predictions(decisions)
        self.counts[f"{stage}_metrics"] = {"decisions": len(rows), "arms": len(counts), "distinct_tensors": expected["distinct_tensors"]}

    def old_predictions(self, decisions):
        old_rows = index_unique(load_json(OLD/"decisions/PREDICTIONS.json"), identity)
        mapped = {"N_P": "N", "ED_S0_P": "ED", "EF_S0_P": "EF"}
        mismatches, compared = [], 0
        for row in decisions:
            if row["arm"] not in mapped:
                continue
            key = (str(row["id"]), mapped[row["arm"]], numeric(row["initial_depth"]))
            previous = old_rows[key]
            compared += 1
            # Deliberately exact: these historical chosen actions must not drift.
            if row["selected_depth"] != previous["selected_depth"]:
                mismatches.append({"key": key, "historical": previous["selected_depth"], "replay": row["selected_depth"]})
            if resolve_file(row["curve_file"], ROOT) != resolve_file(previous["curve_file"], OLD):
                mismatches.append({"key": key, "reason": "historical curve identity differs"})
        self.check("replay:historical_S0P_and_NP_exact", compared == 324 and not mismatches,
                   {"compared": compared, "expected": 324, "mismatches": mismatches})

    def recompute_gate(self, stage, rows):
        main = [row for row in rows if row["arm"] == "ED_S1_M"]
        by_key = {(row["id"], row["initial_depth"]): row for row in main}
        successes = [row for row in rows if row["arm"] == "N_P" and row["error"] <= STRICT_TOL and row["initial_error"] > STRICT_TOL]
        lost = [row for row in successes if by_key[(row["id"], row["initial_depth"])]["error"] > STRICT_TOL]
        directional = {kind: [row["change"] for row in main if row["initial_kind"] == kind] for kind in ("minus", "plus")}
        checks = {"correct_no_harm": all(row["error"] <= STRICT_TOL for row in main if row["initial_kind"] == "correct"),
                  "minus_better_keep": bool(directional["minus"]) and math.fsum(directional["minus"])/len(directional["minus"]) < 0.,
                  "plus_better_keep": bool(directional["plus"]) and math.fsum(directional["plus"])/len(directional["plus"]) < 0.,
                  "retain_full9_success": not lost,
                  "equal_no_move": not any(row["move"] for row in main if row["mechanism"] == "flat_equal")}
        if stage == "confirmation":
            matched = [row for row in rows if row["arm"] == "N_M" and row["error"] <= STRICT_TOL and row["initial_error"] > STRICT_TOL]
            checks["retain_matched_full9_M_success"] = all(by_key[(row["id"], row["initial_depth"])]["error"] <= STRICT_TOL for row in matched)
        expected = {"passed": all(checks.values()), "checks": checks, "full9_successes": len(successes),
                    "lost_full9_successes": len(lost),
                    "lost_rows": [{"id": row["id"], "initial_depth": row["initial_depth"]} for row in lost]}
        differences = compare_values(load_json(ROOT/stage/"evaluation/GATE.json"), expected)
        self.check(f"{stage}:gate_recomputed", not differences, differences)

    def run(self):
        target_ok = self.check("target_identity", socket.gethostname() == "liekkas" and ROOT == EXPECTED_ROOT,
                               {"hostname": socket.gethostname(), "root": str(ROOT)})
        if target_ok:
            self.section("integrity_exception", self.integrity)
            self.section("stage_contract_exception", self.stages)
            for stage in ("replay", "confirmation"):
                self.section(f"{stage}:evaluation_exception", lambda stage=stage: self.evaluate(stage))
        failed = [record["check"] for record in self.checks if not record["passed"]]
        status = "failed" if failed else ("incomplete" if self.pending else "passed")
        return {"checker": "deterministic artifact and metric rechecker",
                "independent_scientific_auditor": False,
                "authorship_limit": "This checker author also wrote the new fixture producer; metric formulas are independently implemented here.",
                "source_file": str(Path(__file__).resolve()), "source_sha256": digest(__file__),
                "root": str(ROOT), "status": status, "passed": status == "passed",
                "failed_checks": failed, "pending": sorted(set(self.pending)),
                "counts": self.counts, "checks": self.checks,
                "scope_limit": "Checks byte integrity, identity, deterministic records and arithmetic, not scientific validity or a security sandbox."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new report path inside this run's audit directory; never overwritten")
    args = parser.parse_args()
    output = None if args.output is None else args.output.resolve()
    if output is not None:
        if not output.is_relative_to(ROOT/"audit"):
            parser.error("report output must be inside this run's audit directory")
        if output.exists():
            parser.error(f"refusing to overwrite {output}")
    report = Rechecker().run()
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(serialized, end="")
    return 0 if report["status"] == "passed" else (2 if report["status"] == "incomplete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
