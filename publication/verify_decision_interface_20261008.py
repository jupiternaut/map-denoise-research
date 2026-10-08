#!/usr/bin/env python3
"""Read-only, standard-library verification of the decision-interface publication.

Run from any directory: python /path/to/publication/verify_decision_interface_20261008.py
Only the publication manifest and files in its archive are opened. Historical
absolute paths are provenance strings, never filesystem fallbacks. No archived
algorithm is imported. A correctly recorded scientific FAIL is verification PASS.
This checks published evidence, not a rerun of rendering, scoring or inference.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys


ARCHIVE = "research_snapshots/2026-10-08/decision-interface-20261008T052305Z"
SOURCE = "/srv/slam-research/grf/map-denoise/runs/decision-interface-20261008T052305Z"
MANIFEST = "publication/DECISION_INTERFACE_20261008_MANIFEST.json"
TOL = 1e-9
KINDS = {"correct", "minus", "plus"}
CONFIRMATION_ARMS = {
    "ED_Mraw", "ED_S0_P", "ED_S1_M", "ED_S1_R", "EF_S1_M", "EF_S1_R",
    "KEEP", "N_M", "N_Mraw", "N_P", "N_R",
}
REPLAY_ARMS = CONFIRMATION_ARMS | {
    "ED_S0_M", "ED_S0_R", "ED_S1_P", "ED_S1_U", "EF_Mraw",
    "EF_S0_M", "EF_S0_P", "EF_S0_R", "EF_S1_P", "EF_S1_U",
}
REJECT_REASONS = {
    "raw_invalid", "nonfinite_curve", "flat_curve", "scale_unavailable", "threshold_empty",
}
EVALUATION_FIELDS = {
    "true_depth", "error", "initial_error", "change", "mechanism", "tensor",
    "initial_kind", "regret",
}


class VerificationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def load(path):
    def reject(value):
        raise VerificationError("Nonstandard JSON number: " + value)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def same(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + ": keys differ")
        for key, value in expected.items():
            same(actual[key], value, label + "/" + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ": lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, label + "/" + str(index))
    elif isinstance(expected, bool) or expected is None:
        require(actual is expected, label + ": value differs")
    elif isinstance(expected, int):
        require(type(actual) is int and actual == expected, label + ": integer differs")
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
                f"{label}: {actual!r} != {expected!r}")
    else:
        require(actual == expected, label + ": value differs")


def check_manifest(root):
    manifest = load(root / MANIFEST)
    require(manifest["archive_roots"] == [ARCHIVE], "Unexpected archive root")
    require(manifest.get("source_host") == "liekkas", "Unexpected source host")
    require(isinstance(manifest["excluded"], list), "Missing exclusion inventory")
    declared, sources = {}, set()
    for record in manifest["files"]:
        relative = record["path"]
        path = PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in relative
                and str(path) == relative and relative.startswith(ARCHIVE + "/"),
                "Unsafe archive path: " + relative)
        require(relative not in declared, "Duplicate archive path: " + relative)
        require(record["source"] not in sources, "Duplicate source path")
        require(PurePosixPath(record["source"]).is_absolute(), "Source key must be absolute")
        require(type(record["bytes"]) is int and record["bytes"] >= 0, "Invalid byte count")
        require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None, "Invalid SHA-256")
        renamed = path.name == "SOURCE_AGENTS.md"
        source_relative = path.relative_to(PurePosixPath(ARCHIVE))
        if renamed:
            source_relative = source_relative.with_name("AGENTS.md")
        require(record["source"] == SOURCE + "/" + str(source_relative),
                "Unexpected source identity: " + relative)
        require(record["transformation"] == ("filename only: archived instructions" if renamed else "none"),
                "Unexpected transformation: " + relative)
        require(path.name != "AGENTS.md" and ".aris" not in path.parts
                and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"),
                "Excluded file in archive: " + relative)
        declared[relative] = record
        sources.add(record["source"])
    archive = root / ARCHIVE
    require(archive.is_dir() and not archive.is_symlink(), "Missing/linked archive")
    require(archive.resolve() == archive, "Linked archive ancestor")
    actual = set()
    for path in archive.rglob("*"):
        require(not path.is_symlink(), "Symlink in archive: " + str(path))
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        else:
            require(path.is_dir(), "Nonregular object: " + str(path))
    require(actual == declared.keys(),
            f"Archive path-set mismatch: missing={sorted(declared.keys() - actual)}, "
            f"extra={sorted(actual - declared.keys())}")
    for relative, record in declared.items():
        path = root / relative
        require(path.stat().st_size == record["bytes"], "Size mismatch: " + relative)
        require(digest(path) == record["sha256"], "SHA-256 mismatch: " + relative)
    return {"files": len(declared), "bytes": sum(row["bytes"] for row in declared.values())}


def index(rows, key, label):
    result = {}
    for row in rows:
        identity = key(row)
        require(identity not in result, f"{label}: duplicate {identity}")
        result[identity] = row
    return result


def row_key(row):
    return row["id"], row["arm"], row["initial_depth"]


def recompute_row(row):
    for field in ("initial_depth", "selected_depth", "true_depth"):
        require(type(row[field]) in (int, float) and math.isfinite(row[field]), "Invalid " + field)
    candidates = row["candidates"]
    require(candidates and all(type(x) in (int, float) and math.isfinite(x) for x in candidates),
            "Invalid candidates")
    initial, selected, truth = row["initial_depth"], row["selected_depth"], row["true_depth"]
    require(selected in candidates and initial in candidates, "Decision outside candidate set")
    error, initial_error = abs(selected - truth), abs(initial - truth)
    result = dict(row, error=error, initial_error=initial_error, change=error - initial_error,
                  initial_kind="correct" if initial_error < TOL else ("minus" if initial < truth else "plus"),
                  regret=error - min(abs(x - truth) for x in candidates), move=abs(selected - initial) > TOL)
    for field in ("error", "initial_error", "change", "initial_kind", "regret", "move"):
        same(row[field], result[field], f"row {row_key(row)}/{field}")
    if row["arm"] == "KEEP":
        require(initial == selected and row["reason"] == "identity", "KEEP is not identity")
    return result


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["initial_kind"])].append(row)
    return [dict(
        arm=arm, initial_kind=kind, n=len(group),
        mae=math.fsum(r["error"] for r in group) / len(group),
        mse=math.fsum(r["error"] ** 2 for r in group) / len(group),
        improved=sum(r["change"] < -TOL for r in group),
        harmed=sum(r["change"] > TOL for r in group),
        unchanged=sum(abs(r["change"]) <= TOL for r in group),
        practical_harm=sum(r["change"] > 7.5 for r in group),
        move=sum(r["move"] for r in group),
        rejected=sum(r["reason"] in REJECT_REASONS for r in group),
        regret=math.fsum(r["regret"] for r in group) / len(group),
    ) for (arm, kind), group in sorted(groups.items())]


def strict_gate(rows, stage):
    main = {(r["id"], r["initial_depth"]): r for r in rows if r["arm"] == "ED_S1_M"}
    successes = [r for r in rows if r["arm"] == "N_P" and r["error"] <= TOL and r["initial_error"] > TOL]
    lost = [r for r in successes if main[(r["id"], r["initial_depth"])]["error"] > TOL]
    checks = dict(
        correct_no_harm=all(r["error"] <= TOL for r in main.values() if r["initial_kind"] == "correct"),
        minus_better_keep=math.fsum(r["change"] for r in main.values() if r["initial_kind"] == "minus") < 0,
        plus_better_keep=math.fsum(r["change"] for r in main.values() if r["initial_kind"] == "plus") < 0,
        retain_full9_success=not lost,
        equal_no_move=not any(r["move"] for r in main.values() if r["mechanism"] == "flat_equal"),
    )
    if stage == "confirmation":
        checks["retain_matched_full9_M_success"] = all(
            main[(r["id"], r["initial_depth"])]["error"] <= TOL for r in rows
            if r["arm"] == "N_M" and r["error"] <= TOL and r["initial_error"] > TOL)
    return dict(passed=all(checks.values()), checks=checks, full9_successes=len(successes),
                lost_full9_successes=len(lost),
                lost_rows=[dict(id=r["id"], initial_depth=r["initial_depth"]) for r in lost])


def verify_stage(archive, stage):
    rows = load(archive / stage / "evaluation/ROWS.json")
    predictions = load(archive / stage / "decisions/PREDICTIONS.json")
    count, objects, arms = (2268, 36, REPLAY_ARMS) if stage == "replay" else (1584, 48, CONFIRMATION_ARMS)
    require(len(rows) == len(predictions) == count, stage + ": row count")
    evaluated, predicted = index(rows, row_key, stage), index(predictions, row_key, stage + " predictions")
    require(evaluated.keys() == predicted.keys(), stage + ": prediction identities differ")
    require({r["arm"] for r in rows} == arms, stage + ": arms differ")
    ids = {r["id"] for r in rows}
    require(len(ids) == objects, stage + ": scene denominator")
    truths = index(load(archive / "confirmation/data/truth/metadata.json"), lambda r: r["id"], "truth") if stage == "confirmation" else None
    if truths is not None:
        require(ids == truths.keys(), "Confirmation truth identities differ")
    scene_labels = {}
    for row in rows:
        pred = predicted[row_key(row)]
        require(not EVALUATION_FIELDS.intersection(pred), "Evaluation fields present in prediction")
        require({k: v for k, v in row.items() if k not in EVALUATION_FIELDS} == pred,
                f"{stage}: prediction changed in evaluation at {row_key(row)}")
        labels = (row["true_depth"], row["mechanism"], row["tensor"])
        require(scene_labels.setdefault(row["id"], labels) == labels, "Inconsistent scene labels")
        require(re.fullmatch(r"[0-9a-f]{64}", row["tensor"]) is not None, "Invalid tensor label")
        if truths is not None:
            same(row["true_depth"], truths[row["id"]]["true_depth"], "Confirmation truth depth")
            same(row["mechanism"], truths[row["id"]]["mechanism"], "Confirmation mechanism")
    rows = [recompute_row(r) for r in rows]
    require(Counter((r["id"], r["arm"], r["initial_kind"]) for r in rows)
            == Counter({(obj, arm, kind): 1 for obj in ids for arm in arms for kind in KINDS}),
            stage + ": every scene/arm must retain all three initial states")
    overall = aggregate(rows)
    require(all(r["n"] == objects for r in overall), stage + ": public denominator changed")
    unique, seen = [], set()
    for row in rows:
        key = row["tensor"], row["arm"], row["initial_depth"]
        if key not in seen:
            unique.append(row)
            seen.add(key)
    summary = dict(overall=overall,
                   by_mechanism={m: aggregate([r for r in rows if r["mechanism"] == m])
                                 for m in sorted({r["mechanism"] for r in rows})},
                   unique_tensors=aggregate(unique), distinct_tensors=len({r["tensor"] for r in rows}))
    same(load(archive / stage / "evaluation/SUMMARY.json"), summary, stage + " summary")
    with (archive / stage / "evaluation/RESULTS.csv").open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    require(len(table) == len(overall), stage + ": CSV row count")
    for saved, expected in zip(table, overall):
        converted = {key: (float(value) if type(expected[key]) is float else int(value)
                           if type(expected[key]) is int else value) for key, value in saved.items()}
        same(converted, expected, stage + " CSV")
    gate = strict_gate(rows, stage)
    same(load(archive / stage / "evaluation/GATE.json"), gate, stage + " strict gate")
    require(gate["passed"] is (stage == "replay"), stage + ": published scientific outcome changed")
    return dict(rows=count, arms=len(arms), public_scene_denominator=objects,
                scientific_gate="PASS" if gate["passed"] else "FAIL", checks=gate["checks"],
                primary=[r for r in overall if r["arm"] == "ED_S1_M"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Published repository root (default: this script's repository)")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        report = {"verification": "PASS", "archive": check_manifest(root)}
        report.update({stage: verify_stage(root / ARCHIVE, stage) for stage in ("replay", "confirmation")})
        report["limits"] = [
            "Artifact verification only; no renderer, predictor, calibration or decision algorithm rerun.",
            "NPZ bytes are hashed; array values and recorded tensor labels are not independently recomputed.",
            "Bootstrap intervals, source-host history/seals and old-replay curve reproduction are not recomputed.",
            "Scientific confirmation gate remains FAIL; all 48 scenes, including KEEP/rejections, remain in each initial-state denominator.",
        ]
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (VerificationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"verification": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
