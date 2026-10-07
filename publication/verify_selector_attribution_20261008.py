"""Verify archived hashes, selector-attribution tables and optional frozen replay.

Uses only the Python standard library and repository files. This does not
regenerate image observations, candidates, or distances to raw laser data.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "research_snapshots/2026-10-07"
RUN = SNAP / "selector-attribution-20261007T180539Z"
PREVIOUS = SNAP / "track-discrimination-20261007T160716Z"
KERNEL = PREVIOUS / "theory/kernel.py"
HISTORICAL_KERNEL = Path(
    "/srv/slam-research/grf/map-denoise/runs/"
    "track-discrimination-20261007T160716Z/theory/kernel.py"
)
MANIFEST = ROOT / "publication/SELECTOR_ATTRIBUTION_20261008_MANIFEST.json"
EVIDENCE = ("star_full", "star", "cycle")
NEW_ARMS = (
    "star_full_P", "star_full_GP", "star_full_R",
    "star_P", "star_GP", "star_R", "cycle_P", "cycle_GP", "cycle_R",
    "star_Q", "star_QG", "cycle_Q", "cycle_QG",
)
OLD_ARMS = ("old_replay", "old_no_ambiguity", "old_raw")
ARMS = ("photo_U11",) + NEW_ARMS + OLD_ARMS
ROI_COUNTS = {
    "scan118_base_ridge": 119, "scan118_upper_fold": 116,
    "scan122_book_edge": 123, "scan122_feather": 126,
}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return row["roi"], int(row["query"])


def index(rows, label):
    indexed = {key(row): row for row in rows}
    require(len(indexed) == len(rows), label + ": duplicate request")
    return indexed


def near(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected), label + ": nonfinite value")
    require(math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-12),
            f"{label}: {actual!r} != {expected!r}")


def compare(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + ": keys")
        for name in expected:
            compare(actual[name], expected[name], label + "." + name)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ": length")
        for number, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{label}[{number}]")
    elif isinstance(expected, float):
        near(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected,
                f"{label}: {actual!r} != {expected!r}")


def check_archive():
    manifest = load(MANIFEST)
    records = manifest["files"]
    names = set()
    for record in records:
        relative = Path(record["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Manifest path must be repository-relative: " + str(relative))
        require(relative.as_posix() not in names, "Duplicate manifest entry: " + str(relative))
        names.add(relative.as_posix())
        path = ROOT / relative
        require(path.resolve() == path and path.is_file(),
                "Manifest entry must be a regular repository file without symlinks: " + str(relative))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == record["sha256"], "SHA256 mismatch: " + str(relative))
        if "bytes" in record:
            require(path.stat().st_size == record["bytes"], "Size mismatch: " + str(relative))
    required = [RUN / name for name in (
        "INPUTS.json", "DECISIONS.json", "PROTOCOL.json", "REPRODUCTION_TARGETS.json",
        "policies.py", "evaluation/POINT_METRICS.csv", "evaluation/RESULTS.json",
        "evaluation/ATTRIBUTION.json", "evaluation/ATTRIBUTION.csv",
        "evaluation/TRANSITIONS.csv",
    )] + [KERNEL, PREVIOUS / "DECISIONS.json"]
    require({p.relative_to(ROOT).as_posix() for p in required} <= names,
            "Manifest does not cover all verifier inputs")
    roots = [ROOT / path for path in manifest["archive_roots"]]
    require(set(roots) == {RUN, PREVIOUS}, "Unexpected archive roots")
    actual = {p.relative_to(ROOT).as_posix() for base in roots
              for p in base.rglob("*") if p.is_file() or p.is_symlink()}
    require(actual == names,
            f"Archive file coverage: missing={sorted(names-actual)}, extra={sorted(actual-names)}")
    return len(records)


def point(row):
    require(row["primary"] in ("True", "False"), "Invalid primary flag")
    distance = float(row["distance_mm"]) if row["distance_mm"] else None
    require(distance is None or (math.isfinite(distance) and distance >= 0), "Invalid distance")
    return dict(arm=row["arm"], roi=row["roi"], query=int(row["query"]),
                primary=row["primary"] == "True",
                selected=int(row["selected"]) if row["selected"] else None,
                reason=row["reason"], distance_mm=distance)


def mean(values):
    return math.fsum(values) / len(values)


def verify_tables():
    inputs = index(load(RUN / "INPUTS.json")["rows"], "inputs")
    decisions = index(load(RUN / "DECISIONS.json")["rows"], "decisions")
    targets = index(load(RUN / "REPRODUCTION_TARGETS.json")["rows"], "targets")
    historical = index(load(PREVIOUS / "DECISIONS.json")["rows"], "historical decisions")
    require(len(inputs) == 21 and inputs.keys() == decisions.keys() == targets.keys(),
            "The same 21 active requests must occur in inputs, decisions and targets")
    require(inputs.keys() == historical.keys(), "Historical request identities differ")
    protocol = load(RUN / "PROTOCOL.json")
    require(protocol["new_arms"] == list(NEW_ARMS) and protocol["old_arms"] == list(OLD_ARMS),
            "The frozen protocol must contain the expected 16 policies")
    summary = load(RUN / "evaluation/RESULTS.json")
    points = [point(row) for row in read_csv(RUN / "evaluation/POINT_METRICS.csv")]
    require(len(points) == 17 * 512 and {p["arm"] for p in points} == set(ARMS),
            "Expected 17 arms and 8,704 point rows")
    by_arm = {arm: index([p for p in points if p["arm"] == arm], arm) for arm in ARMS}
    photo = by_arm["photo_U11"]
    request_keys = {(roi, query) for roi in ROI_COUNTS for query in range(128)}
    require(photo.keys() == request_keys, "Expected four ROI with 128 requests each")
    primary_keys = {k for k, p in photo.items() if p["primary"]}
    require(len(primary_keys) == 484, "Expected 484 fixed primary requests")
    frozen_ids = 0
    for k, row in inputs.items():
        saved, target = decisions[k], targets[k]
        require(k in photo and row["primary"] == photo[k]["primary"], "Active primary identity")
        for name in ("scene", "roi", "query", "primary", "current_id"):
            require(saved[name] == row[name], f"Decision identity {k}: {name}")
        require(set(saved["decisions"]) == set(ARMS[1:]), f"Missing policy at {k}")
        require(row["current_id"] == target["photo_id"], f"Frozen photo target at {k}")
        require(target["photo_id"] == historical[k]["current_id"], f"Historical photo ancestry at {k}")
        require(saved["decisions"]["old_replay"]["selected_candidate_id"] == target["photo_id"],
                f"Historical photo ID reproduction at {k}")
        frozen_ids += 1
        require(set(target["R"]) == set(EVIDENCE), f"Historical R target completeness at {k}")
        for evidence in EVIDENCE:
            require(target["R"][evidence] == historical[k]["decisions"][evidence]["selected_candidate_id"],
                    f"Historical {evidence} R ancestry at {k}")
            require(saved["decisions"][evidence + "_R"]["selected_candidate_id"] == target["R"][evidence],
                    f"Historical {evidence} R ID reproduction at {k}")
            frozen_ids += 1
    expected_transitions = []
    metrics = []
    for arm in ARMS:
        rows = by_arm[arm]
        require(rows.keys() == request_keys, arm + ": expected the same 512 requests")
        require({k for k, p in rows.items() if p["primary"]} == primary_keys,
                arm + ": primary request identities changed")
        for k, p in rows.items():
            if arm == "photo_U11":
                continue
            if k not in inputs:
                require({n: v for n, v in p.items() if n != "arm"} ==
                        {n: v for n, v in photo[k].items() if n != "arm"},
                        f"{arm}: inactive request changed at {k}")
                continue
            choice = decisions[k]["decisions"][arm]
            require(p["selected"] == choice["selected_candidate_id"] and p["reason"] == choice["reason"],
                    f"{arm}: table does not match frozen decision at {k}")
            require(p["selected"] is None or p["selected"] in
                    {obj["candidate_id"] for obj in inputs[k]["objects"]}, f"Unknown object at {k}")
            if p["selected"] == inputs[k]["current_id"]:
                require(p["distance_mm"] == photo[k]["distance_mm"], f"KEEP distance changed at {k}")
            else:
                expected_transitions.append(dict(
                    arm=arm, roi=k[0], query=k[1], primary=p["primary"],
                    photo_mm=photo[k]["distance_mm"], output_mm=p["distance_mm"],
                    photo_id=inputs[k]["current_id"], selected=p["selected"], reason=p["reason"]))
        primary = [p for p in rows.values() if p["primary"]]
        require(all(p["distance_mm"] is not None for p in primary), arm + ": missing primary distance")
        roi_metrics = {}
        for roi, count in ROI_COUNTS.items():
            ds = [p["distance_mm"] for p in primary if p["roi"] == roi]
            require(len(ds) == count, arm + ": ROI primary count")
            roi_metrics[roi] = dict(n=len(ds), mse_mm2=mean([d*d for d in ds]), mae_mm=mean(ds))
        deltas = [p["distance_mm"] - photo[key(p)]["distance_mm"] for p in primary]
        metric = dict(
            arm=arm, primary=484,
            mse_mm2=mean([v["mse_mm2"] for v in roi_metrics.values()]),
            mae_mm=mean([v["mae_mm"] for v in roi_metrics.values()]), roi=roi_metrics,
            scene_mse={str(scene): mean([v["mse_mm2"] for r, v in roi_metrics.items()
                                        if r.startswith("scan" + str(scene) + "_")]) for scene in (118, 122)},
            severe_gt5=sum(p["distance_mm"] > 5 for p in primary),
            improved_vs_photo=sum(d < -1e-9 for d in deltas),
            worsened_vs_photo=sum(d > 1e-9 for d in deltas),
            unchanged_vs_photo=sum(abs(d) <= 1e-9 for d in deltas),
            new_harm_vs_photo=sum(photo[key(p)]["distance_mm"] <= 1 < p["distance_mm"] for p in primary),
            finite=sum(p["distance_mm"] is not None for p in rows.values()),
            missing_finite=sum(not p["primary"] and p["distance_mm"] is not None for p in rows.values()),
            changed_active_ids=sum(t["arm"] == arm for t in expected_transitions),
        )
        baseline = metric["mse_mm2"] if not metrics else metrics[0]["mse_mm2"]
        metric["mse_reduction_pct_vs_photo"] = 100 * (baseline - metric["mse_mm2"]) / baseline
        metrics.append(metric)
    compare(metrics, summary["metrics"], "recomputed metrics")
    transitions = read_csv(RUN / "evaluation/TRANSITIONS.csv")
    for row in transitions:
        for name in ("query", "photo_id", "selected"):
            row[name] = int(row[name]) if row[name] else None
        for name in ("photo_mm", "output_mm"):
            row[name] = float(row[name]) if row[name] else None
        require(row["primary"] in ("True", "False"), "Invalid transition primary flag")
        row["primary"] = row["primary"] == "True"
    compare(expected_transitions, transitions, "transitions")
    compare({ev: sum(bool(r["intervals"][ev]) for r in inputs.values()) for ev in EVIDENCE},
            summary["support_counts"], "support counts")
    counts = {arm: dict(
        moved=sum(r["decisions"][arm]["selected_candidate_id"] != r["current_id"] for r in decisions.values()),
        eligible_alternatives=sum(len(r["decisions"][arm]["eligible_ids"]) for r in decisions.values()),
        reasons=dict(Counter(r["decisions"][arm]["reason"] for r in decisions.values())),
    ) for arm in ARMS[1:]}
    compare(counts, summary["decision_counts"], "decision counts")
    mse = {m["arm"]: m["mse_mm2"] for m in metrics}
    contrasts = []
    for evidence in EVIDENCE:
        for policy, gate in (("P", "GP"),) + (() if evidence == "star_full" else (("Q", "QG"),)):
            p, g, r = (mse[evidence + "_" + name] for name in (policy, gate, "R"))
            terms = [mse["photo_U11"] - p, p - g, g - r]
            near(math.fsum(terms), mse["photo_U11"] - r, "Attribution telescoping identity")
            near(terms[1] + terms[2], p - r, "Attribution extra-gain identity")
            contrasts.append(dict(evidence=evidence, point_policy=policy,
                simple_gain_mm2=terms[0], gate_increment_mm2=terms[1], ranking_increment_mm2=terms[2],
                robust_extra_vs_simple_mm2=p-r, total_robust_gain_mm2=mse["photo_U11"]-r))
    compare(contrasts, load(RUN / "evaluation/ATTRIBUTION.json")["rows"], "attribution JSON")
    csv_contrasts = read_csv(RUN / "evaluation/ATTRIBUTION.csv")
    for row in csv_contrasts:
        for name in row:
            if name not in ("evidence", "point_policy"):
                row[name] = float(row[name])
    compare(contrasts, csv_contrasts, "attribution CSV")
    return dict(point_rows=len(points), arms=len(metrics), requests_per_arm=512,
                fixed_primary_requests=484, frozen_historical_ids=frozen_ids,
                attribution_contrasts=len(contrasts), mse_mm2=mse)


def install_source_read_guard():
    """Audit source reads; this is a portability check, not a security sandbox."""
    blocked = []

    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if ROOT == path or ROOT in path.parents:
            return
        if path == Path("/srv") or Path("/srv") in path.parents or path == Path("/home") or Path("/home") in path.parents:
            blocked.append(str(path))
            raise PermissionError("Portable replay denied external source access: " + str(path))

    sys.addaudithook(guard)
    probes = (HISTORICAL_KERNEL, Path("/home/selector-attribution-portability-probe"))
    for path in probes:
        try:
            with path.open("rb"):
                pass
        except PermissionError:
            continue
        raise ValueError("Source-read guard failed its denial probe: " + str(path))
    require(len(blocked) == len(probes), "Source-read guard probe count")
    return blocked


def replay():
    inputs = load(RUN / "INPUTS.json")["rows"]
    saved = index(load(RUN / "DECISIONS.json")["rows"], "replay decisions")
    original_spec = importlib.util.spec_from_file_location
    mapped = []

    def archived_spec(name, location=None, *args, **kwargs):
        if location is not None and Path(location) == HISTORICAL_KERNEL:
            mapped.append(str(location))
            location = KERNEL
        return original_spec(name, location, *args, **kwargs)

    # Only this one historically absolute import is redirected. Both archived
    # source files remain byte-for-byte unchanged and all other imports pass through.
    importlib.util.spec_from_file_location = archived_spec
    try:
        spec = original_spec("archived_selector_attribution", RUN / "policies.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        importlib.util.spec_from_file_location = original_spec
    require(mapped == [str(HISTORICAL_KERNEL)], "Expected exactly one historical kernel import mapping")
    reproduced = 0
    for row in inputs:
        actual = {}
        for arm in NEW_ARMS:
            evidence, policy = arm.rsplit("_", 1)
            actual[arm] = module.new_policy(row, evidence, policy)
        actual["old_replay"] = module.old_policy(row)
        actual["old_no_ambiguity"] = module.old_policy(row, remove_ambiguity=True)
        actual["old_raw"] = module.old_policy(row, remove_ambiguity=True, remove_margin=True)
        for arm in ARMS[1:]:
            expected = saved[key(row)]["decisions"][arm]
            require(actual[arm]["selected_candidate_id"] == expected["selected_candidate_id"],
                    f"Replayed ID differs at {key(row)}, {arm}")
            require(actual[arm] == expected, f"Replayed full decision differs at {key(row)}, {arm}")
            reproduced += 1
    require(reproduced == 336, "Expected 336 replayed full decisions")
    return dict(replayed_policy_arms=16, replayed_selected_ids=reproduced,
                replayed_exact_full_decisions=reproduced, kernel_import_mappings=len(mapped))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="Replay all 16 policies from archived inputs")
    args = parser.parse_args()
    blocked = install_source_read_guard() if args.replay else None
    result = dict(archive_files=check_archive(), **verify_tables())
    if args.replay:
        result.update(replay())
        require(len(blocked) == 2, "Unexpected external source access during verification or replay")
        result["external_source_read_denial_probes"] = len(blocked)
        result["external_source_reads"] = 0
    print(json.dumps(dict(status="PASS", **result), indent=2))
    print("Checks archived bytes, table arithmetic and optional frozen selector replay only; "
          "no raw-photo, observation, candidate or laser-distance regeneration.")


if __name__ == "__main__":
    main()
