"""Read-only verification of the public subset; no pickle/model loading or GT."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "research_snapshots/2026-09-26"
FIELD = SNAPSHOTS / "multisurface-field-lab-20260926T102842Z"
CONTINUOUS = SNAPSHOTS / "continuous-step-lab-20260926T100617Z"
CONDITIONS = ("native", "minus1", "plus1", "minus3", "plus3")
SCENES = ("55", "65", "69")
ARMS = ("identity", "prior_recovery", "point_wta", "single_field", "multi_field",
        "multi_field_visibility", "multi_field_graph", "oracle_discrete",
        "oracle_continuous", "oracle_proposals", "oracle_all_layers", "oracle_expanded")
METRICS = ("source_MSE_mm2", "source_MAE_mm", "source_p95_mm", "reverse_MAE_mm",
           "precision_1mm", "recall_1mm", "Fscore_1mm", "improved_fraction",
           "harmed_fraction", "moved_support_fraction", "move_RMS_mm")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected), f"Nonfinite {label}")
    require(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=5e-12),
            f"Mismatch {label}: {actual} vs {expected}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest():
    manifest = json.loads((ROOT / "publication/REPLAY_20260928_MANIFEST.json").read_text())
    rows = manifest["files"]
    paths = [row["path"] for row in rows]
    require(len(paths) == len(set(paths)), "Duplicate manifest paths")
    for row in rows:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe path")
        path = ROOT / relative
        require(not path.is_symlink() and path.is_file(), f"Missing/linked {relative}")
        require(path.stat().st_size == row["bytes"], f"Size mismatch {relative}")
        require(sha(path) == row["sha256"], f"SHA-256 mismatch {relative}")
    return len(rows), sum(row["bytes"] for row in rows)


def aggregate(rows):
    require(len(rows) == 720, "Expected 720 main metric rows")
    keys = [(r["scene"], r["roi"], r["condition"], r["arm"]) for r in rows]
    require(len(keys) == len(set(keys)), "Duplicate metric rows")
    expected = {(s, f"scan{s}_roi{i}", c, a) for s in SCENES for i in range(4)
                for c in CONDITIONS for a in ARMS}
    require(set(keys) == expected, "Missing/unexpected scene/ROI/condition/arm")
    result = {}
    for row in rows:
        for metric in METRICS:
            require(math.isfinite(float(row[metric])), f"Nonfinite {metric}")
            require(float(row[metric]) >= 0, f"Negative distance/fraction {metric}")
    for condition in CONDITIONS:
        result[condition] = {}
        for arm in ARMS:
            selected = [r for r in rows if r["condition"] == condition and r["arm"] == arm]
            result[condition][arm] = {
                metric: statistics.mean(statistics.mean(float(r[metric]) for r in selected
                                                        if r["scene"] == scene)
                                        for scene in SCENES)
                for metric in METRICS}
        before = result[condition]["identity"]["source_MSE_mm2"]
        require(before > 0, "Relative improvement needs positive identity MSE")
        for cell in result[condition].values():
            cell["MSE_gain_percent"] = 100 * (1 - cell["source_MSE_mm2"] / before)
    return result


def verify_metrics():
    with (FIELD / "run_evidence/evaluation/METRICS.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual = aggregate(rows)
    recorded = json.loads((FIELD / "run_evidence/evaluation/SUMMARY.json").read_text())["exposed_replay"]
    cells = 0
    for c in CONDITIONS:
        for arm in ARMS:
            for metric, value in actual[c][arm].items():
                close(value, recorded[c][arm][metric], f"{c}/{arm}/{metric}")
                cells += 1
    # These values are the displayed, rounded claims, not a replacement data source.
    displayed = {
        "prior_recovery": [-7.67, 1.16, 10.04, 46.99, 39.02],
        "multi_field": [-83.35, -84.10, -41.00, 23.89, 21.95],
        "oracle_discrete": [32.60, 38.87, 43.90, 69.94, 63.27],
        "oracle_continuous": [38.97, 49.61, 49.46, 74.69, 64.99],
        "oracle_proposals": [64.22, 65.85, 73.32, 90.66, 92.57],
        "oracle_all_layers": [65.36, 67.41, 74.70, 90.95, 92.75],
        "oracle_expanded": [67.64, 70.64, 76.32, 91.94, 93.10],
    }
    readme = (ROOT / "README.md").read_text()
    for arm, values in displayed.items():
        for c, expected in zip(CONDITIONS, values):
            close(round(actual[c][arm]["MSE_gain_percent"], 2), expected, f"rounded {c}/{arm}")
        if arm in ("prior_recovery", "multi_field"):
            line = " | ".join(f"{v:.2f}%".replace("-", "−") for v in values)
            require(line in readme, f"README values differ for {arm}")
        else:
            line = " | ".join(f"{v:.2f}%" for v in values[-2:])
            require(line in readme, f"README Oracle values differ for {arm}")
    native = actual["native"]["identity"]
    for value, precision, label in ((native["source_MAE_mm"], 3, "0.562"),
                                     (native["source_MSE_mm2"], 3, "0.753"),
                                     (math.sqrt(native["source_MSE_mm2"]), 3, "0.868"),
                                     (100 * native["precision_1mm"], 2, "90.18")):
        require(f"{value:.{precision}f}" == label and label in readme, "Native metric claim changed")
    # A deliberate duplicate must be rejected; do not trust a count-only validator.
    broken = rows[:-1] + [rows[0]]
    try:
        aggregate(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate-row negative control escaped")
    return len(rows), cells


def verify_links():
    count = 0
    for rel in ("README.md", "publication/REPLAY_20260928.md",
                "publication/METHOD_POSITIONING_20260928.md"):
        doc = ROOT / rel
        for target in re.findall(r"\]\(([^\s)]+)\)", doc.read_text()):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            relative = target.split("#", 1)[0]
            require((doc.parent / relative).is_file(), f"Broken local link {rel}: {target}")
            count += 1
    return count


def verify_additional_claims():
    """Check continuous-step and posthoc K2 claims against their own CSVs."""
    readme = (ROOT / "README.md").read_text()
    with (CONTINUOUS / "run_evidence/evaluation/METRICS.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_grid = (-6.40, 3.18, 9.81, 47.21, 38.55)
    for condition, expected in zip(CONDITIONS, expected_grid):
        values = {}
        for arm in ("identity", "grid"):
            selected = [r for r in rows if r["condition"] == condition and r["arm"] == arm]
            require({(r["scene"], r["roi"]) for r in selected} ==
                    {(s, f"scan{s}_roi{i}") for s in SCENES for i in range(4)} and len(selected) == 12,
                    f"Continuous grid/identity support changed: {condition}/{arm}")
            values[arm] = statistics.mean(float(r["source_MSE_mm2"]) for r in selected)
        close(round(100 * (1 - values["grid"] / values["identity"]), 2), expected,
              f"continuous grid/{condition}")
    line = " | ".join(f"{v:.2f}%".replace("-", "−") for v in expected_grid)
    require(line in readme, "README continuous-step row differs")
    with (FIELD / "run_evidence/posthoc_field_capacity/METRICS.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 120, "Posthoc metrics row count changed")
    for condition, expected in (("minus3", 62.69), ("plus3", 62.66)):
        selected = [r for r in rows if r["condition"] == condition and r["arm"] == "oracle_K2_only"]
        require(len(selected) == 12, "Missing posthoc K2 cases")
        mse = statistics.mean(float(r["source_MSE_mm2"]) for r in selected)
        baseline = statistics.mean(float(r["identity_MSE_mm2"]) for r in selected)
        close(round(100 * (1 - mse / baseline), 2), expected, f"posthoc K2/{condition}")
    require("62.69%/62.66%" in readme, "README posthoc K2 claim differs")
    return 7


def main():
    files, size = verify_manifest()
    rows, cells = verify_metrics()
    additional = verify_additional_claims()
    links = verify_links()
    print(json.dumps({"status": "PASS", "public_files": files, "bytes": size,
                      "main_metric_rows": rows, "aggregate_cells_checked": cells,
                      "additional_gain_claims_checked": additional,
                      "local_links_checked": links, "duplicate_negative_control": "PASS",
                      "scope": "public archive bytes and table recomputation, not inference replay",
                      "source_datasets_opened": False, "models_deserialized": False}, indent=2))


if __name__ == "__main__":
    main()
