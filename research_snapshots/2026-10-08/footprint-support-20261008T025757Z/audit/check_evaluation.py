"""Compare independent outcomes to published evaluation and diagnose rejections."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT.parent / "surface-owned-support-20261008T022918Z/mechanism"


def read(path):
    return json.loads(path.read_text())


def function_source(path, name):
    source = path.read_text()
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source, node)


def main():
    independent = read(ROOT / "audit/REPLAY_RESULTS.json")
    rows = independent["records"]
    published_rows = {(r["id"], r["arm"]): r for r in read(ROOT / "evaluation/ROWS.json")}
    published = read(ROOT / "evaluation/RESULTS.json")
    failures = []
    for r in rows:
        p = published_rows[(r["fixture_id"], r["arm"])]
        for field in ("name", "seed", "intervals", "selected_depth", "true_depth", "absolute_error", "true_in_support", "empty"):
            if r[field] != p[field]:
                failures.append([r["fixture_id"], r["arm"], field])
    summaries = []
    for p in published["summary"]:
        selected = [r for r in rows if r["arm"] == p["arm"] and (
            p["group"] == "all" or (p["group"] == "ring") == r["name"].startswith("ring"))]
        base = {r["fixture_id"]: r for r in rows if r["arm"] == "full9"}
        changes = [r["absolute_error"] - base[r["fixture_id"]]["absolute_error"] for r in selected]
        result = dict(group=p["group"], arm=p["arm"], n=len(selected),
                      mae=float(np.mean([r["absolute_error"] for r in selected])),
                      mse=float(np.mean([r["absolute_error"] ** 2 for r in selected])),
                      exact=sum(r["absolute_error"] == 0 for r in selected),
                      truth_in_support=sum(r["true_in_support"] for r in selected),
                      empty=sum(r["empty"] for r in selected),
                      improved_vs_full9=sum(x < 0 for x in changes),
                      worsened_vs_full9=sum(x > 0 for x in changes),
                      unchanged_vs_full9=sum(x == 0 for x in changes),
                      retained_full9_exact=sum(r["absolute_error"] == 0 and base[r["fixture_id"]]["absolute_error"] == 0 for r in selected),
                      baseline_exact=sum(base[r["fixture_id"]]["absolute_error"] == 0 for r in selected),
                      improved_vs_incumbent=sum(r["absolute_error"] < abs(900 - r["true_depth"]) for r in selected),
                      worsened_vs_incumbent=sum(r["absolute_error"] > abs(900 - r["true_depth"]) for r in selected))
        if result != p:
            failures.append([p["group"], p["arm"], "summary", result, p])
        summaries.append(result)
    pair_mapping = {(p["pair"], p["seed"], p["arm"]): p for p in read(ROOT / "evaluation/PAIRS.json")}
    pair_fields = {"reference_center3_max_difference": "reference_center_difference",
                   "acceptance_xor": "accepted_xor",
                   "interval_symmetric_difference": "support_symmetric_difference",
                   "selected_depth_delta": "selected_depth_difference",
                   "both_empty": "both_empty", "max_weight_difference": "weight_max_difference"}
    for r in independent["pairs"]:
        p = pair_mapping[(r["pair"], r["seed"], r["arm"])]
        for a, b in pair_fields.items():
            if r[a] != p[b]:
                failures.append([r["pair"], r["seed"], r["arm"], a])
    flips = {arm: sum(p["selected_depth_delta"] > 0 for p in independent["pairs"] if p["arm"] == arm)
             for arm in published["selection_flips"]}
    if flips != published["selection_flips"]:
        failures.append(["selection_flip_summary"])
    primary = next(p for p in summaries if p["group"] == "ring" and p["arm"] == "estimated_footprint")
    baseline = next(p for p in summaries if p["group"] == "ring" and p["arm"] == "full9")
    primary_success = (primary["mae"] < baseline["mae"]
                       and primary["retained_full9_exact"] == baseline["exact"]
                       and flips["estimated_footprint"] < flips["full9"])
    if primary_success != published["primary_success"]:
        failures.append(["primary_success"])

    old_source = function_source(ROOT / "run.py", "evaluate")
    new_source = function_source(ROOT / "evaluate_v2.py", "evaluate")
    exact_repair = old_source.replace("dict(**o,**d,", "dict(o,**d,") == new_source
    if old_source.count("dict(**o,**d,") != 1 or not exact_repair:
        failures.append(["evaluation_repair_exceeds_one_replacement"])

    observations = [r for folder in ("observed_stage", "oracle_stage")
                    for r in read(ROOT / folder / "OBSERVATIONS.json")["rows"]]
    meta = {r["id"]: r for r in read(ROOT / "TRUTH.json")}
    old_rows = {(r["fixture_id"], r["arm"]): r for r in read(HIST / "OBSERVATIONS.json")}
    curve_mapping = {(r["id"], r["arm"]): r for r in observations}
    baseline_checks = []
    diagnostics = []
    method = read(ROOT / "METHOD.json")
    for r in observations:
        with np.load(r["curve_file"]) as raw:
            curve = {key: raw[key] for key in raw.files}
        if r["arm"] in ("full9", "connected9"):
            old = old_rows[(meta[r["id"]]["historical_id"], "plane_" + r["arm"])]
            with np.load(HIST / "curves" / old["curve_file"]) as old_curve:
                finite = np.isfinite(curve["scores"]) & np.isfinite(old_curve["scores"])
                max_difference = float(np.max(abs(curve["scores"][finite] - old_curve["scores"][finite]))) if finite.any() else 0.0
                accepted_same = np.array_equal(curve["accepted"], old_curve["accepted"])
                finite_same = np.array_equal(np.isfinite(curve["scores"]), np.isfinite(old_curve["scores"]))
                score_bitwise_same = np.array_equal(curve["scores"], old_curve["scores"], equal_nan=True)
            if max_difference > 1e-12 or not accepted_same or not finite_same or r["intervals"] != old["intervals"]:
                failures.append([r["id"], r["arm"], "baseline_reproduction"])
            baseline_checks.append(dict(id=r["id"], arm=r["arm"], score_max_difference=max_difference,
                                        score_bitwise_identical=score_bitwise_same, accepted_identical=accepted_same))
        if r["arm"] in ("estimated_footprint", "oracle_pure", "oracle_fraction", "oracle_majority", "oracle_component"):
            masses = curve["mass"] >= method["min_mass"]
            effective = masses & (curve["ess"] >= method["min_ess"] - 1e-10)
            reference = effective & (curve["ref_std"] >= method["std_min"])
            source = reference & np.all(curve["source_std"] >= method["std_min"], axis=0)
            valid = source & np.all(curve["valid_samples"] | (curve["weights"][None] <= 1e-14), axis=(0, 2))
            accepted = valid & np.all(np.isfinite(curve["scores"]) & (curve["scores"] >= method["ncc_min"]), axis=0)
            true_index = np.flatnonzero(curve["grid"] == 600)[0]
            diagnostics.append(dict(id=r["id"], name=meta[r["id"]]["name"], seed=meta[r["id"]]["seed"], arm=r["arm"],
                                    grid_count=len(masses), mass_pass=int(masses.sum()), ess_pass=int(effective.sum()),
                                    ref_std_pass=int(reference.sum()), both_source_std_pass=int(source.sum()), valid_pass=int(valid.sum()),
                                    accepted=int(accepted.sum()), maximum_mass=float(curve["mass"].max()), maximum_ess=float(curve["ess"].max()),
                                    true_mass=float(curve["mass"][true_index]), true_ess=float(curve["ess"][true_index]),
                                    true_reference_std=float(curve["ref_std"][true_index]),
                                    true_source_std=curve["source_std"][:, true_index].tolist()))
    component_comparisons = []
    for fid, t in meta.items():
        with np.load(curve_mapping[(fid, "oracle_majority")]["curve_file"]) as majority, np.load(curve_mapping[(fid, "oracle_component")]["curve_file"]) as component:
            same = np.array_equal(majority["weights"], component["weights"])
        if not same:
            failures.append([fid, "component_majority_weight_mismatch"])
        component_comparisons.append(dict(id=fid, same_weights=same))
    result = dict(failure_count=len(failures), failures=failures, rows_checked=len(rows), summary_rows_checked=len(summaries),
                  paired_rows_checked=len(independent["pairs"]), evaluation_repair_exact_one_replacement=exact_repair,
                  evaluate_v2_sha256=hashlib.sha256((ROOT / "evaluate_v2.py").read_bytes()).hexdigest(),
                  baseline_reproductions=len(baseline_checks), baseline_checks=baseline_checks,
                  maximum_baseline_score_difference=max(r["score_max_difference"] for r in baseline_checks),
                  primary_success=primary_success, selection_flips=flips, summaries=summaries,
                  diagnostics=diagnostics, component_weight_checks=component_comparisons)
    (ROOT / "audit/EVALUATION_CHECKS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("failure_count", "failures", "rows_checked", "summary_rows_checked", "paired_rows_checked", "evaluation_repair_exact_one_replacement", "baseline_reproductions", "maximum_baseline_score_difference", "primary_success", "selection_flips")}, indent=2))
    print(json.dumps([r for r in diagnostics if r["arm"] == "estimated_footprint"], indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
