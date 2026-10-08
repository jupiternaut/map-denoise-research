#!/usr/bin/env python3
"""Read-only verification of the 2026-10-08 mixed-pixel publication.

Run from any working directory with Python 3.9+:
  python publication/verify_mixed_pixel_20261008.py
  python publication/verify_mixed_pixel_20261008.py --arrays

The default mode uses only the standard library. --arrays requires NumPy.
Only published files are opened: historical source paths are opaque manifest
keys, never filesystem fallbacks. No archived algorithm is imported or run.
This checks archived evidence, not an independent rerun of the experiment.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys


RUN = "mixed-pixel-20261008T041249Z"
ARCHIVES = {
    "surface-owned-support-20261008T022918Z",
    "footprint-support-20261008T025757Z",
    RUN,
    "mixed-pixel-study-20261008T035954Z",
}
PREFIX = "research_snapshots/2026-10-08/"
ARMS = ("K", "N", "EF", "ED", "OF", "OD")
INCUMBENTS = (540.0, 600.0, 660.0)
CANDIDATES = (450.0, 540.0, 600.0, 660.0, 900.0)
PREDICTION_SHA256 = "c0aac1709ecd347d58e4712164b642413b8cac980147bd0335cbb57f93bea2b7"
TOL = 1e-9


class VerificationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    def reject_constant(value):
        raise VerificationError("Nonstandard JSON number: " + value)

    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def same(actual, expected, label):
    """Compare independently computed nested results, with numeric tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected),
                label + ": dictionary keys differ")
        for key in expected:
            same(actual[key], expected[key], label + "/" + str(key))
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected),
                label + ": sequence lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, label + "/" + str(index))
    elif isinstance(expected, bool) or expected is None:
        require(actual is expected, label + ": value differs")
    elif isinstance(expected, (int, float)):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=TOL),
                "{}: {} != {}".format(label, actual, expected))
    else:
        require(actual == expected, label + ": value differs")


class Publication:
    def __init__(self, root, manifest):
        self.root = root.resolve()
        self.manifest = load_json(manifest)
        self.by_source = {}
        self.by_path = {}
        roots = self.manifest["archive_roots"]
        require(len(roots) == len(set(roots)), "Duplicate archive roots")
        require(set(roots) == {PREFIX + name for name in ARCHIVES},
                "The four archive roots do not match this publication")
        require(self.manifest.get("source_host") == "liekkas", "Wrong source host")
        for record in self.manifest["files"]:
            relative, source = record["path"], record["source"]
            posix = PurePosixPath(relative)
            require(not posix.is_absolute() and ".." not in posix.parts
                    and str(posix) == relative and "\\" not in relative,
                    "Unsafe/noncanonical archive path: " + relative)
            require(any(relative.startswith(item + "/") for item in roots),
                    "Manifest file lies outside archive roots: " + relative)
            require(relative not in self.by_path, "Duplicate path: " + relative)
            require(source not in self.by_source, "Duplicate source: " + source)
            require(PurePosixPath(source).is_absolute(), "Source key must be absolute")
            require(type(record["bytes"]) is int and record["bytes"] >= 0,
                    "Invalid byte count: " + relative)
            require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
                    "Invalid SHA-256: " + relative)
            require(record["transformation"] in
                    ("none", "filename only: archived instructions"),
                    "Unrecognized byte transformation: " + relative)
            target = self.root.joinpath(*posix.parts)
            require(target.resolve().is_relative_to(self.root),
                    "Archive path escapes repository: " + relative)
            self.by_path[relative] = record
            self.by_source[source] = record

        actual = set()
        for relative in roots:
            archive = self.root / relative
            require(archive.is_dir() and not archive.is_symlink(),
                    "Missing/linked archive root: " + relative)
            for path in archive.rglob("*"):
                require(not path.is_symlink(), "Symlinks are forbidden: " + str(path))
                if path.is_file():
                    actual.add(path.relative_to(self.root).as_posix())
                else:
                    require(path.is_dir(), "Nonregular archive object: " + str(path))
        declared = set(self.by_path)
        require(actual == declared,
                "Archive path-set mismatch; missing={}, extra={}".format(
                    sorted(declared - actual), sorted(actual - declared)))

        for relative, record in self.by_path.items():
            path = self.root / relative
            require(path.stat().st_size == record["bytes"], "Size mismatch: " + relative)
            require(sha256(path) == record["sha256"], "Hash mismatch: " + relative)
        method = self.by_path[PREFIX + RUN + "/METHOD.json"]
        require(method["source"].endswith("/METHOD.json"), "Unexpected method source key")
        self.source_root = method["source"][:-len("/METHOD.json")]

    def path(self, source):
        require(source in self.by_source, "Source is absent from manifest: " + source)
        return self.root / self.by_source[source]["path"]

    def source(self, relative):
        return self.source_root + "/" + relative

    def read(self, relative):
        return load_json(self.path(self.source(relative)))

    def check_hashes(self, mapping, label):
        for source, expected in mapping.items():
            require(sha256(self.path(source)) == expected,
                    label + ": sealed hash differs for " + source)


def decision(intervals, incumbent):
    for low, high in intervals:
        require(math.isfinite(low) and math.isfinite(high) and high > low,
                "Invalid support interval")
    mass = math.fsum(high - low for low, high in intervals)
    if not mass:
        return dict(selected_depth=incumbent, support_mean=None, estimated_squared_gain=0.0)
    mean = math.fsum((high * high - low * low) / 2 for low, high in intervals) / mass
    selected = min(CANDIDATES, key=lambda candidate: (candidate - mean) ** 2)
    gain = (incumbent - mean) ** 2 - (selected - mean) ** 2
    return dict(selected_depth=selected if gain > 0 else incumbent,
                support_mean=mean, estimated_squared_gain=max(0.0, gain))


def stats(rows):
    n = len(rows)
    require(n > 0, "Empty evaluation stratum")
    return dict(n=n, mae=math.fsum(r["error"] for r in rows) / n,
                mse=math.fsum(r["error"] ** 2 for r in rows) / n,
                initial_mae=math.fsum(r["initial_error"] for r in rows) / n,
                improved=sum(r["error"] < r["initial_error"] - TOL for r in rows),
                worsened=sum(r["error"] > r["initial_error"] + TOL for r in rows),
                unchanged=sum(abs(r["error"] - r["initial_error"]) <= TOL for r in rows),
                moved=sum(abs(r["selected_depth"] - r["initial_depth"]) > TOL for r in rows),
                empty_support=sum(not r["intervals"] for r in rows))


def comparison(rows, arm_a, arm_b):
    indexed = {(r["id"], r["arm"], r["initial_depth"]): r for r in rows}
    gains = [indexed[(r["id"], arm_b, r["initial_depth"])]["error"] - r["error"]
             for r in rows if r["arm"] == arm_a]
    return dict(n=len(gains), mae_gain=math.fsum(gains) / len(gains),
                better=sum(g > TOL for g in gains), worse=sum(g < -TOL for g in gains),
                same=sum(abs(g) <= TOL for g in gains))


def verify_standard(pub):
    method = pub.read("METHOD.json")
    same(method["candidates"], CANDIDATES, "candidates")
    same(method["incumbents"], INCUMBENTS, "incumbents")
    same(method["arms"], ARMS, "arms")
    same(method["grid"], [300.0, 1000.0, 2.0], "grid")
    same(method["residual_abs_max"], 9.0, "absolute gate")
    same(method["residual_excess_max"], 4.0, "excess gate")
    same(method["padding_mm"], 1.0, "interval padding")
    lock = pub.read("LOCK.json")
    require(lock["host"] == "liekkas", "Wrong run host")
    for name in ("sources", "inputs", "historical_hashes"):
        pub.check_hashes(lock[name], "LOCK/" + name)
    seal = pub.read("decisions/SEAL.json")
    digest = sha256(pub.path(pub.source("decisions/PREDICTIONS.json")))
    require(digest == seal["sha256"] == PREDICTION_SHA256,
            "Prediction hash differs from frozen publication")
    for name, expected in seal["observation_seals"].items():
        require(name in ("ordinary_stage", "oracle_stage"), "Unknown observation stage")
        require(sha256(pub.path(pub.source(name + "/SEAL.json"))) == expected,
                "Observation seal hash differs")
        pub.check_hashes(pub.read(name + "/SEAL.json")["files"], name)
    require(set(seal["observation_seals"]) == {"ordinary_stage", "oracle_stage"},
            "Missing observation seal")
    for name in ("evaluation", "diagnostics/pixels"):
        pub.check_hashes(pub.read(name + "/SEAL.json")["files"], name)

    truth_list = pub.read("data/truth/metadata.json")
    truth = {row["id"]: row for row in truth_list}
    require(len(truth_list) == len(truth) == 36, "Truth world count differs")
    require(len({r["group_id"] for r in truth_list}) == 24, "Base-group count differs")
    require(all(t["true_depth"] == 600.0 for t in truth_list), "Unexpected E1 true depth")
    predictions = pub.read("decisions/PREDICTIONS.json")
    stored = pub.read("evaluation/ROWS.json")
    key = lambda r: (r["id"], r["arm"], r["initial_depth"])
    expected_keys = set(itertools.product(truth, ARMS, INCUMBENTS))
    require(len(predictions) == len(stored) == len(expected_keys) == 648,
            "Decision-row count differs")
    require({key(r) for r in predictions} == {key(r) for r in stored} == expected_keys,
            "Decision keys contain omissions or duplicates")
    stored_by_key = {key(r): r for r in stored}
    rows = []
    for prediction in predictions:
        incumbent = prediction["initial_depth"]
        selected = prediction["selected_depth"]
        require(selected in CANDIDATES, "Selection outside frozen candidates")
        recomputed = decision(prediction["intervals"], incumbent)
        same({k: prediction[k] for k in recomputed}, recomputed, "P/" + str(key(prediction)))
        t = truth[prediction["id"]]
        derived = dict(prediction, true_depth=t["true_depth"], group_id=t["group_id"],
                       mechanism=t["mechanism"], background_pair=t["background_pair"],
                       initial_offset=incumbent - t["true_depth"],
                       error=abs(selected - t["true_depth"]),
                       initial_error=abs(incumbent - t["true_depth"]))
        published = stored_by_key[key(prediction)]
        same({k: published[k] for k in derived}, derived, "row/" + str(key(prediction)))
        rows.append(derived)

    summary = pub.read("evaluation/SUMMARY.json")
    for name, expected in (("n_worlds", 36), ("n_groups", 24), ("n_decisions", 648),
                           ("prediction_sha256", digest), ("historical_hashes_unchanged", True)):
        same(summary[name], expected, name)
    computed_summary = {}
    for incumbent in INCUMBENTS:
        subset = [r for r in rows if r["initial_depth"] == incumbent]
        by_arm = {arm: stats([r for r in subset if r["arm"] == arm]) for arm in ARMS}
        by_arm.update(ED_vs_N=comparison(subset, "ED", "N"),
                      ED_vs_EF=comparison(subset, "ED", "EF"))
        computed_summary[str(int(incumbent))] = by_arm
    same(summary["summary"], computed_summary, "summary")
    by_mechanism = {
        mechanism: {str(int(incumbent)): {
            arm: stats([r for r in rows if r["mechanism"] == mechanism
                        and r["initial_depth"] == incumbent and r["arm"] == arm])
            for arm in ARMS} for incumbent in INCUMBENTS}
        for mechanism in sorted({t["mechanism"] for t in truth_list})}
    same(summary["by_mechanism"], by_mechanism, "by_mechanism")
    reasons = {arm: dict(Counter(r.get("reason", "baseline_or_keep") for r in rows
                                if r["arm"] == arm)) for arm in ARMS}
    same(summary["reasons"], reasons, "reasons")
    full9_success = {(r["id"], r["initial_depth"]) for r in rows
                     if r["arm"] == "N" and r["error"] <= TOL}
    retention = {arm: dict(n=len(full9_success), retained=sum(
        r["error"] <= TOL for r in rows if r["arm"] == arm
        and (r["id"], r["initial_depth"]) in full9_success)) for arm in ARMS}
    same(summary["full9_success_retention"], retention, "full9_success_retention")
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r["group_id"], r["arm"], r["initial_depth"])][r["background_pair"]] = r
    pair_rows = [dict(group_id=g, arm=a, initial_depth=i,
                      selection_shift=rr[1]["selected_depth"] - rr[0]["selected_depth"],
                      error_shift=rr[1]["error"] - rr[0]["error"])
                 for (g, a, i), rr in sorted(pairs.items()) if 0 in rr and 1 in rr]
    same(pub.read("evaluation/BACKGROUND_PAIRS.json"), pair_rows, "background pairs")
    csv_fields = ["id", "group_id", "mechanism", "background_pair", "arm", "initial_depth",
                  "true_depth", "selected_depth", "initial_error", "error", "reason"]
    with pub.path(pub.source("evaluation/RESULTS.csv")).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == csv_fields, "CSV columns differ")
        csv_rows = list(reader)
    require(len(csv_rows) == len(rows), "CSV row count differs")
    for index, (csv_row, row) in enumerate(zip(csv_rows, rows)):
        same(csv_row, {k: "" if row.get(k) is None else str(row[k]) for k in csv_fields},
             "CSV/" + str(index))

    require(computed_summary["600"]["ED"]["worsened"] == 4, "Correct-incumbent harm differs")
    same([computed_summary[str(int(i))]["ED"]["mae"] for i in INCUMBENTS],
         [170 / 3, 20 / 3, 170 / 3], "ED headline MAE")
    same([computed_summary[str(int(i))]["N"]["mae"] for i in INCUMBENTS],
         [10.0, 0.0, 10.0], "full9 headline MAE")
    same(summary["gate"]["proceed_E2"], False, "E2 did not proceed")
    return dict(decisions=648, worlds=36, base_groups=24, prediction_sha256=digest,
                ed_mae_mm=[170 / 3, 20 / 3, 170 / 3], full9_mae_mm=[10, 0, 10],
                correct_incumbent_ed_harm=4, historical_hashes=len(lock["historical_hashes"])), {
                    "method": method, "truth": truth, "rows": stored, "summary": summary}


def intervals_from_mask(grid, accepted, padding):
    """Independent connected-run construction for the declared regular grid."""
    runs, start = [], None
    half = float(grid[1] - grid[0]) / 2
    for index in range(len(grid) + 1):
        included = index < len(grid) and bool(accepted[index])
        if included and start is None:
            start = index
        elif not included and start is not None:
            interval = [max(float(grid[0]), float(grid[start]) - half - padding),
                        min(float(grid[-1]), float(grid[index - 1]) + half + padding)]
            if runs and interval[0] <= runs[-1][1]:
                runs[-1][1] = max(runs[-1][1], interval[1])
            else:
                runs.append(interval)
            start = None
    return runs


def verify_arrays(pub, state):
    try:
        import numpy as np
    except ImportError as exc:
        raise VerificationError("--arrays requires NumPy; default verification is stdlib-only") from exc

    def close(actual, expected, label):
        require(np.shape(actual) == np.shape(expected), label + ": array shape differs")
        require(bool(np.allclose(actual, expected, rtol=1e-11, atol=TOL, equal_nan=True)),
                label + ": array values differ")

    def read_npz(source):
        with np.load(pub.path(source), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    truth, method = state["truth"], state["method"]
    inputs = pub.read("data/observed/inputs.json")
    require(len(inputs) == 36 and {r["id"] for r in inputs} == set(truth),
            "Observed-input ID set differs")
    images, fingerprints = {}, defaultdict(list)
    for record in inputs:
        require(sha256(pub.path(record["image_file"])) == record["sha256"],
                "Input hash differs from observed manifest")
        array = read_npz(record["image_file"])["images"]
        require(array.shape == (3, 128, 128) and bool(np.isfinite(array).all()),
                "Invalid observed image tensor")
        images[record["id"]] = array
        # NPZ-container hashes do not establish equality of observation tensors.
        fingerprint = (array.dtype.str, array.shape,
                       hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
        fingerprints[fingerprint].append(record["id"])
    require(len(fingerprints) == 30, "Expected 30 distinct observation tensors")
    duplicate_groups = sorted(sorted(ids) for ids in fingerprints.values() if len(ids) > 1)
    require(len(duplicate_groups) == 6 and all(len(ids) == 2 for ids in duplicate_groups),
            "Unexpected duplicate structure")

    curves, observations = {}, []
    raw_ranks = {}
    for stage in ("ordinary_stage", "oracle_stage"):
        payload = pub.read(stage + "/OBSERVATIONS.json")
        observations.extend(payload["rows"])
        auxiliaries = {r["id"]: r for r in payload["auxiliary"]}
        for row in payload["rows"]:
            curve = read_npz(row["curve_file"])
            grid = curve["grid"]
            close(grid, np.arange(300.0, 1001.0, 2.0), "depth grid")
            curves[row["curve_file"]] = curve
            if row["arm"] == "N":
                scores = curve["scores"]
                require(scores.shape == (2, 351), "full9 score shape differs")
                accepted = np.all(np.isfinite(scores) & (scores >= 0.6), axis=0)
            else:
                fold, counts = curve["fold_loss"], curve["pixel_counts"]
                require(fold.shape == (2, 351), "Fold loss shape differs")
                close(counts, [182.0, 182.0], "fixed pixel denominator")
                loss = np.average(fold, weights=counts, axis=0)
                close(curve["loss"], loss, "whole-grid fold aggregation")
                require(bool(np.isfinite(loss).all()), "Nonfinite mixed-model loss")
                auxiliary = auxiliaries[row["id"]]
                expected_sigma = [np.nan if v is None else v for v in auxiliary["sigma_values"]]
                close(curve["sigma_values"], expected_sigma, "training sigma")
                scale_valid = auxiliary["sigma_valid"]
                flat = bool(np.ptp(loss) <= method["flat_tolerance"])
                if scale_valid:
                    sigma_squared = np.average(np.square(curve["sigma_values"]), weights=counts)
                    normalized = loss / sigma_squared
                else:
                    normalized = np.full_like(loss, np.nan)
                close(curve["normalized_loss"], normalized, "whole-grid normalized loss")
                accepted = np.zeros(len(grid), dtype=bool)
                if not row["raw_valid"]:
                    reason = "auxiliary_or_prediction_invalid"
                elif not scale_valid:
                    reason = "scale_unavailable"
                elif flat:
                    reason = "flat_curve"
                else:
                    accepted = (normalized <= 9.0) & (normalized - normalized.min() <= 4.0)
                    reason = "ok" if bool(accepted.any()) else "threshold_empty"
                same(row["reason"], reason, "admission reason")
                same(row["scale_valid"], scale_valid, "scale validity")
                same(row["flat"], flat, "flat curve")
                indices = [int(np.argmin(np.abs(grid - c))) for c in CANDIDATES]
                values = loss[indices]
                correct_index = CANDIDATES.index(truth[row["id"]]["true_depth"])
                margin = float(np.min(np.delete(values, correct_index)) - values[correct_index])
                best = np.flatnonzero(np.abs(values - np.min(values)) <= TOL)
                errors = [abs(CANDIDATES[int(i)] - truth[row["id"]]["true_depth"]) for i in best]
                raw_ranks[row["curve_file"]] = dict(
                    valid=True, flat=bool(np.ptp(values) <= TOL), margin=margin,
                    correct_strict_best=margin > TOL, correct_tied_best=abs(margin) <= TOL,
                    wrong_strictly_better=int(np.sum(np.delete(values, correct_index) < values[correct_index] - TOL)),
                    diagnostic_argmin_error_min=float(min(errors)),
                    diagnostic_argmin_error_max=float(max(errors)), candidate_losses=values.tolist(),
                    whole_grid_range=float(np.ptp(loss)))
            require(np.array_equal(curve["accepted"], accepted), "Whole-grid accepted mask differs")
            same(row["intervals"], intervals_from_mask(grid, accepted, method["padding_mm"]),
                 "support intervals")
    require(len(curves) == len(observations) == 324, "Expected 324 unique score curves")

    for row in state["rows"]:
        if "curve_file" in row and row["arm"] != "N":
            same(row["ranking"], raw_ranks[row["curve_file"]], "candidate ranking")
    ranking = {}
    for mechanism in state["summary"]["by_mechanism"]:
        for arm in ("EF", "ED", "OF", "OD"):
            selected = [r for r in state["rows"] if r["mechanism"] == mechanism and r["arm"] == arm
                        and (not arm.endswith("D") or r["initial_depth"] == 600)]
            ranks = [raw_ranks[r["curve_file"]] for r in selected]
            ranking[mechanism + "/" + arm] = dict(
                n=len(selected), valid=len(ranks), strict_correct=sum(r["correct_strict_best"] for r in ranks),
                flat=sum(r["flat"] for r in ranks), tied_correct=sum(r["correct_tied_best"] for r in ranks),
                mean_margin=math.fsum(r["margin"] for r in ranks) / len(ranks),
                mean_argmin_error_min=math.fsum(r["diagnostic_argmin_error_min"] for r in ranks) / len(ranks),
                mean_argmin_error_max=math.fsum(r["diagnostic_argmin_error_max"] for r in ranks) / len(ranks))
    same(state["summary"]["ranking"], ranking, "ranking summary")
    ed = [r for r in observations if r["arm"] == "ED"]
    informative = [r for r in ed if truth[r["id"]]["mechanism"] != "flat_equal"]
    null = [r for r in ed if truth[r["id"]]["mechanism"] == "flat_equal"]
    require(len(informative) == 30 and all(raw_ranks[r["curve_file"]]["correct_strict_best"] for r in informative),
            "ED informative-world ranking differs")
    require(len(null) == 6 and all(raw_ranks[r["curve_file"]]["flat"] for r in null),
            "ED no-information control differs")
    directional = [state["summary"]["summary"][str(int(i))]["ED_vs_N"]["mae_gain"] for i in (540, 660)]
    gate = dict(
        oracle_boundary_separation=ranking["flat_contrast/OD"]["mean_margin"] > 0
        and ranking["flat_contrast/OD"]["mean_margin"] > ranking["flat_contrast/OF"]["mean_margin"],
        equal_color_no_information=all(raw_ranks[r["curve_file"]]["flat"] for r in observations
                                       if r["arm"] != "N" and truth[r["id"]]["mechanism"] == "flat_equal"),
        estimated_correct_input_no_harm=state["summary"]["summary"]["600"]["ED"]["worsened"] == 0,
        estimated_ncc_each_direction_nonworse=min(directional) >= -TOL,
        estimated_ncc_some_direction_better=max(directional) > TOL,
        estimated_dynamic_beats_fixed=comparison([r for r in state["rows"] if r["initial_depth"] != 600], "ED", "EF")["mae_gain"] > TOL)
    gate["proceed_E2"] = all(gate.values())
    same(state["summary"]["gate"], gate, "scientific gate")

    pixel_count, maximum_difference, rois = 0, 0.0, {}
    for row in observations:
        if row["arm"] == "N":
            continue
        curve = curves[row["curve_file"]]
        indices = [int(np.argmin(np.abs(curve["grid"] - c))) for c in CANDIDATES]
        incumbent = 600 if row["arm"].endswith("D") else int(row["incumbent"])
        for fold, (_, view) in enumerate(method["folds"]):
            relative = "diagnostics/pixels/{}_{}_{}_fold{}.npz".format(
                row["id"], row["arm"], incumbent, fold)
            pixels = read_npz(pub.source(relative))
            require(bool(pixels["valid"]), "Invalid pixel diagnostics")
            close(pixels["candidates"], CANDIDATES, "pixel candidates")
            roi = pixels["roi"]
            require(roi.shape == (182, 2) and np.issubdtype(roi.dtype, np.integer), "Invalid ROI")
            require(bool(((roi >= 0) & (roi < 128)).all()), "ROI outside sensor")
            key = (row["id"], fold)
            if key in rois:
                require(np.array_equal(rois[key], roi), "Arms do not share their ROI")
            rois[key] = roi
            observed = images[row["id"]][view, roi[:, 1], roi[:, 0]]
            close(pixels["observed"], observed, "raw observed sensor pixels")
            residual = pixels["prediction"] - observed[None, :]
            close(pixels["residual"], residual, "pixel residual")
            mse = np.mean(np.square(residual), axis=1)
            expected = curve["fold_loss"][fold, indices]
            close(mse, expected, "candidate pixel-to-curve MSE")
            maximum_difference = max(maximum_difference, float(np.max(np.abs(mse - expected))))
            for name in ("predicted_alpha", "true_alpha_evaluation_only"):
                require(bool(np.isfinite(pixels[name]).all()) and bool(((pixels[name] >= 0) & (pixels[name] <= 1)).all()),
                        "Invalid alpha in pixel diagnostics")
            pixel_count += 1
    require(pixel_count == 576, "Expected 576 pixel diagnostic artifacts")

    witness = next(r for r in state["rows"] if r["id"] == "w002" and r["arm"] == "ED" and r["initial_depth"] == 600)
    same(witness["intervals"], [[486.0, 782.0]], "w002 interval")
    same(witness["support_mean"], 634.0, "w002 interval mean")
    same(witness["selected_depth"], 660.0, "w002 wrong selection")
    require(raw_ranks[witness["curve_file"]]["correct_strict_best"], "w002 lost score minimum")
    return dict(numpy_version=np.__version__, input_worlds=36, distinct_observation_tensors=30,
                duplicate_id_groups=duplicate_groups, ed_informative_strict_first=30, ed_flat_worlds=6,
                full_depth_curves_checked=324, mixed_fold_aggregations_checked=288,
                candidate_pixel_artifacts_checked=pixel_count, maximum_pixel_loss_difference=maximum_difference,
                w002=dict(score_minimum_mm=600, support=[486, 782], mean_mm=634, selected_mm=660),
                limitation="Whole-grid checks recompute aggregation/admission from saved fold losses; pixel residual checks cover the five saved candidates. No rerendering is performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Published repository root (default: parent of publication/)")
    parser.add_argument("--manifest", type=Path, help="Manifest path (default: ROOT/publication/MIXED_PIXEL_20261008_MANIFEST.json)")
    parser.add_argument("--arrays", action="store_true", help="Additionally check archived arrays with NumPy")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.manifest or root / "publication/MIXED_PIXEL_20261008_MANIFEST.json"
    try:
        publication = Publication(root, manifest)
        standard, state = verify_standard(publication)
        result = dict(status="PASS", mode="arrays" if args.arrays else "stdlib", repository=str(root),
                      manifest_files=len(publication.by_path),
                      manifest_bytes=sum(r["bytes"] for r in publication.by_path.values()),
                      exact_archive_path_set=True, standard=standard)
        if args.arrays:
            result["arrays"] = verify_arrays(publication, state)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (VerificationError, OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        print(json.dumps(dict(status="FAIL", error=str(exc)), ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
