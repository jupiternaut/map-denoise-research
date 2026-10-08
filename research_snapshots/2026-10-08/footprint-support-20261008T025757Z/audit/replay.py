"""Independent sealed-curve, candidate-choice, and paired-outcome audit.

No main scorer, estimator, selector, or evaluator is imported. The historical
renderer is used only after decisions exist, to recover oracle target radiance.
All authored and generated outputs remain beneath audit/.
"""

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from independent_math import (choose, length, reconstruct_intervals,
                              symmetric_difference_length)


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT.parent / "surface-owned-support-20261008T022918Z/mechanism"
CANDIDATES = [450.0, 540.0, 600.0, 660.0, 900.0]
INCUMBENT = 900.0


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def fixture_id(row):
    return row.get("fixture_id", row.get("id"))


def bilinear(field, uv):
    """Explicit 4-tap sampling, independent of scipy's interpolation."""
    uv = np.asarray(uv)
    lo = np.floor(uv).astype(int)
    delta = uv - lo
    result = np.zeros(uv.shape[:-1])
    valid = np.isfinite(uv).all(axis=-1)
    valid &= ((uv[..., 0] >= 0) & (uv[..., 0] <= field.shape[1] - 1)
              & (uv[..., 1] >= 0) & (uv[..., 1] <= field.shape[0] - 1))
    for dy in (0, 1):
        for dx in (0, 1):
            coefficient = ((delta[..., 0] if dx else 1 - delta[..., 0])
                           * (delta[..., 1] if dy else 1 - delta[..., 1]))
            x = np.clip(lo[..., 0] + dx, 0, field.shape[1] - 1)
            y = np.clip(lo[..., 1] + dy, 0, field.shape[0] - 1)
            result += coefficient * field[y, x]
    return result, valid


def coordinates(cameras, grid):
    yy, xx = np.mgrid[-4:5, -4:5]
    ref_uv = np.stack((xx.ravel() + 64.0, yy.ravel() + 64.0), axis=-1)
    homogeneous = np.column_stack((ref_uv, np.ones(81)))
    ref = cameras[0]
    rays = homogeneous @ np.linalg.inv(ref["K"]).T @ ref["R"]
    world = ref["C"] + grid[:, None, None] * rays[None]
    projected, positive = [], []
    for camera in cameras[1:]:
        local = (world - camera["C"]) @ camera["R"].T
        uvw = local @ camera["K"].T
        projected.append(uvw[..., :2] / uvw[..., 2:])
        positive.append(local[..., 2] > 0)
    return ref_uv, projected, positive


def target_fields(renderer, metadata, shape):
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    uv = np.stack((xx, yy), axis=-1).astype(float)
    numerators, fractions = [], []
    for raw in metadata["actual_cameras"]:
        camera = {k: np.asarray(v, dtype=float) for k, v in raw.items()}
        numerator, fraction = np.zeros(shape), np.zeros(shape)
        for oy in (-1 / 3, 0, 1 / 3):
            for ox in (-1 / 3, 0, 1 / 3):
                hit = renderer.intersect(metadata["scene"], camera, uv + [ox, oy])
                is_target = hit["owner"] == metadata["scene"]["target_id"]
                numerator += np.where(is_target, hit["value"], 0) / 9
                fraction += is_target / 9
        numerators.append(numerator)
        fractions.append(fraction)
    return np.array(numerators), np.array(fractions)


def independent_scores(values, weight, valid, positive, method):
    """Vectorized weighted means/covariance, independent from run kernel."""
    mass = np.sum(weight, axis=1)
    squared = np.sum(weight * weight, axis=1)
    ess = np.divide(mass * mass, squared, out=np.zeros_like(mass), where=squared > 0)
    safe_mass = np.where(mass > 0, mass, 1)
    avg = np.sum(values * weight[None], axis=2) / safe_mass[None]
    centered = values - avg[:, :, None]
    variance = np.sum(centered * centered * weight[None], axis=2) / safe_mass[None]
    std = np.sqrt(variance)
    scores = np.full((2, len(mass)), np.nan)
    for source in range(2):
        covariance = np.sum(centered[0] * centered[source + 1] * weight, axis=1) / safe_mass
        good = ((mass >= method["min_mass"]) & (ess >= method["min_ess"] - 1e-10)
                & (std[0] >= method["std_min"])
                & (std[source + 1] >= method["std_min"]))
        good &= np.all((weight[None] <= 1e-14) | (valid[0][None] & valid[1:] & positive), axis=(0, 2))
        scores[source, good] = covariance[good] / (std[0, good] * std[source + 1, good])
    return scores, mass, ess, std


def main():
    stages = [read(ROOT / folder / "OBSERVATIONS.json")
              for folder in ("observed_stage", "oracle_stage")]
    observations = [row for stage in stages for row in stage["rows"]]
    choice_path = ROOT / "decisions/DECISIONS.json"
    choices = read(choice_path)  # Must exist before any world truth is read below.
    method = read(ROOT / "METHOD.json")
    manifests = read(HIST / "FIXTURES.json")
    old_metadata = {row["id"]: row for row in manifests}
    truth_mapping = read(ROOT / "TRUTH.json")
    metadata = {row["id"]: old_metadata[row["historical_id"]] for row in truth_mapping}
    historical_to_new = {row["historical_id"]: row["id"] for row in truth_mapping}
    choices = {(fixture_id(row), row["arm"]): row for row in choices}
    assert len(observations) == 30 * len(method["arms"])
    spec = importlib.util.spec_from_file_location("_audit_historical_renderer", HIST / "run_mechanism.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    records, curve_cache, inputs = [], {}, {}
    failures = []
    source_lock = read(ROOT / "LOCK.json")
    for path, expected in source_lock["source_hashes"].items():
        if digest(Path(path)) != expected:
            failures.append([path, "source_hash"])
    if digest(ROOT / "OBS_INPUTS.json") != source_lock["observation_manifest_sha256"]:
        failures.append(["observation_manifest_hash"])
    if digest(ROOT / "TRUTH.json") != source_lock["truth_manifest_sha256"]:
        failures.append(["truth_manifest_hash"])
    decision_seal = read(ROOT / "decisions/SEAL.json")
    if digest(choice_path) != decision_seal["decisions_sha256"]:
        failures.append(["decision_seal"])
    for folder, stage in zip(("observed_stage", "oracle_stage"), stages):
        seal = read(ROOT / folder / "SEAL.json")
        if digest(ROOT / folder / "OBSERVATIONS.json") != seal["observations_sha256"]:
            failures.append([folder, "observation_seal"])
        if digest(ROOT / folder / "SEAL.json") != decision_seal["observation_seals"][folder]:
            failures.append([folder, "decision_chain"])
        if datetime.fromisoformat(seal["created"]) >= datetime.fromisoformat(decision_seal["created"]):
            failures.append([folder, "seal_order"])
    exported_inputs = {row["id"]: row for row in read(ROOT / "OBS_INPUTS.json")}
    for fid, item in exported_inputs.items():
        if set(item) != {"id", "cameras", "image_file", "sha256"}:
            failures.append([fid, "observation_metadata_boundary"])
        if item["cameras"] != metadata[fid]["score_cameras"]:
            failures.append([fid, "supplied_camera_identity"])
        if digest(Path(item["image_file"])) != item["sha256"]:
            failures.append([fid, "observation_image_hash"])
        with np.load(item["image_file"]) as exported, np.load(HIST / "fixtures" / metadata[fid]["arrays"]) as original:
            if exported.files != ["images"] or not np.array_equal(exported["images"], original["images"]):
                failures.append([fid, "image_identity_or_boundary"])
    allowed_ordinary = {str(ROOT / "METHOD.json"), str(ROOT / "OBS_INPUTS.json")}
    allowed_ordinary.update(item["image_file"] for item in exported_inputs.values())
    if set(stages[0]["read_log"]) - allowed_ordinary:
        failures.append(["ordinary_read_log_boundary"])
    expected_blocks = {(str(ROOT / "CANDIDATES.json"), "r"),
                       (str(ROOT / "CANDIDATES.json"), "r+"),
                       (str(ROOT / "TRUTH.json"), "r")}
    if not expected_blocks.issubset({tuple(row) for row in stages[0]["blocked_reads"]}):
        failures.append(["ordinary_negative_read_probes"])
    max_score_error = 0.0
    for observation in observations:
        fid, arm = fixture_id(observation), observation["arm"]
        meta = metadata[fid]
        path = ROOT / "curves" / observation["curve_file"]
        if not path.exists():
            path = ROOT / observation["curve_file"]
        if digest(path) != observation["curve_sha256"]:
            failures.append([fid, arm, "curve_hash"])
        with np.load(path) as raw:
            curve = {key: raw[key] for key in raw.files}
        curve_cache[(fid, arm)] = curve
        if fid not in inputs:
            with np.load(HIST / "fixtures" / meta["arrays"]) as raw:
                images, true_depth = raw["images"], float(raw["depths"][0, 64, 64])
            cameras = [{k: np.array(v, dtype=float) for k, v in c.items()}
                       for c in meta["score_cameras"]]
            ref_uv, projected, positive = coordinates(cameras, curve["grid"])
            a, va = bilinear(images[0], ref_uv)
            samples = [np.broadcast_to(a, (len(curve["grid"]), 81))]
            good = [np.broadcast_to(va, samples[0].shape)]
            for source in range(2):
                b, vb = bilinear(images[source + 1], projected[source])
                samples.append(b)
                good.append(vb)
            inputs[fid] = dict(images=images, true_depth=true_depth, ref_uv=ref_uv,
                               projected=projected, positive=np.array(positive),
                               samples=np.array(samples), good=np.array(good))
        inp = inputs[fid]
        samples = inp["samples"]
        if arm == "oracle_component":
            numerator, fraction = target_fields(renderer, meta, inp["images"].shape[1:])
            isolated = []
            for camera in range(3):
                uv = inp["ref_uv"] if camera == 0 else inp["projected"][camera - 1]
                num, _ = bilinear(numerator[camera], uv)
                frac, _ = bilinear(fraction[camera], uv)
                isolated.append(np.divide(num, frac, out=np.zeros_like(num), where=frac > 0))
            isolated[0] = np.broadcast_to(isolated[0], (len(curve["grid"]), 81))
            samples = np.array(isolated)
        recomputed, mass, ess, std = independent_scores(
            samples, curve["weights"], inp["good"], inp["positive"], method)
        finite = np.isfinite(recomputed)
        if not np.array_equal(finite, np.isfinite(curve["scores"])):
            failures.append([fid, arm, "score_validity"])
        common = finite & np.isfinite(curve["scores"])
        error = float(np.max(abs(recomputed[common] - curve["scores"][common]))) if common.any() else 0.0
        max_score_error = max(max_score_error, error)
        if error > 1e-10:
            failures.append([fid, arm, "score_value", error])
        for name, actual in (("mass", mass), ("ess", ess)):
            if name in curve and not np.allclose(curve[name], actual, atol=1e-10, rtol=0):
                failures.append([fid, arm, name])
        if not np.allclose(curve["ref_std"], std[0], atol=1e-10, rtol=0):
            failures.append([fid, arm, "reference_std"])
        if not np.allclose(curve["source_std"], std[1:], atol=1e-10, rtol=0):
            failures.append([fid, arm, "source_std"])
        accepted = np.all(np.isfinite(recomputed) & (recomputed >= method["ncc_min"]), axis=0)
        if not np.array_equal(accepted, curve["accepted"]):
            failures.append([fid, arm, "acceptance"])
        intervals = reconstruct_intervals(curve["grid"], accepted, method["padding_mm"])
        if intervals != observation["intervals"]:
            failures.append([fid, arm, "intervals"])
        mean, selected, index, gain = choose(intervals, CANDIDATES, INCUMBENT)
        sealed = choices[(fid, arm)]
        if selected != sealed["selected_depth"] or ((mean is None) != (sealed["support_mean"] is None)):
            failures.append([fid, arm, "selected_depth_or_empty"])
        if mean is not None and abs(mean - sealed["support_mean"]) > 1e-9:
            failures.append([fid, arm, "support_mean"])
        squared_gain = 0.0 if mean is None else max(0.0, (INCUMBENT - mean) ** 2 - (selected - mean) ** 2)
        if abs(squared_gain - sealed["estimated_squared_gain"]) > 1e-7:
            failures.append([fid, arm, "estimated_squared_gain"])
        truth = inp["true_depth"]
        near = sum(max(0.0, min(hi, truth + 10) - max(lo, truth - 10)) for lo, hi in intervals)
        records.append(dict(fixture_id=fid, name=meta["name"], seed=meta["seed"], arm=arm,
                            intervals=intervals, support_mean=mean, selected_depth=selected,
                            absolute_error=abs(selected - truth), true_depth=truth,
                            true_in_support=any(lo <= truth <= hi for lo, hi in intervals),
                            empty=not intervals, support_length=length(intervals),
                            off_target_length=length(intervals) - near,
                            accepted_count=int(accepted.sum()), ncc_max_replay_error=error))
    grouped = {(r["fixture_id"], r["arm"]): r for r in records}
    summary = []
    for arm in method["arms"]:
        rows = [r for r in records if r["arm"] == arm]
        bases = [grouped[(r["fixture_id"], "full9")] for r in rows]
        summary.append(dict(arm=arm, n=len(rows), true_in_support=sum(r["true_in_support"] for r in rows),
                            exact=sum(r["absolute_error"] == 0 for r in rows), empty=sum(r["empty"] for r in rows),
                            mae=float(np.mean([r["absolute_error"] for r in rows])),
                            improved_vs_full9=sum(r["absolute_error"] < b["absolute_error"] for r, b in zip(rows, bases)),
                            worse_vs_full9=sum(r["absolute_error"] > b["absolute_error"] for r, b in zip(rows, bases)),
                            unchanged_vs_full9=sum(r["absolute_error"] == b["absolute_error"] for r, b in zip(rows, bases)),
                            repaired_vs_incumbent=sum(r["absolute_error"] < 300 for r in rows),
                            baseline_exact_retained=sum(r["absolute_error"] == 0 and b["absolute_error"] == 0 for r, b in zip(rows, bases))))
    pairs = []
    for size in ("ring9", "ring25"):
        for seed in (11, 29, 47):
            first = historical_to_new[f"{size}_flat_seed{seed}"]
            second = historical_to_new[f"{size}_textured_seed{seed}"]
            for arm in method["arms"]:
                a, b = grouped[(first, arm)], grouped[(second, arm)]
                ca, cb = curve_cache[(first, arm)], curve_cache[(second, arm)]
                pairs.append(dict(pair=size, seed=seed, arm=arm,
                                  reference_center3_max_difference=float(np.max(abs(inputs[first]["images"][0, 63:66, 63:66] - inputs[second]["images"][0, 63:66, 63:66]))),
                                  acceptance_xor=int(np.count_nonzero(ca["accepted"] != cb["accepted"])),
                                  interval_symmetric_difference=symmetric_difference_length(a["intervals"], b["intervals"]),
                                  selected_depth_delta=abs(a["selected_depth"] - b["selected_depth"]),
                                  both_empty=a["empty"] and b["empty"],
                                  max_weight_difference=float(np.max(abs(ca["weights"] - cb["weights"])))))
    result = dict(observations=len(observations), selections=len(choices),
                  source_hashes_checked=len(source_lock["source_hashes"]),
                  ordinary_read_paths=len(stages[0]["read_log"]),
                  ordinary_negative_read_probes=len(stages[0]["blocked_reads"]),
                  max_score_replay_error=max_score_error, failure_count=len(failures),
                  failures=failures, summary=summary, pairs=pairs, records=records)
    (ROOT / "audit/REPLAY_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("observations", "selections", "max_score_replay_error", "failure_count", "failures", "summary")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
