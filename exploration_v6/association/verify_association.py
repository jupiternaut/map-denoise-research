"""Independent saved-output metric and 2x2 contract audit; no estimator call."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
import sys

import numpy as np

ARMS = ("original_independent", "original_shared", "oracle_independent", "oracle_shared")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(**arrays):
    result = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        result.update(name.encode())
        result.update(str((value.shape, value.dtype.str)).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def scores(output, evaluation):
    """Different implementation from the main evaluators, including QR gap fits."""
    points = output*1000.
    nearest = np.full(len(points), np.inf)
    for height, xmin, xmax, ymin, ymax in evaluation["surface_rectangles_mm"]:
        dx = np.maximum(np.maximum(xmin-points[:, 0], points[:, 0]-xmax), 0.)
        dy = np.maximum(np.maximum(ymin-points[:, 1], points[:, 1]-ymax), 0.)
        nearest = np.minimum(nearest, np.sqrt(dx*dx+dy*dy+(points[:, 2]-height)**2))
    delta = output-evaluation["gt_clean_xyz_world"]
    result = dict(surface_accuracy_mean_mm=float(nearest.sum()/len(nearest)),
                  matched_point_rms_mm=float(1000*np.sqrt(np.einsum("ij,ij->", delta, delta)/len(delta))))
    fits = {}
    for label in np.unique(evaluation["gt_layer"]):
        take = points[evaluation["gt_layer"] == label]
        A = np.column_stack((take[:, 0], take[:, 1], np.ones(len(take))))
        if len(take) >= 6 and np.linalg.matrix_rank(A) == 3:
            Q, R = np.linalg.qr(A, mode="reduced")
            fits[int(label)] = np.linalg.solve(R, Q.T@take[:, 2])
    if 0 in fits and 1 in fits:
        gap = float(fits[1][2]-fits[0][2])
        result.update(fitted_gap_at_same_xy_mm=gap,
                      fitted_gap_at_same_xy_error_mm=abs(gap-float(evaluation["true_gap_mm"])))
    return result


def verify(dest):
    assert socket.gethostname().split(".")[0] == "liekkas"
    rows = list(csv.DictReader((dest/"RESULTS.csv").open()))
    assert len(rows) == 96 and len({(r["case"], r["arm"]) for r in rows}) == 96
    inputs = {r["case"]: r for r in json.loads((dest/"INPUT_MANIFEST.json").read_text())}
    checked, max_metric_difference, max_sse_difference = 0, 0., 0.
    bycase = {}
    recomputed = {}
    for row in rows:
        case, arm = row["case"], row["arm"]
        record = inputs[case]
        assert row["ok"] == "True"
        target = Path(row["output"])
        assert digest(target) == row["output_sha256"]
        assert digest(record["input"]) == record["input_sha256"] == digest(dest/"inputs"/(case+".npz"))
        assert digest(record["evaluation"]) == record["evaluation_sha256"]
        with np.load(dest/"inputs"/(case+".npz"), allow_pickle=False) as data:
            world, scans, ids = data["xyz_world"], data["scan_id"], data["source_point_index"]
        with np.load(target, allow_pickle=False) as data:
            output, mask, groups = data["xyz_world"], data["support_mask"], data["group_assignment"]
            active_original = data["active_original_indices"]
            np.testing.assert_array_equal(scans, data["scan_id"])
            np.testing.assert_array_equal(ids, data["source_point_index"])
        assert output.shape == world.shape and np.isfinite(output).all()
        np.testing.assert_array_equal(output[~mask], world[~mask])
        info = json.loads(target.with_suffix(".json").read_text())["info"]
        with np.load(dest/"states"/(case+".npz"), allow_pickle=False) as data:
            state = {key: data[key] for key in data.files}
        state_info = json.loads((dest/"states"/(case+".json")).read_text())
        assert digest(dest/"states"/(case+".npz")) == state_info["state_npz_sha256"]
        np.testing.assert_array_equal(state["world"], world)
        np.testing.assert_array_equal(state["scan_id"], scans)
        keys = ("world", "scan_id", "order", "bias", "basis", "normal", "center", "design",
                "corrected", "weights", "active", "support", "assignment", "node_to_group")
        actual_fingerprints = {key+"_sha256": fingerprint(**{key: state[key]}) for key in keys}
        actual_fingerprints["frozen_common_sha256"] = fingerprint(
            **{key: state[key] for key in keys}, sigma_mm=np.asarray(state_info["sigma_mm"]),
            common_scale_mm=state["common_scale_mm"])
        assert actual_fingerprints == info["common_fingerprints"] == state_info["fingerprints"]
        order, active = state["order"], state["active"]
        np.testing.assert_array_equal(active_original, order[active])
        np.testing.assert_array_equal(mask[order], state["support"])
        np.testing.assert_array_equal(np.flatnonzero(groups >= 0), np.sort(active_original))
        assert info["active_point_count"] == len(active)
        weight = state["weights"][active]
        assert info["total_fit_weight"] == float(weight.sum())
        fit_groups = groups[active_original]
        slopes = np.asarray(info["group_slopes_common_xy"])
        intercepts = np.asarray(info["intercepts_mm_at_common_origin"])
        prediction = intercepts[fit_groups]+np.sum(state["design"][active, 1:]*slopes[fit_groups], axis=1)
        residual = state["corrected"][active]-prediction
        sse = float(np.sum(weight*residual**2))
        max_sse_difference = max(max_sse_difference, abs(sse-info["weighted_measurement_sse_mm2"]))
        assert abs(sse-info["weighted_measurement_sse_mm2"]) < 1e-8
        # Verify fitted coefficients produce actual saved output with fixed mask.
        height = state["corrected"].copy()
        height[active] = prediction
        expected_ordered = world[order].copy()
        use = state["support"]
        expected_ordered[use] += ((height[use]-state["local"][use, 2])/1000.)[:, None]*state["normal"]
        np.testing.assert_allclose(output[order], expected_ordered, rtol=0, atol=1e-14)
        with np.load(record["evaluation"], allow_pickle=False) as data:
            ev = {key: data[key] for key in data.files if key != "json"}
            ev.update(json.loads(data["json"].tobytes().decode()))
        labels = ev["gt_layer"]
        original_groups = state["original_groups"]
        for group in np.unique(fit_groups):
            take = fit_groups == group
            assert len(np.unique(original_groups[take])) == 1
            if arm.startswith("oracle"):
                assert len(np.unique(labels[active_original[take]])) == 1
        if arm.startswith("original"):
            np.testing.assert_array_equal(fit_groups, original_groups)
            assert info["truth_fields_used"] == []
        else:
            expected_pairs = np.c_[original_groups, labels[active_original]]
            _, expected_groups = np.unique(expected_pairs, axis=0, return_inverse=True)
            np.testing.assert_array_equal(fit_groups, expected_groups)
            assert info["truth_fields_used"] == ["gt_layer"]
        values = scores(output, ev)
        for key, value in values.items():
            difference = abs(float(row[key])-value)
            max_metric_difference = max(max_metric_difference, difference)
            assert difference < 1e-8, (case, arm, key, difference)
            checked += 1
        recomputed[case, arm] = values
        bycase.setdefault(case, []).append(info)
    for case, infos in bycase.items():
        assert len(infos) == 4
        for key in ("common_fingerprints", "frozen_common_sha256", "active_point_count", "total_fit_weight", "supported_fraction"):
            assert all(info[key] == infos[0][key] for info in infos)
    contrasts = list(csv.DictReader((dest/"CONTRASTS.csv").open()))
    for row in contrasts:
        case = row["case"]
        for metric in ("surface_accuracy_mean_mm", "matched_point_rms_mm", "fitted_gap_at_same_xy_error_mm"):
            field = metric+"__difference_in_differences"
            if not row.get(field):
                continue
            original = recomputed[case, "original_shared"][metric]-recomputed[case, "original_independent"][metric]
            oracle = recomputed[case, "oracle_shared"][metric]-recomputed[case, "oracle_independent"][metric]
            assert abs(float(row[field])-(oracle-original)) < 1e-8
    before = json.loads((dest/"PROTECTED_BEFORE.json").read_text())
    assert before == json.loads((dest/"PROTECTED_AFTER.json").read_text())
    for path, sha in before.items():
        assert digest(path) == sha, path
    project = Path(__file__).resolve().parent.parents[1]
    source_manifest = json.loads((dest/"SOURCE_MANIFEST.json").read_text())
    for path, sha in source_manifest.items():
        assert digest(path) == sha
        assert digest(dest/"source"/Path(path).relative_to(project)) == sha
    replay = json.loads((dest/"REPRODUCTION.json").read_text())
    for r in replay:
        with np.load(r["reference"]) as old, np.load(dest/"outputs"/(r["case"]+"__"+r["arm"]+".npz")) as new:
            difference = float(np.max(abs(old["xyz_world"]-new["xyz_world"])))
        assert difference == r["max_abs_world_coordinate_difference_m"] and difference <= 1e-12
    report = dict(status="PASS", outputs_checked=len(rows), frozen_common_quartets_checked=len(bycase),
                  independently_recomputed_scores=checked, max_metric_difference_mm=max_metric_difference,
                  max_weighted_sse_difference_mm2=max_sse_difference, replay_outputs=len(replay),
                  max_v4_v5_replay_difference_mm=max(r["max_abs_world_coordinate_difference_mm"] for r in replay),
                  bitwise_replays=sum(r["bitwise_equal"] for r in replay), protected_files=len(before),
                  protected_unchanged=True, no_new_estimator_or_data_run=True)
    with (dest/"VERIFICATION.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    verify(parser.parse_args().run_dir.resolve())
