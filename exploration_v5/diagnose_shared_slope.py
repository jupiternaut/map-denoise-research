"""Evaluation-only diagnosis of TWO already exposed V5 failures.

This script never generates data, searches partitions, selects candidates, or
changes frozen algorithms. Only g4/g8, seed 912101, bias RMS 4 are read. Truth
labels are loaded AFTER the legal-input fits and independent algebra check,
solely for group/node composition counts. No truth coordinate is loaded.

Writes JSON evidence plus exact source snapshots to a new diagnostic-v5-* run.
The signed RHS fractions below can include cancellation and are NOT causal
effect fractions, mixture probabilities, or a suggested filtering criterion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time

sys.dont_write_bytecode = True
import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slope_pooling as frozen

RUNS = Path("/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1")
INPUTS = RUNS / "repair-v2-oyuie4pl" / "synthetics" / "identifiable"
CASES = ("dual_g4_s912101_b4", "dual_g8_s912101_b4")
SIGMA_MM = 1.0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def diagnose(case):
    metadata_path = INPUTS / (case + ".json")
    metadata = json.loads(metadata_path.read_text())
    npz_path = Path(metadata["npz"])
    with np.load(npz_path, allow_pickle=False) as data:
        world, scans = data["xyz_world"].copy(), data["scan_id"].copy()
    # Estimation boundary: only current observed XYZ, scan IDs, supplied sigma.
    state = frozen._freeze(world, scans, SIGMA_MM)
    if state["v4_info"]["status"] != "APPLY":
        raise RuntimeError("expected an applicable already exposed diagnostic case")
    active = state["active"]
    node = state["assignment"][active]
    group = state["node_to_group"][node]
    group_count = int(group.max()) + 1
    xy = state["design"][active, 1:]
    z, weight = state["corrected"][active], state["weights"][active]
    prediction, fit = frozen._fit_model(xy, z, weight, node, state["node_to_group"],
                                        "shared_group_slope")
    before = np.asarray(state["v4_info"]["group_coefficients_common_origin"])
    H, rhs, mean_xy, mean_z = [], [], [], []
    for g in range(group_count):
        take = group == g
        x, y, w = xy[take], z[take], weight[take]
        mx, mz = np.average(x, axis=0, weights=w), np.average(y, weights=w)
        dx, dz = x-mx, y-mz
        H.append(dx.T @ (w[:, None]*dx))
        rhs.append(dx.T @ (w*dz))
        mean_xy.append(mx)
        mean_z.append(mz)
    H, rhs, mean_xy, mean_z = map(np.asarray, (H, rhs, mean_xy, mean_z))
    # Independently eliminate each group intercept, then solve just 2 slopes.
    slope = np.linalg.solve(H.sum(axis=0), rhs.sum(axis=0))
    intercept = mean_z - mean_xy @ slope
    centered_prediction = intercept[group] + xy @ slope
    max_difference = float(np.max(abs(centered_prediction-prediction)))
    if max_difference > 1e-9:
        raise AssertionError("independent weighted-centering and frozen joint SVD disagree")
    # All fit coefficients and predictions above are finalized before GT load.
    evaluation_path = npz_path.parent / "evaluation" / (case + ".eval.npz")
    with np.load(evaluation_path, allow_pickle=False) as data:
        labels_original = data["gt_layer"].copy()  # ONLY truth field accessed.
    labels = labels_original[state["order"]][active]
    layer_ids = sorted(int(k) for k in np.unique(labels))
    measured_mm = state["ordered_world"][active] * 1000.
    node_records = []
    for j, g in enumerate(state["node_to_group"]):
        take = node == j
        counts = {str(k): int(np.sum(labels[take] == k)) for k in layer_ids}
        node_records.append({"node": j, "cell": state["v4_info"]["node_cells"][j],
                             "compatible_group": int(g), "raw_point_count": int(take.sum()),
                             "evaluation_only_true_layer_counts": counts})
    group_records = []
    total_rhs_x = float(rhs[:, 0].sum())
    for g in range(group_count):
        take = group == g
        counts = {str(k): int(np.sum(labels[take] == k)) for k in layer_ids}
        raw_count = int(take.sum())
        group_records.append({
            "group": g, "original_node_ids": state["v4_info"]["group_node_ids"][g],
            "raw_point_count": raw_count,
            "evaluation_only_true_layer_counts": counts,
            "evaluation_only_minority_fraction": float(1.-max(counts.values())/raw_count),
            "measured_world_x_range_mm": [float(measured_mm[take, 0].min()), float(measured_mm[take, 0].max())],
            "measured_world_y_range_mm": [float(measured_mm[take, 1].min()), float(measured_mm[take, 1].max())],
            "measured_world_xy_centroid_mm": measured_mm[take, :2].mean(axis=0).tolist(),
            "weighted_common_xy_mean": mean_xy[g].tolist(),
            "weighted_corrected_height_mean_mm": float(mean_z[g]),
            "raw_fit_points_rejected_from_output": int(np.sum(~state["support"][active[take]])),
            "before_intercept_mm_at_common_origin": float(before[g, 0]),
            "after_intercept_mm_at_common_origin": float(intercept[g]),
            "before_slope_mm_per_mm": (before[g, 1:]/state["common_scale"]).tolist(),
            "after_slope_mm_per_mm": (slope/state["common_scale"]).tolist(),
            "weighted_within_group_H_xy": H[g].tolist(),
            "weighted_within_group_rhs_xy_z": rhs[g].tolist(),
            "signed_rhs_x_fraction": float(rhs[g, 0]/total_rhs_x) if abs(total_rhs_x) > 1e-15 else None,
            "H_xx_information_fraction": float(H[g, 0, 0]/H[:, 0, 0].sum()),
        })
    return {
        "case": case, "classification": "evaluation-only exposed-case mechanism diagnosis",
        "metadata_path": str(metadata_path), "input_npz": str(npz_path),
        "evaluation_npz": str(evaluation_path), "truth_fields_read": ["gt_layer"],
        "truth_use": "node/group composition only, after fits; not estimator, selector or new candidate",
        "legal_fit_inputs": ["xyz_world", "scan_id", "supplied_sigma_mm=1"],
        "normal_world": state["normal"].tolist(), "common_basis_world": state["basis"].tolist(),
        "common_origin_world_m": state["center"].tolist(),
        "common_tangent_scale_mm": state["common_scale"],
        "frozen_bias_mm": state["bias"].tolist(),
        "supported_fraction": float(state["support"].mean()),
        "frozen_array_sha256": frozen._fingerprint(assignment=state["assignment"], weights=state["weights"],
                                                   output_support=state["support"], node_to_group=state["node_to_group"],
                                                   bias=state["bias"], basis=state["basis"], corrected_raw=state["corrected"]),
        "shared_fit_diagnostics": fit,
        "independent_intercept_elimination": {
            "formula": "sum_g H_g slope = sum_g rhs_g; alpha_g = weighted_mean_z_g - weighted_mean_xy_g dot slope",
            "H_sum": H.sum(axis=0).tolist(), "rhs_sum": rhs.sum(axis=0).tolist(),
            "shared_slope_common_xy": slope.tolist(),
            "normal_equation_residual_norm": float(np.linalg.norm(H.sum(axis=0)@slope-rhs.sum(axis=0))),
            "max_prediction_difference_from_joint_svd_mm": max_difference,
        },
        "groups": group_records, "nodes": node_records,
        "interpretation_limits": [
            "Parallel true surfaces need not coincide with frozen mixed inferred groups.",
            "Signed RHS fractions include cancellation; they are not probabilities or causal effect shares.",
            "Counts locate an upstream-association failure; no GT-defined regrouping was performed.",
            "Full rank and small algebraic disagreement do not validate the surface model.",
            "Output-rejected points retain their original positive measurement weights by V4 design.",
        ],
    }


def run():
    if socket.gethostname().split(".")[0] != "liekkas":
        raise RuntimeError("the exact target host is liekkas")
    started = time.perf_counter()
    sources = [Path(__file__).resolve(), HERE/"slope_pooling.py", frozen._V4_PATH, frozen._V4._V3_PATH]
    protected = set(sources)
    for case in CASES:
        metadata_path = INPUTS/(case+".json")
        metadata = json.loads(metadata_path.read_text())
        npz_path = Path(metadata["npz"])
        protected.update([metadata_path, npz_path, npz_path.parent/"evaluation"/(case+".eval.npz")])
    before = {str(path): digest(path) for path in sorted(protected)}
    destination = Path(tempfile.mkdtemp(prefix="diagnostic-v5-", dir=RUNS))
    print(str(destination), flush=True)
    save_json(destination/"PROTECTED_BEFORE.json", before)
    manifest = {}
    project = HERE.parent
    for source in sources:
        snapshot = destination/"source"/source.relative_to(project)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
        manifest[str(source)] = {"sha256": digest(source), "snapshot": str(snapshot)}
    save_json(destination/"SOURCE_MANIFEST.json", manifest)
    records = [diagnose(case) for case in CASES]
    after = {path: digest(path) for path in before}
    save_json(destination/"PROTECTED_AFTER.json", after)
    if before != after:
        raise RuntimeError("protected input or source changed")
    result = {"host": socket.gethostname(), "cases": list(CASES), "case_count": len(records),
              "purpose": "evaluation-only mechanism diagnosis; no method/candidate selection",
              "new_inputs_generated": False, "parameters_changed": False, "new_candidates_generated": False,
              "frozen_algorithms_and_inputs_unchanged": True,
              "python": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__,
              "elapsed_s": time.perf_counter()-started, "diagnostics": records}
    save_json(destination/"DIAGNOSTICS.json", result)
    print(json.dumps({"output": str(destination), "case_count": len(records),
                      "maximum_algebra_difference_mm": max(r["independent_intercept_elimination"]["max_prediction_difference_from_joint_svd_mm"] for r in records),
                      "protected_unchanged": True}), flush=True)


if __name__ == "__main__":
    run()
