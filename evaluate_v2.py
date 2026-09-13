"""World-space geometry scores, separate from algorithm self-reports.

Synthetic truth is finite z-constant rectangles in millimetres. No inferred K
is used to score geometry. Reference sample coverage is sampling dependent.
Source-group gap is a correspondence diagnostic, NOT a physical layer count.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def reported_model_k(info):
    for blob in (info, info.get("backend_info") or {}):
        for key in ("k", "K", "n_layers", "selected_k"):
            if blob.get(key) is not None:
                return int(blob[key])
        fit = blob.get("fit")
        if isinstance(fit, dict) and len(fit) == 1:
            entry = next(iter(fit.values()))
            if isinstance(entry, dict) and entry.get("k") is not None:
                return int(entry["k"])
    return None


def geometry_metrics(output_world, eval_data):
    """Geometry-only and permutation invariant; no algorithm metadata input."""
    out = np.asarray(output_world, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 3 or not len(out) or not np.isfinite(out).all():
        raise ValueError("output must contain finite world-space N x 3 points in metres")
    if not bool(eval_data.get("identifiable", True)):
        return {"identifiable": False, "geometry_scored": False,
                "geometry_note": "scalar ambiguity control; not scored as a unique physical world"}
    rects = np.asarray(eval_data["surface_rectangles_mm"], dtype=float)
    if rects.ndim != 2 or rects.shape[1] != 5:
        raise ValueError("rectangles must be [z, xmin, xmax, ymin, ymax] in mm")
    q = out * 1000.0
    distances = []
    for z, x0, x1, y0, y1 in rects:
        dx = q[:, 0] - np.clip(q[:, 0], x0, x1)
        dy = q[:, 1] - np.clip(q[:, 1], y0, y1)
        distances.append(np.sqrt(dx * dx + dy * dy + (q[:, 2] - z) ** 2))
    d = np.min(distances, axis=0)
    ref = np.asarray(eval_data["gt_clean_xyz_world"], dtype=float)
    coverage_distance = cKDTree(out).query(ref)[0] * 1000.0
    return {
        "identifiable": True, "geometry_scored": True,
        "surface_accuracy_mean_mm": float(np.mean(d)),
        "surface_accuracy_rms_mm": float(np.sqrt(np.mean(d ** 2))),
        "surface_accuracy_p95_mm": float(np.quantile(d, 0.95)),
        "reference_sample_coverage_1mm": float(np.mean(coverage_distance <= 1.0 + 1e-9)),
        "reference_sample_coverage_2mm": float(np.mean(coverage_distance <= 2.0 + 1e-9)),
        "reference_sample_distance_mean_mm": float(np.mean(coverage_distance)),
        "output_point_count_ratio": float(len(out) / len(ref)),
    }


def source_correspondence_metrics(output_world, eval_data):
    """Requires original point IDs/order; not a point-set classifier."""
    if not bool(eval_data.get("identifiable", True)):
        return {}
    out = np.asarray(output_world, dtype=float)
    ref = np.asarray(eval_data["gt_clean_xyz_world"], dtype=float)
    if out.shape != ref.shape:
        raise ValueError("source-correspondence scores require original point order/count")
    labels = np.asarray(eval_data["gt_layer"], dtype=int)
    z = out[:, 2] * 1000.0
    zref = ref[:, 2] * 1000.0
    means = {int(k): float(np.mean(z[labels == k])) for k in np.unique(labels)}
    residual = z - np.array([means[int(k)] for k in labels])
    result = {
        "matched_point_rms_mm": point_rms_mm(out, ref),
        "matched_normal_mae_mm": float(np.mean(np.abs(z - zref))),
        "within_source_layer_rms_mm": float(np.sqrt(np.mean(residual ** 2))),
    }
    if int(eval_data["n_true_layers"]) == 2:
        gap = float(eval_data["true_gap_mm"])
        estimated = means[1] - means[0]
        result.update(source_group_gap_mm=estimated,
                      source_group_gap_error_mm=abs(estimated - gap),
                      source_group_gap_retention=estimated / gap)
    return result


def synthetic_geometry(output_world, info, eval_data):
    result = geometry_metrics(output_world, eval_data)
    result.update(source_correspondence_metrics(output_world, eval_data))
    result["reported_model_k"] = reported_model_k(info or {})
    result["model_k_note"] = "self-report; not geometry-derived layer count or success"
    return result


def point_rms_mm(output_world, reference_world):
    delta = np.asarray(output_world) - np.asarray(reference_world)
    return float(1000 * np.sqrt(np.mean(np.sum(delta ** 2, axis=1))))
