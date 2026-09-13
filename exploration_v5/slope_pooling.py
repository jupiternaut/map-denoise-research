"""V5 conditional slope/intercept ablation on a completely frozen V4 state.

No partition search, new association, target selection, GPU path, or truth input.
`v4_compatible` returns the exact V4 output. `shared_group_slope` keeps each
compatible group's intercept but forces all groups to have parallel slopes.
`node_intercepts` shares slopes only inside each compatible group and frees each
original local node's intercept. All measurement weights and output support are
fixed before selecting the variant. Neither new fit observes projected XYZ.

Parallelism is a hypothesis, not a guarantee. Bad initial associations remain
bad, and nonparallel surfaces can be damaged by shared_group_slope. Rank and
conditioning diagnostics describe the conditional fit, not geometric truth.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


_PROJECT = Path(__file__).resolve().parent.parent
_V4_PATH = _PROJECT / "exploration_v4" / "surface_pooling.py"
_SPEC = importlib.util.spec_from_file_location("_v5_frozen_surface_pooling", _V4_PATH)
_V4 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V4)
VARIANTS = ("v4_compatible", "shared_group_slope", "node_intercepts")
FROZEN_SOURCE_SHA256 = {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (_V4_PATH, _V4._V3_PATH)
}


def _fingerprint(**arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _freeze(world, frames, sigma):
    """Call unchanged V4, then reconstruct its raw-measurement sufficient state.

V4's projected output is retained only for the exact-reproduction arm. All
other arrays below depend on original XYZ and V4's initial diagnostics, not
on projected output coordinates. Reconstructed assignments and support are
checked against V4's recorded state rather than choosing new groups.
    """
    start = time.perf_counter()
    baseline, old = _V4.estimate(world, frames, sigma, variant="compatible")
    v4_seconds = time.perf_counter() - start
    order = _V4._canonical_order((world - world.mean(axis=0)) * 1000., frames)
    ordered = world[order]
    ids, scans = np.unique(frames[order], return_inverse=True)
    n = len(world)
    state = {"baseline": baseline, "v4_info": old, "order": order,
             "ordered_world": ordered, "ids": ids, "scans": scans,
             "bias": np.asarray(old["bias_mm"], dtype=float),
             "support": np.zeros(n, dtype=bool), "weights": np.ones(n),
             "assignment": np.full(n, -1, dtype=np.int64), "node_to_group": np.empty(0, dtype=np.int64),
             "v4_call_seconds": v4_seconds, "normal": np.asarray(old.get("normal_world") or [0., 0., 0.])}
    if old["status"] != "APPLY":
        state["state_reconstruction_seconds"] = time.perf_counter() - start - v4_seconds
        return state
    center = ordered.mean(axis=0)
    centered = (ordered - center) * 1000.
    basis, _ = _V4._V3._basis(centered, scans, sigma)
    local = centered @ basis
    labels, members, geometry, grid = _V4._V3._cells(local)
    corrected = local[:, 2] - state["bias"][scans]
    scale = max(float(np.sqrt(np.mean(np.sum(local[:, :2] ** 2, axis=1)))), sigma)
    design = np.c_[np.ones(n), local[:, :2] / scale]
    node_cells, node_layers, node_counts, local_k = [], [], [], {}
    for cell, take in enumerate(members):
        if len(take) < 24 or len(np.unique(scans[take])) < 2:
            continue
        xy, local_scale = geometry[cell]
        model = _V4._V3._local_model(corrected[take], xy, local_scale, sigma)
        local_k[str(cell)] = model["k"]
        slope = model["beta"][model["k"]:] / local_scale
        fit_ok = np.median(abs(model["residual"])) <= 2.5 * sigma and np.linalg.norm(slope) <= .25
        accepted = (abs(model["residual"]) <= 4. * sigma) & (model["confidence"] >= .8) if fit_ok else np.zeros(len(take), dtype=bool)
        state["support"][take[accepted]] = True
        state["weights"][take] = np.minimum(1., 3. * sigma / np.maximum(abs(model["residual"]), 1e-12))
        for layer in range(model["k"]):
            indices = take[model["assigned"] == layer]
            state["assignment"][indices] = len(node_cells)
            node_cells.append(cell)
            node_layers.append(layer)
            node_counts.append(len(indices))
    if (node_cells != old["node_cells"] or node_layers != old["node_local_layers"]
            or node_counts != old["node_point_counts"] or local_k != old["local_k"]
            or float(state["support"].mean()) != old["supported_fraction"]):
        raise RuntimeError("reconstructed V4 state does not match the frozen initialization")
    # This is the original complete-link grouping, not a new clustering pass.
    mapping = np.asarray(old["node_to_group"], dtype=np.int64)
    for group, nodes in enumerate(old["group_node_ids"]):
        if not np.all(mapping[nodes] == group):
            raise RuntimeError("inconsistent frozen compatible grouping")
    state.update(center=center, basis=basis, normal=basis[:, 2], local=local,
                 corrected=corrected, design=design, common_scale=scale,
                 node_to_group=mapping, node_cells=np.asarray(node_cells, dtype=np.int64),
                 active=np.flatnonzero(state["assignment"] >= 0))
    state["state_reconstruction_seconds"] = time.perf_counter() - start - v4_seconds
    return state


def _fit_model(xy, values, weights, node_ids, node_to_group, variant, frozen_betas=None):
    """Conditional weighted affine least squares; no regrouping or new weights.

The internal labels are reconstructed input-derived V4 nodes, not GT layers.
SVD diagnostics use the weighted DESIGN (not the squared normal equations).
    """
    groups = node_to_group[node_ids]
    group_count = int(node_to_group.max()) + 1
    node_count = len(node_to_group)
    intercept_ids = node_ids if variant == "node_intercepts" else groups
    intercept_count = node_count if variant == "node_intercepts" else group_count
    slope_ids = np.zeros(len(values), dtype=int) if variant == "shared_group_slope" else groups
    slope_count = 1 if variant == "shared_group_slope" else group_count
    design = np.zeros((len(values), intercept_count + 2 * slope_count))
    rows = np.arange(len(values))
    design[rows, intercept_ids] = 1.
    design[rows, intercept_count + 2 * slope_ids] = xy[:, 0]
    design[rows, intercept_count + 2 * slope_ids + 1] = xy[:, 1]
    root_weight = np.sqrt(weights)
    weighted = design * root_weight[:, None]
    beta, _, rank, singular = np.linalg.lstsq(weighted, values * root_weight, rcond=1e-12)
    if variant == "v4_compatible":
        # Preserve the original V4 solve and rounding, not a new SVD solution.
        coefficients = np.asarray(frozen_betas, dtype=float)
        beta = np.r_[coefficients[:, 0], coefficients[:, 1:].ravel()]
    prediction = design @ beta
    residual = values - prediction
    full_rank = int(rank) == design.shape[1]
    condition = float(singular[0] / singular[-1]) if full_rank else None
    effective_condition = float(singular[0] / singular[int(rank) - 1]) if rank else None
    slopes = beta[intercept_count:].reshape(slope_count, 2)
    group_slopes = np.repeat(slopes, group_count, axis=0) if slope_count == 1 else slopes
    return prediction, {
        "fit_rank": int(rank), "fit_parameter_count": design.shape[1], "fit_full_rank": full_rank,
        "fit_condition_number": condition, "fit_effective_condition_number": effective_condition,
        "fit_singular_values": singular.tolist(),
        "fit_condition_scope": "weighted normalized design; null means rank deficient, not infinity hidden",
        "fit_solver": "frozen V4 sufficient-statistic coefficients" if variant == "v4_compatible" else "numpy weighted least squares, rcond=1e-12",
        "intercept_scope": "original local node" if variant == "node_intercepts" else "frozen compatible group",
        "slope_scope": "all compatible groups" if variant == "shared_group_slope" else "within each compatible group",
        "intercepts_mm_at_common_origin": beta[:intercept_count].tolist(),
        "group_slopes_common_xy": group_slopes.tolist(),
        "weighted_measurement_sse_mm2": float(np.dot(weights * residual, residual)),
        "measurement_residual_rms_mm": float(np.sqrt(np.mean(residual ** 2))),
    }


def estimate(xyz_world_m, scan_id, sigma_mm, variant="shared_group_slope"):
    """Return same-order world metres and JSON diagnostics; legal inputs only."""
    start = time.perf_counter()
    world = np.asarray(xyz_world_m, dtype=float)
    frames = np.asarray(scan_id)
    sigma = float(sigma_mm)
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if world.ndim != 2 or world.shape[1] != 3 or not len(world) or not np.isfinite(world).all():
        raise ValueError("finite nonempty N by 3 world metres required")
    if frames.shape != (len(world),) or frames.dtype.kind not in "iu":
        raise ValueError("one integer scan ID per point required")
    if not np.isfinite(sigma) or sigma <= 0.:
        raise ValueError("sigma_mm must be positive and finite")
    state = _freeze(world, frames, sigma)
    support_original = np.empty(len(world), dtype=bool)
    assignment_original = np.empty(len(world), dtype=np.int64)
    weights_original = np.empty(len(world))
    support_original[state["order"]] = state["support"]
    assignment_original[state["order"]] = state["assignment"]
    weights_original[state["order"]] = state["weights"]
    initialization = _fingerprint(xyz_world_m=world, scan_id=frames, sigma_mm=np.asarray(sigma),
                                  order=state["order"], bias_mm=state["bias"], normal=state["normal"],
                                  basis=state.get("basis", np.zeros((3, 3))),
                                  design=state.get("design", np.empty((0, 3))),
                                  corrected_raw_mm=state.get("corrected", np.empty(0)),
                                  assignment=assignment_original, weights=weights_original)
    grouping = _fingerprint(node_to_group=state["node_to_group"])
    mask_hash = _fingerprint(support=support_original)
    frozen_hash = hashlib.sha256(json.dumps([initialization, grouping, mask_hash]).encode()).hexdigest()
    info = {"method": "slope_pooling_v5", "variant": variant, "point_count": len(world),
            "input_units": "world metres", "sigma_mm": sigma,
            "scan_ids": state["ids"].tolist(), "bias_mm": state["bias"].tolist(),
            "normal_world": state["normal"].tolist() if np.linalg.norm(state["normal"]) > 0 else None,
            "gauge": state["v4_info"]["gauge"],
            "gauge_point_counts": state["v4_info"].get("gauge_point_counts", []),
            "initialization_sha256": initialization, "grouping_sha256": grouping,
            "support_mask_sha256": mask_hash, "frozen_state_sha256": frozen_hash,
            "local_assignment_sha256": _fingerprint(assignment=assignment_original),
            "weights_sha256": _fingerprint(weights=weights_original),
            "unsupported_point_indices": np.flatnonzero(~support_original).tolist(),
            "supported_fraction": float(support_original.mean()),
            "unchanged_fraction": float(1. - support_original.mean()),
            "frozen_source_sha256": dict(FROZEN_SOURCE_SHA256),
            "initialization_source": "unchanged V4 compatible on this current legal input",
            "observation_source": "original normal coordinates minus frozen V4 scan bias",
            "projection_used_as_observations": False,
            "frozen_fields": ["canonical ordering", "bias", "normal", "local assignment", "weights", "compatible grouping", "output support"],
            "v4_call_seconds": state["v4_call_seconds"],
            "state_reconstruction_seconds": state["state_reconstruction_seconds"],
            "v4_info": state["v4_info"],
            "interpretation": "conditional parameter-sharing experiment; no probability or gap-preservation guarantee"}
    if state["v4_info"]["status"] != "APPLY":
        result = state["baseline"].copy()
        info.update(status="UNSUPPORTED", fit_rank=0, fit_parameter_count=0, fit_full_rank=False,
                    fit_condition_number=None, fit_effective_condition_number=None, fit_seconds=0.,
                    reason="frozen V4 initialization unsupported")
    else:
        active = state["active"]
        fit_start = time.perf_counter()
        prediction, fit_info = _fit_model(state["design"][active, 1:], state["corrected"][active],
                                         state["weights"][active], state["assignment"][active],
                                         state["node_to_group"], variant,
                                         state["v4_info"]["group_coefficients_common_origin"])
        info.update(fit_info, status="APPLY", fit_seconds=time.perf_counter() - fit_start,
                    local_node_count=len(state["node_to_group"]),
                    pooled_surface_count=state["v4_info"]["pooled_surface_count"],
                    group_node_ids=state["v4_info"]["group_node_ids"],
                    node_to_group=state["node_to_group"].tolist(),
                    common_plane_origin_world_m=state["center"].tolist(),
                    common_tangent_scale_mm=state["common_scale"],
                    rank_deficient_fit_policy="minimum-norm coefficients; in-sample predictions remain defined; no physical extrapolation guarantee")
        if variant == "v4_compatible":
            result = state["baseline"].copy()
        else:
            predicted = state["corrected"].copy()
            predicted[active] = prediction
            ordered_output = state["ordered_world"].copy()
            use = state["support"]
            ordered_output[use] += ((predicted[use] - state["local"][use, 2]) / 1000.)[:, None] * state["normal"]
            result = world.copy()
            result[state["order"]] = ordered_output
    if result.shape != world.shape or not np.isfinite(result).all():
        raise FloatingPointError("invalid same-order world output")
    if not np.array_equal(result[~support_original], world[~support_original]):
        raise RuntimeError("fixed unsupported rows were changed")
    info.update(output_array_sha256=_fingerprint(xyz_world_m=result),
                input_edit_rms_mm=float(np.sqrt(np.mean(np.sum((result-world) ** 2, axis=1)))) * 1000.,
                seconds=time.perf_counter() - start)
    info["total_seconds"] = info["seconds"]
    return result, info
