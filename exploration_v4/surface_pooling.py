"""Conditional cross-cell surface pooling after a shared V3 bias estimate.

All variants use the same input-derived normal, bias, cell/layer assignments,
and acceptance mask. Fits use bias-corrected ORIGINAL normal measurements,
never the V3 projected coordinates as repeated, independent observations.
The independent arm fits one affine surface per assigned local node; global
fits a single affine surface to all active points (a deliberate damage
control); compatible pools only mutually compatible local nodes. This is an
exposed-development prototype, not a physical layer-identifiability test.
"""
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import time

import numpy as np


# Read-only reuse of the frozen V3 implementation at its explicit sibling path.
_V3_PATH = Path(__file__).resolve().parent.parent / "exploration_v3" / "graph_surface.py"
_V3_SHA256 = hashlib.sha256(_V3_PATH.read_bytes()).hexdigest()
_SPEC = importlib.util.spec_from_file_location("_v4_readonly_graph_surface", _V3_PATH)
_V3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V3)


def _canonical_order(centered_mm, scans):
    """Permutation-independent ordering with rigid-coordinate invariant keys.

V3 chooses a subset of normal candidates by input position. Ordering within
each scan by distances (not world XYZ) removes that avoidable input-order
dependence. Sub-nanometre squared-distance rounding only stabilizes ties.
    """
    radius = np.empty(len(centered_mm))
    for sid in np.unique(scans):
        mask = scans == sid
        delta = centered_mm[mask] - centered_mm[mask].mean(axis=0)
        radius[mask] = np.sum(delta * delta, axis=1)
    global_radius = np.sum(centered_mm * centered_mm, axis=1)
    return np.lexsort((np.round(global_radius, 8), np.round(radius, 8), scans))


def _sufficient_fit(indices, design, values, weights, sigma):
    x, y, w = design[indices], values[indices], weights[indices]
    lhs = x.T @ (w[:, None] * x)
    rhs = x.T @ (w * y)
    inverse = np.linalg.pinv(lhs, rcond=1e-12)
    beta = inverse @ rhs
    residual = y - x @ beta
    return {"indices": indices, "lhs": lhs, "rhs": rhs,
            "yy": float(np.dot(w * y, y)), "beta": beta,
            "cov": sigma ** 2 * inverse,
            "sse": float(np.dot(w * residual, residual)),
            "rank": int(np.linalg.matrix_rank(lhs, tol=1e-9))}


def _joined_fit(a, b, sigma):
    lhs, rhs = a["lhs"] + b["lhs"], a["rhs"] + b["rhs"]
    inverse = np.linalg.pinv(lhs, rcond=1e-12)
    beta = inverse @ rhs
    yy = a["yy"] + b["yy"]
    sse = max(0., float(yy - 2. * np.dot(beta, rhs) + beta @ lhs @ beta))
    return {"indices": np.r_[a["indices"], b["indices"]], "lhs": lhs,
            "rhs": rhs, "yy": yy, "beta": beta, "cov": sigma ** 2 * inverse,
            "sse": sse, "rank": int(np.linalg.matrix_rank(lhs, tol=1e-9))}


def _compatibility(nodes, sigma):
    """Fixed complete-link admissibility, not single-link transitive closure.

Every coefficient is expressed at the SAME patch centroid and XY scale.
The squared Mahalanobis discrepancy is a conditional confidence diagnostic
using supplied sigma, not a calibrated post-selection hypothesis test.
12.84 is the approximate 99.5% chi-square(3) quantile. Nodes assigned to
different layers of the same cell cannot merge, even through other cells.
    """
    count = len(nodes)
    allowed = np.eye(count, dtype=bool)
    discrepancy = np.full((count, count), np.inf)
    np.fill_diagonal(discrepancy, 0.)
    for i in range(count):
        for j in range(i):
            a, b = nodes[i], nodes[j]
            if a["cell"] == b["cell"] or min(a["rank"], b["rank"]) < 3:
                continue
            delta = a["beta"] - b["beta"]
            inverse = np.linalg.pinv(a["cov"] + b["cov"], rcond=1e-12)
            distance = float(delta @ inverse @ delta)
            discrepancy[i, j] = discrepancy[j, i] = distance
            allowed[i, j] = allowed[j, i] = distance <= 12.84
    return allowed, discrepancy


def _pool(nodes, allowed, sigma, point_count):
    """Greedy positive data-plus-complexity gain subject to complete linkage."""
    groups = [{"members": [i], "fit": node} for i, node in enumerate(nodes)]
    history = []
    # Conditional weighted-Gaussian BIC-like score: SSE/sigma^2 + 3G log N.
    # Weights/assignments were estimated upstream and frozen for all arms;
    # this is not an unconditional likelihood or evidence for the true K.
    parameter_saving = 3. * np.log(max(point_count, 2))
    while len(groups) > 1:
        best = None
        for i in range(len(groups)):
            for j in range(i):
                a, b = groups[i], groups[j]
                if not allowed[np.ix_(a["members"], b["members"])].all():
                    continue
                fit = _joined_fit(a["fit"], b["fit"], sigma)
                gain = (a["fit"]["sse"] + b["fit"]["sse"] - fit["sse"]) / sigma ** 2 + parameter_saving
                if gain > 0. and (best is None or gain > best[0]):
                    best = (float(gain), i, j, fit)
        if best is None:
            break
        gain, i, j, fit = best
        members = sorted(groups[i]["members"] + groups[j]["members"])
        history.append({"members": members, "score_gain": gain})
        groups.pop(i)
        groups.pop(j)
        groups.append({"members": members, "fit": fit})
    return groups, history


def estimate(xyz_world_m, scan_id, sigma_mm, variant="compatible"):
    """Return same-shape/order world metres plus JSON-friendly diagnostics.

The only legal inputs are measured XYZ, scan IDs and the supplied noise sigma.
The fixed point-count-weighted V3 gauge cannot recover an unobserved absolute
map translation. No ray, injection, ground-truth normal or layer is consumed.
    """
    started = time.perf_counter()
    world = np.asarray(xyz_world_m, dtype=float)
    frames = np.asarray(scan_id)
    sigma = float(sigma_mm)
    if variant not in ("independent", "global", "compatible"):
        raise ValueError("variant must be independent, global, or compatible")
    if world.ndim != 2 or world.shape[1] != 3 or not len(world) or not np.isfinite(world).all():
        raise ValueError("finite nonempty N by 3 world coordinates required")
    if frames.shape != (len(world),) or frames.dtype.kind not in "iu":
        raise ValueError("one integer scan ID per point required")
    if not np.isfinite(sigma) or sigma <= 0.:
        raise ValueError("sigma_mm must be positive and finite")

    center = world.mean(axis=0)
    centered = (world - center) * 1000.
    order = _canonical_order(centered, frames)
    ordered_world = world[order]
    ids, scans = np.unique(frames[order], return_inverse=True)
    # Explicitly discard projected XYZ: they are not observations for pooling.
    _, initial = _V3.estimate(ordered_world, frames[order], sigma, variant="graph")
    info = {"method": "surface_pooling", "variant": variant,
            "point_count": len(world), "scan_ids": ids.tolist(), "sigma_mm": sigma,
            "input_units": "m", "initial_method": "V3 graph; rigid-invariant scan ordering",
            "initial_source_file": str(_V3_PATH), "initial_source_sha256": _V3_SHA256,
            "observation_source": "original measurements minus estimated scan-normal bias",
            "scope": "one common normal; frozen local layer assignments; conditional planar pooling",
            "same_initialization_and_support_across_variants": True,
            "initial_info": initial, "bias_mm": initial["bias_mm"],
            "gauge": initial["gauge"], "gauge_point_counts": initial.get("gauge_point_counts", []),
            "normal_world": initial.get("normal_world"),
            "canonical_order": "scan ID, within-scan squared radius, patch squared radius"}
    if initial["status"] != "APPLY":
        info.update(status="UNSUPPORTED", reason="shared initial geometry has no accepted points",
                    supported_fraction=0., unchanged_fraction=1., seconds=time.perf_counter() - started)
        return world.copy(), info

    # Repeating this deterministic read-only helper reproduces the initial basis.
    ordered_center = ordered_world.mean(axis=0)
    centered = (ordered_world - ordered_center) * 1000.
    basis, _ = _V3._basis(centered, scans, sigma)
    local = centered @ basis
    labels, members, geometry, grid = _V3._cells(local)
    bias = np.asarray(initial["bias_mm"])
    corrected = local[:, 2] - bias[scans]
    common_scale = max(float(np.sqrt(np.mean(np.sum(local[:, :2] ** 2, axis=1)))), sigma)
    design = np.c_[np.ones(len(local)), local[:, :2] / common_scale]
    weights = np.ones(len(local))
    support = np.zeros(len(local), dtype=bool)
    assignment = np.full(len(local), -1, dtype=int)
    local_k, node_records = {}, []
    for cell, take in enumerate(members):
        if len(take) < 24 or len(np.unique(scans[take])) < 2:
            continue
        xy, scale = geometry[cell]
        model = _V3._local_model(corrected[take], xy, scale, sigma)
        local_k[str(cell)] = model["k"]
        slope = model["beta"][model["k"]:] / scale
        fit_ok = np.median(abs(model["residual"])) <= 2.5 * sigma and np.linalg.norm(slope) <= .25
        accepted = (abs(model["residual"]) <= 4. * sigma) & (model["confidence"] >= .8) if fit_ok else np.zeros(len(take), dtype=bool)
        support[take[accepted]] = True
        # Freeze one robust measurement weight for every assigned raw point.
        # Hard upstream layer assignments remain a stated limitation at small gaps.
        weights[take] = np.minimum(1., 3. * sigma / np.maximum(abs(model["residual"]), 1e-12))
        for layer in range(model["k"]):
            indices = take[model["assigned"] == layer]
            assignment[indices] = len(node_records)
            node_records.append({"cell": cell, "local_layer": layer, "indices": indices})

    nodes = []
    for record in node_records:
        fit = _sufficient_fit(record["indices"], design, corrected, weights, sigma)
        fit.update(cell=record["cell"], local_layer=record["local_layer"])
        nodes.append(fit)
    active = np.flatnonzero(assignment >= 0)
    allowed, discrepancies = _compatibility(nodes, sigma)
    history = []
    if variant == "global":
        groups = [{"members": list(range(len(nodes))),
                   "fit": _sufficient_fit(active, design, corrected, weights, sigma)}]
    elif variant == "independent":
        groups = [{"members": [i], "fit": node} for i, node in enumerate(nodes)]
    else:
        groups, history = _pool(nodes, allowed, sigma, len(active))

    prediction = corrected.copy()
    node_to_group = np.empty(len(nodes), dtype=int)
    for group_id, group in enumerate(groups):
        fit = group["fit"]
        prediction[fit["indices"]] = design[fit["indices"]] @ fit["beta"]
        node_to_group[group["members"]] = group_id
    ordered_output = ordered_world.copy()
    ordered_output[support] += ((prediction[support] - local[support, 2]) / 1000.)[:, None] * basis[:, 2]
    result = world.copy()
    result[order] = ordered_output
    residual = corrected[active] - prediction[active]
    info.update(status="APPLY" if support.any() else "UNSUPPORTED",
                supported_fraction=float(support.mean()), unchanged_fraction=float(1. - support.mean()),
                common_plane_origin_world_m=ordered_center.tolist(),
                common_tangent_scale_mm=common_scale, grid_shape=grid.tolist(), local_k=local_k,
                local_node_count=len(nodes), pooled_surface_count=len(groups),
                group_node_ids=[g["members"] for g in groups],
                group_coefficients_common_origin=[g["fit"]["beta"].tolist() for g in groups],
                node_cells=[int(n["cell"]) for n in nodes],
                node_local_layers=[int(n["local_layer"]) for n in nodes],
                node_point_counts=[len(n["indices"]) for n in nodes],
                node_to_group=node_to_group.tolist(), merge_history=history,
                compatible_pair_count=int((allowed.sum() - len(nodes)) // 2),
                complete_link_verified=bool(all(allowed[np.ix_(g["members"], g["members"])].all() for g in groups)) if variant == "compatible" else None,
                same_cell_distinct_layers_protected=variant != "global",
                score_name="conditional weighted SSE/sigma^2 + 3G log N; not calibrated model evidence",
                score=float(sum(g["fit"]["sse"] for g in groups) / sigma ** 2 + 3 * len(groups) * np.log(max(len(active), 2))),
                corrected_measurement_residual_rms_mm=float(np.sqrt(np.mean(residual ** 2))),
                normal_displacement_rms_mm=float(np.sqrt(np.mean(np.sum((result - world) ** 2, axis=1)))) * 1000.,
                seconds=time.perf_counter() - started)
    return result, info
