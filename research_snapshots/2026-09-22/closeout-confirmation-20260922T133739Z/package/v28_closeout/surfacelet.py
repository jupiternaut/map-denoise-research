"""Direct image evidence for a fixed bank of local plane/support hypotheses.

Coordinates and ray offsets are physical millimetres.  The bank order is PCA
k8, k24, k64, camera-fronto; each normal has full/left/right/top/bottom supports.
Hypothesis 5 therefore reproduces the V27 k24 full-footprint estimator A.
No image visibility, evaluator reference, spatial graph, or acceptance gate is
used.  All hypotheses at offset zero pass through the exact incumbent point.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .direct_evidence import _RAY_EPS, _STD_EPS, _bilinear, _camera, _zncc


NORMAL_NAMES = ("pca_k8", "pca_k24", "pca_k64", "fronto")
SUPPORT_NAMES = ("full", "left", "right", "top", "bottom")
A_HYPOTHESIS = 5


def _offsets_array(offsets: np.ndarray, n_points: int) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim == 1:
        offsets = np.broadcast_to(offsets[None], (n_points, len(offsets)))
    if offsets.ndim != 2 or offsets.shape[0] != n_points or offsets.shape[1] < 1:
        raise ValueError("offsets must have shape [K] or [N,K] with K >= 1")
    if not np.isfinite(offsets).all():
        raise ValueError("offsets must be finite")
    if not np.all((offsets == 0.0).any(axis=1)):
        raise ValueError("each point requires an exact zero-offset incumbent candidate")
    return offsets


def score_surfacelets(
    points: np.ndarray,
    normal_bank: np.ndarray,
    offsets: np.ndarray,
    ref: Mapping[str, np.ndarray],
    sources: Sequence[Mapping[str, np.ndarray]],
    batch_size: int = 32,
) -> dict:
    """Score four normals times five overlapping 7x7 spatial footprints.

    Each normal's 49 samples are warped once per source/depth and reused for
    all five supports.  A support needs >=80% shared finite pixels and both
    population standard deviations >1e-4.  ``scores`` is float32 [S,N,20,K]
    with NaN for unsupported candidates.  Normals are supplied by input-only
    geometry; their sign and length do not affect their plane.

    Reference texture features ``ref_variance`` and ``ref_fraction`` have
    shape [N,5] and use photographs only.  ``ref_valid`` is [N,20] and also
    checks the normal/ray orientation.  Source overlap is candidate-dependent.
    """
    points = np.asarray(points, dtype=np.float64)
    normal_bank = np.asarray(normal_bank, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    n_points = len(points)
    if normal_bank.shape != (n_points, 4, 3):
        raise ValueError("normal_bank must have shape [N,4,3]")
    offsets = _offsets_array(offsets, n_points)
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    reference_camera = _camera(ref)
    source_cameras = [_camera(source) for source in sources]
    relative = points - reference_camera.center
    distance = np.linalg.norm(relative, axis=1)
    direction_valid = np.isfinite(relative).all(axis=1) & (distance > _RAY_EPS)
    directions = np.full_like(relative, np.nan)
    np.divide(relative, distance[:, None], out=directions, where=direction_valid[:, None])
    # Explicit assignment protects the exact incumbent even for an invalid ray.
    candidates = points[:, None] + offsets[..., None] * directions[:, None]
    candidates = np.where((offsets == 0)[..., None], points[:, None], candidates)
    normal_lengths = np.linalg.norm(normal_bank, axis=-1)
    normal_valid = np.isfinite(normal_bank).all(axis=-1) & (normal_lengths > _RAY_EPS)
    normals = np.full_like(normal_bank, np.nan)
    np.divide(normal_bank, normal_lengths[..., None], out=normals,
              where=normal_valid[..., None])
    grid = np.arange(-3, 4, dtype=np.float64)
    dx, dy = np.meshgrid(grid, grid, indexing="xy")
    patch_delta = np.stack([dx.ravel(), dy.ravel()], axis=-1)
    supports = (
        np.ones(49, dtype=bool), dx.ravel() <= 0, dx.ravel() >= 0,
        dy.ravel() <= 0, dy.ravel() >= 0,
    )
    n_candidates = offsets.shape[1]
    scores = np.full((len(sources), n_points, 20, n_candidates), np.nan, dtype=np.float32)
    ref_variance = np.zeros((n_points, 5), dtype=np.float64)
    ref_fraction = np.zeros((n_points, 5), dtype=np.float64)
    ref_valid = np.zeros((n_points, 20), dtype=bool)

    for start in range(0, n_points, batch_size):
        stop = min(start + batch_size, n_points)
        ray_projection = relative[start:stop] @ reference_camera.matrix.T
        anchor_valid = (direction_valid[start:stop]
                        & np.isfinite(ray_projection).all(axis=1)
                        & (ray_projection[:, 2] > _RAY_EPS))
        anchor_xy = np.full((stop-start, 2), np.nan)
        np.divide(ray_projection[:, :2], ray_projection[:, 2, None], out=anchor_xy,
                  where=anchor_valid[:, None])
        patch_xy = anchor_xy[:, None] + patch_delta[None]
        reference_values = _bilinear(reference_camera.image, patch_xy, anchor_valid[:, None])
        texture_valid = np.zeros((stop-start, 5), dtype=bool)
        for support_index, support in enumerate(supports):
            values = reference_values[:, support]
            finite = np.isfinite(values)
            count = finite.sum(axis=-1)
            divisor = np.maximum(count, 1)
            mean = np.where(finite, values, 0).sum(axis=-1) / divisor
            variance = np.where(finite, (values-mean[:, None])**2, 0).sum(axis=-1) / divisor
            ref_variance[start:stop, support_index] = variance
            ref_fraction[start:stop, support_index] = count / int(support.sum())
            texture_valid[:, support_index] = ((count >= int(np.ceil(.8*support.sum())))
                                               & (variance > _STD_EPS**2))
        homogeneous = np.concatenate([patch_xy, np.ones((*patch_xy.shape[:-1], 1))], axis=-1)
        patch_rays = homogeneous @ reference_camera.inverse.T
        lengths = np.linalg.norm(patch_rays, axis=-1, keepdims=True)
        np.divide(patch_rays, lengths, out=patch_rays, where=lengths > _RAY_EPS)
        current_normals = normals[start:stop]
        ray_dot_normal = np.einsum("blj,bhj->bhl", patch_rays, current_normals)
        center_dot_normal = np.einsum("bj,bhj->bh", directions[start:stop], current_normals)
        plane_valid = (anchor_valid[:, None] & normal_valid[start:stop]
                       & np.isfinite(center_dot_normal)
                       & (np.abs(center_dot_normal) > _RAY_EPS))
        support_valid = plane_valid[..., None] & texture_valid[:, None]
        ref_valid[start:stop] = support_valid.reshape(stop-start, 20)
        plane_distance = np.einsum("bkj,bhj->bhk", candidates[start:stop]-reference_camera.center,
                                   current_normals)
        intersection_distance = np.full((stop-start, 4, n_candidates, 49), np.nan)
        np.divide(plane_distance[..., None], ray_dot_normal[:, :, None],
                  out=intersection_distance,
                  where=np.abs(ray_dot_normal[:, :, None]) > _RAY_EPS)
        intersection_valid = (plane_valid[:, :, None, None]
                              & np.isfinite(intersection_distance)
                              & (intersection_distance > _RAY_EPS))
        patch_world = (reference_camera.center + intersection_distance[..., None]
                       * patch_rays[:, None, None])
        for source_index, source_camera in enumerate(source_cameras):
            projected = (patch_world-source_camera.center) @ source_camera.matrix.T
            source_valid = (intersection_valid & np.isfinite(projected).all(axis=-1)
                            & (projected[..., 2] > _RAY_EPS))
            xy = np.full((*projected.shape[:-1], 2), np.nan)
            np.divide(projected[..., :2], projected[..., 2, None], out=xy,
                      where=source_valid[..., None])
            source_values = _bilinear(source_camera.image, xy, source_valid)
            for support_index, support in enumerate(supports):
                score = _zncc(reference_values[:, None, None, support],
                              source_values[..., support], int(np.ceil(.8*support.sum())))
                score = np.where(support_valid[:, :, support_index, None], score, np.nan)
                scores[source_index, start:stop, support_index::5] = score.astype(np.float32)
    return {
        "scores": scores, "directions": directions, "candidates": candidates,
        "offsets": offsets, "ref_variance": ref_variance, "ref_fraction": ref_fraction,
        "ref_valid": ref_valid, "normal_names": NORMAL_NAMES, "support_names": SUPPORT_NAMES,
        "hypothesis_normal": np.repeat(np.arange(4), 5),
        "hypothesis_support": np.tile(np.arange(5), 4),
    }


def _aggregate(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(scores)
    count = finite.sum(axis=0)
    selected = np.sort(np.where(finite, scores, -np.inf), axis=0)[-top_k:]
    total = np.where(np.isfinite(selected), selected, 0).sum(axis=0)
    valid = count >= 2
    cost = 1 - total / np.maximum(np.minimum(count, top_k), 1)
    return np.where(valid, cost, np.inf), valid, count


def _choose(cost: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lexicographic minimum: cost, |offset|, hypothesis, depth index."""
    n_points, n_hypotheses, n_candidates = cost.shape
    minimum = cost.min(axis=(1, 2))
    valid = np.isfinite(minimum)
    # isclose(inf, inf) is false, so all-invalid rows get their own zero path.
    tie = np.isclose(cost, minimum[:, None, None], rtol=0, atol=1e-12)
    tie &= valid[:, None, None]
    smallest_offset = np.where(tie, np.abs(offsets[:, None]), np.inf).min(axis=(1, 2))
    tie &= np.abs(offsets[:, None]) == smallest_offset[:, None, None]
    flat = tie.reshape(n_points, n_hypotheses*n_candidates).argmax(axis=1)
    hypothesis, choice = flat // n_candidates, flat % n_candidates
    choice = np.where(valid, choice, (offsets == 0).argmax(axis=1))
    return hypothesis, choice, valid


def _depth_features(cost: np.ndarray, offsets: np.ndarray, choice: np.ndarray,
                    valid: np.ndarray) -> dict[str, np.ndarray]:
    """Depth modes use the hypothesis-minimized cost envelope, sorted by mm."""
    n_points = len(offsets)
    row = np.arange(n_points)
    envelope = cost.min(axis=1)
    order = np.argsort(offsets, axis=1, kind="stable")
    x = np.take_along_axis(offsets, order, axis=1)
    y = np.take_along_axis(envelope, order, axis=1)
    left = np.concatenate([np.full((n_points, 1), np.inf), y[:, :-1]], axis=1)
    right = np.concatenate([y[:, 1:], np.full((n_points, 1), np.inf)], axis=1)
    local_minimum = (y <= left) & (y <= right) & np.isfinite(y)
    selected_offset = offsets[row, choice]
    other_mode = local_minimum & (np.abs(x-selected_offset[:, None]) > .5)
    competitor = np.where(other_mode, y, np.inf).min(axis=1)
    selected_cost = envelope[row, choice]
    gap_valid = valid & np.isfinite(competitor)
    mode_gap = np.zeros(n_points)
    np.subtract(competitor, selected_cost, out=mode_gap, where=gap_valid)
    sorted_choice = (order == choice[:, None]).argmax(axis=1)
    center = sorted_choice
    left_index = np.maximum(center-1, 0)
    right_index = np.minimum(center+1, x.shape[1]-1)
    xl, xc, xr = x[row, left_index], x[row, center], x[row, right_index]
    yl, yc, yr = y[row, left_index], y[row, center], y[row, right_index]
    curvature_valid = (valid & (center > 0) & (center < x.shape[1]-1)
                       & np.isfinite(yl) & np.isfinite(yc) & np.isfinite(yr)
                       & (xc > xl) & (xr > xc))
    curvature = np.zeros(n_points)
    active = np.flatnonzero(curvature_valid)
    curvature[active] = 2*((yr[active]-yc[active])/(xr[active]-xc[active])
                           -(yc[active]-yl[active])/(xc[active]-xl[active]))/(xr[active]-xl[active])
    return {"mode_gap": mode_gap, "mode_gap_valid": gap_valid,
            "depth_curvature": curvature, "depth_curvature_valid": curvature_valid}


def solve_surfacelets(scores: np.ndarray, offsets: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Return ungated A/B choices for all-source and first-two-source fitting.

    All-source arms average their three best scores, requiring >=2 finite.
    Fit arms require both source 0/1 and average them.  Only after those fit
    choices are locked are sources 2/3 inspected for reserved-view features.
    Missing candidates cannot win if a valid candidate exists; all-invalid
    points choose exact zero and the lowest allowed hypothesis.

    Features are finite with explicit validity flags; missing costs use the
    neutral sentinel 1, missing other numeric features use 0.  Costs are 1-ZNCC.
    ``cost_at_zero`` and reserved incumbent costs use the selected hypothesis
    at exact zero, allowing a like-for-like support/orientation comparison.
    ``mode_gap`` is the next local minimum of the depth cost envelope outside
    +/-0.5mm minus the chosen cost (boundary minima included).  Curvature is
    its discrete second derivative in mm^-2 at the chosen depth.  Source
    agreement is the fraction of fitting sources whose independently chosen
    depth is within 0.5mm of the proposal; hypothesis disagreements are allowed.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 4 or scores.shape[0] != 4 or scores.shape[2] != 20:
        raise ValueError("scores must have shape [4,N,20,K]")
    if np.isinf(scores).any() or np.any(np.abs(scores[np.isfinite(scores)]) > 1+1e-6):
        raise ValueError("finite scores must be correlations in [-1,1]; missing scores must be NaN")
    n_points = scores.shape[1]
    offsets = _offsets_array(offsets, n_points)
    if offsets.shape[1] != scores.shape[3]:
        raise ValueError("score candidate dimension must match offsets")
    if n_points == 0:
        raise ValueError("solve_surfacelets requires at least one point")
    row = np.arange(n_points)
    zero = (offsets == 0).argmax(axis=1)
    result = {}
    # Fit proposal computation is structurally isolated from reserved scores.
    for suffix, source_ids, top_k in (("fit", (0, 1), 2), ("all", (0, 1, 2, 3), 3)):
        fit_scores = scores[list(source_ids)]
        for arm, hypotheses in (("A", np.array([A_HYPOTHESIS])), ("B", np.arange(20))):
            candidate_scores = fit_scores[:, :, hypotheses]
            cost, candidate_valid, count = _aggregate(candidate_scores, top_k)
            local_h, choice, valid = _choose(cost, offsets)
            hypothesis = hypotheses[local_h]
            training_cost = cost[row, local_h, choice]
            zero_cost = cost[row, local_h, zero]
            zero_valid = candidate_valid[row, local_h, zero]
            source_best_offset = np.zeros((n_points, len(source_ids)))
            source_best_valid = np.zeros((n_points, len(source_ids)), dtype=bool)
            for index in range(len(source_ids)):
                individual_cost = np.where(np.isfinite(candidate_scores[index]),
                                           1-candidate_scores[index], np.inf)
                _, individual_choice, individual_valid = _choose(individual_cost, offsets)
                source_best_offset[:, index] = offsets[row, individual_choice]
                source_best_valid[:, index] = individual_valid
            delta = offsets[row, choice]
            agreeing = source_best_valid & (np.abs(source_best_offset-delta[:, None]) <= .5)
            agreement_count = source_best_valid.sum(axis=1)
            agreement_valid = valid & (agreement_count > 0)
            agreement = np.where(agreement_valid, agreeing.sum(axis=1)/np.maximum(agreement_count, 1), 0)
            arm_result = {
                "choice": choice, "hypothesis": hypothesis, "offset": delta,
                "valid": valid, "training_cost": np.where(valid, training_cost, 1),
                "cost_at_zero": np.where(zero_valid, zero_cost, 1),
                "cost_at_zero_valid": zero_valid,
                "source_count": count[row, local_h, choice],
                "source_agreement": agreement, "source_agreement_valid": agreement_valid,
                "source_best_offset": source_best_offset, "source_best_valid": source_best_valid,
                **_depth_features(cost, offsets, choice, valid),
            }
            # Postproposal: reading these arrays cannot influence any choices.
            proposed_score = scores[:, row, hypothesis, choice].T
            incumbent_score = scores[:, row, hypothesis, zero].T
            source_valid = np.isfinite(proposed_score)
            incumbent_valid = np.isfinite(incumbent_score)
            arm_result.update({
                "source_cost": np.where(source_valid, 1-proposed_score, 1),
                "source_valid": source_valid,
                "heldout_incumbent_cost": np.where(incumbent_valid[:, 2:], 1-incumbent_score[:, 2:], 1),
                "heldout_incumbent_valid": incumbent_valid[:, 2:],
                "heldout_proposal_cost": np.where(source_valid[:, 2:], 1-proposed_score[:, 2:], 1),
                "heldout_proposal_valid": source_valid[:, 2:],
            })
            result[f"{arm}_{suffix}"] = arm_result
    return result
