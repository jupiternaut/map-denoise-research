"""V22 point-only, leave-self-out moving-surface reconstruction controls.

These are classical moving-least-squares-family constructions, not a novelty
claim. Unlike a single patch-wide pair of parallel planes, each query has its
own frame and quadratic surface. No ground truth, scanner IDs, mesh spacing as
noise scale, or reference geometry is accepted by this interface.

For a query p, fit neighbouring points (excluding p) in a PCA frame:
    z_j = c0 + c1*u_j + c2*v_j
          + c3*u_j**2 + c4*sqrt(2)*u_j*v_j + c5*v_j**2 + residual_j.
Coordinates are relative to p and tangentially normalized by one isotropic
scale. The sqrt(2) basis makes the quadratic ridge invariant to tangent-basis
rotations. The candidate correction is c0*n. Each scale is fitted independently.

The consensus correction projects all quadratic corrections onto the k=64
normal. Inverse estimated prediction variances weight their mean d. With v_n
the median residual-derived point-noise variance, v_p the weighted prediction
variance (NOT divided by the number of correlated scales), and v_s the weighted
3-D disagreement of scale corrections, the applied correction is
    alpha*d*n64, alpha = v_n / (v_n + v_p + v_s + numerical_floor).
This is a continuous heuristic uncertainty shrinkage, not a calibrated posterior
or safety guarantee. Residuals can contain curvature/association errors; scale
agreement can share bias. Multiple sheets, sharp corners, correlated errors and
poorly sampled normals remain explicit possible failure modes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _psd_pinv(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(a)
    threshold = 1e-12 * np.maximum(values[:, -1:], np.finfo(float).tiny)
    keep = values > threshold
    inverse_values = np.divide(1., values, out=np.zeros_like(values), where=keep)
    inverse = np.einsum("nij,nj,nkj->nik", vectors, inverse_values, vectors)
    return inverse, keep.sum(axis=1)


def _fit(points: np.ndarray, neighbours: np.ndarray, quadratic: bool) -> dict:
    """Batched independent fits; the query is never one of the fitting rows."""
    samples = points[neighbours]
    centered = samples - samples.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / samples.shape[1]
    eigenvalues, basis = np.linalg.eigh(covariance)
    normals = basis[:, :, 0]
    tangents = basis[:, :, 1:]
    offsets = samples - points[:, None, :]
    uv = np.einsum("nki,nij->nkj", offsets, tangents)
    # One isotropic normalization preserves rotations of the tangent basis.
    tangent_scale = np.sqrt(np.mean(np.sum(uv * uv, axis=2), axis=1))
    tangent_scale = np.maximum(tangent_scale, np.sqrt(np.finfo(float).tiny))
    uv = uv / tangent_scale[:, None, None]
    z = np.einsum("nki,ni->nk", offsets, normals)
    u, v = uv[:, :, 0], uv[:, :, 1]
    columns = [np.ones_like(u), u, v]
    if quadratic:
        columns.extend([u*u, np.sqrt(2.)*u*v, v*v])
    design = np.stack(columns, axis=2)
    k = samples.shape[1]
    gram = np.einsum("nki,nkj->nij", design, design) / k
    penalized = gram.copy()
    if quadratic:
        penalized[:, 3, 3] += 1e-6
        penalized[:, 4, 4] += 1e-6
        penalized[:, 5, 5] += 1e-6
    inverse, rank = _psd_pinv(penalized)
    rhs = np.einsum("nki,nk->ni", design, z) / k
    coefficients = np.einsum("nij,nj->ni", inverse, rhs)
    residual = z - np.einsum("nki,ni->nk", design, coefficients)
    median = np.median(residual, axis=1, keepdims=True)
    sigma = 1.4826 * np.median(np.abs(residual - median), axis=1)
    sigma *= np.sqrt(k / np.maximum(k - design.shape[2], 1))
    sigma2 = sigma * sigma
    prediction_factor = np.einsum("ni,nij,nj->n", inverse[:, 0], gram, inverse[:, 0]) / k
    prediction_variance = np.maximum(prediction_factor, 0.) * sigma2
    correction = coefficients[:, 0]
    return {
        "output": points + correction[:, None] * normals,
        "normal": normals,
        "correction": correction,
        "noise_variance": sigma2,
        "prediction_variance": prediction_variance,
        "residual_rms": np.sqrt(np.mean(residual * residual, axis=1)),
        "normal_eigenvalues": eigenvalues,
        "rank": rank,
        "tangent_scale": tangent_scale,
        "coefficients": coefficients,
        "k": k,
    }


def construct(points: np.ndarray, ks: tuple[int, ...] = (32, 64, 128)) -> tuple[dict, dict]:
    """Return same-cardinality point variants and input-only diagnostics.

    Coordinates may have any consistent length unit. Defaults are frozen
    reconstruction scales, not a search over evaluator scores. The direct
    controls use min(64, N-1) neighbours. Inputs require at least nine points.
    The method does not promise topology preservation or denoising of every
    input; it leaves tangential coordinates unchanged in the chosen local frame.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 9:
        raise ValueError("points must have shape (N, 3), N >= 9")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if 64 not in ks or any(int(k) != k or k < 8 for k in ks):
        raise ValueError("ks must contain 64 and consist of integers >= 8")
    actual_ks = sorted(set(min(int(k), len(points)-1) for k in ks))
    control_k = min(64, len(points)-1)
    max_k = max(actual_ks)
    _, indexes = cKDTree(points).query(points, k=max_k+1, workers=1)
    # Do not rely on self being first: duplicate coordinates can produce ties.
    neighbours = np.stack([row[row != i][:max_k] for i, row in enumerate(indexes)])
    plane = _fit(points, neighbours[:, :control_k], quadratic=False)
    quadratic = {k: _fit(points, neighbours[:, :k], quadratic=True) for k in actual_ks}
    anchor = quadratic[control_k]
    normals = anchor["normal"]
    delta_vectors = np.stack([quadratic[k]["correction"][:, None] * quadratic[k]["normal"]
                              for k in actual_ks], axis=1)
    projected = np.einsum("nsi,ni->ns", delta_vectors, normals)
    point_variance = np.median(np.stack([quadratic[k]["noise_variance"] for k in actual_ks], axis=1), axis=1)
    prediction_variances = np.stack([quadratic[k]["prediction_variance"] for k in actual_ks], axis=1)
    # Numerical floor is dimensional and tiny; it is not a sensor-noise guess.
    floor = np.maximum(anchor["tangent_scale"]**2 * 1e-24, np.finfo(float).tiny)
    safe_variances = np.maximum(prediction_variances, floor[:, None])
    # Equivalent inverse-variance ratios, without overflow for identical points.
    precision = safe_variances.min(axis=1, keepdims=True) / safe_variances
    weights = precision / precision.sum(axis=1, keepdims=True)
    mean_delta = np.sum(weights * projected, axis=1)
    mean_vector = mean_delta[:, None] * normals
    disagreement = np.sum(weights * np.sum((delta_vectors - mean_vector[:, None])**2, axis=2), axis=1)
    # Fits share observations; do not claim a 1/number_of_scales variance gain.
    prediction_variance = np.sum(weights * prediction_variances, axis=1)
    alpha = point_variance / (point_variance + prediction_variance + disagreement + floor)
    consensus = points + (alpha * mean_delta)[:, None] * normals
    variants = {
        "local_plane64": plane["output"],
        "quadratic64": anchor["output"],
        "multiscale_full": points + mean_vector,
        "multiscale_consensus": consensus,
    }
    diagnostics = {
        "requested_ks": tuple(int(k) for k in ks),
        "actual_ks": actual_ks,
        "n_points": len(points),
        "leave_self_out": True,
        "neighbour_ids": neighbours,
        "plane64": plane,
        "quadratics": quadratic,
        "consensus": {
            "normal": normals, "weights": weights, "alpha": alpha,
            "noise_variance": point_variance,
            "prediction_variance": prediction_variance,
            "scale_disagreement": disagreement,
            "unshrunk_correction": mean_delta,
            "correction": alpha * mean_delta,
        },
    }
    if not all(np.isfinite(output).all() for output in variants.values()):
        raise FloatingPointError("nonfinite reconstructed output")
    return variants, diagnostics
