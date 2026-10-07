"""CPU/NumPy kernels for conditional optical-Z intervals and finite candidates.

No data loading, fitted constants, random state, GT, or calibration claims.
All distances must use one caller-chosen unit; angles are radians.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite three-vector")
    return value


def _budget(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def optical_ray(K, xy):
    """Return K^-1[u,v,1] with camera z component ONE, not unit length."""
    K = np.asarray(K, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if K.shape != (3, 3) or xy.shape != (2,) or not np.isfinite(K).all() or not np.isfinite(xy).all():
        raise ValueError("K and xy must be finite with shapes (3,3) and (2,)")
    ray = np.linalg.solve(K, [*xy, 1.0])
    if not np.isfinite(ray).all() or ray[2] == 0:
        raise ValueError("invalid homogeneous ray")
    return ray / ray[2]


def pixel_ray_error(K, pixel_error):
    """L2 ray bound from L2 pixel error for conventional affine pinhole K.

    Intrinsic/pose uncertainty needs its own budget. A +/-p box per pixel
    coordinate has L2 radius sqrt(2)*p, not p.
    """
    K = np.asarray(K, dtype=float)
    p = _budget(pixel_error, "pixel_error")
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.array_equal(K[2], [0., 0., 1.]):
        raise ValueError("requires K bottom row [0,0,1]")
    return float(np.linalg.norm(np.linalg.inv(K)[:, :2], ord=2) * p)


def anchor_error_from_depth(depth, neighbor_ray, depth_error, neighbor_ray_error):
    """Bound ||z_j r_j - zhat_j rhat_j|| including the product term."""
    r = _vector(neighbor_ray, "neighbor_ray")
    depth = float(depth)
    if not math.isfinite(depth):
        raise ValueError("depth must be finite")
    ez = _budget(depth_error, "depth_error")
    er = _budget(neighbor_ray_error, "neighbor_ray_error")
    return float(ez * np.linalg.norm(r) + (abs(depth) + ez) * er)


def plane_depth_interval(
    anchor, normal, ray, *, normal_angle_rad, anchor_error=0.0,
    camera_center=(0.0, 0.0, 0.0), camera_error=0.0, ray_error=0.0,
    residual_error=0.0, relative_anchor_error=None,
    joint_residual_error=None, depth_domain=(0.0, math.inf),
    denominator_floor=1e-12,
):
    """Enclose true optical-Z under the explicit bounded-error model.

    X=C+z*r; n is unit and n dot (X-Xj) has absolute value <= residual_error.
    normal_angle_rad bounds the acute oriented angle after sign alignment.
    relative_anchor_error directly bounds ||(Xj-C)-(anchor-camera_center)||;
    default is anchor_error+camera_error (safe, potentially loose).
    joint_residual_error optionally bounds ||delta(Xj-C)-zhat*delta(r)||;
    default is relative_anchor_error+abs(zhat)*ray_error.
    Supplied joint bounds must be justified jointly, not fitted from GT.
    """
    X = _vector(anchor, "anchor")
    C = _vector(camera_center, "camera_center")
    n = _vector(normal, "normal")
    r = _vector(ray, "ray")
    norm = float(np.linalg.norm(n))
    if norm <= 1e-15 or np.linalg.norm(r) <= 1e-15:
        raise ValueError("normal and optical ray must be nonzero")
    n = n / norm
    theta = _budget(normal_angle_rad, "normal_angle_rad")
    if theta >= math.pi / 2:
        raise ValueError("normal_angle_rad must be below pi/2")
    eX = _budget(anchor_error, "anchor_error")
    eC = _budget(camera_error, "camera_error")
    er = _budget(ray_error, "ray_error")
    kappa = _budget(residual_error, "residual_error")
    eA = eX + eC if relative_anchor_error is None else _budget(relative_anchor_error, "relative_anchor_error")
    floor = _budget(denominator_floor, "denominator_floor")
    domain_lo, domain_hi = map(float, depth_domain)
    if not math.isfinite(domain_lo) or math.isnan(domain_hi) or domain_lo < 0 or domain_hi <= domain_lo:
        raise ValueError("depth_domain must satisfy 0 <= finite low < high")
    A = X - C
    d = float(n @ r)
    tangent_ray = r - d * n
    gamma = abs(d) * math.cos(theta) - np.linalg.norm(tangent_ray) * math.sin(theta) - er
    out = {
        "status": "uninformative", "reason": "uncertain_or_small_denominator",
        "center": None, "radius": None, "low": domain_lo, "high": domain_hi,
        "denominator_nominal": d, "denominator_margin": float(gamma),
        "relative_anchor_error": eA, "normal_angle_rad": theta,
        "chord_radius": None,
        "interpretation": "conditional_on_supplied_error_budgets_not_calibrated",
    }
    if gamma <= floor or abs(d) <= floor:
        return out
    zhat = float(n @ A / d)
    s = A - zhat * r
    joint = eA + abs(zhat) * er if joint_residual_error is None else _budget(joint_residual_error, "joint_residual_error")
    # The extra parallel residual is zero in real arithmetic; keep its computed
    # magnitude so an imperfect floating-point cancellation cannot shrink R.
    ns = float(n @ s)
    tangent_s = s - ns * n
    numerator = math.sin(theta) * np.linalg.norm(tangent_s) + abs(ns) + joint + kappa
    radius = float(numerator / gamma)
    if not math.isfinite(zhat) or not math.isfinite(radius):
        out["reason"] = "nonfinite_numerical_result"
        return out
    low = max(domain_lo, float(np.nextafter(zhat - radius, -math.inf)))
    high = min(domain_hi, float(np.nextafter(zhat + radius, math.inf)))
    eta = 2 * math.sin(theta / 2)
    chord_gamma = abs(d) - eta * np.linalg.norm(r) - er
    chord_radius = None
    if chord_gamma > floor:
        chord_radius = float((eta * np.linalg.norm(s) + joint + kappa + abs(ns)) / chord_gamma)
    out.update(status="ok" if low <= high else "empty", reason=None,
               center=zhat, radius=radius, low=low, high=high,
               local_baseline=float(np.linalg.norm(s)),
               numerator_bound=float(numerator), joint_residual_error=joint,
               chord_radius=chord_radius)
    return out


def _candidates(candidate_depths):
    z = np.asarray(candidate_depths, dtype=float)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all() or (z <= 0).any():
        raise ValueError("candidate_depths must be nonempty finite positive 1-D values")
    return z


def support_scores(candidate_depths, intervals: Iterable[dict], *, tolerance=0.0):
    """Candidate membership in finite intervals expanded by tolerance.

    Uninformative/empty intervals are excluded. Any contamination budget passed
    to select_candidate therefore applies to THIS retained set.
    """
    z = _candidates(candidate_depths)
    rho = _budget(tolerance, "tolerance")
    retained = [(j, i) for j, i in enumerate(intervals)
                if i.get("status") == "ok" and math.isfinite(i["low"])
                and math.isfinite(i["high"]) and i["low"] <= i["high"]]
    lo = np.array([i["low"] for _, i in retained], dtype=float)
    hi = np.array([i["high"] for _, i in retained], dtype=float)
    support = (z[None, :] >= lo[:, None] - rho) & (z[None, :] <= hi[:, None] + rho)
    separation = np.abs(z[:, None] - z[None, :])
    # An interval of this expanded diameter cannot support both candidates.
    separating = (separation[None, :, :] > (hi - lo + 2 * rho)[:, None, None]).sum(axis=0)
    return {"active_count": len(retained), "active_indices": [j for j, _ in retained],
            "counts": support.sum(axis=0).astype(int).tolist(),
            "support": support.tolist(), "pairwise_separating_counts": separating.astype(int).tolist()}


def select_candidate(candidate_depths, intervals, *, max_bad, tolerance=0.0, min_active=1):
    """Unique-feasible rule; conclusion is conditional, never calibration.

    Assumptions: some candidate is within tolerance of true z; at most max_bad
    RETAINED intervals fail to contain true z. A unique candidate with support
    >= active_count-max_bad is then within tolerance of true z. Otherwise abstain.
    """
    if not isinstance(max_bad, (int, np.integer)) or max_bad < 0:
        raise ValueError("max_bad must be a nonnegative integer")
    if not isinstance(min_active, (int, np.integer)) or min_active < 1:
        raise ValueError("min_active must be a positive integer")
    result = support_scores(candidate_depths, intervals, tolerance=tolerance)
    m = result["active_count"]
    threshold = m - int(max_bad)
    feasible = [k for k, count in enumerate(result["counts"]) if count >= threshold]
    enough = m >= min_active and m > max_bad
    selected = feasible[0] if enough and len(feasible) == 1 else None
    status = "unique_feasible" if selected is not None else (
        "insufficient_active" if not enough else "no_feasible" if not feasible else "ambiguous")
    result.update(status=status, selected_index=selected, feasible_indices=feasible,
                  required_support=threshold, max_bad=int(max_bad),
                  interpretation="conditional_identification_requires_truth_coverage_and_pool_coverage")
    return result


def certain_possible_support(candidate_depths, intervals, *, tolerance,
                             max_radius=math.inf, denominator=None):
    """Observable interval realization bounds for a fixed-band vote.

    An active record has finite status=ok interval and radius <= max_radius.
    Other records abstain. Denominator defaults to ALL supplied records,
    including abstentions. Caller may set a larger original valid-record count.
    L_k <= V_k <= U_k for every active latent depth xi_j within its interval.
    If b active intervals fail coverage, relax each bound by b/denominator.
    """
    z = _candidates(candidate_depths)
    tau = _budget(tolerance, "tolerance")
    max_radius = float(max_radius)
    if math.isnan(max_radius) or max_radius < 0:
        raise ValueError("max_radius must be nonnegative")
    records = list(intervals)
    N = len(records) if denominator is None else denominator
    if not isinstance(N, (int, np.integer)) or N < len(records) or N < 1:
        raise ValueError("denominator must be an integer >= supplied record count and >=1")
    lower = np.zeros(len(z), dtype=int)
    upper = np.zeros(len(z), dtype=int)
    active = []
    for j, interval in enumerate(records):
        if interval.get("status") != "ok":
            continue
        lo, hi = float(interval["low"]), float(interval["high"])
        radius = interval.get("radius")
        radius = (hi - lo) / 2 if radius is None else float(radius)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi
                and math.isfinite(radius) and 0 <= radius <= max_radius):
            continue
        active.append(j)
        lower += (lo >= z - tau) & (hi <= z + tau)
        upper += (hi >= z - tau) & (lo <= z + tau)
    return {"denominator": int(N), "active_count": len(active), "active_indices": active,
            "certain_counts": lower.tolist(), "possible_counts": upper.tolist(),
            "certain": (lower / N).tolist(), "possible": (upper / N).tolist(),
            "interpretation": "robust_ranking_of_active_interval_votes_not_truth_certificate"}
