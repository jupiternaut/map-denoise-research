"""Pure, preregistered decision rules; no experiment data or truth imports.

All new rules use the declared numerical tie tolerance. Physical cell widths
are in the units of ``grid`` (mm in this run), including on nonuniform grids.
The P rule preserves historical candidate order and exact first-minimum ties.
Supplying historical_intervals to P means the admission step is already sealed.
"""
from __future__ import annotations

import math
import numpy as np


ABS_TOL = 1e-10
REL_TOL = 1e-10
LEGACY_FLAT_TOL = 1e-12


def _curve(grid, loss):
    grid = np.asarray(grid, dtype=float)
    loss = np.asarray(loss, dtype=float)
    if grid.ndim != 1 or not len(grid) or loss.shape != grid.shape:
        raise ValueError("grid and loss must be nonempty, equal-length 1D arrays")
    if not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be finite, strictly increasing and unique")
    return grid, loss


def _positive(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError, OverflowError):
        return False


def _near(left, right):
    return np.abs(left - right) <= ABS_TOL + REL_TOL * np.maximum(np.abs(left), np.abs(right))


def _mask(accepted, length):
    accepted = np.asarray(accepted)
    if accepted.shape != (length,) or accepted.dtype.kind != "b":
        raise ValueError("accepted must be a Boolean vector matching the score grid")
    return accepted


def _cell_edges(grid):
    if len(grid) < 2:
        raise ValueError("physical cell integration needs at least two grid nodes")
    return np.concatenate(([grid[0]], grid[:-1] + np.diff(grid) / 2, [grid[-1]]))


def acceptance(grid, loss, sigma2):
    """Return the original 9/4 residual acceptance mask, before flat rejection.

    Invalid scales or nonfinite scores give an empty mask. This helper does not
    certify scale calibration and does not replace decide's flat/validity gates.
    """
    grid, loss = _curve(grid, loss)
    if not _positive(sigma2) or not np.isfinite(loss).all():
        return np.zeros(len(grid), dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = loss / float(sigma2)
        return (normalized <= 9.0) & ((normalized - normalized.min()) <= 4.0)


def distribution(grid, loss, sigma2, temperature, accepted, uniform=False):
    """Normalized masses on the same accepted depth cells for R and U.

    q is a Gibbs score weighting, not a calibrated posterior. Domain endpoints
    bound the first/last cells. Passing reciprocal-depth-spaced *depth values*
    is supported; passing inverse-depth values as though they were mm is not.
    """
    grid, loss = _curve(grid, loss)
    accepted = _mask(accepted, len(grid))
    if not np.isfinite(loss[accepted]).all():
        raise ValueError("distribution requires finite scores on accepted nodes")
    if not _positive(sigma2) or not _positive(temperature):
        raise ValueError("distribution requires positive finite sigma2 and temperature")
    if not accepted.any():
        raise ValueError("distribution is undefined for empty accepted support")
    widths = np.diff(_cell_edges(grid))
    q = np.zeros(len(grid), dtype=float)
    if uniform:
        weights = widths[accepted]
    else:
        # Subtracting the accepted minimum is algebraically identical after
        # normalization and avoids all-zero underflow under an external mask.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            shifted = (loss[accepted] - loss[accepted].min()) / float(sigma2)
            exponent = -shifted / float(temperature)
        weights = widths[accepted] * np.exp(exponent)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("accepted distribution has no finite positive mass")
    q[accepted] = weights / total
    return q


def crps(grid, q, truth):
    """Exact CRPS of the discrete node-mass distribution, in depth units."""
    grid, q = _curve(grid, q)
    if not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("q must contain finite nonnegative masses")
    if not math.isclose(float(q.sum()), 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("q must sum to one")
    if not math.isfinite(float(truth)):
        raise ValueError("truth must be finite")
    # E|Z-y| - 0.5 E|Z-Z'|, using sorted nodes in O(n) memory/time.
    pair_half = np.dot(q * grid, 2 * np.cumsum(q) - q - 1)
    value = float(np.dot(q, np.abs(grid - float(truth))) - pair_half)
    if value < -1e-8:
        raise ValueError("numerically invalid negative CRPS")
    return max(0.0, value)


def combine_scale(sigma_values, pixel_counts, sigma_cal2=None):
    """Combine per-fold variances, retaining every valid local sigma >= 1.

    Only unavailable local folds use sigma_cal2. Sources are 'local',
    'calibration', or 'unavailable'. Any unavailable fold makes the result NaN;
    it is never removed from the denominator. Invalid counts are a contract
    error, not permission to silently drop a scoring fold.
    """
    sigma_values = np.asarray(sigma_values, dtype=float)
    pixel_counts = np.asarray(pixel_counts, dtype=float)
    if (sigma_values.ndim != 1 or not len(sigma_values)
            or pixel_counts.shape != sigma_values.shape):
        raise ValueError("sigma_values and pixel_counts must be matching nonempty vectors")
    if not np.isfinite(pixel_counts).all() or np.any(pixel_counts <= 0):
        raise ValueError("every scoring fold must have a positive finite pixel count")
    fallback_valid = _positive(sigma_cal2) and float(sigma_cal2) >= 1.0
    variances, sources = [], []
    for sigma in sigma_values:
        if math.isfinite(float(sigma)) and sigma >= 1.0:
            variance = float(sigma) ** 2
            if not math.isfinite(variance):
                raise ValueError("local variance overflow")
            variances.append(variance)
            sources.append("local")
        elif fallback_valid:
            variances.append(float(sigma_cal2))
            sources.append("calibration")
        else:
            variances.append(float("nan"))
            sources.append("unavailable")
    if "unavailable" in sources:
        return float("nan"), sources
    return float(np.average(variances, weights=pixel_counts)), sources


def _intervals(grid, accepted, padding=1.0):
    """Connected accepted physical cells, padded/clipped/merged as legacy P.

    On a uniform grid this is exactly the old half-step-plus-padding rule.
    Nonuniform grids use their physical Voronoi cell boundaries.
    """
    if not accepted.any():
        return []
    if len(grid) == 1:
        return [[float(grid[0]), float(grid[0])]]
    edges = _cell_edges(grid)
    starts = np.flatnonzero(accepted & np.r_[True, ~accepted[:-1]])
    stops = np.flatnonzero(accepted & np.r_[~accepted[1:], True]) + 1
    intervals = []
    for start, stop in zip(starts, stops):
        low = float(max(grid[0], edges[start] - padding))
        high = float(min(grid[-1], edges[stop] + padding))
        if intervals and low <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], high)
        else:
            intervals.append([low, high])
    return intervals


def _legacy_selection(intervals, candidates, incumbent):
    normalized = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("each historical interval must have two endpoints")
        low, high = map(float, interval)
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise ValueError("historical interval endpoints must be finite and ordered")
        normalized.append([low, high])
    mass = sum(high - low for low, high in normalized)
    if mass <= 0:
        return incumbent, dict(intervals=normalized, support_mean=None, estimated_squared_gain=0.0)
    mean = sum((high - low) * (low + high) / 2 for low, high in normalized) / mass
    # Intentionally preserve first-index ties and input candidate ordering.
    new = float(candidates[int(np.argmin((candidates - mean) ** 2))])
    gain = (incumbent - mean) ** 2 - (new - mean) ** 2
    return (new if gain > 0 else incumbent), dict(
        intervals=normalized, support_mean=float(mean), estimated_squared_gain=max(0.0, float(gain)))


def decide(grid, loss, candidates, incumbent, *, rule, sigma2, temperature=1.0,
           raw_valid=True, accepted=None, historical_intervals=None, kappa=1.0):
    """Apply one frozen decision rule, returning a JSON-safe decision record.

    P with historical_intervals replays the already-admitted historical support
    without re-gating it. Other P calls preserve the legacy absolute flat test;
    external full9 masks preserve its old P behavior (no added flat rejection).
    M/R/U/Mraw use new flat/tie rules and deduplicated sorted actions including
    KEEP. M/Mraw require exact sampled scores for every action, not a nearest
    grid approximation. External masks replace residual acceptance; P/M then
    need no grayscale scale, whereas R/U still require a positive score scale.
    External full9 masks also declare its finite score domain: new decisions
    ignore nonfinite candidate scores, and R/U leave nonfinite nodes at zero
    mass while keeping cell boundaries on the complete original grid.
    """
    if rule not in {"P", "M", "Mraw", "R", "U"}:
        raise ValueError("unknown decision rule: " + str(rule))
    if historical_intervals is not None and rule != "P":
        raise ValueError("historical_intervals is only valid for P")
    grid, loss = _curve(grid, loss)
    candidates = np.asarray(candidates, dtype=float)
    incumbent = float(incumbent)
    if candidates.ndim != 1 or not len(candidates) or not np.isfinite(candidates).all():
        raise ValueError("candidates must be a nonempty finite vector")
    if not math.isfinite(incumbent):
        raise ValueError("incumbent must be finite")
    if not _positive(kappa):
        raise ValueError("kappa must be positive and finite")
    external = accepted is not None
    finite_nodes = np.isfinite(loss)
    mask = (_mask(accepted, len(grid)) & finite_nodes) if external else acceptance(grid, loss, sigma2)
    finite = bool(finite_nodes.any() if external else finite_nodes.all())
    finite_loss = loss[finite_nodes]
    flat = bool(finite and (float(np.ptp(finite_loss)) <= LEGACY_FLAT_TOL if rule == "P"
                          else _near(float(finite_loss.min()), float(finite_loss.max()))))
    scale_valid = bool(_positive(sigma2))
    actions = candidates if rule == "P" else np.unique(np.r_[candidates, incumbent])
    result = dict(selected_depth=incumbent, reason="keep_incumbent", move=False,
                  scale_valid=scale_valid, accepted_count=int(mask.sum()),
                  raw_valid=bool(raw_valid), flat=flat, rule=rule,
                  candidate_count=int(len(actions)), finite_score_count=int(finite_nodes.sum()))

    def keep(reason):
        result["reason"] = reason
        return result

    def finish(selected, diagnostics):
        result.update(diagnostics)
        result["selected_depth"] = float(selected)
        result["move"] = bool(float(selected) != incumbent)
        result["reason"] = "ok_move" if result["move"] else "keep_incumbent"
        return result

    if rule == "P" and historical_intervals is not None:
        selected, diagnostics = _legacy_selection(historical_intervals, candidates, incumbent)
        finish(selected, diagnostics)
        if diagnostics["support_mean"] is None:
            result["reason"] = "threshold_empty"
        return result
    if not raw_valid:
        return keep("raw_invalid")
    if not finite:
        return keep("nonfinite_curve")
    if rule != "Mraw" and not scale_valid and (not external or rule in {"R", "U"}):
        return keep("scale_unavailable")
    if flat and not (rule == "P" and external):
        return keep("flat_curve")
    if rule != "Mraw" and not mask.any():
        return keep("threshold_empty")
    if rule == "P":
        selected, diagnostics = _legacy_selection(_intervals(grid, mask), candidates, incumbent)
        return finish(selected, diagnostics)
    if rule in {"M", "Mraw"}:
        indices = np.searchsorted(grid, actions)
        if np.any(indices == len(grid)) or not np.array_equal(grid[indices], actions):
            raise ValueError("every M/Mraw action, including incumbent, needs its exact score-grid node")
        values = loss[indices]
        valid_candidates = np.isfinite(values)
        result["finite_candidate_count"] = int(valid_candidates.sum())
        if not valid_candidates.any():
            return keep("no_finite_candidate")
        available_actions = actions[valid_candidates]
        values = values[valid_candidates]
        best = float(values.min())
        tied = np.flatnonzero(_near(values, best))
        result.update(best_loss=best, best_count=int(len(tied)))
        if len(tied) != 1:
            return keep("tied_best")
        return finish(float(available_actions[tied[0]]), {})
    q = distribution(grid, loss, sigma2, temperature, mask, uniform=(rule == "U"))
    delta = np.abs(actions[:, None] - grid[None, :]) - np.abs(incumbent - grid)[None, :]
    asymmetric = float(kappa) * np.maximum(delta, 0.0) + np.minimum(delta, 0.0)
    risks = asymmetric @ q
    risks[actions == incumbent] = 0.0
    best = float(risks.min())
    tied = np.flatnonzero(_near(risks, best))
    result.update(best_count=int(len(tied)), selected_risk=0.0,
                  risks=[dict(depth=float(a), risk=float(r)) for a, r in zip(actions, risks)],
                  kappa=float(kappa), temperature=float(temperature))
    if len(tied) != 1:
        return keep("tied_best")
    if best >= 0:
        return keep("no_positive_gain")
    return finish(float(actions[tied[0]]), dict(selected_risk=best))
