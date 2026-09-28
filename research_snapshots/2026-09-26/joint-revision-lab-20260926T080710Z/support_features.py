"""Fixed-coordinate nuisance-support selection, with no geometry or GT access."""
import numpy as np


def choose_support(fit_new_scores):
    """Choose h from F-only [4,N,20] B scores; -1 means no valid choice.

    Match historical surfacelet aggregation: average the best three finite
    source correlations, requiring at least two; ties within 1e-12 choose
    the lowest hypothesis. No incumbent scores enter this choice.
    """
    scores = np.asarray(fit_new_scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[0] != 4 or scores.shape[2] != 20:
        raise ValueError('fit_new_scores must have shape [4,N,20]')
    if np.isinf(scores).any() or np.any(np.abs(scores) > 1 + 1e-6):
        raise ValueError('scores must be ZNCC or NaN')
    finite = np.isfinite(scores)
    count = finite.sum(axis=0)
    top = np.sort(np.where(finite, scores, -np.inf), axis=0)[-3:]
    total = np.where(np.isfinite(top), top, 0).sum(axis=0)
    cost = np.where(count >= 2, 1 - total / np.maximum(np.minimum(count, 3), 1), np.inf)
    minimum = cost.min(axis=1)
    valid = np.isfinite(minimum)
    tie = np.isclose(cost, minimum[:, None], atol=1e-12, rtol=0) & valid[:, None]
    return np.where(valid, tie.argmax(axis=1), -1).astype(np.int8)


def gather_reserved(reserved_scores, chosen_h):
    """Return [4,N,2], retaining the SAME chosen h for incumbent and B.

    Missing F support produces all-NaN evidence even if R has valid scores.
    The caller summarizes this with the historical neutral+validity encoding.
    """
    scores = np.asarray(reserved_scores)
    h = np.asarray(chosen_h)
    if scores.ndim != 4 or scores.shape[0] != 4 or scores.shape[2:] != (20, 2):
        raise ValueError('reserved_scores must have shape [4,N,20,2]')
    if h.shape != (scores.shape[1],) or h.dtype.kind != 'i' or np.any((h < -1) | (h > 19)):
        raise ValueError('chosen_h must be signed integer [N] in [-1,19]')
    pair = scores[:, np.arange(len(h)), np.maximum(h, 0), :].copy()
    pair[:, h < 0, :] = np.nan
    return pair
