"""Pure, GT-free summaries of four paired incumbent/proposal ZNCC scores.

Rows retain their original order. NaN is the only missing-data marker; a
missing patch is never treated as a positive observation. The four source
slots must already have been selected by the caller's frozen protocol.
"""

from __future__ import annotations

import numpy as np


SCORE_TOLERANCE = 1e-6
FEATURE_NAMES = tuple(
    f"{kind}_{view}"
    for kind in (
        "old_cost", "new_cost", "paired_margin", "pair_valid",
        "old_valid", "new_valid",
    )
    for view in range(4)
) + (
    "paired_mean_margin", "paired_min_margin", "paired_max_margin",
    "paired_std_margin", "paired_win_fraction", "paired_valid_fraction",
    "paired_mean_old_cost", "paired_mean_new_cost",
)


def _numeric_array(value: object, name: str) -> np.ndarray:
    """Copy real numeric input so callers' buffers are never edited or aliased."""
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a real numeric array")
    return np.array(array, dtype=np.float64, copy=True)


def summarize_pairs(scores: np.ndarray) -> np.ndarray:
    """Return float32 [N, 32] features from ZNCC [4, N, 2].

    Last-axis entries are incumbent and proposal, respectively. Larger ZNCC
    means better agreement, so positive paired margins favor the proposal.
    Tiny floating-point excursions (up to SCORE_TOLERANCE) are clipped to the
    legal ZNCC range. Larger excursions and infinity are rejected.

    Costs are 1 - ZNCC. Individual costs retain every valid observation,
    but all eight aggregate features use only views valid for BOTH points.
    Empty paired support gives zero margin statistics/fractions and costs 1.
    Standard deviation is the population value, with denominator valid count.
    """
    array = _numeric_array(scores, "scores")
    if array.ndim != 3 or array.shape[0] != 4 or array.shape[2] != 2:
        raise ValueError("scores must have shape [4, N, 2]")
    if np.isinf(array).any():
        raise ValueError("scores may contain NaN for missing patches, not infinity")
    if np.any(np.abs(array) > 1.0 + SCORE_TOLERANCE):
        raise ValueError("finite ZNCC scores must be within [-1, 1]")
    np.clip(array, -1.0, 1.0, out=array)

    old, new = array[:, :, 0].T, array[:, :, 1].T
    old_valid, new_valid = np.isfinite(old), np.isfinite(new)
    pair_valid = old_valid & new_valid
    old_cost = np.where(old_valid, 1.0 - old, 1.0)
    new_cost = np.where(new_valid, 1.0 - new, 1.0)
    margin = np.where(pair_valid, new - old, 0.0)

    count = pair_valid.sum(axis=1)
    denominator = np.maximum(count, 1)
    supported = count > 0
    mean_margin = margin.sum(axis=1) / denominator
    min_margin = np.where(
        supported, np.where(pair_valid, margin, np.inf).min(axis=1), 0.0
    )
    max_margin = np.where(
        supported, np.where(pair_valid, margin, -np.inf).max(axis=1), 0.0
    )
    squared_deviation = np.where(
        pair_valid, (margin - mean_margin[:, None]) ** 2, 0.0
    )
    std_margin = np.sqrt(squared_deviation.sum(axis=1) / denominator)
    win_fraction = ((margin > 0.0) & pair_valid).sum(axis=1) / denominator
    valid_fraction = count / 4.0
    mean_old_cost = np.where(
        supported, np.where(pair_valid, old_cost, 0.0).sum(axis=1) / denominator,
        1.0,
    )
    mean_new_cost = np.where(
        supported, np.where(pair_valid, new_cost, 0.0).sum(axis=1) / denominator,
        1.0,
    )
    aggregates = np.column_stack((
        mean_margin, min_margin, max_margin, std_margin, win_fraction,
        valid_fraction, mean_old_cost, mean_new_cost,
    ))
    return np.concatenate((
        old_cost, new_cost, margin, pair_valid, old_valid, new_valid, aggregates,
    ), axis=1).astype(np.float32)


def gate_pairs(features: np.ndarray) -> np.ndarray:
    """Select proposals with at least two paired views and positive mean gain."""
    array = _numeric_array(features, "features")
    if array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES):
        raise ValueError("features must have shape [N, 32]")
    if not np.isfinite(array).all():
        raise ValueError("features must be finite")
    flags = array[:, 12:16]
    if not np.isin(flags, (0.0, 1.0)).all():
        raise ValueError("paired-valid flags must be zero or one")
    return (flags.sum(axis=1) >= 2.0) & (array[:, 24] > 0.0)
