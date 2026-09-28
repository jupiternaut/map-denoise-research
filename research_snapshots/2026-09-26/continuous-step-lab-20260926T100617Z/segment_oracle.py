"""Exact segment-to-finite-reference oracle; evaluation-only, never a selector.

For each segment [p0, p1], minimize squared Euclidean distance jointly over
lambda in [0, 1] and all points of the supplied finite reference cloud.  This
is not a point-to-continuous-surface oracle and is not deployable without the
evaluation reference.  No sampling or lambda grid is used.

Candidate retrieval is exact: let m be the segment midpoint, h its half
length, and u a feasible segment-to-reference distance.  Any improving
reference r has ||r-m|| <= h + u by the triangle inequality.  Query that ball,
project every returned reference onto the segment, and take the minimum.
The initial feasible distance is tightened using reference points nearest to
the start, midpoint, and end.  A floating-point outward radius pad can only
add candidates; it does not exclude any.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SegmentNearestResult:
    squared_distance: np.ndarray
    lambda_: np.ndarray
    reference_index: np.ndarray
    candidate_evaluations: int
    max_candidates_per_segment: int
    seconds: float


def _points(value, name):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite")
    return points


def segment_nearest(p0, p1, reference=None, *, tree=None, chunk_size=2048,
                    workers=1):
    """Return the exact best lambda and distance for each finite segment.

    Supply either ``reference`` as an (M, 3) array / cKDTree or ``tree=...``.
    ``tree.data`` is the sole reference if a tree is supplied.  Distances keep
    the input coordinate unit; ``squared_distance`` has its square.  Exact
    floating-point loss ties favor the smallest lambda, so KEEP is favored
    whenever lambda=0 is tied.  Reference-index ties favor the lowest index.

    Memory is O(chunk_size plus the retrieved candidate pairs), not O(N*M).
    Candidate count can still be large for long segments or sparse references;
    callers can reduce chunk_size in that case.  The three-neighbor upper-bound
    probes are not included in ``candidate_evaluations``.
    """
    started = perf_counter()
    start = _points(p0, "p0")
    end = _points(p1, "p1")
    if start.shape != end.shape:
        raise ValueError("p0 and p1 must have the same shape")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if tree is not None and reference is not None:
        raise ValueError("provide reference or tree, not both")
    if tree is None:
        tree = reference if isinstance(reference, cKDTree) else cKDTree(
            _points(reference, "reference"))
    refs = _points(tree.data, "tree.data")
    if len(refs) == 0:
        raise ValueError("reference must not be empty")

    count = len(start)
    best_sq = np.empty(count, dtype=np.float64)
    best_lambda = np.empty(count, dtype=np.float64)
    best_ref = np.empty(count, dtype=np.int64)
    candidate_evaluations = 0
    maximum_candidates = 0
    for lo in range(0, count, chunk_size):
        hi = min(lo + chunk_size, count)
        a, b = start[lo:hi], end[lo:hi]
        direction = b - a
        length_sq = np.einsum("ij,ij->i", direction, direction)
        half_length = 0.5 * np.sqrt(length_sq)
        midpoint = a + 0.5 * direction

        probes = np.stack((a, midpoint, b), axis=1)
        _, probe_indices = tree.query(probes, workers=workers)
        relative = refs[probe_indices] - a[:, None, :]
        dot = np.einsum("nkj,nj->nk", relative, direction)
        probe_lambda = np.divide(dot, length_sq[:, None],
                                 out=np.zeros_like(dot),
                                 where=length_sq[:, None] > 0)
        np.clip(probe_lambda, 0, 1, out=probe_lambda)
        probe_residual = relative - probe_lambda[:, :, None] * direction[:, None, :]
        upper_sq = np.min(np.einsum("nkj,nkj->nk", probe_residual,
                                   probe_residual), axis=1)
        radius = half_length + np.sqrt(upper_sq)
        radius += 64 * np.finfo(np.float64).eps * (
            1 + radius + np.max(np.abs(midpoint), axis=1))
        radius = np.nextafter(radius, np.inf)
        neighbors = tree.query_ball_point(midpoint, radius, workers=workers,
                                          return_sorted=False)
        sizes = np.fromiter((len(ids) for ids in neighbors), dtype=np.int64,
                            count=hi - lo)
        if np.any(sizes == 0):
            raise RuntimeError("outward bounding ball lost its feasible reference")
        candidate_evaluations += int(sizes.sum())
        maximum_candidates = max(maximum_candidates, int(sizes.max()))
        rows = np.repeat(np.arange(hi - lo), sizes)
        ids = np.concatenate(neighbors).astype(np.int64, copy=False)
        relative = refs[ids] - a[rows]
        candidate_dot = np.einsum("ij,ij->i", relative, direction[rows])
        candidate_lambda = np.divide(candidate_dot, length_sq[rows],
                                     out=np.zeros_like(candidate_dot),
                                     where=length_sq[rows] > 0)
        np.clip(candidate_lambda, 0, 1, out=candidate_lambda)
        residual = relative - candidate_lambda[:, None] * direction[rows]
        candidate_sq = np.einsum("ij,ij->i", residual, residual)

        local_sq = np.full(hi - lo, np.inf)
        np.minimum.at(local_sq, rows, candidate_sq)
        tied_distance = candidate_sq == local_sq[rows]
        local_lambda = np.full(hi - lo, np.inf)
        np.minimum.at(local_lambda, rows[tied_distance], candidate_lambda[tied_distance])
        tied_lambda = tied_distance & (candidate_lambda == local_lambda[rows])
        local_ref = np.full(hi - lo, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(local_ref, rows[tied_lambda], ids[tied_lambda])
        best_sq[lo:hi] = local_sq
        best_lambda[lo:hi] = local_lambda
        best_ref[lo:hi] = local_ref

    return SegmentNearestResult(best_sq, best_lambda, best_ref,
                                candidate_evaluations, maximum_candidates,
                                perf_counter() - started)
