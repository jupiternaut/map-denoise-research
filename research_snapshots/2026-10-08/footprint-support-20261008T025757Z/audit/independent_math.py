"""Independent replay math; imports no experimental scorer or evaluator."""

import numpy as np


def reconstruct_intervals(grid, accepted, padding):
    """Union the padded cells of accepted grid points by an explicit scan."""
    grid = np.asarray(grid, dtype=float)
    accepted = np.asarray(accepted, dtype=bool)
    half = (grid[1] - grid[0]) / 2 if len(grid) > 1 else 0.0
    result = []
    for depth in grid[accepted]:
        start = max(float(grid[0]), float(depth - half - padding))
        stop = min(float(grid[-1]), float(depth + half + padding))
        if result and start <= result[-1][1] + 1e-9:
            result[-1][1] = max(result[-1][1], stop)
        else:
            result.append([start, stop])
    return result


def choose(intervals, candidates, incumbent):
    """Length-uniform support mean and frozen nearest fixed candidate rule."""
    mass = sum(hi - lo for lo, hi in intervals)
    if mass <= 0:
        return None, float(incumbent), list(candidates).index(incumbent), 0.0
    first_moment = sum((hi * hi - lo * lo) / 2 for lo, hi in intervals)
    mean = first_moment / mass
    # Explicit tuple order makes the historical first-candidate tie rule clear.
    index = min(range(len(candidates)), key=lambda k: (abs(candidates[k] - mean), k))
    gain = abs(incumbent - mean) - abs(candidates[index] - mean)
    if gain <= 0:
        index = list(candidates).index(incumbent)
    return mean, float(candidates[index]), index, float(max(0, gain))


def length(intervals):
    return sum(hi - lo for lo, hi in intervals)


def overlap_length(first, second):
    return sum(max(0.0, min(a1, b1) - max(a0, b0))
               for a0, a1 in first for b0, b1 in second)


def symmetric_difference_length(first, second):
    return length(first) + length(second) - 2 * overlap_length(first, second)


def weighted_statistics(reference, source, weights):
    """Independent scalar-vector formula with explicit positive-weight slicing."""
    weights = np.asarray(weights, dtype=float)
    use = weights > 0
    a = np.asarray(reference, dtype=float)[use]
    b = np.asarray(source, dtype=float)[use]
    w = weights[use]
    mass = float(sum(w))
    if mass <= 0:
        return dict(mass=0.0, ess=0.0, reference_std=0.0, source_std=0.0,
                    ncc=float("nan"))
    ess = mass * mass / float(sum(w * w))
    ma = float(sum(w * a)) / mass
    mb = float(sum(w * b)) / mass
    va = float(sum(w * (a - ma) ** 2)) / mass
    vb = float(sum(w * (b - mb) ** 2)) / mass
    covariance = float(sum(w * (a - ma) * (b - mb))) / mass
    ncc = covariance / np.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")
    return dict(mass=mass, ess=ess, reference_std=float(np.sqrt(va)),
                source_std=float(np.sqrt(vb)), ncc=float(ncc))
