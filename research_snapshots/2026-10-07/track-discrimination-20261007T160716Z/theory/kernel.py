"""Exact affine gain extrema over finite unions of closed, finite intervals.

Intervals describe feasible depth of ONE reference-ray target. Callers own
the physical-identity and coverage assumptions; this module cannot prove them.
"""
import math
import sys
from fractions import Fraction


def _extrema(intervals):
    intervals = list(intervals)
    if not intervals:
        return None
    lo, hi = math.inf, -math.inf
    for left, right in intervals:
        left, right = float(left), float(right)
        if not (math.isfinite(left) and math.isfinite(right)) or left > right:
            raise ValueError("interval endpoints must be finite and ordered")
        lo, hi = min(lo, left), max(hi, right)
    return lo, hi


def _bounds(constant, slope, intervals):
    extent = _extrema(intervals)
    if extent is None:
        return None
    values = [constant + slope * Fraction(z) for z in extent]
    return _outward(min(values), lower=True), _outward(max(values), lower=False)


def _outward(value, lower):
    """Outward round an exact rational endpoint to a binary float."""
    try:
        rounded = float(value)
    except OverflowError:
        rounded = math.inf if value > 0 else -math.inf
    if math.isinf(rounded):
        if rounded > 0:
            return sys.float_info.max if lower else math.inf
        return -math.inf if lower else -sys.float_info.max
    exact_rounded = Fraction(rounded)
    if lower and exact_rounded > value:
        return math.nextafter(rounded, -math.inf)
    if not lower and exact_rounded < value:
        return math.nextafter(rounded, math.inf)
    return rounded


def gain_bounds(a, b, intervals):
    """Return inf/sup of (a-z)^2-(b-z)^2, or None for an empty set.

    Positive lower bound certifies uniformly positive depth-squared gain,
    conditional on a nonempty feasible set covering the actual same-layer z.
    Gain arithmetic is exact for represented input floats; outputs are rounded
    outwards. Coverage and uncertainty of supplied endpoints remain external.
    """
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("candidate depths must be finite")
    a, b = Fraction(a), Fraction(b)
    return _bounds((a - b) * (a + b), 2 * (b - a), intervals)


def gain_bounds_world(a, b, C, r, intervals):
    """Bound ||a-C-z*r||² - ||b-C-z*r||² for arbitrary 3D a,b.

    r need not be unit: for optical-Z parameterization r is K^-1[u,v,1]
    rotated into world coordinates. All inputs must share coordinate units.
    """
    vectors = [tuple(float(x) for x in v) for v in (a, b, C, r)]
    if any(len(v) != 3 for v in vectors):
        raise ValueError("a, b, C and r must have three coordinates")
    if not all(math.isfinite(x) for v in vectors for x in v):
        raise ValueError("coordinates must be finite")
    a, b, C, r = [tuple(Fraction(x) for x in vector) for vector in vectors]
    if not any(x != 0.0 for x in r):
        raise ValueError("ray direction must be nonzero")
    constant = sum((ai - bi) * (ai + bi - 2 * ci)
                   for ai, bi, ci in zip(a, b, C))
    slope = -2 * sum(ri * (ai - bi) for ri, ai, bi in zip(r, a, b))
    return _bounds(constant, slope, intervals)


def strictly_improves(bounds, gain_floor=0.0):
    """An empty set never yields vacuous acceptance."""
    return bounds is not None and bounds[0] > gain_floor
