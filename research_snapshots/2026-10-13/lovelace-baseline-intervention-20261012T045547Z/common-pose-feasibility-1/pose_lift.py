"""Exact common-camera completion of one frozen twelve-coordinate A6 seed.

All nine darkness values AND all three t values remain fixed. Each view has
one horizontal camera offset, shared over its original 48 pixels; the middle
offset is zero. At fixed depth, rectangle areas are continuous piecewise
affine in that scalar offset. All projected edge crossings partition its
closed domain, and exact one-dimensional halfspace intersections decide
whether this particular seed has a completion. A failed lift says nothing
about other materials/biases or about depth infeasibility.

Only frozen independent geometry and RGB bands are loaded. Importing this
module never loads an observation, a result, a task, or a private fact file.
"""

from fractions import Fraction as Q
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
_source = PARENT / "appearance-1" / "verify_appearance.py"
_spec = importlib.util.spec_from_file_location("_pose_lift_independent_appearance", _source)
_appearance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_appearance)
geometry = _appearance._baseline
if Path(geometry.__file__).resolve() != (PARENT / "recovery-1" / "verify_baseline.py").resolve():
    raise RuntimeError("wrong frozen independent geometry identity")

pixel_bands = _appearance.pixel_bands
DOMAIN, RADIUS = geometry.DOMAIN, geometry.RADIUS
BETA, EPSILON = geometry.BETA, geometry.EPSILON
BASELINE, DRIFT_BOUND = Q(80), Q(1, 10)
BRANCHES = geometry.BRANCHES
STATUSES = ("SEED_LIFTED", "SEED_NOT_LIFTABLE")
PIXELS_PER_VIEW = len(geometry.PIXELS)


def exact(value, name="exact scalar"):
    """Accept frozen rational dictionaries, rational strings, int or Fraction."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be exact, not bool")
    if isinstance(value, (Q, int)):
        return Q(value)
    if isinstance(value, str):
        return Q(value)
    if isinstance(value, dict):
        if set(value) != {"numerator", "denominator"}:
            raise ValueError(f"{name} requires numerator and denominator only")
        n, d = value["numerator"], value["denominator"]
        if any(not isinstance(item, int) or isinstance(item, bool) for item in (n, d)):
            raise TypeError(f"{name} requires integer numerator and denominator")
        if d <= 0:
            raise ValueError(f"{name} requires a positive denominator")
        return Q(n, d)
    raise TypeError(f"{name} must be an exact rational")


def encode(value):
    value = exact(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _encoded_pair(pair):
    return None if pair is None else [encode(value) for value in pair]


def _inputs(public, arm, depth, branch, witness):
    if not isinstance(public, dict) or not isinstance(arm, dict):
        raise TypeError("public and arm must be frozen record dictionaries")
    if branch not in BRANCHES:
        raise ValueError("unregistered branch")
    z = exact(depth, "public depth")
    radius = exact(public["radius"], "public radius")
    baseline = exact(public["baseline"], "public baseline")
    clipped = public["clipped"]
    if baseline != BASELINE or radius != RADIUS or clipped is not True:
        raise ValueError("A7 requires frozen B80, radius 1/5 and clipped=True")
    if not DOMAIN[0] <= z <= DOMAIN[1]:
        raise ValueError("public depth is outside the frozen domain")
    if arm.get("appearance_model", "bounded") != "bounded":
        raise ValueError("seed lift requires the bounded appearance arm")
    if exact(arm.get("drift_bound", DRIFT_BOUND), "drift bound") != DRIFT_BOUND:
        raise ValueError("seed lift retains pairwise drift 1/10")
    if "branches" in arm and arm["branches"][branch].get("mode", "joint") != "joint":
        raise ValueError("A7 lifts joint seeds only")
    image = tuple(exact(value, "observation channel") for value in public["observation"])
    bands = pixel_bands(image, True, epsilon=EPSILON, beta=BETA)
    if isinstance(witness, (str, bytes, dict)):
        raise TypeError("witness must be a twelve-coordinate rational sequence")
    seed = tuple(exact(value, "seed coordinate") for value in witness)
    if len(seed) != 12:
        raise ValueError("a fixed joint seed requires nine darkness and three t coordinates")
    darkness, ts = seed[:9], seed[9:]
    if any(not 0 <= value <= 1 for value in darkness):
        raise ValueError("seed darkness must lie in [0,1]")
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if any(abs(darkness[3 * first + k] - darkness[3 * second + k]) > DRIFT_BOUND
               for k in range(3)):
            raise ValueError("seed violates the fixed same-material pairwise drift")
    calibration = tuple(tuple(exact(value, "calibration endpoint") for value in pair)
                        for pair in arm["calibration_intervals"])
    if len(calibration) != 3 or any(len(pair) != 2 for pair in calibration):
        raise ValueError("three frozen calibration intervals required")
    if any(not 0 <= lo <= hi <= 2 * BETA or not lo <= t <= hi
           for (lo, hi), t in zip(calibration, ts)):
        raise ValueError("fixed seed t lies outside legal frozen calibration")
    return z, radius, darkness, ts, bands


def camera_cells(branch, depth, view, radius=RADIUS):
    """All exact affine-edge crossings, including harmless overpartitioning.

    Horizontal front edges are -20, 0, 20 at z; rear edges are branch-specific
    at 720. Pixel edges are constant image coordinates. Parallel edges never
    cross; each other pair has at most one exact crossing. Closed adjacent
    cells retain their shared endpoints. The reference gauge is {0}.
    """
    z, radius = exact(depth), exact(radius)
    if branch not in BRANCHES or type(view) is not int or view not in (0, 1, 2):
        raise ValueError("registered branch and view required")
    if not DOMAIN[0] <= z <= DOMAIN[1] or not 0 <= radius <= RADIUS:
        raise ValueError("registered depth and camera radius required")
    if view == 1:
        return (Q(0),), ((Q(0), Q(0)),)
    center = (-BASELINE, Q(0), BASELINE)[view]
    edges = [((x - center) / z, -Q(1) / z) for x in (Q(-20), Q(0), Q(20))]
    rear = None if branch == "none" else geometry.REAR if branch == "partial" else geometry.HIDDEN
    if rear is not None:
        edges.extend(((x - center) / geometry.REAR_DEPTH, -Q(1) / geometry.REAR_DEPTH)
                     for x in rear[:2])
    edges.extend((x, Q(0)) for x in sorted({x for pixel in geometry.PIXELS for x in pixel[:2]}))
    cuts = {-radius, radius}
    for index, (first, slope) in enumerate(edges):
        for second, other_slope in edges[index + 1:]:
            if slope == other_slope:
                continue
            crossing = (second - first) / (slope - other_slope)
            if -radius <= crossing <= radius:
                cuts.add(crossing)
    points = tuple(sorted(cuts))
    cells = tuple(zip(points, points[1:])) if len(points) > 1 else ((points[0], points[0]),)
    return points, cells


@lru_cache(maxsize=4096)
def _view_areas(branch, depth, view, offset):
    offsets = tuple(offset if index == view else Q(0) for index in range(3))
    areas = geometry.independent_point_areas(branch, depth, offsets, BASELINE)
    start = PIXELS_PER_VIEW * view
    return areas[start:start + PIXELS_PER_VIEW]


def cell_affine_areas(branch, depth, view, cell):
    """Return 48 material triples of (intercept,slope), valid on one cell."""
    lo, hi = map(exact, cell)
    left, right = _view_areas(branch, depth, view, lo), _view_areas(branch, depth, view, hi)
    result = []
    for first, second in zip(left, right):
        pixel = []
        for a, b in zip(first, second):
            slope = Q(0) if lo == hi else (b - a) / (hi - lo)
            pixel.append((a - slope * lo, slope))
        result.append(tuple(pixel))
    return tuple(result)


def _intersect(interval, slope, intercept, side, bound):
    """Intersect a closed interval with slope*e+intercept <= or >= bound."""
    lo, hi = interval
    if side not in ("upper", "lower"):
        raise ValueError("registered inequality side required")
    if slope == 0:
        good = intercept <= bound if side == "upper" else intercept >= bound
        return interval if good else None
    endpoint = (bound - intercept) / slope
    if (side == "upper") == (slope > 0):
        hi = min(hi, endpoint)
    else:
        lo = max(lo, endpoint)
    return (lo, hi) if lo <= hi else None


def _cell_interval(affine, darkness, t, bands, cell):
    interval = cell
    for triples, (high, low) in zip(affine, bands):
        intercept = t + sum((a * d for (a, _), d in zip(triples, darkness)), Q(0))
        slope = sum((b * d for (_, b), d in zip(triples, darkness)), Q(0))
        for side, bound in (("upper", high), ("lower", low)):
            if bound is not None:
                interval = _intersect(interval, slope, intercept, side, bound)
                if interval is None:
                    return None
    return interval


def _offsets_satisfy(branch, depth, radius, darkness, ts, bands, offsets):
    if len(offsets) != 3 or offsets[1] != 0 or any(abs(e) > radius for e in offsets):
        return False
    areas = geometry.independent_point_areas(branch, depth, offsets, BASELINE)
    for pixel, (triples, (high, low)) in enumerate(zip(areas, bands)):
        view = pixel // PIXELS_PER_VIEW
        value = ts[view] + sum((area * d for area, d in
                               zip(triples, darkness[3 * view:3 * view + 3])), Q(0))
        if (high is not None and value > high) or (low is not None and value < low):
            return False
    return True


def verify_offsets(public, arm, depth, branch, witness, offsets):
    """Direct exact sufficient check, independent of cell interpolation."""
    try:
        z, radius, darkness, ts, bands = _inputs(public, arm, depth, branch, witness)
        if isinstance(offsets, (str, bytes, dict)):
            return False
        offsets = tuple(exact(value, "camera offset") for value in offsets)
        return _offsets_satisfy(branch, z, radius, darkness, ts, bands, offsets)
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return False


def lift_seed(public, arm, depth, branch, witness):
    """Decide exact offset completion for the unchanged d9+t3 seed only.

    JSON-safe per-view records include the entire finite closed partition and
    its exact feasible subintervals, also when a view fails. Successful offsets
    are deterministic midpoints of the first nonempty interval in each view.
    No material, bias, depth, observation or calibration value is refitted.
    """
    z, radius, darkness, ts, bands = _inputs(public, arm, depth, branch, witness)
    records, selected = [], []
    for view in range(3):
        points, cells = camera_cells(branch, z, view, radius)
        start = PIXELS_PER_VIEW * view
        view_bands = bands[start:start + PIXELS_PER_VIEW]
        entries, chosen = [], None
        for cell in cells:
            affine = cell_affine_areas(branch, z, view, cell)
            feasible = _cell_interval(affine, darkness[3 * view:3 * view + 3],
                                      ts[view], view_bands, cell)
            entries.append({"cell": _encoded_pair(cell), "feasible": _encoded_pair(feasible)})
            if chosen is None and feasible is not None:
                chosen = (feasible[0] + feasible[1]) / 2
        records.append({"view": view, "breakpoints": [encode(value) for value in points],
                        "cells": entries, "selected": None if chosen is None else encode(chosen)})
        selected.append(chosen)
    if any(value is None for value in selected):
        return {"status": "SEED_NOT_LIFTABLE", "offsets": None, "views": records,
                "reason": "fixed_seed_has_no_common_offset_completion"}
    offsets = tuple(selected)
    if not _offsets_satisfy(branch, z, radius, darkness, ts, bands, offsets):
        raise ArithmeticError("piecewise candidate failed independent exact point geometry")
    return {"status": "SEED_LIFTED", "offsets": [encode(value) for value in offsets],
            "views": records, "reason": "exact_fixed_seed_offset_completion"}


def clear_geometry_cache():
    _view_areas.cache_clear()
