"""Exact necessary depth outer bounds with shared public view bias.

The only scientific inputs are the observed RGB image, its clipping semantics,
the public camera radius, and registered algorithm parameters. The historical
geometry is read-only. Retention is an incomplete necessary-condition result,
not a proof of joint geometry/material feasibility.
"""
from collections import deque
from fractions import Fraction as Q
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
GEOMETRY_ROOT = Path("/srv/slam-research/grf/map-denoise/runs/lovelace-shared-ablation-20261011T055200Z")
if str(GEOMETRY_ROOT) not in sys.path:
    sys.path.insert(0, str(GEOMETRY_ROOT))
import grouped_solver as historical
from grouped_solver import nuisance_model, interval_model, encode, decode, rational

BRANCHES = historical.BRANCHES
PIXELS_PER_VIEW = historical.PIXELS_PER_VIEW
VIEWS = len(BRANCHES)
METHODS = ("calibrated", "joint")
BETA = Q(1, 100)
EPSILON = Q(1, 100)
ROUNDS = 8
MAX_BOXES = 511
TOLERANCE = Q(15, 32)
MATERIAL_BOUNDS = ((Q(0), Q(1)),) * 3


def _nonnegative(value, name):
    result = rational(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _parameters(method, clipped, beta, epsilon, rounds, max_boxes, tolerance):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not isinstance(clipped, bool):
        raise TypeError("clipped must be bool")
    beta, epsilon = _nonnegative(beta, "beta"), _nonnegative(epsilon, "epsilon")
    for value, name in ((rounds, "rounds"), (max_boxes, "max_boxes")):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    tolerance = rational(tolerance)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    return beta, epsilon, tolerance


def _encoded_pairs(bounds):
    return [[encode(lo), encode(hi)] for lo, hi in bounds]


def pixel_bands(observation, *, clipped, beta=BETA, epsilon=EPSILON):
    """Per-pixel necessary bands for sum(A*d)+t; absent sides are None.

    The three RGB channels share material and view bias, but have separate
    bounded errors. Their present lower/upper sides are intersected exactly.
    Helpers allow small RGB fixtures; solve_structured enforces the full layout.
    """
    if not isinstance(clipped, bool):
        raise TypeError("clipped must be bool")
    beta, epsilon = _nonnegative(beta, "beta"), _nonnegative(epsilon, "epsilon")
    image = tuple(map(rational, observation))
    if not image or len(image) % 3:
        raise ValueError("a nonempty RGB image is required")
    if clipped and any(not 0 <= y <= 1 for y in image):
        raise ValueError("clipped observations must lie in [0,1]")
    result = []
    for pixel in range(len(image) // 3):
        lowers, uppers = [], []
        for y in image[3 * pixel:3 * pixel + 3]:
            if not clipped or y != 1:
                lowers.append(1 + beta - y - epsilon)
            if not clipped or y != 0:
                uppers.append(1 + beta - y + epsilon)
        result.append((max(lowers) if lowers else None,
                       min(uppers) if uppers else None))
    return tuple(result)


def background_mask(radius):
    """Derive guaranteed background from every public branch on full depth.

    Historical geometry bounds already quantify over the allowed continuous
    camera errors. A pixel is background only when every area's upper endpoint
    is zero in every branch. No generating branch or depth enters this mask.
    """
    radius = _nonnegative(radius, "radius")
    geometry = [nuisance_model._geometry_bounds(branch, radius,
                interval_model.DEPTH_MIN, interval_model.DEPTH_MAX)
                for branch in BRANCHES]
    size = VIEWS * PIXELS_PER_VIEW
    if any(len(bounds) != size for bounds in geometry):
        raise ValueError("historical geometry layout mismatch")
    return tuple(all(all(a == 0 for a in branch[pixel][1::2])
                     for branch in geometry) for pixel in range(size))


def calibration_bounds(observation, mask, *, clipped, beta=BETA, epsilon=EPSILON):
    """Intersect same-image background t rows with its public prior.

    The certificate has deterministic pixel order and upper-before-lower
    intersections. Crossed intervals explicitly mean INCOMPATIBLE.
    """
    beta, epsilon = _nonnegative(beta, "beta"), _nonnegative(epsilon, "epsilon")
    bands = pixel_bands(observation, clipped=clipped, beta=beta, epsilon=epsilon)
    mask = tuple(mask)
    if len(mask) != len(bands) or len(mask) != VIEWS * PIXELS_PER_VIEW:
        raise ValueError("calibration requires the full public RGB layout and mask")
    if any(not isinstance(value, bool) for value in mask):
        raise TypeError("background mask entries must be bool")
    bounds = [[Q(0), 2 * beta] for _ in range(VIEWS)]
    steps, contradiction = [], None
    for pixel, ((lower, upper), is_background) in enumerate(zip(bands, mask)):
        if not is_background:
            continue
        view = pixel // PIXELS_PER_VIEW
        for side, value in (("upper", upper), ("lower", lower)):
            if value is None:
                continue
            endpoint = 1 if side == "upper" else 0
            changed = value < bounds[view][1] if endpoint else value > bounds[view][0]
            if changed:
                bounds[view][endpoint] = value
                steps.append({"pixel": pixel, "view": view, "side": side,
                              "value": encode(value)})
            if bounds[view][0] > bounds[view][1]:
                contradiction = {"kind": "interval", "pixel": pixel, "view": view,
                                 "lower": encode(bounds[view][0]),
                                 "upper": encode(bounds[view][1])}
                break
        if contradiction is not None:
            break
    counts = [sum(mask[v * PIXELS_PER_VIEW:(v + 1) * PIXELS_PER_VIEW]) for v in range(VIEWS)]
    return {"intervals": _encoded_pairs(bounds), "excluded": contradiction is not None,
            "certificate": {"steps": steps, "contradiction": contradiction},
            "prior": [encode(Q(0)), encode(2 * beta)],
            "background_pixel_count": sum(mask), "background_pixels_per_view": counts}


def necessary_rows(observation, geometry, method, *, clipped,
                   calibration_intervals=None, beta=BETA, epsilon=EPSILON):
    """Necessary nonnegative upper/lower rows in deterministic pixel order.

    Upper rows use area lower endpoints; lower rows use area upper endpoints.
    Joint rows have d0,d1,d2,t0,t1,t2. Calibrated rows have d0,d1,d2 and
    marginalize the frozen background-derived t intervals. Small geometry
    fixtures may use up to the full public pixel count.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    bands = pixel_bands(observation, clipped=clipped, beta=beta, epsilon=epsilon)
    areas = tuple(tuple(map(rational, pair)) for pair in geometry)
    if len(areas) != len(bands) or len(areas) > VIEWS * PIXELS_PER_VIEW:
        raise ValueError("RGB observation and area bounds must agree")
    if any(len(pair) != 6 or any(not 0 <= lo <= hi <= 1
           for lo, hi in zip(pair[::2], pair[1::2])) for pair in areas):
        raise ValueError("three ordered nonnegative area intervals required per pixel")
    if method == "calibrated":
        if calibration_intervals is None:
            raise ValueError("calibrated rows require frozen t intervals")
        ts = tuple(tuple(map(rational, pair)) for pair in calibration_intervals)
        if len(ts) != VIEWS or any(len(pair) != 2 or not 0 <= pair[0] <= pair[1]
                                  for pair in ts):
            raise ValueError("three ordered nonnegative t intervals required")
    rows = []
    for pixel, (area, (lower, upper)) in enumerate(zip(areas, bands)):
        view = pixel // PIXELS_PER_VIEW
        lows, highs = area[::2], area[1::2]
        if method == "joint":
            t_coefficients = tuple(Q(int(v == view)) for v in range(VIEWS))
            if upper is not None:
                rows.append(("upper", lows + t_coefficients, upper))
            if lower is not None:
                rows.append(("lower", highs + t_coefficients, lower))
        else:
            if upper is not None:
                rows.append(("upper", lows, upper - ts[view][0]))
            if lower is not None:
                rows.append(("lower", highs, lower - ts[view][1]))
    return tuple(rows)


def propagate(rows, bounds=MATERIAL_BOUNDS, rounds=ROUNDS):
    """Finite exact contractor for 3 or 6 nonnegative variables.

    Updates use a row's starting lower endpoints for an upper inequality and
    starting upper endpoints for a lower inequality. Opposite-side tightening
    does not alter that row total. The first explicit row or crossed-interval
    contradiction yields a replayable certificate. Zero rounds retain valid
    priors. Invalid dimensions, signs and inexact floats are rejected.
    """
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer")
    initial = tuple(tuple(map(rational, pair)) for pair in bounds)
    if len(initial) not in (3, 6) or any(len(pair) != 2 or min(pair) < 0 for pair in initial):
        raise ValueError("three or six nonnegative variable intervals required")
    necessary = []
    for kind, coefficients, bound in rows:
        coefficients, bound = tuple(map(rational, coefficients)), rational(bound)
        if kind not in ("upper", "lower") or len(coefficients) != len(initial) or any(a < 0 for a in coefficients):
            raise ValueError("rows require upper/lower and matching nonnegative coefficients")
        necessary.append((kind, coefficients, bound))
    lower, upper = [pair[0] for pair in initial], [pair[1] for pair in initial]
    steps, performed = [], 0

    def excluded(contradiction):
        return {"excluded": True, "rounds": performed, "updates": len(steps),
                "certificate": {"initial_bounds": _encoded_pairs(initial),
                                "steps": steps, "contradiction": contradiction}}

    for variable, (lo, hi) in enumerate(initial):
        if lo > hi:
            return excluded({"kind": "initial_interval", "variable": variable,
                             "lower": encode(lo), "upper": encode(hi)})
    for iteration in range(rounds):
        performed = iteration + 1
        changed = False
        for index, (kind, coefficients, bound) in enumerate(necessary):
            values = lower if kind == "upper" else upper
            total = sum((a * value for a, value in zip(coefficients, values)), Q(0))
            if (total > bound if kind == "upper" else total < bound):
                return excluded({"kind": "row", "round": iteration, "row": index,
                                 "total": encode(total), "bound": encode(bound)})
            for variable, coefficient in enumerate(coefficients):
                if coefficient == 0:
                    continue
                other = total - coefficient * values[variable]
                candidate = (bound - other) / coefficient
                if kind == "upper" and candidate < upper[variable]:
                    upper[variable] = candidate
                    side = "upper"
                elif kind == "lower" and candidate > lower[variable]:
                    lower[variable] = candidate
                    side = "lower"
                else:
                    continue
                steps.append({"round": iteration, "row": index, "variable": variable,
                              "side": side, "value": encode(candidate)})
                changed = True
                if lower[variable] > upper[variable]:
                    return excluded({"kind": "interval", "round": iteration, "row": index,
                                     "variable": variable, "lower": encode(lower[variable]),
                                     "upper": encode(upper[variable])})
        if not changed:
            break
    return {"excluded": False, "rounds": performed, "updates": len(steps),
            "bounds": _encoded_pairs(zip(lower, upper))}


def _solve_branch(image, radius, branch, method, clipped, beta, epsilon, rounds,
                  max_boxes, tolerance, calibration):
    depth_lo, depth_hi = interval_model.DEPTH_MIN, interval_model.DEPTH_MAX
    ts = tuple(tuple(map(decode, pair)) for pair in calibration["intervals"])
    initial = MATERIAL_BOUNDS + ts if method == "joint" else MATERIAL_BOUNDS
    queue = deque([(0, None, depth_lo, depth_hi)])
    records, retained = [], []
    next_id, evaluated, shared_calls = 1, 0, 0
    while queue:
        node_id, parent, left, right = queue.popleft()
        entry = {"id": node_id, "parent": parent, "lo": encode(left), "hi": encode(right)}
        if calibration["excluded"]:
            entry.update(status="excluded_calibration", certificate=calibration["certificate"])
            records.append(entry)
            continue
        if evaluated >= max_boxes:
            entry.update(status="unresolved_budget", darkness_bounds=_encoded_pairs(MATERIAL_BOUNDS),
                         t_bounds=calibration["intervals"])
            records.append(entry)
            retained.append((left, right, node_id))
            continue
        geometry = nuisance_model._geometry_bounds(branch, radius, left, right)
        rows = necessary_rows(image, geometry, method, clipped=clipped,
                              calibration_intervals=ts, beta=beta, epsilon=epsilon)
        evaluated += 1
        contraction = propagate(rows, bounds=initial, rounds=rounds)
        shared_calls += 1
        entry.update(rounds=contraction["rounds"], updates=contraction["updates"])
        if contraction["excluded"]:
            entry.update(status="excluded_shared", group_id="global",
                         certificate=contraction["certificate"])
        else:
            final = contraction["bounds"]
            entry.update(darkness_bounds=final[:3],
                         t_bounds=final[3:] if method == "joint" else calibration["intervals"])
            if right - left <= tolerance:
                entry["status"] = "retained_tolerance"
                retained.append((left, right, node_id))
            else:
                middle = (left + right) / 2
                entry.update(status="split", children=[next_id, next_id + 1])
                queue.append((next_id, node_id, left, middle))
                queue.append((next_id + 1, node_id, middle, right))
                next_id += 2
        records.append(entry)
    hull = None if not retained else [encode(min(a for a, _, _ in retained)),
                                     encode(max(b for _, b, _ in retained))]
    return {"domain": [encode(depth_lo), encode(depth_hi)], "epsilon": encode(epsilon),
            "beta": encode(beta), "clipped": clipped, "tolerance": encode(tolerance),
            "max_boxes": max_boxes, "evaluated_boxes": evaluated, "trace": records,
            "retained_ids": [node_id for _, _, node_id in retained], "hull": hull,
            "shared_tests": shared_calls, "propagation_rounds": rounds,
            "mode": method, "rounds": rounds, "budget": max_boxes,
            "channel_precheck": "none", "initial_bounds": _encoded_pairs(initial),
            "scope": "necessary_shared_bias_halfspaces_with_historical_continuous_geometry_bounds"}


def union_outer(branches, *, method, clipped, beta=BETA, epsilon=EPSILON,
                rounds=ROUNDS, max_boxes=MAX_BOXES, tolerance=TOLERANCE):
    """Preserve the historical outer/action schema with structured metadata."""
    result = historical.union_outer(branches, max_boxes=max_boxes, rounds=rounds,
                                    tolerance=tolerance, epsilon=epsilon, mode="global_gray")
    result.update(mode=method, beta=encode(beta), clipped=clipped, channel_precheck="none")
    return result


def solve_structured(observation, radius, method, *, clipped, beta=BETA,
                     epsilon=EPSILON, rounds=ROUNDS, max_boxes=MAX_BOXES,
                     tolerance=TOLERANCE):
    """Solve the public image across all three registered depth hypotheses.

    Both methods derive the identical calibration once; joint initializes six
    variables with it and refines t per box, while calibrated freezes t and
    contracts three materials. Preparation is included in total method timing.
    No channelwise precheck is used: every exclusion comes from clip-aware
    necessary rows or public background interval intersection.
    """
    beta, epsilon, tolerance = _parameters(method, clipped, beta, epsilon,
                                           rounds, max_boxes, tolerance)
    radius = _nonnegative(radius, "radius")
    image = tuple(map(rational, observation))
    if len(image) != 3 * VIEWS * PIXELS_PER_VIEW:
        raise ValueError("observation must match the historical public RGB layout")
    started = time.perf_counter()
    mask = background_mask(radius)
    calibration = calibration_bounds(image, mask, clipped=clipped, beta=beta, epsilon=epsilon)
    preparation_seconds = time.perf_counter() - started
    branches, seconds = {}, {}
    for branch in BRANCHES:
        tick = time.perf_counter()
        branches[branch] = _solve_branch(image, radius, branch, method, clipped,
                                        beta, epsilon, rounds, max_boxes, tolerance, calibration)
        seconds[branch] = time.perf_counter() - tick
    outer = union_outer(branches, method=method, clipped=clipped, beta=beta,
                        epsilon=epsilon, rounds=rounds, max_boxes=max_boxes, tolerance=tolerance)
    return {"branches": branches, "outer": outer, "branch_seconds": seconds,
            "outer_seconds": time.perf_counter() - started,
            "preparation_seconds": preparation_seconds, "background_mask": list(mask),
            "calibration_intervals": calibration["intervals"], "calibration": calibration}


def clear_geometry_cache():
    """Clear only the historical geometry caches before measured cold solves."""
    historical.clear_geometry_cache()
