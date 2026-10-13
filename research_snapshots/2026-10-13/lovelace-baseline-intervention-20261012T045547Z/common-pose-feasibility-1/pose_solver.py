"""A7 bounded common-pose search; only exact clip-once witnesses are positive.

The two floating pose coordinates and every numerical LP are candidate
generators, never exclusion tests. A candidate is rationalized, its original
point areas are rebuilt, and the frozen A5 classifier recovers exact material
and bias coordinates. Explicit bounded RGB noise is finally checked against
the entire exact forward image. No private files are read by this module.
"""
from fractions import Fraction as Q
import math
from pathlib import Path
import sys
import time
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
POINT_ROOT = ROOT.parent / "point-feasibility-1"
if str(POINT_ROOT) not in sys.path:
    sys.path.insert(0, str(POINT_ROOT))
import point_solver as point

if Path(point.__file__).resolve() != (POINT_ROOT / "point_solver.py").resolve():
    raise RuntimeError("wrong frozen point solver source identity")

import numpy as np
from scipy.optimize import OptimizeWarning, linprog, minimize

geometry, variation = point.geometry, point.variation
encode, exact = point.encode, point.exact
BRANCHES = point.BRANCHES
RADIUS, BASELINE = Q(1, 5), Q(80)
EPSILON, BETA, DRIFT_BOUND = Q(1, 100), Q(1, 100), Q(1, 10)
MAX_LP_CALLS, MAX_EXACT_ATTEMPTS = 200, 20
BRANCH_SECONDS, MAX_POWELL_STARTS, POWELL_MAXFEV = 3.0, 2, 60
OFFSET_DENOMINATOR_LIMIT = 1_000_000_000
CANDIDATE_SLACK = 1e-7
STATUSES = ("FORWARD_WITNESS", "UNKNOWN")
WITNESS_FIELDS = {"offsets", "darkness", "t", "noise"}


def _inputs(public, arm, depth, branch, *, check_depth=True):
    if not isinstance(public, dict) or not isinstance(arm, dict):
        raise TypeError("public and arm must be record dictionaries")
    if branch not in BRANCHES:
        raise ValueError("unregistered occlusion branch")
    depth = exact(depth, "public depth")
    if (exact(public["radius"], "public radius") != RADIUS
            or exact(public["baseline"], "public baseline") != BASELINE
            or public["clipped"] is not True):
        raise ValueError("A7 requires frozen B80, radius 1/5, clip-once observations")
    if check_depth and not geometry.DEPTH_MIN <= depth <= geometry.DEPTH_MAX:
        raise ValueError("depth is outside the original public geometry domain")
    if arm.get("appearance_model", "bounded") != "bounded":
        raise ValueError("A7 requires the bounded appearance arm")
    if exact(arm.get("drift_bound", DRIFT_BOUND), "drift bound") != DRIFT_BOUND:
        raise ValueError("A7 retains pairwise drift 1/10")
    if "branches" in arm and arm["branches"][branch].get("mode", "joint") != "joint":
        raise ValueError("A7 uses the original joint arm only")
    image = variation.appearance_engine._image(public["observation"], True)
    ts = tuple(tuple(exact(value, "calibration endpoint") for value in pair)
               for pair in arm["calibration_intervals"])
    if len(ts) != 3 or any(len(pair) != 2 or not 0 <= pair[0] <= pair[1] <= 2 * BETA
                           for pair in ts):
        raise ValueError("three frozen global calibration intervals in [0,1/50] required")
    return depth, image, variation.PER_VIEW_MATERIAL_BOUNDS + ts


def _offsets(values):
    offsets = tuple(exact(value, "camera offset") for value in values)
    if (len(offsets) != 3 or offsets[1] != 0
            or any(abs(value) > RADIUS for value in offsets)):
        raise ValueError("three bounded rational offsets with zero middle gauge required")
    return offsets


def exact_rows(public, arm, depth, branch, offsets):
    """Return exact fixed-pose joint rows and the unchanged global d/t box."""
    depth, image, bounds = _inputs(public, arm, depth, branch)
    offsets = _offsets(offsets)
    areas = geometry._point_areas(branch, depth, offsets, BASELINE)
    degenerate = tuple(tuple(value for area in triple for value in (area, area))
                       for triple in areas)
    rows = variation.necessary_rows(image, degenerate, "joint", clipped=True,
                                   drift_bound=DRIFT_BOUND, beta=BETA, epsilon=EPSILON)
    return rows, bounds


def _prediction(depth, branch, offsets, darkness, ts):
    result = []
    for pixel, areas in enumerate(geometry._point_areas(branch, depth, offsets, BASELINE)):
        view = pixel // geometry.PIXELS_PER_VIEW
        gray = 1 - sum((area * darkness[3 * view + material]
                        for material, area in enumerate(areas)), Q(0))
        if not 0 <= gray <= 1:
            raise ArithmeticError("invalid original exact point integral")
        result.extend((gray + BETA - ts[view],) * 3)
    return tuple(result)


def check_forward_witness(public, arm, depth, branch, witness):
    """Check strict shapes, all original bounds and exact clip(pred+b+noise)."""
    try:
        depth, image, bounds = _inputs(public, arm, depth, branch)
        if not isinstance(witness, dict) or set(witness) != WITNESS_FIELDS:
            return False
        if any(not isinstance(witness[key], (list, tuple)) for key in WITNESS_FIELDS):
            return False
        offsets = _offsets(witness["offsets"])
        darkness = tuple(exact(value, "darkness") for value in witness["darkness"])
        ts = tuple(exact(value, "bias t") for value in witness["t"])
        noise = tuple(exact(value, "RGB noise") for value in witness["noise"])
        if (len(darkness) != 9 or len(ts) != 3 or len(noise) != 432
                or any(not lo <= value <= hi for value, (lo, hi) in zip(darkness + ts, bounds))
                or any(abs(value) > EPSILON for value in noise)):
            return False
        if any(abs(darkness[3 * first + material] - darkness[3 * second + material]) > DRIFT_BOUND
               for first, second in variation.VIEW_PAIRS for material in range(3)):
            return False
        raw = _prediction(depth, branch, offsets, darkness, ts)
        return all(min(Q(1), max(Q(0), value + error)) == observed
                   for value, error, observed in zip(raw, noise, image))
    except (TypeError, ValueError, KeyError, ZeroDivisionError, OverflowError, ArithmeticError):
        return False


def _reconstruct_witness(public, arm, depth, branch, offsets, coordinates):
    coordinates = tuple(exact(value, "A5 witness coordinate") for value in coordinates)
    if len(coordinates) != 12:
        return None
    darkness, ts = coordinates[:9], coordinates[9:]
    image = variation.appearance_engine._image(public["observation"], True)
    raw = _prediction(depth, branch, offsets, darkness, ts)
    # At a clipped endpoint choose the extremal legal pre-clip noise. For an
    # interior observation equality is mandatory; clipping before adding noise
    # would produce a different, unsound completion model.
    noise = tuple(-EPSILON if y == 0 else EPSILON if y == 1 else y - value
                  for value, y in zip(raw, image))
    witness = {"offsets": list(map(encode, offsets)), "darkness": list(map(encode, darkness)),
               "t": list(map(encode, ts)), "noise": list(map(encode, noise))}
    return witness if check_forward_witness(public, arm, depth, branch, witness) else None


def _intersection(*rectangles):
    if any(rect is None for rect in rectangles):
        return None
    return (max(rect[0] for rect in rectangles), min(rect[1] for rect in rectangles),
            max(rect[2] for rect in rectangles), min(rect[3] for rect in rectangles))


def _area(rectangle):
    if rectangle is None:
        return 0.0
    return max(0.0, rectangle[1] - rectangle[0]) * max(0.0, rectangle[3] - rectangle[2])


def _float_areas(branch, depth, offsets, baseline=BASELINE):
    """Original rectangular point integrals in float, for candidates only."""
    depth, baseline = float(depth), float(baseline)
    rear_world = geometry._rear_rect(branch)
    result = []
    for center, error in zip((-baseline, 0.0, baseline), offsets):
        camera = center + float(error)
        def project(rect, z):
            if rect is None:
                return None
            return ((float(rect[0]) - camera) / z, (float(rect[1]) - camera) / z,
                    float(rect[2]) / z, float(rect[3]) / z)
        left, right, front = (project(rect, depth) for rect in
                              (geometry.LEFT_RECT, geometry.RIGHT_RECT, geometry.FRONT_RECT))
        rear = project(rear_world, float(geometry.REAR_DEPTH))
        for exact_pixel, exact_area in geometry.PIXELS:
            pixel, area = tuple(map(float, exact_pixel)), float(exact_area)
            visible = _area(_intersection(rear, pixel)) - _area(_intersection(rear, front, pixel))
            result.append((_area(_intersection(left, pixel)) / area,
                           _area(_intersection(right, pixel)) / area,
                           max(0.0, visible) / area))
    return np.asarray(result, dtype=float)


def _numeric_rows(areas, bands):
    rows, rhs = [], []
    for pixel, (triple, (lower, upper)) in enumerate(zip(areas, bands)):
        view = pixel // geometry.PIXELS_PER_VIEW
        coefficients = np.zeros(12)
        coefficients[3 * view:3 * view + 3] = triple
        coefficients[9 + view] = 1.0
        if upper is not None:
            rows.append(coefficients)
            rhs.append(float(upper))
        if lower is not None:
            rows.append(-coefficients)
            rhs.append(-float(lower))
    for kind, coefficients, bound in variation.difference_rows("joint", DRIFT_BOUND):
        rows.append(np.asarray(coefficients, dtype=float))
        rhs.append(float(bound))
    return np.asarray(rows), np.asarray(rhs)


def _phase_one(areas, bands, bounds, deadline):
    matrix, rhs = _numeric_rows(areas, bands)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, "deadline"
    objective = np.zeros(13)
    objective[-1] = 1.0
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning,
                                    message=r"Unrecognized options detected:.*threads")
            lp = linprog(objective, A_ub=np.column_stack((matrix, -np.ones(len(matrix)))),
                         b_ub=rhs, bounds=[tuple(map(float, pair)) for pair in bounds] + [(0, None)],
                         method="highs", options={"threads": 1, "time_limit": remaining})
        value = float(lp.fun)
        if lp.status != 0 or not lp.success or not math.isfinite(value):
            return None, "lp_not_successful"
        return max(0.0, value), "ok"
    except Exception as error:
        return None, f"lp_exception:{type(error).__name__}"


class _StopSearch(Exception):
    pass


class _StopPowell(Exception):
    pass


def complete_branch(public, arm, depth, branch, deadline=None):
    """Bounded two-coordinate completion; UNKNOWN is not infeasibility.

    deadline is absolute time.monotonic(), as in the frozen A5 classifier.
    Center, grid, Powell and exact recovery share a single three-second window.
    At most 200 total phase-I LP calls (including A5's LP) and 20 exact recovery
    attempts are permitted. Two Powell starts each allow at most 60 calls.
    """
    started = time.monotonic()
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                 or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite absolute monotonic number")
    end = min(started + BRANCH_SECONDS, float(deadline)) if deadline is not None else started + BRANCH_SECONDS
    depth, image, bounds = _inputs(public, arm, depth, branch, check_depth=False)
    result = {"status": "UNKNOWN", "witness": None, "center_status": "UNKNOWN", "center_witness": None,
              "numeric_evaluations": 0, "classifier_lp_calls": 0, "numeric_lp_calls": 0,
              "exact_attempts": 0, "numeric_failures": 0, "powell_starts": 0, "powell_evaluations": [],
              "best_phase_slack": None, "seconds": 0.0, "termination": "no_exact_witness",
              "search_budget_seconds": max(0.0, end - started),
              "limits": {"lp_calls": MAX_LP_CALLS, "exact_attempts": MAX_EXACT_ATTEMPTS,
                         "branch_seconds": BRANCH_SECONDS, "powell_starts": MAX_POWELL_STARTS,
                         "powell_maxfev": POWELL_MAXFEV, "offset_denominator": OFFSET_DENOMINATOR_LIMIT}}

    def finish():
        result["seconds"] = max(0.0, time.monotonic() - started)
        return result

    if not geometry.DEPTH_MIN <= depth <= geometry.DEPTH_MAX:
        result["termination"] = "outside_domain"
        return finish()
    bands = variation.frozen.pixel_bands(image, clipped=True, beta=BETA, epsilon=EPSILON)
    seen, candidates = set(), []

    def stop_if_limited():
        if time.monotonic() >= end:
            result["termination"] = "whole_deadline" if deadline is not None and deadline <= started + BRANCH_SECONDS else "branch_timeout"
            raise _StopSearch()
        if result["numeric_lp_calls"] >= MAX_LP_CALLS:
            result["termination"] = "lp_budget"
            raise _StopSearch()

    def recover(offsets, *, center=False):
        stop_if_limited()
        if offsets in seen or result["exact_attempts"] >= MAX_EXACT_ATTEMPTS:
            return
        seen.add(offsets)
        result["exact_attempts"] += 1
        rows, exact_bounds = exact_rows(public, arm, depth, branch, offsets)
        stop_if_limited()
        classification = point.classify(rows, exact_bounds, deadline=end)
        # A failed/exceptional HiGHS call can have a null status but still
        # consumed a numerical call. The frozen report records its elapsed LP
        # time even on that path; exact single-row/trivial paths leave it zero.
        if classification["lp_status"] is not None or classification["lp_seconds"] > 0:
            result["classifier_lp_calls"] += 1
            result["numeric_lp_calls"] += 1
        # Even an exact fixed-pose contradiction is not a common-pose exclusion.
        witness = None
        if classification["status"] == "FEASIBLE_WITNESS" and time.monotonic() < end:
            witness = _reconstruct_witness(public, arm, depth, branch, offsets, classification["witness"])
        if center:
            result["center_status"] = "FORWARD_WITNESS" if witness is not None else "UNKNOWN"
            result["center_witness"] = witness
        if witness is not None and time.monotonic() >= end:
            witness = None
            if center:
                result.update(center_status="UNKNOWN", center_witness=None)
        if witness is not None:
            result.update(status="FORWARD_WITNESS", witness=witness, termination="first_exact_witness")
            raise _StopSearch()
        stop_if_limited()

    def objective(coordinates, *, center=False):
        stop_if_limited()
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
            result["numeric_failures"] += 1
            return 1e6
        coordinates = np.clip(coordinates, -float(RADIUS), float(RADIUS))
        offsets = (Q(float(coordinates[0])).limit_denominator(OFFSET_DENOMINATOR_LIMIT), Q(0),
                   Q(float(coordinates[1])).limit_denominator(OFFSET_DENOMINATOR_LIMIT))
        result["numeric_evaluations"] += 1
        result["numeric_lp_calls"] += 1
        value, reason = _phase_one(_float_areas(branch, depth, (coordinates[0], 0, coordinates[1])),
                                   bands, bounds, end)
        if value is None:
            result["numeric_failures"] += 1
            value = 1e6
        else:
            candidates.append((value, tuple(coordinates)))
            result["best_phase_slack"] = value if result["best_phase_slack"] is None else min(result["best_phase_slack"], value)
        if center or (value <= CANDIDATE_SLACK and result["exact_attempts"] < MAX_EXACT_ATTEMPTS):
            recover(offsets, center=center)
        return value

    try:
        objective((0.0, 0.0), center=True)
        radius = float(RADIUS)
        for left in (-radius, 0.0, radius):
            for right in (-radius, 0.0, radius):
                if left != 0.0 or right != 0.0:
                    objective((left, right))
        starts, used = [], set()
        for _score, coordinates in sorted(candidates, key=lambda entry: (entry[0], entry[1])):
            if coordinates not in used:
                starts.append(coordinates)
                used.add(coordinates)
            if len(starts) == MAX_POWELL_STARTS:
                break
        for coordinates in starts:
            stop_if_limited()
            result["powell_starts"] += 1
            before = result["numeric_evaluations"]
            def powell_objective(coordinates):
                if result["numeric_evaluations"] - before >= POWELL_MAXFEV:
                    raise _StopPowell()
                return objective(coordinates)
            try:
                minimize(powell_objective, coordinates, method="Powell", bounds=((-radius, radius),) * 2,
                         options={"maxfev": POWELL_MAXFEV, "xtol": 1e-6, "ftol": 1e-9})
            except _StopSearch:
                raise
            except _StopPowell:
                pass
            except Exception:
                result["numeric_failures"] += 1
            finally:
                result["powell_evaluations"].append(result["numeric_evaluations"] - before)
        stop_if_limited()
        result["termination"] = "search_exhausted" if starts else "numeric_failure"
    except _StopSearch:
        pass
    return finish()
