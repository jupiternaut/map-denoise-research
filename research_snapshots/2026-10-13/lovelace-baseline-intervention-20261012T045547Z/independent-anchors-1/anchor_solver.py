"""A8 public independent anchors, exact outer rows and bounded completion.

Only public inputs are accepted. A necessary varying-pose contradiction can
exclude a branch; a fixed-pose contradiction cannot. Numerical optimization
only proposes rational poses, and positives require clip-once forward checks.
"""
from fractions import Fraction as Q
from functools import lru_cache
import math
from pathlib import Path
import sys
import time
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
POINT_ROOT = PARENT / "point-feasibility-1"
POSE_ROOT = PARENT / "common-pose-feasibility-1"
for _source_root in (POINT_ROOT, POSE_ROOT):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))
import point_solver as point
import pose_solver as pose
import pose_lift as lift

for _module, _path in ((point, POINT_ROOT / "point_solver.py"),
                        (point.polytope, PARENT / "joint-feasibility-1" / "polytope_filter.py"),
                        (pose, POSE_ROOT / "pose_solver.py"),
                        (lift, POSE_ROOT / "pose_lift.py")):
    if Path(_module.__file__).resolve() != _path.resolve():
        raise RuntimeError(f"wrong frozen source identity: {_module.__file__}")

import numpy as np
from scipy.optimize import OptimizeWarning, linprog, minimize

exact, encode = point.exact, point.encode
geometry, variation = point.geometry, point.variation
classify, certificate_check, witness_check = point.classify, point.certificate_check, point.witness_check
ARM_NAMES = ("none", "camera", "appearance", "both")
BRANCHES = point.BRANCHES
RADIUS, BASELINE = Q(1, 5), Q(80)
BETA, EPSILON, DRIFT_BOUND = Q(1, 100), Q(1, 100), Q(1, 10)
SURVEY_BOUND = LOCALIZATION_BOUND = ANCHOR_BIAS_BOUND = Q(1, 100)
TRANSFER_BOUND = ANCHOR_NOISE_BOUND = Q(1, 200)
ANCHOR_BOUNDS = ((Q(0), Q(1, 50)),) * 3
MAX_LP_CALLS, MAX_EXACT_ATTEMPTS = 200, 20
BRANCH_SECONDS, MAX_POWELL_STARTS, POWELL_MAXFEV = 3.0, 2, 60
OFFSET_DENOMINATOR_LIMIT, CANDIDATE_SLACK = 1_000_000_000, 1e-7
WITNESS_FIELDS = pose.WITNESS_FIELDS | {"tA", "camera_k", "camera_noise",
                                      "coupon_transfer", "coupon_noise", "reference_noise"}
STATUSES = ("FORWARD_WITNESS", "UNKNOWN")


def _enabled(name):
    if name not in ARM_NAMES:
        raise ValueError(f"arm_name must be one of {ARM_NAMES}")
    return name in ("camera", "both"), name in ("appearance", "both")


def _sequence(values, count, name):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} requires {count} exact scalars")
    return tuple(exact(value, name) for value in values)


def _anchor_inputs(public, anchors, name):
    camera, appearance = _enabled(name)
    # Disabled sections are not decoded: their information has no effect.
    if camera or appearance:
        if not isinstance(anchors, dict) or anchors.get("observation_id") != public.get("observation_id"):
            raise ValueError("anchor/public observation identity mismatch")
    if camera:
        record = anchors["camera"]
        for key, value in (("Xf", Q(0)), ("Zf", Q(1000)),
                           ("survey_bound", SURVEY_BOUND),
                           ("localization_bound", LOCALIZATION_BOUND)):
            if exact(record[key], key) != value:
                raise ValueError(f"camera {key} differs from the A8 contract")
        _sequence(record["u"], 3, "camera u")
    if appearance:
        record = anchors["appearance"]
        for key, value in (("bias_bound", ANCHOR_BIAS_BOUND),
                           ("transfer_bound", TRANSFER_BOUND),
                           ("noise_bound", ANCHOR_NOISE_BOUND)):
            if exact(record[key], key) != value:
                raise ValueError(f"appearance {key} differs from the A8 contract")
        coupons = _sequence(record["coupons"], 9, "coupon reading")
        references = _sequence(record["references"], 3, "reference reading")
        if any(not 0 <= y <= 1 for y in coupons + references):
            raise ValueError("appearance readings must lie in [0,1]")
    return camera, appearance


def _inputs(public, oldarm, anchors, name, depth, branch, *, check_depth=True):
    depth, image, bounds = pose._inputs(public, oldarm, depth, branch, check_depth=check_depth)
    enabled = _anchor_inputs(public, anchors, name)
    return depth, image, bounds + ANCHOR_BOUNDS, enabled


def _intersect(*intervals):
    if any(interval is None for interval in intervals):
        return None
    lo, hi = max(interval[0] for interval in intervals), min(interval[1] for interval in intervals)
    return (lo, hi) if lo <= hi else None


def _camera_residuals(anchors):
    readings = _sequence(anchors["camera"]["u"], 3, "camera u")
    return tuple(-center - Q(1000) * u for center, u in zip((-BASELINE, Q(0), BASELINE), readings))


def camera_intervals(anchors):
    """Return exact survey J and outer offset I, or (None, None) if empty.

    The reference gauge gives J; marginalizing the common k gives the other
    offset intervals. Dropping their common-k correlation is a sound outer.
    """
    residuals = _camera_residuals(anchors)
    survey = _intersect((-SURVEY_BOUND, SURVEY_BOUND),
                        (residuals[1] - LOCALIZATION_BOUND, residuals[1] + LOCALIZATION_BOUND))
    if survey is None:
        return None, None
    intervals = tuple((Q(0), Q(0)) if view == 1 else _intersect(
        (-RADIUS, RADIUS), (residual - survey[1] - LOCALIZATION_BOUND,
                            residual - survey[0] + LOCALIZATION_BOUND))
                      for view, residual in enumerate(residuals))
    return (survey, intervals) if all(interval is not None for interval in intervals) else (None, None)


def camera_k_interval(anchors, offsets):
    """Exact common survey-error feasibility, retaining cross-view correlation."""
    offsets = pose._offsets(offsets)
    survey, _ = camera_intervals(anchors)
    if survey is None:
        return None
    return _intersect(survey, *((residual - offset - LOCALIZATION_BOUND,
                                 residual - offset + LOCALIZATION_BOUND)
                                for residual, offset in zip(_camera_residuals(anchors), offsets)))


@lru_cache(maxsize=4096)
def area_bounds(branch, depth, intervals):
    """Exact extrema of point areas on all affine cells clipped to each I_v."""
    depth = exact(depth, "depth")
    if len(intervals) != 3:
        raise ValueError("three camera intervals required")
    result = []
    for view, interval in enumerate(intervals):
        lo, hi = (exact(value, "camera interval") for value in interval)
        if not -RADIUS <= lo <= hi <= RADIUS or (view == 1 and (lo, hi) != (Q(0), Q(0))):
            raise ValueError("camera intervals violate the original box/gauge")
        _, cells = lift.camera_cells(branch, depth, view, RADIUS)
        endpoints = sorted({point for cell in cells for point in (_intersect(cell, (lo, hi)) or ())})
        samples = [lift._view_areas(branch, depth, view, offset) for offset in endpoints]
        if not samples:
            raise ArithmeticError("closed camera interval was not covered by affine cells")
        for pixel in range(geometry.PIXELS_PER_VIEW):
            result.append(tuple(value for material in range(3)
                                for value in (min(sample[pixel][material] for sample in samples),
                                              max(sample[pixel][material] for sample in samples))))
    return tuple(result)


def _band(y, base, epsilon):
    """Exact clip-once band for x when y=clip(base+beta-x+error)."""
    y = exact(y, "clipped reading")
    if not 0 <= y <= 1:
        raise ValueError("clipped reading outside [0,1]")
    return (None if y == 0 else base + BETA - y + epsilon,
            None if y == 1 else base + BETA - y - epsilon)


def appearance_rows(anchors):
    record = anchors["appearance"]
    coupons = _sequence(record["coupons"], 9, "coupon reading")
    references = _sequence(record["references"], 3, "reference reading")
    rows = []
    for index, y in enumerate(coupons):
        coefficients = [Q(0)] * 15
        coefficients[index] = coefficients[12 + index // 3] = Q(1)
        high, low = _band(y, Q(1), TRANSFER_BOUND + ANCHOR_NOISE_BOUND)
        if high is not None:
            rows.append(("upper", tuple(coefficients), high))
        if low is not None:
            rows.append(("lower", tuple(coefficients), low))
    for view, y in enumerate(references):
        coefficients = [Q(0)] * 15
        coefficients[12 + view] = Q(1)
        high, low = _band(y, Q(1, 2), ANCHOR_NOISE_BOUND)
        if high is not None:
            rows.append(("upper", tuple(coefficients), high))
        if low is not None:
            rows.append(("lower", tuple(coefficients), low))
    return tuple(rows)


def build_problem(public, oldarm, anchors, arm_name, depth, branch, offsets=None):
    """Necessary all-pose rows, or sufficient photo rows at the given pose.

    The d9/t3 global box remains frozen; independent tA3 is always appended.
    Only offsets=None rows may support a depth/branch exclusion certificate.
    """
    depth, image, bounds, (camera, appearance) = _inputs(public, oldarm, anchors, arm_name, depth, branch)
    impossible = False
    if offsets is None:
        intervals = ((-RADIUS, RADIUS), (Q(0), Q(0)), (-RADIUS, RADIUS))
        if camera:
            _, intervals = camera_intervals(anchors)
            impossible = intervals is None
        if impossible:
            areas = None
        else:
            areas = area_bounds(branch, depth, intervals)
    else:
        offsets = pose._offsets(offsets)
        impossible = camera and camera_k_interval(anchors, offsets) is None
        point_areas = geometry._point_areas(branch, depth, offsets, BASELINE)
        areas = tuple(tuple(value for area in triple for value in (area, area)) for triple in point_areas)
    if areas is None:
        rows = ()
    else:
        rows12 = variation.necessary_rows(image, areas, "joint", clipped=True,
                                         drift_bound=DRIFT_BOUND, beta=BETA, epsilon=EPSILON)
        rows = tuple((kind, coefficients + (Q(0),) * 3, bound) for kind, coefficients, bound in rows12)
    if impossible:
        rows += (("upper", (Q(0),) * 15, Q(-1)),)
    if appearance:
        rows += appearance_rows(anchors)
    return rows, bounds


def check_forward_witness(public, oldarm, anchors, arm_name, depth, branch, witness):
    """Check complete exact shapes and all original/new clip-once measurements."""
    try:
        _depth, _image, _bounds, (camera, appearance) = _inputs(
            public, oldarm, anchors, arm_name, depth, branch)
        if not isinstance(witness, dict) or set(witness) != WITNESS_FIELDS:
            return False
        old_witness = {key: witness[key] for key in pose.WITNESS_FIELDS}
        # Frozen checker verifies the entire 432-channel original photo once.
        if not pose.check_forward_witness(public, oldarm, depth, branch, old_witness):
            return False
        ta = _sequence(witness["tA"], 3, "anchor tA")
        if any(not lo <= value <= hi for value, (lo, hi) in zip(ta, ANCHOR_BOUNDS)):
            return False
        if camera:
            k = exact(witness["camera_k"], "shared camera survey error")
            noise = _sequence(witness["camera_noise"], 3, "camera localization noise")
            offsets = pose._offsets(witness["offsets"])
            if abs(k) > SURVEY_BOUND or any(abs(value) > LOCALIZATION_BOUND for value in noise):
                return False
            if any(r != e + k + n for r, e, n in zip(_camera_residuals(anchors), offsets, noise)):
                return False
        elif witness["camera_k"] is not None or witness["camera_noise"] is not None:
            return False
        if appearance:
            transfers = _sequence(witness["coupon_transfer"], 9, "coupon transfer")
            noise = _sequence(witness["coupon_noise"], 9, "coupon noise")
            ref_noise = _sequence(witness["reference_noise"], 3, "reference noise")
            if (any(abs(value) > TRANSFER_BOUND for value in transfers)
                    or any(abs(value) > ANCHOR_NOISE_BOUND for value in noise + ref_noise)):
                return False
            darkness = _sequence(witness["darkness"], 9, "darkness")
            coupons = _sequence(anchors["appearance"]["coupons"], 9, "coupon reading")
            references = _sequence(anchors["appearance"]["references"], 3, "reference reading")
            if any(_clip(Q(1) - d + BETA - ta[index // 3] + q + n) != y
                   for index, (d, q, n, y) in enumerate(zip(darkness, transfers, noise, coupons))):
                return False
            if any(_clip(Q(1, 2) + BETA - t + n) != y for t, n, y in zip(ta, ref_noise, references)):
                return False
        elif any(witness[key] is not None for key in ("coupon_transfer", "coupon_noise", "reference_noise")):
            return False
        return True
    except (TypeError, ValueError, KeyError, ZeroDivisionError, OverflowError, ArithmeticError):
        return False


def _clip(value):
    return min(Q(1), max(Q(0), value))


def _noise_for(y, raw, epsilon):
    return -epsilon if y == 0 else epsilon if y == 1 else y - raw


def _ta_intervals(anchors, darkness):
    """Anchor-only interval intersection with fixed d9 and one shared tA/view."""
    coupons = _sequence(anchors["appearance"]["coupons"], 9, "coupon reading")
    refs = _sequence(anchors["appearance"]["references"], 3, "reference reading")
    result = []
    for view in range(3):
        interval = ANCHOR_BOUNDS[view]
        for index in range(3 * view, 3 * view + 3):
            high, low = _band(coupons[index], Q(1), TRANSFER_BOUND + ANCHOR_NOISE_BOUND)
            interval = _intersect(interval, (Q(0) if low is None else low - darkness[index],
                                               Q(1, 50) if high is None else high - darkness[index]))
        high, low = _band(refs[view], Q(1, 2), ANCHOR_NOISE_BOUND)
        interval = _intersect(interval, (Q(0) if low is None else low,
                                           Q(1, 50) if high is None else high))
        if interval is None:
            return None
        result.append(interval)
    return tuple(result)


def _extend_witness(public, oldarm, anchors, name, depth, branch, old_witness, ta=None):
    camera, appearance = _anchor_inputs(public, anchors, name)
    if not pose.check_forward_witness(public, oldarm, depth, branch, old_witness):
        return None
    darkness = _sequence(old_witness["darkness"], 9, "darkness")
    if ta is None:
        intervals = _ta_intervals(anchors, darkness) if appearance else ANCHOR_BOUNDS
        if intervals is None:
            return None
        ta = tuple((lo + hi) / 2 for lo, hi in intervals)
    ta = _sequence(ta, 3, "anchor tA")
    result = {key: list(old_witness[key]) for key in pose.WITNESS_FIELDS}
    result.update(tA=list(map(encode, ta)), camera_k=None, camera_noise=None,
                  coupon_transfer=None, coupon_noise=None, reference_noise=None)
    if camera:
        offsets = pose._offsets(old_witness["offsets"])
        interval = camera_k_interval(anchors, offsets)
        if interval is None:
            return None
        k = (interval[0] + interval[1]) / 2
        noise = tuple(r - e - k for r, e in zip(_camera_residuals(anchors), offsets))
        result.update(camera_k=encode(k), camera_noise=list(map(encode, noise)))
    if appearance:
        coupons = _sequence(anchors["appearance"]["coupons"], 9, "coupon reading")
        references = _sequence(anchors["appearance"]["references"], 3, "reference reading")
        residuals = tuple(_noise_for(y, Q(1) - d + BETA - ta[index // 3],
                                    TRANSFER_BOUND + ANCHOR_NOISE_BOUND)
                          for index, (y, d) in enumerate(zip(coupons, darkness)))
        transfers = tuple(min(TRANSFER_BOUND, max(-TRANSFER_BOUND, value)) for value in residuals)
        result.update(coupon_transfer=list(map(encode, transfers)),
                      coupon_noise=[encode(value - q) for value, q in zip(residuals, transfers)],
                      reference_noise=[encode(_noise_for(y, Q(1, 2) + BETA - t,
                                                         ANCHOR_NOISE_BOUND))
                                       for y, t in zip(references, ta)])
    return result if check_forward_witness(public, oldarm, anchors, name, depth, branch, result) else None


def revalidate_old_witness(public, oldarm, anchors, arm_name, depth, branch, oldA7w):
    """Extend an unchanged public old witness only when new anchors admit it."""
    try:
        _inputs(public, oldarm, anchors, arm_name, depth, branch)
        return _extend_witness(public, oldarm, anchors, arm_name, depth, branch, oldA7w)
    except (TypeError, ValueError, KeyError, ZeroDivisionError, OverflowError, ArithmeticError):
        return None


def _reconstruct_witness(public, oldarm, anchors, name, depth, branch, offsets, coordinates):
    coordinates = _sequence(coordinates, 15, "15-coordinate recovery")
    darkness, ts = coordinates[:9], coordinates[9:12]
    image = variation.appearance_engine._image(public["observation"], True)
    raw = pose._prediction(depth, branch, offsets, darkness, ts)
    old = {"offsets": list(map(encode, offsets)), "darkness": list(map(encode, darkness)),
           "t": list(map(encode, ts)),
           "noise": [encode(_noise_for(y, value, EPSILON)) for y, value in zip(image, raw)]}
    return _extend_witness(public, oldarm, anchors, name, depth, branch, old, coordinates[12:])


def _numeric_rows(areas, bands, anchor_rows=()):
    photo_matrix, photo_rhs = pose._numeric_rows(areas, bands)
    matrix = np.column_stack((photo_matrix, np.zeros((len(photo_matrix), 3))))
    rhs = photo_rhs
    if anchor_rows:
        extra = np.asarray([tuple(float(value) * (1 if kind == "upper" else -1)
                                  for value in coefficients) for kind, coefficients, bound in anchor_rows])
        extra_rhs = np.asarray([float(bound) * (1 if kind == "upper" else -1)
                                for kind, coefficients, bound in anchor_rows])
        matrix, rhs = np.vstack((matrix, extra)), np.concatenate((rhs, extra_rhs))
    return matrix, rhs


def _phase_one(areas, bands, bounds, anchor_rows, deadline):
    matrix, rhs = _numeric_rows(areas, bands, anchor_rows)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, "deadline"
    objective = np.zeros(16)
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


def complete_branch(public, oldarm, anchors, arm_name, depth, branch, deadline=None):
    """Bounded common-pose refit; a failed fit always remains UNKNOWN."""
    started = time.monotonic()
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                 or not math.isfinite(deadline)):
        raise ValueError("deadline must be finite absolute time.monotonic()")
    end = min(started + BRANCH_SECONDS, float(deadline)) if deadline is not None else started + BRANCH_SECONDS
    depth, image, bounds, (camera, appearance) = _inputs(
        public, oldarm, anchors, arm_name, depth, branch, check_depth=False)
    result = {"status": "UNKNOWN", "witness": None, "center_status": "UNKNOWN", "center_witness": None,
              "numeric_evaluations": 0, "classifier_lp_calls": 0, "numeric_lp_calls": 0,
              "exact_attempts": 0, "numeric_failures": 0, "skipped_camera_candidates": 0, "powell_starts": 0,
              "powell_evaluations": [], "best_phase_slack": None, "seconds": 0.0,
              "termination": "no_exact_witness", "search_budget_seconds": max(0.0, end - started),
              "limits": {"lp_calls": MAX_LP_CALLS, "exact_attempts": MAX_EXACT_ATTEMPTS,
                         "branch_seconds": BRANCH_SECONDS, "powell_starts": MAX_POWELL_STARTS,
                         "powell_maxfev": POWELL_MAXFEV, "offset_denominator": OFFSET_DENOMINATOR_LIMIT}}

    def finish():
        result["seconds"] = max(0.0, time.monotonic() - started)
        return result

    if not geometry.DEPTH_MIN <= depth <= geometry.DEPTH_MAX:
        result["termination"] = "outside_domain"
        return finish()
    intervals = ((-RADIUS, RADIUS), (Q(0), Q(0)), (-RADIUS, RADIUS))
    if camera:
        _, intervals = camera_intervals(anchors)
        if intervals is None:
            result["termination"] = "empty_camera_outer"
            return finish()
    coordinates_box = (intervals[0], intervals[2])
    bands = variation.frozen.pixel_bands(image, clipped=True, beta=BETA, epsilon=EPSILON)
    extra = appearance_rows(anchors) if appearance else ()
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
        rows, box = build_problem(public, oldarm, anchors, arm_name, depth, branch, offsets)
        stop_if_limited()
        classification = classify(rows, box, deadline=end)
        if classification["lp_status"] is not None or classification["lp_seconds"] > 0:
            result["classifier_lp_calls"] += 1
            result["numeric_lp_calls"] += 1
        witness = None
        if classification["status"] == "FEASIBLE_WITNESS" and time.monotonic() < end:
            witness = _reconstruct_witness(public, oldarm, anchors, arm_name, depth, branch,
                                           offsets, classification["witness"])
        if witness is not None and time.monotonic() >= end:
            witness = None
        if center:
            result.update(center_status="FORWARD_WITNESS" if witness is not None else "UNKNOWN",
                          center_witness=witness)
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
        offsets = tuple(_intersect_coordinate(value, box) for value, box in zip(coordinates, coordinates_box))
        offsets = (offsets[0], Q(0), offsets[1])
        result["numeric_evaluations"] += 1
        if camera and camera_k_interval(anchors, offsets) is None:
            # Reject the common-k inconsistency before proposing an LP.
            result["numeric_failures"] += 1
            result["skipped_camera_candidates"] += 1
            return 1e6
        result["numeric_lp_calls"] += 1
        value, _reason = _phase_one(pose._float_areas(branch, depth, offsets), bands, bounds, extra, end)
        if value is None:
            result["numeric_failures"] += 1
            value = 1e6
        else:
            candidate = (float(offsets[0]), float(offsets[2]))
            candidates.append((value, candidate))
            result["best_phase_slack"] = value if result["best_phase_slack"] is None else min(result["best_phase_slack"], value)
        if center or value <= CANDIDATE_SLACK:
            recover(offsets, center=center)
        return value

    try:
        midpoint = tuple(float((lo + hi) / 2) for lo, hi in coordinates_box)
        objective(midpoint, center=True)
        grid = tuple(tuple(float(x) for x in (lo, (lo + hi) / 2, hi)) for lo, hi in coordinates_box)
        for left in grid[0]:
            for right in grid[1]:
                if (left, right) != midpoint:
                    objective((left, right))
        starts = []
        for _score, coordinates in sorted(candidates, key=lambda entry: (entry[0], entry[1])):
            if coordinates not in starts:
                starts.append(coordinates)
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
                minimize(powell_objective, coordinates, method="Powell",
                         bounds=tuple(tuple(map(float, box)) for box in coordinates_box),
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


def _intersect_coordinate(value, interval):
    candidate = Q(float(value)).limit_denominator(OFFSET_DENOMINATOR_LIMIT)
    return min(interval[1], max(interval[0], candidate))


def clear_geometry_cache():
    area_bounds.cache_clear()
    lift.clear_geometry_cache()
