"""Independent exact checker for the frozen A13 physical-model relaxation.

Standard library only. No solver, certificate builder, estimator, or truth input.
The source packet contains binary64 roundings of ideal rational observations;
closed rounding-preimage intervals conservatively recover those observations.
"""
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import json
import math
import re


SCHEMA = "a13-rational-box-farkas-v1"
BRANCHES = ("none", "partial", "full")
VARIABLE_ORDER = [f"d{v}_{k}" for v in range(3) for k in range(3)] + [
    "b0", "b1", "b2", "ba0", "ba1", "ba2"]
ZERO, ONE = Q(0), Q(1)
FRONT = (-20, 20, -15, 15)
MATERIALS = ((-20, 0, -15, 15), (0, 20, -15, 15))
REARS = {"none": None, "partial": (-20, 36, -17, 23),
         "full": (-100, 100, -80, 80)}
PIXELS = tuple((Q(-1, 5) + Q(i, 20), Q(-1, 5) + Q(i + 1, 20),
                Q(-2, 25) + Q(2 * j, 75), Q(-2, 25) + Q(2 * (j + 1), 75))
               for j in range(6) for i in range(8))
PIXEL_AREA = Q(1, 750)


class VerificationError(ValueError):
    pass


def fraction_text(value):
    """Canonical exact rational used in serialized proof records."""
    value = Q(value)
    return f"{value.numerator}/{value.denominator}"


def _fraction(value):
    if not isinstance(value, str) or not re.fullmatch(r"-?[0-9]+/[1-9][0-9]*", value):
        raise VerificationError("proof rationals must be canonical n/d strings")
    result = Q(value)
    if fraction_text(result) != value:
        raise VerificationError("noncanonical proof rational")
    return result


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise VerificationError("expected finite JSON number")
    if not math.isfinite(value):
        raise VerificationError("nonfinite JSON number")
    return Q(value)


def _endpoint(value):
    return _fraction(value) if isinstance(value, str) else _number(value)


def _rounding_interval(value, clipped=False):
    """Closed enclosure of all exact reals rounding to this finite binary64."""
    _number(value)
    x = float(value)
    if clipped and not 0 <= x <= 1:
        raise VerificationError("clipped observation outside [0,1]")
    before, after = math.nextafter(x, -math.inf), math.nextafter(x, math.inf)
    if not math.isfinite(before) or not math.isfinite(after):
        raise VerificationError("unbounded binary64 rounding preimage")
    low = (Q(before) + Q(x)) / 2
    high = (Q(x) + Q(after)) / 2
    return (max(ZERO, low), min(ONE, high)) if clipped else (low, high)


def _canonical_observation(observation):
    if not isinstance(observation, dict):
        raise VerificationError("observation must be an object")
    selected = {key: observation[key] for key in ("photo", "camera", "appearance")}
    return json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False)


def observation_sha256(observation):
    return hashlib.sha256(_canonical_observation(observation).encode()).hexdigest()


def _require_metadata(obj, expected):
    if not isinstance(obj, dict):
        raise VerificationError("observation metadata must be an object")
    for key, value in expected.items():
        if key not in obj or isinstance(obj[key], bool) or obj[key] != value:
            raise VerificationError(f"unsupported frozen metadata {key}")


def _vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise VerificationError(f"{label} must have length {length}")
    return value


@lru_cache(maxsize=32)
def _decode_observation(canonical):
    o = json.loads(canonical)
    camera, appearance = o["camera"], o["appearance"]
    _require_metadata(camera, {"Xf": 0, "Zf": 1000, "survey_bound": .01,
                               "localization_bound": .01})
    _require_metadata(appearance, {"bias_bound": .01, "transfer_bound": .005,
                                   "noise_bound": .005})
    u = [_rounding_interval(x) for x in _vector(camera["u"], 3, "camera.u")]
    # The middle camera is exact. Eliminate the shared survey error k first.
    k_low = max(Q(-1, 100), -1000 * u[1][1] - Q(1, 100))
    k_high = min(Q(1, 100), -1000 * u[1][0] + Q(1, 100))
    if k_low > k_high:
        raise VerificationError("inconsistent camera observation; no supported certificate")
    cameras = []
    for v, nominal in enumerate((-80, 0, 80)):
        if v == 1:
            cameras.append((ZERO, ZERO))
            continue
        low = max(Q(nominal) - Q(1, 5), -1000 * u[v][1] - k_high - Q(1, 100))
        high = min(Q(nominal) + Q(1, 5), -1000 * u[v][0] - k_low + Q(1, 100))
        if low > high:
            raise VerificationError("inconsistent camera interval; no supported certificate")
        cameras.append((low, high))
    photo = [_rounding_interval(x, True) for x in _vector(o["photo"], 432, "photo")]
    photo_bounds = []
    for v in range(3):
        view = []
        for p in range(48):
            channels = photo[144 * v + 3 * p:144 * v + 3 * p + 3]
            lower = [a - Q(1, 100) for a, b in channels if a > 0]
            upper = [b + Q(1, 100) for a, b in channels if b < 1]
            view.append((max(lower) if lower else None, min(upper) if upper else None))
        photo_bounds.append(tuple(view))
    coupons = _vector(appearance["coupons"], 9, "appearance.coupons")
    refs = _vector(appearance["references"], 3, "appearance.references")
    boxes = [(ZERO, ONE)] * 9 + [(Q(-1, 100), Q(1, 100))] * 3
    rows = []
    for v in range(3):
        for k, value in enumerate(coupons[3 * v:3 * v + 3]):
            low, high = _rounding_interval(value, True)
            if low > 0:
                rows.append((((3 * v + k, ONE), (12 + v, -ONE)), ONE - low + Q(1, 100)))
            if high < 1:
                rows.append((((3 * v + k, -ONE), (12 + v, ONE)), high - ONE + Q(1, 100)))
        low, high = _rounding_interval(refs[v], True)
        b_low = max(Q(-1, 100), low - Q(1, 2) - Q(1, 200)) if low > 0 else Q(-1, 100)
        b_high = min(Q(1, 100), high - Q(1, 2) + Q(1, 200)) if high < 1 else Q(1, 100)
        if b_low > b_high:
            raise VerificationError("inconsistent appearance reference; no supported certificate")
        boxes.append((b_low, b_high))
    for k in range(3):
        for v in range(3):
            for w in range(v):
                rows.append((((3 * v + k, ONE), (3 * w + k, -ONE)), Q(1, 10)))
                rows.append((((3 * v + k, -ONE), (3 * w + k, ONE)), Q(1, 10)))
    return tuple(cameras), tuple(photo_bounds), tuple(boxes), tuple(rows)


def _rect_enclosure(rect, depth_low, depth_high, camera):
    if rect is None:
        return None, None
    c_low, c_high = camera
    ranges = []
    for coordinate in rect[:2]:
        values = [(Q(coordinate) - c) / z for c in (c_low, c_high)
                  for z in (depth_low, depth_high)]
        ranges.append((min(values), max(values)))
    for coordinate in rect[2:]:
        values = [Q(coordinate) / z for z in (depth_low, depth_high)]
        ranges.append((min(values), max(values)))
    inner = (ranges[0][1], ranges[1][0], ranges[2][1], ranges[3][0])
    outer = (ranges[0][0], ranges[1][1], ranges[2][0], ranges[3][1])
    return inner, outer


def _intersection(a, b):
    if a is None or b is None:
        return None
    return max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])


def _area(rect, pixel):
    if rect is None:
        return ZERO
    width = max(ZERO, min(rect[1], pixel[1]) - max(rect[0], pixel[0]))
    height = max(ZERO, min(rect[3], pixel[3]) - max(rect[2], pixel[2]))
    return width * height / PIXEL_AREA


def _view_geometry(branch, low, high, camera):
    front_inner, front_outer = _rect_enclosure(FRONT, low, high, camera)
    left = _rect_enclosure(MATERIALS[0], low, high, camera)
    right = _rect_enclosure(MATERIALS[1], low, high, camera)
    rear_inner, rear_outer = _rect_enclosure(REARS[branch], Q(720), Q(720), camera)
    rear_lower_cover = _intersection(rear_inner, front_outer)
    rear_upper_cover = _intersection(rear_outer, front_inner)
    return left, right, rear_inner, rear_outer, rear_lower_cover, rear_upper_cover


def _pixel_geometry(geometry, pixel):
    left, right, r_inner, r_outer, low_cover, high_cover = geometry
    lower = (_area(left[0], pixel), _area(right[0], pixel),
             _area(r_inner, pixel) - _area(low_cover, pixel))
    upper = (_area(left[1], pixel), _area(right[1], pixel),
             _area(r_outer, pixel) - _area(high_cover, pixel))
    if any(not 0 <= a <= b <= 1 for a, b in zip(lower, upper)):
        raise VerificationError("invalid independently reconstructed area bounds")
    return lower, upper


def reconstruct_constraints(observation, branch, low, high):
    """Return exact sparse A,b rows and box, from physical inputs alone.

    Useful for independent audits. Verification itself materializes only rows
    touched by the sparse proof, while computing the same complete row layout.
    """
    canonical = _canonical_observation(observation)
    low = low if isinstance(low, Q) else _endpoint(low)
    high = high if isinstance(high, Q) else _endpoint(high)
    if branch not in BRANCHES or not 540 <= low <= high <= 660:
        raise VerificationError("invalid geometric cell")
    cameras, photo, boxes, base = _decode_observation(canonical)
    rows = list(base)
    for v in range(3):
        geometry = _view_geometry(branch, low, high, cameras[v])
        areas = [_pixel_geometry(geometry, pixel) for pixel in PIXELS]
        for p in range(48):
            if photo[v][p][0] is not None:
                row = tuple((3 * v + k, value) for k, value in enumerate(areas[p][0]) if value)
                rows.append((row + ((9 + v, -ONE),), ONE - photo[v][p][0]))
        for p in range(48):
            if photo[v][p][1] is not None:
                row = tuple((3 * v + k, -value) for k, value in enumerate(areas[p][1]) if value)
                rows.append((row + ((9 + v, ONE),), photo[v][p][1] - ONE))
    return rows, boxes


def _selected_rows(canonical, branch, low, high, support):
    cameras, photo, boxes, base = _decode_observation(canonical)
    selected = {i: base[i] for i in support if i < len(base)}
    count = len(base)
    for v in range(3):
        geometry = None
        cached_areas = {}
        for side in range(2):
            for p in range(48):
                bound = photo[v][p][side]
                if bound is None:
                    continue
                if count in support:
                    if geometry is None:
                        geometry = _view_geometry(branch, low, high, cameras[v])
                    if p not in cached_areas:
                        cached_areas[p] = _pixel_geometry(geometry, PIXELS[p])
                    sign = ONE if side == 0 else -ONE
                    row = tuple((3 * v + k, sign * value)
                                for k, value in enumerate(cached_areas[p][side]) if value)
                    selected[count] = (row + ((9 + v, -sign),), sign * (ONE - bound))
                count += 1
    return selected, boxes, count


def verify_certificate(observation, record):
    """Accept only a strictly positive exact box-dual contradiction.

    Also accepts the search exclusion wrapper containing ``certificate`` and
    binds its branch and endpoints. All expected malformed inputs fail closed.
    """
    try:
        if not isinstance(record, dict):
            raise VerificationError("certificate must be an object")
        wrapper = record if "certificate" in record else None
        cert = record["certificate"] if wrapper is not None else record
        if not isinstance(cert, dict) or cert.get("schema") != SCHEMA:
            raise VerificationError("unsupported certificate schema")
        canonical = _canonical_observation(observation)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if cert.get("observation_sha256") != digest:
            raise VerificationError("observation digest mismatch")
        if cert.get("variable_order") != VARIABLE_ORDER:
            raise VerificationError("variable order mismatch")
        branch, low, high = cert["branch"], _fraction(cert["lo"]), _fraction(cert["hi"])
        if branch not in BRANCHES or not 540 <= low <= high <= 660:
            raise VerificationError("invalid geometric cell")
        if wrapper is not None:
            if (wrapper.get("branch") != branch or _endpoint(wrapper["lower"]) != low
                    or _endpoint(wrapper["upper"]) != high):
                raise VerificationError("exclusion wrapper differs from certified cell")
        terms = cert["multipliers"]
        if not isinstance(terms, list) or not terms:
            raise VerificationError("empty or malformed multipliers")
        multipliers = {}
        for term in terms:
            if not isinstance(term, list) or len(term) != 2:
                raise VerificationError("malformed sparse multiplier")
            i, value = term
            if isinstance(i, bool) or not isinstance(i, int) or i < 0 or i in multipliers:
                raise VerificationError("invalid or duplicate multiplier index")
            y = _fraction(value)
            if y <= 0:
                raise VerificationError("sparse multipliers must be strictly positive")
            multipliers[i] = y
        rows, boxes, count = _selected_rows(canonical, branch, low, high, set(multipliers))
        if type(cert.get("row_count")) is not int or cert.get("row_count") != count:
            raise VerificationError("constraint row count mismatch")
        if any(i >= count for i in multipliers):
            raise VerificationError("multiplier row index out of range")
        coefficients, rhs = [ZERO] * 15, ZERO
        for i, weight in multipliers.items():
            row, b = rows[i]
            rhs += weight * b
            for j, a in row:
                coefficients[j] += weight * a
        minimum = sum(a * (box[0] if a >= 0 else box[1]) for a, box in zip(coefficients, boxes))
        margin = minimum - rhs
        if _fraction(cert["margin"]) != margin:
            raise VerificationError("claimed margin differs from exact recomputation")
        if margin <= 0:
            raise VerificationError("no strictly positive exact contradiction")
        return {"valid": True, "reason": "exact rational box contradiction", "margin": fraction_text(margin),
                "branch": branch, "lo": fraction_text(low), "hi": fraction_text(high),
                "row_count": count, "support_count": len(multipliers)}
    except (VerificationError, KeyError, ValueError, TypeError, OverflowError, IndexError) as exc:
        return {"valid": False, "reason": str(exc)}


def verify_state(observation, state):
    """Verify all exclusions plus exact closed-domain coverage per branch.

    A safe fallback is accepted only with explicit fallback metadata and a full
    retained domain in every branch; its historical excluded cells may overlap.
    """
    try:
        if not isinstance(state, dict):
            raise VerificationError("state must be an object")
        cells, exclusions = state["cells"], state.get("exclusions", [])
        if not isinstance(cells, list) or not isinstance(exclusions, list):
            raise VerificationError("cells and exclusions must be arrays")
        intervals = {branch: [] for branch in BRANCHES}
        retained = {branch: [] for branch in BRANCHES}
        certificates = []
        for index, item in enumerate(cells):
            branch = item["branch"]
            low, high = _endpoint(item["lower"]), _endpoint(item["upper"])
            if branch not in BRANCHES or not 540 <= low <= high <= 660:
                raise VerificationError(f"invalid retained cell {index}")
            intervals[branch].append((low, high, "retained"))
            retained[branch].append((low, high))
        if not any(retained.values()):
            raise VerificationError("empty retained state cannot support projection")
        for index, item in enumerate(exclusions):
            result = verify_certificate(observation, item)
            certificates.append(result)
            if not result["valid"]:
                raise VerificationError(f"exclusion {index}: {result['reason']}")
            intervals[result["branch"]].append((_fraction(result["lo"]), _fraction(result["hi"]), "excluded"))
        fallback = bool(state.get("fallback_reason") or state.get("fallback")
                        or state.get("reason") in ("encoding_or_solver_error", "empty_enclosure_identity_fallback"))
        full_retained = all(any(a == 540 and b == 660 for a, b in retained[branch]) for branch in BRANCHES)
        conservative_fallback = fallback and full_retained
        if not conservative_fallback:
            for branch, spans in intervals.items():
                spans.sort()
                cursor = Q(540)
                if not spans:
                    raise VerificationError(f"uncovered branch {branch}")
                for low, high, kind in spans:
                    if low > cursor:
                        raise VerificationError(f"unrecorded gap in {branch}: {fraction_text(cursor)} to {fraction_text(low)}")
                    if low < cursor and high > low:
                        raise VerificationError(f"interior overlap in branch {branch}")
                    cursor = max(cursor, high)
                if cursor != 660:
                    raise VerificationError(f"uncovered upper domain in {branch}")
        lower = min(a for spans in retained.values() for a, b in spans)
        upper = max(b for spans in retained.values() for a, b in spans)
        if not ("lower" in state and "upper" in state) and "hull" not in state:
            raise VerificationError("state must record its retained hull")
        if "hull" in state:
            if not isinstance(state["hull"], list) or len(state["hull"]) != 2:
                raise VerificationError("malformed state hull")
            if [_endpoint(x) for x in state["hull"]] != [lower, upper]:
                raise VerificationError("state hull differs from retained-cell hull")
        if "lower" in state and _endpoint(state["lower"]) != lower:
            raise VerificationError("state lower differs from retained-cell hull")
        if "upper" in state and _endpoint(state["upper"]) != upper:
            raise VerificationError("state upper differs from retained-cell hull")
        return {"valid": True, "reason": "conservative full-domain fallback" if conservative_fallback
                else "all certificates valid and closed partition covers every branch",
                "partitioned": not conservative_fallback, "conservative": True,
                "checked_exclusions": len(certificates), "retained_cells": len(cells),
                "hull": [fraction_text(lower), fraction_text(upper)]}
    except (VerificationError, KeyError, ValueError, TypeError, OverflowError, IndexError) as exc:
        return {"valid": False, "reason": str(exc), "conservative": False}
