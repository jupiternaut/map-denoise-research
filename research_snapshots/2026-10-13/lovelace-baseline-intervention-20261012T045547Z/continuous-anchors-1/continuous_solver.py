"""A9 continuous public-depth outer support with exact cell exclusions.

Rectangle inner/outer sets cover the whole depth x camera interval. A second
bound uses exact all-camera areas at the inverse-depth midpoint plus an exact
Lipschitz radius: four moving edge strips bound the rectangle symmetric
difference inside each pixel. No grid or affine-vertex claim is used.
"""
from collections import deque
from fractions import Fraction as Q
from functools import lru_cache
import inspect
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
ANCHOR_ROOT = PARENT / "independent-anchors-1"
if str(ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(ANCHOR_ROOT))
import anchor_solver as anchor

geometry, variation = anchor.geometry, anchor.variation
exact, encode = anchor.exact, anchor.encode
classify, certificate_check, witness_check = anchor.classify, anchor.certificate_check, anchor.witness_check
PRIMITIVE_ROOT = Path("/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z")
for _actual, _expected in (
        (anchor.__file__, ANCHOR_ROOT / "anchor_solver.py"),
        (geometry.__file__, PARENT / "recovery-1" / "baseline_geometry.py"),
        (inspect.getfile(geometry._project_rect_boxes), PRIMITIVE_ROOT / "layered_model.py"),
        (inspect.getfile(geometry.Interval), PRIMITIVE_ROOT / "source_snapshot" / "continuous_world" / "interval_model.py")):
    if Path(_actual).resolve() != _expected.resolve():
        raise RuntimeError(f"wrong frozen continuous source identity: {_actual}")

DOMAIN = (geometry.DEPTH_MIN, geometry.DEPTH_MAX)
DEPTH_MIN, DEPTH_MAX = DOMAIN
MIN_WIDTH = variation.TOLERANCE
if DOMAIN != (Q(540), Q(660)) or MIN_WIDTH != Q(15, 32):
    raise RuntimeError("frozen continuous domain/tolerance changed")
MAX_NODES = 127  # Classification budget; all pending cells remain in trace.
BRANCH_SECONDS = 3.0
ARM_NAMES, BRANCHES = anchor.ARM_NAMES, anchor.BRANCHES
POLICIES = ("KEEP", "HULL", "UNION")
DEFAULT_CAMERA_INTERVALS = ((-anchor.RADIUS, anchor.RADIUS), (Q(0), Q(0)),
                            (-anchor.RADIUS, anchor.RADIUS))


def strip_method_inputs(public, oldarm, anchors, arm_name):
    """Copy only registered measurement/box fields; labels and IDs are ignored."""
    if not isinstance(public, dict) or not isinstance(oldarm, dict):
        raise TypeError("public photo and calibration must be record dictionaries")
    camera, appearance = anchor._enabled(arm_name)
    photo = {key: public[key] for key in ("observation", "radius", "baseline", "clipped")}
    calibration = {"calibration_intervals": oldarm["calibration_intervals"],
                   "appearance_model": oldarm.get("appearance_model", "bounded"),
                   "drift_bound": oldarm.get("drift_bound", anchor.DRIFT_BOUND)}
    measurements = {}
    if camera or appearance:
        if not isinstance(anchors, dict):
            raise TypeError("enabled anchors must be a measurement record")
    if camera:
        measurements["camera"] = {key: anchors["camera"][key] for key in
                                    ("u", "Xf", "Zf", "survey_bound", "localization_bound")}
    if appearance:
        measurements["appearance"] = {key: anchors["appearance"][key] for key in
                                        ("coupons", "references", "bias_bound", "transfer_bound", "noise_bound")}
    # A8's record-identity guard sees a fixed internal identity, never a task ID.
    photo["observation_id"] = measurements["observation_id"] = "A9-public-measurements"
    return photo, calibration, measurements


def _depth_pair(lo, hi):
    lo, hi = exact(lo, "depth lo"), exact(hi, "depth hi")
    if not DEPTH_MIN <= lo <= hi <= DEPTH_MAX:
        raise ValueError("closed depth cell must lie in frozen [540,660]")
    return lo, hi


def _camera_intervals(intervals):
    if not isinstance(intervals, (tuple, list)) or len(intervals) != 3:
        raise ValueError("three camera intervals required")
    result = tuple(tuple(exact(value, "camera interval") for value in pair) for pair in intervals)
    if any(len(pair) != 2 or not -anchor.RADIUS <= pair[0] <= pair[1] <= anchor.RADIUS for pair in result):
        raise ValueError("ordered camera intervals must remain in the frozen radius")
    if result[1] != (Q(0), Q(0)):
        raise ValueError("middle camera gauge must be {0}")
    return result


@lru_cache(maxsize=4096)
def _rectangle_bounds(branch, lo, hi, intervals):
    depth = geometry.Interval(lo, hi)
    rear_world = geometry._rear_rect(branch)
    result = []
    for center, (elo, ehi) in zip(geometry.camera_centers(anchor.BASELINE), intervals):
        camera = geometry.Interval(center + elo, center + ehi)
        left, right, front = tuple(geometry._inner_outer(geometry._project_rect_boxes(rect, depth, camera))
                                   for rect in (geometry.LEFT_RECT, geometry.RIGHT_RECT, geometry.FRONT_RECT))
        rear = ((None, None) if rear_world is None else geometry._inner_outer(
            geometry._project_rect_boxes(rear_world, geometry.Interval(geometry.REAR_DEPTH), camera)))
        for pixel, area in geometry.PIXELS:
            result.append((geometry.rectangle_area(left[0], pixel) / area,
                           geometry.rectangle_area(left[1], pixel) / area,
                           geometry.rectangle_area(right[0], pixel) / area,
                           geometry.rectangle_area(right[1], pixel) / area,
                           geometry.rectangle_difference_area(rear[0], front[1], pixel) / area,
                           geometry.rectangle_difference_area(rear[1], front[0], pixel) / area))
    return tuple(result)


def rectangle_lipschitz(rectangle, center, interval, pixel):
    """Normalized symmetric-difference edge-strip coefficient in inverse depth.

    Inside a pixel of width w and height h, horizontal edge motion costs at
    most h times the sum of edge displacements, and vertical motion costs at
    most w times that sum. The maximum over the entire e interval is exact.
    """
    width, height = pixel[1] - pixel[0], pixel[3] - pixel[2]
    horizontal = sum((max(abs(x - center - interval[0]), abs(x - center - interval[1]))
                      for x in rectangle[:2]), Q(0)) / width
    vertical = (abs(rectangle[2]) + abs(rectangle[3])) / height
    return horizontal + vertical


@lru_cache(maxsize=4096)
def _interval_area_bounds(branch, lo, hi, intervals):
    generic = _rectangle_bounds(branch, lo, hi, intervals)
    qc = (Q(1) / lo + Q(1) / hi) / 2
    midpoint_depth = Q(1) / qc
    delta_q = max(abs(Q(1) / lo - qc), abs(Q(1) / hi - qc))
    center_bounds = anchor.area_bounds(branch, midpoint_depth, intervals)
    rectangles = (geometry.LEFT_RECT, geometry.RIGHT_RECT, geometry.FRONT_RECT)
    result = []
    for view, (center, interval) in enumerate(zip(geometry.camera_centers(anchor.BASELINE), intervals)):
        for pixel_index, (pixel, _area) in enumerate(geometry.PIXELS):
            index = view * geometry.PIXELS_PER_VIEW + pixel_index
            lipschitz = tuple(rectangle_lipschitz(rect, center, interval, pixel) for rect in rectangles)
            enclosure = []
            for material, coefficient in enumerate(lipschitz):
                lower = max(generic[index][2 * material], Q(0),
                            center_bounds[index][2 * material] - coefficient * delta_q)
                upper = min(generic[index][2 * material + 1], Q(1),
                            center_bounds[index][2 * material + 1] + coefficient * delta_q)
                if lower > upper:
                    raise ArithmeticError("independently sound area bounds disagree")
                enclosure.extend((lower, upper))
            result.append(tuple(enclosure))
    return tuple(result)


def interval_area_bounds(branch, lo, hi, intervals):
    """Enclose all normalized material areas for every z/e in a closed cell."""
    lo, hi = _depth_pair(lo, hi)
    if branch not in BRANCHES:
        raise ValueError("unregistered occlusion branch")
    intervals = _camera_intervals(intervals)
    return _interval_area_bounds(branch, lo, hi, intervals)


def build_interval_problem(public, oldarm, anchors, arm_name, lo, hi, branch):
    """Return whole-cell necessary rows with unchanged d9/t3/tA3 bounds."""
    lo, hi = _depth_pair(lo, hi)
    photo, calibration, measurements = strip_method_inputs(public, oldarm, anchors, arm_name)
    _lo, image, bounds, (camera, appearance) = anchor._inputs(
        photo, calibration, measurements, arm_name, lo, branch)
    intervals = DEFAULT_CAMERA_INTERVALS
    if camera:
        _survey, intervals = anchor.camera_intervals(measurements)
    if intervals is None:
        rows = (("upper", (Q(0),) * 15, Q(-1)),)
    else:
        areas = interval_area_bounds(branch, lo, hi, intervals)
        rows12 = variation.necessary_rows(image, areas, "joint", clipped=True,
                                         drift_bound=anchor.DRIFT_BOUND,
                                         beta=anchor.BETA, epsilon=anchor.EPSILON)
        rows = tuple((kind, coefficients + (Q(0),) * 3, bound) for kind, coefficients, bound in rows12)
    if appearance:
        rows += anchor.appearance_rows(measurements)
    return rows, bounds


def merge_intervals(intervals):
    """Exact closed union; overlaps and touching endpoints merge."""
    if isinstance(intervals, (str, bytes, dict)):
        raise TypeError("outer support must be a sequence of closed intervals")
    pairs = []
    for pair in intervals:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("outer support requires closed endpoint pairs")
        pairs.append(_depth_pair(*pair))
    merged = []
    for lo, hi in sorted(pairs):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def _encode_intervals(intervals):
    return [[encode(lo), encode(hi)] for lo, hi in intervals]


def solve_interval_branch(public, oldarm, anchors, arm_name, branch, deadline=None):
    """Breadth-first whole-domain enclosure; every unpruned cell is retained.

    MAX_NODES counts classified cells. Pending cells at that cap or deadline
    are recorded without a classification, preserving the full denominator.
    A FEASIBLE_WITNESS proves the outer relaxation only, never a forward world.
    """
    started = time.monotonic()
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                 or not math.isfinite(deadline)):
        raise ValueError("deadline must be finite absolute time.monotonic()")
    end = started + BRANCH_SECONDS if deadline is None else min(started + BRANCH_SECONDS, float(deadline))
    if type(MAX_NODES) is not int or MAX_NODES < 0:
        raise ValueError("MAX_NODES must be a nonnegative classification budget")
    # Validate inputs even when no classifier call is permitted.
    photo, calibration, measurements = strip_method_inputs(public, oldarm, anchors, arm_name)
    anchor._inputs(photo, calibration, measurements, arm_name, DEPTH_MIN, branch)
    queue = deque([(0, None, DEPTH_MIN, DEPTH_MAX)])
    trace, retained = [], []
    next_id, classified, pruned = 1, 0, 0
    termination = "tree_complete"
    while queue:
        node_id, parent, lo, hi = queue.popleft()
        record = {"id": node_id, "parent": parent, "lo": encode(lo), "hi": encode(hi),
                  "classification": None}
        if time.monotonic() >= end:
            record["status"] = "retained_deadline"
            termination = "deadline"
        elif classified >= MAX_NODES:
            record["status"] = "retained_node_budget"
            if termination != "deadline":
                termination = "node_budget"
        else:
            rows, bounds = build_interval_problem(photo, calibration, measurements, arm_name, lo, hi, branch)
            # Geometry/row construction shares the absolute deadline.
            if time.monotonic() >= end:
                record["status"] = "retained_deadline"
                termination = "deadline"
            else:
                classification = classify(rows, bounds, deadline=end)
                classified += 1
                record["classification"] = classification
                status = classification["status"]
                if status == "CERTIFIED_INFEASIBLE":
                    if not certificate_check(rows, bounds, classification["certificate"]):
                        raise ArithmeticError("whole-cell certificate failed exact recheck")
                    record["status"] = "excluded_certificate"
                    pruned += 1
                else:
                    if status == "FEASIBLE_WITNESS":
                        if not witness_check(rows, bounds, classification["witness"]):
                            raise ArithmeticError("outer-relaxation witness failed exact recheck")
                    elif status != "UNKNOWN":
                        raise ValueError("unregistered frozen classification status")
                    if hi - lo <= MIN_WIDTH:
                        record["status"] = "retained_tolerance"
                    else:
                        midpoint = (lo + hi) / 2
                        record.update(status="split", children=[next_id, next_id + 1])
                        queue.append((next_id, node_id, lo, midpoint))
                        queue.append((next_id + 1, node_id, midpoint, hi))
                        next_id += 2
        trace.append(record)
        if record["status"].startswith("retained_"):
            retained.append((lo, hi, node_id))
    outer = merge_intervals((lo, hi) for lo, hi, _node_id in retained)
    hull = None if not outer else [encode(outer[0][0]), encode(outer[-1][1])]
    return {"domain": [encode(value) for value in DOMAIN], "trace": trace,
            "retained_ids": [node_id for _lo, _hi, node_id in retained],
            "outer_intervals": _encode_intervals(outer), "hull": hull,
            "status": "OUTER_EMPTY" if not outer else "OUTER_RETAINED",
            "classified_nodes": classified, "pruned_nodes": pruned,
            "max_nodes": MAX_NODES, "min_width": encode(MIN_WIDTH),
            "branch_seconds": BRANCH_SECONDS, "search_budget_seconds": max(0.0, end - started),
            "termination": termination, "seconds": max(0.0, time.monotonic() - started)}


def select_depth(outer_intervals, incumbent, policy):
    """Select one coordinate, independently of interval search and labels.

    HULL projection cannot increase absolute/squared error if the true depth
    belongs to the outer support. Nearest UNION projection has no such general
    guarantee for a disconnected support. Both are conditional on model cover.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    incumbent = exact(incumbent, "incumbent depth")
    outer = merge_intervals(outer_intervals)
    hull = None if not outer else (outer[0][0], outer[-1][1])
    selected, status = incumbent, "KEEP"
    if policy != "KEEP":
        if not outer:
            status = "SUPPORT_FAILED"
        elif policy == "HULL":
            selected = min(hull[1], max(hull[0], incumbent))
            status = "PROJECTED"
        else:
            candidates = tuple(min(hi, max(lo, incumbent)) for lo, hi in outer)
            selected = min(candidates, key=lambda value: (abs(value - incumbent), value))
            status = "PROJECTED"
    return {"policy": policy, "status": status, "incumbent": encode(incumbent),
            "selected": encode(selected), "moved": selected != incumbent,
            "outer_intervals": _encode_intervals(outer),
            "hull": None if hull is None else [encode(value) for value in hull],
            "model_conditional_nonregression": policy in ("KEEP", "HULL")}


def clear_geometry_cache():
    _rectangle_bounds.cache_clear()
    _interval_area_bounds.cache_clear()
    anchor.clear_geometry_cache()
