"""Exact necessary material constraints for the registered sharing ablation.

Only an image, the public pose radius and algorithm parameters enter solve_arm.
The contractor is incomplete: absence of an explicit contradiction retains a
box and does not assert that a joint geometry/material assignment exists.
"""
from collections import deque
from fractions import Fraction as Q
from pathlib import Path
import sys
import time

# Imported historical sources are read-only, including their bytecode paths.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
NUISANCE_ROOT = ROOT.parent / "lovelace-nuisance-20261011T044500Z"
if str(NUISANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(NUISANCE_ROOT))
import nuisance_model
from nuisance_model import ColoredModel, PUBLIC_COLORS, BRANCHES
import layered_model
from continuous_world import interval_model
from continuous_world.interval_solver import solve_outer, encode, decode, rational
sys.path.insert(0, str(ROOT))

EPSILON = Q(1, 100)
TOLERANCE = Q(15, 32)
MAX_BOXES = 511
ROUNDS = 8
MODES = ("channel", "global_gray", "pixel_gray", "view_gray", "global_rgb")
PIXELS_PER_VIEW = 48


def _parameters(mode, max_boxes, rounds, tolerance, epsilon):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not isinstance(max_boxes, int) or isinstance(max_boxes, bool) or max_boxes < 0:
        raise ValueError("max_boxes must be a nonnegative integer")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer")
    tolerance, epsilon = rational(tolerance), rational(epsilon)
    if tolerance <= 0 or epsilon < 0:
        raise ValueError("positive tolerance and nonnegative epsilon required")
    return tolerance, epsilon


def grouped_rows(observation, geometry, mode, epsilon=EPSILON):
    """Return necessary rows by independently contracted material group.

    Geometry rows contain (L0,U0,L1,U1,L2,U2). Gray groups intersect the three
    noisy RGB bands at each pixel. RGB groups use each channel's own band and
    have independent material triples. Row numbers are local to each group.
    """
    if mode not in MODES or mode == "channel":
        raise ValueError("a grouped material mode is required")
    epsilon = rational(epsilon)
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    image = tuple(map(rational, observation))
    areas = tuple(tuple(map(rational, bounds)) for bounds in geometry)
    if not image or len(image) != 3 * len(areas):
        raise ValueError("nonempty RGB layout and area bounds must agree")
    if any(len(bounds) != 6 or any(not 0 <= a <= b <= 1
           for a, b in zip(bounds[::2], bounds[1::2])) for bounds in areas):
        raise ValueError("geometry requires three ordered nonnegative area bounds")
    groups = {f"c{k}": [] for k in range(3)} if mode == "global_rgb" else {}
    for pixel, bounds in enumerate(areas):
        lower_coefficients, upper_coefficients = bounds[::2], bounds[1::2]
        values = image[3 * pixel:3 * pixel + 3]
        if mode == "global_rgb":
            for channel, value in enumerate(values):
                groups[f"c{channel}"].extend((
                    ("upper", lower_coefficients, 1 - value + epsilon),
                    ("lower", upper_coefficients, 1 - value - epsilon)))
        else:
            group_id = ("global" if mode == "global_gray" else
                        f"p{pixel}" if mode == "pixel_gray" else
                        f"v{pixel // PIXELS_PER_VIEW}")
            groups.setdefault(group_id, []).extend((
                ("upper", lower_coefficients, min(1 - y + epsilon for y in values)),
                ("lower", upper_coefficients, max(1 - y - epsilon for y in values))))
    return {group_id: tuple(rows) for group_id, rows in groups.items()}


def propagate(rows, colors=PUBLIC_COLORS, rounds=ROUNDS):
    """The historical exact bounded contractor, with fresh local bounds.

    Explicit contradictions alone permit exclusion. Zero rounds performs no
    contraction and conservatively retains all valid public material bounds.
    """
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer")
    pairs = tuple(tuple(map(rational, pair)) for pair in colors)
    if len(pairs) != 3 or any(len(pair) != 2 or not 0 <= pair[0] <= pair[1] <= 1
                              for pair in pairs):
        raise ValueError("three valid material intervals required")
    necessary = []
    for kind, coefficients, bound in rows:
        coefficients = tuple(map(rational, coefficients))
        bound = rational(bound)
        if kind not in ("upper", "lower") or len(coefficients) != 3 or any(a < 0 for a in coefficients):
            raise ValueError("necessary rows require upper/lower and three nonnegative coefficients")
        necessary.append((kind, coefficients, bound))
    lower = [1 - high for _low, high in pairs]
    upper = [1 - low for low, _high in pairs]
    steps, performed = [], 0
    for iteration in range(rounds):
        performed = iteration + 1
        changed = False
        for index, (kind, coefficients, bound) in enumerate(necessary):
            values = lower if kind == "upper" else upper
            total = sum((a * value for a, value in zip(coefficients, values)), Q(0))
            conflict = total > bound if kind == "upper" else total < bound
            if conflict:
                return {"excluded": True, "certificate": {"steps": steps,
                        "contradiction": {"row": index}}, "rounds": performed}
            for variable, coefficient in enumerate(coefficients):
                if coefficient == 0:
                    continue
                other = total - coefficient * values[variable]
                candidate = (bound - other) / coefficient
                if kind == "upper" and candidate < upper[variable]:
                    upper[variable] = candidate
                    steps.append({"row": index, "variable": variable, "side": "upper", "value": encode(candidate)})
                    changed = True
                elif kind == "lower" and candidate > lower[variable]:
                    lower[variable] = candidate
                    steps.append({"row": index, "variable": variable, "side": "lower", "value": encode(candidate)})
                    changed = True
        if not changed:
            break
    return {"excluded": False, "rounds": performed,
            "darkness_bounds": [[encode(a), encode(b)] for a, b in zip(lower, upper)],
            "updates": len(steps)}


def solve_branch(observation, model, mode="global_gray", max_boxes=MAX_BOXES,
                 rounds=ROUNDS, tolerance=TOLERANCE, epsilon=EPSILON):
    """Same depth BFS and channel precheck, followed by grouped witnesses."""
    tolerance, epsilon = _parameters(mode, max_boxes, rounds, tolerance, epsilon)
    image = tuple(map(rational, observation))
    if len(image) != model.observation_size:
        raise ValueError("observation must match the public model RGB layout")
    depth_lo, depth_hi = model.depth_range
    if mode == "channel":
        result = solve_outer(image, model.enclosure, lo=depth_lo, hi=depth_hi,
                             epsilon=epsilon, tolerance=tolerance, max_boxes=max_boxes)
        result.update(mode=mode, rounds=rounds, budget=max_boxes,
                      shared_tests=0, propagation_rounds=rounds)
        return result
    queue = deque([(0, None, depth_lo, depth_hi)])
    records, retained = [], []
    next_id, evaluated, shared_calls = 1, 0, 0
    while queue:
        node_id, parent, left, right = queue.popleft()
        entry = {"id": node_id, "parent": parent, "lo": encode(left), "hi": encode(right)}
        if evaluated >= max_boxes:
            entry["status"] = "unresolved_budget"
            records.append(entry)
            retained.append((left, right, node_id))
            continue
        bounds = tuple((rational(a), rational(b)) for a, b in model.enclosure(left, right))
        evaluated += 1
        if len(bounds) != len(image) or any(a > b for a, b in bounds):
            raise ValueError("enclosure must match dimension with ordered intervals")
        conflict = next(((channel, a, b) for channel, (a, b) in enumerate(bounds)
                         if a > image[channel] + epsilon or b < image[channel] - epsilon), None)
        if conflict is not None:
            channel, a, b = conflict
            entry.update(status="excluded", witness_channel=channel,
                         prediction_interval=[encode(a), encode(b)])
            records.append(entry)
            continue
        geometry = nuisance_model._geometry_bounds(model.branch, model.camera_radius, left, right)
        groups = grouped_rows(image, geometry, mode, epsilon=epsilon)
        excluded = False
        for group_id, rows in groups.items():
            contraction = propagate(rows, colors=PUBLIC_COLORS, rounds=rounds)
            shared_calls += 1
            if contraction["excluded"]:
                entry.update(status="excluded_shared", group_id=group_id,
                             certificate=contraction["certificate"])
                excluded = True
                break
        if not excluded:
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
            "tolerance": encode(tolerance), "max_boxes": max_boxes, "evaluated_boxes": evaluated,
            "trace": records, "retained_ids": [node_id for _, _, node_id in retained], "hull": hull,
            "shared_tests": shared_calls, "propagation_rounds": rounds,
            "mode": mode, "rounds": rounds, "budget": max_boxes,
            "scope": "necessary_grouped_material_halfspaces_with_conservative_continuous_geometry_bounds"}


def union_outer(branch_outers, max_boxes=MAX_BOXES, rounds=ROUNDS,
                tolerance=TOLERANCE, epsilon=EPSILON, mode="global_gray"):
    """Merge all retained branch leaves, recording the actual public budget."""
    tolerance, epsilon = _parameters(mode, max_boxes, rounds, tolerance, epsilon)
    leaves = []
    for branch, outer in branch_outers.items():
        retained = set(outer["retained_ids"])
        for node in outer["trace"]:
            if node["id"] in retained:
                leaves.append({"branch": branch, "id": node["id"], "lo": node["lo"], "hi": node["hi"]})
    merged = []
    for left, right in sorted((decode(node["lo"]), decode(node["hi"])) for node in leaves):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return {"domain": [encode(interval_model.DEPTH_MIN), encode(interval_model.DEPTH_MAX)],
            "epsilon": encode(epsilon), "tolerance": encode(tolerance), "max_boxes_per_branch": max_boxes,
            "evaluated_boxes": sum(outer["evaluated_boxes"] for outer in branch_outers.values()),
            "unresolved_budget_leaves": sum(node["status"] == "unresolved_budget"
                for outer in branch_outers.values() for node in outer["trace"]),
            "source_leaves": leaves, "retained_intervals": [[encode(lo), encode(hi)] for lo, hi in merged],
            "hull": None if not merged else [encode(merged[0][0]), encode(merged[-1][1])],
            "retained_branches": [branch for branch, outer in branch_outers.items() if outer["hull"] is not None],
            "mode": mode, "rounds": rounds, "budget": max_boxes,
            "scope": "union_of_branch_depth_outer_leaves_then_convex_hull"}


def solve_arm(observation, radius, mode, max_boxes=MAX_BOXES, rounds=ROUNDS,
              tolerance=TOLERANCE, epsilon=EPSILON):
    """Solve all three scene-wide hypotheses with identical public priors."""
    tolerance, epsilon = _parameters(mode, max_boxes, rounds, tolerance, epsilon)
    image, radius = tuple(map(rational, observation)), rational(radius)
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    started = time.perf_counter()
    branches, seconds = {}, {}
    for branch in BRANCHES:
        tick = time.perf_counter()
        model = ColoredModel(branch, camera_radius=radius, colors=PUBLIC_COLORS)
        branches[branch] = solve_branch(image, model, mode=mode, max_boxes=max_boxes,
                                       rounds=rounds, tolerance=tolerance, epsilon=epsilon)
        seconds[branch] = time.perf_counter() - tick
    return {"branches": branches,
            "outer": union_outer(branches, max_boxes=max_boxes, rounds=rounds,
                                 tolerance=tolerance, epsilon=epsilon, mode=mode),
            "branch_seconds": seconds, "outer_seconds": time.perf_counter() - started}


def clear_geometry_cache():
    """Clear all historical geometry/model caches before a cold measured arm.

    The rectangle projection helpers themselves are uncached. The old interval
    triangle camera-data cache is the historical projection cache. No clearing
    occurs inside solve_arm, so its three branches may share geometry normally.
    """
    for cached in (nuisance_model._front_boxes, nuisance_model._front_area_bounds,
                   nuisance_model._rear_boxes, nuisance_model._geometry_bounds,
                   nuisance_model._point_geometry, ColoredModel._enclosure_cached,
                   layered_model.LayeredModel._point_prediction_cached,
                   layered_model.LayeredModel._enclosure_cached,
                   interval_model.Model._point_prediction_cached,
                   interval_model.Model._triangle_camera_data,
                   interval_model.Model._enclosure_cached):
        cached.cache_clear()
