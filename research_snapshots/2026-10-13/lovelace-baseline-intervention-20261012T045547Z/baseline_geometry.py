"""Explicit registered camera baselines over read-only rational primitives."""
from fractions import Fraction as Q
from functools import lru_cache
from pathlib import Path
import sys

sys.dont_write_bytecode = True
NUISANCE_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-nuisance-20261011T044500Z')
if str(NUISANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(NUISANCE_ROOT))
import nuisance_model as historical
from layered_model import (FRONT_RECT, REAR_RECT, HIDDEN_REAR_RECT, REAR_DEPTH,
                           _project_rect, _project_rect_boxes, _inner_outer,
                           rectangle_area, rectangle_difference_area)
from continuous_world.interval_model import DEPTH_MIN, DEPTH_MAX, GRID, Interval, _q

BRANCHES = historical.BRANCHES
LEFT_RECT, RIGHT_RECT = historical.LEFT_RECT, historical.RIGHT_RECT
PIXELS = historical._PIXELS
PIXELS_PER_VIEW = len(PIXELS)
BASELINES = (Q(70), Q(80))


def camera_centers(baseline):
    baseline = _q(baseline, 'baseline')
    if baseline not in BASELINES:
        raise ValueError('baseline must be one of the registered 70 or 80 mm')
    return -baseline, Q(0), baseline


def _inputs(branch, radius, low, high, baseline):
    if branch not in BRANCHES:
        raise ValueError('unregistered occlusion branch')
    radius, low, high, baseline = (_q(value, name) for value, name in
        ((radius, 'radius'), (low, 'depth lo'), (high, 'depth hi'), (baseline, 'baseline')))
    camera_centers(baseline)
    if radius < 0 or not DEPTH_MIN <= low <= high <= DEPTH_MAX:
        raise ValueError('nonnegative radius and ordered registered depth range required')
    return radius, low, high, baseline


def _rear_rect(branch):
    return None if branch == 'none' else REAR_RECT if branch == 'partial' else HIDDEN_REAR_RECT


@lru_cache(maxsize=4096)
def _front_boxes(radius, low, high, baseline):
    depth = Interval(low, high)
    result = []
    for index, center in enumerate(camera_centers(baseline)):
        error = Q(0) if index == 1 else radius
        camera = Interval(center - error, center + error)
        result.append(tuple(_inner_outer(_project_rect_boxes(rect, depth, camera))
                            for rect in (LEFT_RECT, RIGHT_RECT, FRONT_RECT)))
    return tuple(result)


@lru_cache(maxsize=4096)
def _front_area_bounds(radius, low, high, baseline):
    result = []
    for left, right, _front in _front_boxes(radius, low, high, baseline):
        for pixel, area in PIXELS:
            result.append((rectangle_area(left[0], pixel) / area,
                           rectangle_area(left[1], pixel) / area,
                           rectangle_area(right[0], pixel) / area,
                           rectangle_area(right[1], pixel) / area))
    return tuple(result)


@lru_cache(maxsize=128)
def _rear_boxes(branch, radius, baseline):
    rect = _rear_rect(branch)
    if rect is None:
        return ((None, None),) * 3
    result = []
    for index, center in enumerate(camera_centers(baseline)):
        error = Q(0) if index == 1 else radius
        camera = Interval(center - error, center + error)
        result.append(_inner_outer(_project_rect_boxes(rect, Interval(REAR_DEPTH), camera)))
    return tuple(result)


@lru_cache(maxsize=8192)
def _geometry_bounds(branch, radius, low, high, baseline):
    fronts = _front_boxes(radius, low, high, baseline)
    front_areas = _front_area_bounds(radius, low, high, baseline)
    rears = _rear_boxes(branch, radius, baseline)
    result = []
    for index in range(3):
        front_inner, front_outer = fronts[index][2]
        rear_inner, rear_outer = rears[index]
        for pixel_index, (pixel, area) in enumerate(PIXELS):
            visible_low = rectangle_difference_area(rear_inner, front_outer, pixel) / area
            visible_high = rectangle_difference_area(rear_outer, front_inner, pixel) / area
            result.append((*front_areas[index * PIXELS_PER_VIEW + pixel_index],
                           visible_low, visible_high))
    return tuple(result)


def geometry_bounds(branch, radius, lo, hi, baseline):
    """Enclose each material's visible area over a continuous depth/pose box."""
    radius, lo, hi, baseline = _inputs(branch, radius, lo, hi, baseline)
    return _geometry_bounds(branch, radius, lo, hi, baseline)


@lru_cache(maxsize=2048)
def _point_areas(branch, depth, errors, baseline):
    result = []
    rear_world = _rear_rect(branch)
    for center, error in zip(camera_centers(baseline), errors):
        camera_x = center + error
        left = _project_rect(LEFT_RECT, depth, camera_x)
        right = _project_rect(RIGHT_RECT, depth, camera_x)
        front = _project_rect(FRONT_RECT, depth, camera_x)
        rear = None if rear_world is None else _project_rect(rear_world, REAR_DEPTH, camera_x)
        for pixel, area in PIXELS:
            result.append((rectangle_area(left, pixel) / area,
                           rectangle_area(right, pixel) / area,
                           rectangle_difference_area(rear, front, pixel) / area))
    return tuple(result)


def point_rgb(branch, depth, errors, colors, baseline):
    """Generate an exact point image; generator labels never enter enclosures."""
    _radius, depth, _high, baseline = _inputs(branch, Q(0), depth, depth, baseline)
    errors = tuple(_q(value, 'camera error') for value in errors)
    colors = tuple(_q(value, 'material gray') for value in colors)
    if len(errors) != 3 or errors[1] != 0:
        raise ValueError('three camera errors with zero reference gauge required')
    if len(colors) != 3 or any(not 0 <= value <= 1 for value in colors):
        raise ValueError('three physical material grays required')
    result = []
    for areas in _point_areas(branch, depth, errors, baseline):
        gray = Q(1) - sum((area * (1 - color) for area, color in zip(areas, colors)), Q(0))
        if not 0 <= gray <= 1:
            raise ArithmeticError('invalid exact pixel integral')
        result.extend((gray,) * 3)
    return tuple(result)


def clear_geometry_cache():
    for cached in (_front_boxes, _front_area_bounds, _rear_boxes,
                   _geometry_bounds, _point_areas):
        cached.cache_clear()
