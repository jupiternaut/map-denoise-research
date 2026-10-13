"""Exact two-patch grayscale rectangles and conservative continuous enclosures.

The historical rectangle/interval primitives are imported read-only. Color
intervals describe three independent, view-invariant materials; channelwise
enclosures do not assert an exact jointly feasible image or equal materials.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from pathlib import Path
import sys

DEPENDENCY_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z')
if str(DEPENDENCY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPENDENCY_ROOT))

from layered_model import (FRONT_RECT, REAR_RECT, HIDDEN_REAR_RECT, REAR_DEPTH,
                           _project_rect, _project_rect_boxes, _inner_outer,
                           rectangle_area, rectangle_difference_area)
from continuous_world.interval_model import DEPTH_MIN, DEPTH_MAX, GRID, NOMINAL_CX, Interval, _q

BRANCHES = ('none', 'partial', 'full')
COLOR_LABELS = ('front-left', 'front-right', 'rear')
LEFT_RECT = (FRONT_RECT[0], Q(0), FRONT_RECT[2], FRONT_RECT[3])
RIGHT_RECT = (Q(0), FRONT_RECT[1], FRONT_RECT[2], FRONT_RECT[3])
PUBLIC_COLORS = ((Q(0), Q(1)),) * 3
_PIXELS = tuple((GRID.pixel_bounds(column, row),
                 rectangle_area(GRID.pixel_bounds(column, row)))
                for row in range(GRID.height) for column in range(GRID.width))


def _color_intervals(colors):
    pairs = tuple(tuple(pair) for pair in colors)
    if len(pairs) != 3 or any(len(pair) != 2 for pair in pairs):
        raise ValueError('colors must give three ordered closed material intervals')
    result = tuple((_q(low, 'color lower'), _q(high, 'color upper')) for low, high in pairs)
    if any(not Q(0) <= low <= high <= Q(1) for low, high in result):
        raise ValueError('color intervals must be ordered subsets of [0,1]')
    return result


def _rear_rect(branch):
    return None if branch == 'none' else REAR_RECT if branch == 'partial' else HIDDEN_REAR_RECT


@lru_cache(maxsize=2048)
def _front_boxes(radius, low, high):
    depth = Interval(low, high)
    result = []
    for index, center in enumerate(NOMINAL_CX):
        error = Q(0) if index == 1 else radius
        camera = Interval(center - error, center + error)
        result.append(tuple(_inner_outer(_project_rect_boxes(rect, depth, camera))
                            for rect in (LEFT_RECT, RIGHT_RECT, FRONT_RECT)))
    return tuple(result)


@lru_cache(maxsize=2048)
def _front_area_bounds(radius, low, high):
    result = []
    for left, right, _front in _front_boxes(radius, low, high):
        for pixel, area in _PIXELS:
            result.append((rectangle_area(left[0], pixel) / area,
                           rectangle_area(left[1], pixel) / area,
                           rectangle_area(right[0], pixel) / area,
                           rectangle_area(right[1], pixel) / area))
    return tuple(result)


@lru_cache(maxsize=64)
def _rear_boxes(branch, radius):
    rect = _rear_rect(branch)
    if rect is None:
        return ((None, None),) * 3
    result = []
    for index, center in enumerate(NOMINAL_CX):
        error = Q(0) if index == 1 else radius
        camera = Interval(center - error, center + error)
        result.append(_inner_outer(_project_rect_boxes(rect, Interval(REAR_DEPTH), camera)))
    return tuple(result)


@lru_cache(maxsize=4096)
def _geometry_bounds(branch, radius, low, high):
    """Color-free normalized visible-area bounds, shared across all models."""
    fronts = _front_boxes(radius, low, high)
    front_areas = _front_area_bounds(radius, low, high)
    rears = _rear_boxes(branch, radius)
    result = []
    for index in range(3):
        front_inner, front_outer = fronts[index][2]
        rear_inner, rear_outer = rears[index]
        for pixel_index, (pixel, area) in enumerate(_PIXELS):
            visible_low = rectangle_difference_area(rear_inner, front_outer, pixel) / area
            visible_high = rectangle_difference_area(rear_outer, front_inner, pixel) / area
            result.append((*front_areas[index * len(_PIXELS) + pixel_index], visible_low, visible_high))
    return tuple(result)


@lru_cache(maxsize=1024)
def _point_geometry(branch, depth, errors):
    """Exact areas at one generating geometry; no hidden colors enter cache."""
    result = []
    rear_world = _rear_rect(branch)
    for center, error in zip(NOMINAL_CX, errors):
        left = _project_rect(LEFT_RECT, depth, center + error)
        right = _project_rect(RIGHT_RECT, depth, center + error)
        front = _project_rect(FRONT_RECT, depth, center + error)
        rear = None if rear_world is None else _project_rect(rear_world, REAR_DEPTH, center + error)
        for pixel, area in _PIXELS:
            result.append((rectangle_area(left, pixel) / area,
                           rectangle_area(right, pixel) / area,
                           rectangle_difference_area(rear, front, pixel) / area))
    return tuple(result)


@dataclass(frozen=True)
class ColoredModel:
    branch: str
    camera_radius: Q = Q(1, 5)
    colors: tuple = PUBLIC_COLORS

    def __post_init__(self):
        if self.branch not in BRANCHES:
            raise ValueError(f'branch must be one of {BRANCHES}')
        radius = _q(self.camera_radius, 'camera_radius')
        if radius < 0:
            raise ValueError('camera_radius must be nonnegative')
        object.__setattr__(self, 'camera_radius', radius)
        object.__setattr__(self, 'colors', _color_intervals(self.colors))

    @property
    def grid(self):
        return GRID

    @property
    def depth_range(self):
        return DEPTH_MIN, DEPTH_MAX

    @property
    def observation_size(self):
        return 3 * len(_PIXELS) * 3

    def _depth_box(self, low, high):
        depth = Interval(_q(low, 'depth_lo'), _q(high, 'depth_hi'))
        if depth.lo < DEPTH_MIN or depth.hi > DEPTH_MAX:
            raise ValueError('depth must be contained in [540,660] mm')
        return depth

    def _offsets(self, offsets):
        errors = tuple(_q(value, 'camera offset') for value in offsets)
        if len(errors) != 3:
            raise ValueError('offsets must give left, reference and right camera errors')
        if errors[1] != 0:
            raise ValueError('reference camera offset must be zero')
        if any(abs(errors[index]) > self.camera_radius for index in (0, 2)):
            raise ValueError('side camera offset exceeds camera_radius')
        return errors

    def point_prediction(self, depth, offsets=(Q(0), Q(0), Q(0)), colors=None):
        """Generate an exact point image; colors are explicit material point values.

        Omitted colors are permitted only for three singleton public intervals.
        This generation API is separate from enclosure, which receives no actual
        colors, depth, offsets or generating branch beyond its public hypothesis.
        """
        depth = self._depth_box(depth, depth).lo
        errors = self._offsets(offsets)
        if colors is None:
            if any(low != high for low, high in self.colors):
                raise ValueError('point_prediction requires explicit colors unless all intervals are singletons')
            values = tuple(low for low, _high in self.colors)
        else:
            values = tuple(_q(value, 'point color') for value in colors)
            if len(values) != 3:
                raise ValueError('point colors must give front-left, front-right and rear values')
            if any(not low <= value <= high for value, (low, high) in zip(values, self.colors)):
                raise ValueError('point colors must lie in the model material intervals')
        result = []
        for areas in _point_geometry(self.branch, depth, errors):
            gray = Q(1) - sum((area * (1 - color) for area, color in zip(areas, values)), Q(0))
            if not Q(0) <= gray <= Q(1):
                raise ArithmeticError('invalid exact grayscale integral')
            result.extend((gray,) * 3)
        return tuple(result)

    def enclosure(self, low, high):
        """Conservative image bounds for every continuous geometry/color assignment."""
        depth = self._depth_box(low, high)
        return self._enclosure_cached(depth.lo, depth.hi)

    @lru_cache(maxsize=512)
    def _enclosure_cached(self, low, high):
        result = []
        active = self.colors[:2] if self.branch == 'none' else self.colors
        floor = min(color_low for color_low, _color_high in active)
        for bounds in _geometry_bounds(self.branch, self.camera_radius, low, high):
            lowers, uppers = bounds[::2], bounds[1::2]
            minimum = Q(1) - sum((area * (1 - color_low)
                                  for area, (color_low, _color_high) in zip(uppers, self.colors)), Q(0))
            maximum = Q(1) - sum((area * (1 - color_high)
                                  for area, (_color_low, color_high) in zip(lowers, self.colors)), Q(0))
            pair = max(floor, minimum), min(Q(1), maximum)
            if pair[0] > pair[1]:
                raise ArithmeticError('invalid conservative grayscale enclosure')
            result.extend((pair,) * 3)
        return tuple(result)

    def to_dict(self):
        rect = _rear_rect(self.branch)
        return {'branch': self.branch, 'camera_radius_mm': str(self.camera_radius),
                'color_labels': list(COLOR_LABELS),
                'color_intervals': [[str(low), str(high)] for low, high in self.colors],
                'depth_range_mm': [str(DEPTH_MIN), str(DEPTH_MAX)],
                'front_left_rect_mm': list(map(str, LEFT_RECT)),
                'front_right_rect_mm': list(map(str, RIGHT_RECT)),
                'rear_rect_mm': None if rect is None else list(map(str, rect)),
                'rear_depth_mm': str(REAR_DEPTH), 'background': '1',
                'material_contract': 'three independent constant grayscale materials shared across views',
                'geometry_contract': 'shared foreground depth, disjoint patches, rear minus entire foreground',
                'reference_camera_offset_mm': '0',
                'camera_nominal_cx_mm': list(map(str, NOMINAL_CX)), 'grid': GRID.to_dict(),
                'observation_layout': 'camera,row,column,RGB; grayscale repeated in three channels',
                'enclosure_scope': 'conservative channelwise continuous depth/pose/material outer bounds',
                'dependency_root': str(DEPENDENCY_ROOT)}
