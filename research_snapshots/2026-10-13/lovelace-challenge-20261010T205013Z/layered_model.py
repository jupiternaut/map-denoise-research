"""Exact layered-rectangle images and continuous depth/pose enclosures.

The copied source_snapshot supplies the frozen rational Interval, image grid,
camera gauge and overlap primitive. It is imported, never modified. Every
certified depth lies in [540,660], strictly in front of the fixed rear at 720.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import sys
from typing import Iterable

_SOURCE = Path(__file__).resolve().parent / "source_snapshot"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from continuous_world.interval_model import (
    DEPTH_MAX, DEPTH_MIN, GRID, NOMINAL_CX, Interval, Pair, RationalInput,
    _overlap, _q,
)
from finite_world.renderer import Camera

Q = Fraction
Rect = tuple[Fraction, Fraction, Fraction, Fraction]
MaybeRect = Rect | None
FRONT_RECT: Rect = (Q(-20), Q(20), Q(-15), Q(15))
REAR_RECT: Rect = (Q(-20), Q(36), Q(-17), Q(23))
HIDDEN_REAR_RECT: Rect = (Q(-100), Q(100), Q(-80), Q(80))
REAR_DEPTH = Q(720)
KINDS = ("layered_contrast", "layered_same", "layered_hidden")


def rectangle_intersection(first: MaybeRect, second: MaybeRect) -> MaybeRect:
    """Intersect image-coordinate rectangles; empty/zero area becomes None."""
    if first is None or second is None:
        return None
    left, right = max(first[0], second[0]), min(first[1], second[1])
    bottom, top = max(first[2], second[2]), min(first[3], second[3])
    return (left, right, bottom, top) if left < right and bottom < top else None


def rectangle_area(rect: MaybeRect, pixel: MaybeRect = None) -> Fraction:
    """Exact area, optionally clipped to one pixel using frozen overlaps."""
    if rect is None:
        return Q(0)
    if pixel is None:
        return max(Q(0), rect[1] - rect[0]) * max(Q(0), rect[3] - rect[2])
    return (_overlap(rect[0], rect[1], pixel[0], pixel[1])
            * _overlap(rect[2], rect[3], pixel[2], pixel[3]))


def rectangle_difference_area(first: MaybeRect, second: MaybeRect, pixel: Rect) -> Fraction:
    """Area of (first minus second) inside pixel; both sets are rectangles."""
    return rectangle_area(first, pixel) - rectangle_area(rectangle_intersection(first, second), pixel)


def rectangle_union_area(first: MaybeRect, second: MaybeRect, pixel: Rect) -> Fraction:
    return rectangle_area(first, pixel) + rectangle_difference_area(second, first, pixel)


def _project_rect(rect: Rect, depth: Fraction, camera_x: Fraction) -> Rect:
    if depth <= 0:
        raise ValueError("Projected rectangle Z denominator must be strictly positive")
    return ((rect[0] - camera_x) / depth, (rect[1] - camera_x) / depth,
            rect[2] / depth, rect[3] / depth)


def _project_rect_boxes(rect: Rect, depth: Interval, camera_x: Interval) -> tuple[Interval, ...]:
    if depth.lo <= 0:
        raise ValueError("Projected rectangle Z denominator must stay strictly positive")
    return ((Interval(rect[0]) - camera_x) / depth,
            (Interval(rect[1]) - camera_x) / depth,
            Interval(rect[2]) / depth, Interval(rect[3]) / depth)


def _inner_outer(boxes: tuple[Interval, ...]) -> tuple[MaybeRect, Rect]:
    """Intersection/superset from coordinate bounds, without endpoint images."""
    left, right, bottom, top = boxes
    inner = (left.hi, right.lo, bottom.hi, top.lo)
    if inner[0] >= inner[1] or inner[2] >= inner[3]:
        inner = None
    outer = (left.lo, right.hi, bottom.lo, top.hi)
    return inner, outer


def _clamp_color_pair(low: Fraction, high: Fraction) -> Pair:
    # Clipping to the known physical [0,1] range preserves inclusion even
    # when independently bounded terms overcount a shared coverage region.
    result = (max(Q(0), min(Q(1), low)), max(Q(0), min(Q(1), high)))
    if result[0] > result[1]:
        raise ArithmeticError("Invalid layered pixel enclosure")
    return result


def _front_first_gray(front: MaybeRect, rear: MaybeRect, rear_color: Fraction, pixel: Rect) -> Fraction:
    area = rectangle_area(pixel)
    return (1 - rectangle_area(front, pixel) / area
            - (1 - rear_color) * rectangle_difference_area(rear, front, pixel) / area)


@dataclass(frozen=True)
class LayeredModel:
    kind: str = "layered_contrast"
    camera_radius: Fraction = Q(0)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"LayeredModel.kind must be one of {KINDS}")
        radius = _q(self.camera_radius, "camera_radius")
        if radius < 0:
            raise ValueError("camera_radius must be nonnegative")
        # For the hidden control F(z,c) subset R(c) for all |c|<=220 and
        # z>=540. The side-camera centers have |nominal c|=70.
        if self.kind == "layered_hidden" and radius > 150:
            raise ValueError("layered_hidden camera_radius must be <=150 for its containment guarantee")
        object.__setattr__(self, "camera_radius", radius)

    @property
    def grid(self):
        return GRID

    @property
    def nominal_cameras(self) -> tuple[Camera, ...]:
        return tuple(Camera(cx, name) for cx, name in zip(NOMINAL_CX, ("left", "reference", "right")))

    @property
    def depth_range(self) -> Pair:
        return DEPTH_MIN, DEPTH_MAX

    @property
    def observation_size(self) -> int:
        return 3 * GRID.width * GRID.height * 3

    @property
    def rear_rect(self) -> Rect:
        return HIDDEN_REAR_RECT if self.kind == "layered_hidden" else REAR_RECT

    @property
    def rear_color(self) -> Fraction:
        return Q(1, 3) if self.kind == "layered_contrast" else Q(0)

    def _depth_box(self, low: RationalInput, high: RationalInput) -> Interval:
        depth = Interval(_q(low, "depth_lo"), _q(high, "depth_hi"))
        if depth.lo < DEPTH_MIN or depth.hi > DEPTH_MAX:
            raise ValueError("Depth box must be contained in [540,660] mm")
        return depth

    def _offsets(self, offsets: Iterable[RationalInput]) -> tuple[Fraction, ...]:
        errors = tuple(_q(value, "camera offset") for value in offsets)
        if len(errors) != 3:
            raise ValueError("offsets must give errors for left, reference and right cameras")
        if errors[1] != 0:
            raise ValueError("Reference camera gauge is fixed: middle offset must be zero")
        if any(abs(errors[i]) > self.camera_radius for i in (0, 2)):
            raise ValueError("Side-camera offset exceeds camera_radius")
        return errors

    def _camera_box(self, index: int) -> Interval:
        radius = Q(0) if index == 1 else self.camera_radius
        return Interval(NOMINAL_CX[index] - radius, NOMINAL_CX[index] + radius)

    def point_prediction(self, depth: RationalInput,
                         offsets: Iterable[RationalInput] = (Q(0), Q(0), Q(0))) -> tuple[Fraction, ...]:
        """Render an exact positive-depth hypothesis, including outside the contract.

        At z>720 the rear becomes the nearer opaque rectangle. At coincident
        z=720 the black front owns their overlap (a deterministic tie convention).
        Such hypotheses do not change the [540,660] certified target domain.
        """
        depth = _q(depth, "depth")
        if depth <= 0:
            raise ValueError("Projected rectangle Z denominator must be strictly positive")
        return self._point_prediction_cached(depth, self._offsets(offsets))

    @lru_cache(maxsize=1024)
    def _point_prediction_cached(self, depth: Fraction, errors: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
        result = []
        for center, error in zip(NOMINAL_CX, errors):
            front = _project_rect(FRONT_RECT, depth, center + error)
            rear = _project_rect(self.rear_rect, REAR_DEPTH, center + error)
            for row in range(GRID.height):
                for column in range(GRID.width):
                    pixel = GRID.pixel_bounds(column, row)
                    if depth <= REAR_DEPTH:
                        gray = _front_first_gray(front, rear, self.rear_color, pixel)
                    else:
                        area = rectangle_area(pixel)
                        gray = (1 - (1 - self.rear_color) * rectangle_area(rear, pixel) / area
                                - rectangle_difference_area(front, rear, pixel) / area)
                    if not 0 <= gray <= 1:
                        raise ArithmeticError("Invalid exact layered pixel color")
                    result.extend((gray,) * 3)
        return tuple(result)

    def enclosure(self, depth_lo: RationalInput, depth_hi: RationalInput) -> tuple[Pair, ...]:
        """Cover all depths and all allowed additive camera errors continuously."""
        return self._enclosure_cached(self._depth_box(depth_lo, depth_hi))

    @lru_cache(maxsize=2048)
    def _enclosure_cached(self, depth: Interval) -> tuple[Pair, ...]:
        result = []
        for index in range(3):
            camera = self._camera_box(index)
            front_inner, front_outer = _inner_outer(_project_rect_boxes(FRONT_RECT, depth, camera))
            rear_inner, rear_outer = _inner_outer(_project_rect_boxes(self.rear_rect, Interval(REAR_DEPTH), camera))
            for row in range(GRID.height):
                for column in range(GRID.width):
                    pixel = GRID.pixel_bounds(column, row)
                    area = rectangle_area(pixel)
                    if self.kind == "layered_hidden":
                        # Pointwise F subset R holds for shared camera pose; no
                        # cross-pose Fouter subset Rinner assumption is made.
                        low = 1 - rectangle_area(rear_outer, pixel) / area
                        high = 1 - rectangle_area(rear_inner, pixel) / area
                    elif self.rear_color == 0:
                        low = 1 - rectangle_union_area(front_outer, rear_outer, pixel) / area
                        high = 1 - rectangle_union_area(front_inner, rear_inner, pixel) / area
                    else:
                        rear_visible_low = rectangle_difference_area(rear_inner, front_outer, pixel)
                        rear_visible_high = rectangle_difference_area(rear_outer, front_inner, pixel)
                        low = (1 - rectangle_area(front_outer, pixel) / area
                               - (1 - self.rear_color) * rear_visible_high / area)
                        high = (1 - rectangle_area(front_inner, pixel) / area
                                - (1 - self.rear_color) * rear_visible_low / area)
                    result.extend((_clamp_color_pair(low, high),) * 3)
        return tuple(result)

    def enclosure_details(self, depth_lo: RationalInput, depth_hi: RationalInput) -> dict:
        depth = self._depth_box(depth_lo, depth_hi)
        cameras = []
        for index in range(3):
            camera = self._camera_box(index)
            item = {"camera_cx_box_mm": [str(c) for c in camera.pair()]}
            for label, rect, box in (("front", FRONT_RECT, depth),
                                     ("rear", self.rear_rect, Interval(REAR_DEPTH))):
                coordinates = _project_rect_boxes(rect, box, camera)
                inner, outer = _inner_outer(coordinates)
                item[label] = {"coordinate_intervals": [[str(c) for c in b.pair()] for b in coordinates],
                               "inner": [str(c) for c in inner] if inner is not None else None,
                               "outer": [str(c) for c in outer]}
            cameras.append(item)
        return {"kind": self.kind, "depth_box_mm": [str(c) for c in depth.pair()],
                "camera_radius_mm": str(self.camera_radius), "reference_offset_mm": "0",
                "method": "exact interval projected rectangle inclusions and clipped set areas",
                "visible_rear_inclusion": "Rinner minus Fouter subset R minus F subset Router minus Finner",
                "equal_black_inclusion": "Finner union Rinner subset F union R subset Fouter union Router",
                "hidden_depth_independence": self.kind == "layered_hidden",
                "camera_rectangles": cameras}

    def to_dict(self) -> dict:
        return {"kind": self.kind, "depth_range_mm": [str(DEPTH_MIN), str(DEPTH_MAX)],
                "side_camera_offset_radius_mm": str(self.camera_radius), "reference_camera_offset_mm": "0",
                "camera_nominal_cx_mm": [str(c) for c in NOMINAL_CX], "grid": GRID.to_dict(),
                "front_rect_mm": [str(c) for c in FRONT_RECT], "front_color": "0",
                "rear_rect_mm": [str(c) for c in self.rear_rect], "rear_depth_mm": str(REAR_DEPTH),
                "rear_color": str(self.rear_color), "background": "1", "opaque": True,
                "evaluation_target": "reference center-ray first hit depth z on foreground",
                "observation_layout": "camera,row,column,RGB",
                "scope": "declared rational parallel-camera two-rectangle simulation",
                "hidden_containment_camera_radius_limit_mm": "150" if self.kind == "layered_hidden" else None}
