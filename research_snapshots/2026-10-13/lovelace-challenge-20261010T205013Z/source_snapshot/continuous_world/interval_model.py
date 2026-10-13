"""Rational enclosures for a declared continuous-depth, bounded-camera world.

This is a camera/geometry model, not endpoint sampling. The LOVELACE case
isolates the same M4 seed facet used in stage one, retaining its fixed opaque
black material. Every depth in [540,660] and every admissible side-camera
offset is enclosed. The reference camera is a fixed gauge: its offset is zero.

Triangle coverage changes are bounded through projected-vertex intervals and
a triangle symmetric-difference area bound. A rectangle supplies a separate
analytic baseline with direct interval overlap. Neither certifies full-character
reconstruction, continuous material uncertainty, or a physical camera pipeline.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from finite_world.mesh_patch import (DEFAULT_ASSET, M4_SHA256, TriangularWorld,
                                     centered_triangle, clip_polygon, polygon_area, render_triangle)
from finite_world.renderer import Camera, ImageGrid

Q = Fraction
RationalInput = Fraction | int | str
Pair = tuple[Fraction, Fraction]
DEPTH_MIN, DEPTH_MAX = Q(540), Q(660)
NOMINAL_CX = (Q(-70), Q(0), Q(70))
GRID = ImageGrid(8, 6, Q(-1, 5), Q(1, 5), Q(-2, 25), Q(2, 25))
RECTANGLE_HALF_WIDTH, RECTANGLE_HALF_HEIGHT = Q(20), Q(15)


def _q(value: RationalInput, field_name: str = "rational") -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int, str)):
        raise TypeError(f"{field_name} must be Fraction, int or rational string; floats are forbidden")
    return Q(value)


@dataclass(frozen=True)
class Interval:
    """A closed interval with exact rational endpoints; point intervals are valid."""

    lo: Fraction
    hi: Fraction | None = None

    def __post_init__(self) -> None:
        low = _q(self.lo, "interval lo")
        high = low if self.hi is None else _q(self.hi, "interval hi")
        if low > high:
            raise ValueError("Interval lower endpoint must not exceed upper endpoint")
        object.__setattr__(self, "lo", low)
        object.__setattr__(self, "hi", high)

    def __add__(self, other) -> "Interval":
        other = _iv(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other) -> "Interval":
        return self + (-_iv(other))

    def __rsub__(self, other) -> "Interval":
        return _iv(other) - self

    def __mul__(self, other) -> "Interval":
        other = _iv(other)
        products = (self.lo * other.lo, self.lo * other.hi,
                    self.hi * other.lo, self.hi * other.hi)
        return Interval(min(products), max(products))

    __rmul__ = __mul__

    def reciprocal(self) -> "Interval":
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError("Cannot invert an interval containing zero")
        return Interval(1 / self.hi, 1 / self.lo)

    def __truediv__(self, other) -> "Interval":
        return self * _iv(other).reciprocal()

    def __rtruediv__(self, other) -> "Interval":
        return _iv(other) / self

    @property
    def midpoint(self) -> Fraction:
        return (self.lo + self.hi) / 2

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    def contains(self, value: RationalInput) -> bool:
        value = _q(value)
        return self.lo <= value <= self.hi

    def pair(self) -> Pair:
        return self.lo, self.hi


def _iv(value) -> Interval:
    return value if isinstance(value, Interval) else Interval(_q(value))


def _interval_min(first, second) -> Interval:
    first, second = _iv(first), _iv(second)
    return Interval(min(first.lo, second.lo), min(first.hi, second.hi))


def _interval_max(first, second) -> Interval:
    first, second = _iv(first), _iv(second)
    return Interval(max(first.lo, second.lo), max(first.hi, second.hi))


def _unit_clamp(value: Interval) -> Interval:
    return _interval_min(1, _interval_max(0, value))


def _overlap(left: Fraction, right: Fraction, low: Fraction, high: Fraction) -> Fraction:
    return max(Q(0), min(right, high) - max(left, low))


def _overlap_interval(left: Interval, right: Interval,
                       low: Fraction, high: Fraction) -> Interval:
    # min/max are monotone in each argument, hence their endpoint extensions
    # enclose every correlated geometry assignment as well as independent ones.
    length = _interval_max(0, _interval_min(right, high) - _interval_max(left, low))
    return _interval_min(length, high - low)


def _convex_hull(points: Iterable[tuple[Fraction, Fraction]]) -> tuple:
    """Exact monotone-chain hull, CCW without repeated/collinear boundary points."""
    points = sorted(set(points))
    if len(points) <= 1:
        return tuple(points)

    def turn(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _clip_linear(polygon: tuple, a: Fraction, b: Fraction, bound: Fraction) -> tuple:
    """Clip to the closed half-plane a*x+b*y >= bound, with rational intersections."""
    if not polygon:
        return ()
    result = []
    previous = polygon[-1]
    previous_value = a * previous[0] + b * previous[1] - bound
    for current in polygon:
        current_value = a * current[0] + b * current[1] - bound
        if (current_value >= 0) != (previous_value >= 0):
            parameter = previous_value / (previous_value - current_value)
            result.append((previous[0] + parameter * (current[0] - previous[0]),
                           previous[1] + parameter * (current[1] - previous[1])))
        if current_value >= 0:
            result.append(current)
        previous, previous_value = current, current_value
    return tuple(result)


def _dilate_erode_triangle(nominal: tuple, rho: Fraction) -> tuple[tuple, tuple]:
    """Return K0 plus/minus the closed L-infinity radius-rho square.

    Outer is the convex hull of the 12 vertex-plus-square-corner sums. Inner
    moves each CCW triangle edge inward by rho*(abs(A)+abs(B)) in its defining
    inequality A*x+B*y >= c. A collapsed/empty inner set has zero area.
    """
    hull = _convex_hull(nominal)
    corners = ((-rho, -rho), (-rho, rho), (rho, -rho), (rho, rho))
    outer = _convex_hull((p[0] + dx, p[1] + dy) for p in hull for dx, dy in corners)
    if len(hull) < 3 or polygon_area(hull) == 0:
        return outer, ()
    inner = hull
    for first, second in zip(hull, hull[1:] + hull[:1]):
        a, b = -(second[1] - first[1]), second[0] - first[0]
        bound = a * first[0] + b * first[1] + rho * (abs(a) + abs(b))
        inner = _clip_linear(inner, a, b, bound)
    return outer, inner


@dataclass(frozen=True)
class Model:
    kind: str = "lovelace_triangle"
    camera_radius: Fraction = Q(0)
    asset_path: str | Path | None = None
    _vertices: tuple = field(init=False, repr=False, default=())
    _asset: Path | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.kind not in ("lovelace_triangle", "rectangle"):
            raise ValueError("Model.kind must be lovelace_triangle or rectangle")
        radius = _q(self.camera_radius, "camera_radius")
        if radius < 0:
            raise ValueError("camera_radius must be nonnegative")
        object.__setattr__(self, "camera_radius", radius)
        if self.kind == "lovelace_triangle":
            copied_asset = Path(__file__).parent / "assets" / "lovelace_patch.json"
            asset = Path(self.asset_path) if self.asset_path is not None else (
                copied_asset if copied_asset.exists() else DEFAULT_ASSET)
            vertices = centered_triangle(asset)
            TriangularWorld("continuous_domain_validation", vertices, DEPTH_MIN)
            object.__setattr__(self, "_vertices", vertices)
            object.__setattr__(self, "_asset", asset)

    @property
    def grid(self) -> ImageGrid:
        return GRID

    @property
    def nominal_cameras(self) -> tuple[Camera, ...]:
        return tuple(Camera(cx, label) for cx, label in zip(NOMINAL_CX, ("left", "reference", "right")))

    @property
    def depth_range(self) -> Pair:
        return DEPTH_MIN, DEPTH_MAX

    @property
    def observation_size(self) -> int:
        return 3 * GRID.width * GRID.height * 3

    def _depth_box(self, low: RationalInput, high: RationalInput) -> Interval:
        box = Interval(_q(low, "depth_lo"), _q(high, "depth_hi"))
        if box.lo < DEPTH_MIN or box.hi > DEPTH_MAX:
            raise ValueError("Depth box must be contained in [540,660] mm")
        return box

    def _offsets(self, offsets: Iterable[RationalInput]) -> tuple[Fraction, ...]:
        errors = tuple(_q(value, "camera offset") for value in offsets)
        if len(errors) != 3:
            raise ValueError("offsets must give errors for left, reference and right cameras")
        if errors[1] != 0:
            raise ValueError("Reference camera gauge is fixed: middle offset must be zero")
        if any(abs(errors[i]) > self.camera_radius for i in (0, 2)):
            raise ValueError("Side-camera offset exceeds camera_radius")
        return errors

    def point_prediction(self, depth: RationalInput,
                          offsets: Iterable[RationalInput] = (Q(0), Q(0), Q(0))) -> tuple[Fraction, ...]:
        """Exact image at an actual depth and additive camera-offset triple.

        Point depths may lie outside the certified [540,660] domain, e.g. for
        scoring an unrepaired nominal baseline. All vertices must still have
        positive Z. These points do not enlarge the enclosure's certified domain.
        """
        depth = _q(depth, "depth")
        errors = self._offsets(offsets)
        return self._point_prediction_cached(depth, errors)

    @lru_cache(maxsize=1024)
    def _point_prediction_cached(self, depth: Fraction,
                                  errors: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
        cameras = tuple(Camera(cx + error, name) for cx, error, name
                        in zip(NOMINAL_CX, errors, ("left", "reference", "right")))
        if self.kind == "lovelace_triangle":
            world = TriangularWorld(f"lovelace_triangle_z_{depth}", self._vertices, depth)
            return render_triangle(world, cameras, GRID)
        if depth <= 0:
            raise ValueError("Rectangle point depth must be strictly positive")
        result = []
        for camera in cameras:
            left = (-RECTANGLE_HALF_WIDTH - camera.cx) / depth
            right = (RECTANGLE_HALF_WIDTH - camera.cx) / depth
            bottom, top = -RECTANGLE_HALF_HEIGHT / depth, RECTANGLE_HALF_HEIGHT / depth
            for row in range(GRID.height):
                for column in range(GRID.width):
                    u0, u1, v0, v1 = GRID.pixel_bounds(column, row)
                    covered = _overlap(left, right, u0, u1) * _overlap(bottom, top, v0, v1)
                    gray = 1 - covered / ((u1 - u0) * (v1 - v0))
                    result.extend((gray,) * 3)
        return tuple(result)

    def _camera_box(self, index: int) -> Interval:
        radius = Q(0) if index == 1 else self.camera_radius
        return Interval(NOMINAL_CX[index] - radius, NOMINAL_CX[index] + radius)

    @lru_cache(maxsize=1024)
    def _triangle_camera_data(self, depth: Interval, index: int) -> dict:
        camera_box = self._camera_box(index)
        projected_boxes = []
        for x, y, z in self._vertices:
            denominator = depth + z
            if denominator.lo <= 0:
                raise ValueError("Projected triangle Z denominator must stay strictly positive")
            projected_boxes.append(((Interval(x) - camera_box) / denominator,
                                    Interval(y) / denominator))
        projected_boxes = tuple(projected_boxes)
        nominal = tuple(((x - NOMINAL_CX[index]) / (depth.midpoint + z),
                          y / (depth.midpoint + z)) for x, y, z in self._vertices)
        rho = max(max(abs(box.lo - coordinate), abs(box.hi - coordinate))
                  for vertex_boxes, vertex in zip(projected_boxes, nominal)
                  for box, coordinate in zip(vertex_boxes, vertex))
        perimeter_l1 = sum((abs(a[0] - b[0]) + abs(a[1] - b[1])
                            for a, b in zip(nominal, nominal[1:] + nominal[:1])), Q(0))
        area_bound = 2 * perimeter_l1 * rho + 20 * rho * rho
        outer, inner = _dilate_erode_triangle(nominal, rho)
        hull_box = (min(p[0].lo for p in projected_boxes), max(p[0].hi for p in projected_boxes),
                    min(p[1].lo for p in projected_boxes), max(p[1].hi for p in projected_boxes))
        return {"vertex_boxes": projected_boxes, "nominal_vertices": nominal,
                "rho": rho, "perimeter_l1": perimeter_l1,
                "symmetric_difference_area_bound": area_bound, "hull_box": hull_box,
                "outer_polygon": outer, "inner_polygon": inner}

    def _triangle_enclosure(self, depth: Interval) -> tuple[Pair, ...]:
        result = []
        for camera_index in range(3):
            data = self._triangle_camera_data(depth, camera_index)
            xlow, xhigh, ylow, yhigh = data["hull_box"]
            for row in range(GRID.height):
                for column in range(GRID.width):
                    u0, u1, v0, v1 = GRID.pixel_bounds(column, row)
                    # Every possible triangle is inside this hull bounding box.
                    # Boundary-only contact contributes zero area.
                    outside = xhigh <= u0 or xlow >= u1 or yhigh <= v0 or ylow >= v1
                    if outside:
                        color = (Q(1), Q(1))
                    else:
                        pixel_area = (u1 - u0) * (v1 - v0)
                        upper_area = polygon_area(clip_polygon(data["outer_polygon"], (u0, u1, v0, v1)))
                        lower_area = polygon_area(clip_polygon(data["inner_polygon"], (u0, u1, v0, v1)))
                        if not 0 <= lower_area <= upper_area <= pixel_area:
                            raise ArithmeticError("Invalid exact inner/outer pixel-area bounds")
                        color = (1 - upper_area / pixel_area, 1 - lower_area / pixel_area)
                    result.extend((color,) * 3)
        return tuple(result)

    def _rectangle_enclosure(self, depth: Interval) -> tuple[Pair, ...]:
        if depth.lo <= 0:
            raise ValueError("Rectangle Z denominator must stay strictly positive")
        result = []
        for camera_index in range(3):
            camera = self._camera_box(camera_index)
            left = (-RECTANGLE_HALF_WIDTH - camera) / depth
            right = (RECTANGLE_HALF_WIDTH - camera) / depth
            bottom, top = -RECTANGLE_HALF_HEIGHT / depth, RECTANGLE_HALF_HEIGHT / depth
            for row in range(GRID.height):
                for column in range(GRID.width):
                    u0, u1, v0, v1 = GRID.pixel_bounds(column, row)
                    overlap_x = _overlap_interval(left, right, u0, u1)
                    overlap_y = _overlap_interval(bottom, top, v0, v1)
                    coverage = overlap_x * overlap_y / ((u1 - u0) * (v1 - v0))
                    gray = _unit_clamp(1 - coverage).pair()
                    result.extend((gray,) * 3)
        return tuple(result)

    def enclosure(self, depth_lo: RationalInput, depth_hi: RationalInput) -> tuple[Pair, ...]:
        """Contain all predictions for every depth/camera error in the declared box.

        The operation uses interval arithmetic and the geometric area bound;
        it does not render endpoints and interpolate or assume monotone images.
        """
        depth = self._depth_box(depth_lo, depth_hi)
        return self._enclosure_cached(depth)

    @lru_cache(maxsize=2048)
    def _enclosure_cached(self, depth: Interval) -> tuple[Pair, ...]:
        if self.kind == "lovelace_triangle":
            return self._triangle_enclosure(depth)
        return self._rectangle_enclosure(depth)

    def enclosure_details(self, depth_lo: RationalInput, depth_hi: RationalInput) -> dict:
        """Serializable bound ingredients for independent audits, not extra samples."""
        depth = self._depth_box(depth_lo, depth_hi)
        details = {"depth_box": [str(depth.lo), str(depth.hi)], "kind": self.kind,
                   "nominal_depth_mm": str(depth.midpoint),
                   "camera_radius_mm": str(self.camera_radius), "reference_offset_mm": "0"}
        if self.kind == "lovelace_triangle":
            details["method"] = "exact convex outer Minkowski square dilation and inner half-plane erosion; per-pixel clipped area"
            details["inclusion"] = "K0 eroded by L-infinity ball is contained in every K; every K lies in dilated K0"
            details["triangle_cameras"] = []
            for index in range(3):
                data = self._triangle_camera_data(depth, index)
                details["triangle_cameras"].append({
                    "nominal_cx_mm": str(NOMINAL_CX[index]),
                    "camera_cx_box_mm": [str(c) for c in self._camera_box(index).pair()],
                    "nominal_projected_vertices": [[str(c) for c in p] for p in data["nominal_vertices"]],
                    "rho_linf": str(data["rho"]), "nominal_perimeter_l1": str(data["perimeter_l1"]),
                    "symmetric_difference_area_bound": str(data["symmetric_difference_area_bound"]),
                    "outer_polygon": [[str(c) for c in p] for p in data["outer_polygon"]],
                    "inner_polygon": [[str(c) for c in p] for p in data["inner_polygon"]],
                    "projected_vertex_intervals": [
                        [[str(c) for c in coordinate.pair()] for coordinate in vertex]
                        for vertex in data["vertex_boxes"]],
                    "projected_hull_box": [str(c) for c in data["hull_box"]]})
        else:
            details["method"] = "rational interval extension of clipped rectangle overlap area"
        return details

    def to_dict(self) -> dict:
        return {"kind": self.kind, "depth_range_mm": [str(DEPTH_MIN), str(DEPTH_MAX)],
                "side_camera_offset_radius_mm": str(self.camera_radius),
                "reference_camera_offset_mm": "0", "camera_nominal_cx_mm": [str(c) for c in NOMINAL_CX],
                "grid": GRID.to_dict(), "material": "constant opaque black, same for every camera",
                "background": "constant white", "observation_layout": "camera,row,column,RGB",
                "triangle_area_bound": "2*nominal_perimeter_L1*rho_Linf+20*rho_Linf^2",
                "triangle_enclosure_method": "exact L-infinity dilation/erosion polygons clipped to each pixel",
                "rectangle_half_dimensions_mm": [str(RECTANGLE_HALF_WIDTH), str(RECTANGLE_HALF_HEIGHT)],
                "source_sha256": M4_SHA256 if self.kind == "lovelace_triangle" else None,
                "source_asset": str(self._asset) if self._asset is not None else None,
                "scope": "isolated M4 seed facet for lovelace_triangle; separate analytic toy for rectangle"}
