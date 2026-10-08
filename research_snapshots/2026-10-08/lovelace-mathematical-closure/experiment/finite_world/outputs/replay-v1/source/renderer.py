"""Exact finite-world pinhole renderer using only Python's standard library.

The model has parallel +Z pinhole cameras, opaque axis-aligned rectangular
surfaces and world-coordinate solid/checker materials. Pixel values are exact
area averages in normalized image coordinates. This certifies this declared
finite model, not a continuous scene family or a physical image pipeline.

Observation layout: camera, pixel row (increasing v), pixel column (increasing
u), then RGB. Geometry, colors and arithmetic are rational; float input is
rejected. Surface boundaries are closed and checker tiles use floor indexing;
these conventions only affect measure-zero pixel boundaries.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

Q = Fraction
RGB = tuple[Fraction, Fraction, Fraction]
RationalInput = Fraction | int | str


def _q(value: RationalInput, field: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int, str)):
        raise TypeError(f"{field} must be Fraction, int or rational string; floats are forbidden")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{field} is not a valid rational number") from exc


def _rgb(value: Iterable[RationalInput], field: str) -> RGB:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field} must contain exactly three rational channels") from exc
    if len(items) != 3:
        raise ValueError(f"{field} must contain exactly three channels")
    result = tuple(_q(channel, f"{field}[{i}]") for i, channel in enumerate(items))
    if any(channel < 0 or channel > 1 for channel in result):
        raise ValueError(f"{field} channels must be in [0, 1]")
    return result  # type: ignore[return-value]


def _name(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


@dataclass(frozen=True)
class Camera:
    """Center (cx, 0, 0), unit focal scale, fixed orientation toward +Z."""

    cx: Fraction = Q(0)
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "cx", _q(self.cx, "Camera.cx"))
        _name(self.name, "Camera.name")

    def to_dict(self) -> dict:
        return {"name": self.name, "center": [str(self.cx), "0", "0"],
                "direction": ["0", "0", "1"], "projection": "u=(x-cx)/z; v=y/z"}


@dataclass(frozen=True)
class ImageGrid:
    width: int = 8
    height: int = 6
    u_min: Fraction = Q(-1, 2)
    u_max: Fraction = Q(1, 2)
    v_min: Fraction = Q(-3, 8)
    v_max: Fraction = Q(3, 8)

    def __post_init__(self) -> None:
        for field in ("width", "height"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"ImageGrid.{field} must be a positive integer")
        for field in ("u_min", "u_max", "v_min", "v_max"):
            object.__setattr__(self, field, _q(getattr(self, field), f"ImageGrid.{field}"))
        if self.u_min >= self.u_max or self.v_min >= self.v_max:
            raise ValueError("ImageGrid bounds must have positive area")

    def pixel_bounds(self, column: int, row: int) -> tuple[Fraction, ...]:
        if (isinstance(column, bool) or isinstance(row, bool)
                or not isinstance(column, int) or not isinstance(row, int)):
            raise TypeError("Pixel indices must be integers")
        if not (0 <= column < self.width and 0 <= row < self.height):
            raise IndexError("Pixel index outside ImageGrid")
        du = (self.u_max - self.u_min) / self.width
        dv = (self.v_max - self.v_min) / self.height
        return (self.u_min + column * du, self.u_min + (column + 1) * du,
                self.v_min + row * dv, self.v_min + (row + 1) * dv)

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height,
                "u_min": str(self.u_min), "u_max": str(self.u_max),
                "v_min": str(self.v_min), "v_max": str(self.v_max),
                "pixel_measure": "uniform normalized-image area", "row_order": "increasing v"}


@dataclass(frozen=True)
class Material:
    """A single physical material rule, evaluated in world XY for every view."""

    kind: str = "solid"
    color0: RGB = (Q(0), Q(0), Q(0))
    color1: RGB = (Q(1), Q(1), Q(1))
    tile_size: Fraction = Q(1)
    origin_x: Fraction = Q(0)
    origin_y: Fraction = Q(0)

    def __post_init__(self) -> None:
        if self.kind not in ("solid", "checker"):
            raise ValueError("Material.kind must be 'solid' or 'checker'")
        for field in ("color0", "color1"):
            object.__setattr__(self, field, _rgb(getattr(self, field), f"Material.{field}"))
        for field in ("tile_size", "origin_x", "origin_y"):
            object.__setattr__(self, field, _q(getattr(self, field), f"Material.{field}"))
        if self.tile_size <= 0:
            raise ValueError("Material.tile_size must be positive")

    def sample(self, x: RationalInput, y: RationalInput) -> RGB:
        x, y = _q(x, "material sample x"), _q(y, "material sample y")
        if self.kind == "solid":
            return self.color0
        fx, fy = (x - self.origin_x) / self.tile_size, (y - self.origin_y) / self.tile_size
        ix, iy = fx.numerator // fx.denominator, fy.numerator // fy.denominator
        return self.color0 if (ix + iy) % 2 == 0 else self.color1

    def to_dict(self) -> dict:
        return {"kind": self.kind, "color0": [str(c) for c in self.color0],
                "color1": [str(c) for c in self.color1], "tile_size": str(self.tile_size),
                "origin_x": str(self.origin_x), "origin_y": str(self.origin_y),
                "coordinates": "world XY shared across views",
                "checker_rule": "(floor((x-origin_x)/tile_size)+floor((y-origin_y)/tile_size)) mod 2"}


@dataclass(frozen=True)
class Surface:
    z: Fraction
    x_min: Fraction
    x_max: Fraction
    y_min: Fraction
    y_max: Fraction
    material: Material
    label: str = ""

    def __post_init__(self) -> None:
        for field in ("z", "x_min", "x_max", "y_min", "y_max"):
            object.__setattr__(self, field, _q(getattr(self, field), f"Surface.{field}"))
        if self.z <= 0 or self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Surface requires positive Z and a positive-area rectangle")
        if not isinstance(self.material, Material):
            raise TypeError("Surface.material must be a Material")
        _name(self.label, "Surface.label")

    def contains(self, x: Fraction, y: Fraction) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def to_dict(self) -> dict:
        return {"label": self.label, "z": str(self.z), "x_min": str(self.x_min),
                "x_max": str(self.x_max), "y_min": str(self.y_min),
                "y_max": str(self.y_max), "opaque": True, "material": self.material.to_dict()}


@dataclass(frozen=True)
class World:
    name: str
    surfaces: tuple[Surface, ...]
    background: RGB = (Q(1), Q(1), Q(1))

    def __post_init__(self) -> None:
        _name(self.name, "World.name")
        surfaces = tuple(self.surfaces)
        if not surfaces or not all(isinstance(s, Surface) for s in surfaces):
            raise ValueError("World requires at least one Surface")
        if len({s.z for s in surfaces}) != len(surfaces):
            raise ValueError("World surfaces must have distinct Z; coplanar overlap is excluded")
        object.__setattr__(self, "surfaces", tuple(sorted(surfaces, key=lambda s: s.z)))
        object.__setattr__(self, "background", _rgb(self.background, "World.background"))

    def to_dict(self) -> dict:
        return {"name": self.name, "surfaces": [s.to_dict() for s in self.surfaces],
                "background": [str(c) for c in self.background],
                "visibility": "nearest positive-Z opaque rectangle hit"}


DEFAULT_CAMERAS = (Camera(Q(-1), "left"), Camera(Q(0), "reference"), Camera(Q(1), "right"))


def _tile_boundaries(lo: Fraction, hi: Fraction, origin: Fraction,
                     tile_size: Fraction) -> tuple[Fraction, ...]:
    first = (lo - origin) / tile_size
    last = (hi - origin) / tile_size
    start = first.numerator // first.denominator + 1
    # ceil(last)-1 excludes a tile edge exactly on the rectangle's upper edge.
    stop = -((-last.numerator) // last.denominator) - 1
    return tuple(origin + k * tile_size for k in range(start, stop + 1))


def _projection_boundaries(world: World, camera: Camera) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    u_edges, v_edges = set(), set()
    for surface in world.surfaces:
        x_edges = (surface.x_min, surface.x_max)
        y_edges = (surface.y_min, surface.y_max)
        if surface.material.kind == "checker":
            material = surface.material
            x_edges += _tile_boundaries(surface.x_min, surface.x_max,
                                        material.origin_x, material.tile_size)
            y_edges += _tile_boundaries(surface.y_min, surface.y_max,
                                        material.origin_y, material.tile_size)
        u_edges.update((x - camera.cx) / surface.z for x in x_edges)
        v_edges.update(y / surface.z for y in y_edges)
    return tuple(sorted(u_edges)), tuple(sorted(v_edges))


def _ray_color(world: World, camera: Camera, u: Fraction, v: Fraction) -> RGB:
    for surface in world.surfaces:  # World canonicalizes increasing positive Z.
        x, y = camera.cx + surface.z * u, surface.z * v
        if surface.contains(x, y):
            return surface.material.sample(x, y)
    return world.background


def render_image(world: World, camera: Camera, grid: ImageGrid = ImageGrid()) -> tuple[Fraction, ...]:
    """Return an exact row-major RGB image, flattened to immutable Fractions."""
    if not isinstance(world, World) or not isinstance(camera, Camera) or not isinstance(grid, ImageGrid):
        raise TypeError("render_image expects World, Camera and ImageGrid")
    u_edges, v_edges = _projection_boundaries(world, camera)
    result = []
    for row in range(grid.height):
        for column in range(grid.width):
            u0, u1, v0, v1 = grid.pixel_bounds(column, row)
            us = (u0,) + tuple(edge for edge in u_edges if u0 < edge < u1) + (u1,)
            vs = (v0,) + tuple(edge for edge in v_edges if v0 < edge < v1) + (v1,)
            total = [Q(0), Q(0), Q(0)]
            for low_v, high_v in zip(vs, vs[1:]):
                for low_u, high_u in zip(us, us[1:]):
                    # Visibility and material are constant on this open cell.
                    color = _ray_color(world, camera, (low_u + high_u) / 2,
                                       (low_v + high_v) / 2)
                    area = (high_u - low_u) * (high_v - low_v)
                    for channel in range(3):
                        total[channel] += area * color[channel]
            pixel_area = (u1 - u0) * (v1 - v0)
            result.extend(value / pixel_area for value in total)
    return tuple(result)


def render(world: World, cameras: Iterable[Camera] = DEFAULT_CAMERAS,
           grid: ImageGrid = ImageGrid()) -> tuple[Fraction, ...]:
    """Concatenate exact observations from the same world and material rules."""
    cameras = tuple(cameras)
    if not cameras or not all(isinstance(camera, Camera) for camera in cameras):
        raise ValueError("render requires at least one Camera")
    return tuple(value for camera in cameras for value in render_image(world, camera, grid))


def evaluation_target_z(world: World) -> Fraction:
    """Evaluation/library registration only: first hit of x=y=0, cx=0 ray.

    This helper reads geometry. A solver should consume the public candidate
    library and observations without receiving the actual world's identity.
    The reference ray is unit +Z, so its ray distance equals camera-axis Z.
    """
    if not isinstance(world, World):
        raise TypeError("evaluation_target_z expects World")
    for surface in world.surfaces:
        if surface.contains(Q(0), Q(0)):
            return surface.z
    raise ValueError("Reference ray has no target hit in this world")


def demo_worlds() -> tuple[World, ...]:
    """Small examples; the last two have exactly identical default observations."""
    checker = Material("checker", (0, 0, 0), (1, 1, 1), Q(1))
    plain = Material("solid", (Q(1, 3), Q(1, 2), Q(2, 3)))
    red = Material("solid", (1, 0, 0))
    return (
        World("single_checker_6", (Surface(6, -4, 4, -3, 3, checker, "surface"),)),
        World("single_checker_8", (Surface(8, -4, 4, -3, 3, checker, "surface"),)),
        World("two_layers", (Surface(6, -1, 1, -2, 2, red, "front"),
                              Surface(8, -4, 4, -3, 3, checker, "rear"))),
        World("same_color_6", (Surface(6, -10, 10, -10, 10, plain, "surface"),)),
        World("same_color_8", (Surface(8, -10, 10, -10, 10, plain, "surface"),)),
    )
