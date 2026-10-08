"""An exact single-facet abstraction extracted from the existing LOVELACE M4.

The asset retains a 49-vertex, 74-face local patch as provenance, but this
renderer isolates only its seed triangle. It does not reconstruct or render
the full character, infer a physical truth, or certify a continuous family.
All source float32 coordinates become their exact dyadic rational values.
Pixels average a constant opaque foreground over exact projected area.

No renderer import or third-party package is needed. Cameras provide ``cx``;
grids provide width, height and pixel_bounds(column, row). Observation layout
is camera, increasing-v row, increasing-u column, then RGB, as in renderer.py.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
from typing import Iterable

Q = Fraction
Point2 = tuple[Fraction, Fraction]
Point3 = tuple[Fraction, Fraction, Fraction]
RGB = tuple[Fraction, Fraction, Fraction]
RationalInput = Fraction | int | str
DEFAULT_ASSET = Path(__file__).parent / "assets" / "lovelace_patch.json"
M4_SOURCE = Path("C:/Users/gengr/Downloads/Lovelace/global-search-20260924/arm-M/M4_seed11/mesh_raw.glb")
M4_SHA256 = "fcd1d454f926bc076cbe7a54a0e810ecff5200a8d4c904024490b7483b5bf10a"
SEED_FACE_INDEX = 133336


def _q(value: RationalInput, field: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int, str)):
        raise TypeError(f"{field} must be Fraction, int or rational string; floats are forbidden")
    return Q(value)


def _rgb(values: Iterable[RationalInput], field: str) -> RGB:
    channels = tuple(values)
    if len(channels) != 3:
        raise ValueError(f"{field} must have three channels")
    result = tuple(_q(c, field) for c in channels)
    if any(c < 0 or c > 1 for c in result):
        raise ValueError(f"{field} channels must be in [0, 1]")
    return result


def _point3(values: Iterable[RationalInput], field: str) -> Point3:
    values = tuple(values)
    if len(values) != 3:
        raise ValueError(f"{field} must have three coordinates")
    return tuple(_q(v, field) for v in values)


def _twice_area(polygon: Iterable[Point2]) -> Fraction:
    """Absolute shoelace area times two; either orientation is accepted."""
    points = tuple(polygon)
    if len(points) < 3:
        return Q(0)
    return abs(sum((p[0] * q[1] - q[0] * p[1]
                    for p, q in zip(points, points[1:] + points[:1])), Q(0)))


def polygon_area(polygon: Iterable[Point2]) -> Fraction:
    return _twice_area(polygon) / 2


def _clip_halfplane(polygon: tuple[Point2, ...], axis: int,
                    bound: Fraction, keep_greater: bool) -> tuple[Point2, ...]:
    """Sutherland-Hodgman clipping against one closed axis half-plane."""
    if not polygon:
        return ()

    def inside(point: Point2) -> bool:
        return point[axis] >= bound if keep_greater else point[axis] <= bound

    def intersection(a: Point2, b: Point2) -> Point2:
        # A crossing edge cannot be parallel to this clipping line.
        t = (bound - a[axis]) / (b[axis] - a[axis])
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    result = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                result.append(intersection(previous, current))
            result.append(current)
        elif previous_inside:
            result.append(intersection(previous, current))
        previous, previous_inside = current, current_inside
    return tuple(result)


def clip_polygon(polygon: Iterable[Point2], bounds: Iterable[RationalInput]) -> tuple[Point2, ...]:
    """Clip to (u0, u1, v0, v1), entirely with exact rational arithmetic."""
    box = tuple(_q(value, "clip bounds") for value in bounds)
    if len(box) != 4 or box[0] >= box[1] or box[2] >= box[3]:
        raise ValueError("clip bounds must describe a positive-area rectangle")
    points = tuple(tuple(_q(c, "polygon coordinate") for c in p) for p in polygon)
    if any(len(p) != 2 for p in points):
        raise ValueError("polygon coordinates must have two components")
    for axis, bound, keep_greater in ((0, box[0], True), (0, box[1], False),
                                      (1, box[2], True), (1, box[3], False)):
        points = _clip_halfplane(points, axis, bound, keep_greater)
    return points


@dataclass(frozen=True)
class TriangularWorld:
    name: str
    centered_vertices: tuple[Point3, Point3, Point3]
    translation_z: Fraction
    foreground: RGB = (Q(0), Q(0), Q(0))
    background: RGB = (Q(1), Q(1), Q(1))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("TriangularWorld.name must be a string")
        vertices = tuple(_point3(v, "triangle vertex") for v in self.centered_vertices)
        if len(vertices) != 3:
            raise ValueError("TriangularWorld requires exactly three vertices")
        if any(sum((v[i] for v in vertices), Q(0)) != 0 for i in range(3)):
            raise ValueError("centered_vertices must have exact centroid (0, 0, 0)")
        if polygon_area(tuple((v[0], v[1]) for v in vertices)) == 0:
            raise ValueError("triangle XY projection must be nondegenerate")
        object.__setattr__(self, "centered_vertices", vertices)
        object.__setattr__(self, "translation_z", _q(self.translation_z, "translation_z"))
        object.__setattr__(self, "foreground", _rgb(self.foreground, "foreground"))
        object.__setattr__(self, "background", _rgb(self.background, "background"))
        if any(v[2] <= 0 for v in self.vertices):
            raise ValueError("all translated triangle vertices must lie at positive Z")

    @property
    def vertices(self) -> tuple[Point3, Point3, Point3]:
        return tuple((x, y, z + self.translation_z) for x, y, z in self.centered_vertices)

    @property
    def target_z(self) -> Fraction:
        # The reference +Z ray at x=y=0 passes through the triangle centroid.
        return self.translation_z

    def to_dict(self) -> dict:
        return {"name": self.name, "model": "isolated LOVELACE M4 seed triangle",
                "centered_vertices": [[str(c) for c in v] for v in self.centered_vertices],
                "translation_z": str(self.translation_z),
                "vertices": [[str(c) for c in v] for v in self.vertices],
                "foreground": [str(c) for c in self.foreground],
                "background": [str(c) for c in self.background],
                "opaque": True, "double_sided": True,
                "target": "reference-ray Z equals translated centroid Z",
                "units": "millimetres in the declared abstract world"}


def centered_triangle(asset_path: str | Path | None = None) -> tuple[Point3, Point3, Point3]:
    """Read the scaled rational triangle; the original GLB need not be installed."""
    asset = json.loads(Path(asset_path or DEFAULT_ASSET).read_text(encoding="utf-8"))
    if asset["source"]["sha256"] != M4_SHA256 or asset["source"]["seed_face_index"] != SEED_FACE_INDEX:
        raise ValueError("Unexpected source provenance in the triangle asset")
    if asset["schema"] != "lovelace-m4-exact-local-patch/1":
        raise ValueError("Unsupported LOVELACE patch schema")
    vertices = tuple(_point3(v, "asset scaled vertex")
                     for v in asset["abstraction"]["scaled_centered_seed_vertices"])
    if len(vertices) != 3:
        raise ValueError("asset must contain exactly three seed vertices")
    return vertices


def lovelace_world(depth: RationalInput, name: str | None = None,
                   foreground: Iterable[RationalInput] = (0, 0, 0),
                   background: Iterable[RationalInput] = (1, 1, 1),
                   asset_path: str | Path | None = None) -> TriangularWorld:
    depth = _q(depth, "depth")
    return TriangularWorld(name if name is not None else f"lovelace_triangle_z_{depth}",
                           centered_triangle(asset_path), depth, tuple(foreground), tuple(background))


def evaluation_target_z(world: TriangularWorld) -> Fraction:
    """Library/evaluation registration; do not reveal the actual world ID to a solver."""
    if not isinstance(world, TriangularWorld):
        raise TypeError("evaluation_target_z expects TriangularWorld")
    return world.target_z


def render_triangle(world: TriangularWorld, cameras: Iterable, grid) -> tuple[Fraction, ...]:
    """Exact pinhole area-averaged RGB observation of one opaque, double-sided facet.

    Every camera center is (cx,0,0), oriented along +Z with unit focal scale.
    Vertices project as ((x-cx)/z, y/z). If a camera sees the triangle exactly
    edge-on, its projected area and contribution are zero. There are no other
    surfaces, shadows, texture or lighting; background is constant.
    """
    if not isinstance(world, TriangularWorld):
        raise TypeError("render_triangle expects TriangularWorld")
    cameras = tuple(cameras)
    if not cameras:
        raise ValueError("render_triangle requires at least one camera")
    for field in ("width", "height"):
        value = getattr(grid, field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"grid.{field} must be a positive integer")
    result = []
    for camera in cameras:
        cx = _q(camera.cx, "camera.cx")
        projected = tuple(((x - cx) / z, y / z) for x, y, z in world.vertices)
        for row in range(grid.height):
            for column in range(grid.width):
                bounds = tuple(_q(v, "pixel bounds") for v in grid.pixel_bounds(column, row))
                if len(bounds) != 4:
                    raise ValueError("pixel_bounds must return (u0,u1,v0,v1)")
                u0, u1, v0, v1 = bounds
                if u0 >= u1 or v0 >= v1:
                    raise ValueError("pixel bounds must have positive area")
                clipped = clip_polygon(projected, bounds)
                coverage = polygon_area(clipped) / ((u1 - u0) * (v1 - v0))
                if not 0 <= coverage <= 1:
                    raise ArithmeticError("Exact clipped coverage outside [0,1]")
                result.extend(coverage * fg + (1 - coverage) * bg
                              for fg, bg in zip(world.foreground, world.background))
    return tuple(result)


def _read_glb(source: Path) -> tuple[dict, bytes, bytes]:
    raw = source.read_bytes()
    if len(raw) < 12:
        raise ValueError("Truncated GLB")
    magic, version, total = struct.unpack_from("<4sII", raw)
    if magic != b"glTF" or version != 2 or total != len(raw):
        raise ValueError("Expected complete GLB 2 file")
    chunks = {}
    offset = 12
    while offset < total:
        if offset + 8 > total:
            raise ValueError("Truncated GLB chunk header")
        size, kind = struct.unpack_from("<I4s", raw, offset)
        offset += 8
        if offset + size > total or kind in chunks:
            raise ValueError("Truncated or duplicate GLB chunk")
        chunks[kind] = raw[offset:offset + size]
        offset += size
    doc = json.loads(chunks[b"JSON"].decode("utf-8"))
    if len(doc.get("buffers", [])) != 1 or "uri" in doc["buffers"][0]:
        raise ValueError("Expected a single embedded GLB buffer")
    if any(any(key in node for key in ("matrix", "translation", "rotation", "scale"))
           for node in doc.get("nodes", [])):
        raise ValueError("This extractor expects the frozen M4's identity nodes")
    return doc, chunks[b"BIN\x00"], raw


def _read_accessor(doc: dict, binary: bytes, index: int) -> tuple:
    accessor = doc["accessors"][index]
    if accessor.get("sparse") or accessor.get("normalized"):
        raise ValueError("Sparse/normalized accessors are outside this frozen extractor")
    view = doc["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise ValueError("Accessor uses an unsupported buffer")
    code = {5121: "B", 5123: "H", 5125: "I", 5126: "f"}[accessor["componentType"]]
    width = {"SCALAR": 1, "VEC3": 3}[accessor["type"]]
    fmt = "<" + code * width
    item_size = struct.calcsize(fmt)
    stride = view.get("byteStride", item_size)
    if stride < item_size:
        raise ValueError("Invalid accessor byte stride")
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    if count and start + (count - 1) * stride + item_size > view.get("byteOffset", 0) + view["byteLength"]:
        raise ValueError("Accessor escapes its buffer view")
    return tuple(struct.unpack_from(fmt, binary, start + i * stride) for i in range(count))


def build_patch_asset(source: str | Path = M4_SOURCE,
                      destination: str | Path = DEFAULT_ASSET) -> dict:
    """Create only the declared JSON asset from the frozen M4, using stdlib only."""
    source, destination = Path(source), Path(destination)
    doc, binary, raw = _read_glb(source)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != M4_SHA256:
        raise ValueError("Original LOVELACE M4 hash differs from the frozen source")
    primitive = doc["meshes"][0]["primitives"][0]
    if primitive.get("mode", 4) != 4:
        raise ValueError("Expected a triangular primitive")
    position = doc["accessors"][primitive["attributes"]["POSITION"]]
    if position["componentType"] != 5126 or position["type"] != "VEC3":
        raise ValueError("Expected source float32 VEC3 positions")
    vertices_float = _read_accessor(doc, binary, primitive["attributes"]["POSITION"])
    flat_indices = tuple(value[0] for value in _read_accessor(doc, binary, primitive["indices"]))
    if len(flat_indices) % 3 or any(i >= len(vertices_float) for i in flat_indices):
        raise ValueError("Invalid triangular index buffer")
    faces = tuple(tuple(flat_indices[i:i + 3]) for i in range(0, len(flat_indices), 3))
    exact_cache = {}

    def exact_vertex(index: int) -> Point3:
        if index not in exact_cache:
            exact_cache[index] = tuple(Q(*value.as_integer_ratio()) for value in vertices_float[index])
        return exact_cache[index]

    def nonzero_face(index: int) -> bool:
        a, b, c = (exact_vertex(i) for i in faces[index])
        ab, ac = tuple(b[i] - a[i] for i in range(3)), tuple(c[i] - a[i] for i in range(3))
        return any(ab[i] * ac[j] != ab[j] * ac[i] for i, j in ((0, 1), (0, 2), (1, 2)))

    adjacent = [[] for _ in vertices_float]
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            adjacent[vertex_index].append(face_index)
    included = {SEED_FACE_INDEX}
    for _ in range(3):
        patch_vertices = {i for face_index in included for i in faces[face_index]}
        candidates = {face_index for i in patch_vertices for face_index in adjacent[i]}
        included.update(face_index for face_index in candidates if nonzero_face(face_index))
    face_ids = sorted(included)
    vertex_ids = sorted({i for face_index in face_ids for i in faces[face_index]})
    if (len(vertex_ids), len(face_ids)) != (49, 74):
        raise ValueError("Frozen source patch topology differs from 49 vertices / 74 faces")
    local_index = {old: new for new, old in enumerate(vertex_ids)}
    patch_faces = [[local_index[i] for i in faces[index]] for index in face_ids]
    seed_ids = faces[SEED_FACE_INDEX]
    seed = tuple(exact_vertex(i) for i in seed_ids)
    centroid = tuple(sum((v[axis] for v in seed), Q(0)) / 3 for axis in range(3))
    centered = tuple(tuple(v[i] - centroid[i] for i in range(3)) for v in seed)
    radius_xy = max(abs(v[i]) for v in centered for i in (0, 1))
    scale = Q(20) / radius_xy
    scaled = tuple(tuple(c * scale for c in v) for v in centered)
    # Validate the declared world and its positive-Z reference-ray target.
    TriangularWorld("asset_validation_z_540", scaled, Q(540))
    asset = {
        "schema": "lovelace-m4-exact-local-patch/1",
        "source": {"path": source.as_posix(), "sha256": digest, "bytes": len(raw),
                   "format": "GLB 2", "mesh_index": 0, "primitive_index": 0,
                   "vertex_count": len(vertices_float), "triangle_count": len(faces),
                   "seed_face_index": SEED_FACE_INDEX,
                   "seed_original_vertex_indices": list(seed_ids),
                   "seed_vertices_exact": [[str(c) for c in v] for v in seed],
                   "coordinate_rule": "original GLB POSITION float32 via as_integer_ratio; no axis remap"},
        "patch": {"selection": "seed face 133336; three vertex-adjacency expansion rounds; exact zero-area faces excluded",
                  "vertex_count": len(vertex_ids), "triangle_count": len(face_ids),
                  "original_vertex_indices": vertex_ids, "original_face_indices": face_ids,
                  "vertices_exact": [[str(c) for c in exact_vertex(i)] for i in vertex_ids],
                  "triangles_local_indices": patch_faces},
        "abstraction": {"scope": "single seed facet isolated from archived LOVELACE M4 local patch",
                        "omitted": "all other faces, full-character occlusion, texture, rig, lighting and physical calibration",
                        "not_claimed": "complete LOVELACE reconstruction, accurate character identity or true physical geometry",
                        "units": "declared abstract-world millimetres; source units are not physical millimetres",
                        "source_seed_centroid_exact": [str(c) for c in centroid],
                        "uniform_scale_exact": str(scale), "max_abs_centered_xy_mm": "20",
                        "scaled_centered_seed_vertices": [[str(c) for c in v] for v in scaled],
                        "transform": "world vertex = scale * (source seed vertex - source seed centroid) + (0,0,translation_z)",
                        "suggested_translation_z_mm": ["540", "600", "660"],
                        "material": "constant opaque double-sided foreground shared by all cameras",
                        "reference_target": "reference camera cx=0 +Z ray passes through triangle centroid; t=translation_z",
                        "pixel_rule": "exact pinhole projection, Sutherland-Hodgman rectangle clipping, shoelace area average"}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return asset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the exact LOVELACE M4 local patch provenance asset")
    parser.add_argument("--source", type=Path, default=M4_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_ASSET)
    options = parser.parse_args()
    asset = build_patch_asset(options.source, options.destination)
    print(json.dumps({"asset": str(options.destination), "source_sha256": asset["source"]["sha256"],
                      "patch_vertices": asset["patch"]["vertex_count"],
                      "patch_triangles": asset["patch"]["triangle_count"]}))
