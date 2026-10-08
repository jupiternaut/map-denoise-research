"""Hand-computable checks for the exact isolated LOVELACE triangle renderer."""

from fractions import Fraction as Q
import hashlib
import json

from mesh_patch import (DEFAULT_ASSET, M4_SHA256, TriangularWorld, centered_triangle,
                        clip_polygon, evaluation_target_z, lovelace_world,
                        polygon_area, render_triangle)
from renderer import Camera, ImageGrid


def main() -> None:
    triangle = ((Q(0), Q(0)), (Q(2), Q(0)), (Q(0), Q(2)))
    checks = 0

    def check(condition, label):
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1

    check(polygon_area(triangle) == 2, "shoelace full triangle")
    check(polygon_area(clip_polygon(triangle, (0, 1, 0, 1))) == 1,
          "unit square entirely covered up to a boundary")
    check(polygon_area(clip_polygon(triangle, (1, 2, 0, 1))) == Q(1, 2),
          "half-square triangular coverage")
    check(polygon_area(clip_polygon(triangle, (3, 4, 3, 4))) == 0,
          "disjoint clipping")
    check(polygon_area(clip_polygon(triangle, (2, 3, 0, 1))) == 0,
          "single-point boundary contact has zero area")
    check(polygon_area(clip_polygon(tuple(reversed(triangle)), (1, 2, 0, 1))) == Q(1, 2),
          "orientation does not alter double-sided coverage")
    check(polygon_area(clip_polygon(triangle, (-1, 3, -1, 3))) == 2,
          "clipping preserves fully contained area")

    vertices = ((Q(-1, 3), Q(-1, 3), Q(0)),
                (Q(2, 3), Q(-1, 3), Q(0)),
                (Q(-1, 3), Q(2, 3), Q(0)))
    grid = ImageGrid(1, 1, Q(-1, 3), Q(2, 3), Q(-1, 3), Q(2, 3))
    world = TriangularWorld("hand_half_pixel", vertices, Q(1))
    near = render_triangle(world, (Camera(0),), grid)
    check(near == (Q(1, 2),) * 3, "half-pixel area mix equals one-half gray")
    farther = render_triangle(TriangularWorld("hand_eighth_pixel", vertices, Q(2)),
                              (Camera(0),), grid)
    check(farther == (Q(7, 8),) * 3, "doubling Z quarters projected area")
    check(render_triangle(world, (Camera(0), Camera(100)), grid) == near + (Q(1),) * 3,
          "camera-major observation and off-image background")
    reverse_world = TriangularWorld("reverse_winding", tuple(reversed(vertices)), Q(1))
    check(render_triangle(reverse_world, (Camera(0),), grid) == near,
          "renderer is double-sided")
    try:
        TriangularWorld("behind_camera", vertices, Q(-1))
    except ValueError:
        check(True, "nonpositive vertex depth rejected")
    else:
        check(False, "nonpositive vertex depth rejected")
    try:
        TriangularWorld("float_depth", vertices, 1.0)
    except TypeError:
        check(True, "float world depth rejected")
    else:
        check(False, "float world depth rejected")

    asset = json.loads(DEFAULT_ASSET.read_text(encoding="utf-8"))
    actual = centered_triangle()
    check(asset["source"]["sha256"] == M4_SHA256, "source provenance hash")
    check(asset["patch"]["vertex_count"] == 49 and asset["patch"]["triangle_count"] == 74,
          "local patch topology")
    check(all(sum((v[i] for v in actual), Q(0)) == 0 for i in range(3)),
          "asset triangle exact centroid zero")
    check(max(abs(v[i]) for v in actual for i in (0, 1)) == 20,
          "asset exact uniform scaling")
    source_seed = tuple(tuple(Q(c) for c in v) for v in asset["source"]["seed_vertices_exact"])
    centroid = tuple(Q(c) for c in asset["abstraction"]["source_seed_centroid_exact"])
    scale = Q(asset["abstraction"]["uniform_scale_exact"])
    check(actual == tuple(tuple((v[i] - centroid[i]) * scale for i in range(3))
                          for v in source_seed), "asset source-to-abstraction transform")

    cameras = (Camera(-60, "left"), Camera(0, "reference"), Camera(60, "right"))
    grid = ImageGrid(8, 6, Q(-1, 5), Q(1, 5), Q(-3, 20), Q(3, 20))
    worlds = tuple(lovelace_world(Q(z)) for z in (540, 600, 660))
    observations = tuple(render_triangle(w, cameras, grid) for w in worlds)
    check(all(evaluation_target_z(w) == w.translation_z for w in worlds),
          "centroid reference-ray target")
    check(all(len(y) == 3 * grid.width * grid.height * 3 for y in observations),
          "actual asset observation shape")
    check(all(isinstance(v, Q) and 0 <= v <= 1 for y in observations for v in y),
          "actual asset channels are exact bounded rationals")
    check(len(set(observations)) == 3, "three smoke states differ for this declared grid")
    print(json.dumps({"status": "passed", "checks": checks,
                      "asset_sha256": hashlib.sha256(DEFAULT_ASSET.read_bytes()).hexdigest(),
                      "source_sha256": M4_SHA256,
                      "patch_vertices": 49, "patch_triangles": 74,
                      "rendered_triangles_per_world": 1,
                      "sample_depths_mm": [540, 600, 660],
                      "scope": "hand-checks and finite sample consistency; no continuous-family or full-character claim"}))


if __name__ == "__main__":
    main()
