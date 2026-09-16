"""COLMAP cameras, physical-mm projection, and mesh loading.

Adapted from V23 setup() camera/scale/reprojection checks. This module does not
run the old runner and does not read laser references.
"""

from __future__ import annotations

import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from PIL import Image

from . import paths
from .io_util import sha256_file

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "third_party") not in sys.path:
    sys.path.insert(0, str(ROOT / "third_party"))
import calibration_v23 as colmap_legacy  # noqa: E402


@dataclass
class View:
    name: str
    index: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    P_colmap: np.ndarray
    P_phys: np.ndarray
    width: int
    height: int
    path: Optional[Path]
    center_phys: np.ndarray


@dataclass
class Scene:
    scene_id: int
    scale: np.ndarray
    scale_factor: float
    views: list[View]
    mesh_vertices_phys: np.ndarray
    mesh_path: Path
    cameras_npz: Path
    colmap_dir: Path
    image_dir: Path
    provenance: dict = field(default_factory=dict)
    _images: dict[str, np.ndarray] = field(default_factory=dict)

    def view_by_name(self, name: str) -> View:
        for view in self.views:
            if view.name == name:
                return view
        raise KeyError(name)

    def load_image(self, name: str) -> np.ndarray:
        if name not in self._images:
            view = self.view_by_name(name)
            image = np.asarray(Image.open(view.path).convert("RGB"))
            if image.shape[1] != view.width or image.shape[0] != view.height:
                raise ValueError(f"{name} shape {image.shape} != {view.width}x{view.height}")
            self._images[name] = image
        return self._images[name]


def require_host() -> None:
    host = socket.gethostname()
    if host != paths.EXPECTED_HOST:
        raise RuntimeError(f"host {host} is not {paths.EXPECTED_HOST}")
    if Path.cwd().resolve() != paths.WORKSPACE:
        raise RuntimeError(f"cwd {Path.cwd()} is not {paths.WORKSPACE}")


def project_matrix(points: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        matrix = matrix[:3]
    homogeneous = np.column_stack([points, np.ones(len(points))])
    clip = homogeneous @ matrix.T
    depth = clip[:, 2]
    uv = np.divide(
        clip[:, :2],
        depth[:, None],
        out=np.full((len(points), 2), np.nan),
        where=np.abs(depth[:, None]) > 1e-12,
    )
    return uv, depth


def unproject_colmap_depth(view: View, uv: np.ndarray, depth_norm: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Pixel + COLMAP camera-Z (normalized) -> physical millimetres."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    depth_norm = np.asarray(depth_norm, dtype=np.float64).reshape(-1)
    pixel = np.column_stack([uv, np.ones(len(uv))])
    camera_xy = np.linalg.inv(view.K) @ pixel.T
    camera_xyz = camera_xy * depth_norm[None, :]
    world_norm = view.R.T @ (camera_xyz - view.t[:, None])
    physical = scale[:3, :3] @ world_norm + scale[:3, 3:4]
    return physical.T


def physical_to_colmap_depth(view: View, points_phys: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale_inv = np.linalg.inv(scale)
    homogeneous = np.column_stack([points_phys, np.ones(len(points_phys))])
    world_norm = (homogeneous @ scale_inv.T)[:, :3]
    camera = (view.R @ world_norm.T + view.t[:, None]).T
    return camera[:, 2]


def load_scene(scene_id: int, load_mesh: bool = True) -> Scene:
    require_host()
    if scene_id not in paths.SCENE_INPUTS:
        raise KeyError(scene_id)
    spec = paths.SCENE_INPUTS[scene_id]
    cameras = np.load(spec["cameras_npz"])
    scale = cameras["scale_mat_0"].astype(np.float64)
    if not np.allclose(scale[:3, :3], np.eye(3) * scale[0, 0], atol=1e-8):
        raise ValueError("scale_mat_0 is not isotropic")
    image_paths = sorted(spec["images"].glob("*.png"))
    if len(image_paths) != 49 or [int(p.stem) for p in image_paths] != list(range(49)):
        raise ValueError(f"expected 0000-0048.png in {spec['images']}")
    calibration, sparse_points = colmap_legacy.load(spec["colmap"])
    if set(calibration) != {p.name for p in image_paths}:
        raise ValueError("COLMAP image names do not match PNG filenames")

    views: list[View] = []
    center_errors = []
    reprojection = []
    scale_inv = np.linalg.inv(scale)
    for index, image_path in enumerate(image_paths):
        raw = calibration[image_path.name]
        with Image.open(image_path) as probe:
            width, height = probe.size
        if (width, height) != (raw["width"], raw["height"]):
            raise ValueError(f"{image_path.name} size mismatch")
        world = cameras[f"world_mat_{index}"]
        world_center = -np.linalg.solve(world[:3, :3], world[:3, 3])
        normalized_center = (world_center - scale[:3, 3]) / scale[0, 0]
        colmap_center = -raw["R"].T @ raw["t"]
        center_errors.append(float(np.linalg.norm(normalized_center - colmap_center)))
        if center_errors[-1] >= 1e-4:
            raise ValueError(f"camera center mismatch {image_path.name}: {center_errors[-1]}")
        p_phys = raw["P"] @ scale_inv
        center_phys = -np.linalg.solve(p_phys[:3, :3], p_phys[:3, 3])
        view = View(
            name=image_path.name,
            index=index,
            K=np.asarray(raw["K"], dtype=np.float64),
            R=np.asarray(raw["R"], dtype=np.float64),
            t=np.asarray(raw["t"], dtype=np.float64),
            P_colmap=np.asarray(raw["P"], dtype=np.float64),
            P_phys=np.asarray(p_phys, dtype=np.float64),
            width=width,
            height=height,
            path=image_path,
            center_phys=np.asarray(center_phys, dtype=np.float64),
        )
        views.append(view)
        tracks = raw["tracks"]
        keep = np.array([int(tid) in sparse_points for tid in tracks["id"]], dtype=bool)
        tracks = tracks[keep]
        if len(tracks) == 0:
            continue
        points = np.array([sparse_points[int(tid)] for tid in tracks["id"]], dtype=np.float64)
        uv, _ = project_matrix(points, view.P_colmap)
        reprojection.extend(np.linalg.norm(uv - np.column_stack([tracks["x"], tracks["y"]]), axis=1))

    if not reprojection:
        raise ValueError("no sparse tracks for reprojection check")
    reprojection = np.asarray(reprojection, dtype=np.float64)
    median_px = float(np.median(reprojection))
    p95_px = float(np.quantile(reprojection, 0.95))
    if p95_px > 5.0:
        raise ValueError(f"sparse reprojection P95 {p95_px:.3f} px exceeds 5 px")

    mesh_vertices = np.zeros((0, 3), dtype=np.float64)
    if load_mesh:
        mesh = o3d.io.read_triangle_mesh(str(spec["mesh"]))
        mesh.transform(scale)
        mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(mesh_vertices) < 1000:
            raise ValueError("mesh vertex count unexpectedly low")

    provenance = {
        "scene": scene_id,
        "camera": str(spec["cameras_npz"]),
        "camera_sha256": sha256_file(spec["cameras_npz"]),
        "mesh": str(spec["mesh"]),
        "mesh_sha256": sha256_file(spec["mesh"]),
        "colmap": str(spec["colmap"]),
        "colmap_sha256": {p.name: sha256_file(p) for p in sorted(spec["colmap"].glob("*.bin"))},
        "image_count": len(image_paths),
        "image_size": [views[0].width, views[0].height],
        "normalized_center_max_difference": float(max(center_errors)),
        "sparse_reprojection_median_px": median_px,
        "sparse_reprojection_p95_px": p95_px,
        "scale_factor": float(scale[0, 0]),
        "units": "physical_mm_after_scale_mat",
        "mesh_vertex_count": int(len(mesh_vertices)),
    }
    return Scene(
        scene_id=scene_id,
        scale=scale,
        scale_factor=float(scale[0, 0]),
        views=views,
        mesh_vertices_phys=mesh_vertices,
        mesh_path=spec["mesh"],
        cameras_npz=spec["cameras_npz"],
        colmap_dir=spec["colmap"],
        image_dir=spec["images"],
        provenance=provenance,
    )
