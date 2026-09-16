"""ROI definition from photos + input mesh, crop/scale, and view selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .cameras import Scene, View, physical_to_colmap_depth, project_matrix
from .io_util import read_json


@dataclass
class Roi:
    roi_id: str
    scene_id: int
    kind: str
    reason: str
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    ref_view: str
    box_xyxy: list[int]
    n_mesh_points: int
    centroid: np.ndarray


@dataclass
class Crop:
    view_name: str
    x0: int
    y0: int
    side: int
    scale: float
    K: np.ndarray
    gray: np.ndarray
    rgb: np.ndarray


def load_roi_specs(path) -> dict:
    return read_json(path)


def mesh_points_in_image_box(scene: Scene, view: View, box_xyxy: list[int]) -> np.ndarray:
    uv, depth = project_matrix(scene.mesh_vertices_phys, view.P_phys)
    x0, y0, x1, y1 = box_xyxy
    inside = (
        np.isfinite(uv).all(1)
        & (depth > 0)
        & (uv[:, 0] >= x0)
        & (uv[:, 0] < x1)
        & (uv[:, 1] >= y0)
        & (uv[:, 1] < y1)
    )
    return scene.mesh_vertices_phys[inside]


def aabb_from_points(points: np.ndarray, pad_mm: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    lo = points.min(axis=0) - pad_mm
    hi = points.max(axis=0) + pad_mm
    return lo, hi


def points_in_aabb(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.all((points >= lo) & (points <= hi), axis=1)


def define_rois(scene: Scene, spec: dict, min_points: int = 80) -> list[Roi]:
    ref_name = spec["ref_view"]
    view = scene.view_by_name(ref_name)
    rois = []
    for item in spec["scenes"][str(scene.scene_id)]:
        selected = mesh_points_in_image_box(scene, view, item["box_xyxy"])
        if len(selected) < min_points:
            raise ValueError(f"{item['roi_id']} has only {len(selected)} mesh points in the photo box")
        lo, hi = aabb_from_points(selected, pad_mm=4.0)
        inside = scene.mesh_vertices_phys[points_in_aabb(scene.mesh_vertices_phys, lo, hi)]
        rois.append(
            Roi(
                roi_id=item["roi_id"],
                scene_id=scene.scene_id,
                kind=item["kind"],
                reason=item["reason"],
                aabb_min=lo,
                aabb_max=hi,
                ref_view=ref_name,
                box_xyxy=list(item["box_xyxy"]),
                n_mesh_points=int(len(inside)),
                centroid=inside.mean(axis=0),
            )
        )
    return rois


def shrink_roi(scene: Scene, roi: Roi, keep_frac: float = 0.4) -> Roi:
    """Keep the nearest keep_frac of input-mesh points to the ROI centroid."""
    inside = scene.mesh_vertices_phys[points_in_aabb(scene.mesh_vertices_phys, roi.aabb_min, roi.aabb_max)]
    if len(inside) < 40:
        return roi
    dist = np.linalg.norm(inside - roi.centroid[None, :], axis=1)
    cutoff = np.quantile(dist, keep_frac)
    core = inside[dist <= cutoff]
    lo, hi = aabb_from_points(core, pad_mm=2.0)
    return Roi(
        roi_id=roi.roi_id + "_core",
        scene_id=roi.scene_id,
        kind=roi.kind,
        reason=roi.reason + " | core=nearest_40pct_input_mesh",
        aabb_min=lo,
        aabb_max=hi,
        ref_view=roi.ref_view,
        box_xyxy=roi.box_xyxy,
        n_mesh_points=int(len(core)),
        centroid=core.mean(axis=0),
    )


def roi_to_dict(roi: Roi) -> dict:
    return {
        "roi_id": roi.roi_id,
        "scene_id": roi.scene_id,
        "kind": roi.kind,
        "reason": roi.reason,
        "aabb_min_mm": roi.aabb_min.tolist(),
        "aabb_max_mm": roi.aabb_max.tolist(),
        "ref_view": roi.ref_view,
        "box_xyxy": roi.box_xyxy,
        "n_mesh_points": roi.n_mesh_points,
        "centroid_mm": roi.centroid.tolist(),
        "selection": "photo_box_on_0022_plus_input_mesh_aabb",
        "used_laser": False,
    }


def depth_range_from_mesh(scene: Scene, view: View, roi: Roi, margin_mm: float) -> tuple[float, float]:
    inside = scene.mesh_vertices_phys[points_in_aabb(scene.mesh_vertices_phys, roi.aabb_min, roi.aabb_max)]
    if len(inside) < 8:
        raise ValueError(f"{roi.roi_id} mesh support too small for a depth range")
    depths = physical_to_colmap_depth(view, inside, scene.scale)
    depths = depths[np.isfinite(depths) & (depths > 1e-6)]
    if len(depths) < 8:
        raise ValueError(f"{roi.roi_id} has no positive camera depths")
    margin_norm = margin_mm / scene.scale_factor
    return float(depths.min() - margin_norm), float(depths.max() + margin_norm)


def select_views(scene: Scene, roi: Roi, n_views: int) -> list[str]:
    scores = []
    for view in scene.views:
        uv, depth = project_matrix(np.asarray([roi.centroid]), view.P_phys)
        if not np.isfinite(uv).all() or depth[0] <= 0:
            continue
        u, v = uv[0]
        if u < 8 or v < 8 or u >= view.width - 8 or v >= view.height - 8:
            continue
        inside = scene.mesh_vertices_phys[points_in_aabb(scene.mesh_vertices_phys, roi.aabb_min, roi.aabb_max)]
        uv_i, z_i = project_matrix(inside, view.P_phys)
        visible = np.isfinite(uv_i).all(1) & (z_i > 0)
        visible &= (uv_i[:, 0] >= 0) & (uv_i[:, 0] < view.width) & (uv_i[:, 1] >= 0) & (uv_i[:, 1] < view.height)
        n_vis = int(visible.sum())
        if n_vis < 20:
            continue
        look = roi.centroid - view.center_phys
        look /= max(np.linalg.norm(look), 1e-9)
        # Prefer views that see many ROI points and are not grazing.
        scores.append((n_vis, view.name))
    if not scores:
        raise ValueError(f"no views see {roi.roi_id}")
    scores.sort(reverse=True)
    preferred = [roi.ref_view] if any(name == roi.ref_view for _, name in scores) else []
    names = preferred + [name for _, name in scores if name not in preferred]
    return names[:n_views]


def crop_view(scene: Scene, view: View, roi: Roi, crop_px: int) -> Crop:
    inside = scene.mesh_vertices_phys[points_in_aabb(scene.mesh_vertices_phys, roi.aabb_min, roi.aabb_max)]
    uv, depth = project_matrix(inside, view.P_phys)
    keep = np.isfinite(uv).all(1) & (depth > 0)
    if keep.sum() < 8:
        raise ValueError(f"{view.name} does not see {roi.roi_id}")
    uv = uv[keep]
    x0 = int(np.floor(uv[:, 0].min())) - 8
    y0 = int(np.floor(uv[:, 1].min())) - 8
    x1 = int(np.ceil(uv[:, 0].max())) + 8
    y1 = int(np.ceil(uv[:, 1].max())) + 8
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(view.width, x1)
    y1 = min(view.height, y1)
    side = max(x1 - x0, y1 - y0, 32)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    x0 = int(np.floor(cx - side / 2.0))
    y0 = int(np.floor(cy - side / 2.0))
    x0 = min(max(0, x0), view.width - 2)
    y0 = min(max(0, y0), view.height - 2)
    side = int(min(side, view.width - x0, view.height - y0))
    rgb = scene.load_image(view.name)[y0 : y0 + side, x0 : x0 + side]
    resized = np.asarray(Image.fromarray(rgb).resize((crop_px, crop_px), Image.BILINEAR))
    scale = crop_px / float(side)
    k = view.K.copy()
    k[0, 0] *= scale
    k[1, 1] *= scale
    k[0, 2] = (k[0, 2] - x0) * scale
    k[1, 2] = (k[1, 2] - y0) * scale
    gray = (0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]) / 255.0
    return Crop(view.name, x0, y0, side, scale, k, gray.astype(np.float64), resized)


def crop_to_view(view: View, crop: Crop, scale: np.ndarray) -> View:
    """Return a view whose K matches the cropped/scaled image."""
    p_colmap = crop.K @ np.column_stack([view.R, view.t])
    p_phys = p_colmap @ np.linalg.inv(scale)
    return View(
        name=view.name,
        index=view.index,
        K=crop.K.copy(),
        R=view.R.copy(),
        t=view.t.copy(),
        P_colmap=p_colmap,
        P_phys=p_phys,
        width=crop.gray.shape[1],
        height=crop.gray.shape[0],
        path=view.path,
        center_phys=view.center_phys.copy(),
    )
