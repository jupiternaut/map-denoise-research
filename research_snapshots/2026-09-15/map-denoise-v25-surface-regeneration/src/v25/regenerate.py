"""Constructors that consume an evidence bundle. No GT / laser / mask arguments."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .cameras import Scene, unproject_colmap_depth
from .evidence import EvidenceBundle, peak_depths, wta_depth
from .io_util import voxel_downsample
from .observations import Roi, crop_to_view, crop_view, points_in_aabb
from .surfaces import assign_and_sample, sequential_atlas


def _ref_cropped(scene: Scene, bundle: EvidenceBundle):
    ref = scene.view_by_name(bundle.ref_name)
    roi = Roi(
        roi_id=bundle.roi_id,
        scene_id=bundle.scene_id,
        kind="unknown",
        reason="",
        aabb_min=bundle.aabb_min,
        aabb_max=bundle.aabb_max,
        ref_view=bundle.ref_view if hasattr(bundle, "ref_view") else bundle.ref_name,
        box_xyxy=[0, 0, 1, 1],
        n_mesh_points=0,
        centroid=0.5 * (bundle.aabb_min + bundle.aabb_max),
    )
    crop = crop_view(scene, ref, roi, int(bundle.config["crop_px"]))
    return crop_to_view(ref, crop, scene.scale), crop


def identity_points(scene: Scene, bundle: EvidenceBundle) -> tuple[np.ndarray, dict]:
    mask = points_in_aabb(scene.mesh_vertices_phys, bundle.aabb_min, bundle.aabb_max)
    points = scene.mesh_vertices_phys[mask]
    voxel = float(bundle.config["voxel_mm"])
    sampled = voxel_downsample(points, voxel)
    return sampled, {
        "arm": "identity",
        "n_mesh": int(len(points)),
        "n_output": int(len(sampled)),
        "resample": f"vertex_voxel_{voxel}mm",
        "used_evidence_volume": False,
    }


def fusion_wta_points(scene: Scene, bundle: EvidenceBundle) -> tuple[np.ndarray, dict]:
    depth, score, valid = wta_depth(bundle)
    ref_cview, _ = _ref_cropped(scene, bundle)
    yy, xx = np.nonzero(valid)
    if len(xx) == 0:
        return np.zeros((0, 3), dtype=np.float64), {"arm": "fusion_wta", "n_output": 0, "status": "EMPTY_EVIDENCE"}
    uv = np.column_stack([xx.astype(np.float64), yy.astype(np.float64)])
    xyz = unproject_colmap_depth(ref_cview, uv, depth[valid], scene.scale)
    inside = points_in_aabb(xyz, bundle.aabb_min, bundle.aabb_max)
    xyz = xyz[inside]
    sampled = voxel_downsample(xyz, float(bundle.config["voxel_mm"]))
    return sampled, {
        "arm": "fusion_wta",
        "n_valid_pixels": int(valid.sum()),
        "n_unprojected": int(len(xyz)),
        "n_output": int(len(sampled)),
        "mean_zncc": float(score[valid].mean()) if valid.any() else None,
        "name": "cpu_zncc_plane_sweep_wta",
        "not_official_colmap": True,
    }


def _candidate_points(scene: Scene, bundle: EvidenceBundle, multi_peak: bool) -> tuple[np.ndarray, dict]:
    ref_cview, _ = _ref_cropped(scene, bundle)
    if multi_peak:
        peaks = peak_depths(bundle)
        if not peaks:
            return np.zeros((0, 3), dtype=np.float64), {"n_peaks": 0}
        uv = np.array([[p["u"], p["v"]] for p in peaks], dtype=np.float64)
        depths = np.array([p["depth_norm"] for p in peaks], dtype=np.float64)
        xyz = unproject_colmap_depth(ref_cview, uv, depths, scene.scale)
        n_second = int(sum(p["rank"] == 1 for p in peaks))
        info = {"n_peaks": len(peaks), "n_second_peaks": n_second}
    else:
        depth, score, valid = wta_depth(bundle)
        yy, xx = np.nonzero(valid)
        if len(xx) == 0:
            return np.zeros((0, 3), dtype=np.float64), {"n_peaks": 0}
        uv = np.column_stack([xx.astype(np.float64), yy.astype(np.float64)])
        xyz = unproject_colmap_depth(ref_cview, uv, depth[valid], scene.scale)
        info = {"n_peaks": int(len(xyz)), "n_second_peaks": 0, "mean_zncc": float(score[valid].mean())}
    inside = points_in_aabb(xyz, bundle.aabb_min, bundle.aabb_max)
    return xyz[inside], info


def restricted_single_points(scene: Scene, bundle: EvidenceBundle) -> tuple[np.ndarray, dict]:
    xyz, info = _candidate_points(scene, bundle, multi_peak=False)
    patches = sequential_atlas(
        xyz,
        max_patches=1,
        thresh_mm=float(bundle.config["plane_thresh_mm"]),
        min_points=int(bundle.config["min_patch_points"]),
    )
    sampled, records = assign_and_sample(
        xyz, patches, float(bundle.config["voxel_mm"]), float(bundle.config["support_radius_mm"])
    )
    sampled = voxel_downsample(sampled, float(bundle.config["voxel_mm"]))
    return sampled, {
        "arm": "restricted_single",
        "n_candidate": int(len(xyz)),
        "n_output": int(len(sampled)),
        "n_patches": len(patches),
        "patches": records,
        **info,
    }


def v25_atlas_points(scene: Scene, bundle: EvidenceBundle) -> tuple[np.ndarray, dict]:
    xyz, info = _candidate_points(scene, bundle, multi_peak=True)
    patches = sequential_atlas(
        xyz,
        max_patches=int(bundle.config["max_patches"]),
        thresh_mm=float(bundle.config["plane_thresh_mm"]),
        min_points=int(bundle.config["min_patch_points"]),
    )
    sampled, records = assign_and_sample(
        xyz, patches, float(bundle.config["voxel_mm"]), float(bundle.config["support_radius_mm"])
    )
    sampled = voxel_downsample(sampled, float(bundle.config["voxel_mm"]))
    return sampled, {
        "arm": "v25_atlas",
        "n_candidate": int(len(xyz)),
        "n_output": int(len(sampled)),
        "n_patches": len(patches),
        "patches": records,
        "variable_support": True,
        **info,
    }


def restricted_point_move(scene: Scene, bundle: EvidenceBundle) -> tuple[np.ndarray, dict]:
    """Move identity points onto nearest WTA evidence; same cardinality after voxel match is not required."""
    identity, _ = identity_points(scene, bundle)
    evidence, info = _candidate_points(scene, bundle, multi_peak=False)
    if len(identity) == 0 or len(evidence) == 0:
        return identity, {"arm": "restricted_point_move", "n_output": int(len(identity)), "status": "IDENTITY_FALLBACK", **info}
    dist, idx = cKDTree(evidence).query(identity, k=1, workers=1)
    moved = identity.copy()
    take = dist <= float(bundle.config["support_radius_mm"]) * 2.5
    moved[take] = evidence[idx[take]]
    return moved, {
        "arm": "restricted_point_move",
        "n_output": int(len(moved)),
        "n_moved": int(take.sum()),
        "median_move_mm": float(np.median(dist[take])) if take.any() else 0.0,
        **info,
    }


def v25_wta_atlas_points(scene: Scene, bundle: EvidenceBundle) -> tuple[np.ndarray, dict]:
    """Same atlas, but only WTA peaks. Tests whether second ZNCC peaks were aliases."""
    xyz, info = _candidate_points(scene, bundle, multi_peak=False)
    patches = sequential_atlas(
        xyz,
        max_patches=int(bundle.config["max_patches"]),
        thresh_mm=float(bundle.config["plane_thresh_mm"]),
        min_points=int(bundle.config["min_patch_points"]),
    )
    sampled, records = assign_and_sample(
        xyz, patches, float(bundle.config["voxel_mm"]), float(bundle.config["support_radius_mm"])
    )
    sampled = voxel_downsample(sampled, float(bundle.config["voxel_mm"]))
    return sampled, {
        "arm": "v25_wta_atlas",
        "n_candidate": int(len(xyz)),
        "n_output": int(len(sampled)),
        "n_patches": len(patches),
        "patches": records,
        "variable_support": True,
        "used_second_peaks": False,
        **info,
    }


def v25_depth_cc_points(scene: Scene, bundle: EvidenceBundle) -> tuple[np.ndarray, dict]:
    """Split the WTA depth map on jumps, then keep each component's unprojected points."""
    depth, score, valid = wta_depth(bundle)
    scale = float(bundle.config.get("scale_factor") or 1.0)
    sep = float(bundle.config["peak_sep_mm"]) / scale
    labels, n_comp, sizes = _depth_components(depth, valid, sep)
    ref_cview, _ = _ref_cropped(scene, bundle)
    min_pix = max(24, int(bundle.config["min_patch_points"]) // 2)
    chunks = []
    records = []
    for lab in range(1, n_comp + 1):
        if sizes[lab] < min_pix:
            continue
        yy, xx = np.nonzero(labels == lab)
        uv = np.column_stack([xx.astype(np.float64), yy.astype(np.float64)])
        xyz = unproject_colmap_depth(ref_cview, uv, depth[yy, xx], scene.scale)
        inside = points_in_aabb(xyz, bundle.aabb_min, bundle.aabb_max)
        xyz = xyz[inside]
        if len(xyz) < 12:
            continue
        chunks.append(xyz)
        records.append(
            {
                "component": int(lab),
                "n_pixels": int(sizes[lab]),
                "n_points": int(len(xyz)),
                "mean_zncc": float(score[yy, xx].mean()),
            }
        )
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64), {"arm": "v25_depth_cc", "n_output": 0, "n_components_kept": 0}
    sampled = voxel_downsample(np.concatenate(chunks, axis=0), float(bundle.config["voxel_mm"]))
    return sampled, {
        "arm": "v25_depth_cc",
        "n_output": int(len(sampled)),
        "n_components_total": int(n_comp),
        "n_components_kept": len(records),
        "components": records[:24],
        "variable_support": True,
    }


def v25_gated_move(scene: Scene, bundle: EvidenceBundle) -> tuple[np.ndarray, dict]:
    """Keep identity unless WTA evidence is strong and disagrees by more than 3 mm."""
    identity, _ = identity_points(scene, bundle)
    evidence, info = _candidate_points(scene, bundle, multi_peak=False)
    if len(identity) == 0 or len(evidence) == 0:
        return identity, {"arm": "v25_gated_move", "n_output": int(len(identity)), "status": "IDENTITY_FALLBACK", **info}
    dist, idx = cKDTree(evidence).query(identity, k=1, workers=1)
    gate = (dist >= 3.0) & (dist <= 12.0)
    moved = identity.copy()
    moved[gate] = evidence[idx[gate]]
    return moved, {
        "arm": "v25_gated_move",
        "n_output": int(len(moved)),
        "n_moved": int(gate.sum()),
        "n_kept": int((~gate).sum()),
        "median_move_mm": float(np.median(dist[gate])) if gate.any() else 0.0,
        "variable_support": True,
        **info,
    }


def _depth_components(depth: np.ndarray, valid: np.ndarray, sep: float) -> tuple[np.ndarray, int, dict]:
    height, width = depth.shape
    parent = np.arange(height * width, dtype=np.int32)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for y in range(height):
        for x in range(width):
            if not valid[y, x]:
                continue
            i = y * width + x
            if x + 1 < width and valid[y, x + 1] and abs(depth[y, x] - depth[y, x + 1]) < sep:
                union(i, i + 1)
            if y + 1 < height and valid[y + 1, x] and abs(depth[y, x] - depth[y + 1, x]) < sep:
                union(i, i + width)
    roots = {}
    labels = np.zeros((height, width), dtype=np.int32)
    next_lab = 1
    sizes = {0: 0}
    for y in range(height):
        for x in range(width):
            if not valid[y, x]:
                continue
            root = find(y * width + x)
            if root not in roots:
                roots[root] = next_lab
                sizes[next_lab] = 0
                next_lab += 1
            lab = roots[root]
            labels[y, x] = lab
            sizes[lab] += 1
    return labels, next_lab - 1, sizes


ARMS = {
    "identity": identity_points,
    "fusion_wta": fusion_wta_points,
    "restricted_single": restricted_single_points,
    "restricted_point_move": restricted_point_move,
    "v25_atlas": v25_atlas_points,
    "v25_wta_atlas": v25_wta_atlas_points,
    "v25_depth_cc": v25_depth_cc_points,
    "v25_gated_move": v25_gated_move,
}


def run_arm(name: str, scene: Scene, bundle: EvidenceBundle) -> tuple[np.ndarray, dict]:
    if name not in ARMS:
        raise KeyError(name)
    return ARMS[name](scene, bundle)
