"""Variable-support local planar atlas. Fits observation candidates only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Patch:
    origin: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    uv_min: np.ndarray
    uv_max: np.ndarray
    n_inliers: int
    residual_rms_mm: float


def _plane_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= max(np.linalg.norm(normal), 1e-12)
    return center, normal


def fit_plane_ransac(points: np.ndarray, thresh_mm: float, rng: np.random.Generator, iters: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n = len(points)
    if n < 12:
        return None
    best_count = 0
    best = None
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal / length
        dist = np.abs((points - p0) @ normal)
        inliers = dist < thresh_mm
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best = inliers
    if best is None or best_count < 12:
        return None
    origin, normal = _plane_from_points(points[best])
    dist = np.abs((points - origin) @ normal)
    inliers = dist < thresh_mm
    if inliers.sum() < 12:
        return None
    origin, normal = _plane_from_points(points[inliers])
    return origin, normal, inliers


def _frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent_u = np.cross(normal, helper)
    tangent_u /= max(np.linalg.norm(tangent_u), 1e-12)
    tangent_v = np.cross(normal, tangent_u)
    tangent_v /= max(np.linalg.norm(tangent_v), 1e-12)
    return tangent_u, tangent_v


def sequential_atlas(points: np.ndarray, max_patches: int, thresh_mm: float, min_points: int, seed: int = 0) -> list[Patch]:
    remaining = np.arange(len(points))
    rng = np.random.default_rng(seed)
    patches: list[Patch] = []
    xyz = np.asarray(points, dtype=np.float64)
    for _ in range(max_patches):
        if len(remaining) < min_points:
            break
        fitted = fit_plane_ransac(xyz[remaining], thresh_mm, rng)
        if fitted is None:
            break
        origin, normal, local_inliers = fitted
        if local_inliers.sum() < min_points:
            break
        chosen = remaining[local_inliers]
        tangent_u, tangent_v = _frame(normal)
        local = xyz[chosen] - origin
        uv = np.column_stack([local @ tangent_u, local @ tangent_v])
        residual = (xyz[chosen] - origin) @ normal
        patches.append(
            Patch(
                origin=origin,
                normal=normal,
                tangent_u=tangent_u,
                tangent_v=tangent_v,
                uv_min=uv.min(axis=0),
                uv_max=uv.max(axis=0),
                n_inliers=int(len(chosen)),
                residual_rms_mm=float(np.sqrt(np.mean(residual * residual))),
            )
        )
        remaining = remaining[~local_inliers]
    return patches


def sample_patch(patch: Patch, voxel_mm: float, inlier_uv: np.ndarray | None = None) -> np.ndarray:
    u0, v0 = patch.uv_min
    u1, v1 = patch.uv_max
    if u1 <= u0 or v1 <= v0:
        return np.zeros((0, 3), dtype=np.float64)
    us = np.arange(u0, u1 + 0.5 * voxel_mm, voxel_mm)
    vs = np.arange(v0, v1 + 0.5 * voxel_mm, voxel_mm)
    grid_u, grid_v = np.meshgrid(us, vs, indexing="xy")
    samples = (
        patch.origin[None, :]
        + grid_u.ravel()[:, None] * patch.tangent_u[None, :]
        + grid_v.ravel()[:, None] * patch.tangent_v[None, :]
    )
    if inlier_uv is None or len(inlier_uv) == 0:
        return samples
    # Keep samples near observed inlier UV support, so holes stay holes.
    tree = cKDTree(inlier_uv)
    dist, _ = tree.query(np.column_stack([grid_u.ravel(), grid_v.ravel()]), k=1, workers=1)
    return samples[dist <= 2.5 * voxel_mm]


def patch_uv(points: np.ndarray, patch: Patch) -> np.ndarray:
    local = points - patch.origin
    return np.column_stack([local @ patch.tangent_u, local @ patch.tangent_v])


def assign_and_sample(points: np.ndarray, patches: list[Patch], voxel_mm: float, support_radius_mm: float) -> tuple[np.ndarray, list[dict]]:
    if not patches or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64), []
    chunks = []
    records = []
    for index, patch in enumerate(patches):
        residual = np.abs((points - patch.origin) @ patch.normal)
        inliers = residual <= support_radius_mm
        uv = patch_uv(points[inliers], patch) if inliers.any() else np.zeros((0, 2))
        sampled = sample_patch(patch, voxel_mm, uv if len(uv) else None)
        chunks.append(sampled)
        records.append(
            {
                "patch_id": index,
                "n_inliers": int(inliers.sum()),
                "n_samples": int(len(sampled)),
                "residual_rms_mm": patch.residual_rms_mm,
                "normal": patch.normal.tolist(),
                "origin_mm": patch.origin.tolist(),
                "uv_min": patch.uv_min.tolist(),
                "uv_max": patch.uv_max.tolist(),
            }
        )
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64), records
    return np.concatenate(chunks, axis=0), records
