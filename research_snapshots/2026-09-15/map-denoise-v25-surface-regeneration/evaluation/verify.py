"""Independent metric recompute. Does not import evaluation.metrics scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.io import loadmat
from scipy.spatial import cKDTree

EMPTY_PENALTY_MM = 1000.0


def _points(path: Path | str) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(cloud.points, dtype=np.float64)
    return xyz if xyz.size else np.zeros((0, 3), dtype=np.float64)


def _voxel(points: np.ndarray, voxel_mm: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    out = np.asarray(cloud.voxel_down_sample(voxel_mm).points, dtype=np.float64)
    return out if len(out) else np.zeros((0, 3), dtype=np.float64)


def _observed(points: np.ndarray, mask_path: Path | str) -> np.ndarray:
    data = loadmat(str(mask_path))
    mask = np.asarray(data["ObsMask"])
    origin = np.asarray(data["BB"], dtype=np.float64).reshape(-1, 3)[:1]
    res = float(np.asarray(data["Res"]).reshape(-1)[0])
    grid = np.around((points - origin) / res).astype(np.int32)
    valid = np.all((grid >= 0) & (grid < np.asarray(mask.shape)), axis=1)
    keep = np.zeros(len(points), dtype=bool)
    if valid.any():
        g = grid[valid]
        keep[valid] = mask[g[:, 0], g[:, 1], g[:, 2]].astype(bool)
    return keep


def recompute(output_ply, laser_ply, mask_path, aabb_min, aabb_max, voxel_mm: float) -> dict:
    lo = np.asarray(aabb_min, dtype=np.float64)
    hi = np.asarray(aabb_max, dtype=np.float64)
    output = _points(output_ply)
    laser = _points(laser_ply)
    out = output[np.all((output >= lo) & (output <= hi), axis=1)] if len(output) else output
    ref = laser[np.all((laser >= lo) & (laser <= hi), axis=1)] if len(laser) else laser
    if len(out):
        out = out[_observed(out, mask_path)]
    if len(ref):
        ref = ref[_observed(ref, mask_path)]
    out = _voxel(out, voxel_mm)
    ref = _voxel(ref, voxel_mm)
    if len(out) == 0:
        return {"status": "EMPTY_OUTPUT", "E_sym_mm": EMPTY_PENALTY_MM, "accuracy_mm": EMPTY_PENALTY_MM, "completeness_mm": EMPTY_PENALTY_MM, "n_output_eval": 0, "n_laser_eval": int(len(ref))}
    acc = cKDTree(ref).query(out, workers=1)[0] if len(ref) else np.full(len(out), EMPTY_PENALTY_MM)
    if len(ref) == 0:
        completeness = EMPTY_PENALTY_MM
        recall = 0.0
    else:
        comp = cKDTree(out).query(ref, workers=1)[0]
        completeness = float(comp.mean())
        recall = float(np.mean(comp <= 1.0))
    accuracy = float(acc.mean())
    precision = float(np.mean(acc <= 1.0))
    fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "status": "SCORED",
        "accuracy_mm": accuracy,
        "completeness_mm": completeness,
        "E_sym_mm": 0.5 * (accuracy + completeness),
        "precision_1mm": precision,
        "recall_1mm": recall,
        "fscore_1mm": fscore,
        "n_output_eval": int(len(out)),
        "n_laser_eval": int(len(ref)),
    }
