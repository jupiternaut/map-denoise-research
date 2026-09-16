"""Fixed-space ROI scoring. Independent of src.v25 constructors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .mask import load_obs_mask, observed

EMPTY_PENALTY_MM = 1000.0


def read_points(path: Path | str) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    return points if points.size else np.zeros((0, 3), dtype=np.float64)


def voxel_downsample(points: np.ndarray, voxel_mm: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or voxel_mm <= 0:
        return points.reshape(0, 3) if len(points) == 0 else points
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    down = np.asarray(cloud.voxel_down_sample(voxel_mm).points, dtype=np.float64)
    return down if len(down) else np.zeros((0, 3), dtype=np.float64)


def in_aabb(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.all((points >= lo) & (points <= hi), axis=1)


def _stats(distances: np.ndarray) -> dict:
    if len(distances) == 0:
        return {"mean": None, "p95": None, "frac_1mm": None}
    return {
        "mean": float(distances.mean()),
        "p95": float(np.quantile(distances, 0.95)),
        "frac_1mm": float(np.mean(distances <= 1.0)),
    }


def score_arrays(
    output: np.ndarray,
    laser: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    obs: dict | None,
    voxel_mm: float,
    empty_penalty_mm: float = EMPTY_PENALTY_MM,
) -> dict:
    output = np.asarray(output, dtype=np.float64).reshape(-1, 3)
    laser = np.asarray(laser, dtype=np.float64).reshape(-1, 3)
    raw_n = int(len(output))
    out_roi = output[in_aabb(output, aabb_min, aabb_max)]
    laser_roi = laser[in_aabb(laser, aabb_min, aabb_max)]
    if obs is not None:
        if len(out_roi):
            out_roi = out_roi[observed(out_roi, obs)]
        if len(laser_roi):
            laser_roi = laser_roi[observed(laser_roi, obs)]
    out_ds = voxel_downsample(out_roi, voxel_mm)
    laser_ds = voxel_downsample(laser_roi, voxel_mm)
    if len(out_ds) == 0:
        return {
            "status": "EMPTY_OUTPUT",
            "n_output_raw": raw_n,
            "n_output_roi": int(len(out_roi)),
            "n_output_eval": 0,
            "n_laser_eval": int(len(laser_ds)),
            "accuracy_mm": empty_penalty_mm,
            "completeness_mm": empty_penalty_mm,
            "accuracy_p95_mm": empty_penalty_mm,
            "completeness_p95_mm": empty_penalty_mm,
            "precision_1mm": 0.0,
            "recall_1mm": 0.0,
            "fscore_1mm": 0.0,
            "E_sym_mm": empty_penalty_mm,
            "coverage_laser_points": int(len(laser_ds)),
            "voxel_mm": voxel_mm,
        }
    acc = cKDTree(laser_ds).query(out_ds, k=1, workers=1)[0] if len(laser_ds) else np.full(len(out_ds), empty_penalty_mm)
    if len(laser_ds) == 0:
        comp = np.full(0, np.nan)
        completeness = empty_penalty_mm
        recall = 0.0
        comp_p95 = empty_penalty_mm
    else:
        comp = cKDTree(out_ds).query(laser_ds, k=1, workers=1)[0]
        completeness = float(comp.mean())
        recall = float(np.mean(comp <= 1.0))
        comp_p95 = float(np.quantile(comp, 0.95))
    precision = float(np.mean(acc <= 1.0))
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = float(acc.mean())
    return {
        "status": "SCORED",
        "n_output_raw": raw_n,
        "n_output_roi": int(len(out_roi)),
        "n_output_eval": int(len(out_ds)),
        "n_laser_eval": int(len(laser_ds)),
        "accuracy_mm": accuracy,
        "completeness_mm": completeness,
        "accuracy_p95_mm": float(np.quantile(acc, 0.95)),
        "completeness_p95_mm": comp_p95,
        "precision_1mm": precision,
        "recall_1mm": recall,
        "fscore_1mm": fscore,
        "E_sym_mm": 0.5 * (accuracy + completeness),
        "coverage_laser_points": int(len(laser_ds)),
        "voxel_mm": voxel_mm,
    }


def score_file(
    output_ply: Path | str,
    laser_ply: Path | str,
    mask_path: Path | str,
    aabb_min,
    aabb_max,
    voxel_mm: float,
    empty_penalty_mm: float = EMPTY_PENALTY_MM,
) -> dict:
    obs = load_obs_mask(mask_path)
    return score_arrays(
        read_points(output_ply),
        read_points(laser_ply),
        np.asarray(aabb_min, dtype=np.float64),
        np.asarray(aabb_max, dtype=np.float64),
        obs,
        voxel_mm,
        empty_penalty_mm,
    )
