"""Hashing, JSON, and point-cloud I/O. No evaluator imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    payload = np.ascontiguousarray(array)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def write_json(path: Path | str, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default)
    path.write_text(text + "\n")


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def write_ply(path: Path | str, points: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"failed to write {path}")


def read_ply(path: Path | str) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return points


def voxel_downsample(points: np.ndarray, voxel_mm: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return points
    if voxel_mm <= 0:
        raise ValueError("voxel_mm must be positive")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    down = cloud.voxel_down_sample(voxel_mm)
    out = np.asarray(down.points, dtype=np.float64)
    return out if len(out) else points[:0].reshape(0, 3)


def append_jsonl(path: Path | str, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
