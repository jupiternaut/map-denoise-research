"""Lightweight NPZ+JSON patch schema. Evaluation fields stay off the estimator path."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from hashutil import dump_json, sha256_array, sha256_file
from transforms import as_matrix44, origin_from_pose, apply_pose, check_homogeneous

POINT_KEYS = (
    "xyz_world",
    "xyz_local",
    "scan_id",
    "source_point_index",
    "scanner_origin_world",
    "ray_direction",
    "xyz_local_valid",
    "ray_valid",
)


def empty_points(n: int) -> dict:
    nan = np.full((n, 3), np.nan, dtype=np.float64)
    return {
        "xyz_world": np.zeros((n, 3), dtype=np.float64),
        "xyz_local": nan.copy(),
        "scan_id": np.zeros(n, dtype=np.int64),
        "source_point_index": np.full(n, -1, dtype=np.int64),
        "scanner_origin_world": nan.copy(),
        "ray_direction": nan.copy(),
        "xyz_local_valid": np.zeros(n, dtype=np.bool_),
        "ray_valid": np.zeros(n, dtype=np.bool_),
    }


def fill_rays(points: dict) -> None:
    valid = points["xyz_local_valid"] & np.isfinite(points["scanner_origin_world"]).all(axis=1)
    vectors = points["xyz_world"] - points["scanner_origin_world"]
    norms = np.linalg.norm(vectors, axis=1)
    usable = valid & (norms > 1e-12)
    points["ray_direction"] = np.full((len(vectors), 3), np.nan, dtype=np.float64)
    points["ray_direction"][usable] = vectors[usable] / norms[usable, None]
    points["ray_valid"] = usable
    points["ray_note"] = (
        "ray_direction is derived from scanner origin to the returned point; "
        "it is not an independently measured direction. First-return points "
        "are not a complete visibility ground truth."
    )


def validate_points(points: dict) -> None:
    n = len(points["xyz_world"])
    if n == 0:
        raise ValueError("patch has no points")
    for key in POINT_KEYS:
        if key not in points:
            raise KeyError(key)
        if len(points[key]) != n:
            raise ValueError(f"{key} length {len(points[key])} != {n}")
    if points["xyz_world"].dtype != np.float64:
        raise TypeError("xyz_world must be float64 metres")
    if not np.isfinite(points["xyz_world"]).all():
        raise ValueError("xyz_world must be finite")
    fake_zero_rays = (
        (~points["ray_valid"])
        & np.isfinite(points["ray_direction"]).all(axis=1)
        & (np.linalg.norm(points["ray_direction"], axis=1) == 0)
    )
    if np.any(fake_zero_rays):
        raise ValueError("invalid rays must not be stored as zero vectors")


def write_patch(directory: Path, patch_id: str, points: dict, meta: dict, evaluation: dict | None = None) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    validate_points(points)
    arrays = {key: np.asarray(points[key]) for key in POINT_KEYS}
    npz_path = directory / f"{patch_id}.npz"
    json_path = directory / f"{patch_id}.json"
    targets = [npz_path, json_path, directory / "evaluation" / f"{patch_id}.eval.npz",
               directory / "evaluation" / f"{patch_id}.eval.json"]
    if any(p.exists() for p in targets):
        raise FileExistsError(f"refusing to overwrite patch {patch_id}; use a new run directory")
    validate_pose_consistency(points, meta)
    with npz_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    record = dict(meta)
    record.update(
        {
            "patch_id": patch_id,
            "n_points": int(len(arrays["xyz_world"])),
            "scan_ids": sorted(int(s) for s in np.unique(arrays["scan_id"])),
            "npz": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "array_sha256": {key: sha256_array(arrays[key]) for key in POINT_KEYS},
            "ray_note": points.get("ray_note"),
        }
    )
    dump_json(json_path, record)
    eval_path = None
    if evaluation is not None:
        eval_dir = directory / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_path = eval_dir / f"{patch_id}.eval.npz"
        payload = {}
        extra = {}
        for key, value in evaluation.items():
            if isinstance(value, np.ndarray):
                payload[key] = value
            else:
                extra[key] = value
        if extra:
            payload["json"] = np.frombuffer(json.dumps(extra).encode(), dtype=np.uint8)
        np.savez_compressed(eval_path, **payload)
        dump_json(eval_dir / f"{patch_id}.eval.json", {
            "patch_id": patch_id,
            "eval_npz": str(eval_path),
            "eval_sha256": sha256_file(eval_path),
            "keys": sorted(payload),
            "note": "evaluation only; never pass this file to estimate()",
        })
    return {"meta": str(json_path), "npz": str(npz_path), "evaluation": str(eval_path) if eval_path else None}


def validate_pose_consistency(points: dict, meta: dict, atol=1e-8) -> None:
    poses = pose_table(meta)
    for sid in np.unique(points["scan_id"]):
        if int(sid) not in poses:
            raise ValueError(f"missing scan pose {sid}")
        pose = poses[int(sid)]
        if not check_homogeneous(pose)["ok"]:
            raise ValueError(f"invalid pose for scan {sid}")
        valid = (points["scan_id"] == sid) & points["xyz_local_valid"]
        if not np.any(valid):
            continue
        world = apply_pose(points["xyz_local"][valid], pose)
        if not np.allclose(world, points["xyz_world"][valid], atol=atol, rtol=0):
            raise ValueError(f"local/world coordinates inconsistent for scan {sid}")
        if not np.allclose(points["scanner_origin_world"][valid], pose[:3, 3], atol=atol, rtol=0):
            raise ValueError(f"scanner origins inconsistent for scan {sid}")
    mask = points["ray_valid"]
    if np.any(mask):
        vectors = points["xyz_world"][mask] - points["scanner_origin_world"][mask]
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        if not np.allclose(vectors, points["ray_direction"][mask], atol=atol, rtol=0):
            raise ValueError("ray directions inconsistent with returned points")


def read_patch(json_path: Path) -> tuple[dict, dict]:
    meta = json.loads(Path(json_path).read_text())
    with np.load(meta["npz"]) as data:
        points = {key: data[key] for key in POINT_KEYS}
    return points, meta


def read_evaluation(json_path: Path) -> dict:
    meta = json.loads(Path(json_path).read_text())
    eval_json = Path(meta["npz"]).parent / "evaluation" / f"{meta['patch_id']}.eval.npz"
    if not eval_json.exists():
        raise FileNotFoundError(eval_json)
    out = {}
    with np.load(eval_json, allow_pickle=False) as data:
        for key in data.files:
            if key == "json":
                out.update(json.loads(bytes(data[key]).decode()))
            else:
                out[key] = data[key]
    return out


def estimator_arrays(points: dict, scan_to_frame=None):
    """Only coordinates, frame IDs and later a supplied sigma leave this helper."""
    frames = np.asarray(points["scan_id"], dtype=np.int64)
    if scan_to_frame is not None:
        frames = np.array([scan_to_frame[int(s)] for s in frames], dtype=np.int64)
    return {
        "xyz_world": np.array(points["xyz_world"], copy=True),
        "frame": frames,
        "scan_id": np.array(points["scan_id"], copy=True),
        "source_point_index": np.array(points["source_point_index"], copy=True),
    }


def pose_table(meta: dict) -> dict[int, np.ndarray]:
    table = {}
    for key, value in meta["T_world_from_scan_input"].items():
        table[int(key)] = as_matrix44(value)
    return table


def scanner_origins(meta: dict) -> dict[int, np.ndarray]:
    return {sid: origin_from_pose(matrix) for sid, matrix in pose_table(meta).items()}
