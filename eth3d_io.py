"""ETH3D per-station PLY + MeshLab alignment. Local origin is the scanner origin."""
from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d

from hashutil import sha256_file
from transforms import apply_pose, as_matrix44, check_homogeneous, origin_from_pose


def _scan_id(name: str, fallback: int) -> int:
    match = re.search(r"(\d+)", Path(name).stem)
    return int(match.group(1)) if match else int(fallback)


def parse_mlp(path: Path) -> list[dict]:
    text = Path(path).read_text(errors="replace")
    root = ET.fromstring(text)
    stations = []
    for mesh in root.iter("MLMesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        matrix_node = None
        for child in mesh.iter():
            if child.tag.endswith("MLMatrix44"):
                matrix_node = child
                break
        if matrix_node is None or not (matrix_node.text or "").strip():
            raise ValueError(f"missing MLMatrix44 for {filename} in {path}")
        numbers = [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", matrix_node.text)]
        if len(numbers) != 16:
            raise ValueError(f"MLMatrix44 for {filename} has {len(numbers)} numbers")
        matrix = as_matrix44(numbers)
        check = check_homogeneous(matrix)
        stations.append(
            {
                "filename": filename,
                "name": Path(filename).name,
                "matrix": matrix,
                "origin_world": origin_from_pose(matrix),
                "pose_check": check,
            }
        )
    if not stations:
        raise ValueError(f"no MLMesh stations in {path}")
    return stations


def load_ply_xyz(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"empty PLY: {path}")
    return np.asarray(cloud.points, dtype=np.float64)


def inspect_scene(extracted_root: Path, scene: str, variant: str) -> dict:
    root = Path(extracted_root) / scene / variant
    candidates = list(root.rglob("scan_alignment.mlp"))
    if not candidates:
        # some archives unpack as scene/scans + mlp
        candidates = list(root.rglob("*.mlp"))
    if not candidates:
        ply_only = sorted(p for p in root.rglob("*.ply") if p.is_file())
        return {
            "scene": scene,
            "variant": variant,
            "root": str(root),
            "status": "no_alignment_mlp",
            "ply_files": [str(p) for p in ply_only],
            "usable_for_frame_bias": False,
        }
    mlp = candidates[0]
    stations = parse_mlp(mlp)
    records = []
    for index, station in enumerate(stations):
        ply = (mlp.parent / station["filename"]).resolve()
        if not ply.exists():
            alt = next(root.rglob(station["name"]), None)
            ply = alt if alt else ply
        rec = {
            "scan_id": _scan_id(station["name"], index),
            "name": station["name"],
            "filename": station["filename"],
            "ply": str(ply),
            "ply_exists": ply.exists(),
            "T_world_from_scan": station["matrix"].tolist(),
            "scanner_origin_world": station["origin_world"].tolist(),
            "pose_check": station["pose_check"],
        }
        if ply.exists():
            rec["ply_sha256"] = sha256_file(ply)
            rec["ply_bytes"] = ply.stat().st_size
            xyz = load_ply_xyz(ply)
            rec["n_points"] = int(len(xyz))
            rec["local_centroid"] = xyz.mean(axis=0).tolist()
            rec["local_min"] = xyz.min(axis=0).tolist()
            rec["local_max"] = xyz.max(axis=0).tolist()
            # Official docs: scanner origin is the PLY coordinate origin.
            rec["local_centroid_norm_m"] = float(np.linalg.norm(xyz.mean(axis=0)))
            rec["scanner_origin_is_ply_origin"] = True
            rec["median_range_m"] = float(np.median(np.linalg.norm(xyz, axis=1)))
            world = apply_pose(xyz[:: max(1, len(xyz) // 20000)], station["matrix"])
            rec["world_centroid_sample"] = world.mean(axis=0).tolist()
            rec["already_in_world_warning"] = bool(
                np.linalg.norm(xyz.mean(axis=0) - station["origin_world"]) < 0.25
                and np.linalg.norm(station["origin_world"]) > 1.0
            )
        records.append(rec)
    return {
        "scene": scene,
        "variant": variant,
        "root": str(root),
        "mlp": str(mlp),
        "mlp_sha256": sha256_file(mlp),
        "n_stations": len(records),
        "stations": records,
        "all_poses_ok": all(r["pose_check"]["ok"] for r in records),
        "usable_for_frame_bias": all(r.get("ply_exists") and r["pose_check"]["ok"] for r in records) and len(records) >= 2,
        "units_assumed": "metres, as published by ETH3D",
        "pose_applied_once": "PLY stays scanner-local; world = T_world_from_scan @ p_local",
    }
