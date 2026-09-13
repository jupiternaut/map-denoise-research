"""ETH3D overlap patches. Boxes come from multi-station voxels, not method scores."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile

import numpy as np
import open3d as o3d

from eth3d_io import inspect_scene, load_ply_xyz
from hashutil import dump_json
from paths import ETH3D_EXTRACTED, PREVIEW, PATCHES, require_liekkas
from schema import POINT_KEYS, empty_points, fill_rays, write_patch
from transforms import apply_pose, as_matrix44, origin_from_pose

PATCH_SPECS = {
    "delivery_area": [
        {"patch_id": "da_wall", "kind": "wall", "slot": 0},
        {"patch_id": "da_junction", "kind": "junction", "slot": 1},
        {"patch_id": "da_thin", "kind": "thin_candidate", "slot": 2},
    ],
    "courtyard": [
        {"patch_id": "cy_wall", "kind": "wall", "slot": 0},
        {"patch_id": "cy_junction", "kind": "junction", "slot": 1},
        {"patch_id": "cy_thin", "kind": "thin_candidate", "slot": 2},
    ],
}


def _voxel_keys(xyz, size):
    return np.floor(np.asarray(xyz, dtype=np.float64) / size).astype(np.int64)


def overlap_stats(station_world, voxel_m=0.10) -> dict:
    occupancy = {}
    for sid, xyz in station_world.items():
        keys = np.unique(_voxel_keys(xyz, voxel_m), axis=0)
        for key in map(tuple, keys):
            occupancy.setdefault(key, set()).add(int(sid))
    counts = np.array([len(v) for v in occupancy.values()], dtype=np.int64) if occupancy else np.zeros(0, dtype=np.int64)
    pairs = {}
    ids = sorted(station_world)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            pairs[f"{a}-{b}"] = int(sum(1 for s in occupancy.values() if a in s and b in s))
    return {
        "voxel_m": voxel_m,
        "occupied_voxels": int(len(occupancy)),
        "voxels_with_ge2": int(np.sum(counts >= 2)) if len(counts) else 0,
        "voxels_with_ge3": int(np.sum(counts >= 3)) if len(counts) else 0,
        "max_stations_in_voxel": int(counts.max()) if len(counts) else 0,
        "pairwise_overlap_voxels": pairs,
    }


def overlap_points(station_world, voxel_m=0.10):
    occupancy = {}
    for sid, xyz in station_world.items():
        keys = _voxel_keys(xyz, voxel_m)
        for key in map(tuple, np.unique(keys, axis=0)):
            occupancy.setdefault(key, set()).add(int(sid))
    multi = {key for key, sids in occupancy.items() if len(sids) >= 2}
    kept = {}
    for sid, xyz in station_world.items():
        keys = _voxel_keys(xyz, voxel_m)
        mask = np.array([tuple(k) in multi for k in keys], dtype=bool)
        kept[int(sid)] = xyz[mask]
    if not any(len(v) for v in kept.values()):
        return None
    return np.concatenate([v for v in kept.values() if len(v)])


def _aabb(points, pad=0.05):
    return points.min(axis=0) - pad, points.max(axis=0) + pad


def _vertical_peaks(overlap_xyz, bins=160):
    """Sharp XY histogram peaks. Floors are ignored by only searching X and Y."""
    peaks = []
    for axis in (0, 1):
        values = overlap_xyz[:, axis]
        hist, edges = np.histogram(values, bins=bins)
        if hist.max() <= 0:
            continue
        peak = int(np.argmax(hist))
        centre = 0.5 * (edges[peak] + edges[peak + 1])
        sharpness = float(hist[peak]) / float(max(hist.mean(), 1.0))
        peaks.append({"axis": axis, "centre": centre, "count": int(hist[peak]), "sharpness": sharpness})
    peaks.sort(key=lambda p: p["sharpness"], reverse=True)
    return peaks


def _copy_bounds(lo, hi):
    return np.array(lo, dtype=np.float64, copy=True), np.array(hi, dtype=np.float64, copy=True)


def _along_wall_slab(wall_lo, wall_hi, other, centre, half=1.2):
    """Keep the wall-normal thickness; crop a local span along the wall."""
    lo, hi = _copy_bounds(wall_lo, wall_hi)
    lo[other] = max(float(centre) - half, float(wall_lo[other]))
    hi[other] = min(float(centre) + half, float(wall_hi[other]))
    return lo, hi


def overlap_boxes(overlap_xyz, n=3):
    """Wall / junction-or-second-slab / local candidate from overlap geometry only.

    Returns (primary_boxes, extra_junction_candidates). Boxes are chosen by
    histogram geometry, not method scores. A same-axis second slab is not a
    thin-layer label.
    """
    if overlap_xyz is None or len(overlap_xyz) < 500:
        return [], []
    z0, z1 = np.quantile(overlap_xyz[:, 2], [0.15, 0.85])
    peaks = _vertical_peaks(overlap_xyz)
    if not peaks:
        return [], []
    wall = peaks[0]
    lo = overlap_xyz.min(axis=0).copy()
    hi = overlap_xyz.max(axis=0).copy()
    lo[wall["axis"]] = wall["centre"] - 0.15
    hi[wall["axis"]] = wall["centre"] + 0.15
    lo[2], hi[2] = z0, z1
    other = 1 - wall["axis"]
    mid = float(np.median(overlap_xyz[:, other]))
    lo[other] = mid - 4.0
    hi[other] = mid + 4.0
    boxes = [{"lo": lo, "hi": hi, "role": "wall_slab"}]
    extras = []
    thin_lo, thin_hi = _copy_bounds(lo, hi)
    thin_lo[other] = thin_hi[other] - 1.2
    # Both X and Y always have some histogram argmax. A second axis is a
    # junction only if that peak is sharp and the crossing box is populated.
    other_peak = next((p for p in peaks[1:] if p["axis"] != wall["axis"]), None)
    used_ortho = False
    if other_peak is not None and other_peak["sharpness"] >= 2.5:
        jlo = overlap_xyz.mean(axis=0) - 1.2
        jhi = overlap_xyz.mean(axis=0) + 1.2
        jlo[other_peak["axis"]] = other_peak["centre"] - 1.2
        jhi[other_peak["axis"]] = other_peak["centre"] + 1.2
        jlo[wall["axis"]] = wall["centre"] - 1.2
        jhi[wall["axis"]] = wall["centre"] + 1.2
        n_cross = int(np.sum(np.all((overlap_xyz >= jlo) & (overlap_xyz <= jhi), axis=1)))
        if n_cross >= 500:
            boxes.append({"lo": jlo, "hi": jhi, "role": "orthogonal_crossing"})
            used_ortho = True
    if not used_ortho:
        # Same-axis fallback: quantile of WALL-SLAB points, then clamp to the wall.
        # Using the whole-scene overlap quantile can place the box off the slab.
        wall_mask = np.all((overlap_xyz >= lo) & (overlap_xyz <= hi), axis=1)
        along = overlap_xyz[wall_mask, other] if np.any(wall_mask) else overlap_xyz[:, other]
        q = float(np.quantile(along, 0.28))
        jlo, jhi = _along_wall_slab(lo, hi, other, q, half=1.2)
        boxes.append({"lo": jlo, "hi": jhi, "role": "same_axis_second_slab"})
        for qlev, half in (
            (0.18, 1.2),
            (0.22, 1.2),
            (0.35, 1.2),
            (0.42, 1.2),
            (0.55, 1.2),
            (0.28, 1.6),
            (0.20, 1.6),
        ):
            cq = float(np.quantile(along, qlev))
            elo, ehi = _along_wall_slab(lo, hi, other, cq, half=half)
            extras.append({"lo": elo, "hi": ehi, "role": "same_axis_second_slab"})
    boxes.append({"lo": thin_lo, "hi": thin_hi, "role": "local_end_slab"})
    return boxes[:n], extras


def _remove_stale_patch(directory: Path, patch_id: str) -> None:
    """Never delete an old patch: new extracts must use a fresh directory."""
    directory = Path(directory)
    for path in (
        directory / f"{patch_id}.json",
        directory / f"{patch_id}.npz",
        directory / "evaluation" / f"{patch_id}.eval.json",
        directory / "evaluation" / f"{patch_id}.eval.npz",
    ):
        if path.exists():
            raise FileExistsError(f"existing patch {path}; use a new extraction directory")


def load_scene_stations(scene: str, variant: str, stride: int = 12) -> dict:
    info = inspect_scene(ETH3D_EXTRACTED, scene, variant)
    if not info.get("usable_for_frame_bias"):
        return {"info": info, "world": {}, "local": {}, "poses": {}}
    world = {}
    local = {}
    poses = {}
    for rec in info["stations"]:
        xyz = load_ply_xyz(Path(rec["ply"]))
        if stride > 1:
            xyz = xyz[::stride]
        matrix = as_matrix44(rec["T_world_from_scan"])
        local[rec["scan_id"]] = xyz
        world[rec["scan_id"]] = apply_pose(xyz, matrix)
        poses[rec["scan_id"]] = matrix
    return {"info": info, "world": world, "local": local, "poses": poses, "stride": stride}


def crop(station_world, station_local, lo, hi, max_per_station):
    kept = []
    for sid, xyz in station_world.items():
        mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        if len(idx) > max_per_station:
            sel = np.linspace(0, len(idx) - 1, max_per_station).astype(np.int64)
            idx = idx[sel]
        kept.append((int(sid), idx, xyz[idx], station_local[sid][idx]))
    return kept


def extract_named_patch(scene, variant, spec, loaded, box, max_per_station=800):
    lo, hi = box
    kept = crop(loaded["world"], loaded["local"], lo, hi, max_per_station)
    if len(kept) < 2:
        return None
    parts = []
    for sid, idx, xyz_w, xyz_l in kept:
        n = len(idx)
        block = empty_points(n)
        block["xyz_world"] = xyz_w
        block["xyz_local"] = xyz_l
        block["scan_id"] = np.full(n, sid, dtype=np.int64)
        block["source_point_index"] = (idx * int(loaded.get("stride") or 1)).astype(np.int64)
        block["scanner_origin_world"] = np.repeat(origin_from_pose(loaded["poses"][sid])[None], n, axis=0)
        block["xyz_local_valid"] = np.ones(n, dtype=np.bool_)
        fill_rays(block)
        for key in POINT_KEYS:
            value = np.asarray(block[key])
            if value.ndim == 0:
                raise ValueError(f"{spec['patch_id']} scan {sid} key {key} is scalar {value!r}")
            block[key] = value
        parts.append(block)
    points = {key: np.concatenate([p[key] for p in parts], axis=0) for key in POINT_KEYS}
    points["ray_note"] = parts[0].get("ray_note")
    if len(points["xyz_world"]) > 3000:
        rng = np.random.default_rng(912301)
        pick = np.sort(rng.choice(len(points["xyz_world"]), 2400, replace=False))
        points = {key: (points[key][pick] if key in POINT_KEYS else points[key]) for key in points}
    if len(points["xyz_world"]) < 300:
        return None
    poses = {str(sid): loaded["poses"][sid].tolist() for sid, *_ in kept}
    station_counts = {int(sid): int(np.sum(points["scan_id"] == sid)) for sid, *_ in kept}
    meta = {
        "dataset": "eth3d",
        "scene": scene,
        "variant": variant,
        "kind": spec["kind"],
        "crop_world_min_m": np.asarray(lo).tolist(),
        "crop_world_max_m": np.asarray(hi).tolist(),
        "sampling": {
            "load_stride": loaded.get("stride"),
            "max_per_station": max_per_station,
            "rule": (
                "overlap histogram wall slab; orthogonal crossing or same-axis "
                "second slab; stride load; even subsample in-box; "
                "source_point_index is original PLY row"
            ),
        },
        "T_world_from_scan_input": poses,
        "station_counts": station_counts,
        "source_mlp": loaded["info"].get("mlp"),
        "source_mlp_sha256": loaded["info"].get("mlp_sha256"),
        "thin_layer_label": "not_invented; no independent thin-layer GT on ETH3D raw/clean",
        "raw_clean_note": "raw and clean are the same scanner source; clean is cleaned, not independent millimetre GT",
        "pose_applied_once": True,
    }
    evaluation = {
        "reference_T_world_from_scan": poses,
        "independent_geometry_gt": False,
        "independent_geometry_note": "本轮尚未测得",
    }
    return points, meta, evaluation, {"lo": np.asarray(lo).tolist(), "hi": np.asarray(hi).tolist(), "station_counts": station_counts}


def preview_scene(scene, variant, loaded, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for sid, xyz in loaded["world"].items():
        sample = xyz[:: max(1, len(xyz) // 8000)]
        ax.scatter(sample[:, 0], sample[:, 1], s=0.4, label=f"scan {sid}")
        origin = origin_from_pose(loaded["poses"][sid])
        ax.scatter([origin[0]], [origin[1]], s=40, marker="x")
        ax.text(origin[0], origin[1], f"S{sid}")
    ax.set_aspect("equal")
    ax.set_xlabel("x world / m")
    ax.set_ylabel("y world / m")
    ax.set_title(f"ETH3D {scene} {variant} stations")
    ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(dest / f"{scene}_{variant}_stations_xy.png", dpi=140)
    plt.close(fig)
    clouds = []
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(loaded["world"]), 1)))
    for i, (sid, xyz) in enumerate(loaded["world"].items()):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz[:: max(1, len(xyz) // 40000)])
        cloud.paint_uniform_color(colors[i][:3])
        clouds.append(cloud)
    if clouds:
        merged = clouds[0]
        for extra in clouds[1:]:
            merged += extra
        o3d.io.write_point_cloud(str(dest / f"{scene}_{variant}_preview.ply"), merged)


def run(variant="clean", stride=12, destination=None):
    global PATCHES, PREVIEW
    require_liekkas()
    if destination is None:
        PATCHES.parent.mkdir(parents=True, exist_ok=True)
        destination = tempfile.mkdtemp(prefix="extraction-v2-", dir=PATCHES.parent)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"extraction destination not empty: {destination}")
    PATCHES, PREVIEW = destination / "patches", destination / "preview"
    PATCHES.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    report = {"variant_used_for_patches": variant, "load_stride": stride, "scenes": {}}
    for scene in ("delivery_area", "courtyard"):
        root = ETH3D_EXTRACTED / scene / variant
        if not root.exists():
            report["scenes"][scene] = {"status": "not_extracted_yet"}
            continue
        loaded = load_scene_stations(scene, variant, stride=stride)
        info = loaded["info"]
        scene_rep = {"inspect": {k: info[k] for k in info if k != "stations"}}
        scene_rep["stations"] = [
            {k: v for k, v in rec.items() if k != "T_world_from_scan"}
            for rec in info.get("stations", [])
        ]
        if loaded["world"]:
            scene_rep["overlap"] = overlap_stats(loaded["world"])
            preview_scene(scene, variant, loaded, PREVIEW / scene)
            ov = overlap_points(loaded["world"])
            boxes, extras = overlap_boxes(ov, n=3)
            extracted = []
            for spec in PATCH_SPECS[scene]:
                candidates = []
                if spec["slot"] < len(boxes):
                    candidates.append(boxes[spec["slot"]])
                if spec["kind"] == "junction":
                    candidates.extend(extras)
                got = None
                used = None
                for cand in candidates:
                    got = extract_named_patch(scene, variant, spec, loaded, (cand["lo"], cand["hi"]))
                    if got is not None:
                        used = cand
                        break
                if got is None:
                    _remove_stale_patch(PATCHES / scene, spec["patch_id"])
                    extracted.append({"patch_id": spec["patch_id"], "status": "insufficient_overlap_or_points"})
                    continue
                points, meta, ev, box = got
                meta["geometry_role"] = used["role"]
                if used["role"] == "same_axis_second_slab":
                    meta["kind_note"] = (
                        "no orthogonal histogram peak; second local slab on the "
                        "same wall, not a verified multi-surface junction and "
                        "not a thin-layer label"
                    )
                paths = write_patch(PATCHES / scene, spec["patch_id"], points, meta, ev)
                extracted.append({
                    "status": "ok",
                    "box": box,
                    "geometry_role": used["role"],
                    **paths,
                    "n": int(len(points["xyz_world"])),
                })
            scene_rep["patches"] = extracted
        report["scenes"][scene] = scene_rep
    dump_json(PATCHES / "EXTRACT_REPORT.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str)[:5000])
