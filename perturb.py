"""Controlled extra rigid motion about a fixed patch centre. Not a full sensor-pose model."""
from __future__ import annotations

import numpy as np

from transforms import as_matrix44


def _rotation_from_rotvec(rotvec) -> np.ndarray:
    vec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vec))
    matrix = np.eye(4)
    if angle <= 0:
        return matrix
    axis = vec / angle
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    matrix[:3, :3] = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )
    return matrix


def _rigid_about_center(rotation_4, translation, center) -> np.ndarray:
    rot = as_matrix44(rotation_4)
    center = np.asarray(center, dtype=np.float64)
    trans = np.asarray(translation, dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = rot[:3, :3]
    matrix[:3, 3] = trans + center - rot[:3, :3] @ center
    return matrix


def apply_station_transforms(xyz_world, scan_id, transforms):
    out = np.array(xyz_world, dtype=np.float64, copy=True)
    for sid, matrix in transforms.items():
        mask = np.asarray(scan_id) == int(sid)
        if not np.any(mask):
            continue
        pts = out[mask]
        out[mask] = pts @ matrix[:3, :3].T + matrix[:3, 3]
    return out


def compose_pose(extra, original):
    return as_matrix44(extra) @ as_matrix44(original)


def make_perturbation(xyz_world, scan_id, poses, seed, translation_rms_m, rotation_rms_rad, reference_scan_id):
    """Return extra T per station, perturbed points, and actual displacement stats.

    The reference station is held fixed. Remaining stations share a scaled random
    rigid motion about the fixed unperturbed patch centroid.
    """
    points = np.asarray(xyz_world, dtype=np.float64)
    scans = np.asarray(scan_id, dtype=np.int64)
    center = points.mean(axis=0)
    ids = sorted(int(s) for s in np.unique(scans))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 505]))
    trans = {sid: np.zeros(3) for sid in ids}
    rotvec = {sid: np.zeros(3) for sid in ids}
    movable = [sid for sid in ids if sid != int(reference_scan_id)]
    if movable:
        raw_t = {sid: rng.normal(size=3) for sid in movable}
        if translation_rms_m > 0:
            stacked = np.concatenate([
                np.repeat(raw_t[sid][None], int(np.sum(scans == sid)), axis=0)
                for sid in movable
            ])
            scale = translation_rms_m / np.sqrt(np.mean(np.sum(stacked ** 2, axis=1)))
            for sid in movable:
                trans[sid] = raw_t[sid] * scale
        if rotation_rms_rad > 0:
            raw_r = {sid: rng.normal(size=3) for sid in movable}
            angles = np.array([np.linalg.norm(raw_r[sid]) for sid in movable])
            # Point-weighted angle RMS.
            weights = np.concatenate([
                np.repeat(angles[i], int(np.sum(scans == sid)))
                for i, sid in enumerate(movable)
            ])
            scale = rotation_rms_rad / np.sqrt(np.mean(weights ** 2))
            for sid in movable:
                rotvec[sid] = raw_r[sid] * scale
    extras = {}
    for sid in ids:
        extras[sid] = _rigid_about_center(_rotation_from_rotvec(rotvec[sid]), trans[sid], center)
    perturbed = apply_station_transforms(points, scans, extras)
    delta = perturbed - points
    current_poses = {sid: compose_pose(extras[sid], poses[sid]) for sid in ids}
    inverse = {sid: as_matrix44(np.linalg.inv(extras[sid])) for sid in ids}
    return {
        "extra_T_world": {str(sid): extras[sid].tolist() for sid in ids},
        "inverse_extra_T_world": {str(sid): inverse[sid].tolist() for sid in ids},
        "current_T_world_from_scan": {str(sid): current_poses[sid].tolist() for sid in ids},
        "reference_scan_id": int(reference_scan_id),
        "patch_center_m": center.tolist(),
        "requested_translation_rms_m": translation_rms_m,
        "requested_rotation_rms_rad": rotation_rms_rad,
        "actual_point_delta_rms_m": float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))),
        "actual_point_delta_rms_mm": float(1000.0 * np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))),
        "note": (
            "extra rigid motion about a fixed patch centre; "
            "this is not a complete sensor-pose error model"
        ),
        "xyz_world_perturbed": perturbed,
    }
