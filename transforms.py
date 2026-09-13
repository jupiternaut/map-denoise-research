"""Pose and local-frame helpers. Units are metres unless a name says millimetres."""
from __future__ import annotations

import numpy as np


def as_matrix44(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected 4x4 pose, got {matrix.shape}")
    return np.array(matrix, copy=True)


def check_homogeneous(matrix, atol=1e-8) -> dict:
    matrix = as_matrix44(matrix)
    bottom = matrix[3]
    rotation = matrix[:3, :3]
    gram = rotation.T @ rotation
    det = float(np.linalg.det(rotation))
    ortho_err = float(np.max(np.abs(gram - np.eye(3))))
    bottom_ok = bool(np.allclose(bottom, (0.0, 0.0, 0.0, 1.0), atol=atol))
    return {
        "homogeneous_bottom_ok": bottom_ok,
        "rotation_det": det,
        "rotation_orthogonality_err": ortho_err,
        "rotation_orthogonal": bool(ortho_err <= 1e-6 and abs(det - 1.0) <= 1e-6),
        "ok": bool(bottom_ok and ortho_err <= 1e-6 and abs(det - 1.0) <= 1e-6),
    }


def apply_pose(points_local, matrix) -> np.ndarray:
    points = np.asarray(points_local, dtype=np.float64)
    matrix = as_matrix44(matrix)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be N by 3")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def invert_pose(matrix) -> np.ndarray:
    matrix = as_matrix44(matrix)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def origin_from_pose(matrix) -> np.ndarray:
    return as_matrix44(matrix)[:3, 3].copy()


def local_world_roundtrip(points_local, matrix, atol=1e-9) -> dict:
    world = apply_pose(points_local, matrix)
    recovered = apply_pose(world, invert_pose(matrix))
    err = float(np.max(np.abs(recovered - np.asarray(points_local, dtype=np.float64))))
    return {"max_abs_err_m": err, "ok": bool(err <= atol)}


def orthonormal_basis(normal) -> np.ndarray:
    axis = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("normal must be a finite nonzero vector")
    z = axis / norm
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.column_stack((x, y, z))


def world_to_local_mm(xyz_world_m, origin_m, basis) -> np.ndarray:
    centered = np.asarray(xyz_world_m, dtype=np.float64) - np.asarray(origin_m, dtype=np.float64)
    return (centered @ np.asarray(basis, dtype=np.float64)) * 1000.0


def local_mm_to_world(xyz_local_mm, origin_m, basis) -> np.ndarray:
    local_m = np.asarray(xyz_local_mm, dtype=np.float64) / 1000.0
    return local_m @ np.asarray(basis, dtype=np.float64).T + np.asarray(origin_m, dtype=np.float64)


def estimate_patch_frame(xyz_world_m) -> dict:
    """Local frame from algorithm-visible points only. Third axis is the thin direction."""
    points = np.asarray(xyz_world_m, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("need at least three points to estimate a patch frame")
    origin = points.mean(axis=0)
    centered = points - origin
    _, _, vt = np.linalg.svd(centered, full_matrices=True)
    basis = vt[:3].T
    if basis.shape != (3, 3):
        raise ValueError(f"unexpected basis shape {basis.shape}")
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1
    # Keep a stable sign for the estimated normal.
    if basis[2, 2] < 0:
        basis[:, 2] *= -1
        basis[:, 1] *= -1
    return {
        "origin_m": origin,
        "basis": basis,
        "normal_world": basis[:, 2].copy(),
        "source": "thin-direction PCA on current algorithm-visible xyz_world; no evaluation GT",
    }
