"""Known-geometry plane/step fixtures. Used to test depth localization, not as DTU proof."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .cameras import View, project_matrix, unproject_colmap_depth


def make_pinhole(name: str, index: int, center: np.ndarray, look_at: np.ndarray, fx: float = 400.0, size: int = 128) -> View:
    up = np.array([0.0, 0.0, 1.0])
    forward = look_at - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    # World-to-camera: camera X=right, Y=-up, Z=forward (OpenCV)
    R = np.stack([right, -true_up, forward], axis=0)
    t = -R @ center
    K = np.array([[fx, 0.0, size / 2.0], [0.0, fx, size / 2.0], [0.0, 0.0, 1.0]])
    P = K @ np.column_stack([R, t])
    return View(name, index, K, R, t, P, P, size, size, path=None, center_phys=center.copy())


def render_planes(view: View, planes: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]], size: int, z_far: float = 400.0) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    uv = np.column_stack([xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)])
    best_z = np.full(size * size, z_far)
    best_color = np.zeros((size * size, 3), dtype=np.uint8)
    # Ray-plane: X = C + d * D_world, (X-origin).n = 0
    pixel = np.column_stack([uv, np.ones(len(uv))])
    cam = np.linalg.inv(view.K) @ pixel.T
    cam = cam / cam[2]
    dirs = (view.R.T @ cam).T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origin = view.center_phys
    for plane_o, plane_n, color in planes:
        plane_n = plane_n / np.linalg.norm(plane_n)
        denom = dirs @ plane_n
        numer = (plane_o - origin) @ plane_n
        t = np.divide(numer, denom, out=np.full(len(dirs), np.nan), where=np.abs(denom) > 1e-8)
        ok = np.isfinite(t) & (t > 0.5)
        xyz = origin[None, :] + t[:, None] * dirs
        # Keep a bounded square on the plane so the background stays empty.
        tangent = xyz - plane_o
        helper = np.array([1.0, 0.0, 0.0]) if abs(plane_n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(plane_n, helper)
        u /= np.linalg.norm(u)
        v = np.cross(plane_n, u)
        uu = tangent @ u
        vv = tangent @ v
        ok &= (np.abs(uu) < 40) & (np.abs(vv) < 40)
        checker = ((np.floor(uu / 5.0) + np.floor(vv / 5.0)) % 2) * 50
        color_arr = np.clip(np.asarray(color, dtype=np.float64) + checker[:, None], 0, 255)
        better = ok & (t < best_z)
        best_z[better] = t[better]
        best_color[better] = color_arr[better].astype(np.uint8)
    return best_color.reshape(size, size, 3)


def textured_step_images(size: int = 128):
    """A single fronto-parallel textured plane at camera Z=80."""
    planes = [
        (np.array([0.0, 0.0, 80.0]), np.array([0.0, 0.0, 1.0]), (160, 90, 70)),
    ]
    views = [
        make_pinhole("ref.png", 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 80.0]), size=size),
        make_pinhole("src1.png", 1, np.array([10.0, 0.0, 2.0]), np.array([0.0, 0.0, 80.0]), size=size),
        make_pinhole("src2.png", 2, np.array([-8.0, 6.0, 3.0]), np.array([0.0, 0.0, 80.0]), size=size),
    ]
    images = [render_planes(view, planes, size) for view in views]
    return views, images, planes
