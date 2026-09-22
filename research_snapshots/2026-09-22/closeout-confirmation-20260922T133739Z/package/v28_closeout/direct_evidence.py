"""Direct, arbitrary-position plane-patch photometric evidence.

World coordinates and displacements use the same physical unit (mm here).  Each
camera is a mapping with a grayscale ``image``, physical ``P`` (3, 4), and camera
``center`` (3,).  Images are sampled directly; there is no depth/score volume,
geometry-derived visibility veto, score threshold, or evaluator input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import map_coordinates


_STD_EPS = 1e-4
_RAY_EPS = 1e-12


@dataclass(frozen=True)
class _Camera:
    image: np.ndarray
    matrix: np.ndarray
    inverse: np.ndarray
    center: np.ndarray


def _camera(camera: Mapping[str, np.ndarray]) -> _Camera:
    image = np.asarray(camera["image"], dtype=np.float64)
    projection = np.asarray(camera["P"], dtype=np.float64)
    center = np.asarray(camera["center"], dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 1:
        raise ValueError("camera image must be a nonempty grayscale [H,W] array")
    if projection.shape != (3, 4) or not np.isfinite(projection).all():
        raise ValueError("camera P must be a finite [3,4] projection matrix")
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("camera center must be a finite [3] vector")
    matrix = projection[:, :3]
    row_scale = np.linalg.norm(matrix[2])
    if not np.isfinite(row_scale) or row_scale <= 0:
        raise ValueError("camera P must have a nonzero optical-axis row")
    # A physical K[R|-RC], with positive focal lengths and proper R, has det(KR)
    # > 0.  Canonicalize its arbitrary homogeneous scale, including its sign.
    scaled = matrix / row_scale
    sign, _ = np.linalg.slogdet(scaled)
    if sign == 0:
        raise ValueError("camera P left block must be invertible")
    matrix = scaled * sign
    try:
        inverse = np.linalg.inv(matrix)
        recovered_center = -inverse @ (projection[:, 3] / row_scale * sign)
    except np.linalg.LinAlgError as error:
        raise ValueError("camera P left block must be invertible") from error
    if not np.allclose(recovered_center, center, rtol=1e-9, atol=1e-6):
        raise ValueError("camera center is inconsistent with P")
    return _Camera(image=image, matrix=matrix, inverse=inverse, center=center)


def _bilinear(image: np.ndarray, xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Sample full photographs; missing/outside/invalid samples remain NaN."""
    finite = valid & np.isfinite(xy).all(axis=-1)
    upper = np.array([image.shape[1] - 1, image.shape[0] - 1])
    # A round trip through calibrated projection can put a boundary pixel a
    # few ulps outside the image.  Snap only this numerical boundary error.
    finite &= ((xy >= -1e-9) & (xy <= upper + 1e-9)).all(axis=-1)
    coordinates = np.where(finite[..., None], np.clip(xy, 0, upper), -1.0)
    values = map_coordinates(
        image,
        [coordinates[..., 1].ravel(), coordinates[..., 0].ravel()],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    ).reshape(xy.shape[:-1])
    return np.where(finite, values, np.nan)


def _zncc(reference: np.ndarray, source: np.ndarray, min_pixels: int) -> np.ndarray:
    """ZNCC on the shared finite footprint, with a population-std floor."""
    valid = np.isfinite(reference) & np.isfinite(source)
    count = valid.sum(axis=-1)
    divisor = np.maximum(count, 1)
    reference_mean = np.where(valid, reference, 0.0).sum(axis=-1) / divisor
    source_mean = np.where(valid, source, 0.0).sum(axis=-1) / divisor
    centered_reference = np.where(valid, reference - reference_mean[..., None], 0.0)
    centered_source = np.where(valid, source - source_mean[..., None], 0.0)
    reference_ss = np.square(centered_reference).sum(axis=-1)
    source_ss = np.square(centered_source).sum(axis=-1)
    supported = (
        (count >= min_pixels)
        & (reference_ss / divisor > _STD_EPS**2)
        & (source_ss / divisor > _STD_EPS**2)
    )
    numerator = (centered_reference * centered_source).sum(axis=-1)
    denominator = np.sqrt(reference_ss * source_ss)
    score = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=score, where=supported)
    return np.clip(score, -1.0, 1.0)


def score_patches(
    points: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    ref: Mapping[str, np.ndarray],
    sources: Sequence[Mapping[str, np.ndarray]],
    mode: str = "tangent",
    patch_radius: int = 3,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Directly evaluate plane-induced image patches at physical candidates.

    ``offsets`` is [K] or [N,K] and need not be a regular grid.  A zero offset
    evaluates the exact input point through the same projection and sampling
    path as every other candidate.  Normals are [N,3] input-surface normals;
    their sign is immaterial.  ``fronto`` instead uses the reference optical
    axis.  Projection matrices may carry arbitrary nonzero homogeneous scale.

    Returns ``scores`` [S,N,K] (float32, NaN when unsupported), ``candidates``
    [N,K,3], unit ``directions`` [N,3], and ``ref_valid`` [N].  The latter
    indicates valid anchor ray/plane orientation and a textured reference
    patch; source overlap and each candidate's plane intersections can still
    be invalid.  No visibility mask is inferred from the input geometry.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    if normals.shape != points.shape:
        raise ValueError("normals must have the same [N,3] shape as points")
    n_points = len(points)
    if offsets.ndim == 1:
        offsets = np.broadcast_to(offsets[None, :], (n_points, len(offsets)))
    elif offsets.ndim != 2 or offsets.shape[0] != n_points:
        raise ValueError("offsets must have shape [K] or [N,K]")
    if not np.isfinite(offsets).all() or offsets.shape[1] < 1:
        raise ValueError("offsets must contain at least one finite candidate per point")
    if mode not in ("tangent", "fronto"):
        raise ValueError("mode must be 'tangent' or 'fronto'")
    if not isinstance(patch_radius, (int, np.integer)) or patch_radius < 0:
        raise ValueError("patch_radius must be a nonnegative integer")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    reference_camera = _camera(ref)
    source_cameras = [_camera(source) for source in sources]
    relative = points - reference_camera.center
    distances = np.linalg.norm(relative, axis=1)
    directions = np.full_like(relative, np.nan)
    direction_valid = np.isfinite(relative).all(axis=1) & (distances > _RAY_EPS)
    np.divide(relative, distances[:, None], out=directions, where=direction_valid[:, None])
    candidates = points[:, None, :] + offsets[..., None] * directions[:, None, :]
    n_candidates = offsets.shape[1]
    scores = np.full((len(source_cameras), n_points, n_candidates), np.nan, dtype=np.float32)
    ref_valid = np.zeros(n_points, dtype=bool)

    if mode == "fronto":
        plane_normals = np.broadcast_to(reference_camera.matrix[2], points.shape)
        normal_valid = np.ones(n_points, dtype=bool)
    else:
        normal_lengths = np.linalg.norm(normals, axis=1)
        normal_valid = np.isfinite(normals).all(axis=1) & (normal_lengths > _RAY_EPS)
        plane_normals = np.full_like(normals, np.nan)
        np.divide(normals, normal_lengths[:, None], out=plane_normals, where=normal_valid[:, None])

    grid = np.arange(-patch_radius, patch_radius + 1, dtype=np.float64)
    dx, dy = np.meshgrid(grid, grid, indexing="xy")
    patch_delta = np.stack([dx.ravel(), dy.ravel()], axis=-1)
    min_pixels = int(np.ceil(0.8 * len(patch_delta)))

    for start in range(0, n_points, batch_size):
        stop = min(start + batch_size, n_points)
        ray_projection = relative[start:stop] @ reference_camera.matrix.T
        base_valid = (
            direction_valid[start:stop]
            & normal_valid[start:stop]
            & np.isfinite(ray_projection).all(axis=1)
            & (ray_projection[:, 2] > _RAY_EPS)
        )
        anchor_xy = np.full((stop - start, 2), np.nan, dtype=np.float64)
        np.divide(ray_projection[:, :2], ray_projection[:, 2, None], out=anchor_xy,
                  where=base_valid[:, None])
        patch_xy = anchor_xy[:, None, :] + patch_delta[None, :, :]
        reference_values = _bilinear(reference_camera.image, patch_xy, base_valid[:, None])
        finite_reference = np.isfinite(reference_values)
        reference_count = finite_reference.sum(axis=-1)
        reference_mean = np.where(finite_reference, reference_values, 0).sum(axis=-1) / np.maximum(reference_count, 1)
        reference_variance = np.where(
            finite_reference, (reference_values - reference_mean[:, None]) ** 2, 0
        ).sum(axis=-1) / np.maximum(reference_count, 1)
        base_valid &= (reference_count >= min_pixels) & (reference_variance > _STD_EPS**2)

        homogeneous_pixels = np.concatenate([patch_xy, np.ones((*patch_xy.shape[:-1], 1))], axis=-1)
        patch_rays = homogeneous_pixels @ reference_camera.inverse.T
        patch_ray_lengths = np.linalg.norm(patch_rays, axis=-1, keepdims=True)
        np.divide(patch_rays, patch_ray_lengths, out=patch_rays,
                  where=patch_ray_lengths > _RAY_EPS)
        current_normals = plane_normals[start:stop]
        ray_dot_normal = np.einsum("blj,bj->bl", patch_rays, current_normals)
        center_dot_normal = np.einsum("bj,bj->b", directions[start:stop], current_normals)
        base_valid &= np.isfinite(center_dot_normal) & (np.abs(center_dot_normal) > _RAY_EPS)
        ref_valid[start:stop] = base_valid

        # Compute plane intersections anew for every offset; no raster-depth
        # lookup/interpolation or nearest precomputed hypothesis is involved.
        plane_distance = np.einsum(
            "bkj,bj->bk", candidates[start:stop] - reference_camera.center, current_normals
        )
        intersection_distances = np.full((stop - start, n_candidates, len(patch_delta)), np.nan)
        np.divide(plane_distance[..., None], ray_dot_normal[:, None, :],
                  out=intersection_distances,
                  where=np.abs(ray_dot_normal[:, None, :]) > _RAY_EPS)
        intersection_valid = (
            base_valid[:, None, None]
            & np.isfinite(intersection_distances)
            & (intersection_distances > _RAY_EPS)
        )
        patch_world = reference_camera.center + intersection_distances[..., None] * patch_rays[:, None, :, :]
        for source_index, source_camera in enumerate(source_cameras):
            projected = (patch_world - source_camera.center) @ source_camera.matrix.T
            source_valid = (
                intersection_valid
                & np.isfinite(projected).all(axis=-1)
                & (projected[..., 2] > _RAY_EPS)
            )
            source_xy = np.full((*projected.shape[:-1], 2), np.nan)
            np.divide(projected[..., :2], projected[..., 2, None], out=source_xy,
                      where=source_valid[..., None])
            source_values = _bilinear(source_camera.image, source_xy, source_valid)
            scores[source_index, start:stop] = _zncc(
                reference_values[:, None, :], source_values, min_pixels
            ).astype(np.float32)

    return {"scores": scores, "candidates": candidates, "directions": directions, "ref_valid": ref_valid}


def aggregate_scores(
    scores: np.ndarray, top_k: int = 3, min_sources: int = 2
) -> dict[str, np.ndarray]:
    """Return neutral cost 1 when fewer than ``min_sources`` scores are finite.

    Valid cost is one minus the mean of the highest min(top_k, finite_count)
    source scores.  ``valid`` is separate from cost; in particular an actual
    zero-correlation hypothesis is not confused with missing evidence.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError("scores must have shape [S,N,K]")
    if not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(min_sources, (int, np.integer)) or min_sources < 1:
        raise ValueError("min_sources must be a positive integer")
    finite = np.isfinite(scores)
    count = finite.sum(axis=0, dtype=np.int32)
    valid = count >= min_sources
    sorted_scores = np.sort(np.where(finite, scores, -np.inf), axis=0)
    selected = sorted_scores[-min(top_k, scores.shape[0]):] if scores.shape[0] else sorted_scores
    total = np.where(np.isfinite(selected), selected, 0.0).sum(axis=0)
    mean = total / np.maximum(np.minimum(count, top_k), 1)
    cost = np.where(valid, 1.0 - mean, 1.0).astype(np.float32)
    return {"cost": cost, "valid": valid, "count": count}
