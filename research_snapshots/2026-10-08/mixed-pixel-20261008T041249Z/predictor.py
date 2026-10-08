"""Frozen image auxiliaries and raw sensor-pixel mixture prediction.

This module has no fixture, renderer, scene-label, or evaluation-truth imports.
Coordinates are (x, y); depth is optical Z in the reference camera.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree, QhullError


METHOD_VERSION = "image-rectangle-linear-v1"
DEFAULTS = {
    "query": (64.0, 64.0),
    "ref_window": 1,
    "patch_radius": 20,
    "outer_inner_radius": 12,
    "contrast_floor": 5.0,
    "contrast_mad_multiplier": 3.0,
    "background_ncc_min": 0.2,
    "background_std_min": 1.0,
    "sigma_block_size": 2,
    "sigma_guard": 2,
    "sigma_min_blocks": 4,
    "sigma_min_samples": 12,
    "sigma_min_remaining": 8,
    "depth_chunk": 64,
}
_OFFSET = np.array([(x, y) for y in (-1 / 3, 0, 1 / 3)
                    for x in (-1 / 3, 0, 1 / 3)], dtype=float)


def _params(params):
    p = DEFAULTS.copy()
    if params is not None:
        p.update(params)
    return p


def _camera(cam):
    return (np.asarray(cam["K"], float), np.asarray(cam["R"], float),
            np.asarray(cam["C"], float))


def _mad(values):
    values = np.asarray(values, float)
    return float(np.median(np.abs(values - np.median(values))))


def _sample(field, xy):
    xy = np.asarray(xy, float)
    return ndimage.map_coordinates(np.asarray(field, float),
                                  [xy[..., 1], xy[..., 0]], order=1,
                                  mode="nearest", prefilter=False)


def project_reference_pixels(xy, depths, ref_cam, target_cam):
    """Project reference coordinates at optical depths; output (D,N,2)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    depths = np.atleast_1d(np.asarray(depths, float))
    kr, rr, cr = _camera(ref_cam)
    kt, rt, ct = _camera(target_cam)
    rays = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(kr).T
    rays /= rays[:, 2:3]
    world = (rays[None] * depths[:, None, None]) @ rr + cr
    camera = (world - ct) @ rt.T
    homogeneous = camera @ kt.T
    uv = homogeneous[..., :2] / homogeneous[..., 2:3]
    return uv, camera[..., 2] > 0


def _reference_corners(params):
    p = _params(params)
    qx, qy = p["query"]
    w = np.asarray(p["ref_window"], float)
    if w.ndim == 0:
        x0, x1, y0, y1 = -float(w), float(w), -float(w), float(w)
    elif w.shape == (4,):
        x0, x1, y0, y1 = w
    elif w.ndim == 2 and w.shape[1] == 2:
        x0, y0 = np.min(w, axis=0)
        x1, y1 = np.max(w, axis=0)
    else:
        raise ValueError("ref_window must be a radius, (x0,x1,y0,y1), or Nx2 offsets")
    return np.array([[qx+x0-.5, qy+y0-.5], [qx+x1+.5, qy+y0-.5],
                     [qx+x1+.5, qy+y1+.5], [qx+x0-.5, qy+y1+.5]])


def sensor_roi(ref_cam, eval_cam, grid, shape=(128, 128), params=None):
    """Fixed integer-pixel bounding ROI of the public search projection union.

    Depends on cameras, image dimensions, public window, and the complete grid;
    it never depends on the image, auxiliaries, candidate ownership, or incumbent.
    Pixel squares intersecting that projected bounding box are included.
    """
    grid = np.asarray(grid, float)
    if grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError("grid must contain finite positive optical depths")
    uv, front = project_reference_pixels(_reference_corners(params), grid, ref_cam, eval_cam)
    if not np.all(front) or not np.all(np.isfinite(uv)):
        return np.empty((0, 2), dtype=int)
    lo = np.ceil(np.min(uv, axis=(0, 1)) - .5).astype(int)
    hi = np.floor(np.max(uv, axis=(0, 1)) + .5).astype(int)
    lo = np.maximum(lo, [0, 0])
    hi = np.minimum(hi, [shape[1]-1, shape[0]-1])
    if np.any(hi < lo):
        return np.empty((0, 2), dtype=int)
    xx, yy = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
    return np.column_stack((xx.ravel(), yy.ravel()))


def _inverse_plane(eval_xy, depths, ref_cam, eval_cam):
    """Intersect evaluation rays with reference-Z planes, including infinity."""
    kr, rr, cr = _camera(ref_cam)
    ke, re, ce = _camera(eval_cam)
    eval_xy = np.asarray(eval_xy, float)
    ray_camera = np.concatenate((eval_xy, np.ones(eval_xy.shape[:-1]+(1,))), axis=-1)
    directions = (ray_camera @ np.linalg.inv(ke).T) @ re @ rr.T
    origin = (ce-cr) @ rr.T
    depths = np.atleast_1d(np.asarray(depths, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (depths[:, None, None] - origin[2]) / directions[None, ..., 2]
        xyz = origin + distance[..., None] * directions[None]
        infinite = np.isposinf(depths)
        xyz[infinite] = directions
        h = xyz @ kr.T
        uv = h[..., :2] / h[..., 2:3]
    valid = (distance > 0) & np.all(np.isfinite(uv), axis=-1)
    return uv, distance, valid


def _inside(aux, uv):
    if str(aux["mode"]) == "single":
        return np.ones(uv.shape[:-1], dtype=bool)
    polygon = np.asarray(aux.get("polygon", np.empty((0, 2))), float)
    if len(polygon) >= 3:
        positive = np.ones(uv.shape[:-1], dtype=bool)
        negative = positive.copy()
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            cross = (b[0]-a[0])*(uv[..., 1]-a[1]) - (b[1]-a[1])*(uv[..., 0]-a[0])
            positive &= cross >= -1e-10
            negative &= cross <= 1e-10
        return positive | negative
    return _sample(aux["mask"], uv) >= .5


def predict_pixels(aux, ref_cam, eval_cam, grid, incumbent, params=None, mode="dynamic"):
    """Return raw pixel predictions (D,N), alpha, and per-subray ownership.

    The fixed arm freezes ownership INCLUDING visibility/depth order at incumbent,
    while foreground texture coordinates are recomputed at every candidate depth.
    """
    p = _params(params)
    grid = np.asarray(grid, float)
    if mode not in ("dynamic", "fixed"):
        raise ValueError("mode must be dynamic or fixed")
    shape = np.asarray(aux["foreground"]).shape
    roi = sensor_roi(ref_cam, eval_cam, grid, shape, p)
    n = len(roi)
    prediction = np.full((len(grid), n), np.nan)
    ownership = np.zeros((len(grid), n, 9), dtype=bool)
    depth_valid = np.zeros(len(grid), dtype=bool)
    if not bool(aux.get("valid", False)) or str(aux["mode"]) == "unavailable" or n == 0:
        return {"prediction": prediction, "alpha": ownership.mean(axis=-1),
                "ownership": ownership, "roi": roi, "depth_valid": depth_valid}
    rays = roi[:, None, :] + _OFFSET[None]
    background_depth = float(aux.get("background_depth", np.inf))
    uv_b, t_b, ok_b = _inverse_plane(rays, [background_depth], ref_cam, eval_cam)
    back = _sample(aux["background"], uv_b[0])
    single = str(aux["mode"]) == "single"
    fixed = None
    if mode == "fixed":
        uv_0, t_0, ok_0 = _inverse_plane(rays, [float(incumbent)], ref_cam, eval_cam)
        fixed = _inside(aux, uv_0[0]) & (single | (t_0[0] < t_b[0]))
        fixed_valid = bool(np.all(ok_0))
    for start in range(0, len(grid), int(p["depth_chunk"])):
        stop = min(start+int(p["depth_chunk"]), len(grid))
        uv_f, t_f, ok_f = _inverse_plane(rays, grid[start:stop], ref_cam, eval_cam)
        cover = _inside(aux, uv_f) & (single | (t_f < t_b)) if fixed is None else np.broadcast_to(fixed, ok_f.shape)
        foreground = _sample(aux["foreground"], uv_f)
        prediction[start:stop] = np.mean(np.where(cover, foreground, back[None]), axis=-1)
        ownership[start:stop] = cover
        depth_valid[start:stop] = np.all(ok_f, axis=(1, 2)) & (single or bool(np.all(ok_b)))
        if fixed is not None:
            depth_valid[start:stop] &= fixed_valid
    prediction[~depth_valid] = np.nan
    return {"prediction": prediction, "alpha": ownership.mean(axis=-1),
            "ownership": ownership, "roi": roi, "depth_valid": depth_valid}


def predict_curve(aux, ref_cam, eval_cam, eval_image, grid, incumbent, params=None, mode="dynamic"):
    """MSE on one frozen set of observed raw integer sensor pixels."""
    image = np.asarray(eval_image, float)
    pixels = predict_pixels(aux, ref_cam, eval_cam, grid, incumbent, params, mode)
    roi = pixels["roi"]
    valid = bool(len(roi) and np.all(pixels["depth_valid"]))
    loss = np.full(len(grid), np.nan)
    if valid:
        observed = image[roi[:, 1], roi[:, 0]]
        valid = bool(np.all(np.isfinite(observed)))
        if valid:
            loss = np.mean((pixels["prediction"] - observed[None])**2, axis=1)
    sigma_valid = bool(aux.get("sigma_valid", False))
    sigma = float(aux.get("sigma", np.nan))
    reason = "ok" if valid else "auxiliary_or_projection_unavailable"
    return {"loss": loss, "sigma": sigma, "sigma_valid": sigma_valid,
            "valid": valid, "reason": reason, "pixel_count": len(roi),
            "roi": roi, "flat": bool(valid and np.ptp(loss) <= 1e-12),
            "auxiliary_valid": bool(aux.get("valid", False)),
            "scale_reason": "ok" if sigma_valid else "training_scale_unavailable"}


def _texture_field(donor_xy, values, shape, output_xy=None):
    """Linear triangulation on frozen donors; nearest donor outside convex hull."""
    donor_xy = np.asarray(donor_xy, float)
    values = np.asarray(values, float)
    if not len(donor_xy):
        raise ValueError("no appearance donors")
    if output_xy is None:
        yy, xx = np.indices(shape)
        output_xy = np.column_stack((xx.ravel(), yy.ravel()))
        outshape = shape
    else:
        output_xy = np.asarray(output_xy, float)
        outshape = output_xy.shape[:-1]
        output_xy = output_xy.reshape(-1, 2)
    nearest = cKDTree(donor_xy).query(output_xy, workers=1)[1]
    result = values[nearest]
    if len(donor_xy) >= 3:
        try:
            linear = LinearNDInterpolator(donor_xy, values)(output_xy)
            finite = np.isfinite(linear)
            result[finite] = linear[finite]
        except QhullError:
            pass  # Collinear donors have the same declared nearest extension.
    return result.reshape(outshape)


def _held_block_residuals(donor_xy, values, shape, p):
    donor_xy = np.asarray(donor_xy, int)
    values = np.asarray(values, float)
    block, guard = int(p["sigma_block_size"]), int(p["sigma_guard"])
    stride = block+2*guard
    residuals, blocks = [], 0
    # Disjoint halo tiles on a fixed sensor-coordinate lattice.
    for y in range(guard, shape[0]-block+1, stride):
        for x in range(guard, shape[1]-block+1, stride):
            held = ((donor_xy[:, 0] >= x) & (donor_xy[:, 0] < x+block) &
                    (donor_xy[:, 1] >= y) & (donor_xy[:, 1] < y+block))
            if np.count_nonzero(held) < 3:
                continue
            excluded = ((donor_xy[:, 0] >= x-guard) & (donor_xy[:, 0] < x+block+guard) &
                        (donor_xy[:, 1] >= y-guard) & (donor_xy[:, 1] < y+block+guard))
            if np.count_nonzero(~excluded) < int(p["sigma_min_remaining"]):
                continue
            pred = _texture_field(donor_xy[~excluded], values[~excluded], shape, donor_xy[held])
            residuals.extend((values[held]-pred).tolist())
            blocks += 1
    valid = blocks >= int(p["sigma_min_blocks"]) and len(residuals) >= int(p["sigma_min_samples"])
    return np.asarray(residuals), {"blocks": blocks, "samples": len(residuals), "valid": bool(valid)}


def _rectangle_from_component(alpha, component, x0, y0):
    """Interpolate the connected half-level contour, then fit one PCA rectangle."""
    points = []
    # Each connected inside pixel contributes its crossing toward adjacent outside.
    for y, x in np.argwhere(component):
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            ny, nx = y+dy, x+dx
            if 0 <= ny < alpha.shape[0] and 0 <= nx < alpha.shape[1] and not component[ny, nx]:
                a, b = alpha[y, x], alpha[ny, nx]
                fraction = np.clip((a-.5)/(a-b), 0., 1.) if abs(a-b) > 1e-12 else .5
                points.append((x0+x+fraction*dx, y0+y+fraction*dy))
    points = np.asarray(points, float)
    if len(points) < 4:
        raise ValueError("insufficient boundary samples")
    yy, xx = np.nonzero(component)
    centres = np.column_stack((xx+x0, yy+y0)).astype(float)
    centre = np.mean(centres, axis=0)
    _, vectors = np.linalg.eigh(np.cov((centres-centre).T))
    uv = (points-centre) @ vectors
    lo, hi = np.min(uv, axis=0), np.max(uv, axis=0)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return corners @ vectors.T + centre


def _estimate_background(ref, train, xy, ref_cam, train_cam, p):
    values = ref[xy[:, 1], xy[:, 0]]
    sd = float(np.std(values))
    if sd < float(p["background_std_min"]):
        return np.inf, {"valid": False, "usable": True, "geometry": "constant_background",
                        "reason": "outer_ring_has_no_depth_texture", "reference_std": sd}
    grid = np.asarray(p.get("background_grid", np.arange(300., 1001., 2.)), float)
    uv, positive = project_reference_pixels(xy, grid, ref_cam, train_cam)
    h, w = train.shape
    inside = positive & (uv[..., 0] >= 0) & (uv[..., 0] <= w-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h-1)
    common = np.all(inside, axis=0)
    if np.count_nonzero(common) < 32:
        return np.nan, {"valid": False, "usable": False, "geometry": "unavailable", "reason": "insufficient_common_outer_pixels"}
    a = values[common] - np.mean(values[common])
    b = _sample(train, uv[:, common])
    b -= np.mean(b, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(a*a) * np.sum(b*b, axis=1))
    ncc = np.divide(np.sum(b*a, axis=1), denom, out=np.full(len(grid), -np.inf), where=denom > 1e-12)
    index = int(np.argmax(ncc))
    valid = bool(np.isfinite(ncc[index]) and ncc[index] >= float(p["background_ncc_min"]))
    return (float(grid[index]) if valid else np.nan), {
        "valid": valid, "usable": valid, "geometry": "outer_ring_ncc",
        "reason": "ok" if valid else "outer_ring_ncc_insufficient",
        "reference_std": sd, "pixel_count": int(np.count_nonzero(common)),
        "best_ncc": float(ncc[index]) if np.isfinite(ncc[index]) else None,
        "grid_count": len(grid), "depth": float(grid[index]) if valid else None}


def estimate_aux(ref, train, ref_cam, train_cam, params=None):
    """Estimate once from reference + one training view only; no scoring input."""
    p = _params(params)
    ref, train = np.asarray(ref, float), np.asarray(train, float)
    if ref.ndim != 2 or train.shape != ref.shape or not np.all(np.isfinite(ref)) or not np.all(np.isfinite(train)):
        raise ValueError("reference and training inputs must be finite, same-size gray images")
    shape = ref.shape
    qx, qy = np.rint(p["query"]).astype(int)
    radius = int(p["patch_radius"])
    x0, x1 = max(0, qx-radius), min(shape[1], qx+radius+1)
    y0, y1 = max(0, qy-radius), min(shape[0], qy+radius+1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.maximum(np.abs(xx-qx), np.abs(yy-qy))
    ring = distance >= int(p["outer_inner_radius"])
    patch = ref[y0:y1, x0:x1]
    metadata = {"method_version": METHOD_VERSION, "patch_bounds": [x0, x1, y0, y1],
                "shape_assumption": "one query-connected opaque PCA rectangle or a single plane",
                "appearance": "piecewise_linear_donors_nearest_extension",
                "exposure": "identity"}
    aux = {"mode": "unavailable", "foreground": np.full(shape, np.nan),
           "background": np.full(shape, np.nan), "mask": np.zeros(shape),
           "polygon": np.empty((0, 2)), "background_depth": np.nan,
           "background_depth_valid": False, "background_geometry": "unavailable",
           "sigma": np.nan, "sigma_valid": False, "valid": False, "metadata": metadata}
    if not (0 <= qx < shape[1] and 0 <= qy < shape[0]) or np.count_nonzero(ring) < 32:
        metadata["reason"] = "insufficient_reference_patch"
        return aux
    centre = (np.abs(xx-qx)+np.abs(yy-qy)) <= 1
    foreground_endpoint = float(np.median(patch[centre]))
    background_endpoint = float(np.median(patch[ring]))
    contrast = foreground_endpoint-background_endpoint
    threshold = max(float(p["contrast_floor"]), float(p["contrast_mad_multiplier"])*1.4826*_mad(patch[ring]))
    metadata.update({"foreground_endpoint": foreground_endpoint, "background_endpoint": background_endpoint,
                     "contrast": contrast, "contrast_threshold": threshold})
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    if abs(contrast) <= threshold:
        # A single frozen sampled texture plane; constant inputs stay constant.
        foreground_donors = np.ones(patch.shape, dtype=bool)
        background_donors = None
        aux.update({"mode": "single", "mask": np.ones(shape), "background_depth": np.inf,
                    "background_geometry": "single_plane", "valid": True})
    else:
        alpha = np.clip((patch-background_endpoint)/contrast, 0., 1.)
        labels, _ = ndimage.label(alpha >= .5)
        label = int(labels[qy-y0, qx-x0])
        component = labels == label if label else np.zeros(patch.shape, bool)
        if np.count_nonzero(component) < 4 or np.any(component[[0, -1], :]) or np.any(component[:, [0, -1]]):
            metadata["reason"] = "query_component_missing_or_exits_patch"
            return aux
        try:
            polygon = _rectangle_from_component(alpha, component, x0, y0)
        except ValueError as exc:
            metadata["reason"] = str(exc)
            return aux
        # One spatial dilation admits mixed initialization pixels; it is not
        # iterated per depth and is never used to discard scoring sensor pixels.
        near_component = ndimage.binary_dilation(component, iterations=1)
        corner_offsets = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
        polygon_aux = {"mode": "two", "polygon": polygon}
        corner_xy = xy[:, None, :] + corner_offsets[None]
        foreground_donors = np.all(_inside(polygon_aux, corner_xy), axis=-1).reshape(patch.shape)
        background_donors = ~near_component
        if np.count_nonzero(foreground_donors) < 3 or np.count_nonzero(background_donors) < 8:
            metadata["reason"] = "insufficient_appearance_donors"
            return aux
        background_depth, background_meta = _estimate_background(ref, train, np.column_stack((xx[ring], yy[ring])), ref_cam, train_cam, p)
        metadata["background_estimate"] = background_meta
        if not background_meta["usable"]:
            metadata["reason"] = background_meta["reason"]
            return aux
        aux.update({"mode": "two", "polygon": polygon, "background_depth": background_depth,
                    "background_depth_valid": background_meta["valid"],
                    "background_geometry": background_meta["geometry"], "valid": True})
        fy, fx = np.indices(shape)
        aux["mask"] = _inside(aux, np.stack((fx, fy), axis=-1)).astype(float)
    fg_xy = xy[foreground_donors.ravel()]
    fg_values = patch[foreground_donors]
    aux["foreground"] = _texture_field(fg_xy, fg_values, shape)
    fg_residual, fg_sigma_meta = _held_block_residuals(fg_xy, fg_values, shape, p)
    metadata["foreground_donors"] = len(fg_xy)
    metadata["sigma_foreground"] = fg_sigma_meta
    residuals = [fg_residual]
    scale_valid = fg_sigma_meta["valid"]
    if background_donors is not None:
        bg_xy, bg_values = xy[background_donors.ravel()], patch[background_donors]
        aux["background"] = _texture_field(bg_xy, bg_values, shape)
        bg_residual, bg_sigma_meta = _held_block_residuals(bg_xy, bg_values, shape, p)
        residuals.append(bg_residual)
        metadata["background_donors"] = len(bg_xy)
        metadata["sigma_background"] = bg_sigma_meta
        scale_valid = scale_valid and bg_sigma_meta["valid"]
    else:
        aux["background"] = aux["foreground"].copy()
    residuals = np.concatenate(residuals)
    aux["sigma_valid"] = bool(scale_valid)
    aux["sigma"] = max(1., 1.4826*_mad(residuals)) if scale_valid else np.nan
    metadata["sigma_residual_count"] = len(residuals)
    metadata["reason"] = "ok" if scale_valid else "raw_auxiliary_available_but_training_scale_insufficient"
    return aux
