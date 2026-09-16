"""Shared multi-view plane-sweep evidence. No laser / mask / GT fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter

from .cameras import Scene, unproject_colmap_depth
from .io_util import sha256_array
from .observations import Crop, Roi, crop_to_view, crop_view, depth_range_from_mesh, select_views


@dataclass
class EvidenceBundle:
    scene_id: int
    roi_id: str
    view_names: list[str]
    ref_name: str
    depths_norm: np.ndarray
    zncc: np.ndarray
    n_valid_src: np.ndarray
    crop_meta: list[dict]
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    config: dict
    hashes: dict


def _masked_zncc(ref: np.ndarray, src: np.ndarray, valid: np.ndarray, win: int) -> np.ndarray:
    window = float(win * win)
    mask = valid.astype(np.float64)
    count = uniform_filter(mask, size=win, mode="constant") * window
    def _sum(values: np.ndarray) -> np.ndarray:
        return uniform_filter(values, size=win, mode="constant") * window
    sum_r = _sum(np.where(valid, ref, 0.0))
    sum_s = _sum(np.where(valid, src, 0.0))
    sum_rr = _sum(np.where(valid, ref * ref, 0.0))
    sum_ss = _sum(np.where(valid, src * src, 0.0))
    sum_rs = _sum(np.where(valid, ref * src, 0.0))
    safe = np.maximum(count, 1.0)
    mean_r = sum_r / safe
    mean_s = sum_s / safe
    cov = sum_rs / safe - mean_r * mean_s
    var_r = np.maximum(sum_rr / safe - mean_r * mean_r, 0.0)
    var_s = np.maximum(sum_ss / safe - mean_s * mean_s, 0.0)
    denom = np.sqrt(var_r * var_s) + 1e-8
    score = cov / denom
    score[count < 0.65 * window] = np.nan
    return score


def _warp_source(src_gray: np.ndarray, uv: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(uv).all(axis=1)
    safe = np.where(ok[:, None], uv, 0.0)
    sample = map_coordinates(src_gray, [safe[:, 1], safe[:, 0]], order=1, mode="constant", cval=np.nan)
    sample = np.where(ok, sample, np.nan)
    warped = sample.reshape(shape)
    valid = np.isfinite(warped)
    return np.where(valid, warped, 0.0), valid


def build_evidence(scene: Scene, roi: Roi, config: dict) -> EvidenceBundle:
    names = select_views(scene, roi, int(config["n_views"]))
    ref_name = names[0]
    ref_view = scene.view_by_name(ref_name)
    crops: list[Crop] = [crop_view(scene, scene.view_by_name(name), roi, int(config["crop_px"])) for name in names]
    cropped_views = [crop_to_view(scene.view_by_name(name), crop, scene.scale) for name, crop in zip(names, crops)]
    ref_crop = crops[0]
    ref_cview = cropped_views[0]
    d0, d1 = depth_range_from_mesh(scene, ref_view, roi, float(config["depth_margin_mm"]))
    d0 = max(d0, 1e-4)
    if d1 <= d0:
        raise ValueError(f"empty depth range for {roi.roi_id}")
    depths = np.linspace(d0, d1, int(config["n_depths"]))
    height, width = ref_crop.gray.shape
    yy, xx = np.mgrid[0:height, 0:width]
    pixels = np.column_stack([xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)])
    volume = np.full((len(depths), height, width), np.nan, dtype=np.float32)
    counts = np.zeros((len(depths), height, width), dtype=np.uint8)
    win = int(config["zncc_window"])
    src_items = list(zip(crops[1:], cropped_views[1:]))
    for di, depth in enumerate(depths):
        if di % 20 == 0:
            print(f"  plane {di}/{len(depths)}", flush=True)
        xyz = unproject_colmap_depth(ref_cview, pixels, np.full(len(pixels), depth), scene.scale)
        acc = np.zeros((height, width), dtype=np.float64)
        n_ok = np.zeros((height, width), dtype=np.float64)
        for crop, cview in src_items:
            uv, z = _project(xyz, cview.P_phys)
            in_front = z > 1e-6
            uv[~in_front] = np.nan
            warped, valid = _warp_source(crop.gray, uv, (height, width))
            valid &= in_front.reshape(height, width)
            score = _masked_zncc(ref_crop.gray, warped, valid, win)
            good = np.isfinite(score)
            acc = np.where(good, acc + score, acc)
            n_ok += good
        with np.errstate(invalid="ignore"):
            volume[di] = np.where(n_ok >= 1, acc / np.maximum(n_ok, 1.0), np.nan).astype(np.float32)
        counts[di] = n_ok.astype(np.uint8)
    crop_meta = [
        {
            "view": crop.view_name,
            "x0": crop.x0,
            "y0": crop.y0,
            "side": crop.side,
            "scale": crop.scale,
            "K": crop.K.tolist(),
        }
        for crop in crops
    ]
    config = dict(config)
    config["scale_factor"] = float(scene.scale_factor)
    hashes = {
        "zncc": sha256_array(volume),
        "depths": sha256_array(depths),
        "counts": sha256_array(counts),
        "aabb": sha256_array(np.stack([roi.aabb_min, roi.aabb_max])),
    }
    return EvidenceBundle(
        scene_id=scene.scene_id,
        roi_id=roi.roi_id,
        view_names=names,
        ref_name=ref_name,
        depths_norm=depths,
        zncc=volume,
        n_valid_src=counts,
        crop_meta=crop_meta,
        aabb_min=roi.aabb_min.copy(),
        aabb_max=roi.aabb_max.copy(),
        config={k: config[k] for k in config},
        hashes=hashes,
    )


def _project(points: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    clip = homogeneous @ matrix.T
    depth = clip[:, 2]
    uv = np.divide(
        clip[:, :2],
        depth[:, None],
        out=np.full((len(points), 2), np.nan),
        where=np.abs(depth[:, None]) > 1e-12,
    )
    return uv, depth


def save_bundle(path, bundle: EvidenceBundle) -> None:
    np.savez_compressed(
        path,
        scene_id=bundle.scene_id,
        roi_id=bundle.roi_id,
        view_names=np.asarray(bundle.view_names),
        ref_name=bundle.ref_name,
        depths_norm=bundle.depths_norm,
        zncc=bundle.zncc,
        n_valid_src=bundle.n_valid_src,
        aabb_min=bundle.aabb_min,
        aabb_max=bundle.aabb_max,
        crop_meta_json=np.asarray([str(bundle.crop_meta)], dtype=object),
        config_keys=np.asarray(list(bundle.config.keys())),
        hashes_json=np.asarray([str(bundle.hashes)], dtype=object),
    )


def bundle_public_dict(bundle: EvidenceBundle) -> dict:
    return {
        "scene_id": bundle.scene_id,
        "roi_id": bundle.roi_id,
        "view_names": bundle.view_names,
        "ref_name": bundle.ref_name,
        "n_depths": int(len(bundle.depths_norm)),
        "crop_px": int(bundle.zncc.shape[1]),
        "depth_norm_min": float(bundle.depths_norm.min()),
        "depth_norm_max": float(bundle.depths_norm.max()),
        "hashes": bundle.hashes,
        "crop_meta": bundle.crop_meta,
        "aabb_min_mm": bundle.aabb_min.tolist(),
        "aabb_max_mm": bundle.aabb_max.tolist(),
        "contains_laser_or_gt": False,
        "arm_shared": True,
    }


def wta_depth(bundle: EvidenceBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = bundle.zncc.astype(np.float64)
    counts = bundle.n_valid_src
    min_zncc = float(bundle.config["min_zncc"])
    min_views = int(bundle.config["min_views_for_valid"])
    best_idx = np.nanargmax(np.where(np.isfinite(scores), scores, -1e9), axis=0)
    grid_i, grid_j = np.indices(scores.shape[1:])
    best = scores[best_idx, grid_i, grid_j]
    nsrc = counts[best_idx, grid_i, grid_j]
    valid = np.isfinite(best) & (best >= min_zncc) & (nsrc >= min_views)
    depth = np.where(valid, bundle.depths_norm[best_idx], np.nan)
    return depth, best, valid


def peak_depths(bundle: EvidenceBundle) -> list[dict]:
    """Keep WTA and an optional second 1-D peak per pixel. Not a physical-layer label."""
    scores = bundle.zncc.astype(np.float64)
    min_zncc = float(bundle.config["min_zncc"])
    min_views = int(bundle.config["min_views_for_valid"])
    sep = float(bundle.config["peak_sep_mm"]) / float(bundle.config.get("scale_factor", 1.0))
    # scale_factor is stored later; if missing, treat depths as already separated in norm units via config.
    scale_factor = float(bundle.config.get("scale_factor", 0.0))
    sep_norm = float(bundle.config["peak_sep_mm"]) / scale_factor if scale_factor else 0.02
    ratio = float(bundle.config["second_peak_ratio"])
    n_d, height, width = scores.shape
    peaks = []
    for y in range(height):
        for x in range(width):
            col = scores[:, y, x]
            cnt = bundle.n_valid_src[:, y, x]
            if not np.isfinite(col).any():
                continue
            local = []
            for i in range(1, n_d - 1):
                if not np.isfinite(col[i]) or cnt[i] < min_views or col[i] < min_zncc:
                    continue
                if col[i] >= col[i - 1] and col[i] >= col[i + 1]:
                    local.append((float(col[i]), i))
            if not local:
                i = int(np.nanargmax(col))
                if np.isfinite(col[i]) and col[i] >= min_zncc and cnt[i] >= min_views:
                    local = [(float(col[i]), i)]
            local.sort(reverse=True)
            kept = []
            for score, i in local:
                if not kept:
                    kept.append((score, i))
                    continue
                if score < ratio * kept[0][0]:
                    break
                if all(abs(bundle.depths_norm[i] - bundle.depths_norm[j]) >= sep_norm for _, j in kept):
                    kept.append((score, i))
                if len(kept) >= 2:
                    break
            for rank, (score, i) in enumerate(kept):
                peaks.append({"u": x, "v": y, "depth_norm": float(bundle.depths_norm[i]), "zncc": score, "rank": rank})
    return peaks
