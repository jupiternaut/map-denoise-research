"""Q-only plane sweep initializer. No reference geometry or H pixels."""
import argparse
import time
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from rebuild import (ROOT, MAX_CONTEXT, MAX_QUERIES, check_host, sha, write_json,
                     half_gray, mutual_matches, triangulate, projection_center)


def sample(image, xy):
    shape = xy.shape[:-1]
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    valid = np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 1] >= 0)
    valid &= (xy[:, 0] <= image.shape[1] - 1) & (xy[:, 1] <= image.shape[0] - 1)
    safe = np.where(valid[:, None], xy, 0)
    # OpenCV remap has a 32767 dimension limit; chunk the flattened map.
    values = np.empty(len(xy), np.float32)
    for start in range(0, len(xy), 16000):
        stop = min(len(xy), start + 16000)
        values[start:stop] = cv2.remap(image.astype(np.float32), safe[start:stop, 0][None],
                                     safe[start:stop, 1][None], cv2.INTER_LINEAR).ravel()
    values[~valid] = np.nan
    return values.reshape(shape)


def zncc(ref, values):
    valid = np.isfinite(ref) & np.isfinite(values)
    count = valid.sum(-1)
    a, b = np.where(valid, ref, 0), np.where(valid, values, 0)
    ma, mb = a.sum(-1) / np.maximum(count, 1), b.sum(-1) / np.maximum(count, 1)
    a, b = np.where(valid, a - ma[..., None], 0), np.where(valid, b - mb[..., None], 0)
    denom = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    score = (a * b).sum(-1) / np.maximum(denom, 1e-12)
    return np.where((count >= 20) & (denom > 1e-10), score, np.nan)


def plane_scores(reference_image, source_images, reference_P, source_Ps, pixels, depths):
    """Scores[D,N] on a fixed 5x5 reference-plane patch."""
    xx, yy = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    patch_uv = pixels[:, None] + np.column_stack((xx.ravel(), yy.ravel()))[None]
    ref_patch = sample(reference_image, patch_uv) / 255.
    rays = np.column_stack((patch_uv.reshape(-1, 2), np.ones(len(pixels) * 25))) @ np.linalg.inv(reference_P[:, :3]).T
    center = projection_center(reference_P)
    result = np.full((len(depths), len(pixels)), np.nan, np.float32)
    # All camera matrices are canonically given world_mat, with third-row norm1.
    for start in range(0, len(depths), 16):
        dep = depths[start:start + 16]
        points = center + dep[:, None, None] * rays[None]
        view_scores = []
        for image, P in zip(source_images, source_Ps):
            h = points @ P[:, :3].T + P[:, 3]
            uv = h[..., :2] / h[..., 2:3]
            vals = sample(image, uv).reshape(len(dep), len(pixels), 25) / 255.
            vals = np.where((h[..., 2] > 0).reshape(len(dep), len(pixels), 25), vals, np.nan)
            view_scores.append(zncc(ref_patch[None], vals))
        scores = np.stack(view_scores, axis=2)
        valid = np.isfinite(scores).sum(2) >= 2
        top = np.sort(np.where(np.isfinite(scores), scores, -2), axis=2)[:, :, -2:].mean(2)
        result[start:start + len(dep)] = np.where(valid, top, np.nan)
    return result, np.nanvar(ref_patch, axis=1)


def run(sid):
    check_host()
    started = time.monotonic()
    spec = json.loads((ROOT / "PLAN.json").read_text())["scenes"][str(sid)]
    cameras = spec["cameras"]
    reference = spec["reference"]
    allowed = {reference, *spec["Q"]}
    dest = ROOT / "construction-dense" / f"scan{sid}"
    dest.mkdir(parents=True, exist_ok=False)
    events, sources = [], {}
    def load(name):
        if name not in allowed:
            raise PermissionError("Q-only rebuilding")
        path = Path(cameras[name]["image_path"])
        digest = sha(path)
        sources[str(path)] = digest
        events.append(dict(sequence=len(events), phase="rebuild", kind="photo", path=str(path),
                           view_id=name, group="reference" if name == reference else "Q", sha256=digest))
        return half_gray(path, cameras[name])
    image = load(reference)
    source_images = [load(name) for name in spec["Q"]]
    P = np.asarray(cameras[reference]["P"], float)
    source_Ps = [np.asarray(cameras[n]["P"], float) for n in spec["Q"]]
    sift = cv2.SIFT_create(nfeatures=12000)
    kp, des = sift.detectAndCompute(image, None)
    optical_depths = []
    for src_image, S in zip(source_images, source_Ps):
        skp, sdes = sift.detectAndCompute(src_image, None)
        matches = mutual_matches(des, sdes)
        if not matches:
            continue
        uv = np.array([kp[m.queryIdx].pt for m in matches], float)
        xy = np.array([skp[m.trainIdx].pt for m in matches], float)
        points, good, _, _ = triangulate(P, S, uv, xy)
        optical_depths.extend((points[good] @ P[2, :3] + P[2, 3]).tolist())
    if len(optical_depths) >= 10:
        low, high = np.quantile(optical_depths, [.05, .95]) + np.array([-20., 20.])
        rule = "Q-only global SIFT optical_depth q05/q95+-20mm"
    else:
        axes = np.array([np.asarray(cameras[n]["P"])[2, :3] for n in [reference] + spec["Q"]])
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        projectors = np.eye(3)[None] - axes[:, :, None] * axes[:, None, :]
        centers = np.array([cameras[n]["center"] for n in [reference] + spec["Q"]])
        focus = np.linalg.solve(projectors.sum(0), np.einsum("nij,nj->i", projectors, centers))
        d = P[2, :3] @ focus + P[2, 3]
        low, high = d - 150, d + 150
        rule = "Q-only rig focus +-150mm"
    if not (np.isfinite([low, high]).all() and 0 < low < high and high - low < 2000):
        raise ValueError("Unusable construction-only depth range")
    depths = np.arange(np.floor(low), np.ceil(high) + .1, 1.)
    print("DEPTH_RANGE", sid, low, high, len(depths), flush=True)
    summaries = []
    for roi in spec["rois"]:
        tick = time.monotonic()
        x0, y0, x1, y1 = roi["box_xyxy_original"]
        xx, yy = np.meshgrid(np.arange(x0 / 2 + 4, x1 / 2 - 4, 8.),
                             np.arange(y0 / 2 + 4, y1 / 2 - 4, 8.))
        pixels = np.column_stack((xx.ravel(), yy.ravel()))
        scores, variance = plane_scores(image, source_images, P, source_Ps, pixels, depths)
        finite = np.isfinite(scores)
        safe = np.where(finite, scores, -2)
        best = safe.argmax(0)
        best_score = safe[best, np.arange(len(pixels))]
        selected = finite.any(0) & (best_score >= .6) & (variance > 1e-5)
        ids = np.flatnonzero(selected)
        z = depths[best].copy()
        # Refine only a well-formed interior maximum; no new data enter.
        interior = (best > 0) & (best < len(depths) - 1)
        for i in np.flatnonzero(interior):
            a, b, c = safe[best[i] - 1:best[i] + 2, i]
            denominator = a - 2 * b + c
            if denominator < -1e-8 and min(a, b, c) > -1:
                z[i] += np.clip(.5 * (a - c) / denominator, -.5, .5)
        order = sorted(ids, key=lambda i: hashlib.sha256(pixels[i].tobytes()).digest())[:MAX_CONTEXT]
        order = np.array(order, np.int64)
        rays = np.column_stack((pixels[order], np.ones(len(order)))) @ np.linalg.inv(P[:, :3]).T
        points = projection_center(P) + z[order, None] * rays
        queries = np.linspace(0, len(points) - 1, min(len(points), MAX_QUERIES), dtype=np.int64)
        folder = dest / roi["id"]
        folder.mkdir()
        with (folder / "input.npz").open("xb") as stream:
            np.savez_compressed(stream, points_mm=points, reference_pixel_xy=pixels[order], query_ids=queries,
                                construction_zncc=best_score[order], optical_depth_mm=z[order])
        rec = dict(scene=sid, roi=roi["id"], status="READY" if len(points) >= 64 else "INSUFFICIENT_CONSTRUCTION_MATCHES",
                   grid_rows=len(pixels), passing_rows=int(selected.sum()), context_rows=len(points), query_rows=len(queries),
                   depth_rule=rule, depth_range_mm=[float(low), float(high)], global_sparse_depth_count=len(optical_depths),
                   max_context=MAX_CONTEXT, reconstruction="CPU Q-only calibrated plane sweep", min_zncc=.6,
                   construction_views=[reference] + spec["Q"], gt_accessed=False, mesh_accessed=False,
                   sparse_accessed=False, seconds=time.monotonic() - tick)
        write_json(folder / "REBUILD.json", rec)
        summaries.append(rec)
        print("DENSE_REBUILT", sid, roi["id"], rec["status"], rec["passing_rows"], flush=True)
    write_json(dest / "READ_EVENTS.json", events)
    write_json(dest / "SUMMARY.json", dict(records=summaries, source_files=sources, seconds=time.monotonic() - started,
                                             gpu=False, calibration_condition="given world_mat"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=int, choices=(24, 37))
    run(ap.parse_args().scene)
