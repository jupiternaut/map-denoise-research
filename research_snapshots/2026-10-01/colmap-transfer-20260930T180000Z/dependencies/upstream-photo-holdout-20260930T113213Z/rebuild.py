"""Construction-photo-only calibrated SIFT triangulation. No old scene geometry.

Run with system Python (OpenCV available there); downstream historical arrays
use the existing scientific environment. Existing world_mat is an explicitly
conditioned input, not certified independently calibrated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = Path("/srv/slam-research/grf/map-denoise/datasets")
ROI_FILE = Path("/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z/vendor/configs/roi_image_boxes.json")
SCENES = (24, 37)
CONDITIONS = ("native", "minus3", "plus3")
MAX_CONTEXT = 384
MAX_QUERIES = 128
cv2.setNumThreads(1)
cv2.setRNGSeed(20260930)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def check_host():
    if socket.gethostname() != "liekkas":
        raise RuntimeError("Wrong host")
    if ROOT != Path("/srv/slam-research/grf/map-denoise/runs/upstream-photo-holdout-20260930T113213Z"):
        raise RuntimeError("Wrong experiment root")


def source_paths(sid):
    images = DATA / "loss-alignment-v23" / f"scan{sid}" / "image"
    calibration = DATA / ("real-closure-v21/cameras_geosvr_linked.npz" if sid == 24
                          else "reconstruction-v22-scan37/cameras.npz")
    return images, calibration


def projection_center(P):
    return -np.linalg.solve(P[:, :3], P[:, 3])


def half_projection(P, width, height):
    nw, nh = width // 2, height // 2
    sx, sy = nw / width, nh / height
    A = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])
    return A @ P, nw, nh


def plan():
    check_host()
    configuration = json.loads(ROI_FILE.read_text())
    reference = configuration["ref_view"]
    scenes, cases, sources = {}, [], {str(ROI_FILE): sha(ROI_FILE)}
    for sid in SCENES:
        images, calibration = source_paths(sid)
        sources[str(calibration)] = sha(calibration)
        # Only the reference photo is opened. Held-out pixels are not consulted.
        ref_path = images / reference
        with Image.open(ref_path) as im:
            width, height = im.size
        sources[str(ref_path)] = sha(ref_path)
        with np.load(calibration, allow_pickle=False) as z:
            matrices = {name: z[f"world_mat_{int(Path(name).stem)}"][:3].astype(float)
                        for name in sorted(p.name for p in images.glob("*.png"))}
        axes = {name: P[2, :3] / np.linalg.norm(P[2, :3]) for name, P in matrices.items()}
        ranked = sorted((float(np.degrees(np.arccos(np.clip(axis @ axes[reference], -1, 1)))), name)
                        for name, axis in axes.items() if name != reference)
        eligible = [(angle, name) for angle, name in ranked if 5 <= angle <= 45]
        if len(eligible) < 12:
            raise RuntimeError(f"scan{sid}: only {len(eligible)} eligible source cameras")
        triplets = [eligible[i:i + 3] for i in range(0, 12, 3)]
        groups = {key: [triple[index][1] for triple in triplets]
                  for index, key in enumerate(("Q", "C", "H"))}
        cameras = {}
        for name in [reference] + groups["Q"] + groups["C"] + groups["H"]:
            P, nw, nh = half_projection(matrices[name], width, height)
            cameras[name] = dict(P=P.tolist(), center=projection_center(P).tolist(),
                                 image_path=str(images / name), width=nw, height=nh)
        rois = [dict(id=r["roi_id"], box_xyxy_original=r["box_xyxy"], kind=r["kind"])
                for r in configuration["scenes"][str(sid)][:2]]
        scenes[str(sid)] = dict(images_dir=str(images), camera_file=str(calibration),
                                reference=reference, cameras=cameras, rois=rois, **groups,
                                angle_triplets=[[dict(name=n, angle_deg=a) for a, n in t] for t in triplets])
        for roi in rois:
            for condition in CONDITIONS:
                case_id = roi["id"] + "__" + condition
                cases.append(dict(case_id=case_id, scene=sid, roi=roi["id"], condition=condition,
                                  output_dir=str(ROOT / "inference" / f"scan{sid}" / case_id)))
    record = dict(host=socket.gethostname(), root=str(ROOT), scenes=scenes, cases=cases,
                  calibration_condition="Conditioned on provided world_mat; pose upstream provenance unverified",
                  coordinate_units="mm", use_scale_mat=False, source_files=sources, view_selection="camera-only consecutive angle triplets",
                  max_context=MAX_CONTEXT, max_queries=MAX_QUERIES, gt_accessed=False)
    write_json(ROOT / "PLAN.json", record)
    print("PLAN_LOCKED", {s: {k: v[k] for k in ("reference", "Q", "C", "H")} for s, v in scenes.items()}, flush=True)


def project(P, points):
    h = np.column_stack((points, np.ones(len(points)))) @ P.T
    return h[:, :2] / h[:, 2:3], h[:, 2]


def triangulate(P, S, uv, xy):
    h = cv2.triangulatePoints(P, S, uv.T, xy.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        points = h[:, :3] / h[:, 3:4]
        up, z0 = project(P, points)
        xp, z1 = project(S, points)
        error = np.maximum(np.linalg.norm(up - uv, axis=1), np.linalg.norm(xp - xy, axis=1))
        ray0 = points - projection_center(P)
        ray1 = points - projection_center(S)
        cosine = np.sum(ray0 * ray1, axis=1) / (np.linalg.norm(ray0, axis=1) * np.linalg.norm(ray1, axis=1))
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    good = np.isfinite(points).all(1) & np.isfinite(error) & (z0 > 0) & (z1 > 0) & (error <= 1.) & (angle >= 2.)
    return points, good, error, angle


def half_gray(path, camera):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize((camera["width"], camera["height"]), Image.Resampling.BILINEAR))
    return np.clip(rgb @ np.array([.299, .587, .114]), 0, 255).astype(np.uint8)


def mutual_matches(des0, des1):
    if des0 is None or des1 is None or len(des0) < 2 or len(des1) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    forward = {m.queryIdx: m for pair in bf.knnMatch(des0, des1, k=2) if len(pair) == 2
               for m, n in [pair] if m.distance < .75 * n.distance}
    reverse = {m.queryIdx: m.trainIdx for pair in bf.knnMatch(des1, des0, k=2) if len(pair) == 2
               for m, n in [pair] if m.distance < .75 * n.distance}
    return [m for i, m in forward.items() if reverse.get(m.trainIdx) == i]


def rebuild(sid):
    check_host()
    started = time.monotonic()
    spec = json.loads((ROOT / "PLAN.json").read_text())["scenes"][str(sid)]
    dest = ROOT / "construction" / f"scan{sid}"
    dest.mkdir(parents=True, exist_ok=False)
    allowed = {spec["reference"], *spec["Q"]}
    events, sources = [], {}
    cameras = spec["cameras"]
    def load(name):
        if name not in allowed:
            raise PermissionError("Only Q+reference photo pixels allowed in rebuilding")
        path = Path(cameras[name]["image_path"])
        digest = sha(path)
        sources[str(path)] = digest
        events.append(dict(sequence=len(events), phase="rebuild", kind="photo", path=str(path), view_id=name,
                           group="reference" if name == spec["reference"] else "Q", sha256=digest))
        return half_gray(path, cameras[name])
    reference = spec["reference"]
    image = load(reference)
    sift = cv2.SIFT_create(nfeatures=12000)
    source_features = {}
    for name in spec["Q"]:
        source_features[name] = sift.detectAndCompute(load(name), None)
    summaries = []
    for roi in spec["rois"]:
        tick = time.monotonic()
        x0, y0, x1, y1 = roi["box_xyxy_original"]
        mask = np.zeros_like(image)
        mask[max(0, int(y0 / 2)):min(image.shape[0], int(np.ceil(y1 / 2))),
             max(0, int(x0 / 2)):min(image.shape[1], int(np.ceil(x1 / 2)))] = 255
        keys, des = sift.detectAndCompute(image, mask)
        candidates, stats = {}, []
        P = np.asarray(cameras[reference]["P"])
        for name in spec["Q"]:
            skeys, sdes = source_features[name]
            matches = mutual_matches(des, sdes)
            if matches:
                uv = np.array([keys[m.queryIdx].pt for m in matches], dtype=float)
                xy = np.array([skeys[m.trainIdx].pt for m in matches], dtype=float)
                points, good, error, angles = triangulate(P, np.asarray(cameras[name]["P"]), uv, xy)
                for i in np.flatnonzero(good):
                    m = matches[i]
                    entry = (float(error[i]), name, points[i], uv[i], float(angles[i]))
                    old = candidates.get(m.queryIdx)
                    if old is None or entry[:2] < old[:2]:
                        candidates[m.queryIdx] = entry
            else:
                good = np.array([], bool)
            stats.append(dict(view=name, mutual_matches=len(matches), geometric_matches=int(good.sum())))
        # SIFT may report multiple orientations at one pixel. Keep one geometric
        # solution per quantized reference location to avoid duplicate samples.
        locations = {}
        for entry in candidates.values():
            pixel = tuple(np.round(entry[3], 2))
            if pixel not in locations or entry[:2] < locations[pixel][:2]:
                locations[pixel] = entry
        ordered = sorted(locations.values(), key=lambda e: hashlib.sha256(np.round(e[3], 2).tobytes()).digest())
        selected = ordered[:MAX_CONTEXT]
        status = "READY" if len(selected) >= 64 else "INSUFFICIENT_CONSTRUCTION_MATCHES"
        folder = dest / roi["id"]
        folder.mkdir()
        if selected:
            points = np.array([e[2] for e in selected])
            pixels = np.array([e[3] for e in selected])
            ids = np.linspace(0, len(points) - 1, min(MAX_QUERIES, len(points)), dtype=np.int64)
            with (folder / "input.npz").open("xb") as stream:
                np.savez_compressed(stream, points_mm=points, reference_pixel_xy=pixels, query_ids=ids)
        rec = dict(scene=sid, roi=roi["id"], status=status, keypoints=len(keys), unique_geometric_points=len(ordered),
                   context_rows=len(selected), query_rows=min(MAX_QUERIES, len(selected)), matches=stats,
                   max_selected_reprojection_px=max((e[0] for e in selected), default=None),
                   sources=sorted(set(e[1] for e in selected)), construction_views=[reference] + spec["Q"],
                   gt_accessed=False, mesh_accessed=False, sparse_accessed=False, seconds=time.monotonic() - tick)
        write_json(folder / "REBUILD.json", rec)
        summaries.append(rec)
        print("REBUILT", sid, roi["id"], status, len(ordered), "->", len(selected), flush=True)
    write_json(dest / "READ_EVENTS.json", events)
    write_json(dest / "SUMMARY.json", dict(records=summaries, source_files=sources, seconds=time.monotonic() - started,
                                             cv2_version=cv2.__version__, numpy_version=np.__version__, gpu=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--scene", type=int, choices=SCENES)
    args = ap.parse_args()
    if args.plan:
        plan()
    elif args.scene:
        rebuild(args.scene)
    else:
        ap.error("choose --plan or --scene")


if __name__ == "__main__":
    main()
