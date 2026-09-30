"""Frozen Q-only CPU baseline. Run only after parent seals RUN_LOCK.json."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import json
import socket
import time

import cv2
import numpy as np

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/colmap-transfer-20260930T180000Z')
OLD = Path('/srv/slam-research/grf/map-denoise/runs/upstream-photo-holdout-20260930T113213Z')
sys.path.insert(0, str(OLD))
from dense_rebuild import plane_scores
from rebuild import half_gray, mutual_matches, triangulate, projection_center, sha, write_json
from mvs_transfer import verify_lock


def prediction(scores, variance, depths, pixels, P):
    finite = np.isfinite(scores)
    safe = np.where(finite, scores, -2)
    best = safe.argmax(0)
    score = safe[best, np.arange(len(pixels))]
    valid = finite.any(0) & (score >= .6) & (variance > 1e-5)
    z = depths[best].copy()
    for i in np.flatnonzero((best > 0) & (best < len(depths) - 1)):
        a, b, c = safe[best[i] - 1:best[i] + 2, i]
        den = a - 2 * b + c
        if den < -1e-8 and min(a, b, c) > -1:
            z[i] += np.clip(.5 * (a - c) / den, -.5, .5)
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(P[:, :3]).T
    xyz = projection_center(P) + z[:, None] * rays
    xyz[~valid] = np.nan
    z[~valid] = np.nan
    return xyz, valid, z, score


def run(sid):
    if socket.gethostname() != 'liekkas' or Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Wrong target identity')
    lock_path = ROOT / 'RUN_LOCK.json'
    verify_lock()
    plan = json.loads((ROOT / 'PLAN.json').read_text())
    for filename, expected in plan['source_files'].items():
        if sha(filename) != expected:
            raise RuntimeError('Input/camera-plan source changed: ' + filename)
    spec = plan['scenes'][str(sid)]
    output = ROOT / 'cpu'
    output.mkdir(exist_ok=True)
    targets = [output / (r['id'] + '.npz') for r in spec['rois']]
    targets += [ROOT / f'RANGES_{sid}.json', output / f'SUMMARY_{sid}.json']
    if any(p.exists() for p in targets):
        raise FileExistsError('Will not overwrite baseline outputs')
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260930)
    start = time.monotonic()
    cameras = spec['cameras']
    names = [spec['reference']] + spec['Q']
    images = [half_gray(Path(cameras[n]['image_path']), cameras[n]) for n in names]
    P = np.array(cameras[spec['reference']]['P'])
    Ps = [np.array(cameras[n]['P']) for n in spec['Q']]
    sift = cv2.SIFT_create(nfeatures=12000)
    kp, des = sift.detectAndCompute(images[0], None)
    zs, match_records = [], []
    for name, im, S in zip(spec['Q'], images[1:], Ps):
        skp, sdes = sift.detectAndCompute(im, None)
        matches = mutual_matches(des, sdes)
        good_count = 0
        if matches:
            uv = np.array([kp[m.queryIdx].pt for m in matches])
            xy = np.array([skp[m.trainIdx].pt for m in matches])
            pts, good, _, _ = triangulate(P, S, uv, xy)
            zs.extend((pts[good] @ P[2, :3] + P[2, 3]).tolist())
            good_count = int(good.sum())
        match_records.append(dict(view=name, mutual_matches=len(matches), qualified=good_count))
    if len(zs) >= 10:
        low, high = np.quantile(zs, [.05, .95]) + [-20., 20.]
        rule = 'Q-only SIFT q05/q95 +-20mm'
    else:
        axes = np.array([cameras[n]['P'][2][:3] for n in names])
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        proj = np.eye(3)[None] - axes[:, :, None] * axes[:, None, :]
        centers = np.array([cameras[n]['center'] for n in names])
        focus = np.linalg.solve(proj.sum(0), np.einsum('nij,nj->i', proj, centers))
        z = P[2, :3] @ focus + P[2, 3]
        low, high = z - 150, z + 150
        rule = 'Q-only rig focus +-150mm'
    if not (np.isfinite([low, high]).all() and 0 < low < high and high - low < 2000):
        raise ValueError('Frozen range rule produced invalid interval; no new-scene retuning')
    depths = np.arange(np.floor(low), np.ceil(high) + .1, 1.)
    write_json(ROOT / f'RANGES_{sid}.json', dict(scene=sid,
        depth_range_mm=[float(np.floor(low)), float(np.ceil(high))],
        raw_quantile_range_mm=[float(low), float(high)], depth_step_mm=1.,
        rule=rule, sift_count=len(zs), matches=match_records,
        plan_sha256=sha(ROOT / 'PLAN.json'), run_lock_sha256=sha(lock_path), gt_accessed=False))
    print('DEPTH_RANGE', sid, float(low), float(high), len(zs), flush=True)
    records = []
    for roi in spec['rois']:
        tick = time.monotonic()
        pixels = np.asarray(roi['pixel_xy'], np.float64)
        scores, variance = plane_scores(images[0], images[1:], P, Ps, pixels, depths)
        xyz, valid, z, score = prediction(scores, variance, depths, pixels, P)
        dest = output / (roi['id'] + '.npz')
        with dest.open('xb') as f:
            np.savez_compressed(f, pixel_xy=pixels.astype(np.int64), xyz_mm=xyz,
                                valid=valid, depth_mm=z, score=score, variance=variance)
        record = dict(roi=roi['id'], requested=len(pixels), valid=int(valid.sum()),
                      seconds=time.monotonic() - tick, output_sha256=sha(dest))
        records.append(record)
        print('CPU_BASELINE', sid, roi['id'], record['valid'], len(pixels), flush=True)
    write_json(output / f'SUMMARY_{sid}.json', dict(scene=sid, records=records,
        seconds=time.monotonic() - start, frozen_algorithm='old corrected_rebuild CPU5x5 top2of4 ZNCC',
        package_versions=dict(numpy=np.__version__, opencv=cv2.__version__),
        source_files={str(p): sha(p) for p in (Path(__file__), OLD / 'dense_rebuild.py',
                                               OLD / 'rebuild.py', ROOT / 'PLAN.json')},
        run_lock_sha256=sha(lock_path), gt_accessed=False, points3D_accessed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, choices=(118, 122))
    parser.add_argument('--both', action='store_true')
    args = parser.parse_args()
    if args.both:
        for sid in (118, 122):
            run(sid)
    elif args.scene:
        run(args.scene)
    else:
        parser.error('Use --scene or --both')
