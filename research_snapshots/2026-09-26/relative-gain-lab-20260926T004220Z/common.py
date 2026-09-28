"""Local I/O and measurement primitives; no policy selection or hidden data reads."""
from pathlib import Path
import hashlib
import json
import os
import socket
import sys

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z')
CLOSEOUT = Path('/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z')
DATA = Path('/srv/slam-research/grf/map-denoise/datasets')
SEED = 20260926
SCENES = (55, 65, 69)
CONDS = ('native', 'minus1', 'plus1', 'minus3', 'plus3')


def check_host():
    if socket.gethostname() != 'liekkas':
        raise RuntimeError('target host is liekkas')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def save_npz(path, **arrays):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)


def read_points(path):
    import open3d as o3d
    p = np.asarray(o3d.io.read_point_cloud(str(path)).points).copy()
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('invalid point file: ' + str(path))
    return p


def write_points(path, p):
    import open3d as o3d
    if Path(path).exists():
        raise FileExistsError(path)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(p, dtype=float))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError('PLY export failed')


def in_box(p, lo, hi):
    return np.all((p >= lo) & (p <= hi), axis=1)


def observed(p, obs):
    origin = np.asarray(obs['BB']).reshape(-1, 3)[0]
    res = float(np.asarray(obs['Res']).ravel()[0])
    ids = np.around((p - origin) / res).astype(np.int64)
    valid = np.all((ids >= 0) & (ids < np.asarray(obs['ObsMask'].shape)), axis=1)
    keep = np.zeros(len(p), dtype=bool)
    keep[valid] = obs['ObsMask'][tuple(ids[valid].T)].astype(bool)
    return keep


def voxel(p):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(p)
    return np.asarray(cloud.voxel_down_sample(0.8).points).copy()


def seal(folder, source=None):
    save_json(folder / 'SEALED.json', {
        'files': {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
        'source': source or {},
    })


def verify_seal(folder):
    manifest = json.loads((folder / 'SEALED.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(folder / name) != digest:
            raise RuntimeError('seal mismatch: ' + str(folder / name))
    return manifest
