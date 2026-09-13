"""Frozen paired correlation intervention; only input NPZ files are legal inputs.

All lengths are mm. Displacement is along the fixed map-normal axis e_z.
The curved probe uses shared h(x,y)=0.0015*(x**2+0.5*y**2), so its
displacement is axial, not a spatially varying local-surface-normal displacement.
It is a controlled model-mismatch probe, not a scanning simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import socket

import numpy as np


ROOT = Path(__file__).resolve().parent
OLD_COMMON = Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-t2-boundary-v1/common.py')
OLD_SHA256 = '8c12d4a5cfaebfc72bea4e45c88ad04401f2a1a353024f8c70e2f90b138e57f4'
SEEDS = (911031, 911047, 911059)
AMPLITUDES = (0., 2., 4.)
FRAMES = 8
POINTS_PER_FRAME = 96
SIGMA_MM = 1.
SCENES = (
    {'family': 'single', 'group': 'planar', 'gap_mm': 0., 'imbalance': 0.},
    {'family': 'dual_balanced', 'group': 'planar', 'gap_mm': 6., 'imbalance': 0.},
    {'family': 'dual_imbalanced', 'group': 'planar', 'gap_mm': 6., 'imbalance': .9},
    {'family': 'raycast_gap4', 'group': 'raycast', 'gap_mm': 4., 'imbalance': .85},
    {'family': 'raycast_gap8', 'group': 'raycast', 'gap_mm': 8., 'imbalance': .85},
    {'family': 'curved_dual', 'group': 'curved', 'gap_mm': 6., 'imbalance': 0.},
)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha256(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.dtype.str, array.shape)).encode() + array.tobytes()).hexdigest()


# Verbatim function copied from OLD_COMMON; its frozen SHA256 is recorded above.
# This uses analytic rays and frontmost intersections with finite opaque plates.
# The old module is never imported (it has unrelated import-time dependencies).
def ray_hits(rng, frame, frames, gap, count, proportion):
    """Visible hits on two finite parallel opaque plates, not random overlap labels.

    Camera at z=220 mm. Rear at z=0, x in [-60,15]; front z=gap,
    x in [-15,60]. Select among physically visible hits to vary scan coverage.
    """
    camera = np.array([-35.+70.*frame/max(frames-1,1), -80., 220.])
    q = np.c_[rng.uniform(-70,70, max(count*30,6000)),
              rng.uniform(-45,45,max(count*30,6000)), np.zeros(max(count*30,6000))]
    direction = q-camera
    hits=[]; valid=[]
    for z, low, high in [(0.,-60.,15.),(gap,-15.,60.)]:
        t=(z-camera[2])/direction[:,2]
        hit=camera+t[:,None]*direction
        mask=(hit[:,0]>=low)&(hit[:,0]<=high)&(abs(hit[:,1])<=50)&(t>0)
        hits.append(hit); valid.append(mask)
    labels=np.where(valid[1],1,0)  # front intersection is always nearer
    usable=valid[0]|valid[1]
    physical=np.where(labels[:,None]==1,hits[1],hits[0])
    n1=int(round(count*proportion)); indices=[]
    for label,n in [(0,count-n1),(1,n1)]:
        choices=np.flatnonzero(usable&(labels==label))
        if n and len(choices)==0: raise RuntimeError('Ray generator has no requested visible surface')
        if n: indices.extend(rng.choice(choices,n,replace=len(choices)<n).tolist())
    indices=np.array(indices); rng.shuffle(indices)
    p=physical[indices]
    return p,labels[indices],np.repeat(camera[None],count,axis=0)


def scene_geometry(scene, seed):
    """Geometry stream independent of epsilon, amplitude and shuffle repeat."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    xyz, labels, frames, origins = [], [], [], []
    for frame in range(FRAMES):
        proportion = .5 + scene['imbalance'] * (frame / (FRAMES - 1) - .5)
        if scene['family'] == 'single':
            proportion = 0.
        if scene['group'] == 'raycast':
            clean, layer, origin = ray_hits(rng, frame, FRAMES, scene['gap_mm'], POINTS_PER_FRAME, proportion)
            # Canonical exact plate heights remove only ray intersection roundoff.
            clean[:, 2] = scene['gap_mm'] * layer
        else:
            xy = rng.uniform(-40., 40., (POINTS_PER_FRAME, 2))
            n1 = int(round(POINTS_PER_FRAME * proportion))
            layer = np.r_[np.zeros(POINTS_PER_FRAME - n1, dtype=np.int64), np.ones(n1, dtype=np.int64)]
            rng.shuffle(layer)
            clean = np.c_[xy, scene['gap_mm'] * layer]
            if scene['group'] == 'curved':
                clean[:, 2] += .0015 * (xy[:, 0] ** 2 + .5 * xy[:, 1] ** 2)
            origin = np.full((POINTS_PER_FRAME, 3), np.nan)
        xyz.append(clean)
        labels.append(layer)
        frames.append(np.full(POINTS_PER_FRAME, frame, dtype=np.int64))
        origins.append(origin)
    return np.concatenate(xyz), np.concatenate(labels), np.concatenate(frames), np.concatenate(origins)


def frame_bias(seed, amplitude):
    """Centered across equally populated frames; requested RMS, not sine amplitude."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 303]))
    raw = rng.normal(size=FRAMES)
    raw -= raw.mean()
    if amplitude == 0:
        return np.zeros(FRAMES)
    return raw * (amplitude / np.sqrt(np.mean(raw ** 2)))


def identity_error_stats(perturbation):
    """Order-independent accurate reductions of the actual stored scalar errors."""
    values = sorted(float(value) for value in np.asarray(perturbation))
    return {
        'normal_mae_mm': math.fsum(abs(value) for value in values) / len(values),
        'point_rmse_mm': math.sqrt(math.fsum(value * value for value in values) / len(values)),
    }


def paired_arrays(scene, seed, amplitude):
    clean, labels, frames, origins = scene_geometry(scene, seed)
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    eps = noise_rng.normal(0., SIGMA_MM, len(frames))
    bias = frame_bias(seed, amplitude)
    total = eps + bias[frames]
    base = clean.copy()
    base[:, 2] += eps
    arms = [('correlated', -1, total.copy(), None)]
    for repeat in range(2):
        # Repeat stream is frozen and independent of geometry/epsilon/bias streams.
        shuffle_seed = [seed, 404, repeat]
        rng = np.random.default_rng(np.random.SeedSequence(shuffle_seed))
        shuffled = total.copy()
        for layer in np.unique(labels):
            indices = np.flatnonzero(labels == layer)
            shuffled[indices] = total[rng.permutation(indices)]
        arms.append((f'shuffle_{repeat}', repeat, shuffled, shuffle_seed))
    return clean, labels, frames, origins, eps, bias, base, arms


def generate_dataset(destination=ROOT / 'data'):
    destination = Path(destination).resolve()
    if socket.gethostname() != 'liekkas':
        raise RuntimeError('Expected target host liekkas; refusing another environment')
    if destination != ROOT / 'data':
        raise ValueError('This frozen run writes only the designated data/ directory')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {destination}')
    if sha256(OLD_COMMON) != OLD_SHA256:
        raise RuntimeError('Frozen source hash changed')
    source_files = [ROOT / 'PROTOCOL.md', Path(__file__).resolve(), ROOT / 'test_generation.py', OLD_COMMON]
    source_hashes = {str(path): sha256(path) for path in source_files}
    destination.mkdir(exist_ok=False)
    (destination / 'inputs').mkdir()
    (destination / 'eval').mkdir()
    entries = []
    for scene in SCENES:
        for seed in SEEDS:
            for amplitude in AMPLITUDES:
                clean, labels, frames, origins, eps, bias, base, arms = paired_arrays(scene, seed, amplitude)
                pair_id = f"{scene['family']}_s{seed}_a{amplitude:g}"
                for arm, repeat, error, shuffle_seed in arms:
                    case_id = f'{pair_id}_{arm}'
                    input_relative = f'inputs/{case_id}.npz'
                    eval_relative = f'eval/{case_id}.npz'
                    observed = clean.copy()
                    observed[:, 2] += error
                    with (destination / input_relative).open('xb') as stream:
                        np.savez_compressed(stream, xyz_mm=observed, frame=frames, sigma_mm=np.array(SIGMA_MM))
                    with (destination / eval_relative).open('xb') as stream:
                        np.savez_compressed(stream, clean_xyz_mm=clean, labels=labels,
                                            perturbation_mm=error, base_xyz_mm=base)
                    stats = identity_error_stats(error)
                    entry = {
                        'id': case_id, 'pair_id': pair_id, 'family': scene['family'], 'group': scene['group'],
                        'gap_mm': scene['gap_mm'], 'seed': seed, 'amplitude_mm': amplitude,
                        'arm': arm, 'repeat': repeat, 'input_path': input_relative, 'eval_path': eval_relative,
                        'input_sha256': sha256(destination / input_relative),
                        'eval_sha256': sha256(destination / eval_relative),
                        'geometry_sha256': array_sha256(clean), 'frame_sha256': array_sha256(frames),
                        'labels_sha256': array_sha256(labels), 'base_xyz_sha256': array_sha256(base),
                        'point_count': len(frames), 'frames': FRAMES, 'points_per_frame': POINTS_PER_FRAME,
                        'sigma_mm': SIGMA_MM, 'shuffle_seed_entropy': shuffle_seed,
                        'frame_bias_mm': bias.tolist(), 'frame_bias_mean_mm': float(bias.mean()),
                        'frame_bias_rms_mm': float(np.sqrt(np.mean(bias ** 2))),
                        'epsilon_sample_mean_mm': float(eps.mean()), 'epsilon_sample_rms_mm': float(np.sqrt(np.mean(eps ** 2))),
                        'total_error_rms_mm': stats['point_rmse_mm'], 'identity_error_from_u': stats,
                        'frame_mean_total_error_mm': [float(error[frames == frame].mean()) for frame in range(FRAMES)],
                        'layer1_fraction_by_frame': [float(labels[frames == frame].mean()) for frame in range(FRAMES)],
                    }
                    if scene['group'] == 'raycast':
                        entry['camera_origins_by_frame_mm'] = origins[::POINTS_PER_FRAME].tolist()
                    entries.append(entry)
    manifest = {
        'protocol_version': 'correlation-v1', 'host': socket.gethostname(), 'data_directory': str(destination),
        'source_sha256': source_hashes, 'numpy_version': np.__version__, 'seeds': list(SEEDS),
        'amplitudes_mm': list(AMPLITUDES), 'scene_count': len(SCENES), 'pair_count': 54,
        'input_count': len(entries), 'legal_input_keys': ['xyz_mm', 'frame', 'sigma_mm'],
        'eval_keys': ['clean_xyz_mm', 'labels', 'perturbation_mm', 'base_xyz_mm'],
        'normal_axis': [0., 0., 1.], 'curved_height_formula_mm': 'h(x,y)=0.0015*(x^2+0.5*y^2), x/y in mm',
        'geometry_seed_entropy': '[seed,101]', 'epsilon_seed_entropy': '[seed,202]',
        'bias_seed_entropy': '[seed,303]', 'shuffle_seed_entropy': '[seed,404,repeat]',
        'notes': [
            'u=b_frame+epsilon; epsilon sampled independently from Normal(0,1 mm), without sample centering or normalization.',
            'b is centered across eight equally populated frames and normalized to the requested RMS within float64 precision.',
            'Shuffle is a within-true-layer finite-population permutation of total u; it is not strict iid.',
            'All arms keep row identity, clean geometry, frame membership and base_xyz_mm=clean+epsilon fixed.',
            'At amplitude zero, shuffles still permute epsilon and are not the same input array.',
            'dual_imbalanced means changing per-frame layer proportions (0.05 to 0.95); aggregate layer counts remain balanced.',
            'Ray scenes select frontmost physically visible hits to prescribed coverage; this is not uniform camera pixel sampling.',
            'Curved probe uses shared height plus a 6 mm axial separation; displacement is along e_z, not local curved normals.',
            'Identity MAE/RMSE from sorted u use stable sums and agree exactly within each pair; coordinate subtraction may differ at roundoff.',
            'Additional sensitivity uses the configuration scale hypot(sigma,amplitude), without reading EVAL, and is distinct from the fixed sigma=1 mm primary run.',
        ],
        'entries': entries,
    }
    if sha256(OLD_COMMON) != OLD_SHA256:
        raise RuntimeError('Frozen source changed during generation')
    with (destination / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = generate_dataset()
    print(json.dumps({'data': result['data_directory'], 'inputs': result['input_count'],
                      'pairs': result['pair_count'], 'scenes': result['scene_count']}))
