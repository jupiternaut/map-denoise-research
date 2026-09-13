"""T4: explicit geometry-metric probes and a real provenance support experiment.

No filter is credited with the hand-constructed evaluator probe outputs. Oxford
patches are selected only from observations; no GT is used to decide eligibility.
"""
import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OBS = Path('/srv/slam-research/grf/map-denoise/runs/parallel-geometry-v5/oxford-observed-space-1zisp54k')
OLD = Path('/srv/slam-research/grf/map-denoise/runs/parallel-geometry-v4/real-data/run-ghbm8z8u')
DEST = Path('/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600/t4')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def metric(points, reference, levels):
    a = cKDTree(reference).query(points)[0]
    b = cKDTree(points).query(reference)[0]
    prf = []
    for tolerance in [0.001, 0.003, 0.05]:
        precision, recall = float(np.mean(a <= tolerance)), float(np.mean(b <= tolerance))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        prf.append({'tolerance_m': tolerance, 'precision': precision, 'recall': recall, 'f1': f1})
    support = [float(np.mean(np.abs(points[:, 2] - z) <= .001)) for z in levels]
    return {'chamfer_mm': float(.5 * (a.mean() + b.mean()) * 1000),
            'plane_distance_mm': float(np.min(np.abs(points[:, 2, None] - np.array(levels)[None, :]), axis=1).mean() * 1000),
            'prf': prf, 'layer_band_support_fraction_1mm': support,
            'z_q90_minus_q10_mm': float(np.diff(np.quantile(points[:, 2], [.1, .9]))[0] * 1000)}


def metric_probes(run):
    # Same observed coordinates; two different evaluator-only truths. This is a
    # metric/information counterexample, not performance of a deployable method.
    axis = np.linspace(-.05, .05, 24)
    xx, yy = np.meshgrid(axis, axis)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    lower = np.column_stack([xy, np.full(len(xy), -.004)])
    upper = np.column_stack([xy, np.full(len(xy), .004)])
    split = np.concatenate([lower, upper])
    flat = split.copy(); flat[:, 2] = 0
    records = []
    for name, levels in [('true_two_layers', [-.004, .004]), ('single_wall_with_ghost', [0.])]:
        folder = run / name; folder.mkdir()
        reference = np.concatenate([np.column_stack([xy, np.full(len(xy), z)]) for z in levels])
        np.savez_compressed(folder / 'input.npz', points=split)
        np.savez_compressed(folder / 'evaluator_reference.npz', points=reference, levels=levels)
        for label, out in [('preserve_two_layers', split), ('merge_to_midplane', flat)]:
            np.savez_compressed(folder / f'{label}.npz', points=out)
            records.append({'world': name, 'constructed_output': label,
                            'metrics': metric(out, reference, levels)})
    assert np.array_equal(np.load(run / 'true_two_layers/input.npz')['points'],
                          np.load(run / 'single_wall_with_ghost/input.npz')['points'])
    assert all(r['metrics']['prf'][-1]['f1'] == 1. for r in records)
    assert records[0]['metrics']['plane_distance_mm'] == 0.
    assert records[1]['metrics']['plane_distance_mm'] > 3.99
    assert records[2]['metrics']['plane_distance_mm'] > 3.99
    assert records[3]['metrics']['plane_distance_mm'] == 0.
    return records


def support_graph(points, frames, normal):
    """Per-frame normal residual components; descriptive, not proof of layers."""
    residual = (points - points.mean(0)) @ normal
    q25, q75 = np.quantile(residual, [.25, .75])
    labels = residual > np.median(residual)
    ids, inverse = np.unique(frames, return_inverse=True)
    counts = np.zeros((len(ids), 2), dtype=int)
    np.add.at(counts, (inverse, labels.astype(int)), 1)
    both = np.min(counts, axis=1) >= 3
    overall = np.var(residual)
    between = sum(len(residual[inverse == i]) * float(np.mean(residual[inverse == i])) ** 2
                  for i in range(len(ids))) / len(residual)
    return {'points': len(points), 'frames': len(ids), 'frames_min3_each_median_side': int(both.sum()),
            'q75_minus_q25_mm': float((q75 - q25) * 1000),
            'between_frame_variance_fraction': float(between / max(overall, 1e-30)),
            'median_points_per_frame': float(np.median(np.sum(counts, axis=1)))}


def real_patches(run):
    summaries, rows, frozen = [], [], {}
    for roi in ['left', 'center', 'right']:
        path = OBS / roi / 'scan_provenance.npz'
        frozen[str(path)] = sha(path)
        data = np.load(path)
        points, frames = data['points'], data['frame_index']
        old = np.load(OLD / roi / 'input.npz')['raw']
        assert np.array_equal(points, old)
        roi_rows = []
        # A scale sweep checks whether the 5cm occupancy support statistic was
        # unfairly pessimistic for a neighbourhood-based algorithm.
        for size in [.05, .15, .30]:
            grid = np.floor(points / size).astype(np.int64)
            _, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
            order = np.argsort(inverse, kind='stable')
            offsets = np.r_[0, np.cumsum(counts)]
            eligible = np.flatnonzero(counts >= 30)
            # Deterministic bounded sample, selected without geometry scores/GT.
            if len(eligible) > 300:
                eligible = eligible[np.linspace(0, len(eligible)-1, 300).astype(int)]
            local = []
            for cell in eligible:
                indices = order[offsets[cell]:offsets[cell+1]]
                patch = points[indices]
                centered = patch - patch.mean(0)
                eig, vec = np.linalg.eigh(centered.T @ centered / len(patch))
                if eig[1] <= 1e-12:
                    continue
                planarity_ratio = float(eig[0] / eig[1])
                info = support_graph(patch, frames[indices], vec[:, 0])
                info.update({'roi': roi, 'cell_m': size, 'cell_index': int(cell),
                             'planarity_lambda_min_over_middle': planarity_ratio})
                local.append(info); roi_rows.append(info); rows.append(info)
                # Representative real cloud: first suitable patch, never selected
                # on output improvement, no output denoising claimed here.
                sample = run / f'{roi}-{size:.2f}-sample.npz'
                if not sample.exists() and info['frames'] >= 3 and planarity_ratio < .1:
                    np.savez_compressed(sample, points=patch, frame_index=frames[indices], normal=vec[:, 0])
            planar = [x for x in local if x['planarity_lambda_min_over_middle'] < .1]
            summaries.append({'roi': roi, 'cell_m': size, 'raw_points': len(points),
                              'eligible_cells_at_least_30_points': int(np.sum(counts >= 30)),
                              'tested_cells': len(local), 'planar_cells': len(planar),
                              'planar_cells_at_least_3_frames': sum(x['frames'] >= 3 for x in planar),
                              'planar_cells_at_least_2_crossing_frames': sum(x['frames_min3_each_median_side'] >= 2 for x in planar),
                              'median_between_frame_fraction': float(np.median([x['between_frame_variance_fraction'] for x in planar])) if planar else None})
        print(f'Oxford {roi}: {len(points)} points, {len(roi_rows)} tested cells', flush=True)
    return summaries, rows, frozen


def main():
    start = time.perf_counter()
    DEST.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-', dir=DEST))
    print(str(run), flush=True)
    probes = metric_probes(run)
    summaries, patches, hashes = real_patches(run)
    source = Path(__file__)
    (run / 'experiment.py').write_bytes(source.read_bytes())
    write(run / 'results.json', {'metric_probes': probes, 'real_summary': summaries,
          'real_patch_records': patches, 'input_hashes': hashes,
          'source_sha256': sha(source), 'elapsed_seconds': time.perf_counter()-start,
          'scope': 'evaluator counterexamples and observation-only real-data eligibility, NOT denoising success',
          'cautions': ['median split does not identify two physical surfaces',
                       'frame variation may include real geometry and visibility changes',
                       '5cm F-score cannot distinguish the 8mm synthetic layer alternatives',
                       'patch size increases support but can violate the common-bias/planarity model',
                       'Oxford sample normals are estimated, not evaluator truth']})
    print(json.dumps({'run': str(run), 'summary': summaries, 'seconds': time.perf_counter()-start}), flush=True)


if __name__ == '__main__':
    main()
