"""Descriptive depth-domain audit of sealed outputs; no alternative scoring run."""
import csv
import hashlib
import json
from pathlib import Path
import socket
import numpy as np

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/colmap-transfer-20260930T180000Z')
assert socket.gethostname() == 'liekkas'
assert Path(__file__).resolve().parent == ROOT / 'review'
plan = json.loads((ROOT / 'PLAN.json').read_text())
seal = json.loads((ROOT / 'PREDICTIONS_SEALED.json').read_text())
with (ROOT / 'evaluation/POINT_METRICS.csv').open() as f:
    scores = {(r['roi'], r['method'], int(r['query'])): r for r in csv.DictReader(f)}
records = []
for sid, scene in plan['scenes'].items():
    interval = json.loads((ROOT / f'RANGES_{sid}.json').read_text())['depth_range_mm']
    for roi in scene['rois']:
        pred_path = ROOT / 'predictions' / (roi['id'] + '.npz')
        cpu_path = ROOT / 'cpu' / (roi['id'] + '.npz')
        for path in [pred_path, cpu_path]:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == seal['files'][str(path.relative_to(ROOT))]
        with np.load(pred_path, allow_pickle=False) as new, np.load(cpu_path, allow_pickle=False) as cpu:
            cv = cpu['valid']; n = int(cv.sum())
            assert ((cpu['depth_mm'][cv] >= interval[0]) & (cpu['depth_mm'][cv] <= interval[1])).all()
            for arm in ['photo', 'geo']:
                z, valid = new[arm + '_depth_mm'], new[arm + '_valid']
                outside = valid & ((z < interval[0]) | (z > interval[1]))
                paired = outside & cv
                contribution = 0.
                for i in np.flatnonzero(paired):
                    old_d = float(scores[(roi['id'], 'CPU', int(i))]['distance_mm'])
                    new_d = float(scores[(roi['id'], arm, int(i))]['distance_mm'])
                    contribution += (old_d ** 2 - new_d ** 2) / (4 * n)
                records.append({'roi': roi['id'], 'scene': int(sid), 'arm': arm,
                    'configured_depth_minmax_mm': interval, 'cpu_valid': n,
                    'mvs_valid': int(valid.sum()), 'mvs_valid_depth_minmax_mm': [float(z[valid].min()), float(z[valid].max())],
                    'mvs_outside_configured_interval': int(outside.sum()),
                    'mvs_outside_on_cpu_support': int(paired.sum()),
                    'outside_query_indices': np.flatnonzero(outside).tolist(),
                    'outside_cpu_support_roi_equal_mse_gain_contribution_mm2': contribution})
summary = {}
for arm in ['photo', 'geo']:
    selected = [r for r in records if r['arm'] == arm]
    summary[arm] = {k: sum(r[k] for r in selected) for k in ['mvs_valid', 'mvs_outside_configured_interval',
        'mvs_outside_on_cpu_support', 'outside_cpu_support_roi_equal_mse_gain_contribution_mm2']}
payload = {'status': 'DESCRIPTIVE_ONLY', 'rule': 'All fixed points retained; no score recomputed after selecting or excluding queries.',
           'observation': 'CPU estimates stay within the configured sweep interval; COLMAP accepted estimates can leave it.',
           'source': 'https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/patch_match_cuda.cu',
           'primary_numeric_verifier': 'INDEPENDENT_NUMERIC_CHECK.json', 'summary': summary, 'records': records,
           'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (ROOT / 'review/DEPTH_RANGE_SUPPORT_CHECK.json').open('x') as f:
    json.dump(payload, f, indent=2, allow_nan=False); f.write('\n')
print(json.dumps(summary, indent=2))
