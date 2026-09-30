"""Independent audit of frozen inputs/predictions; never opens reference geometry.

Run on liekkas with Python + NumPy/Pillow. Read-only; prints deterministic checks.
No experiment modules are imported, and nothing is written by this verifier.
"""
import hashlib
import json
from pathlib import Path
import socket
import numpy as np
from PIL import Image

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/colmap-transfer-20260930T180000Z')
assert socket.gethostname() == 'liekkas'
assert Path(__file__).resolve().parent == ROOT / 'review'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def verify(mapping):
    for p, digest in mapping.items():
        target = ROOT / p
        assert 'references' not in target.parts, target
        assert sha(target) == digest, target
    return len(mapping)


def depth(p):
    with p.open('rb') as f:
        fields = []
        part = b''
        while len(fields) < 3:
            b = f.read(1)
            assert b, p
            if b == b'&':
                fields.append(int(part))
                part = b''
            else:
                part += b
        width, height, channels = fields
        a = np.fromfile(f, '<f4')
    assert len(a) == width * height * channels
    a = a.reshape((width, height, channels), order='F').transpose(1, 0, 2)
    return a[..., 0] if channels == 1 else a


def backproject(P, uv, z):
    P = np.asarray(P)
    A, b = P[:, :3], P[:, 3]
    return np.linalg.solve(A, (np.c_[uv, np.ones(len(uv))] * z[:, None] - b).T).T


run = load(ROOT / 'RUN_LOCK.json')
plan = load(ROOT / 'PLAN.json')
mvs = load(ROOT / 'MVS_LOCK.json')
seal = load(ROOT / 'PREDICTIONS_SEALED.json')
out = {'host': socket.gethostname(), 'root': str(ROOT), 'ground_truth_opened': False,
       'hash_counts': {'run': verify(run['files']), 'dependencies': verify(run['dependencies']),
                       'plan': verify(plan['source_files']), 'mvs': verify(mvs['files']),
                       'seal': verify(seal['files'])}, 'rois': [], 'jobs': []}
assert set(plan['scenes']) == {'118', '122'}
assert len(mvs['jobs']) == 14
checks = load(ROOT / 'CAMERA_CHECKS.json')
assert checks['plan_sha256'] == sha(ROOT / 'PLAN.json')
for sid, scene in plan['scenes'].items():
    assert scene['reference'] == '0022.png'
    assert scene['Q'] == ['0016.png', '0035.png', '0017.png', '0007.png']
    assert len(scene['rois']) == 2
    for roi in scene['rois']:
        uv = np.asarray(roi['pixel_xy'])
        x0, y0, x1, y1 = roi['box_xyxy_original']
        xx, yy = np.meshgrid(np.arange(x0 / 2 + 4, x1 / 2 - 4, 8.),
                             np.arange(y0 / 2 + 4, y1 / 2 - 4, 8.))
        grid = np.c_[xx.ravel(), yy.ravel()]
        expected = sorted(grid, key=lambda p: hashlib.sha256(p.astype(np.float64).tobytes()).digest())[:128]
        assert np.array_equal(uv, expected)
        assert len(set(map(tuple, uv))) == len(uv) == 128
        with np.load(ROOT / 'cpu' / (roi['id'] + '.npz'), allow_pickle=False) as cpu:
            assert np.array_equal(cpu['pixel_xy'], uv)
            cv = cpu['valid']
            assert np.array_equal(cv, np.isfinite(cpu['xyz_mm']).all(axis=1))
            cpu_xyz = backproject(scene['cameras'][scene['reference']]['P'], uv[cv], cpu['depth_mm'][cv])
            assert np.max(np.abs(cpu_xyz - cpu['xyz_mm'][cv])) < 1e-8
            assert np.array_equal(cv, np.isfinite(cpu['depth_mm']) & (cpu['score'] >= .6) & (cpu['variance'] > 1e-5))
            record = {'roi': roi['id'], 'requested': len(uv), 'cpu_valid': int(cv.sum())}
            with np.load(ROOT / 'predictions' / (roi['id'] + '.npz'), allow_pickle=False) as pred:
                assert np.array_equal(pred['pixel_xy'], uv)
                for arm, kind in [('photo', 'photometric'), ('geo', 'geometric')]:
                    dmap = depth(ROOT / 'workspaces' / f'scan{sid}' / arm / 'stereo/depth_maps' / (scene['reference'] + '.' + kind + '.bin'))
                    assert dmap.shape == (581, 777)
                    z = dmap[uv[:, 1], uv[:, 0]].astype(float)
                    pv = np.isfinite(z) & (z > 0)
                    assert np.array_equal(z, pred[arm + '_depth_mm'], equal_nan=True)
                    assert np.array_equal(pv, pred[arm + '_valid'])
                    assert np.array_equal(pv, np.isfinite(pred[arm + '_xyz_mm']).all(axis=1))
                    xyz = backproject(scene['cameras'][scene['reference']]['P'], uv[pv], z[pv])
                    assert np.max(np.abs(xyz - pred[arm + '_xyz_mm'][pv])) < 1e-8
                    record[arm] = {'valid': int(pv.sum()), 'cpu_overlap': int((pv & cv).sum()),
                                   'fallback_cpu_retained': int((~pv & cv).sum()), 'new_support': int((pv & ~cv).sum())}
            out['rois'].append(record)
    for name, camera in scene['cameras'].items():
        with Image.open(camera['image_path']) as im:
            rgb = np.asarray(im.convert('RGB').resize((777, 581), Image.Resampling.BILINEAR))
        gray = np.clip(rgb @ [.299, .587, .114], 0, 255).astype('uint8')
        for arm in ['photo', 'geo']:
            with Image.open(ROOT / 'workspaces' / f'scan{sid}' / arm / 'images' / name) as im:
                assert np.array_equal(np.asarray(im), gray)
for i, job in enumerate(mvs['jobs']):
    record = load(ROOT / 'job_records' / f'{i:02d}.json')
    assert record['job'] == i and record['seconds'] > 0
    verify(record['files'])
    kind = 'geometric' if job['options']['geom_consistency'] else 'photometric'
    dmap = depth(Path(job['workspace']) / 'stereo/depth_maps' / (job['reference'] + '.' + kind + '.bin'))
    assert record['valid_depths'] == int((dmap > 0).sum())
    out['jobs'].append({'job': i, 'scene': job['scene'], 'arm': job['variant'],
                        'reference': job['reference'], 'geom': job['options']['geom_consistency'],
                        'filter': job['options']['filter'], 'valid_depths': record['valid_depths']})
assert sum(r['requested'] for r in out['rois']) == 512
out['cpu_valid_total'] = sum(r['cpu_valid'] for r in out['rois'])
print(json.dumps(out, indent=2, allow_nan=False))
