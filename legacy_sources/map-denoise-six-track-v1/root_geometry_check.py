"""Post-freeze parameter/seed replication; not a new object-family benchmark."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
TRACK = ROOT / 'tracks/t3_t6'
OLD = Path('/home/grf/Documents/Codex/2026-09-10/map-denoise-v0')
sys.path.insert(0, str(OLD)); sys.path.insert(0, str(TRACK))
from geometry import sample, rotation, evaluate
from local_operator import denoise as candidate
from parallel_geometry_v5.association.operator import denoise as old_a
from parallel_geometry_v4.baselines.operators import denoise as baseline

DEST = Path('/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600/t3_t6/root-postfreeze-01')
CASES = [('parallel_plates', .0075, .0016, .65),
         ('slotted_sheet', .008, .0013, .5),
         ('concentric_shells', .006, .0012, .5)]
ARMS = ['identity', 'bilateral', 'old_A', 'pcl_mls_k8', 'pcl_mls_k12', 'pursuit_reference']


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    frozen = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in [TRACK/'local_operator.py', TRACK/'geometry.py']}
    manifest = {'candidate_hashes': frozen, 'cases': CASES, 'seeds': [82019, 82021],
                'arms': ARMS, 'n': 2400,
                'scope': 'new parameter/seed replication of exposed families, not independent real transfer'}
    (DEST/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for family, gap, sigma, balance in CASES:
        for seed in manifest['seeds']:
            rng = np.random.default_rng(seed)
            clean, _, _ = sample(family, 2400, gap, balance, rng)
            rot = rotation(seed+17)
            points = (clean + rng.normal(0, sigma, clean.shape)) @ rot.T
            ref, _, labels = sample(family, 20000, gap, .5, np.random.default_rng(seed+991))
            key = f'{family}-{seed}'
            np.savez_compressed(DEST/f'{key}-input.npz', points=points, rotation=rot)
            for arm in ARMS:
                start = time.perf_counter()
                if arm == 'identity': out = points.copy()
                elif arm == 'old_A': out, _ = old_a(points, {'fallback':'bilateral'})
                elif arm == 'pursuit_reference': out, _ = candidate(points)
                elif arm == 'bilateral': out, _ = baseline(points, {'method':'bilateral_normal','k':48,'iterations':2,'step':.8})
                else: out, _ = baseline(points, {'method':'pcl_mls','k':int(arm.rsplit('k',1)[1]),'polynomial_order':2})
                elapsed = time.perf_counter()-start
                path = DEST/f'{key}-{arm}.npy'; np.save(path, out)
                metrics = evaluate(family, np.load(path) @ rot, gap, ref, labels, len(points))
                rows.append({'family':family,'seed':seed,'arm':arm,'seconds':elapsed,'metrics':metrics})
            print(key, 'done', flush=True)
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==v for f,v in frozen.items())
    (DEST/'results.json').write_text(json.dumps({'manifest':manifest,'records':rows}, indent=2))
    print(json.dumps({'run':str(DEST),'outputs':len(rows)}), flush=True)


if __name__ == '__main__': main()
