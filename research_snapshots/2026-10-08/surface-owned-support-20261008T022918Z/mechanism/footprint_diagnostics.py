"""Post-seal physical footprint audit; never fed into observation policy."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_mechanism as render


def ownership(scene, camera, uv):
    """Target contribution of bilinear interpolation over integrated pixels."""
    lo = np.floor(uv)
    frac = uv-lo
    result = np.zeros(len(uv))
    for by in (0, 1):
        for bx in (0, 1):
            weight = (frac[:, 0] if bx else 1-frac[:, 0])*(frac[:, 1] if by else 1-frac[:, 1])
            pixel = lo + np.array([bx, by])
            value = np.zeros(len(uv))
            for oy in (-1/3, 0, 1/3):
                for ox in (-1/3, 0, 1/3):
                    hit = render.intersect(scene, camera, pixel+np.array([ox, oy]))
                    value += (hit['owner'] == scene['target_id'])/9
            result += weight*value
    return result


def main():
    observations = json.loads((ROOT/'OBSERVATIONS.json').read_text())
    assert render.digest(ROOT/'OBSERVATIONS.json') == json.loads((ROOT/'OBSERVATIONS_LOCK.json').read_text())['observations_sha256']
    fixtures = {f['id']:f for f in json.loads((ROOT/'FIXTURES.json').read_text())}
    rows = []
    for obs in observations:
        meta = fixtures[obs['fixture_id']]
        actual = [render.deserialize_cam(c) for c in meta['actual_cameras']]
        supplied = [render.deserialize_cam(c) for c in meta['score_cameras']]
        with np.load(ROOT/'curves'/obs['curve_file']) as data:
            mask = data['mask']
        yy, xx = np.mgrid[-4:5, -4:5]
        offsets = np.stack((xx, yy), -1)[mask]
        xy = np.array([64.,64.])
        target = render.intersect(meta['scene'], actual[0], xy)
        z = target['depth']
        points = supplied[0]['C']+z*render.ray(supplied[0], xy+offsets)
        per_source = []
        for i in (1,2):
            if obs['arm'].startswith('translation_'):
                uv = render.project(supplied[i], target['point'])+offsets
            else:
                uv = render.project(supplied[i], points)
            fractions = ownership(meta['scene'], actual[i], uv)
            per_source.append(dict(mean_target_contribution=float(fractions.mean()),
                                   min_target_contribution=float(fractions.min()),
                                   mixed_samples=int(np.sum((fractions>1e-8)&(fractions<1-1e-8))),
                                   fully_target_samples=int(np.sum(fractions>=1-1e-8))))
        rows.append(dict(fixture_id=obs['fixture_id'],name=meta['name'],seed=meta['seed'],arm=obs['arm'],
                         samples=int(mask.sum()),source_footprints_at_true_depth=per_source))
    render.dump(ROOT/'FOOTPRINT_DIAGNOSTICS.json', rows)
    for name in ('ring9_flat','ring25_flat'):
        sample = next(r for r in rows if r['name']==name and r['arm']=='plane_connected9')
        print(json.dumps(sample))


if __name__ == '__main__':
    main()
