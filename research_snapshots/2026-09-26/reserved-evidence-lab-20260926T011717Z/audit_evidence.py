"""Independent array/provenance checks; no evaluator input access.

Run only after all five evidence scene seals exist. Each case is checked; one
fixed lexicographic case per scene also gets a nine-row direct-score replay.
"""
from common import *
import time
from scene_adapter import load_scene, camera
from v28_closeout.graph_field import estimate_normals
from v28_closeout.direct_evidence import score_patches
from threadpoolctl import threadpool_limits


def independently_summarize(scores):
    a = np.asarray(scores, dtype=float)
    assert a.ndim == 3 and a.shape[0] == 4 and a.shape[2] == 2
    assert not np.isinf(a).any()
    assert np.all(np.abs(a[np.isfinite(a)]) <= 1.000001)
    old, new = a[:, :, 0].T, a[:, :, 1].T
    v0, v1 = np.isfinite(old), np.isfinite(new)
    pair = v0 & v1
    c0, c1 = np.where(v0, 1-old, 1), np.where(v1, 1-new, 1)
    gain = np.where(pair, new-old, 0)
    n = pair.sum(1)
    denom = np.maximum(n, 1)
    mean = gain.sum(1)/denom
    low = np.min(np.where(pair, gain, np.inf), axis=1)
    high = np.max(np.where(pair, gain, -np.inf), axis=1)
    low[n == 0] = 0
    high[n == 0] = 0
    std = np.sqrt(np.where(pair, (gain-mean[:, None])**2, 0).sum(1)/denom)
    m0 = np.where(n > 0, np.where(pair, c0, 0).sum(1)/denom, 1)
    m1 = np.where(n > 0, np.where(pair, c1, 0).sum(1)/denom, 1)
    return np.column_stack((c0, c1, gain, pair, v0, v1, mean, low, high, std,
                            ((gain > 0) & pair).sum(1)/denom, n/4, m0, m1))


def input_spec(sid):
    if sid not in (24, 37):
        return dict(root=DATA/'closeout-confirmation-v1/inputs'/f'scan{sid}')
    return dict(spec=dict(images=DATA/f'loss-alignment-v23/scan{sid}/image',
        colmap=DATA/f'loss-alignment-v23/scan{sid}/sparse/0',
        cameras=DATA/('real-closure-v21/cameras_geosvr_linked.npz' if sid == 24 else
                      'reconstruction-v22-scan37/cameras.npz'),
        mesh=DATA/('published-outputs-v1/scan24_mesh.ply' if sid == 24 else
                   'reconstruction-v22-scan37/scan37_mesh.ply')))


def independent_rank(points, views, excluded):
    ranking = []
    homogeneous = np.column_stack((points, np.ones(len(points))))
    for name, cam in views.items():
        if name in excluded:
            continue
        h = homogeneous@cam['P'].T
        uv = np.divide(h[:, :2], h[:, 2, None],
                       out=np.full((len(points), 2), np.nan), where=abs(h[:, 2, None]) > 1e-12)
        keep = (np.isfinite(uv).all(1) & (h[:, 2] > 0) & (uv[:, 0] >= 8) &
                (uv[:, 1] >= 8) & (uv[:, 0] < cam['width']-8) &
                (uv[:, 1] < cam['height']-8))
        if keep.sum() >= 20:
            ranking.append((int(keep.sum()), name))
    return sorted(ranking, reverse=True)


def main():
    check_host()
    start = time.monotonic()
    training_manifest = json.loads((PREV/'data/TRAIN_MANIFEST.json').read_text())['records']
    with np.load(PREV/'data/train.npz', allow_pickle=False) as data:
        case_id, row_id = data['case_id'], data['row_id']
    train_ids = {r['case']: row_id[case_id == r['case_id']] for r in training_manifest}
    cases, view_checks, direct_checks = [], [], []
    for sid in (24, 37, 55, 65, 69):
        folder = ROOT/'evidence'/f'scan{sid}'
        verify_seal(folder)
        scene = load_scene(**input_spec(sid))
        views = json.loads((folder/'VIEWS.json').read_text())
        paths = cases_for_scene(sid)
        assert len(paths) == (12 if sid in (24, 37) else 20)
        for path in (p for p in paths if p.name.endswith('__native')):
            rid = path.name.split('__')[0]
            v = views[rid]
            original, reserved = v['original_views'], v['reserved_views']
            assert len(original) == len(set(original)) == 5
            assert len(reserved) == len(set(reserved)) == 4
            assert not set(original).intersection(reserved)
            assert original == case_metadata(path)['views']
            rank = independent_rank(read_points(path/'identity.ply'), scene['views'], set(original))
            assert reserved == [name for _, name in rank[:4]]
            assert v['eligible_rank'] == [list(r) for r in rank]
            assert all(sha(scene['views'][name]['path']) == v['image_sha256'][name]
                       for name in original+reserved)
            view_checks.append(dict(scene=sid, roi=rid, originals=original,
                                    reserved=reserved, eligible=len(rank)))
        camera_cache = {}
        for case_index, path in enumerate(paths):
            target = folder/path.name
            meta = json.loads((target/'META.json').read_text())
            v = views[path.name.split('__')[0]]
            assert case_metadata(path)['views'] == v['original_views']
            assert meta['views'] == v and meta['reference_access'] is False
            assert all(sha(path/f) == digest for f, digest in meta['source_sha256'].items())
            with np.load(target/'PAIRED.npz', allow_pickle=False) as arr:
                ids, fit, reserved = arr['row_ids'], arr['fit_scores'], arr['reserved_scores']
                F, R = arr['F'], arr['R']
            p, a = read_points(path/'identity.ply'), read_points(path/'A_all.ply')
            expected = train_ids[path.name] if sid in (24, 37) else np.arange(len(p))
            assert np.array_equal(ids, expected)
            assert p.shape == a.shape and len(ids) == meta['rows']
            assert np.unique(ids).size == len(ids) and ids.min() >= 0 and ids.max() < len(p)
            assert F.shape == R.shape == (len(ids), 32)
            assert fit.shape == reserved.shape == (4, len(ids), 2)
            assert np.isfinite(F).all() and np.isfinite(R).all()
            summary_error = max(float(np.max(abs(F-independently_summarize(fit)))),
                                float(np.max(abs(R-independently_summarize(reserved)))))
            assert summary_error <= 3e-7, (path.name, 'summary', summary_error)
            names = v['original_views']+v['reserved_views']
            for name in names:
                if name not in camera_cache:
                    camera_cache[name] = camera(scene['views'][name])
            reference = camera_cache[names[0]]
            rays = p[ids]-reference['center']
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            delta = np.sum((a[ids]-p[ids])*rays, axis=1)
            residual = float(np.max(abs(p[ids]+delta[:, None]*rays-a[ids])))
            assert residual <= 1e-8
            assert abs(residual-meta['max_ray_residual_mm']) <= 1e-12
            cases.append(dict(case=path.name, rows=len(ids), summary_max_abs_error=summary_error,
                              max_ray_residual_mm=residual))
            if case_index == 0:
                sample = np.unique(np.linspace(0, len(ids)-1, 9).astype(int))
                physical = ids[sample]
                normals = estimate_normals(p, 24)
                # Both original and held-out cohorts in one call use identical
                # geometry, normal and reference footprint; no GT or labels.
                evidence = score_patches(p[physical], normals[physical],
                    np.column_stack((np.zeros(len(sample)), delta[sample])),
                    reference, [camera_cache[name] for name in names[1:]],
                    mode='tangent', patch_radius=3, batch_size=256)
                saved = np.concatenate((fit[:, sample], reserved[:, sample]), axis=0)
                got = evidence['scores']
                assert np.array_equal(np.isnan(got), np.isnan(saved))
                finite = np.isfinite(saved)
                error = float(np.max(abs(got[finite]-saved[finite]))) if finite.any() else 0.
                assert error <= 1e-7
                assert np.allclose(evidence['candidates'][:, 1], a[physical], rtol=0, atol=1e-8)
                direct_checks.append(dict(case=path.name, sampled_rows=physical.tolist(),
                                          max_score_error=error))
        print('AUDITED EVIDENCE', sid, flush=True)
    assert len(cases) == 84 and len(view_checks) == 20 and len(direct_checks) == 5
    save_json(ROOT/'AUDIT_EVIDENCE.json', dict(status='PASS', checks=cases, views=view_checks,
        sampled_direct_replays=direct_checks, case_count=len(cases), roi_count=len(view_checks),
        reference_access=False, evaluator_results_read=False,
        scope='correction-source-disjoint; neither initial-MVS independence nor visibility certified',
        source_sha256={str(ROOT/name): sha(ROOT/name) for name in
                       ('audit_evidence.py', 'extract_evidence.py', 'paired_features.py', 'common.py')},
        wall_seconds=time.monotonic()-start))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
