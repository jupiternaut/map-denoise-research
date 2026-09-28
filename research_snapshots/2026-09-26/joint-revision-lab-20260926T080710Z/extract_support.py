"""Candidate-specific paired R evidence at unchanged archived p/A/B positions."""
from common import *
import argparse
import resource
import time
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from scene_adapter import load_scene, camera
from v28_closeout.surfacelet import score_surfacelets
from v28_closeout.graph_field import estimate_normals
from paired_features import summarize_pairs
from extract_evidence import scene_spec
from support_features import choose_support, gather_reserved


def input_normal_bank(p, row_ids, tree, ref):
    """Original k8/k24/k64 PCA formula, queried only at requested input rows."""
    bank = []
    for k in (8, 24, 64):
        _, neighbors = tree.query(p[row_ids], k=min(k, len(p)), workers=1)
        q = p[neighbors] - p[neighbors].mean(axis=1, keepdims=True)
        covariance = np.einsum('nki,nkj->nij', q, q)
        _, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, :, 0]
        sign = np.where(normal[np.arange(len(normal)), np.argmax(abs(normal), axis=1)] < 0, -1, 1)
        bank.append(normal * sign[:, None])
    optical = np.asarray(ref['P'])[2, :3]
    optical = optical / np.linalg.norm(optical)
    return np.stack(bank + [np.broadcast_to(optical, (len(row_ids), 3))], axis=1)


def maximum_score_error(actual, archived):
    if actual.shape != archived.shape or not np.array_equal(np.isnan(actual), np.isnan(archived)):
        raise AssertionError('h5 archived score validity or shape mismatch')
    finite = np.isfinite(actual)
    error = float(np.max(abs(actual[finite] - archived[finite]))) if finite.any() else 0.
    if error > 1e-6:
        raise AssertionError(('h5 archived score mismatch', error))
    return error


def main():
    check_host()
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, required=True, choices=(24, 37, 55, 65, 69))
    parser.add_argument('--chunk', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.chunk <= 4096 or not 1 <= args.batch <= args.chunk:
        raise ValueError('require 1 <= batch <= chunk <= 4096')
    sid = args.scene
    if not (ROOT / 'PROTOCOL.md').is_file():
        raise RuntimeError('freeze protocol before extraction')
    for source in (RESERVED, CROSS):
        verify_seal(source / 'evidence' / f'scan{sid}')
    out = ROOT / 'evidence' / f'scan{sid}'
    out.mkdir(parents=True, exist_ok=False)
    paths = cases_for_scene(sid)
    assert len(paths) == (12 if sid in (24, 37) else 20)
    source_paths = [ROOT / n for n in ('common.py', 'PROTOCOL.md', 'support_features.py', 'extract_support.py', 'test_support.py')]
    source_paths += [RESERVED / 'paired_features.py', RESERVED / 'extract_evidence.py',
                    CLOSEOUT / 'scene_adapter.py', CLOSEOUT / 'package/v28_closeout/surfacelet.py',
                    CLOSEOUT / 'package/v28_closeout/direct_evidence.py',
                    CLOSEOUT / 'package/v28_closeout/graph_field.py']
    source_paths += [source / 'evidence' / f'scan{sid}' / 'SEALED.json' for source in (RESERVED, CROSS)]
    train_ids = {}
    if sid in (24, 37):
        verify_seal(BASE / 'data')
        source_paths += [BASE / 'data/train.npz', BASE / 'data/TRAIN_MANIFEST.json']
        manifest = json.loads((BASE / 'data/TRAIN_MANIFEST.json').read_text())['records']
        with np.load(BASE / 'data/train.npz', allow_pickle=False) as z:
            ci, ri = z['case_id'], z['row_id']
        train_ids = {r['case']: ri[ci == r['case_id']] for r in manifest if r['scene'] == sid}
    sources = {str(p): sha(p) for p in source_paths}
    save_json(out / 'LOCK.json', dict(source_sha256=sources, scene=sid, reference_access=False,
        chunk=args.chunk, batch=args.batch, coordinates='frozen p/A/B',
        support_choice='B final coordinate, original F only, best three requiring two finite',
        R='same chosen h at both p and B', A='reuse sealed R32 at fixed h5'))
    scene = load_scene(**scene_spec(sid))
    assert scene['audit'] == json.loads((RESERVED / 'evidence' / f'scan{sid}' / 'CALIBRATION.json').read_text())
    save_json(out / 'CALIBRATION.json', scene['audit'])
    views = json.loads((RESERVED / 'evidence' / f'scan{sid}' / 'VIEWS.json').read_text())
    assert views == json.loads((CROSS / 'evidence' / f'scan{sid}' / 'VIEWS.json').read_text())
    save_json(out / 'VIEWS.json', views)
    cams = {}
    for view in views.values():
        original, reserved = view['original_views'], view['reserved_views']
        assert len(original) == len(set(original)) == 5
        assert len(reserved) == len(set(reserved)) == 4 and not set(original) & set(reserved)
        for name in original + reserved:
            assert sha(scene['views'][name]['path']) == view['image_sha256'][name]
            if name not in cams:
                cams[name] = camera(scene['views'][name])
    records = []
    start = time.monotonic()
    for path in paths:
        tick = time.monotonic()
        dest = out / path.name
        dest.mkdir()
        view = views[path.name.split('__')[0]]
        construction_name = 'construction.json' if sid in (24, 37) else 'CONSTRUCTION.json'
        filenames = ['identity.ply', 'A_all.ply', 'B_all.ply', construction_name]
        hashes = {f: sha(path / f) for f in filenames}
        frozen = json.loads((path.parent / 'SEALED.json').read_text())['files']
        assert all(h == frozen[path.name + '/' + f] for f, h in hashes.items())
        assert json.loads((path / construction_name).read_text())['views'] == view['original_views']
        p, a, b = (read_points(path / (name + '.ply')) for name in ('identity', 'A_all', 'B_all'))
        assert p.shape == a.shape == b.shape
        ids = train_ids[path.name] if sid in (24, 37) else np.arange(len(p))
        prior_a = RESERVED / 'evidence' / f'scan{sid}' / path.name / 'PAIRED.npz'
        prior_b = CROSS / 'evidence' / f'scan{sid}' / path.name / 'PAIRED.npz'
        with np.load(prior_a, allow_pickle=False) as z:
            assert np.array_equal(z['row_ids'], ids)
            A = z['R'].copy()
            a_scores = np.concatenate([z['fit_scores'], z['reserved_scores']])
        with np.load(prior_b, allow_pickle=False) as z:
            assert np.array_equal(z['row_ids'], ids)
            b_scores = np.concatenate([z['fit_scores'], z['reserved_scores']])
        B = np.empty((len(ids), 32), dtype=np.float32)
        chosen_h = np.empty(len(ids), dtype=np.int8)
        ref = cams[view['original_views'][0]]
        cameras = [cams[n] for n in view['original_views'][1:] + view['reserved_views']]
        ray = p[ids] - ref['center']
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        offset_b = np.sum((b[ids] - p[ids]) * ray, axis=1)
        residual = float(np.max(abs(p[ids] + offset_b[:, None] * ray - b[ids])))
        assert residual <= 1e-8, 'B coordinate is not frozen ray position'
        tree = cKDTree(p)
        max_b_error = 0.
        max_a_error = 0.
        max_normal_error = 0.
        # Direct h5 A replay is small; A's full feature array is sealed reuse.
        sample = np.unique(np.linspace(0, len(ids) - 1, min(16, len(ids)), dtype=int))
        bank = input_normal_bank(p, ids[sample], tree, ref)
        for j, k in enumerate((8, 24, 64)):
            expected = estimate_normals(p, k)[ids[sample]]
            max_normal_error = max(max_normal_error, float(np.max(abs(bank[:, j] - expected))))
        assert max_normal_error <= 1e-12
        offset_a = np.sum((a[ids[sample]] - p[ids[sample]]) * ray[sample], axis=1)
        ev_a = score_surfacelets(p[ids[sample]], bank,
            np.column_stack([np.zeros(len(sample)), offset_a]), ref, cameras, batch_size=args.batch)
        assert np.allclose(ev_a['candidates'][:, 1], a[ids[sample]], atol=1e-8, rtol=0)
        max_a_error = maximum_score_error(ev_a['scores'][:, :, 5], a_scores[:, sample])
        for begin in range(0, len(ids), args.chunk):
            end = min(begin + args.chunk, len(ids))
            bank = input_normal_bank(p, ids[begin:end], tree, ref)
            offsets = np.column_stack([np.zeros(end - begin), offset_b[begin:end]])
            ev = score_surfacelets(p[ids[begin:end]], bank, offsets, ref, cameras, batch_size=args.batch)
            assert np.array_equal(ev['candidates'][:, 0], p[ids[begin:end]])
            assert np.allclose(ev['candidates'][:, 1], b[ids[begin:end]], atol=1e-8, rtol=0)
            max_b_error = max(max_b_error, maximum_score_error(ev['scores'][:, :, 5], b_scores[:, begin:end]))
            h = choose_support(ev['scores'][:4, :, :, 1])
            pair = gather_reserved(ev['scores'][4:], h)
            B[begin:end] = summarize_pairs(pair)
            chosen_h[begin:end] = h
        assert np.isfinite(A).all() and np.isfinite(B).all()
        assert A.shape == B.shape == (len(ids), 32)
        save_npz(dest / 'SUPPORT.npz', row_ids=ids, A=A, B=B, chosen_h=chosen_h)
        record = dict(case=path.name, rows=len(ids), full_rows=len(p), source_path=str(path),
            source_sha256=hashes, paired_source_sha256={str(f): sha(f) for f in (prior_a, prior_b)},
            max_B_ray_residual_mm=residual, h5_B_max_score_error=max_b_error,
            h5_A_sample_max_score_error=max_a_error, h5_A_sample_row_ids=ids[sample].tolist(),
            sampled_normal_max_error=max_normal_error, A_R_reuse_exact=True,
            chosen_h_counts=np.bincount(chosen_h.astype(int) + 1, minlength=21).tolist(),
            missing_F_fraction=float(np.mean(chosen_h < 0)),
            R_two_valid_fraction=float(np.mean(B[:, 29] >= .5)),
            reference_access=False, wall_seconds=time.monotonic() - tick)
        save_json(dest / 'META.json', record)
        records.append(record)
        print('SUPPORT', path.name, len(ids), 'rows', round(record['wall_seconds'], 2), 's', flush=True)
    assert all(sha(p) == h for p, h in sources.items()), 'source changed during extraction'
    for view in views.values():
        assert all(sha(scene['views'][n]['path']) == h for n, h in view['image_sha256'].items())
    save_json(out / 'SUMMARY.json', dict(records=records, wall_seconds=time.monotonic() - start,
        rows=sum(r['rows'] for r in records), gpu=False, chunk=args.chunk,
        peak_own_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    seal(out, sources)
    print('SUPPORT SEALED', sid, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
