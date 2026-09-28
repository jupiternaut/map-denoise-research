"""Use the same archived development rows and independently verify sealed B labels."""
from common import *
from scipy.io import loadmat
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
import time


def main():
    check_host()
    start = time.monotonic()
    verify_seal(BASE/'data')
    out = ROOT/'data'
    out.mkdir(exist_ok=False)
    specs = {
        24: (DATA/'published-outputs-v2-reference/stl024_total.ply',
             DATA/'published-outputs-v2-reference/ObsMask24_10.mat'),
        37: (DATA/'reconstruction-v22-scan37/stl037_total.ply',
             DATA/'reconstruction-v22-scan37/ObsMask37_10.mat'),
    }
    raw_manifest = json.loads((OLD/'real_results/SEALED.json').read_text())['files']
    label_manifest = json.loads((OLD/'selector_results/SEALED.json').read_text())['files']
    base_manifest = json.loads((BASE/'data/TRAIN_MANIFEST.json').read_text())
    source_paths = [ROOT/'PROTOCOL.md', ROOT/'prepare_data.py', ROOT/'common.py',
                    BASE/'data/train.npz', BASE/'data/TRAIN_MANIFEST.json', BASE/'data/SEALED.json',
                    BASE/'prepare_data.py', OLD/'real_results/SEALED.json',
                    OLD/'selector_results/SEALED.json']
    sources = {str(p): sha(p) for p in source_paths}
    with np.load(BASE/'data/train.npz', allow_pickle=False) as z:
        base_data = {k: z[k] for k in z.files}
    metadata_keys = ('e0', 'scene', 'condition', 'case_id', 'row_id', 'calibration_support')
    arrays = {k: base_data[k].copy() for k in metadata_keys}
    n = len(arrays['e0'])
    if n != 79594 or int(arrays['calibration_support'].sum()) != 78598:
        raise AssertionError('development row/support contract changed')
    arrays['X'] = np.empty((n, 64), dtype=base_data['X'].dtype)
    arrays['gain'] = np.empty(n, dtype=base_data['gain'].dtype)
    arrays['e1'] = np.empty(n, dtype=base_data['e1'].dtype)
    filled = np.zeros(n, bool)
    records, recomputation = [], []
    for scene, test in ((24, 37), (37, 24)):
        reference_path, obs_path = specs[scene]
        for path in (reference_path, obs_path):
            digest = sha(path)
            if digest != base_manifest['source_sha256'][str(path)]:
                raise AssertionError('development reference changed')
            sources[str(path)] = digest
        laser, obs = read_points(reference_path), loadmat(obs_path)
        trees, supports = {}, {}
        for record in base_manifest['records']:
            if record['scene'] != scene:
                continue
            case, cid, roi = record['case'], record['case_id'], record['roi']
            select = arrays['case_id'] == cid
            ids = arrays['row_id'][select]
            raw = OLD/'real_results'/case
            label_path = OLD/'selector_results'/f'train{scene}_test{test}'/(case+'_training_labels.npz')
            feature_path = raw/'features.npz'
            for path, prefix, manifest in ((label_path, OLD/'selector_results', label_manifest),
                                           (feature_path, OLD/'real_results', raw_manifest),
                                           (raw/'identity.ply', OLD/'real_results', raw_manifest),
                                           (raw/'B_all.ply', OLD/'real_results', raw_manifest),
                                           (raw/'roi.json', OLD/'real_results', raw_manifest)):
                digest = sha(path)
                if digest != manifest[str(path.relative_to(prefix))]:
                    raise AssertionError('archived source changed: '+str(path))
                sources[str(path)] = digest
            with np.load(label_path, allow_pickle=False) as z:
                if not np.array_equal(z['row_ids'], ids):
                    raise AssertionError('archived sampled row mismatch')
                gain = z['B_gain']
            with np.load(feature_path, allow_pickle=False) as z:
                features = z['B_all_post'][ids]
            e0 = arrays['e0'][select]
            e1 = e0-gain
            if features.shape != (len(ids), 64) or not np.isfinite(features).all():
                raise AssertionError('B64 feature contract')
            if not np.isfinite(gain).all() or np.min(e1) < -1e-8:
                raise AssertionError('B reference loss invalid')
            n_clipped = int(np.sum(e1 < 0))
            e1 = np.maximum(e1, 0)
            meta = json.loads((raw/'roi.json').read_text())
            lo, hi = np.asarray(meta['aabb_min_mm']), np.asarray(meta['aabb_max_mm'])
            if roi not in trees:
                ref = voxel(laser[in_box(laser, lo, hi) & observed(laser, obs)])
                trees[roi] = cKDTree(ref)
                native_path = OLD/'real_results'/(roi+'__native')/'identity.ply'
                native_digest = sha(native_path)
                if native_digest != raw_manifest[str(native_path.relative_to(OLD/'real_results'))]:
                    raise AssertionError('archived native geometry changed')
                sources[str(native_path)] = native_digest
                native = read_points(native_path)
                supports[roi] = in_box(native, lo, hi) & observed(native, obs)
            p, b = read_points(raw/'identity.ply'), read_points(raw/'B_all.ply')
            if p.shape != b.shape or len(p) != len(supports[roi]):
                raise AssertionError('B row correspondence broken')
            if not np.array_equal(arrays['calibration_support'][select], supports[roi][ids]):
                raise AssertionError('BASE calibration support changed')
            e0_check = trees[roi].query(p[ids], workers=1)[0]**2
            e1_check = trees[roi].query(b[ids], workers=1)[0]**2
            differences = dict(e0_max_abs=float(np.max(np.abs(e0_check-e0))),
                               e1_max_abs=float(np.max(np.abs(e1_check-e1))),
                               gain_max_abs=float(np.max(np.abs(e0_check-e1_check-gain))))
            if max(differences.values()) > 1e-8:
                raise AssertionError(('B label recomputation mismatch', case, differences))
            if len(ids) != record['rows'] or not np.all(arrays['scene'][select] == scene):
                raise AssertionError('BASE metadata identity mismatch')
            arrays['X'][select] = features
            arrays['gain'][select] = gain
            arrays['e1'][select] = e1
            filled[select] = True
            records.append(dict(record, B_e1_tiny_negative_clipped=n_clipped))
            recomputation.append(dict(case=case, rows=len(ids), **differences))
            print('B DEVELOPMENT', case, len(ids), differences, flush=True)
        del laser
    if not filled.all() or len(records) != 24:
        raise AssertionError('incomplete development assembly')
    for key in metadata_keys:
        if not np.array_equal(arrays[key], base_data[key]):
            raise AssertionError('BASE metadata was changed: '+key)
    save_npz(out/'train.npz', **arrays)
    save_json(out/'TRAIN_MANIFEST.json', dict(records=records, rows=n,
              calibration_rows=int(arrays['calibration_support'].sum()),
              source_sha256=sources, wall_seconds=time.monotonic()-start,
              feature_schema='B_all_post cached64 at exact previous sampled row IDs',
              label_source='sealed historical B_gain, e1=e0-B_gain; only tiny negatives clipped',
              error_unit='e0,e1,gain are squared millimetres; gain=e0-e1',
              labels_accessed_scenes=[24,37], new_replay_reference_access=False,
              same_base_metadata=list(metadata_keys),
              calibration='exact BASE sampled rows and fixed native support'))
    save_json(out/'B_LABEL_REPRODUCTION.json', dict(status='PASS', checks=recomputation,
              all_selected_development_rows_recomputed=True,
              maximum_absolute_difference=max(max(r[k] for k in ('e0_max_abs','e1_max_abs','gain_max_abs'))
                                              for r in recomputation)))
    if any(sha(Path(p)) != digest for p, digest in sources.items()):
        raise AssertionError('source modified during preparation')
    seal(out, sources)
    print('B DATA SEALED', n, 'rows', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
