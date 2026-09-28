"""Recompute development labels only; preserve archived sampled row identities."""
from common import *
from scipy.io import loadmat
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
import time


def main():
    check_host()
    out = ROOT / 'data'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    specs = {
        24: (DATA/'published-outputs-v2-reference/stl024_total.ply', DATA/'published-outputs-v2-reference/ObsMask24_10.mat'),
        37: (DATA/'reconstruction-v22-scan37/stl037_total.ply', DATA/'reconstruction-v22-scan37/ObsMask37_10.mat'),
    }
    raw_seal = json.loads((OLD/'real_results/SEALED.json').read_text())['files']
    label_seal = json.loads((OLD/'selector_results/SEALED.json').read_text())['files']
    source = {str(p): sha(p) for p in [ROOT/'PROTOCOL.md', ROOT/'prepare_data.py', ROOT/'common.py']}
    arrays = {k: [] for k in ('X', 'gain', 'e0', 'e1', 'scene', 'condition', 'case_id', 'row_id', 'calibration_support')}
    records = []
    for scene, test in ((24, 37), (37, 24)):
        ref_path, obs_path = specs[scene]
        source[str(ref_path)] = sha(ref_path)
        source[str(obs_path)] = sha(obs_path)
        full_ref = read_points(ref_path)
        obs = loadmat(obs_path)
        folder = OLD/'selector_results'/f'train{scene}_test{test}'
        files = sorted(folder.glob('*_training_labels.npz'))
        if len(files) != 12:
            raise AssertionError('expected twelve development cases per scene')
        support_cache, trees = {}, {}
        for label_path in files:
            case = label_path.name.removesuffix('_training_labels.npz')
            roi, condition = case.split('__')
            raw = OLD/'real_results'/case
            feature_path = raw/'features.npz'
            for path, prefix, manifest in ((label_path, OLD/'selector_results', label_seal),
                                            (feature_path, OLD/'real_results', raw_seal)):
                digest = sha(path)
                if digest != manifest[str(path.relative_to(prefix))]:
                    raise AssertionError('archived data changed')
                source[str(path)] = digest
            meta_path = raw/'roi.json'
            meta = json.loads(meta_path.read_text())
            lo, hi = np.asarray(meta['aabb_min_mm']), np.asarray(meta['aabb_max_mm'])
            source[str(meta_path)] = sha(meta_path)
            if roi not in trees:
                ref = voxel(full_ref[in_box(full_ref, lo, hi) & observed(full_ref, obs)])
                trees[roi] = cKDTree(ref)
                native_path = OLD/'real_results'/(roi+'__native')/'identity.ply'
                native = read_points(native_path)
                support_cache[roi] = in_box(native, lo, hi) & observed(native, obs)
                source[str(native_path)] = sha(native_path)
            with np.load(label_path, allow_pickle=False) as z:
                ids, previous_gain = z['row_ids'], z['A_gain']
            p, q = [read_points(raw/(arm+'.ply')) for arm in ('identity', 'A_all')]
            for arm in ('identity', 'A_all'):
                source[str(raw/(arm+'.ply'))] = sha(raw/(arm+'.ply'))
            if p.shape != q.shape or len(support_cache[roi]) != len(p):
                raise AssertionError('row correspondence broken')
            e0 = trees[roi].query(p[ids], workers=1)[0] ** 2
            e1 = trees[roi].query(q[ids], workers=1)[0] ** 2
            gain = e0 - e1
            diff = float(np.max(np.abs(gain-previous_gain)))
            if diff > 1e-8:
                raise AssertionError(('archived gain reproduction', case, diff))
            with np.load(feature_path, allow_pickle=False) as z:
                X = z['A_all_post'][ids]
            if X.shape != (len(ids), 64) or not np.isfinite(X).all():
                raise AssertionError('feature contract')
            cid = len(records)
            valid = support_cache[roi][ids]
            values = dict(X=X, gain=gain, e0=e0, e1=e1, row_id=ids,
                          scene=np.full(len(ids), scene, dtype=np.int16),
                          condition=np.full(len(ids), condition),
                          case_id=np.full(len(ids), cid, dtype=np.int16),
                          calibration_support=valid)
            for key, value in values.items():
                arrays[key].append(value)
            records.append(dict(case=case, scene=scene, roi=roi, condition=condition,
                                case_id=cid, rows=len(ids), calibration_rows=int(valid.sum()),
                                archived_gain_max_abs_difference=diff))
            print('DEVELOPMENT', case, len(ids), 'calibration', valid.sum(), flush=True)
        del full_ref
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    save_npz(out/'train.npz', **arrays)
    save_json(out/'TRAIN_MANIFEST.json', dict(records=records, rows=len(arrays['gain']),
              source_sha256=source, wall_seconds=time.monotonic()-start,
              error_unit='e0,e1,gain are squared millimetres; gain=e0-e1',
              labels_accessed_scenes=[24,37], new_replay_reference_access=False,
              calibration='archived sampled rows intersected with native fixed support'))
    if any(sha(Path(p)) != digest for p, digest in source.items()):
        raise AssertionError('source modified during preparation')
    seal(out, source)
    print('PREPARED', len(arrays['gain']), 'rows', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
