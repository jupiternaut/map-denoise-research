"""Read sealed inference, then score fixed native rows against held-out reference.

Random and oracle decisions exist only here, never in the deployable interface.
"""
from common import *
import csv
import time
from scipy.io import loadmat
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

BINS = np.array([0., 1e-7, .25, .5, 1., 2., 3., 4., 6.000001, np.inf])
NUMERIC = ('source_MSE_mm2', 'source_MAE_mm', 'source_p95_mm',
           'improved_fraction', 'harmed_fraction', 'accepted_fraction',
           'accepted_support_fraction', 'moved_fraction', 'move_RMS_mm',
           'benefit_sum_mm2', 'harm_sum_mm2')


def matched_random(accept, support, movement, mode, seed):
    rng = np.random.default_rng(seed)
    bins = np.digitize(movement, BINS[1:-1], right=False) if mode == 'bin' else np.zeros(len(accept), int)
    groups = bins * 2 + support.astype(int)
    result = np.zeros(len(accept), dtype=bool)
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        count = int(accept[ids].sum())
        result[rng.choice(ids, size=count, replace=False)] = True
        if result[ids].sum() != count:
            raise AssertionError('random control count mismatch')
    return result


def measure(d0, d1, support, move2, accept):
    after = np.where(accept, d1, d0)
    delta = after[support] - d0[support]
    gain = d0[support] ** 2 - after[support] ** 2
    actual_move2 = np.where(accept, move2, 0.)
    return dict(source_MSE_mm2=float(np.mean(after[support] ** 2)),
                source_MAE_mm=float(np.mean(after[support])),
                source_p95_mm=float(np.quantile(after[support], .95)),
                improved_fraction=float(np.mean(delta < -.1)),
                harmed_fraction=float(np.mean(delta > .1)),
                accepted_fraction=float(accept.mean()),
                accepted_support_fraction=float(accept[support].mean()),
                moved_fraction=float(np.mean(actual_move2 > 1e-14)),
                move_RMS_mm=float(np.sqrt(actual_move2.mean())),
                benefit_sum_mm2=float(np.maximum(gain, 0).sum()),
                harm_sum_mm2=float(np.maximum(-gain, 0).sum()))


def summarize(rows):
    methods = sorted({r['arm'] for r in rows})
    lookup = {(r['scene'], r['roi'], r['condition'], r['arm']): r for r in rows}
    per_scene = {str(s): {c: {} for c in CONDS} for s in SCENES}
    pooled = {c: {} for c in CONDS}
    for condition in CONDS:
        for arm in methods:
            rr = [r for r in rows if r['condition'] == condition and r['arm'] == arm]
            if len(rr) != 12:
                raise AssertionError('incomplete twelve-ROI method cell')
            for s in SCENES:
                ss = [r for r in rr if r['scene'] == s]
                per_scene[str(s)][condition][arm] = {k:float(np.mean([r[k] for r in ss])) for k in NUMERIC}
            means = {k:float(np.mean([per_scene[str(s)][condition][arm][k] for s in SCENES])) for k in NUMERIC}
            base = [lookup[(r['scene'],r['roi'],condition,'identity')]['source_MSE_mm2'] for r in rr]
            values = [r['source_MSE_mm2'] for r in rr]
            means.update(MSE_gain_percent=100 * (1 - np.mean(values) / np.mean(base)),
                         wins=int(np.sum(np.array(values) < np.array(base) - 1e-12)),
                         ties=int(np.sum(np.abs(np.array(values)-np.array(base)) <= 1e-12)),
                         losses=int(np.sum(np.array(values) > np.array(base) + 1e-12)),
                         n_rois=12)
            pooled[condition][arm] = means
    return dict(per_scene=per_scene, exposed_replay=pooled, rows=len(rows), methods=methods)


def main():
    check_host()
    start = time.monotonic()
    inference_seals = {str(s):verify_seal(ROOT/'inference'/f'scan{s}') for s in SCENES}
    out = ROOT/'evaluation'
    out.mkdir(exist_ok=False)
    reference_root = DATA/'closeout-confirmation-v1'
    reference_manifest = json.loads((reference_root/'REFERENCE_MANIFEST.json').read_text())
    reference_hashes = {}
    for item in reference_manifest['records']:
        path = Path(item['path'])
        if path.parent.name in {f'scan{s}' for s in SCENES}:
            if sha(path) != item['sha256']:
                raise RuntimeError('reference hash changed')
            reference_hashes[str(path)] = item['sha256']
    save_json(out/'LOCK.json', dict(evaluator_sha256=sha(ROOT/'evaluate_replay.py'),
              inference_seals={str(s):sha(ROOT/'inference'/f'scan{s}'/'SEALED.json') for s in SCENES},
              reference_sha256=reference_hashes, data_role='EXPOSED_REPLAY_NOT_NEW_CONFIRMATION',
              support='native rows fixed across conditions and arms', voxel_mm=.8,
              primary='source_MSE_mm2', geometry='fixed row preserving A or identity'))
    with (CLOSEOUT/'evaluation/METRICS.csv').open() as f:
        old = {(int(r['scene']),r['roi'],r['condition'],r['arm']):r for r in csv.DictReader(f)}
    rows, checks, exported_checks = [], [], []
    for sid in SCENES:
        raw = CLOSEOUT/'confirmation'/f'scan{sid}'
        base = reference_root/'evaluation_only'/f'scan{sid}'
        laser = read_points(base/f'stl{sid:03d}_total.ply')
        obs = loadmat(base/f'ObsMask{sid}_10.mat')
        for roi in json.loads((raw/'ROIS.json').read_text()):
            if roi['status'] != 'READY':
                raise AssertionError('locked replay ROI unavailable')
            rid = roi['id']
            native = read_points(raw/(rid+'__native')/'identity.ply')
            support = in_box(native, roi['lo'], roi['hi']) & observed(native, obs)
            cached = np.load(CLOSEOUT/'evaluation'/(rid+'_native_support.npy'), allow_pickle=False)
            if not np.array_equal(cached, support) or not support.any():
                raise AssertionError('native support changed')
            ref = voxel(laser[in_box(laser, roi['lo'], roi['hi']) & observed(laser, obs)])
            tree = cKDTree(ref)
            for condition in CONDS:
                name = rid+'__'+condition
                case = raw/name
                inf = ROOT/'inference'/f'scan{sid}'/name
                dest = out/name
                dest.mkdir()
                p, a = read_points(case/'identity.ply'), read_points(case/'A_all.ply')
                if p.shape != a.shape or p.shape != native.shape:
                    raise AssertionError('row-preserving candidate changed')
                d0, d1 = tree.query(p, workers=1)[0], tree.query(a, workers=1)[0]
                move2 = np.sum((a-p)**2, axis=1)
                save_npz(dest/'point_errors.npz', d0=d0, d1=d1, support=support, movement_squared=move2)
                with np.load(inf/'decisions.npz', allow_pickle=False) as z:
                    masks = {k:z[k] for k in z.files}
                for target in ('frozen_gain','normalized_gain__balanced'):
                    for mode in ('count','bin'):
                        for seed in range(10):
                            masks[f'random__{target}__{mode}__seed{seed}'] = matched_random(
                                masks[target], support, np.sqrt(move2), mode, SEED+seed)
                masks['oracle_fixed_A'] = d1 < d0
                save_npz(dest/'diagnostic_decisions.npz', **{
                    k:v for k,v in masks.items() if k.startswith('random__') or k == 'oracle_fixed_A'})
                for arm, mask in masks.items():
                    if mask.dtype != bool or mask.shape != (len(p),):
                        raise AssertionError('invalid decision mask')
                    m = measure(d0, d1, support, move2, mask)
                    rows.append(dict(scene=sid, roi=rid, condition=condition, arm=arm,
                                     n_rows=len(p), n_source=int(support.sum()), n_reference=len(ref), **m))
                    if arm in ('identity','A_all','frozen_gain'):
                        previous = old[(sid,rid,condition,'post_A_keep' if arm=='frozen_gain' else arm)]
                        diff = max(abs(m[k]-float(previous[k])) for k in NUMERIC if k in previous)
                        if diff > 1e-8:
                            raise AssertionError(f'archived metrics mismatch {name} {arm} {diff}')
                        checks.append(dict(case=name, arm=arm, maximum_absolute_difference=diff))
                    if arm in ('frozen_gain','normalized_gain__balanced','normalized_gain__native_priority'):
                        saved = read_points(inf/(arm+'.ply'))
                        err = float(np.max(np.abs(saved-np.where(mask[:,None],a,p))))
                        if err != 0:
                            raise AssertionError('PLY differs from decisions')
                        exported_checks.append(dict(case=name,arm=arm,max_coordinate_difference_mm=err))
                print('SCORED', name, len(masks), 'arms', flush=True)
    with (out/'METRICS.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows)
    summary.update(wall_seconds=time.monotonic()-start,
                   aggregation='four ROI means per scene then three equally weighted scenes',
                   random_repeats_are_not_independent_scenes=True,
                   scope='fixed-row distance, not completeness or layer identity')
    save_json(out/'SUMMARY.json', summary)
    save_json(out/'BASELINE_REPRODUCTION.json', dict(status='PASS', checks=checks,
               maximum_absolute_difference=max(r['maximum_absolute_difference'] for r in checks),
               exported_ply_checks=exported_checks))
    seal(out)
    print('EVALUATION SEALED', len(rows), 'rows', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
