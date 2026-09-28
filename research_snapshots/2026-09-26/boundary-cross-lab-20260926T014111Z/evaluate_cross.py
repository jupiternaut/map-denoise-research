"""Evaluate frozen A/B decisions; GT-derived diagnostics never enter inference."""
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
COMPLEMENT = ('B_better_than_A_KEEP_fraction', 'B_incremental_MSE_mm2',
              'B_R_captured_incremental_MSE_mm2', 'B_R_selected_complement_fraction',
              'identity_MSE_mm2', 'oracle_A_MSE_mm2', 'oracle_AB_MSE_mm2')
POLICIES = ('balanced', 'native_priority', 'natural')
EXPECTED_ARMS = {'identity', 'A_all', 'B_all', 'frozen_gain', 'previous_normalized'} | {
    c+'_'+e+'__'+p for c in ('A', 'B') for e in ('F', 'R') for p in POLICIES} | {
    c+'_'+e+'_pair' for c in ('A', 'B') for e in ('F', 'R')}


def matched_random(accept, support, movement, mode, seed):
    rng = np.random.default_rng(seed)
    bins = np.digitize(movement, BINS[1:-1], right=False) if mode == 'bin' else np.zeros(len(accept), int)
    groups = bins*2 + support.astype(int)
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
    delta = after[support]-d0[support]
    gain = d0[support]**2-after[support]**2
    actual_move2 = np.where(accept, move2, 0.)
    return dict(source_MSE_mm2=float(np.mean(after[support]**2)),
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
    if len(lookup) != len(rows):
        raise AssertionError('duplicate metric row')
    per_scene = {str(s): {c: {} for c in CONDS} for s in SCENES}
    pooled = {c: {} for c in CONDS}
    for condition in CONDS:
        for arm in methods:
            rr = [r for r in rows if r['condition'] == condition and r['arm'] == arm]
            for sid in SCENES:
                ss = [r for r in rr if r['scene'] == sid]
                if len(ss) != 4 or len({r['roi'] for r in ss}) != 4:
                    raise AssertionError('each arm requires four ROIs in every scene/condition')
                cell = {k: float(np.mean([r[k] for r in ss])) for k in NUMERIC}
                base = np.mean([lookup[(sid, r['roi'], condition, 'identity')]['source_MSE_mm2'] for r in ss])
                cell['MSE_gain_percent'] = float(100*(1-cell['source_MSE_mm2']/base))
                per_scene[str(sid)][condition][arm] = cell
            means = {k: float(np.mean([per_scene[str(s)][condition][arm][k] for s in SCENES]))
                     for k in NUMERIC}
            base = np.array([lookup[(r['scene'], r['roi'], condition, 'identity')]['source_MSE_mm2'] for r in rr])
            values = np.array([r['source_MSE_mm2'] for r in rr])
            base_mse = np.mean([np.mean([lookup[(s, r['roi'], condition, 'identity')]['source_MSE_mm2']
                                        for r in rr if r['scene'] == s]) for s in SCENES])
            means.update(MSE_gain_percent=float(100*(1-means['source_MSE_mm2']/base_mse)),
                         wins=int(np.sum(values < base-1e-12)),
                         ties=int(np.sum(np.abs(values-base) <= 1e-12)),
                         losses=int(np.sum(values > base+1e-12)), n_rois=len(rr))
            pooled[condition][arm] = means
    interactions = {}
    for condition in CONDS:
        interaction = {}
        for policy in (*POLICIES, 'pair'):
            suffix = '_pair' if policy == 'pair' else '__'+policy
            gains = {k: pooled[condition][k+suffix]['MSE_gain_percent'] for k in ('A_F','A_R','B_F','B_R')}
            interaction[policy] = dict(A_R_minus_A_F_gain_pp=gains['A_R']-gains['A_F'],
                B_R_minus_B_F_gain_pp=gains['B_R']-gains['B_F'],
                candidate_by_evidence_interaction_gain_pp=(gains['B_R']-gains['B_F'])-(gains['A_R']-gains['A_F']))
        interactions[condition] = interaction
    return dict(per_scene=per_scene, exposed_replay=pooled, interactions=interactions,
                rows=len(rows), methods=methods)


def summarize_complement(rows):
    per_scene = {str(s): {} for s in SCENES}
    pooled = {}
    for condition in CONDS:
        for sid in SCENES:
            ss = [r for r in rows if r['scene'] == sid and r['condition'] == condition]
            if len(ss) != 4:
                raise AssertionError('incomplete complement cell')
            per_scene[str(sid)][condition] = {k: float(np.mean([r[k] for r in ss])) for k in COMPLEMENT}
        cell = {k: float(np.mean([per_scene[str(s)][condition][k] for s in SCENES])) for k in COMPLEMENT}
        cell['B_incremental_gain_pp'] = 100*cell['B_incremental_MSE_mm2']/cell['identity_MSE_mm2']
        cell['B_R_captured_incremental_gain_pp'] = 100*cell['B_R_captured_incremental_MSE_mm2']/cell['identity_MSE_mm2']
        cell['B_R_incremental_benefit_capture_fraction'] = (
            cell['B_R_captured_incremental_MSE_mm2']/cell['B_incremental_MSE_mm2']
            if cell['B_incremental_MSE_mm2'] > 0 else None)
        pooled[condition] = cell
    return dict(per_scene=per_scene, exposed_replay=pooled, cases=rows,
                interpretation='evaluator-only B improvement over pointwise min(A,identity); captured contribution is gross selected benefit, not net deployed gain',
                definition='incremental=max(min(d0^2,dA^2)-dB^2,0); capture=mean(incremental*B_R__balanced)')


def read_csv(path):
    with path.open() as f:
        return {(int(r['scene']), r['roi'], r['condition'], r['arm']): r for r in csv.DictReader(f)}


def main():
    check_host()
    start = time.monotonic()
    for sid in SCENES:
        verify_seal(ROOT/'inference'/f'scan{sid}')
    verify_seal(BASE/'evaluation')
    verify_seal(PREV/'evaluation')
    out = ROOT/'evaluation'
    out.mkdir(exist_ok=False)
    sources = {str(p): sha(p) for p in (ROOT/'evaluate_cross.py', ROOT/'common.py', ROOT/'PROTOCOL.md',
               BASE/'evaluate_replay.py', BASE/'evaluation/SEALED.json', PREV/'evaluation/SEALED.json',
               PREV/'evaluation/METRICS.csv', CLOSEOUT/'evaluation/METRICS.csv')}
    for sid in SCENES:
        seal_path = ROOT/'inference'/f'scan{sid}'/'SEALED.json'
        sources[str(seal_path)] = sha(seal_path)
    reference_root = DATA/'closeout-confirmation-v1'
    reference_manifest = json.loads((reference_root/'REFERENCE_MANIFEST.json').read_text())
    references = {}
    for item in reference_manifest['records']:
        path = Path(item['path'])
        if path.parent.name in {f'scan{s}' for s in SCENES}:
            if sha(path) != item['sha256']:
                raise AssertionError('replay reference changed')
            references[str(path)] = item['sha256']
    sources.update(references)
    save_json(out/'LOCK.json', dict(source_sha256=sources, reference_sha256=references,
              inference_sealed_before_reference_read=True, data_role='EXPOSED_REPLAY_NOT_NEW_CONFIRMATION',
              support='fixed native rows; exact BASE reference voxelization at 0.8 mm',
              primary='source_MSE_mm2', geometry='row preserving A, B, or identity',
              diagnostic_oracles_evaluation_only=True, expected_deployable_arms=sorted(EXPECTED_ARMS)))
    previous, historical = read_csv(PREV/'evaluation/METRICS.csv'), read_csv(CLOSEOUT/'evaluation/METRICS.csv')
    rows, complements, distance_checks, metric_checks, ply_checks = [], [], [], [], []
    for sid in SCENES:
        raw = CLOSEOUT/'confirmation'/f'scan{sid}'
        raw_manifest = json.loads((raw/'SEALED.json').read_text())['files']
        base = reference_root/'evaluation_only'/f'scan{sid}'
        laser = read_points(base/f'stl{sid:03d}_total.ply')
        obs = loadmat(base/f'ObsMask{sid}_10.mat')
        roi_path = raw/'ROIS.json'
        if sha(roi_path) != raw_manifest['ROIS.json']:
            raise AssertionError('ROI protocol changed')
        sources[str(roi_path)] = sha(roi_path)
        rois = json.loads(roi_path.read_text())
        if len(rois) != 4:
            raise AssertionError('four ROIs required per scene')
        for roi in rois:
            if roi['status'] != 'READY':
                raise AssertionError('frozen ROI unavailable')
            rid = roi['id']
            native_path = raw/(rid+'__native')/'identity.ply'
            native = read_points(native_path)
            support = in_box(native, roi['lo'], roi['hi']) & observed(native, obs)
            cached = np.load(CLOSEOUT/'evaluation'/(rid+'_native_support.npy'), allow_pickle=False)
            if not np.array_equal(support, cached) or not support.any():
                raise AssertionError('fixed native support changed')
            ref = voxel(laser[in_box(laser, roi['lo'], roi['hi']) & observed(laser, obs)])
            tree = cKDTree(ref)
            for condition in CONDS:
                name = rid+'__'+condition
                case, dest = raw/name, out/name
                dest.mkdir()
                inf = ROOT/'inference'/f'scan{sid}'/name
                for arm in ('identity', 'A_all', 'B_all'):
                    path = case/(arm+'.ply')
                    digest = sha(path)
                    if digest != raw_manifest[str(path.relative_to(raw))]:
                        raise AssertionError('frozen candidate changed')
                    sources[str(path)] = digest
                p, a, b = [read_points(case/(arm+'.ply')) for arm in ('identity', 'A_all', 'B_all')]
                if p.shape != a.shape or p.shape != b.shape or p.shape != native.shape:
                    raise AssertionError('candidate row correspondence changed')
                d0, dA, dB = [tree.query(points, workers=1)[0] for points in (p, a, b)]
                move2_A, move2_B = np.sum((a-p)**2, axis=1), np.sum((b-p)**2, axis=1)
                old_error_path = BASE/'evaluation'/name/'point_errors.npz'
                with np.load(old_error_path, allow_pickle=False) as z:
                    error = max(float(np.max(np.abs(d0-z['d0']))), float(np.max(np.abs(dA-z['d1']))))
                    if error > 1e-8 or not np.array_equal(z['support'], support):
                        raise AssertionError(('BASE distance/support reproduction failed', name, error))
                    if not np.array_equal(z['movement_squared'], move2_A):
                        raise AssertionError('BASE A displacement changed')
                sources[str(old_error_path)] = sha(old_error_path)
                distance_checks.append(dict(case=name, max_distance_difference_mm=error,
                                            support_equal=True, A_movement_equal=True))
                save_npz(dest/'point_errors.npz', d0=d0, dA=dA, dB=dB, support=support,
                         move2_A=move2_A, move2_B=move2_B)
                with np.load(inf/'decisions.npz', allow_pickle=False) as z:
                    masks = {k: z[k] for k in z.files}
                if set(masks) != EXPECTED_ARMS:
                    raise AssertionError(('decision arm contract', set(masks)^EXPECTED_ARMS))
                if masks['identity'].any() or not masks['A_all'].all() or not masks['B_all'].all():
                    raise AssertionError('raw/identity masks invalid')
                for mode in ('count', 'bin'):
                    for seed in range(10):
                        masks[f'random__B_R__balanced__{mode}__seed{seed}'] = matched_random(
                            masks['B_R__balanced'], support, np.sqrt(move2_B), mode, SEED+seed)
                masks['oracle_A'], masks['oracle_B'] = dA < d0, dB < d0
                choose_B = dB < dA
                dAB, move2_AB = np.where(choose_B, dB, dA), np.where(choose_B, move2_B, move2_A)
                masks['oracle_AB'] = dAB < d0
                route_AB = np.where(masks['oracle_AB'], np.where(choose_B, 2, 1), 0).astype(np.uint8)
                save_npz(dest/'diagnostic_decisions.npz', **{
                    k: v for k, v in masks.items() if k.startswith(('random__', 'oracle_'))}, oracle_AB_route=route_AB)
                incremental = np.maximum(np.minimum(d0**2, dA**2)-dB**2, 0.)
                complements.append(dict(scene=sid, roi=rid, condition=condition,
                    B_better_than_A_KEEP_fraction=float(np.mean((dB < np.minimum(d0,dA))[support])),
                    B_incremental_MSE_mm2=float(np.mean(incremental[support])),
                    B_R_captured_incremental_MSE_mm2=float(np.mean((incremental*masks['B_R__balanced'])[support])),
                    B_R_selected_complement_fraction=float(np.mean(((incremental > 0)&masks['B_R__balanced'])[support])),
                    identity_MSE_mm2=float(np.mean(d0[support]**2)),
                    oracle_A_MSE_mm2=float(np.mean(np.minimum(d0,dA)[support]**2)),
                    oracle_AB_MSE_mm2=float(np.mean(np.minimum(d0,dAB)[support]**2))))
                for arm, mask in masks.items():
                    if mask.dtype != bool or mask.shape != d0.shape:
                        raise AssertionError('invalid decision mask')
                    if arm == 'oracle_AB':
                        d1, move2 = dAB, move2_AB
                    elif arm.startswith(('B_', 'random__')) or arm == 'oracle_B':
                        d1, move2 = dB, move2_B
                    else:
                        d1, move2 = dA, move2_A
                    metric = measure(d0, d1, support, move2, mask)
                    rows.append(dict(scene=sid, roi=rid, condition=condition, arm=arm,
                                     n_rows=len(p), n_source=int(support.sum()), n_reference=len(ref), **metric))
                    old_name = None
                    if arm in ('identity','A_all','frozen_gain','previous_normalized'):
                        old_name = arm
                    elif arm.startswith('A_F__'):
                        old_name = arm.replace('A_F__','fit_aug__')
                    elif arm.startswith('A_R__'):
                        old_name = arm.replace('A_R__','reserved_aug__')
                    if old_name:
                        old = previous[(sid,rid,condition,old_name)]
                        error = max(abs(metric[k]-float(old[k])) for k in NUMERIC)
                        if error > 1e-8:
                            raise AssertionError(('previous A arm reproduction', name, arm, error))
                        metric_checks.append(dict(case=name, arm=arm, source='PREV', max_metric_difference=error))
                    if arm == 'B_all':
                        old = historical[(sid,rid,condition,'B_all')]
                        error = max(abs(metric[k]-float(old[k])) for k in NUMERIC if k in old and old[k] != '')
                        if error > 1e-8:
                            raise AssertionError(('historical B reproduction', name, error))
                        metric_checks.append(dict(case=name, arm=arm, source='CLOSEOUT', max_metric_difference=error))
                for arm in ('B_F__balanced', 'B_R__balanced', 'B_R__native_priority'):
                    exported = read_points(inf/(arm+'.ply'))
                    expected = np.where(masks[arm][:,None], b, p)
                    error = float(np.max(np.abs(exported-expected))) if exported.shape == expected.shape else float('inf')
                    if error != 0:
                        raise AssertionError('B exported PLY differs from decision geometry')
                    ply_checks.append(dict(case=name, arm=arm, max_coordinate_difference_mm=error))
                print('CROSS SCORED', name, len(masks), 'arms', flush=True)
        del laser
    if len(rows) != 2640 or len(ply_checks) != 180 or len(complements) != 60:
        raise AssertionError('evaluation completeness failed')
    with (out/'METRICS.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary.update(wall_seconds=time.monotonic()-start, aggregation='four equally weighted ROI means, then three equally weighted scenes',
                   random_repeats_are_not_independent_scenes=True, scope='fixed-row error, not completeness or physical layer identity',
                   primary_arm='B_R__balanced', not_new_confirmation=True)
    save_json(out/'SUMMARY.json', summary)
    save_json(out/'COMPLEMENT.json', summarize_complement(complements))
    save_json(out/'REPRODUCTION.json', dict(status='PASS', distance_checks=distance_checks,
              metric_checks=metric_checks, ply_checks=ply_checks,
              max_distance_difference_mm=max(r['max_distance_difference_mm'] for r in distance_checks),
              max_metric_difference=max(r['max_metric_difference'] for r in metric_checks)))
    if any(sha(Path(p)) != digest for p, digest in sources.items()):
        raise AssertionError('evaluation source modified during scoring')
    seal(out, sources)
    print('CROSS EVALUATION SEALED', len(rows), 'rows', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
