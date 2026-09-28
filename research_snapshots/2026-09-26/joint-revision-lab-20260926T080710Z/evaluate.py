"""Score sealed KEEP/A/B routes on exposed replay; no import by inference."""
from common import *
import csv
import time
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

PRIMARY = 'joint_support__balanced'
METHODS = ('independent_absolute', 'joint_common', 'joint_support',
           'normalized_max', 'support_margin')
POLICIES = ('balanced', 'native_priority', 'natural')
EXPECTED_ARMS = {f'{method}__{policy}' for method in METHODS for policy in POLICIES}
BASELINES = ('identity', 'A_R__balanced', 'B_R__balanced', 'B_R__native_priority',
             'oracle_A', 'oracle_B', 'oracle_AB')
CORE = ('source_MSE_mm2', 'source_MAE_mm', 'source_p95_mm',
        'improved_fraction', 'harmed_fraction', 'accepted_fraction',
        'accepted_support_fraction', 'moved_fraction', 'move_RMS_mm',
        'benefit_sum_mm2', 'harm_sum_mm2')
EXTRA = ('KEEP_fraction', 'A_fraction', 'B_fraction', 'KEEP_support_fraction',
         'A_support_fraction', 'B_support_fraction', 'B_incremental_MSE_mm2',
         'B_captured_incremental_MSE_mm2', 'B_selected_complement_fraction',
         'B_net_incremental_MSE_mm2', 'B_incremental_harm_MSE_mm2',
         'B_selected_oracle_regret_MSE_mm2',
         'joint_oracle_MSE_mm2', 'joint_oracle_regret_MSE_mm2')
NUMERIC = CORE + EXTRA


def valid_route(route, n):
    if route.dtype != np.uint8 or route.shape != (n,) or np.any(route > 2):
        raise AssertionError('route must be uint8[N] with 0=KEEP, 1=A, 2=B')


def route_count_random(route, support, seed):
    """Match each of KEEP/A/B counts within fixed-support/not-support groups."""
    valid_route(route, len(support))
    rng = np.random.default_rng(seed)
    result = np.empty_like(route)
    for supported in (False, True):
        ids = np.flatnonzero(support == supported)
        result[ids] = rng.permutation(route[ids])
        if not np.array_equal(np.bincount(result[ids], minlength=3),
                              np.bincount(route[ids], minlength=3)):
            raise AssertionError('route-count control mismatch')
    return result


def measure(errors, route):
    d0, dA, dB = (errors[k] for k in ('d0', 'dA', 'dB'))
    support = errors['support']
    valid_route(route, len(d0))
    after = np.choose(route, (d0, dA, dB))
    move2 = np.choose(route, (np.zeros_like(d0), errors['move2_A'], errors['move2_B']))
    delta = after[support] - d0[support]
    gain = d0[support]**2 - after[support]**2
    signed_incremental = np.minimum(d0**2, dA**2) - dB**2
    incremental = np.maximum(signed_incremental, 0.)
    oracle2 = np.minimum.reduce((d0, dA, dB))[support]**2
    result = dict(source_MSE_mm2=float(np.mean(after[support]**2)),
                  source_MAE_mm=float(np.mean(after[support])),
                  source_p95_mm=float(np.quantile(after[support], .95)),
                  improved_fraction=float(np.mean(delta < -.1)),
                  harmed_fraction=float(np.mean(delta > .1)),
                  accepted_fraction=float(np.mean(route != 0)),
                  accepted_support_fraction=float(np.mean(route[support] != 0)),
                  moved_fraction=float(np.mean(move2 > 1e-14)),
                  move_RMS_mm=float(np.sqrt(move2.mean())),
                  benefit_sum_mm2=float(np.maximum(gain, 0).sum()),
                  harm_sum_mm2=float(np.maximum(-gain, 0).sum()))
    for i, name in enumerate(('KEEP', 'A', 'B')):
        result[name+'_fraction'] = float(np.mean(route == i))
        result[name+'_support_fraction'] = float(np.mean(route[support] == i))
    result.update(B_incremental_MSE_mm2=float(np.mean(incremental[support])),
        B_captured_incremental_MSE_mm2=float(np.mean((incremental*(route == 2))[support])),
        B_selected_complement_fraction=float(np.mean(((incremental > 0) & (route == 2))[support])),
        B_net_incremental_MSE_mm2=float(np.mean((signed_incremental*(route == 2))[support])),
        B_incremental_harm_MSE_mm2=float(np.mean((np.maximum(-signed_incremental,0.)*(route == 2))[support])),
        B_selected_oracle_regret_MSE_mm2=float(np.mean((after[support]**2-oracle2)*(route[support] == 2))),
        joint_oracle_MSE_mm2=float(oracle2.mean()),
        joint_oracle_regret_MSE_mm2=float(np.mean(after[support]**2 - oracle2)))
    add_ratios(result)
    return result


def add_ratios(cell):
    total = cell['B_incremental_MSE_mm2']
    cell['B_incremental_capture_fraction'] = cell['B_captured_incremental_MSE_mm2']/total if total > 0 else None


def read_csv(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    index = {(int(r['scene']), r['roi'], r['condition'], r['arm']): r for r in rows}
    if len(index) != len(rows):
        raise AssertionError('duplicate archive metric row')
    return index


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
                    raise AssertionError('four ROI rows required for every arm/scene/condition')
                cell = {k: float(np.mean([r[k] for r in ss])) for k in NUMERIC}
                baseline = np.mean([lookup[(sid, r['roi'], condition, 'identity')]['source_MSE_mm2'] for r in ss])
                cell['MSE_gain_percent'] = float(100*(1-cell['source_MSE_mm2']/baseline))
                add_ratios(cell)
                per_scene[str(sid)][condition][arm] = cell
            cell = {k: float(np.mean([per_scene[str(s)][condition][arm][k] for s in SCENES])) for k in NUMERIC}
            before = np.array([lookup[(r['scene'], r['roi'], condition, 'identity')]['source_MSE_mm2'] for r in rr])
            after = np.array([r['source_MSE_mm2'] for r in rr])
            base_mse = float(np.mean([per_scene[str(s)][condition]['identity']['source_MSE_mm2'] for s in SCENES])) if 'identity' in per_scene[str(SCENES[0])][condition] else float(before.mean())
            cell.update(MSE_gain_percent=float(100*(1-cell['source_MSE_mm2']/base_mse)),
                        wins=int(np.sum(after < before-1e-12)),
                        ties=int(np.sum(np.abs(after-before) <= 1e-12)),
                        losses=int(np.sum(after > before+1e-12)), n_rois=len(rr))
            add_ratios(cell)
            pooled[condition][arm] = cell
    comparisons, random_summary = {}, {}
    comparators = ('independent_absolute__balanced', 'joint_common__balanced',
                   'normalized_max__balanced', 'support_margin__balanced',
                   'A_R__balanced', 'B_R__balanced', 'B_R__native_priority', 'historical_post_AB')
    for condition in CONDS:
        p = pooled[condition][PRIMARY]
        comparisons[condition] = {}
        for arm in comparators:
            if arm not in methods:
                continue
            other = pooled[condition][arm]
            delta = [r['source_MSE_mm2'] - lookup[(r['scene'], r['roi'], condition, arm)]['source_MSE_mm2']
                     for r in rows if r['condition'] == condition and r['arm'] == PRIMARY]
            comparisons[condition][arm] = dict(
                primary_minus_comparator_MSE_mm2=p['source_MSE_mm2']-other['source_MSE_mm2'],
                primary_minus_comparator_gain_pp=p['MSE_gain_percent']-other['MSE_gain_percent'],
                primary_wins=int(np.sum(np.asarray(delta) < -1e-12)),
                ties=int(np.sum(np.abs(delta) <= 1e-12)),
                primary_losses=int(np.sum(np.asarray(delta) > 1e-12)))
        random = [pooled[condition][f'random_route_count__{PRIMARY}__seed{i}'] for i in range(10)]
        random_summary[condition] = {k: dict(mean=float(np.mean([r[k] for r in random])),
            min=float(np.min([r[k] for r in random])), max=float(np.max([r[k] for r in random])),
            sd=float(np.std([r[k] for r in random]))) for k in ('source_MSE_mm2', 'MSE_gain_percent', 'harmed_fraction')}
    return dict(per_scene=per_scene, exposed_replay=pooled, primary_comparisons=comparisons,
                random_route_count_controls=random_summary, rows=len(rows), methods=methods)


def archived_routes(case, errors, cross_inference, sources, raw_manifest):
    """Read archived decisions, without replacing or relabelling old outcomes."""
    name = case.name
    path = cross_inference/name/'decisions.npz'
    verify_file(path, cross_inference, sources)
    with np.load(path, allow_pickle=False) as z:
        routes = {'identity': np.zeros(len(errors['d0']), dtype=np.uint8)}
        for arm, code in (('A_R__balanced', 1), ('B_R__balanced', 2), ('B_R__native_priority', 2)):
            mask = z[arm]
            if mask.dtype != bool or mask.shape != errors['d0'].shape:
                raise AssertionError('invalid archived binary decision')
            routes[arm] = mask.astype(np.uint8)*code
    d0, dA, dB = (errors[k] for k in ('d0','dA','dB'))
    routes['oracle_A'] = (dA < d0).astype(np.uint8)
    routes['oracle_B'] = (dB < d0).astype(np.uint8)*2
    # Archive ties choose A rather than B, and KEEP on ties with identity.
    choose_b = dB < dA
    routes['oracle_AB'] = np.where(np.minimum(dA,dB) < d0, np.where(choose_b,2,1),0).astype(np.uint8)
    path = case/'DECISIONS.npz'
    verify_file(path, case.parent, sources, raw_manifest)
    with np.load(path, allow_pickle=False) as z:
        old = z['post_AB_keep']
        if old.shape != d0.shape or np.any((old < 0) | (old > 2)):
            raise AssertionError('invalid historical joint decision')
        routes['historical_post_AB'] = old.astype(np.uint8)
    return routes


def verify_file(path, folder, sources, manifest=None):
    if manifest is None:
        manifest = json.loads((folder/'SEALED.json').read_text())
    digest = sha(path)
    if digest != manifest['files'][str(path.relative_to(folder))]:
        raise AssertionError('sealed file changed: '+str(path))
    sources[str(path)] = digest


def independent_distances(points, reference):
    """Independent Open3D NN backend for a small fixed row sample."""
    import open3d as o3d
    src, dst = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(points)
    dst.points = o3d.utility.Vector3dVector(reference)
    return np.asarray(src.compute_point_cloud_distance(dst))


def main():
    check_host()
    start = time.monotonic()
    # This precondition runs before opening any error cache, metric CSV or reference.
    for sid in SCENES:
        verify_seal(ROOT/'inference'/f'scan{sid}')
    protocol = (ROOT/'PROTOCOL.md').read_text()
    if not protocol.strip():
        raise AssertionError('protocol missing')
    cross_seal = verify_seal(CROSS/'evaluation')
    verify_seal(CLOSEOUT/'evaluation')
    out = ROOT/'evaluation'
    out.mkdir(exist_ok=False)
    inputs = [ROOT/'evaluate.py', ROOT/'test_evaluation.py', ROOT/'common.py', ROOT/'PROTOCOL.md',
              CROSS/'evaluation/SEALED.json', CLOSEOUT/'evaluation/SEALED.json',
              CROSS/'evaluation/METRICS.csv', CLOSEOUT/'evaluation/METRICS.csv',
              BASE/'common.py', DATA/'closeout-confirmation-v1/REFERENCE_MANIFEST.json']
    inputs += [ROOT/'inference'/f'scan{s}'/'SEALED.json' for s in SCENES]
    inputs += [CROSS/'inference'/f'scan{s}'/'SEALED.json' for s in SCENES]
    inputs += [CLOSEOUT/'confirmation'/f'scan{s}'/'SEALED.json' for s in SCENES]
    sources = {str(p): sha(p) for p in inputs}
    save_json(out/'LOCK.json', dict(source_sha256=sources.copy(),
        inference_sealed_before_evaluation_read=True,
        data_role='EXPOSED_REPLAY_NOT_NEW_CONFIRMATION', primary_arm=PRIMARY,
        primary_metric='source_MSE_mm2', expected_route_arms=sorted(EXPECTED_ARMS),
        archived_baselines=BASELINES, historical_joint_baseline='historical_post_AB',
        support='fixed native point rows, shared across conditions and methods',
        cached_errors='sealed CROSS d0/dA/dB/support/move2_A/move2_B, never copied',
        independent_sample='128 equally spaced row indices per case, Open3D NN backend',
        random_controls='10 seeds; match KEEP/A/B counts within support and outside support; not movement matched',
        reference_voxel_mm=.8, diagnostic_oracles_evaluation_only=True))
    archive, historical = read_csv(CROSS/'evaluation/METRICS.csv'), read_csv(CLOSEOUT/'evaluation/METRICS.csv')
    reference_root = DATA/'closeout-confirmation-v1'
    manifest = json.loads((reference_root/'REFERENCE_MANIFEST.json').read_text())
    for record in manifest['records']:
        p = Path(record['path'])
        if p.parent.name in {f'scan{s}' for s in SCENES}:
            if sha(p) != record['sha256']:
                raise AssertionError('reference manifest changed')
            sources[str(p)] = record['sha256']
    rows, metric_checks, ply_checks, sample_checks, support_checks, controls = [], [], [], [], [], []
    for sid in SCENES:
        raw = CLOSEOUT/'confirmation'/f'scan{sid}'
        raw_manifest = json.loads((raw/'SEALED.json').read_text())
        verify_file(raw/'ROIS.json', raw, sources, raw_manifest)
        rois = json.loads((raw/'ROIS.json').read_text())
        if len(rois) != 4 or any(r['status'] != 'READY' for r in rois):
            raise AssertionError('four ready ROIs required')
        base = reference_root/'evaluation_only'/f'scan{sid}'
        laser = read_points(base/f'stl{sid:03d}_total.ply')
        obs = loadmat(base/f'ObsMask{sid}_10.mat')
        for roi in rois:
            rid = roi['id']
            native_path = raw/(rid+'__native')/'identity.ply'
            verify_file(native_path, raw, sources, raw_manifest)
            native = read_points(native_path)
            support = io.in_box(native, roi['lo'], roi['hi']) & io.observed(native, obs)
            support_path = CLOSEOUT/'evaluation'/(rid+'_native_support.npy')
            verify_file(support_path, CLOSEOUT/'evaluation', sources)
            if not np.array_equal(support, np.load(support_path, allow_pickle=False)) or not support.any():
                raise AssertionError('historical support mismatch')
            reference = io.voxel(laser[io.in_box(laser, roi['lo'], roi['hi']) & io.observed(laser, obs)])
            support_checks.append(dict(scene=sid, roi=rid, n_source=int(support.sum()),
                                       historical_support_exact=True, n_reference=len(reference)))
            for condition in CONDS:
                case = raw/(rid+'__'+condition)
                error_path = CROSS/'evaluation'/case.name/'point_errors.npz'
                verify_file(error_path, CROSS/'evaluation', sources, cross_seal)
                with np.load(error_path, allow_pickle=False) as z:
                    errors = {k: z[k] for k in ('d0','dA','dB','support','move2_A','move2_B')}
                if not np.array_equal(errors['support'], support):
                    raise AssertionError('cached support mismatch')
                points = []
                for arm in ('identity','A_all','B_all'):
                    path = case/(arm+'.ply')
                    verify_file(path, raw, sources, raw_manifest)
                    points.append(read_points(path))
                if any(p.shape != native.shape for p in points):
                    raise AssertionError('candidate row correspondence mismatch')
                for code in ('A','B'):
                    move2 = np.sum((points[1 if code == 'A' else 2]-points[0])**2, axis=1)
                    if not np.array_equal(move2, errors['move2_'+code]):
                        raise AssertionError('cached candidate movement mismatch')
                ids = np.unique(np.linspace(0, len(native)-1, min(128,len(native)), dtype=int))
                max_error = max(float(np.max(np.abs(independent_distances(p[ids],reference)-errors[k][ids])))
                                for p,k in zip(points,('d0','dA','dB')))
                if max_error > 1e-8:
                    raise AssertionError('independent reference sample mismatch')
                sample_checks.append(dict(case=case.name, rows=ids.tolist(),
                    candidate_distances_checked=3*len(ids), max_distance_difference_mm=max_error))
                inf = ROOT/'inference'/f'scan{sid}'/case.name
                verify_file(inf/'routes.npz', inf.parent, sources)
                with np.load(inf/'routes.npz', allow_pickle=False) as z:
                    routes = {k:z[k] for k in z.files}
                if set(routes) != EXPECTED_ARMS:
                    raise AssertionError(('route arm mismatch', set(routes)^EXPECTED_ARMS))
                for route in routes.values():
                    valid_route(route, len(native))
                routes.update(archived_routes(case, errors, CROSS/'inference'/f'scan{sid}', sources, raw_manifest))
                control_record = dict(case=case.name, seeds=[], primary_counts={})
                for group in (False,True):
                    control_record['primary_counts']['support' if group else 'outside_support'] = np.bincount(routes[PRIMARY][support==group],minlength=3).tolist()
                for seed in range(10):
                    arm = f'random_route_count__{PRIMARY}__seed{seed}'
                    routes[arm] = route_count_random(routes[PRIMARY], support, SEED+seed)
                    control_record['seeds'].append(dict(seed=SEED+seed, group_counts_exact=True))
                controls.append(control_record)
                for arm, route in routes.items():
                    metric = measure(errors, route)
                    archived = historical[(sid,rid,condition,'post_AB_keep')] if arm == 'historical_post_AB' else archive.get((sid,rid,condition,arm))
                    if arm in BASELINES or arm == 'historical_post_AB':
                        shared = [k for k in CORE if k in archived and archived[k] != '']
                        error = max(abs(metric[k]-float(archived[k])) for k in shared)
                        if error > 1e-8 or any(int(archived[k]) != n for k,n in
                            (('n_rows',len(native)),('n_source',int(support.sum())),('n_reference',len(reference)))):
                            raise AssertionError(('archived metric incompatibility',case.name,arm,error))
                        # Original archive values remain exact; only new diagnostics are appended.
                        metric.update({k:float(archived[k]) for k in shared})
                        metric_checks.append(dict(case=case.name, arm=arm, max_metric_difference=error,
                                                  source='CLOSEOUT' if arm == 'historical_post_AB' else 'CROSS'))
                    family = ('diagnostic_oracle' if arm.startswith('oracle_') else
                              'route_count_random_control' if arm.startswith('random_route_count__') else
                              'historical_joint_baseline' if arm == 'historical_post_AB' else
                              'archived_baseline' if arm in BASELINES else 'new_observation_only_method')
                    rows.append(dict(scene=sid,roi=rid,condition=condition,arm=arm,family=family,
                                     n_rows=len(native),n_source=int(support.sum()),n_reference=len(reference),**metric))
                for arm in (PRIMARY,'joint_common__balanced','historical_post_AB'):
                    path = case/'post_AB_keep.ply' if arm == 'historical_post_AB' else inf/(arm+'.ply')
                    verify_file(path, raw if arm == 'historical_post_AB' else inf.parent, sources)
                    exported = read_points(path)
                    expected = np.choose(routes[arm][:,None], points)
                    error = float(np.max(np.abs(exported-expected))) if exported.shape == expected.shape else float('inf')
                    if error != 0:
                        raise AssertionError('exported PLY differs from route geometry: '+arm)
                    ply_checks.append(dict(case=case.name,arm=arm,max_coordinate_difference_mm=error))
                print('JOINT SCORED',case.name,len(routes),'arms',flush=True)
        del laser
    if len(rows) != 1980 or len(ply_checks) != 180 or len(sample_checks) != 60 or len(metric_checks) != 480:
        raise AssertionError('evaluation completeness mismatch')
    with (out/'METRICS.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary = summarize(rows)
    summary.update(wall_seconds=time.monotonic()-start,
        aggregation='four equally weighted ROI means, then three equally weighted scenes; condition cells kept separate',
        primary_arm=PRIMARY, not_new_confirmation=True, data_role='EXPOSED_REPLAY_NOT_NEW_CONFIRMATION',
        scope='fixed-row source error; no completeness, coverage or physical layer identity claim',
        random_repeats_are_not_independent_scenes=True,
        B_incremental_definition='max(min(d0^2,dA^2)-dB^2,0); capture is gross benefit on selected B rows, not net deployed gain',
        B_net_incremental_definition='mean((min(d0^2,dA^2)-dB^2)*(route==B)); signed comparison with evaluator-only best of KEEP/A, not a causal B ablation',
        B_selected_regret_definition='mean((dB^2-min(d0,dA,dB)^2)*(route==B)); selected-B portion of joint oracle regret',
        ratio_aggregation='B capture ratio computed after ROI-then-scene aggregation of numerator and denominator',
        p95_aggregation='mean of per-ROI 95th percentiles, not percentile of pooled points',
        oracle_regret_definition='mean(selected_distance^2-min(d0,dA,dB)^2) on fixed support',
        harm_definition='fraction of fixed-support rows with distance increase greater than 0.1 mm')
    save_json(out/'SUMMARY.json',summary)
    save_json(out/'REPRODUCTION.json',dict(status='PASS',sample_checks=sample_checks,
        support_checks=support_checks,metric_checks=metric_checks,ply_checks=ply_checks,
        new_exported_ply_checks=120,historical_exported_ply_checks=60,
        max_distance_difference_mm=max(r['max_distance_difference_mm'] for r in sample_checks),
        max_metric_difference=max(r['max_metric_difference'] for r in metric_checks),
        point_error_cache_copied=False))
    save_json(out/'RANDOM_CONTROL_AUDIT.json',dict(kind='route_count_matched_within_fixed_support_groups',
        movement_matched=False,random_repeats=10,seed_base=SEED,cases=controls))
    if any(sha(Path(p)) != digest for p,digest in sources.items()):
        raise AssertionError('evaluation source changed during scoring')
    seal(out,sources)
    print('JOINT EVALUATION SEALED',len(rows),'rows',flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
