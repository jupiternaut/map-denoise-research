"""Evaluate frozen visibility routes; reference access begins only after inference seals."""
from common import *
import csv
import time
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

frozen = module('frozen_joint_metrics', JOINT/'evaluate.py')
measure, valid_route = frozen.measure, frozen.valid_route
route_count_random = frozen.route_count_random
verify_file, read_csv = frozen.verify_file, frozen.read_csv
CORE, NUMERIC = frozen.CORE, frozen.NUMERIC
BASELINES = ('identity', 'A_R__balanced', 'B_R__balanced', 'B_R__native_priority',
             'oracle_A', 'oracle_B', 'oracle_AB')
PRIOR_ARMS = ('independent_absolute__balanced', 'joint_support__balanced',
              'joint_support__native_priority', 'joint_common__balanced',
              'normalized_max__balanced')


def add_ratios(cell, identity_mse):
    frozen.add_ratios(cell)
    available = identity_mse-cell['joint_oracle_MSE_mm2']
    net = identity_mse-cell['source_MSE_mm2']
    cell.update(net_MSE_reduction_mm2=float(net),
                attainable_MSE_reduction_mm2=float(available),
                oracle_headroom_capture_fraction=float(net/available) if available > 0 else None,
                MSE_gain_percent=float(100*net/identity_mse) if identity_mse > 0 else None)


def summarize(rows, primary, new_arms):
    methods = sorted({r['arm'] for r in rows})
    lookup = {(r['scene'], r['roi'], r['condition'], r['arm']):r for r in rows}
    if len(lookup) != len(rows):
        raise AssertionError('duplicate metric row')
    per_scene = {str(s):{c:{} for c in CONDS} for s in SCENES}
    pooled = {c:{} for c in CONDS}
    comparisons, random_summary, maximum = {}, {}, {}
    for condition in CONDS:
        for arm in methods:
            rr = [r for r in rows if r['condition'] == condition and r['arm'] == arm]
            for sid in SCENES:
                ss = [r for r in rr if r['scene'] == sid]
                if len(ss) != 4 or len({r['roi'] for r in ss}) != 4:
                    raise AssertionError('four ROI rows required per arm/scene/condition')
                cell = {k:float(np.mean([r[k] for r in ss])) for k in NUMERIC}
                baseline = np.mean([lookup[(sid,r['roi'],condition,'identity')]['source_MSE_mm2'] for r in ss])
                add_ratios(cell, float(baseline))
                per_scene[str(sid)][condition][arm] = cell
            cell = {k:float(np.mean([per_scene[str(s)][condition][arm][k] for s in SCENES])) for k in NUMERIC}
            before = np.array([lookup[(r['scene'],r['roi'],condition,'identity')]['source_MSE_mm2'] for r in rr])
            after = np.array([r['source_MSE_mm2'] for r in rr])
            add_ratios(cell, float(before.mean()))
            cell.update(wins=int(np.sum(after < before-1e-12)),
                        ties=int(np.sum(np.abs(after-before) <= 1e-12)),
                        losses=int(np.sum(after > before+1e-12)), n_rois=len(rr))
            pooled[condition][arm] = cell
        p = pooled[condition][primary]
        comparisons[condition] = {}
        for arm in methods:
            if arm == primary or arm.startswith('random_route_count__') or arm.startswith('oracle_'):
                continue
            other = pooled[condition][arm]
            delta = np.array([r['source_MSE_mm2']-lookup[(r['scene'],r['roi'],condition,arm)]['source_MSE_mm2']
                              for r in rows if r['condition'] == condition and r['arm'] == primary])
            comparisons[condition][arm] = dict(
                primary_minus_comparator_MSE_mm2=p['source_MSE_mm2']-other['source_MSE_mm2'],
                primary_minus_comparator_gain_pp=p['MSE_gain_percent']-other['MSE_gain_percent'],
                primary_wins=int(np.sum(delta < -1e-12)), ties=int(np.sum(np.abs(delta) <= 1e-12)),
                primary_losses=int(np.sum(delta > 1e-12)))
        random = [pooled[condition][f'random_route_count__{primary}__seed{i}'] for i in range(10)]
        random_summary[condition] = {k:dict(mean=float(np.mean([r[k] for r in random])),
            min=float(np.min([r[k] for r in random])), max=float(np.max([r[k] for r in random])),
            sd=float(np.std([r[k] for r in random]))) for k in ('source_MSE_mm2','MSE_gain_percent','harmed_fraction')}
        best = min(new_arms,key=lambda arm:pooled[condition][arm]['source_MSE_mm2'])
        maximum[condition] = dict(arm=best, **pooled[condition][best],
            interpretation='descriptive best among locked arms on this condition; hindsight selection, not a deployable composite method')
    return dict(per_scene=per_scene, exposed_replay=pooled, primary_comparisons=comparisons,
                random_route_count_controls=random_summary, descriptive_best_locked_arm=maximum,
                rows=len(rows), methods=methods, new_arms=sorted(new_arms))


def prior_routes(case, source_manifest, sources):
    sid = int(case.name.split('_')[0][4:])
    folder = JOINT/'inference'/f'scan{sid}'
    path = folder/case.name/'routes.npz'
    verify_file(path, folder, sources, source_manifest)
    with np.load(path,allow_pickle=False) as z:
        return {'prior__'+arm:z[arm] for arm in PRIOR_ARMS}


def main():
    check_host()
    start = time.monotonic()
    # No evaluation cache, CSV, laser or support is opened before all output seals.
    for sid in SCENES:
        verify_seal(OUT/'inference'/f'scan{sid}')
    verify_seal(OUT/'training')
    lock = json.loads((OUT/'training/MODEL_LOCK.json').read_text())
    primary = lock['primary_arm']
    recovery = lock['recovery_arm']
    exported_arms = sorted({primary,recovery})
    new_arms = set(lock['expected_route_arms'])
    if not set(exported_arms).issubset(new_arms) or not new_arms:
        raise AssertionError('primary/recovery not in frozen route arms')
    cross_seal = verify_seal(CROSS/'evaluation')
    verify_seal(CLOSEOUT/'evaluation')
    verify_seal(JOINT/'evaluation')
    prior_seals = {sid:verify_seal(JOINT/'inference'/f'scan{sid}') for sid in SCENES}
    out = OUT/'evaluation'
    out.mkdir(exist_ok=False)
    inputs = [ROOT/'evaluate.py',ROOT/'test_evaluation.py',ROOT/'common.py',ROOT/'PROTOCOL.md',
        OUT/'training/MODEL_LOCK.json',OUT/'training/SEALED.json',JOINT/'evaluate.py',
        CROSS/'evaluation/SEALED.json',CLOSEOUT/'evaluation/SEALED.json',JOINT/'evaluation/SEALED.json',
        CROSS/'evaluation/METRICS.csv',CLOSEOUT/'evaluation/METRICS.csv',JOINT/'evaluation/METRICS.csv',
        BASE/'common.py',DATA/'closeout-confirmation-v1/REFERENCE_MANIFEST.json']
    inputs += [OUT/'inference'/f'scan{s}'/'SEALED.json' for s in SCENES]
    inputs += [JOINT/'inference'/f'scan{s}'/'SEALED.json' for s in SCENES]
    inputs += [CROSS/'inference'/f'scan{s}'/'SEALED.json' for s in SCENES]
    inputs += [CLOSEOUT/'confirmation'/f'scan{s}'/'SEALED.json' for s in SCENES]
    sources = {str(p):sha(p) for p in inputs}
    save_json(out/'LOCK.json',dict(source_sha256=sources.copy(),
        inference_sealed_before_evaluation_read=True,primary_arm=primary,recovery_arm=recovery,
        data_role='EXPOSED_REPLAY_NOT_NEW_CONFIRMATION',expected_route_arms=sorted(new_arms),
        primary_metric='source_MSE_mm2',support='fixed native point rows, shared by all arms and conditions',
        cached_errors='sealed CROSS d0/dA/dB/support/move2_A/move2_B; never copied',
        random_controls='10 seeds matching KEEP/A/B counts separately within/outside fixed support, not actual edit counts',
        independent_sample='128 equally spaced row indices per case; Open3D NN',
        reference_voxel_mm=.8, diagnostic_oracles_evaluation_only=True,
        hindsight_maximum='conditionwise best among locked arms is descriptive, never relabelled as primary'))
    archive = read_csv(CROSS/'evaluation/METRICS.csv')
    historical = read_csv(CLOSEOUT/'evaluation/METRICS.csv')
    old_joint = read_csv(JOINT/'evaluation/METRICS.csv')
    reference_root = DATA/'closeout-confirmation-v1'
    manifest = json.loads((reference_root/'REFERENCE_MANIFEST.json').read_text())
    for record in manifest['records']:
        p = Path(record['path'])
        if p.parent.name in {f'scan{s}' for s in SCENES}:
            if sha(p) != record['sha256']:
                raise AssertionError('reference manifest changed')
            sources[str(p)] = record['sha256']
    rows, metric_checks, ply_checks, sample_checks, support_checks, controls = [],[],[],[],[],[]
    for sid in SCENES:
        raw = CLOSEOUT/'confirmation'/f'scan{sid}'
        raw_manifest = json.loads((raw/'SEALED.json').read_text())
        verify_file(raw/'ROIS.json',raw,sources,raw_manifest)
        rois = json.loads((raw/'ROIS.json').read_text())
        if len(rois) != 4 or any(r['status'] != 'READY' for r in rois):
            raise AssertionError('four ready ROI records required')
        base = reference_root/'evaluation_only'/f'scan{sid}'
        laser = read_points(base/f'stl{sid:03d}_total.ply')
        obs = loadmat(base/f'ObsMask{sid}_10.mat')
        for roi in rois:
            rid = roi['id']
            native_path = raw/(rid+'__native')/'identity.ply'
            verify_file(native_path,raw,sources,raw_manifest)
            native = read_points(native_path)
            support = io.in_box(native,roi['lo'],roi['hi']) & io.observed(native,obs)
            support_path = CLOSEOUT/'evaluation'/(rid+'_native_support.npy')
            verify_file(support_path,CLOSEOUT/'evaluation',sources)
            if not np.array_equal(support,np.load(support_path,allow_pickle=False)) or not support.any():
                raise AssertionError('support differs from archived support')
            reference = io.voxel(laser[io.in_box(laser,roi['lo'],roi['hi']) & io.observed(laser,obs)])
            support_checks.append(dict(scene=sid,roi=rid,n_source=int(support.sum()),n_reference=len(reference),historical_support_exact=True))
            for condition in CONDS:
                case = raw/(rid+'__'+condition)
                error_path = CROSS/'evaluation'/case.name/'point_errors.npz'
                verify_file(error_path,CROSS/'evaluation',sources,cross_seal)
                with np.load(error_path,allow_pickle=False) as z:
                    errors = {k:z[k] for k in ('d0','dA','dB','support','move2_A','move2_B')}
                if not np.array_equal(errors['support'],support):
                    raise AssertionError('cached support mismatch')
                points = []
                for arm in ('identity','A_all','B_all'):
                    path = case/(arm+'.ply')
                    verify_file(path,raw,sources,raw_manifest)
                    points.append(read_points(path))
                if any(p.shape != native.shape for p in points):
                    raise AssertionError('candidate point row mismatch')
                for idx,code in ((1,'A'),(2,'B')):
                    if not np.array_equal(np.sum((points[idx]-points[0])**2,axis=1),errors['move2_'+code]):
                        raise AssertionError('cached candidate movement mismatch')
                ids = np.unique(np.linspace(0,len(native)-1,min(128,len(native)),dtype=int))
                err = max(float(np.max(np.abs(frozen.independent_distances(p[ids],reference)-errors[k][ids])))
                          for p,k in zip(points,('d0','dA','dB')))
                if err > 1e-8:
                    raise AssertionError('independent reference sample mismatch')
                sample_checks.append(dict(case=case.name,rows=ids.tolist(),candidate_distances_checked=3*len(ids),max_distance_difference_mm=err))
                inf = OUT/'inference'/f'scan{sid}'/case.name
                verify_file(inf/'routes.npz',inf.parent,sources)
                with np.load(inf/'routes.npz',allow_pickle=False) as z:
                    routes = {k:z[k] for k in z.files}
                if set(routes) != new_arms:
                    raise AssertionError('new route arms differ from model lock')
                routes.update(frozen.archived_routes(case,errors,CROSS/'inference'/f'scan{sid}',sources,raw_manifest))
                routes.update(prior_routes(case,prior_seals[sid],sources))
                for route in routes.values():
                    valid_route(route,len(native))
                control = dict(case=case.name,seeds=[],primary_counts={})
                for group in (False,True):
                    control['primary_counts']['support' if group else 'outside_support'] = np.bincount(routes[primary][support==group],minlength=3).tolist()
                for seed in range(10):
                    routes[f'random_route_count__{primary}__seed{seed}'] = route_count_random(routes[primary],support,SEED+seed)
                    control['seeds'].append(dict(seed=SEED+seed,group_counts_exact=True))
                controls.append(control)
                for arm,route in routes.items():
                    metric = measure(errors,route)
                    archived = None
                    if arm == 'historical_post_AB':
                        archived,origin = historical[(sid,rid,condition,'post_AB_keep')],'CLOSEOUT'
                    elif arm in BASELINES:
                        archived,origin = archive[(sid,rid,condition,arm)],'CROSS'
                    elif arm.startswith('prior__'):
                        archived,origin = old_joint[(sid,rid,condition,arm[len('prior__'):])],'JOINT'
                    if archived is not None:
                        keys = [k for k in NUMERIC if k in archived and archived[k] != '']
                        err = max(abs(metric[k]-float(archived[k])) for k in keys)
                        if err > 1e-8 or any(int(archived[k]) != n for k,n in
                            (('n_rows',len(native)),('n_source',int(support.sum())),('n_reference',len(reference)))):
                            raise AssertionError(('archive metric mismatch',case.name,arm,err))
                        metric.update({k:float(archived[k]) for k in keys})
                        metric_checks.append(dict(case=case.name,arm=arm,source=origin,max_metric_difference=err))
                    add_ratios(metric, float(np.mean(errors['d0'][support]**2)))
                    family = ('diagnostic_oracle' if arm.startswith('oracle_') else
                        'route_count_random_control' if arm.startswith('random_route_count__') else
                        'archived_baseline' if archived is not None else 'new_observation_only_method')
                    rows.append(dict(scene=sid,roi=rid,condition=condition,arm=arm,family=family,
                        n_rows=len(native),n_source=int(support.sum()),n_reference=len(reference),**metric))
                for arm in exported_arms:
                    path = inf/(arm+'.ply')
                    verify_file(path,inf.parent,sources)
                    exported = read_points(path)
                    expected = np.choose(routes[arm][:,None],points)
                    err = float(np.max(np.abs(exported-expected))) if exported.shape == expected.shape else float('inf')
                    if err != 0:
                        raise AssertionError('exported PLY differs from selected geometry')
                    ply_checks.append(dict(case=case.name,arm=arm,max_coordinate_difference_mm=err))
                print('VISIBILITY SCORED',case.name,len(routes),'arms',flush=True)
        del laser
    expected_methods = len(new_arms)+len(BASELINES)+1+len(PRIOR_ARMS)+10
    if (len(rows),len(ply_checks),len(sample_checks),len(metric_checks)) != (60*expected_methods,60*len(exported_arms),60,60*(len(BASELINES)+1+len(PRIOR_ARMS))):
        raise AssertionError('evaluation completeness mismatch')
    with (out/'METRICS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=summarize(rows,primary,new_arms)
    summary.update(wall_seconds=time.monotonic()-start,primary_arm=primary,recovery_arm=recovery,
        locked_recovery_result={c:summary['exposed_replay'][c][recovery] for c in CONDS},
        recovery_selection='selected by development OOF before replay; distinct from descriptive best locked arm',
        aggregation='four equal ROI means, then three equal scene means; conditions separate',
        data_role='EXPOSED_REPLAY_NOT_NEW_CONFIRMATION',not_new_confirmation=True,
        scope='fixed-row source distances; not completeness or physical surface identity',
        harm_definition='fixed-support fraction with distance increase greater than 0.1 mm',
        oracle_headroom='(identity MSE-method MSE)/(identity MSE-oracle_AB MSE); ratio after aggregation; negative values retained',
        oracle_scope='attainable only by GT selection among existing p/A/B on fixed support, not global geometric optimum',
        descriptive_best_warning='each condition picks a different arm after evaluation; not a deployable combined result',
        random_repeats_are_not_independent_scenes=True,
        B_incremental_definition='gross max(min(d0^2,dA^2)-dB^2,0) captured on B routes, not net outcome',
        B_net_incremental_definition='signed comparison to evaluator-only best KEEP/A; not a causal B-removed ablation',
        p95_aggregation='mean of ROI percentiles, not percentile of pooled points')
    save_json(out/'SUMMARY.json',summary)
    save_json(out/'REPRODUCTION.json',dict(status='PASS',support_checks=support_checks,
        sample_checks=sample_checks,metric_checks=metric_checks,ply_checks=ply_checks,
        new_exported_ply_checks=60*len(exported_arms),point_error_cache_copied=False,
        max_coordinate_difference_mm=max(r['max_coordinate_difference_mm'] for r in ply_checks),
        max_distance_difference_mm=max(r['max_distance_difference_mm'] for r in sample_checks),
        max_metric_difference=max(r['max_metric_difference'] for r in metric_checks)))
    save_json(out/'RANDOM_CONTROL_AUDIT.json',dict(kind='route_count_matched_within_fixed_support_groups',
        movement_matched=False,random_repeats=10,seed_base=SEED,cases=controls))
    if any(sha(Path(p)) != digest for p,digest in sources.items()):
        raise AssertionError('evaluation source changed during scoring')
    seal(out,sources)
    print('VISIBILITY EVALUATION SEALED',len(rows),'rows',flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
