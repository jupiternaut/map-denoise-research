"""Fixed-support field evaluation; laser access only after all inference seals."""
from common import *
import argparse, csv, time, resource
from scipy.io import loadmat
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ACTUAL = ('identity','prior_recovery','point_wta','single_field','multi_field',
          'multi_field_visibility','multi_field_graph')
ORACLES = ('oracle_discrete','oracle_continuous','oracle_proposals','oracle_all_layers','oracle_expanded')
ARMS = ACTUAL+ORACLES
PROPOSAL_NAMES = ('keep','old_A','old_B','half_A','half_B','minus6','minus3','plus3','plus6')
NUMERIC = ('source_MSE_mm2','source_MAE_mm','source_p95_mm','reverse_MAE_mm',
           'precision_1mm','recall_1mm','Fscore_1mm','improved_fraction','harmed_fraction',
           'moved_support_fraction','move_RMS_mm')


def metrics(output,initial,reference,support,tree,d0):
    q = output[support]
    d = tree.query(q,workers=1)[0]
    rev = cKDTree(q).query(reference,workers=1)[0]
    precision,recall = float(np.mean(d<=1)),float(np.mean(rev<=1))
    displacement = np.sum((q-initial[support])**2,axis=1)
    return dict(source_MSE_mm2=float(np.mean(d*d)),source_MAE_mm=float(d.mean()),
        source_p95_mm=float(np.quantile(d,.95)),reverse_MAE_mm=float(rev.mean()),
        precision_1mm=precision,recall_1mm=recall,
        Fscore_1mm=2*precision*recall/(precision+recall) if precision+recall else 0.,
        improved_fraction=float(np.mean(d-d0 < -.1)),harmed_fraction=float(np.mean(d-d0 > .1)),
        moved_support_fraction=float(np.mean(displacement>1e-14)),
        move_RMS_mm=float(np.sqrt(displacement.mean())))


def pointwise_oracle(initial,support,tree,candidates):
    """GT diagnostic only: finite-candidate minimum with KEEP-preferred ties.

    `candidates` is an iterable of (name, [N,3] coordinates). Output rows outside
    evaluator support stay at the input; no physical-surface identity is inferred.
    """
    q = initial.copy()
    best = tree.query(initial[support],workers=1)[0]
    choice = np.zeros(int(support.sum()),np.int32)
    names = ['identity']
    ids = np.flatnonzero(support)
    for index,(name,points) in enumerate(candidates,1):
        if points.shape != initial.shape or not np.isfinite(points).all():
            raise AssertionError('candidate must be finite and preserve point rows')
        if name in names: raise AssertionError('duplicate oracle candidate name')
        names.append(name)
        d = tree.query(points[support],workers=1)[0]
        better = d < best
        q[ids[better]] = points[ids[better]]
        choice[better] = index
        best[better] = d[better]
    return q,best,choice,names


def proposal_candidates(pool,names):
    """Only the initial nine physical positions: explicitly exclude new fields."""
    if pool.ndim!=3 or pool.shape[2]!=3 or pool.shape[1]!=len(names) or tuple(names[:9])!=PROPOSAL_NAMES:
        raise AssertionError('initial proposal labels/order differ from frozen contract')
    return [('proposal__'+str(name),pool[:,j]) for j,name in enumerate(names[:9])]


def summarize(rows):
    result = {}
    lookup = {(r['scene'],r['roi'],r['condition'],r['arm']):r for r in rows}
    if len(lookup)!=len(rows): raise AssertionError('duplicate metric row')
    for condition in CONDS:
        result[condition] = {}
        before = np.mean([r['source_MSE_mm2'] for r in rows if r['condition']==condition and r['arm']=='identity'])
        for arm in ARMS:
            rr = [r for r in rows if r['condition']==condition and r['arm']==arm]
            if len(rr)!=12 or any(sum(r['scene']==s for r in rr)!=4 for s in SCENES):
                raise AssertionError('12 rows, four per scene required')
            cell = {key:float(np.mean([np.mean([r[key] for r in rr if r['scene']==s]) for s in SCENES])) for key in NUMERIC}
            delta = np.array([r['source_MSE_mm2']-lookup[(r['scene'],r['roi'],condition,'identity')]['source_MSE_mm2'] for r in rr])
            cell.update(MSE_gain_percent=float(100*(1-cell['source_MSE_mm2']/before)),
                wins=int(np.sum(delta < -1e-12)),ties=int(np.sum(abs(delta)<=1e-12)),losses=int(np.sum(delta>1e-12)))
            result[condition][arm] = cell
        comparisons = {}
        for a,b in (('single_field','point_wta'),('multi_field','single_field'),
                    ('multi_field_visibility','multi_field'),('multi_field_graph','multi_field_visibility')):
            delta = np.array([lookup[(s,r['roi'],condition,a)]['source_MSE_mm2']-lookup[(s,r['roi'],condition,b)]['source_MSE_mm2']
                for s in SCENES for r in rows if r['scene']==s and r['condition']==condition and r['arm']=='identity'])
            comparisons[a+'_minus_'+b] = dict(MSE_difference_mm2=float(delta.mean()),
                wins=int(np.sum(delta < -1e-12)),ties=int(np.sum(abs(delta)<=1e-12)),losses=int(np.sum(delta>1e-12)))
        result[condition]['matched_comparisons'] = comparisons
        result[condition]['oracle_headroom'] = dict(
            discrete_MSE_mm2=result[condition]['oracle_discrete']['source_MSE_mm2'],
            previous_continuous_MSE_mm2=result[condition]['oracle_continuous']['source_MSE_mm2'],
            proposals_MSE_mm2=result[condition]['oracle_proposals']['source_MSE_mm2'],
            new_pool_MSE_mm2=result[condition]['oracle_all_layers']['source_MSE_mm2'],
            expanded_MSE_mm2=result[condition]['oracle_expanded']['source_MSE_mm2'],
            new_pool_gain_vs_discrete_mm2=result[condition]['oracle_discrete']['source_MSE_mm2']-result[condition]['oracle_all_layers']['source_MSE_mm2'],
            proposal_expansion_gain_vs_discrete_mm2=result[condition]['oracle_discrete']['source_MSE_mm2']-result[condition]['oracle_proposals']['source_MSE_mm2'],
            field_added_gain_vs_proposals_mm2=result[condition]['oracle_proposals']['source_MSE_mm2']-result[condition]['oracle_all_layers']['source_MSE_mm2'],
            expansion_gain_vs_continuous_mm2=result[condition]['oracle_continuous']['source_MSE_mm2']-result[condition]['oracle_expanded']['source_MSE_mm2'])
    return result


def evaluate_scene(sid):
    check_host()
    # This gate executes before any laser, support mask or evaluator scores open.
    for scene in SCENES: verify_seal(OUT/'inference'/f'scan{scene}')
    dest = OUT/'evaluation'/f'scan{sid}'
    dest.mkdir(parents=True,exist_ok=False)
    sources = {str(ROOT/n):sha(ROOT/n) for n in ('PROTOCOL.md','EVALUATION_ADDENDUM.md','common.py','evaluate.py','test_evaluation.py')}
    for scene in SCENES:
        path=OUT/'inference'/f'scan{scene}'/'SEALED.json';sources[str(path)]=sha(path)
    save_json(dest/'LOCK.json',dict(sources=sources.copy(),scene=sid,
        all_actual_outputs_sealed_before_reference_access=True,primary='multi_field',
        data_role='exposed_replay',fixed_native_row_support=True,reference_voxel_mm=.8,
        actual_arms=ACTUAL,diagnostic_oracle_arms=ORACLES,
        oracle_proposals='initial nine POOL positions only; excludes all fitted field predictions',
        oracle_pool='identity, A, B, actual outputs, all exported POOL entries; pointwise GT selection',
        oracle_expanded='same pool plus prior GT continuous-union output; evaluation only'))
    raw=CLOSEOUT/'confirmation'/f'scan{sid}'
    frozen_file(raw/'ROIS.json',raw,sources)
    rois=json.loads((raw/'ROIS.json').read_text())
    if len(rois)!=4: raise AssertionError('four ROI required')
    refroot=DATA/'closeout-confirmation-v1';manifest_path=refroot/'REFERENCE_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text());sources[str(manifest_path)]=sha(manifest_path)
    folder=refroot/'evaluation_only'/f'scan{sid}'
    for rec in manifest['records']:
        path=Path(rec['path'])
        if path.parent==folder:
            if sha(path)!=rec['sha256']: raise AssertionError('reference changed')
            sources[str(path)]=rec['sha256']
    laser=read_points(folder/f'stl{sid:03d}_total.ply');obs=loadmat(folder/f'ObsMask{sid}_10.mat')
    old_path=CONT_OUT/'evaluation/METRICS.csv';frozen_file(old_path,CONT_OUT/'evaluation',sources)
    with old_path.open() as f:
        archive={(int(r['scene']),r['roi'],r['condition'],r['arm']):r for r in csv.DictReader(f)}
    rows,checks,timings=[],[],[];start=time.monotonic()
    for roi in rois:
        rid=roi['id'];native_path=raw/(rid+'__native')/'identity.ply'
        frozen_file(native_path,raw,sources);native=read_points(native_path)
        support=io.in_box(native,roi['lo'],roi['hi'])&io.observed(native,obs)
        spath=CLOSEOUT/'evaluation'/(rid+'_native_support.npy');frozen_file(spath,CLOSEOUT/'evaluation',sources)
        if not support.any() or not np.array_equal(support,np.load(spath,allow_pickle=False)): raise AssertionError('support changed')
        ids=np.flatnonzero(support)
        reference=io.voxel(laser[io.in_box(laser,roi['lo'],roi['hi'])&io.observed(laser,obs)]);tree=cKDTree(reference)
        for cond in CONDS:
            tick=time.monotonic();case=raw/(rid+'__'+cond);inf=OUT/'inference'/f'scan{sid}'/case.name
            geometry,route=geometry_and_route(case,sources);p,a,b=geometry[:,0],geometry[:,1],geometry[:,2]
            endpoint=selected_endpoint(geometry,route);d0=tree.query(p[support],workers=1)[0]
            outputs={}
            for arm in ACTUAL:
                path=inf/(arm+'.ply');frozen_file(path,inf.parent,sources);outputs[arm]=read_points(path)
                if outputs[arm].shape!=p.shape or not np.isfinite(outputs[arm]).all(): raise AssertionError('actual output invalid')
            if not np.array_equal(outputs['identity'],p) or not np.array_equal(outputs['prior_recovery'],endpoint):
                raise AssertionError('frozen baseline coordinate changed')
            pool_path=inf/'POOL.npz';frozen_file(pool_path,inf.parent,sources)
            with np.load(pool_path,allow_pickle=False) as z:
                pool,names=z['points'],z['candidate_names'].tolist()
            if pool.ndim!=3 or pool.shape[0]!=len(p) or pool.shape[2]!=3 or len(names)!=pool.shape[1] or not np.isfinite(pool).all():
                raise AssertionError('invalid POOL contract')
            if len(set(names))!=len(names): raise AssertionError('duplicate POOL labels')
            old_case=CONT_OUT/'evaluation'/f'scan{sid}'/case.name
            old_cont=old_case/'DIAGNOSTIC_oracle_segment_union.ply';frozen_file(old_cont,old_case.parent,sources)
            continuous=read_points(old_cont)
            if continuous.shape!=p.shape: raise AssertionError('old oracle row mismatch')
            discrete,dd,dc,dnames=pointwise_oracle(p,support,tree,[('A',a),('B',b)])
            proposals,dp,pc,pnames=pointwise_oracle(p,support,tree,proposal_candidates(pool,names))
            candidates=[('A',a),('B',b)]+[(arm,outputs[arm]) for arm in ACTUAL if arm!='identity']
            candidates += [('pool__'+str(name),pool[:,j]) for j,name in enumerate(names)]
            layer,dl,lc,lnames=pointwise_oracle(p,support,tree,candidates)
            expanded,de,ec,enames=pointwise_oracle(p,support,tree,[('all_layers_oracle',layer),('previous_continuous_oracle',continuous)])
            outputs.update(oracle_discrete=discrete,oracle_continuous=continuous,oracle_proposals=proposals,oracle_all_layers=layer,oracle_expanded=expanded)
            dcont=tree.query(continuous[support],workers=1)[0]
            if np.any(dp>dd+1e-8) or np.any(dl>np.minimum(dd,dp)+1e-10) or np.any(de>np.minimum(dl,dcont)+1e-10):
                raise AssertionError('oracle candidate inclusion failed')
            outcase=dest/case.name;outcase.mkdir()
            write_points(outcase/'DIAGNOSTIC_oracle_all_layers.ply',layer)
            write_points(outcase/'DIAGNOSTIC_oracle_expanded.ply',expanded)
            save_npz(outcase/'ORACLE_CHOICES.npz',support_row_ids=ids,proposal_choice=pc,
                proposal_candidate_names=np.asarray(pnames),all_layers_choice=lc,
                all_layers_candidate_names=np.asarray(lnames),expanded_choice=ec,expanded_candidate_names=np.asarray(enames))
            for arm,q in outputs.items():
                metric=metrics(q,p,reference,support,tree,d0)
                row=dict(scene=sid,roi=rid,condition=cond,arm=arm,n_rows=len(p),n_source=len(ids),n_reference=len(reference),
                    family='diagnostic_oracle' if arm in ORACLES else 'observation_only',**metric)
                if arm in ('identity','prior_recovery','oracle_discrete','oracle_continuous'):
                    old_name='oracle_segment_union' if arm=='oracle_continuous' else arm
                    old=archive[(sid,rid,cond,old_name)]
                    diff=max(abs(metric[k]-float(old[k])) for k in NUMERIC)
                    if diff>1e-8: raise AssertionError(('historical metric changed',case.name,arm,diff))
                    checks.append(dict(case=case.name,check='historical_'+arm,max_difference=diff))
                rows.append(row)
            # Different NN implementation checks both a new actual and oracle output.
            import open3d as o3d
            sample=ids[np.linspace(0,len(ids)-1,min(64,len(ids)),dtype=int)]
            dst=o3d.geometry.PointCloud();dst.points=o3d.utility.Vector3dVector(reference)
            for arm in ('multi_field_visibility','oracle_all_layers'):
                src=o3d.geometry.PointCloud();src.points=o3d.utility.Vector3dVector(outputs[arm][sample])
                independent=np.asarray(src.compute_point_cloud_distance(dst));expected=tree.query(outputs[arm][sample],workers=1)[0]
                diff=float(np.max(abs(independent-expected)))
                if diff>1e-8: raise AssertionError('independent Open3D NN mismatch')
                checks.append(dict(case=case.name,check='open3d_'+arm,max_difference=diff,n=len(sample)))
            timings.append(dict(case=case.name,seconds=time.monotonic()-tick,pool_candidates=len(names),n_source=len(ids)))
            print('FIELD EVALUATED',case.name,round(timings[-1]['seconds'],2),'s',flush=True)
    if len(rows)!=20*len(ARMS): raise AssertionError('incomplete scene evaluation')
    with (dest/'METRICS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(dest/'RESULTS.json',dict(status='PASS',rows=rows,checks=checks,timing=timings,
        wall_seconds=time.monotonic()-start,own_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    if any(sha(path)!=digest for path,digest in sources.items()): raise AssertionError('evaluation source changed')
    seal(dest,sources)


def finalize():
    rows,checks=[],[];sources={str(ROOT/n):sha(ROOT/n) for n in ('evaluate.py','PROTOCOL.md','EVALUATION_ADDENDUM.md')}
    for sid in SCENES:
        folder=OUT/'evaluation'/f'scan{sid}';verify_seal(folder);rec=json.loads((folder/'RESULTS.json').read_text())
        rows+=rec['rows'];checks+=rec['checks'];sources[str(folder/'SEALED.json')]=sha(folder/'SEALED.json')
    if len(rows)!=60*len(ARMS): raise AssertionError('incomplete evaluation')
    folder=OUT/'evaluation'
    with (folder/'METRICS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(folder/'SUMMARY.json',dict(exposed_replay=summarize(rows),primary='multi_field',secondary=['multi_field_visibility','multi_field_graph'],
        rows=len(rows),cases=60,roi_count=12,independent_scenes=0,source_aggregation='ROI equally then scene equally',
        oracle_scope='finite reference and explicit candidate coordinates only; GT selectors are not deployable methods',
        reverse_metric_scope='cropped reference against supported output footprint; not official full-scene completeness',
        movement_denominator='fixed support'))
    save_json(folder/'REPRODUCTION.json',dict(status='PASS',checks=checks))
    seal(folder,sources);print('FIELD EVALUATION SEALED',len(rows),'rows',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,choices=SCENES);ap.add_argument('--finalize',action='store_true');args=ap.parse_args()
    with threadpool_limits(limits=1):
        if args.finalize:finalize()
        elif args.scene:evaluate_scene(args.scene)
        else:ap.error('choose --scene or --finalize')
