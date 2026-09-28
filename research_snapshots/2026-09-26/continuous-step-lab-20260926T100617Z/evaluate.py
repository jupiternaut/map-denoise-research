"""Fixed-support replay evaluation. No import from observation-only construction."""
from common import *
import argparse, csv, time, resource
from scipy.io import loadmat
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from segment_oracle import segment_nearest

ARMS = ('identity','prior_recovery','constant_025','constant_050','constant_075',
        'grid','quadratic','oracle_discrete','oracle_segment_union','oracle_selected_segment')
NUMERIC = ('source_MSE_mm2','source_MAE_mm','source_p95_mm','reverse_MAE_mm',
           'precision_1mm','recall_1mm','Fscore_1mm','improved_fraction',
           'harmed_fraction','moved_support_fraction','move_RMS_mm',
           'interior_step_support_fraction')

def metrics(output, initial, reference, support, tree, d0, step=None):
    q = output[support]
    d = tree.query(q, workers=1)[0]
    rev = cKDTree(q).query(reference, workers=1)[0]
    precision, recall = float(np.mean(d<=1)), float(np.mean(rev<=1))
    displacement = np.sum((q-initial[support])**2,axis=1)
    interior = 0. if step is None else float(np.mean((step[support]>1e-8)&(step[support]<1-1e-8)&(displacement>1e-14)))
    return dict(source_MSE_mm2=float(np.mean(d*d)),source_MAE_mm=float(d.mean()),
        source_p95_mm=float(np.quantile(d,.95)),reverse_MAE_mm=float(rev.mean()),
        precision_1mm=precision,recall_1mm=recall,
        Fscore_1mm=2*precision*recall/(precision+recall) if precision+recall else 0.,
        improved_fraction=float(np.mean(d-d0 < -.1)),harmed_fraction=float(np.mean(d-d0 > .1)),
        moved_support_fraction=float(np.mean(displacement>1e-14)),
        move_RMS_mm=float(np.sqrt(displacement.mean())),interior_step_support_fraction=interior),d

def summarize(rows):
    result = {}
    for condition in CONDS:
        result[condition] = {}
        before = np.mean([r['source_MSE_mm2'] for r in rows if r['condition']==condition and r['arm']=='identity'])
        for arm in ARMS:
            rr = [r for r in rows if r['condition']==condition and r['arm']==arm]
            if len(rr)!=12: raise AssertionError(('need12ROIs',condition,arm,len(rr)))
            cell = {key:float(np.mean([r[key] for r in rr])) for key in NUMERIC}
            baseline = {(r['scene'],r['roi']):r['source_MSE_mm2'] for r in rows if r['condition']==condition and r['arm']=='identity'}
            delta = np.array([r['source_MSE_mm2']-baseline[(r['scene'],r['roi'])] for r in rr])
            cell.update(MSE_gain_percent=float(100*(1-cell['source_MSE_mm2']/before)),
                wins=int(np.sum(delta < -1e-12)),ties=int(np.sum(abs(delta)<=1e-12)),losses=int(np.sum(delta>1e-12)))
            result[condition][arm] = cell
        disc,seg,route,actual = [result[condition][a]['source_MSE_mm2'] for a in ('oracle_discrete','oracle_segment_union','oracle_selected_segment','prior_recovery')]
        result[condition]['decomposition'] = dict(discrete_to_segment_MSE_reduction_mm2=disc-seg,
            discrete_oracle_residual_reduction_percent=100*(disc-seg)/disc,
            selected_branch_step_headroom_mm2=actual-route,
            remaining_branch_or_KEEP_gap_mm2=route-seg,
            reference_sampling_and_path_residual_mm2=seg)
    return result

def evaluate_scene(sid):
    check_host()
    # All observation-only methods sealed before loading any evaluator geometry.
    inf_seals = {s:verify_seal(OUT/'inference'/f'scan{s}') for s in SCENES}
    dest = OUT/'evaluation'/f'scan{sid}'
    dest.mkdir(parents=True,exist_ok=False)
    sources = {str(ROOT/n):sha(ROOT/n) for n in ('PROTOCOL.md','common.py','evaluate.py','segment_oracle.py')}
    for s in SCENES:
        sources[str(OUT/'inference'/f'scan{s}'/'SEALED.json')] = sha(OUT/'inference'/f'scan{s}'/'SEALED.json')
    save_json(dest/'LOCK.json',dict(sources=sources.copy(),scene=sid,
        observation_outputs_sealed_before_reference_access=True,data_role='exposed_replay',
        metric='point-to-finite-reference MSE',primary_actual_arm='grid',
        fixed_support=True,reference_voxel_mm=.8))
    raw = CLOSEOUT/'confirmation'/f'scan{sid}'
    frozen_file(raw/'ROIS.json',raw,sources)
    rois = json.loads((raw/'ROIS.json').read_text())
    refroot = DATA/'closeout-confirmation-v1'
    manifest_path = refroot/'REFERENCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    sources[str(manifest_path)] = sha(manifest_path)
    folder = refroot/'evaluation_only'/f'scan{sid}'
    for rec in manifest['records']:
        path = Path(rec['path'])
        if path.parent == folder:
            if sha(path)!=rec['sha256']: raise AssertionError('reference changed')
            sources[str(path)] = rec['sha256']
    laser = read_points(folder/f'stl{sid:03d}_total.ply')
    obs = loadmat(folder/f'ObsMask{sid}_10.mat')
    old_summary = json.loads((VIS/'evaluation'/'SUMMARY.json').read_text())
    old_rows = {}
    with (VIS/'evaluation'/'METRICS.csv').open() as f:
        for r in csv.DictReader(f):
            if r['arm'] in ('identity',RECOVERY,'oracle_AB'):
                old_rows[(r['roi'],r['condition'],r['arm'])] = float(r['source_MSE_mm2'])
    for name in ('SUMMARY.json','METRICS.csv'):
        frozen_file(VIS/'evaluation'/name,VIS/'evaluation',sources)
    rows, checks, timing = [],[],[]
    start = time.monotonic()
    for roi in rois:
        rid = roi['id']
        native = read_points(raw/(rid+'__native')/'identity.ply')
        support = io.in_box(native,roi['lo'],roi['hi']) & io.observed(native,obs)
        spath = CLOSEOUT/'evaluation'/(rid+'_native_support.npy')
        frozen_file(spath,CLOSEOUT/'evaluation',sources)
        if not np.array_equal(support,np.load(spath,allow_pickle=False)): raise AssertionError('support changed')
        ids = np.flatnonzero(support)
        reference = io.voxel(laser[io.in_box(laser,roi['lo'],roi['hi']) & io.observed(laser,obs)])
        tree = cKDTree(reference)
        for cond in CONDS:
            tick = time.monotonic()
            case = raw/(rid+'__'+cond)
            geometry,route = geometry_and_route(case,sources)
            p,a,b = geometry[:,0],geometry[:,1],geometry[:,2]
            endpoint = selected_endpoint(geometry,route)
            d0,dA,dB = [tree.query(x[support],workers=1)[0] for x in (p,a,b)]
            olderr = CROSS/'evaluation'/case.name/'point_errors.npz'
            frozen_file(olderr,CROSS/'evaluation',sources)
            with np.load(olderr,allow_pickle=False) as z:
                cache_error = max(float(np.max(abs(d-z[k][support]))) for d,k in zip((d0,dA,dB),('d0','dA','dB')))
            if cache_error>1e-8: raise AssertionError(('old distances mismatch',case.name,cache_error))
            oa = segment_nearest(p[support],a[support],tree=tree)
            ob = segment_nearest(p[support],b[support],tree=tree)
            best_b = ob.squared_distance < oa.squared_distance
            union_loss = np.minimum(oa.squared_distance,ob.squared_distance)
            discrete_loss = np.minimum.reduce((d0*d0,dA*dA,dB*dB))
            if np.any(union_loss>discrete_loss+1e-9): raise AssertionError('oracle inclusion violated')
            q_union = p.copy()
            alpha_union = np.zeros(len(p))
            alpha_union[support] = np.where(best_b,ob.lambda_,oa.lambda_)
            q_union[support] = p[support]+alpha_union[support,None]*(np.where(best_b[:,None],b[support],a[support])-p[support])
            alpha_route = np.zeros(len(p))
            alpha_route[support] = np.where(route[support]==1,oa.lambda_,np.where(route[support]==2,ob.lambda_,0.))
            q_route = materialize_step(geometry,route,alpha_route)
            discrete_choice = np.argmin(np.column_stack((d0,dA,dB)),axis=1)
            q_discrete = p.copy()
            q_discrete[support] = geometry[ids,discrete_choice]
            inf = OUT/'inference'/f'scan{sid}'/case.name
            for filename in ('grid.ply','quadratic.ply','STEPS.npz'):
                frozen_file(inf/filename,inf.parent,sources)
            with np.load(inf/'STEPS.npz',allow_pickle=False) as z:
                steps = {key:z[key] for key in ('grid','quadratic')}
            outputs = dict(identity=p,prior_recovery=endpoint,
                constant_025=p+.25*(endpoint-p),constant_050=p+.5*(endpoint-p),constant_075=p+.75*(endpoint-p),
                grid=read_points(inf/'grid.ply'),quadratic=read_points(inf/'quadratic.ply'),
                oracle_discrete=q_discrete,oracle_segment_union=q_union,oracle_selected_segment=q_route)
            outcase = dest/case.name
            outcase.mkdir()
            write_points(outcase/'DIAGNOSTIC_oracle_segment_union.ply',q_union)
            save_npz(outcase/'ORACLE_STEPS.npz',support_row_ids=ids,A_step=oa.lambda_,B_step=ob.lambda_,
                A_squared_distance=oa.squared_distance,B_squared_distance=ob.squared_distance,
                union_step=alpha_union[support],union_branch=np.where(best_b,2,1).astype(np.uint8),
                selected_route_step=alpha_route[support])
            for arm,q in outputs.items():
                if q.shape!=p.shape or not np.isfinite(q).all(): raise AssertionError('bad geometry')
                step = steps.get(arm)
                if arm.startswith('constant_'):
                    step = np.full(len(p),{'constant_025':.25,'constant_050':.5,'constant_075':.75}[arm])
                if arm in ('grid','quadratic'):
                    expected = materialize_step(geometry,route,step)
                    if not np.allclose(q,expected,rtol=0,atol=1e-10): raise AssertionError('wrong step geometry')
                    if np.any((step<0)|(step>1)) or not np.array_equal(q[route==0],p[route==0]): raise AssertionError('KEEP or interval violated')
                if arm=='oracle_segment_union': step=alpha_union
                if arm=='oracle_selected_segment': step=alpha_route
                m,d = metrics(q,p,reference,support,tree,d0,step)
                m.update(scene=sid,roi=rid,condition=cond,arm=arm,n_rows=len(p),n_source=len(ids),n_reference=len(reference),
                    family='diagnostic_oracle' if arm.startswith('oracle') else 'observation_only')
                if arm in ('identity','prior_recovery','oracle_discrete'):
                    old_name = dict(identity='identity',prior_recovery=RECOVERY,oracle_discrete='oracle_AB')[arm]
                    diff = abs(m['source_MSE_mm2']-old_rows[(rid,cond,old_name)])
                    if diff>1e-9: raise AssertionError(('baseline changed',arm,diff))
                    checks.append(dict(case=case.name,check=arm,max_difference=diff))
                if arm=='oracle_segment_union':
                    diff = float(np.max(abs(d*d-union_loss)))
                    if diff>1e-8: raise AssertionError(('segment coordinate mismatch',diff))
                    checks.append(dict(case=case.name,check='continuous_coordinate_loss',max_difference=diff))
                rows.append(m)
            cm = {r['arm']:r['source_MSE_mm2'] for r in rows if r['roi']==rid and r['condition']==cond}
            if cm['oracle_selected_segment']>cm['prior_recovery']+1e-9: raise AssertionError('route oracle monotonicity')
            # Independent Open3D point-to-point nearest backend on 64 fixed support rows.
            import open3d as o3d
            sample = np.linspace(0,len(ids)-1,min(64,len(ids)),dtype=int)
            src,dst = o3d.geometry.PointCloud(),o3d.geometry.PointCloud()
            src.points=o3d.utility.Vector3dVector(q_union[ids[sample]])
            dst.points=o3d.utility.Vector3dVector(reference)
            independent=np.asarray(src.compute_point_cloud_distance(dst))**2
            diff=float(np.max(abs(independent-union_loss[sample])))
            if diff>1e-8: raise AssertionError('independent NN mismatch')
            checks.append(dict(case=case.name,check='open3d_oracle',max_difference=diff,n=len(sample)))
            record=dict(case=case.name,wall_seconds=time.monotonic()-tick,n_source=len(ids),
                A_candidate_evaluations=int(oa.candidate_evaluations),B_candidate_evaluations=int(ob.candidate_evaluations),
                A_max_candidates=int(oa.max_candidates_per_segment),B_max_candidates=int(ob.max_candidates_per_segment))
            timing.append(record)
            print('EVALUATED',case.name,round(record['wall_seconds'],2),'s',flush=True)
    if len(rows)!=200: raise AssertionError('incomplete scene')
    with (dest/'METRICS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(dest/'RESULTS.json',dict(rows=rows,checks=checks,timing=timing,wall_seconds=time.monotonic()-start,
        own_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,status='PASS'))
    seal(dest,sources)

def finalize():
    rows,checks=[],[]
    sources={str(ROOT/'evaluate.py'):sha(ROOT/'evaluate.py'),str(ROOT/'PROTOCOL.md'):sha(ROOT/'PROTOCOL.md')}
    for sid in SCENES:
        folder=OUT/'evaluation'/f'scan{sid}'
        verify_seal(folder)
        rec=json.loads((folder/'RESULTS.json').read_text())
        rows+=rec['rows'];checks+=rec['checks']
        sources[str(folder/'SEALED.json')]=sha(folder/'SEALED.json')
    if len(rows)!=600: raise AssertionError('expected 600 metric rows')
    folder=OUT/'evaluation'
    with (folder/'METRICS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(folder/'SUMMARY.json',dict(exposed_replay=summarize(rows),primary='grid',secondary='quadratic',
        rows=len(rows),cases=60,roi_count=12,independent_scenes=0,source_aggregation='ROI equally then scene equally'))
    save_json(folder/'REPRODUCTION.json',dict(status='PASS',checks=checks))
    seal(folder,sources)
    print('EVALUATION SEALED',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,choices=SCENES);ap.add_argument('--finalize',action='store_true')
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        if args.finalize: finalize()
        elif args.scene: evaluate_scene(args.scene)
        else: ap.error('choose --scene or --finalize')
