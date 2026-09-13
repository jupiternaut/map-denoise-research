"""Read-only V6 replay plus tiny train-only physical-plane diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import resource
import shutil
import socket
import sys
import tempfile
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT/'exploration_v6'/'direction'))
import direction_pooling as wrapper
from geometry import graph_plane_distances, fit_planes, plane_errors, spatial_stripe_holdout

RUNS = Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
FROZEN = RUNS/'direction-v6-edpxxdsb'
PATCHES = ('da_wall', 'da_junction', 'da_thin', 'cy_wall', 'cy_junction', 'cy_thin')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def save_csv(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def finite_median(a):
    a = np.asarray(a)
    return float(np.median(a[np.isfinite(a)])) if np.isfinite(a).any() else None


def physical_replay(dest, graph, pool):
    rows, states, replay = [], {}, []
    for patch in PATCHES:
        for mode in ('zero', 'normal_translation'):
            case = patch+'__'+mode
            path = FROZEN/'outputs'/('real__'+case+'__pooled_pca.json')
            info = json.loads(path.read_text())['info']
            saved = info['v4_info']['initial_info']
            with np.load(FROZEN/'real'/'inputs'/(case+'.npz'), allow_pickle=False) as a:
                world, frames = a['xyz_world'], a['scan_id']
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as a:
                stored_support = a['support_mask']
            order = pool._canonical_order((world-world.mean(0))*1000., frames)
            ordered = world[order]
            ids, scans = np.unique(frames[order], return_inverse=True)
            centered = (ordered-ordered.mean(0))*1000.
            basis = np.asarray(info['basis_calls'][0]['basis_world'])
            assert wrapper.fingerprint(centered) == info['basis_calls'][0]['point_coordinates_sha256']
            np.testing.assert_allclose(basis.T@basis, np.eye(3), atol=1e-12)
            bias = np.asarray(info['bias_mm'])
            local = centered@basis
            labels, members, geometry, grid = graph._cells(local)
            assert grid.tolist() == saved['grid_shape']
            predicted_support = np.zeros(len(world), bool)
            residual_full = np.full((len(world), 2), np.nan)
            for cell, take in enumerate(members):
                if len(take)<24 or len(np.unique(scans[take]))<2:
                    continue
                xy, scale = geometry[cell]
                values = local[take, 2]-bias[scans[take]]
                model = graph._local_model(values, xy, scale, float(info['sigma_mm']))
                physical = graph_plane_distances(values, xy, model['beta'], model['assigned'], scale)
                height, orth = physical['height_mm'], physical['orthogonal_mm']
                np.testing.assert_allclose(height, abs(model['residual']), atol=1e-11)
                sigma = float(info['sigma_mm'])
                raw_ok = (height <= 4*sigma)&(model['confidence']>=.8)
                cell_ok = np.median(height)<=2.5*sigma and np.linalg.norm(physical['physical_slopes'])<=.25
                accepted = raw_ok if cell_ok else np.zeros(len(take), bool)
                predicted_support[take[accepted]] = True
                assert model['k'] == saved['local_k'][str(cell)]
                assert int((~accepted).sum()) == saved['rejected_points_by_cell'][str(cell)]
                residual_full[take, 0] = height
                residual_full[take, 1] = orth
                rows.append(dict(case=case, patch=patch, mode=mode, cell=cell,
                    point_count=len(take), k=int(model['k']), accepted_count=int(accepted.sum()),
                    height_median_mm=float(np.median(height)), orthogonal_median_mm=float(np.median(orth)),
                    height_p95_mm=float(np.quantile(height,.95)), orthogonal_p95_mm=float(np.quantile(orth,.95)),
                    axis_cosine=physical['axis_cosine'], axis_condition=physical['axis_condition'],
                    physical_slope_norm=float(np.linalg.norm(physical['physical_slopes'])),
                    old_height_gate_failed=bool(np.median(height)>2.5*sigma),
                    orthogonal_median_le_5mm=bool(np.median(orth)<=5.),
                    direction_interpretation='saved-input-PCA; conditional on saved estimated scan bias'))
            support = np.zeros(len(world), bool); support[order] = predicted_support
            np.testing.assert_array_equal(support, stored_support)
            np.savez_compressed(dest/'replays'/(case+'.npz'), ordered_point_indices=order,
                conditional_height_orthogonal_residuals_mm=residual_full, support_sorted=predicted_support)
            replay.append(dict(case=case, support_exact=True, supported_points=int(support.sum()),
                               covered_points=int(np.isfinite(residual_full[:,0]).sum()), points=len(world)))
            if mode == 'zero':
                states[patch] = dict(local=local, scans=scans, ids=ids, basis=basis, members=members, grid=grid)
    return rows, states, replay


def spatial_anchors(local, members):
    eligible = [i for i, take in enumerate(members) if len(take)>=24]
    if not eligible:
        return []
    centers = np.array([local[members[i],:2].mean(0) for i in eligible])
    extent = np.maximum(np.ptp(centers, axis=0), 1.)
    centers = centers/extent
    picked = [int(np.argmin(np.sum((centers-centers.mean(0))**2, axis=1)))]
    while len(picked) < min(3, len(eligible)):
        distances = np.min(np.sum((centers[:,None]-centers[picked])**2,axis=2),axis=1)
        distances[picked] = -1
        picked.append(int(np.argmax(distances)))
    return [eligible[i] for i in picked]


def holdout_diagnostic(dest, states):
    rows, splits, models_saved = [], [], []
    for patch, state in states.items():
        local, scans = state['local'], state['scans']
        for cell in spatial_anchors(local, state['members']):
            take = state['members'][cell]
            low, high = local[take,:2].min(0), local[take,:2].max(0)
            center, width = (low+high)/2., np.maximum(high-low, 1.)
            is_test = spatial_stripe_holdout(local[take],scans[take])
            half = np.all(abs(local[take,:2]-center)<=width*.25, axis=1)
            core_test = take[is_test & half]
            splits.append(dict(patch=patch, cell=cell, total=len(take), common_test=len(core_test),
                full_test=int(is_test.sum()), half_test=int((is_test&half).sum()),
                full_train=int((~is_test).sum()), half_train=int((~is_test&half).sum()),
                full_test_not_in_core=int((is_test&~half).sum()),
                rule='per-scan major-axis 8 empirical-quantile spatial stripes; stripes 1,5 test; common half-scale core',
                patch_basis_scope='current all-input PCA used only for fixed spatial design and diagnostics; no all-input bias used'))
            np.savez_compressed(dest/'splits'/f'{patch}__c{cell}.npz', original_cell_sorted_indices=take,
                full_train_sorted_indices=take[~is_test], half_train_sorted_indices=take[~is_test&half],
                common_test_sorted_indices=core_test)
            for scale_name, subset in (('full',np.ones(len(take),bool)),('half',half)):
                train = take[~is_test&subset]
                assert not np.intersect1d(train,core_test).size
                for k in (1,2):
                    for sharing in ('joint','per_scan'):
                        fitted, fail = {}, {}
                        fit_scans = [None] if sharing=='joint' else list(np.unique(scans[take]))
                        for sid in fit_scans:
                            fit_take = train if sid is None else train[scans[train]==sid]
                            key = -1 if sid is None else int(sid)
                            try:
                                fitted[key] = fit_planes(local[fit_take],k)
                            except ValueError as exc:
                                fail[key] = str(exc)
                        faces = sum(model['actual_faces'] for model in fitted.values())
                        for key, model in fitted.items():
                            models_saved.append(dict(patch=patch,cell=cell,scale=scale_name,k=k,sharing=sharing,scan_key=key,
                                **{name:(value.tolist() if isinstance(value,np.ndarray) else value) for name,value in model.items()}))
                        for sid in np.unique(scans[take]):
                            eval_take = core_test[scans[core_test]==sid]
                            key = -1 if sharing=='joint' else int(sid)
                            row = dict(patch=patch,cell=cell,scale=scale_name,per_fit_capacity=k,sharing=sharing,
                                evaluation_scan_id=int(state['ids'][sid]),train_points=int(len(train)),
                                scan_train_points=int(np.count_nonzero(scans[train]==sid)),test_points=len(eval_take),
                                requested_total_faces=k*len(fit_scans),actual_total_faces=faces,
                                geometry_degrees_of_freedom=3*faces,coverage_count=0,status='NO_CORE_TEST',
                                orthogonal_median_mm=None,orthogonal_rms_mm=None,axis_distance_median_mm=None,
                                axis_cosine_median=None,axis_cosine_min=None,axis_infinite_count=0,
                                fitted_covariance_ranks='',failure_reason='')
                            if len(eval_take) and key not in fitted:
                                row.update(status='UNFIT',failure_reason=fail.get(key,'missing model'))
                            elif len(eval_take):
                                model = fitted[key]
                                orth, axis, cosine, assigned = plane_errors(local[eval_take],model,np.array([0.,0.,1.]))
                                row.update(status='SCORED',coverage_count=len(eval_take),
                                    orthogonal_median_mm=float(np.median(orth)),orthogonal_rms_mm=float(np.sqrt(np.mean(orth**2))),
                                    axis_distance_median_mm=finite_median(axis),axis_cosine_median=float(np.median(cosine)),
                                    axis_cosine_min=float(cosine.min()),axis_infinite_count=int((~np.isfinite(axis)).sum()),
                                    fitted_covariance_ranks=';'.join(map(str,model['ranks'])))
                            rows.append(row)
    return rows,splits,models_saved


def main():
    if socket.gethostname() != 'liekkas':
        raise RuntimeError('exact target host liekkas required')
    started = time.perf_counter()
    private = wrapper._load_private(); pool=private._V4; graph=pool._V3
    protected = sorted({p for p in FROZEN.rglob('*') if p.is_file()} |
        {Path(wrapper.__file__),Path(private.__file__),Path(pool.__file__),Path(graph.__file__)})
    before = {str(p):sha(p) for p in protected}
    dest = Path(tempfile.mkdtemp(prefix='real-geometry-v7-',dir=RUNS))
    print(dest,flush=True)
    (dest/'replays').mkdir(); (dest/'splits').mkdir(); (dest/'source').mkdir()
    for path in HERE.glob('*.py'):
        shutil.copyfile(path,dest/'source'/path.name)
    save_json(dest/'PROTECTED_BEFORE.json',before)
    rows, states, replay = physical_replay(dest,graph,pool)
    holdout,splits,models = holdout_diagnostic(dest,states)
    save_csv(dest/'PHYSICAL_REPLAY.csv',rows)
    save_csv(dest/'HOLDOUT.csv',holdout)
    save_json(dest/'REPLAY_CASES.json',replay)
    save_json(dest/'SPLIT_DESIGN.json',splits)
    save_json(dest/'TRAINING_MODELS.json',models)
    after = {path:sha(path) for path in before}
    assert before == after
    save_json(dest/'PROTECTED_AFTER.json',after)
    zero = [r for r in rows if r['mode']=='zero']
    status = dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    summary = dict(run_dir=str(dest),wall_seconds=time.perf_counter()-started,
        process_reported_ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        proc_vm_hwm_kib=int(status['VmHWM'].split()[0]),proc_vm_rss_kib=int(status['VmRSS'].split()[0]),
        memory_scope='single current process; proc VmHWM is this address-space high water; ru_maxrss may include pre-exec process lifetime',
        split_version='v2 per-scan spatial stripes; earlier v1 grid split run retained at real-geometry-v7-rlp6r6j5',
        replay_cases=len(replay),replayed_local_fits=len(rows),all_support_exact=True,
        zero_replayed_fits=len(zero),zero_height_failed_cells=sum(r['old_height_gate_failed'] for r in zero),
        zero_height_failed_but_orthogonal_le5_cells=sum(r['old_height_gate_failed'] and r['orthogonal_median_le_5mm'] for r in zero),
        zero_axis_condition_gt10_cells=sum(r['axis_condition']>10 for r in zero),
        holdout_rows=len(holdout),holdout_scored_rows=sum(r['status']=='SCORED' for r in holdout),
        holdout_unfit_rows=sum(r['status']=='UNFIT' for r in holdout),
        heldout_anchor_cells=len(splits),heldout_common_point_count=sum(s['common_test'] for s in splits),
        protected_files=len(before),protected_files_unchanged=True,
        estimator_output_generated=False,independent_geometry_truth=False,
        interpretation='conditional old-state plane metric replay + fixed-design train-only TLS diagnostic; not denoising quality',
        holdout_scope='all-input PCA/cells define spatial design; fitted planes use training XYZ only, no scan bias; fixed-capacity nearest-plane test scoring',
        capacity_note='joint K planes has 3K geometry DOF; two per-scan fits have up to 2K planes and 6K DOF; not equal capacity')
    save_json(dest/'SUMMARY.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
