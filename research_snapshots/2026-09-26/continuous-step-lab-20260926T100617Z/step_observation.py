"""Observation-only line search on a frozen KEEP/A/B route.

Only step length changes. Direct tangent-plane ZNCC is recomputed at physical
intermediate points; no evaluator distances, reference points, or native parent
of an injected case are read. All five grid positions share precisely the same
valid reserved source views. An optional parabolic vertex is scored afresh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

GRID = np.array([0., .25, .5, .75, 1.])
ROUTE_ARM = 'base_shallow__recovery'
PREVIOUS_OUT = Path('/srv/slam-research/grf/map-denoise/runs/visibility-revision-20260926T085529Z')
MIN_COMMON_VIEWS = 2
ARMS = ('grid', 'quadratic')


def common_view_choice(scores):
    """[views, points, five steps] -> common-view costs and smallest-step argmin."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 3 or scores.shape[2] != len(GRID):
        raise ValueError('scores must have shape [views, points, 5]')
    if np.isinf(scores).any():
        raise ValueError('missing scores must be NaN, not infinity')
    common = np.isfinite(scores).all(axis=2)
    count = common.sum(axis=0)
    mean = np.where(common[:, :, None], scores, 0.).sum(axis=0) / np.maximum(count[:, None], 1)
    cost = 1. - mean
    selected = np.argmin(cost, axis=1)
    selected[count < MIN_COMMON_VIEWS] = len(GRID)-1
    return dict(cost=cost, common=common, count=count,
                index=selected, step=GRID[selected])


def parabolic_vertex(cost, index, supported):
    """Convex three-neighbor interpolation around an interior grid winner.

The candidate is not accepted here. Actual photographs must be re-scored before
using it; a fitted parabola alone is not evidence for a better coordinate.
"""
    cost = np.asarray(cost, dtype=float)
    index = np.asarray(index, dtype=int)
    valid = np.asarray(supported, dtype=bool) & (index > 0) & (index < len(GRID) - 1)
    row = np.arange(len(index))
    safe = np.clip(index, 1, len(GRID) - 2)
    left, center, right = cost[row, safe-1], cost[row, safe], cost[row, safe+1]
    curvature = left - 2*center + right
    valid &= np.isfinite(left + center + right) & (curvature > 1e-12)
    delta = np.divide(.5*(left-right)*.25, curvature,
                      out=np.zeros(len(index)), where=valid)
    step = GRID[safe] + delta
    valid &= (step >= GRID[safe-1]) & (step <= GRID[safe+1])
    valid &= np.abs(step - GRID[safe]) > 1e-10
    return step, valid


def revise_path(points, endpoint, normals, ref, sources, scorer=None):
    """Return both actual point outputs and their input-only decision evidence."""
    if scorer is None:
        from v28_closeout.direct_evidence import score_patches
        scorer = score_patches
    points = np.asarray(points, dtype=float)
    endpoint = np.asarray(endpoint, dtype=float)
    if points.shape != endpoint.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points and endpoint must share [N,3] shape')
    rays = points - ref['center']
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    displacement = endpoint - points
    length = np.einsum('ij,ij->i', displacement, rays)
    ray_residual = float(np.max(np.abs(length[:, None]*rays-displacement), initial=0))
    if ray_residual > 1e-8:
        raise ValueError('frozen candidate is not on its current input reference ray')
    ev = scorer(points, normals, length[:, None]*GRID[None, :], ref, sources,
                mode='tangent', patch_radius=3, batch_size=256)
    expected = points[:, None, :] + GRID[None, :, None]*displacement[:, None, :]
    if not np.allclose(ev['candidates'], expected, atol=1e-8, rtol=0):
        raise AssertionError('photometry scored different physical candidate positions')
    choice = common_view_choice(ev['scores'])
    grid_step = choice['step']
    quadratic_step = grid_step.copy()
    candidate_t, trial = parabolic_vertex(choice['cost'], choice['index'],
                                          choice['count'] >= MIN_COMMON_VIEWS)
    accepted = np.zeros(len(points), dtype=bool)
    trial_scores = np.full((len(sources), len(points)), np.nan, dtype=np.float32)
    grid_cost = choice['cost'][np.arange(len(points)), choice['index']]
    selected_cost = grid_cost.copy()
    ids = np.flatnonzero(trial)
    if len(ids):
        fresh = scorer(points[ids], normals[ids], (length[ids]*candidate_t[ids])[:, None],
                       ref, sources, mode='tangent', patch_radius=3, batch_size=256)
        trial_scores[:, ids] = fresh['scores'][:, :, 0]
        common = choice['common'][:, ids]
        same_support = np.all(~common | np.isfinite(trial_scores[:, ids]), axis=0)
        trial_cost = 1. - np.where(common, trial_scores[:, ids], 0.).sum(axis=0) / choice['count'][ids]
        # No threshold fitted to geometry: accept only actual lower photo cost.
        improve = same_support & (trial_cost < grid_cost[ids])
        accepted[ids[improve]] = True
        quadratic_step[ids[improve]] = candidate_t[ids[improve]]
        selected_cost[ids[improve]] = trial_cost[improve]
    return dict(
        step_grid=points + grid_step[:, None]*displacement,
        step_quadratic=points + quadratic_step[:, None]*displacement,
        lambda_grid=grid_step, lambda_quadratic=quadratic_step,
        grid_scores=ev['scores'], common_views=choice['common'], common_count=choice['count'],
        grid_cost=grid_cost, selected_quadratic_cost=selected_cost,
        quadratic_tried=trial, quadratic_accepted=accepted,
        quadratic_trial_step=candidate_t, quadratic_trial_scores=trial_scores,
        ray_residual_mm=ray_residual)


def load_cameras(sid, view_records, context):
    """Calibration and selected photographs only; never load_scene/native mesh."""
    from scene_adapter import colmap, camera
    folder = context.DATA/'closeout-confirmation-v1/inputs'/f'scan{sid}'
    calibration, sparse = folder/'cameras.npz', folder/'sparse/0'
    with np.load(calibration, allow_pickle=False) as z:
        inverse = np.linalg.inv(z['scale_mat_0'].astype(float))
    raw, _ = colmap(sparse)
    names = sorted({n for v in view_records.values()
                    for n in [v['reference'], *v['reserved_views']]})
    cams = {}
    paths = [calibration, *sorted(sparse.glob('*.bin'))]
    for name in names:
        original = raw[name]
        P = original['P'] @ inverse
        path = folder/'images'/name
        expected = {v['image_sha256'][name] for v in view_records.values()
                    if name in v['image_sha256']}
        if expected != {context.sha(path)}:
            raise AssertionError('frozen photograph hash changed')
        cams[name] = camera(dict(P=P, center=-np.linalg.solve(P[:,:3], P[:,3]),
                                width=original['width'], height=original['height'], path=path))
        paths.append(path)
    return cams, paths


def main():
    import common as context
    from v28_closeout.graph_field import estimate_normals
    context.check_host()
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', type=int, required=True, choices=(55,65,69))
    ap.add_argument('--case', help='optional exact case for smoke; writes separate smoke subtree')
    args = ap.parse_args()
    cases = context.cases_for_scene(args.scene)
    if args.case:
        cases = [case for case in cases if case.name == args.case]
        if len(cases) != 1:
            raise ValueError('requested exact case not found')
    out = context.OUT/('observation_smoke' if args.case else 'inference')/f'scan{args.scene}'
    out.mkdir(parents=True, exist_ok=False)
    views_file = context.RESERVED/'evidence'/f'scan{args.scene}'/'VIEWS.json'
    view_records = json.loads(views_file.read_text())
    cams, camera_files = load_cameras(args.scene, view_records, context)
    source_files = [Path(__file__), Path(__file__).with_name('test_step_observation.py'),
                    context.ROOT/'common.py', context.ROOT/'PROTOCOL.md', views_file,
                    context.CLOSEOUT/'scene_adapter.py',
                    context.CLOSEOUT/'package/v28_closeout/direct_evidence.py',
                    context.CLOSEOUT/'package/v28_closeout/graph_field.py', *camera_files]
    sources = {str(p): context.sha(p) for p in source_files}
    context.save_json(out/'LOCK.json', dict(source_sha256=sources, route=ROUTE_ARM,
        grid=GRID.tolist(), min_common_views=MIN_COMMON_VIEWS, reference_access=False,
        current_case_only=True, no_training=True, shared_source_view_set=True,
        comparator='same frozen recovery route at full step; only step changes'))
    records = []
    start = time.monotonic()
    for case in cases:
        tick = time.monotonic()
        dest = out/case.name
        dest.mkdir()
        hashes = {}
        geometry, route = context.geometry_and_route(case, hashes)
        sources.update(hashes)
        p = geometry[:,0]
        route_file = PREVIOUS_OUT/'inference'/f'scan{args.scene}'/case.name/'routes.npz'
        endpoint = context.selected_endpoint(geometry,route)
        active = np.flatnonzero((route != 0) & (np.linalg.norm(endpoint-p, axis=1) > 1e-7))
        normals = estimate_normals(p,24)
        v = view_records[case.name.split('__')[0]]
        if set(v['reserved_views']) & set(v['original_views']):
            raise AssertionError('reserved views overlap candidate construction')
        result = revise_path(p[active], endpoint[active], normals[active], cams[v['reference']],
                             [cams[n] for n in v['reserved_views']])
        arrays = {k:val for k,val in result.items() if isinstance(val,np.ndarray)}
        arrays.update(row_ids=active, route=route)
        for arm in ARMS:
            full_step = np.ones(len(p), dtype=float)
            full_step[active] = result['lambda_'+arm]
            arrays[arm] = full_step
        context.save_npz(dest/'STEPS.npz', **arrays)
        for arm in ARMS:
            output = p.copy()
            output[active] = result['step_'+arm]
            context.write_points(dest/(arm+'.ply'), output)
        record = dict(case=case.name, rows=len(p), active_rows=len(active),
            source_sha256=hashes, route_sha256=context.sha(route_file),
            reference_access=False, current_case_normals=True,
            common_two_view_rows=int(np.sum(result['common_count'] >= MIN_COMMON_VIEWS)),
            lambda_grid_counts=np.bincount(np.rint(result['lambda_grid']*4).astype(int), minlength=5).tolist(),
            quadratic_trial_rows=int(result['quadratic_tried'].sum()),
            quadratic_accepted_rows=int(result['quadratic_accepted'].sum()),
            max_ray_residual_mm=result['ray_residual_mm'], seconds=time.monotonic()-tick)
        context.save_json(dest/'META.json',record)
        records.append(record)
        print('STEP_OBSERVATION',case.name,len(active),'active',round(record['seconds'],2),'s',flush=True)
    if any(context.sha(path) != h for path,h in sources.items()):
        raise AssertionError('observation source changed during execution')
    context.save_json(out/'SUMMARY.json',dict(records=records,seconds=time.monotonic()-start,
        peak_own_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpu=False))
    context.seal(out,sources)
    print('OBSERVATION SEALED',args.scene,flush=True)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
