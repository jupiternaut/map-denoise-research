"""Direct, GT-free multi-candidate photometry and rendered-field visibility.

Visibility is a candidate-conditioned self-rendering proxy, not independently
measured depth. Unknown support has unit visibility weight. The common source
set is frozen across all candidates before either cost aggregation is applied.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

MIN_VIEWS = 2
VISIBILITY_SCALE_MM = .75
CONTINUOUS = Path('/home/grf/Documents/Codex/2026-09-26/continuous-step-lab-20260926T100617Z')
VISIBILITY = Path('/home/grf/Documents/Codex/2026-09-26/visibility-revision-lab-20260926T085529Z')


def _module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def load_cameras(sid, view_records, context):
    """Reuse frozen calibration/photo loader; no native mesh or laser access."""
    old = _module('field_frozen_observation_loader', CONTINUOUS/'step_observation.py')
    return old.load_cameras(sid, view_records, context)


def candidate_offsets(points, a, b, ref_center):
    """Nine ray offsets: KEEP,A,B,half-A,half-B,-6,-3,+3,+6 mm."""
    points = np.asarray(points, float)
    rays = points - np.asarray(ref_center)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    offsets = []
    for candidate in (a,b):
        d = np.asarray(candidate)-points
        t = np.sum(d*rays, axis=1)
        if np.max(np.abs(d-t[:,None]*rays), initial=0.) > 1e-8:
            raise ValueError('archived candidate is not on the current reference ray')
        offsets.append(t)
    ta,tb = offsets
    return np.column_stack([np.zeros(len(points)),ta,tb,.5*ta,.5*tb,
                            np.full((len(points),4),[-6.,-3.,3.,6.])])


def common_cost(scores):
    """All candidates use one identical set of finite source-view scores."""
    scores = np.asarray(scores,float)
    if scores.ndim != 3 or scores.shape[2] < 1 or np.isinf(scores).any():
        raise ValueError('scores must be [sources,points,candidates], NaN for missing')
    common = np.isfinite(scores).all(axis=2)
    count = common.sum(axis=0)
    supported = count >= MIN_VIEWS
    mean = np.where(common[:,:,None],scores,0.).sum(axis=0) / np.maximum(count[:,None],1)
    cost = 1.-mean
    cost[~supported] = 1.
    return dict(cost=cost, common_valid=common, common_count=count, supported=supported)


def score_positions(points, offsets, normals, ref, views, scorer=None):
    """Score actual ray coordinates, tangent PCA normal, full 7x7 image patch.

    Returns scores[S,N,C], candidates[N,C,3], cost[N,C], common_valid[S,N],
    common_count[N], supported[N], and scorer ref_valid/directions. Unsupported
    costs are neutral1, not evidence; caller must preserve those rows explicitly.
    """
    if scorer is None:
        from v28_closeout.direct_evidence import score_patches
        scorer = score_patches
    evidence = scorer(points,normals,offsets,ref,views,mode='tangent',
                      patch_radius=3,batch_size=256)
    return {**evidence, **common_cost(evidence['scores'])}


def weighted_common_cost(scores, gap_mm, common=None):
    """Mean photo cost weighted by exp(-positive_depth_gap/.75 mm).

    Stabilized normalization is mathematically the same weighted mean, including
    when all raw weights are tiny. Missing source views never become evidence.
    """
    scores = np.asarray(scores,float)
    gap = np.asarray(gap_mm,float)
    if gap.shape != scores.shape or not np.isfinite(gap).all():
        raise ValueError('finite gaps and scores must share [S,N,C] shape')
    base = common_cost(scores)
    if common is None:
        common = base['common_valid']
    elif not np.array_equal(common,base['common_valid']):
        raise ValueError('visibility may not change the common photometric view set')
    logweight = -np.maximum(gap,0.)/VISIBILITY_SCALE_MM
    valid = common[:,:,None]
    maximum = np.max(np.where(valid,logweight,-np.inf),axis=0)
    maximum = np.where(np.isfinite(maximum),maximum,0.)
    stable = np.where(valid,np.exp(np.where(valid,logweight-maximum[None],-np.inf)),0.)
    denom = stable.sum(axis=0)
    mean = np.divide(np.where(valid,scores,0.)*stable,denom[None],
                     out=np.zeros_like(scores),where=denom[None]>0).sum(axis=0)
    cost = 1.-mean
    cost[~base['supported']] = 1.
    raw_weights = np.exp(logweight)
    normalized_weights = np.divide(stable,denom[None],out=np.zeros_like(stable),where=denom[None]>0)
    return dict(cost=cost,weights=raw_weights,normalized_weights=normalized_weights,**{
        name:base[name] for name in ('common_valid','common_count','supported')})


def _raster_camera(camera):
    if 'width' in camera and 'height' in camera:
        return camera
    height,width = np.asarray(camera['image']).shape[:2]
    return {**camera,'width':width,'height':height}


def visibility_cost(scores, candidates, render_points, views, chunk=8192):
    """Reweight identical scores with a single current output field's z-buffers.

    candidates[N,C,3] are queried against render_points[N,3], one current surface
    point per input row. Every source has a one-pixel nearest-depth z-buffer. A
    3x3 fit of inverse depth at actual subpixel coordinates provides planar gap;
    if unavailable, use known center depth. If neither is known, gap=0 and weight1.
    Self-front and unknown flags remain explicit evidence, never treated as an
    independent observation. Original row identity is preserved for self flags.
    """
    candidates = np.asarray(candidates,float)
    render_points = np.asarray(render_points,float)
    n,c,_ = candidates.shape
    if render_points.shape != (n,3) or np.asarray(scores).shape != (len(views),n,c):
        raise ValueError('render field, candidate and score rows must align')
    helper = _module('field_frozen_visibility',VISIBILITY/'visibility_features.py')
    shape=(len(views),n,c)
    gap=np.zeros(shape,np.float32)
    known=np.zeros(shape,bool)
    self_front=np.zeros(shape,bool)
    plane_known=np.zeros(shape,bool)
    valid_projection=np.zeros(shape,bool)
    for source_index,view in enumerate(views):
        raster=helper.rasterize(render_points,_raster_camera(view))
        for start in range(0,n,chunk):
            stop=min(start+chunk,n)
            rows=np.repeat(np.arange(start,stop),c)
            support=helper.query_support(candidates[start:stop].reshape(-1,3),rows,raster)
            current_known=support['fit']|support['known']
            signed=np.where(support['fit'],support['planegap'],support['centergap'])
            gap[source_index,start:stop]=np.where(current_known,signed,0.).reshape(stop-start,c)
            known[source_index,start:stop]=current_known.reshape(stop-start,c)
            self_front[source_index,start:stop]=support['self_front'].reshape(stop-start,c)
            plane_known[source_index,start:stop]=support['fit'].reshape(stop-start,c)
            valid_projection[source_index,start:stop]=support['valid'].reshape(stop-start,c)
    result=weighted_common_cost(scores,gap)
    return {**result,'gap_mm':gap,'known':known,'unknown':~known,
            'plane_known':plane_known,'self_front':self_front,'project_valid':valid_projection,
            'evidence_kind':'candidate-conditioned self-rendered depth proxy'}
