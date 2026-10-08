"""Candidate-blind, center-anchored image supports. No truth or candidates enter.

A connected intensity mask and a frontoparallel warp are hypotheses, NOT
physical surface labels. Scores and their finite support are the shared object.
"""
from collections import deque
import numpy as np
from scipy.ndimage import map_coordinates

RADIUS = 4
EDGE_MAX = 20.0
MIN_PIXELS = 9
STD_MIN = 3.0
NCC_MIN = 0.6


def camera(data):
    return dict(K=np.asarray(data['K_half'], float), R=np.asarray(data['R'], float),
                C=np.asarray(data['center'], float))


def project(cam, points):
    xyz = (np.asarray(points)-cam['C']) @ cam['R'].T
    uvw = xyz @ cam['K'].T
    uv = np.divide(uvw[..., :2], uvw[..., 2:], out=np.full_like(uvw[..., :2], np.nan),
                   where=np.abs(uvw[..., 2:]) > 1e-12)
    return uv, xyz[..., 2]


def rays(cam, pixels):
    p = np.asarray(pixels, float)
    return np.concatenate((p, np.ones(p.shape[:-1]+(1,))), axis=-1) @ np.linalg.inv(cam['K']).T @ cam['R']


def sample(image, uv):
    uv = np.asarray(uv, float)
    good = np.isfinite(uv).all(axis=-1)
    good &= (uv[..., 0] >= 0) & (uv[..., 0] <= image.shape[1]-1)
    good &= (uv[..., 1] >= 0) & (uv[..., 1] <= image.shape[0]-1)
    clean = np.where(np.isfinite(uv), uv, 0.)
    values = map_coordinates(image, [clean[..., 1], clean[..., 0]], order=1, mode='constant', cval=0.)
    return values, good


def center_mask(patch, kind):
    if np.asarray(patch).shape != (9, 9):
        raise ValueError('expected 9x9 patch')
    if kind == 'full9':
        return np.ones((9, 9), bool)
    if kind == 'center3':
        mask = np.zeros((9, 9), bool)
        mask[3:6, 3:6] = True
        return mask
    if kind != 'connected9':
        raise ValueError('unknown mask')
    mask = np.zeros((9, 9), bool)
    mask[4, 4] = True
    todo = deque([(4, 4)])
    while todo:
        y, x = todo.popleft()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < 9 and 0 <= xx < 9 and not mask[yy, xx]:
                if abs(float(patch[yy, xx])-float(patch[y, x])) <= EDGE_MAX:
                    mask[yy, xx] = True
                    todo.append((yy, xx))
    return mask


def score_support(ref_image, source_image, ref_cam, source_cam, xy, grid, warp, mask_kind):
    grid = np.asarray(grid, float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any():
        raise ValueError('finite increasing 1D depth grid required')
    if warp not in ('translation', 'plane'):
        raise ValueError('unknown warp')
    yy, xx = np.mgrid[-4:5, -4:5]
    offsets = np.column_stack((xx.ravel(), yy.ravel()))
    ref_uv = np.asarray(xy)[None, :]+offsets
    patch, ref_good = sample(ref_image, ref_uv)
    mask = center_mask(patch.reshape(9, 9), mask_kind)
    use = mask.ravel()
    count = int(use.sum())
    scores = np.full(len(grid), np.nan)
    a = patch[use]
    a = a-a.mean()
    std_a = float(np.sqrt(np.mean(a*a)))
    base = dict(scores=scores, mask=mask, mask_count=count, anchor_std=std_a,
                valid_count=0, center_selected=bool(mask[4, 4]), warp=warp, mask_kind=mask_kind)
    if count < MIN_PIXELS or not ref_good[use].all() or std_a < STD_MIN:
        return base
    if warp == 'translation':
        center_world = ref_cam['C']+grid[:, None]*rays(ref_cam, np.asarray(xy))
        centers, z = project(source_cam, center_world)
        uv = centers[:, None, :]+offsets[None, use, :]
        positive = (z > 0) & (grid > 0)
    else:
        # Every selected reference pixel intersects the SAME frontoparallel
        # plane at optical Z=grid[z]; this is not a candidate-dependent plane.
        world = ref_cam['C']+grid[:, None, None]*rays(ref_cam, ref_uv[use])[None, :, :]
        uv, z = project(source_cam, world)
        positive = (z > 0).all(axis=1) & (grid > 0)
    b, source_good = sample(source_image, uv)
    b = b-b.mean(axis=1, keepdims=True)
    std_b = np.sqrt(np.mean(b*b, axis=1))
    valid = source_good.all(axis=1) & positive & (std_b >= STD_MIN)
    scores[valid] = (b[valid] @ a)/(count*std_a*std_b[valid])
    base['valid_count'] = int(valid.sum())
    return base


def merge_intervals(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]+1e-9:
            out[-1][1] = max(out[-1][1], float(hi))
        else:
            out.append([float(lo), float(hi)])
    return out


def support_intervals(grid, mask, padding=1.0):
    grid = np.asarray(grid, float)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    chunks = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
    half = (grid[1]-grid[0])/2 if len(grid) > 1 else 0.
    return merge_intervals([[max(grid[0], grid[c[0]]-half-padding),
                             min(grid[-1], grid[c[-1]]+half+padding)] for c in chunks])


def intersection(a, b):
    return merge_intervals([[max(x[0], y[0]), min(x[1], y[1])] for x in a for y in b
                            if max(x[0], y[0]) <= min(x[1], y[1])])
