"""Depth intervals obtained by eliminating bounded photometric nuisances.

Only the public forward-model geometry constants are imported.  There is no
development-data dependency in this module.
"""
import math
import time
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning

from forward import PIXELS, PIXEL_AREA, FRONT, LEFT, RIGHT, REARS, BRANCHES


DEPTH_RANGE = (540.0, 660.0)
CELL_TOLERANCE = 0.025
LP_SLACK = 2e-8


def _project_enclosure(rect, depths, cameras):
    """Inner and outer axis-aligned rectangles enclosing every projection."""
    vals = []
    for i, edge in enumerate(rect):
        ns = (edge-cameras[1], edge-cameras[0]) if i < 2 else (edge, edge)
        candidates = [n/z for n in ns for z in depths]
        vals.append((min(candidates), max(candidates)))
    outer = (vals[0][0], vals[1][1], vals[2][0], vals[3][1])
    inner = (vals[0][1], vals[1][0], vals[2][1], vals[3][0])
    return inner, outer


def _area(rect):
    return (np.maximum(0., np.minimum(rect[1], PIXELS[:, 1])-
                       np.maximum(rect[0], PIXELS[:, 0])) *
            np.maximum(0., np.minimum(rect[3], PIXELS[:, 3])-
                       np.maximum(rect[2], PIXELS[:, 2]))) / PIXEL_AREA


def _intersect(a, b):
    return (max(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), min(a[3], b[3]))


def _area_enclosures(depths, branch, cameras):
    lower = np.zeros((3, 48, 3))
    upper = np.zeros_like(lower)
    for v, camera in enumerate(cameras):
        fi, fo = _project_enclosure(FRONT, depths, camera)
        for k, rect in enumerate((LEFT, RIGHT)):
            ri, ro = _project_enclosure(rect, depths, camera)
            lower[v, :, k], upper[v, :, k] = _area(ri), _area(ro)
        if REARS[branch] is not None:
            ri, ro = _project_enclosure(REARS[branch], (720., 720.), camera)
            # Set inclusion: ri\fo <= R\F <= ro\fi.  Using the same
            # enclosing rear in both terms keeps camera translations coupled.
            lower[v, :, 2] = _area(ri)-_area(_intersect(ri, fo))
            upper[v, :, 2] = _area(ro)-_area(_intersect(ro, fi))
    return np.clip(lower-1e-12, 0., 1.), np.clip(upper+1e-12, 0., 1.)


def _camera_intervals(camera):
    centers = np.array([-80., 0., 80.])
    u = np.asarray(camera['u'], dtype=float).reshape(3)
    zf, xf = float(camera.get('Zf', 1000.)), float(camera.get('Xf', 0.))
    survey = float(camera.get('survey_bound', .01))
    local = float(camera.get('localization_bound', .01))
    residual = xf-zf*u-centers
    klo, khi = max(-survey, residual[1]-local), min(survey, residual[1]+local)
    if klo > khi+1e-10:
        raise ValueError('Inconsistent middle-camera calibration')
    intervals = []
    for v in range(3):
        if v == 1:
            intervals.append((0., 0.))
        else:
            elo = max(-.2, residual[v]-khi-local)-1e-12
            ehi = min(.2, residual[v]-klo+local)+1e-12
            if elo > ehi:
                raise ValueError('Inconsistent side-camera calibration')
            intervals.append((centers[v]+elo, centers[v]+ehi))
    return intervals


def _encode(observation, shared_materials):
    photo = np.asarray(observation['photo'], dtype=float).reshape(3, 48, 3)
    low = np.max(np.where(photo > 0., photo-.01, -np.inf), axis=2)
    high = np.min(np.where(photo < 1., photo+.01, np.inf), axis=2)
    ap = observation['appearance']
    coupons = np.asarray(ap['coupons'], dtype=float).reshape(3, 3)
    references = np.asarray(ap['references'], dtype=float).reshape(3)
    noise = float(ap.get('noise_bound', .005))
    transfer = float(ap.get('transfer_bound', .005))
    bias = float(ap.get('bias_bound', .01))
    bounds = [(0., 1.)]*9 + [(-.01, .01)]*3
    rows, rhs = [], []
    for v in range(3):
        ref = references[v]
        blo = max(-bias, ref-.5-noise) if ref > 0. else -bias
        bhi = min(bias, ref-.5+noise) if ref < 1. else bias
        if blo > bhi+1e-10:
            raise ValueError('Inconsistent reference measurement')
        bounds.append((blo-1e-12, bhi+1e-12))
        for k in range(3):
            coupon = coupons[v, k]
            row = np.zeros(15)
            row[v*3+k], row[12+v] = 1., -1.
            if coupon > 0.:
                rows.append(row)
                rhs.append(1.-coupon+noise+transfer)
            if coupon < 1.:
                rows.append(-row)
                rhs.append(coupon-1.+noise+transfer)
    if shared_materials:
        for k in range(3):
            for v in range(3):
                for w in range(v):
                    row = np.zeros(15)
                    row[v*3+k], row[w*3+k] = 1., -1.
                    rows.extend((row, -row))
                    rhs.extend((.1, .1))
    fixed_a = np.asarray(rows, dtype=float).reshape(-1, 15)
    fixed_b = np.asarray(rhs, dtype=float)
    return low, high, bounds, fixed_a, fixed_b, _camera_intervals(observation['camera'])


def _lp_for_cell(encoded, branch, lo, hi, time_limit):
    low, high, bounds, fixed_a, fixed_b, cameras = encoded
    amin, amax = _area_enclosures((lo, hi), branch, cameras)
    rows, rhs = [fixed_a], [fixed_b]
    for v in range(3):
        mask = np.isfinite(low[v])
        a = np.zeros((int(mask.sum()), 15))
        a[:, v*3:v*3+3] = amin[v, mask]
        a[:, 9+v] = -1.
        rows.append(a)
        rhs.append(1.-low[v, mask])
        mask = np.isfinite(high[v])
        a = np.zeros((int(mask.sum()), 15))
        a[:, v*3:v*3+3] = -amax[v, mask]
        a[:, 9+v] = 1.
        rows.append(a)
        rhs.append(high[v, mask]-1.)
    # SciPy forwards this HiGHS-native option even on releases whose wrapper
    # does not list it.  Environment variables alone need not limit HiGHS.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected:.*',
                                category=OptimizeWarning)
        return linprog(np.zeros(15), A_ub=np.concatenate(rows),
                       b_ub=np.concatenate(rhs)+LP_SLACK, bounds=bounds,
                       method='highs', options={'time_limit': time_limit, 'threads': 1,
                                               'primal_feasibility_tolerance': 1e-8,
                                               'dual_feasibility_tolerance': 1e-8})


def _prepare(observation, budget_seconds, shared_materials):
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    budget = max(0., float(budget_seconds))
    deadline = wall_start+budget
    stats = {'lp_calls': 0, 'infeasible_cells': 0, 'splits': 0,
             'atomic_over_025_seconds': [], 'solver_failures': []}
    # Each box is [branch, lower, upper, checked, terminal].  Unchecked boxes
    # always remain in the returned enclosure if the clock interrupts work.
    boxes = [[b, 540., 660., False, False] for b in BRANCHES]
    reason = 'budget'
    try:
        encoded = _encode(observation, shared_materials)
        side = 0
        while boxes and time.perf_counter() < deadline-.04:
            extrema = [min(boxes, key=lambda b: (b[1], b[4], b[2])),
                       max(boxes, key=lambda b: (b[2], -int(b[4]), -b[1]))]
            done = [x[4] for x in extrema]
            if all(done):
                reason = 'boundary_resolution'
                break
            if done[side]:
                side = 1-side
            cell = extrema[side]
            side = 1-side
            if not cell[3]:
                remaining = deadline-time.perf_counter()-.025
                if remaining <= .01:
                    break
                atomic_start = time.perf_counter()
                result = _lp_for_cell(encoded, cell[0], cell[1], cell[2],
                                      max(.001, min(.20, remaining)))
                elapsed = time.perf_counter()-atomic_start
                stats['lp_calls'] += 1
                if elapsed > .25:
                    stats['atomic_over_025_seconds'].append(elapsed)
                if result.status == 2:
                    boxes.remove(cell)
                    stats['infeasible_cells'] += 1
                    continue
                cell[3] = True
                if result.status != 0:
                    stats['solver_failures'].append({'status': int(result.status),
                                                     'seconds': elapsed})
                    cell[4] = True
                    continue
            if cell[2]-cell[1] <= CELL_TOLERANCE:
                cell[4] = True
                continue
            # Bisect reciprocal depth, the coordinate in which perspective
            # rectangle edges are affine for a fixed camera.
            middle = 2./(1./cell[1]+1./cell[2])
            boxes.remove(cell)
            boxes.extend(([cell[0], cell[1], middle, False, False],
                          [cell[0], middle, cell[2], False, False]))
            stats['splits'] += 1
    except Exception as exc:
        stats['error'] = type(exc).__name__+': '+str(exc)
        reason = 'encoding_or_solver_error'
        boxes = [[b, 540., 660., False, False] for b in BRANCHES]
    if boxes:
        lo, hi = min(b[1] for b in boxes), max(b[2] for b in boxes)
    else:
        # Incompatible data or numerical rejection should never force a
        # confident repair.  Identity is the explicitly recorded fallback.
        lo, hi, reason = 540., 660., 'empty_enclosure_identity_fallback'
    return {'method': 'silhouette_enclosure_lp_hull',
            'shared_materials': bool(shared_materials),
            'lower': float(lo), 'upper': float(hi),
            'cells': [{'branch': b[0], 'lower': float(b[1]), 'upper': float(b[2]),
                       'checked': bool(b[3]), 'terminal': bool(b[4])} for b in boxes],
            'reason': reason, 'stats': stats,
            'wall_seconds': time.perf_counter()-wall_start,
            'cpu_seconds': time.process_time()-cpu_start,
            'requested_budget_seconds': budget}


def prepare(observation, budget_seconds):
    return _prepare(observation, budget_seconds, shared_materials=True)


def prepare_ablation(observation, budget_seconds):
    """Frozen structure ablation: omit cross-view material range constraints."""
    return _prepare(observation, budget_seconds, shared_materials=False)


def decode(state, incumbent):
    value = float(incumbent)
    if not math.isfinite(value):
        value = (float(state['lower'])+float(state['upper']))/2.
    return float(np.clip(value, max(540., float(state['lower'])),
                         min(660., float(state['upper']))))
