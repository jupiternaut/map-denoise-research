"""A13 exact rational necessary constraints and independently checkable exclusions.

Only the three public observation fields are read. Float optimization proposes
multipliers; every accepted contradiction is evaluated with Fraction arithmetic.
No historical solver, generator, renderer, development data, or labels imported.
"""
from __future__ import annotations

from fractions import Fraction as Q
import hashlib
import json
import math
import time
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning


BRANCHES = ('none', 'partial', 'full')
FRONT = (-20, 20, -15, 15)
LEFT = (-20, 0, -15, 15)
RIGHT = (0, 20, -15, 15)
REARS = {'none': None, 'partial': (-20, 36, -17, 23),
         'full': (-100, 100, -80, 80)}
PIXELS = tuple((Q(-1, 5)+Q(j, 20), Q(-1, 5)+Q(j+1, 20),
                Q(-2, 25)+Q(2*i, 75), Q(-2, 25)+Q(2*(i+1), 75))
               for i in range(6) for j in range(8))
PIXEL_AREA = Q(1, 750)
VARIABLE_ORDER = tuple([f'd{v}_{k}' for v in range(3) for k in range(3)]
                       + [f'b{v}' for v in range(3)]
                       + [f'ba{v}' for v in range(3)])
SCHEMA = 'a13-rational-box-farkas-v1'
ZERO, ONE = Q(0), Q(1)
PHOTO_NOISE, BIAS, ANCHOR_NOISE = Q(1, 100), Q(1, 100), Q(1, 200)


def fraction_text(value):
    value = Q(value)
    return f'{value.numerator}/{value.denominator}'


def observation_digest(observation):
    selected = {key: observation[key] for key in ('photo', 'camera', 'appearance')}
    return hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def rounded_interval(value, clipped=False):
    """Closed binary64 round-to-nearest preimage; includes both tie endpoints.

    Generator public_packet calls float(exact_rational); JSON's shortest decimal
    round trip returns that binary64. This interval includes the original rational.
    Physical constants are specified separately as exact task rationals.
    """
    x = float(value)
    if not math.isfinite(x):
        raise ValueError('nonfinite measurement')
    if clipped and not 0. <= x <= 1.:
        raise ValueError('clipped measurement outside [0,1]')
    previous, following = math.nextafter(x, -math.inf), math.nextafter(x, math.inf)
    if not math.isfinite(previous) or not math.isfinite(following):
        raise ValueError('unbounded rounding bin')
    center = Q.from_float(x)
    lo = (Q.from_float(previous)+center)/2
    hi = (center+Q.from_float(following))/2
    return (max(ZERO, lo), min(ONE, hi)) if clipped else (lo, hi)


def _flatten(value, count):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != count:
        raise ValueError(f'expected {count} measurement values')
    return tuple(float(x) for x in arr)


def _metadata(record, expected):
    for name, value in expected.items():
        if float(record.get(name, value)) != value:
            raise ValueError(f'unsupported frozen model metadata: {name}')


def project_enclosure(rect, depths, cameras):
    """Inner/outer rectangles over positive depth and camera intervals."""
    values = []
    for i, edge in enumerate(rect):
        numerators = (Q(edge)-cameras[1], Q(edge)-cameras[0]) if i < 2 else (Q(edge),)
        candidates = tuple(n / z for n in numerators for z in depths)
        values.append((min(candidates), max(candidates)))
    inner = (values[0][1], values[1][0], values[2][1], values[3][0])
    outer = (values[0][0], values[1][1], values[2][0], values[3][1])
    return inner, outer


def intersect(a, b):
    return (max(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), min(a[3], b[3]))


def pixel_area(rect, pixel):
    return (max(ZERO, min(rect[1], pixel[1])-max(rect[0], pixel[0])) *
            max(ZERO, min(rect[3], pixel[3])-max(rect[2], pixel[2]))) / PIXEL_AREA


def area_enclosures(branch, lo, hi, cameras, deadline=None):
    """Exact A12 free rectangle bounds, with no Lipschitz strengthening."""
    if branch not in BRANCHES or not Q(540) <= lo <= hi <= Q(660):
        raise ValueError('invalid branch or cell')
    result = []
    for camera in cameras:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('deadline during geometry')
        fi, fo = project_enclosure(FRONT, (lo, hi), camera)
        li, lout = project_enclosure(LEFT, (lo, hi), camera)
        ri, rout = project_enclosure(RIGHT, (lo, hi), camera)
        if REARS[branch] is not None:
            bi, bo = project_enclosure(REARS[branch], (Q(720), Q(720)), camera)
            lower_cut, upper_cut = intersect(bi, fo), intersect(bo, fi)
        view = []
        for pixel in PIXELS:
            lower = [pixel_area(li, pixel), pixel_area(ri, pixel), ZERO]
            upper = [pixel_area(lout, pixel), pixel_area(rout, pixel), ZERO]
            if REARS[branch] is not None:
                lower[2] = pixel_area(bi, pixel)-pixel_area(lower_cut, pixel)
                upper[2] = pixel_area(bo, pixel)-pixel_area(upper_cut, pixel)
            if any(not ZERO <= a <= b <= ONE for a, b in zip(lower, upper)):
                raise ArithmeticError('invalid exact area enclosure')
            view.append((tuple(lower), tuple(upper)))
        result.append(tuple(view))
    return tuple(result)


def box_margin(rows, rhs, bounds, multipliers, deadline=None):
    """min_box (sum_i y_i A_i)x - sum_i y_i b_i, y >= 0.

    Rows are sparse coefficient dictionaries. No numerical tolerance is used.
    A positive margin contradicts every point in the explicit nuisance box.
    """
    total = [ZERO]*15
    beta = ZERO
    seen = set()
    for index, weight in multipliers:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('deadline during exact check')
        if (type(index) is not int or index in seen or not 0 <= index < len(rows)
                or not isinstance(weight, Q) or weight <= ZERO):
            raise ValueError('invalid sparse multiplier')
        seen.add(index)
        beta += weight * rhs[index]
        for variable, coefficient in rows[index].items():
            total[variable] += weight * coefficient
    if not seen or len(bounds) != 15 or any(a > b for a, b in bounds):
        raise ValueError('empty multiplier or invalid box')
    return sum((coefficient*(bounds[j][0] if coefficient >= 0 else bounds[j][1])
                for j, coefficient in enumerate(total)), ZERO)-beta


class ExactContext:
    """Observation-only context. Invalid input is retained as an unknown result."""

    def __init__(self, observation):
        started, cpu = time.perf_counter(), time.process_time()
        self.error = None
        try:
            self.digest = observation_digest(observation)
            camera, appearance = observation['camera'], observation['appearance']
            _metadata(camera, {'Xf': 0., 'Zf': 1000., 'survey_bound': .01,
                               'localization_bound': .01})
            _metadata(appearance, {'bias_bound': .01, 'transfer_bound': .005,
                                   'noise_bound': .005})
            us = [rounded_interval(x) for x in _flatten(camera['u'], 3)]
            centers = (-80, 0, 80)
            residuals = [(-1000*uhi-centers[v], -1000*ulo-centers[v])
                         for v, (ulo, uhi) in enumerate(us)]
            klo = max(-BIAS, residuals[1][0]-BIAS)
            khi = min(BIAS, residuals[1][1]+BIAS)
            if klo > khi:
                raise ValueError('inconsistent middle camera')
            cameras = []
            for v, center in enumerate(centers):
                elo = max(-Q(1, 5), residuals[v][0]-khi-BIAS) if v != 1 else ZERO
                ehi = min(Q(1, 5), residuals[v][1]-klo+BIAS) if v != 1 else ZERO
                if elo > ehi:
                    raise ValueError('inconsistent side camera')
                cameras.append((Q(center)+elo, Q(center)+ehi))
            self.cameras = tuple(cameras)
            photos = [rounded_interval(x, clipped=True)
                      for x in _flatten(observation['photo'], 432)]
            low, high = [], []
            for v in range(3):
                low_view, high_view = [], []
                for p in range(48):
                    channels = photos[144*v+3*p:144*v+3*p+3]
                    lows = [a-PHOTO_NOISE for a, b in channels if a > ZERO]
                    highs = [b+PHOTO_NOISE for a, b in channels if b < ONE]
                    low_view.append(max(lows) if lows else None)
                    high_view.append(min(highs) if highs else None)
                low.append(tuple(low_view)); high.append(tuple(high_view))
            self.low, self.high = tuple(low), tuple(high)
            coupons = [rounded_interval(x, clipped=True)
                       for x in _flatten(appearance['coupons'], 9)]
            references = [rounded_interval(x, clipped=True)
                          for x in _flatten(appearance['references'], 3)]
            bounds = [(ZERO, ONE)]*9+[(-BIAS, BIAS)]*3
            rows, rhs = [], []
            for v, (reflo, refhi) in enumerate(references):
                blo = max(-BIAS, reflo-Q(1, 2)-ANCHOR_NOISE) if reflo > ZERO else -BIAS
                bhi = min(BIAS, refhi-Q(1, 2)+ANCHOR_NOISE) if refhi < ONE else BIAS
                if blo > bhi:
                    raise ValueError('inconsistent appearance reference')
                bounds.append((blo, bhi))
                for k in range(3):
                    qlo, qhi = coupons[3*v+k]
                    if qlo > ZERO:
                        rows.append({3*v+k: ONE, 12+v: -ONE})
                        rhs.append(ONE-qlo+2*ANCHOR_NOISE)
                    if qhi < ONE:
                        rows.append({3*v+k: -ONE, 12+v: ONE})
                        rhs.append(qhi-ONE+2*ANCHOR_NOISE)
            for k in range(3):
                for v in range(3):
                    for w in range(v):
                        rows.extend(({3*v+k: ONE, 3*w+k: -ONE},
                                     {3*v+k: -ONE, 3*w+k: ONE}))
                        rhs.extend((Q(1, 10), Q(1, 10)))
            self.bounds, self.fixed_rows, self.fixed_rhs = tuple(bounds), tuple(rows), tuple(rhs)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        self.setup_seconds = time.perf_counter()-started
        self.setup_cpu_seconds = time.process_time()-cpu

    def cell_constraints(self, branch, lo, hi, deadline=None):
        if self.error is not None:
            raise ValueError(self.error)
        lo, hi = Q(lo), Q(hi)
        areas = area_enclosures(branch, lo, hi, self.cameras, deadline)
        rows, rhs = list(self.fixed_rows), list(self.fixed_rhs)
        for v in range(3):
            for p, lower in enumerate(self.low[v]):
                if lower is not None:
                    row = {3*v+k: a for k, a in enumerate(areas[v][p][0]) if a}
                    row[9+v] = -ONE
                    rows.append(row); rhs.append(ONE-lower)
            for p, upper in enumerate(self.high[v]):
                if upper is not None:
                    row = {3*v+k: -a for k, a in enumerate(areas[v][p][1]) if a}
                    row[9+v] = ONE
                    rows.append(row); rhs.append(upper-ONE)
        return tuple(rows), tuple(rhs), self.bounds

    def check_cell(self, branch, lo, hi, deadline):
        """Return JSON only. No verified contradiction means no exclusion.

        deadline is an absolute time.perf_counter timestamp. Phase-I LP has a
        0.20 s solver limit and one HiGHS thread; exact work checks the deadline.
        """
        started, cpu = time.perf_counter(), time.process_time()
        out = {'verified': False, 'certificate': None, 'reason': 'unknown',
               'geometry_seconds': 0., 'certificate_search_seconds': 0.,
               'exact_check_seconds': 0., 'lp_status': None}
        try:
            if self.error is not None:
                raise ValueError(self.error)
            if not math.isfinite(float(deadline)) or time.perf_counter() >= deadline:
                raise TimeoutError('deadline before geometry')
            # The search's chosen binary64 endpoints themselves define the cell.
            qlo, qhi = Q.from_float(float(lo)), Q.from_float(float(hi))
            stage = time.perf_counter()
            try:
                rows, rhs, bounds = self.cell_constraints(branch, qlo, qhi, deadline)
            finally:
                out['geometry_seconds'] = time.perf_counter()-stage
            stage = time.perf_counter()
            try:
                remain = deadline-time.perf_counter()-.005
                if remain <= .001:
                    raise TimeoutError('deadline before certificate proposal')
                a = np.zeros((len(rows), 16))
                for i, row in enumerate(rows):
                    for j, value in row.items():
                        a[i, j] = float(value)
                a[:, 15] = -1.
                cost = np.zeros(16); cost[15] = 1.
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Unrecognized options detected:.*',
                                            category=OptimizeWarning)
                    proposal = linprog(cost, A_ub=a, b_ub=np.array(list(map(float, rhs))),
                                       bounds=[(float(x), float(y)) for x, y in bounds]+[(0., None)],
                                       method='highs', options={'time_limit': max(.001, min(.20, remain)),
                                       'threads': 1, 'primal_feasibility_tolerance': 1e-8,
                                       'dual_feasibility_tolerance': 1e-8})
                out['lp_status'] = int(proposal.status)
            finally:
                out['certificate_search_seconds'] = time.perf_counter()-stage
            if proposal.status != 0 or proposal.fun is None or proposal.fun <= 0.:
                out['reason'] = 'no_positive_phase_one_proposal'
            else:
                weights = tuple((i, Q.from_float(float(-weight)))
                                for i, weight in enumerate(proposal.ineqlin.marginals)
                                if math.isfinite(float(weight)) and weight < 0.)
                stage = time.perf_counter()
                try:
                    margin = box_margin(rows, rhs, bounds, weights, deadline)
                finally:
                    out['exact_check_seconds'] = time.perf_counter()-stage
                if time.perf_counter() >= deadline:
                    raise TimeoutError('deadline after exact check')
                if margin > ZERO:
                    out['verified'] = True
                    out['reason'] = 'positive_exact_box_margin'
                    out['certificate'] = {
                        'schema': SCHEMA, 'observation_sha256': self.digest,
                        'branch': branch, 'lo': fraction_text(qlo), 'hi': fraction_text(qhi),
                        'variable_order': list(VARIABLE_ORDER), 'row_count': len(rows),
                        'multipliers': [[i, fraction_text(w)] for i, w in weights],
                        'margin': fraction_text(margin)}
                else:
                    out['reason'] = 'nonpositive_exact_box_margin'
        except TimeoutError as exc:
            out.update(verified=False, certificate=None, reason='deadline', error=str(exc))
        except Exception as exc:
            out.update(verified=False, certificate=None, reason='error',
                       error=f'{type(exc).__name__}: {exc}')
        out['wall_seconds'] = time.perf_counter()-started
        out['cpu_seconds'] = time.process_time()-cpu
        return out
