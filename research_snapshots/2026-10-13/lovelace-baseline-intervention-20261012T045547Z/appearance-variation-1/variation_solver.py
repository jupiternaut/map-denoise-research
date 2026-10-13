"""Bounded same-material view variation with exact frozen endpoint bridges.

drift_bound is the maximum pairwise darkness difference for the same material
across views. Intermediate bounds append 18 signed halfspaces to the original
view-major photometric rows. Each view still shares three materials and one
joint bias over its original pixels; no nuisance variables are split.
"""
from collections import deque
from fractions import Fraction as Q
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
APPEARANCE_ROOT = Path(__file__).resolve().parent.parent / 'appearance-1'
if str(APPEARANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(APPEARANCE_ROOT))
import appearance_solver as appearance_engine

baseline_engine = appearance_engine.baseline_engine
frozen, geometry = appearance_engine.frozen, appearance_engine.geometry
BRANCHES = appearance_engine.BRANCHES
BETA, EPSILON = appearance_engine.BETA, appearance_engine.EPSILON
ROUNDS, MAX_BOXES, TOLERANCE = (appearance_engine.ROUNDS, appearance_engine.MAX_BOXES,
                               appearance_engine.TOLERANCE)
METHODS = appearance_engine.METHODS
VIEWS = appearance_engine.VIEWS
PIXELS_PER_VIEW = appearance_engine.PIXELS_PER_VIEW
PIXEL_COUNT, RGB_COUNT = appearance_engine.PIXEL_COUNT, appearance_engine.RGB_COUNT
MATERIAL_BOUNDS = appearance_engine.MATERIAL_BOUNDS
PER_VIEW_MATERIAL_BOUNDS = appearance_engine.PER_VIEW_MATERIAL_BOUNDS
VIEW_PAIRS = ((0, 1), (0, 2), (1, 2))
APPEARANCE_MODELS = ('strict_shared', 'independent', 'bounded')
BOUNDED_SCOPE = 'necessary_bounded_per_view_material_shared_bias_halfspaces_with_historical_continuous_geometry_bounds'
encode, decode, rational = appearance_engine.encode, appearance_engine.decode, appearance_engine.rational
_exact, _encoded_pairs = appearance_engine._exact, appearance_engine._encoded_pairs


def _drift_bound(value):
    result = _exact(value, 'drift_bound')
    if not 0 <= result <= 1:
        raise ValueError('drift_bound must lie in [0,1]')
    return result


def difference_rows(method, drift_bound):
    """Pair-major, material-major, forward/reverse upper rows, exactly 18.

    Forward is d_vk-d_wk<=delta and reverse is d_wk-d_vk<=delta. Bias
    coefficients are zero. delta describes pairwise spread, not distance
    from a latent material center or only from one reference view.
    """
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    delta = _drift_bound(drift_bound)
    dimension = 12 if method == 'joint' else 9
    rows = []
    for first, second in VIEW_PAIRS:
        for material in range(3):
            for direction in (1, -1):
                coefficients = [Q(0)] * dimension
                coefficients[3 * first + material] = Q(direction)
                coefficients[3 * second + material] = Q(-direction)
                rows.append(('upper', tuple(coefficients), delta))
    return tuple(rows)


def necessary_rows(observation, areas, method, *, clipped, drift_bound,
                   calibration_intervals=None, beta=BETA, epsilon=EPSILON):
    """Preserve all A4 photometric rows, then append the fixed signed rows."""
    links = difference_rows(method, drift_bound)
    photometric = appearance_engine.necessary_rows(
        observation, areas, method, clipped=clipped,
        calibration_intervals=calibration_intervals, appearance_scope='per_view',
        beta=beta, epsilon=epsilon)
    return photometric + links


def propagate(rows, bounds=MATERIAL_BOUNDS, rounds=ROUNDS):
    """Exact signed contractor with historical arithmetic on nonnegative rows.

    Each upper row snapshots its minimizing endpoints (lo for positive and hi
    for negative coefficients); each lower row snapshots maximizing endpoints.
    The row total and all candidates use that immutable snapshot. A negative
    coefficient reverses the endpoint being tightened. Update order, sweep
    counters and first-contradiction certificates retain frozen semantics.
    """
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
        raise ValueError('rounds must be a nonnegative integer')
    initial = tuple(tuple(_exact(value, 'variable interval') for value in pair)
                    for pair in bounds)
    if len(initial) not in (3, 6, 9, 12) or any(len(pair) != 2 or min(pair) < 0
                                              for pair in initial):
        raise ValueError('3, 6, 9 or 12 nonnegative variable intervals required')
    necessary = []
    for kind, coefficients, bound in rows:
        coefficients = tuple(_exact(value, 'coefficient') for value in coefficients)
        bound = _exact(bound, 'row bound')
        if kind not in ('upper', 'lower') or len(coefficients) != len(initial):
            raise ValueError('rows require upper/lower and matching exact coefficients')
        necessary.append((kind, coefficients, bound))
    lower, upper = [pair[0] for pair in initial], [pair[1] for pair in initial]
    steps, performed = [], 0

    def excluded(contradiction):
        return {'excluded': True, 'rounds': performed, 'updates': len(steps),
                'certificate': {'initial_bounds': _encoded_pairs(initial),
                                'steps': steps, 'contradiction': contradiction}}

    for variable, (lo, hi) in enumerate(initial):
        if lo > hi:
            return excluded({'kind': 'initial_interval', 'variable': variable,
                             'lower': encode(lo), 'upper': encode(hi)})
    for iteration in range(rounds):
        performed = iteration + 1
        changed = False
        for index, (kind, coefficients, bound) in enumerate(necessary):
            if kind == 'upper':
                values = tuple(lower[variable] if coefficient >= 0 else upper[variable]
                               for variable, coefficient in enumerate(coefficients))
            else:
                values = tuple(upper[variable] if coefficient >= 0 else lower[variable]
                               for variable, coefficient in enumerate(coefficients))
            total = sum((a * value for a, value in zip(coefficients, values)), Q(0))
            if (total > bound if kind == 'upper' else total < bound):
                return excluded({'kind': 'row', 'round': iteration, 'row': index,
                                 'total': encode(total), 'bound': encode(bound)})
            for variable, coefficient in enumerate(coefficients):
                if coefficient == 0:
                    continue
                other = total - coefficient * values[variable]
                candidate = (bound - other) / coefficient
                if kind == 'upper':
                    side = 'upper' if coefficient > 0 else 'lower'
                else:
                    side = 'lower' if coefficient > 0 else 'upper'
                if side == 'upper' and candidate < upper[variable]:
                    upper[variable] = candidate
                elif side == 'lower' and candidate > lower[variable]:
                    lower[variable] = candidate
                else:
                    continue
                steps.append({'round': iteration, 'row': index, 'variable': variable,
                              'side': side, 'value': encode(candidate)})
                changed = True
                if lower[variable] > upper[variable]:
                    return excluded({'kind': 'interval', 'round': iteration, 'row': index,
                                     'variable': variable, 'lower': encode(lower[variable]),
                                     'upper': encode(upper[variable])})
        if not changed:
            break
    return {'excluded': False, 'rounds': performed, 'updates': len(steps),
            'bounds': _encoded_pairs(zip(lower, upper))}


def _solve_branch(image, radius, baseline, branch, method, delta, clipped, calibration):
    depth_lo, depth_hi = geometry.DEPTH_MIN, geometry.DEPTH_MAX
    ts = tuple(tuple(map(decode, pair)) for pair in calibration['intervals'])
    initial = PER_VIEW_MATERIAL_BOUNDS + ts if method == 'joint' else PER_VIEW_MATERIAL_BOUNDS
    queue = deque([(0, None, depth_lo, depth_hi)])
    records, retained = [], []
    next_id, evaluated, shared_calls = 1, 0, 0
    while queue:
        node_id, parent, left, right = queue.popleft()
        entry = {'id': node_id, 'parent': parent, 'lo': encode(left), 'hi': encode(right)}
        if calibration['excluded']:
            entry.update(status='excluded_calibration', certificate=calibration['certificate'])
        elif evaluated >= MAX_BOXES:
            entry.update(status='unresolved_budget',
                         darkness_bounds=_encoded_pairs(PER_VIEW_MATERIAL_BOUNDS),
                         t_bounds=calibration['intervals'])
            retained.append((left, right, node_id))
        else:
            areas = geometry.geometry_bounds(branch, radius, left, right, baseline)
            rows = necessary_rows(image, areas, method, clipped=clipped, drift_bound=delta,
                                  calibration_intervals=ts, beta=BETA, epsilon=EPSILON)
            evaluated += 1
            contraction = propagate(rows, bounds=initial, rounds=ROUNDS)
            shared_calls += 1
            entry.update(rounds=contraction['rounds'], updates=contraction['updates'])
            if contraction['excluded']:
                entry.update(status='excluded_shared', group_id='global',
                             certificate=contraction['certificate'])
            else:
                final = contraction['bounds']
                entry.update(darkness_bounds=final[:9],
                             t_bounds=final[9:] if method == 'joint' else calibration['intervals'])
                if right - left <= TOLERANCE:
                    entry['status'] = 'retained_tolerance'
                    retained.append((left, right, node_id))
                else:
                    middle = (left + right) / 2
                    entry.update(status='split', children=[next_id, next_id + 1])
                    queue.append((next_id, node_id, left, middle))
                    queue.append((next_id + 1, node_id, middle, right))
                    next_id += 2
        records.append(entry)
    hull = None if not retained else [encode(min(a for a, _, _ in retained)),
                                     encode(max(b for _, b, _ in retained))]
    return {'domain': [encode(depth_lo), encode(depth_hi)], 'epsilon': encode(EPSILON),
            'beta': encode(BETA), 'clipped': clipped, 'tolerance': encode(TOLERANCE),
            'max_boxes': MAX_BOXES, 'evaluated_boxes': evaluated, 'trace': records,
            'retained_ids': [node_id for _, _, node_id in retained], 'hull': hull,
            'shared_tests': shared_calls, 'propagation_rounds': ROUNDS,
            'mode': method, 'rounds': ROUNDS, 'budget': MAX_BOXES,
            'channel_precheck': 'none', 'initial_bounds': _encoded_pairs(initial),
            'scope': BOUNDED_SCOPE, 'appearance_scope': 'per_view',
            'material_variable_names': list(appearance_engine.material_variable_names())}


def solve_public(observation, radius, baseline, method, drift_bound, clipped=True):
    """Bridge exact endpoints or solve the bounded pairwise-variation model."""
    frozen._parameters(method, clipped, BETA, EPSILON, ROUNDS, MAX_BOXES, TOLERANCE)
    delta = _drift_bound(drift_bound)
    radius, baseline = _exact(radius, 'radius'), _exact(baseline, 'baseline')
    if radius < 0:
        raise ValueError('radius must be nonnegative')
    geometry.camera_centers(baseline)
    image = appearance_engine._image(observation, clipped)
    if delta == 0:
        result = baseline_engine.solve_public(image, radius, baseline, method, clipped=clipped)
        result.update(appearance_model='strict_shared', drift_bound=encode(delta))
        return result
    if delta == 1:
        result = appearance_engine.solve_public(image, radius, baseline, method,
                                               appearance_scope='per_view', clipped=clipped)
        result.update(appearance_model='independent', drift_bound=encode(delta))
        return result
    started = time.perf_counter()
    mask = baseline_engine.background_mask(radius, baseline)
    calibration = frozen.calibration_bounds(image, mask, clipped=clipped, beta=BETA, epsilon=EPSILON)
    preparation_seconds = time.perf_counter() - started
    branches, seconds = {}, {}
    for branch in BRANCHES:
        tick = time.perf_counter()
        branches[branch] = _solve_branch(image, radius, baseline, branch, method,
                                       delta, clipped, calibration)
        seconds[branch] = time.perf_counter() - tick
    outer = frozen.union_outer(branches, method=method, clipped=clipped, beta=BETA,
                               epsilon=EPSILON, rounds=ROUNDS, max_boxes=MAX_BOXES,
                               tolerance=TOLERANCE)
    return {'branches': branches, 'outer': outer, 'branch_seconds': seconds,
            'outer_seconds': time.perf_counter() - started,
            'preparation_seconds': preparation_seconds, 'background_mask': list(mask),
            'calibration_intervals': calibration['intervals'], 'calibration': calibration,
            'appearance_scope': 'per_view',
            'material_variable_names': list(appearance_engine.material_variable_names()),
            'appearance_model': 'bounded', 'drift_bound': encode(delta)}


def clear_geometry_cache():
    """Clear all explicit-baseline and historical caches before cold solves."""
    appearance_engine.clear_geometry_cache()
