"""View-specific material sharing over the frozen full-image depth engine.

Each view shares its material triple over all 48 original pixels. The joint
method also shares one bias variable per view. Geometry, clipping, calibration,
row order, finite sweeps and the depth search retain their frozen semantics.
The global_shared scope is a nonformal identity fixture for the old engine.
"""
from collections import deque
from fractions import Fraction as Q
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
BASELINE_ROOT = Path(__file__).resolve().parent.parent / 'recovery-1'
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))
import baseline_solver as baseline_engine

frozen, geometry = baseline_engine.frozen, baseline_engine.geometry
BRANCHES = baseline_engine.BRANCHES
BETA, EPSILON = baseline_engine.BETA, baseline_engine.EPSILON
ROUNDS, MAX_BOXES, TOLERANCE = (baseline_engine.ROUNDS, baseline_engine.MAX_BOXES,
                               baseline_engine.TOLERANCE)
METHODS = frozen.METHODS
APPEARANCE_SCOPES = ('per_view', 'global_shared')
VIEWS = 3
PIXELS_PER_VIEW = geometry.PIXELS_PER_VIEW
PIXEL_COUNT = VIEWS * PIXELS_PER_VIEW
RGB_COUNT = 3 * PIXEL_COUNT
MATERIAL_BOUNDS = frozen.MATERIAL_BOUNDS
PER_VIEW_MATERIAL_BOUNDS = MATERIAL_BOUNDS * VIEWS
GLOBAL_SCOPE = 'necessary_shared_bias_halfspaces_with_historical_continuous_geometry_bounds'
PER_VIEW_SCOPE = 'necessary_per_view_material_shared_bias_halfspaces_with_historical_continuous_geometry_bounds'
rational = frozen.rational


def encode(value):
    """Encode an exact rational, rejecting float and bool coercion."""
    return frozen.encode(rational(value))


def decode(value):
    """Decode only an exact two-integer public rational representation."""
    if not isinstance(value, dict) or set(value) != {'numerator', 'denominator'}:
        raise ValueError('public rational requires numerator and denominator only')
    numerator, denominator = value['numerator'], value['denominator']
    if any(not isinstance(item, int) or isinstance(item, bool)
           for item in (numerator, denominator)):
        raise TypeError('public rational requires exact integer numerator and denominator')
    if denominator <= 0:
        raise ValueError('public rational denominator must be positive')
    return Q(numerator, denominator)


def _exact(value, name):
    """Accept exact scalars or strict public rational dictionaries."""
    try:
        return decode(value) if isinstance(value, dict) else rational(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise type(error)(f'{name}: {error}') from error


def _encoded_pairs(bounds):
    return [[encode(lo), encode(hi)] for lo, hi in bounds]


def _scope(appearance_scope):
    if appearance_scope not in APPEARANCE_SCOPES:
        raise ValueError(f'appearance_scope must be one of {APPEARANCE_SCOPES}')
    return appearance_scope


def material_variable_names(appearance_scope='per_view'):
    """Return the exact darkness-variable order; t variables follow in joint."""
    _scope(appearance_scope)
    return (tuple(f'd{view}{material}' for view in range(VIEWS) for material in range(3))
            if appearance_scope == 'per_view' else ('d0', 'd1', 'd2'))


def material_bounds(appearance_scope='per_view'):
    _scope(appearance_scope)
    return PER_VIEW_MATERIAL_BOUNDS if appearance_scope == 'per_view' else MATERIAL_BOUNDS


def _image(observation, clipped):
    if not isinstance(clipped, bool):
        raise TypeError('clipped must be bool')
    image = tuple(_exact(value, 'observation') for value in observation)
    if len(image) != RGB_COUNT:
        raise ValueError('observation must match the public 432-scalar RGB layout')
    frozen.pixel_bands(image, clipped=clipped, beta=BETA, epsilon=EPSILON)
    return image


def necessary_rows(observation, areas, method, *, clipped,
                   calibration_intervals=None, appearance_scope='per_view',
                   beta=BETA, epsilon=EPSILON):
    """Build original upper-before-lower rows with view-major material slots.

    Geometry still supplies three area intervals per original pixel. Upper
    rows use area lower endpoints and lower rows use area upper endpoints.
    Calibrated rows marginalize frozen t; joint appends t0,t1,t2 after the
    material slots. Small hand-built fixtures retain original pixel indexing.
    """
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    _scope(appearance_scope)
    image = tuple(_exact(value, 'observation') for value in observation)
    bands = frozen.pixel_bands(image, clipped=clipped,
                               beta=_exact(beta, 'beta'), epsilon=_exact(epsilon, 'epsilon'))
    areas = tuple(tuple(_exact(value, 'area') for value in pair) for pair in areas)
    if len(areas) != len(bands) or len(areas) > PIXEL_COUNT:
        raise ValueError('RGB observation and area bounds must agree')
    if any(len(pair) != 6 or any(not 0 <= lo <= hi <= 1
           for lo, hi in zip(pair[::2], pair[1::2])) for pair in areas):
        raise ValueError('three ordered nonnegative area intervals required per pixel')
    if method == 'calibrated':
        if calibration_intervals is None:
            raise ValueError('calibrated rows require frozen t intervals')
        ts = tuple(tuple(_exact(value, 'calibration interval') for value in pair)
                   for pair in calibration_intervals)
        if len(ts) != VIEWS or any(len(pair) != 2 or not 0 <= pair[0] <= pair[1]
                                  for pair in ts):
            raise ValueError('three ordered nonnegative t intervals required')
    rows = []
    for pixel, (area, (lower, upper)) in enumerate(zip(areas, bands)):
        view = pixel // PIXELS_PER_VIEW
        lows, highs = area[::2], area[1::2]
        if appearance_scope == 'per_view':
            before, after = (Q(0),) * (3 * view), (Q(0),) * (3 * (VIEWS - view - 1))
            lows, highs = before + lows + after, before + highs + after
        if method == 'joint':
            t_coefficients = tuple(Q(int(index == view)) for index in range(VIEWS))
            if upper is not None:
                rows.append(('upper', lows + t_coefficients, upper))
            if lower is not None:
                rows.append(('lower', highs + t_coefficients, lower))
        else:
            if upper is not None:
                rows.append(('upper', lows, upper - ts[view][0]))
            if lower is not None:
                rows.append(('lower', highs, lower - ts[view][1]))
    return tuple(rows)


def propagate(rows, bounds=MATERIAL_BOUNDS, rounds=ROUNDS):
    """Frozen exact row arithmetic generalized to 3, 6, 9 or 12 variables.

    Every row total uses its starting lower endpoints for an upper inequality
    or starting upper endpoints for a lower inequality. Opposite-side updates
    do not change that total. Counters, update order and first-contradiction
    certificate semantics are identical to the historical 3/6 contractor.
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
        if kind not in ('upper', 'lower') or len(coefficients) != len(initial) or any(a < 0 for a in coefficients):
            raise ValueError('rows require upper/lower and matching nonnegative coefficients')
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
            values = lower if kind == 'upper' else upper
            total = sum((a * value for a, value in zip(coefficients, values)), Q(0))
            if (total > bound if kind == 'upper' else total < bound):
                return excluded({'kind': 'row', 'round': iteration, 'row': index,
                                 'total': encode(total), 'bound': encode(bound)})
            for variable, coefficient in enumerate(coefficients):
                if coefficient == 0:
                    continue
                other = total - coefficient * values[variable]
                candidate = (bound - other) / coefficient
                if kind == 'upper' and candidate < upper[variable]:
                    upper[variable] = candidate
                    side = 'upper'
                elif kind == 'lower' and candidate > lower[variable]:
                    lower[variable] = candidate
                    side = 'lower'
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


def _solve_branch(image, radius, baseline, branch, method, appearance_scope, clipped, calibration):
    depth_lo, depth_hi = geometry.DEPTH_MIN, geometry.DEPTH_MAX
    ts = tuple(tuple(map(decode, pair)) for pair in calibration['intervals'])
    materials = material_bounds(appearance_scope)
    material_count = len(materials)
    initial = materials + ts if method == 'joint' else materials
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
                         darkness_bounds=_encoded_pairs(materials),
                         t_bounds=calibration['intervals'])
            retained.append((left, right, node_id))
        else:
            areas = geometry.geometry_bounds(branch, radius, left, right, baseline)
            rows = necessary_rows(image, areas, method, clipped=clipped,
                                  calibration_intervals=ts, appearance_scope=appearance_scope,
                                  beta=BETA, epsilon=EPSILON)
            evaluated += 1
            contraction = propagate(rows, bounds=initial, rounds=ROUNDS)
            shared_calls += 1
            entry.update(rounds=contraction['rounds'], updates=contraction['updates'])
            if contraction['excluded']:
                entry.update(status='excluded_shared', group_id='global',
                             certificate=contraction['certificate'])
            else:
                final = contraction['bounds']
                entry.update(darkness_bounds=final[:material_count],
                             t_bounds=final[material_count:] if method == 'joint' else calibration['intervals'])
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
            'scope': PER_VIEW_SCOPE if appearance_scope == 'per_view' else GLOBAL_SCOPE,
            'appearance_scope': appearance_scope,
            'material_variable_names': list(material_variable_names(appearance_scope))}


def solve_public(observation, radius, baseline, method, appearance_scope='per_view', clipped=True):
    """Run the registered finite search with the chosen public sharing scope."""
    frozen._parameters(method, clipped, BETA, EPSILON, ROUNDS, MAX_BOXES, TOLERANCE)
    _scope(appearance_scope)
    radius, baseline = _exact(radius, 'radius'), _exact(baseline, 'baseline')
    if radius < 0:
        raise ValueError('radius must be nonnegative')
    geometry.camera_centers(baseline)
    image = _image(observation, clipped)
    started = time.perf_counter()
    mask = baseline_engine.background_mask(radius, baseline)
    calibration = frozen.calibration_bounds(image, mask, clipped=clipped, beta=BETA, epsilon=EPSILON)
    preparation_seconds = time.perf_counter() - started
    branches, seconds = {}, {}
    for branch in BRANCHES:
        tick = time.perf_counter()
        branches[branch] = _solve_branch(image, radius, baseline, branch, method,
                                       appearance_scope, clipped, calibration)
        seconds[branch] = time.perf_counter() - tick
    outer = frozen.union_outer(branches, method=method, clipped=clipped, beta=BETA,
                               epsilon=EPSILON, rounds=ROUNDS, max_boxes=MAX_BOXES,
                               tolerance=TOLERANCE)
    return {'branches': branches, 'outer': outer, 'branch_seconds': seconds,
            'outer_seconds': time.perf_counter() - started,
            'preparation_seconds': preparation_seconds, 'background_mask': list(mask),
            'calibration_intervals': calibration['intervals'], 'calibration': calibration,
            'appearance_scope': appearance_scope,
            'material_variable_names': list(material_variable_names(appearance_scope))}


def clear_geometry_cache():
    """Clear explicit-baseline and frozen historical caches for cold solves."""
    baseline_engine.clear_geometry_cache()
