"""Frozen photometric constraints with explicit public camera geometry."""
from collections import deque
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
FROZEN_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z')
if str(FROZEN_ROOT) not in sys.path:
    sys.path.insert(0, str(FROZEN_ROOT))
import structured_solver as frozen
import baseline_geometry as geometry

BRANCHES = geometry.BRANCHES
BETA, EPSILON = frozen.BETA, frozen.EPSILON
ROUNDS, MAX_BOXES, TOLERANCE = frozen.ROUNDS, frozen.MAX_BOXES, frozen.TOLERANCE
encode, decode, rational = frozen.encode, frozen.decode, frozen.rational


def background_mask(radius, baseline):
    areas = [geometry.geometry_bounds(branch, radius, geometry.DEPTH_MIN,
                                     geometry.DEPTH_MAX, baseline) for branch in BRANCHES]
    size = len(BRANCHES) * geometry.PIXELS_PER_VIEW
    return tuple(all(all(a == 0 for a in branch[pixel][1::2]) for branch in areas)
                 for pixel in range(size))


def _solve_branch(image, radius, baseline, branch, method, clipped, calibration):
    depth_lo, depth_hi = geometry.DEPTH_MIN, geometry.DEPTH_MAX
    ts = tuple(tuple(map(decode, pair)) for pair in calibration['intervals'])
    initial = frozen.MATERIAL_BOUNDS + ts if method == 'joint' else frozen.MATERIAL_BOUNDS
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
                         darkness_bounds=frozen._encoded_pairs(frozen.MATERIAL_BOUNDS),
                         t_bounds=calibration['intervals'])
            retained.append((left, right, node_id))
        else:
            areas = geometry.geometry_bounds(branch, radius, left, right, baseline)
            rows = frozen.necessary_rows(image, areas, method, clipped=clipped,
                                        calibration_intervals=ts, beta=BETA, epsilon=EPSILON)
            evaluated += 1
            contraction = frozen.propagate(rows, bounds=initial, rounds=ROUNDS)
            shared_calls += 1
            entry.update(rounds=contraction['rounds'], updates=contraction['updates'])
            if contraction['excluded']:
                entry.update(status='excluded_shared', group_id='global',
                             certificate=contraction['certificate'])
            else:
                final = contraction['bounds']
                entry.update(darkness_bounds=final[:3],
                             t_bounds=final[3:] if method == 'joint' else calibration['intervals'])
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
            'channel_precheck': 'none', 'initial_bounds': frozen._encoded_pairs(initial),
            'scope': 'necessary_shared_bias_halfspaces_with_historical_continuous_geometry_bounds'}


def solve_public(observation, radius, baseline, method, clipped=True):
    """Same registered solver and arm schema, conditioned on public baseline."""
    frozen._parameters(method, clipped, BETA, EPSILON, ROUNDS, MAX_BOXES, TOLERANCE)
    radius, baseline = frozen._nonnegative(radius, 'radius'), rational(baseline)
    geometry.camera_centers(baseline)
    image = tuple(decode(value) if isinstance(value, dict) else rational(value) for value in observation)
    if len(image) != 3 * len(BRANCHES) * geometry.PIXELS_PER_VIEW:
        raise ValueError('observation must match the public RGB layout')
    started = time.perf_counter()
    mask = background_mask(radius, baseline)
    calibration = frozen.calibration_bounds(image, mask, clipped=clipped, beta=BETA, epsilon=EPSILON)
    preparation_seconds = time.perf_counter() - started
    branches, seconds = {}, {}
    for branch in BRANCHES:
        tick = time.perf_counter()
        branches[branch] = _solve_branch(image, radius, baseline, branch, method, clipped, calibration)
        seconds[branch] = time.perf_counter() - tick
    outer = frozen.union_outer(branches, method=method, clipped=clipped, beta=BETA,
                               epsilon=EPSILON, rounds=ROUNDS, max_boxes=MAX_BOXES,
                               tolerance=TOLERANCE)
    return {'branches': branches, 'outer': outer, 'branch_seconds': seconds,
            'outer_seconds': time.perf_counter() - started,
            'preparation_seconds': preparation_seconds, 'background_mask': list(mask),
            'calibration_intervals': calibration['intervals'], 'calibration': calibration}


def clear_geometry_cache():
    """Clear explicit-baseline and all read-only historical geometry caches."""
    geometry.clear_geometry_cache()
    frozen.clear_geometry_cache()
