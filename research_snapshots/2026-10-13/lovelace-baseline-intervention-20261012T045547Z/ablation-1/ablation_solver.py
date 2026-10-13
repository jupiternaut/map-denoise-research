"""Registered inference-row ablations over the frozen full-image engine.

The RGB layout remains intact. Only necessary inference rows selected by a
public pixel mask are omitted; guaranteed-background calibration always uses
the complete image. Certificate row indices address the retained row order.
"""
from collections import deque
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
encode, decode, rational = baseline_engine.encode, baseline_engine.decode, baseline_engine.rational
MASK_NAMES = ('drop_F', 'matched_drop')
PIXELS_PER_VIEW = geometry.PIXELS_PER_VIEW
PIXEL_COUNT = 3 * PIXELS_PER_VIEW
RGB_COUNT = 3 * PIXEL_COUNT


def _exact(value, name):
    """Decode lossless public rationals without bool/float integer coercion."""
    if isinstance(value, dict):
        if set(value) != {'numerator', 'denominator'}:
            raise ValueError(f'{name} requires numerator and denominator only')
        numerator, denominator = value['numerator'], value['denominator']
        if any(not isinstance(x, int) or isinstance(x, bool)
               for x in (numerator, denominator)):
            raise TypeError(f'{name} requires exact integer numerator and denominator')
        if denominator <= 0:
            raise ValueError(f'{name} denominator must be positive')
        return decode(value)
    return rational(value)


def _image(observation, clipped):
    if not isinstance(clipped, bool):
        raise TypeError('clipped must be bool')
    image = tuple(_exact(value, 'observation') for value in observation)
    if len(image) != RGB_COUNT:
        raise ValueError('observation must match the public 432-scalar RGB layout')
    # This also enforces the registered clipping semantics before any geometry.
    frozen.pixel_bands(image, clipped=clipped, beta=BETA, epsilon=EPSILON)
    return image


def dropped_pixels(mask_name):
    """Return fixed original view*48+row*8+column pixel IDs.

    The empty full mask supports identity tests and is not a formal ablation.
    """
    if mask_name == 'full':
        return ()
    if mask_name not in MASK_NAMES:
        raise ValueError(f'mask_name must be one of {MASK_NAMES} or test-only full')
    left_column, right_column = (7, 0) if mask_name == 'drop_F' else (6, 1)
    return tuple([row * 8 + left_column for row in range(6)] +
                 [2 * PIXELS_PER_VIEW + row * 8 + right_column for row in range(6)])


def _row_metadata(image, clipped):
    bands = frozen.pixel_bands(image, clipped=clipped, beta=BETA, epsilon=EPSILON)
    records = []
    for pixel, (lower, upper) in enumerate(bands):
        view, local_pixel = divmod(pixel, PIXELS_PER_VIEW)
        row, column = divmod(local_pixel, 8)
        for side, endpoint in (('upper', upper), ('lower', lower)):
            if endpoint is not None:
                records.append({'row_id': len(records), 'pixel': pixel, 'view': view,
                                'row': row, 'column': column, 'side': side})
    return tuple(records)


def row_metadata(observation, clipped=True):
    """Map original necessary-row indices to public pixel/view/side records."""
    return _row_metadata(_image(observation, clipped), clipped)


def _filter_rows(rows, kept_row_ids, original_row_count):
    """Select by original indices, rejecting any layout mismatch, never zip."""
    if len(rows) != original_row_count:
        raise ValueError('necessary rows and original row metadata disagree')
    return tuple(rows[index] for index in kept_row_ids)


def _solve_branch(image, radius, baseline, branch, method, clipped, calibration,
                  kept_row_ids, original_row_count):
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
            original_rows = frozen.necessary_rows(image, areas, method, clipped=clipped,
                                  calibration_intervals=ts, beta=BETA, epsilon=EPSILON)
            rows = _filter_rows(original_rows, kept_row_ids, original_row_count)
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


def solve_public(observation, radius, baseline, method, mask_name, clipped=True):
    """Run the registered engine after deterministic public inference-row removal."""
    frozen._parameters(method, clipped, BETA, EPSILON, ROUNDS, MAX_BOXES, TOLERANCE)
    radius, baseline = _exact(radius, 'radius'), _exact(baseline, 'baseline')
    if radius < 0:
        raise ValueError('radius must be nonnegative')
    geometry.camera_centers(baseline)
    image = _image(observation, clipped)
    dropped = dropped_pixels(mask_name)
    started = time.perf_counter()
    metadata = _row_metadata(image, clipped)
    dropped_set = frozenset(dropped)
    removed = [record for record in metadata if record['pixel'] in dropped_set]
    kept = tuple(record['row_id'] for record in metadata if record['pixel'] not in dropped_set)
    background = baseline_engine.background_mask(radius, baseline)
    calibration = frozen.calibration_bounds(image, background, clipped=clipped,
                                           beta=BETA, epsilon=EPSILON)
    preparation_seconds = time.perf_counter() - started
    branches, seconds = {}, {}
    for branch in BRANCHES:
        tick = time.perf_counter()
        branches[branch] = _solve_branch(image, radius, baseline, branch, method, clipped,
                                        calibration, kept, len(metadata))
        seconds[branch] = time.perf_counter() - tick
    outer = frozen.union_outer(branches, method=method, clipped=clipped, beta=BETA,
                               epsilon=EPSILON, rounds=ROUNDS, max_boxes=MAX_BOXES,
                               tolerance=TOLERANCE)
    return {'branches': branches, 'outer': outer, 'branch_seconds': seconds,
            'outer_seconds': time.perf_counter() - started,
            'preparation_seconds': preparation_seconds, 'background_mask': list(background),
            'calibration_intervals': calibration['intervals'], 'calibration': calibration,
            'inference_mask': {'name': mask_name, 'dropped_pixels': list(dropped),
                               'dropped_rows': removed, 'kept_row_count': len(kept)}}


def clear_geometry_cache():
    """Clear the explicit-baseline and frozen historical geometry caches."""
    baseline_engine.clear_geometry_cache()
