#!/usr/bin/env python3
"""Independent A9 whole-depth enclosure, tree, and coordinate verifier.

Only sealed independent A8/A6/A5 arithmetic and geometry are imported. This
module never imports the continuous solver, its runner, or data. A retained
outer-row witness is deliberately not a complete forward-world assertion.
All scientific guarantees are conditional on the registered explanation family.
"""
from __future__ import annotations

from collections import Counter, deque
from fractions import Fraction as Q
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode = True
for _continuous_verifier_thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_continuous_verifier_thread] = '1'

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/continuous-anchors-1')
PYTHON = Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python')
_a8_path = ROOT.parent / 'independent-anchors-1/verify_anchors.py'
_spec = importlib.util.spec_from_file_location('_continuous_frozen_independent_anchors', _a8_path)
_a8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a8)
_geometry, _rows, _proof = _a8._geometry, _a8._old, _a8._proof
VerificationError = _a8.VerificationError
require, same, exact, encoded = _a8.require, _a8.same, _a8.exact, _a8.encoded
enc, rational, finite_seconds = encoded, _a8.rational, _a8.finite_seconds
verify_certificate, verify_witness = _proof.verify_certificate, _proof.verify_witness
DOMAIN, MIN_WIDTH, MAX_NODES, BRANCH_SECONDS = (Q(540), Q(660)), Q(15, 32), 127, 3.0
BRANCHES, ARMS, POLICIES = ('none', 'partial', 'full'), ('none', 'camera', 'appearance', 'both'), ('KEEP', 'HULL', 'UNION')
RADIUS, BETA, EPSILON, DRIFT = Q(1, 5), Q(1, 100), Q(1, 100), Q(1, 10)
SURVEY, LOCALIZATION, TRANSFER, ANCHOR_NOISE = Q(1, 100), Q(1, 100), Q(1, 200), Q(1, 200)
CENTERS = (-Q(80), Q(0), Q(80))
DEFAULT_CAMERA_INTERVALS = ((-RADIUS, RADIUS), (Q(0), Q(0)), (-RADIUS, RADIUS))
BRANCH_FIELDS = {'domain', 'trace', 'retained_ids', 'outer_intervals', 'hull', 'status',
                 'classified_nodes', 'pruned_nodes', 'max_nodes', 'min_width',
                 'branch_seconds', 'search_budget_seconds', 'termination', 'seconds'}
NODE_FIELDS = {'id', 'parent', 'lo', 'hi', 'status', 'classification'}
SELECTION_FIELDS = {'policy', 'status', 'incumbent', 'selected', 'moved', 'outer_intervals',
                    'hull', 'model_conditional_nonregression'}
NODE_STATES = ('split', 'excluded_certificate', 'retained_tolerance', 'retained_node_budget', 'retained_deadline')

require(Path(_a8.__file__).resolve() == ROOT.parent / 'independent-anchors-1/verify_anchors.py' and
        Path(_geometry.__file__).resolve() == ROOT.parent / 'recovery-1/verify_baseline.py',
        'source: exact independently authored A8/recovery geometry identity')
require(_geometry.DOMAIN == DOMAIN and _rows.TOLERANCE == MIN_WIDTH and
        _a8.CENTERS == CENTERS and _a8.RADIUS == RADIUS,
        'source: frozen geometry, depth, and camera contract')


def check_target():
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT,
            'target: exact A9 root on liekkas required')
    require(Path(sys.executable) == PYTHON and sys.flags.dont_write_bytecode,
            'target: existing open3d-019 Python -B required')
    require(all(os.environ.get(key) == '1' for key in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
            'target: single-thread CPU environment required')
    return True


def _depth_pair(lo, hi):
    lo, hi = exact(lo), exact(hi)
    require(DOMAIN[0] <= lo <= hi <= DOMAIN[1], 'depth: ordered closed cell inside fixed domain required')
    return lo, hi


def _camera_box(intervals):
    require(isinstance(intervals, (tuple, list)) and len(intervals) == 3,
            'area: exact three camera intervals required')
    result = []
    for pair in intervals:
        require(isinstance(pair, (tuple, list)) and len(pair) == 2, 'area: closed camera pair required')
        lo, hi = map(exact, pair)
        require(-RADIUS <= lo <= hi <= RADIUS, 'area: camera pair outside frozen uncertainty')
        result.append((lo, hi))
    require(result[1] == (Q(0), Q(0)), 'area: middle camera gauge must equal zero')
    return tuple(result)


def _project_rectangle(rectangle, depth, camera):
    """Independent quotient edge extrema over the complete depth/camera box."""
    require(depth[0] > 0, 'area: positive depth denominator required')
    edge_ranges = []
    for index, edge in enumerate(rectangle):
        numerator = (edge - camera[1], edge - camera[0]) if index < 2 else (edge, edge)
        values = tuple(n / z for n in numerator for z in depth)
        edge_ranges.append((min(values), max(values)))
    inner = (edge_ranges[0][1], edge_ranges[1][0], edge_ranges[2][1], edge_ranges[3][0])
    if inner[0] >= inner[1] or inner[2] >= inner[3]:
        inner = None
    outer = (edge_ranges[0][0], edge_ranges[1][1], edge_ranges[2][0], edge_ranges[3][1])
    return inner, outer


def rectangle_lipschitz(rectangle, center, interval, pixel):
    """Normalized moving-edge strip bound in inverse depth, for every camera."""
    width, height = pixel[1] - pixel[0], pixel[3] - pixel[2]
    require(width > 0 and height > 0, 'area: nondegenerate pixel required')
    horizontal_motion = sum((max(abs(edge - center - interval[0]), abs(edge - center - interval[1]))
                             for edge in rectangle[:2]), Q(0))
    vertical_motion = sum((abs(edge) for edge in rectangle[2:]), Q(0))
    return horizontal_motion / width + vertical_motion / height


@lru_cache(maxsize=16384)
def _rectangle_enclosure(branch, lo, hi, intervals):
    rear = None if branch == 'none' else _geometry.REAR if branch == 'partial' else _geometry.HIDDEN
    result = []
    for center, interval in zip(CENTERS, intervals):
        camera = (center + interval[0], center + interval[1])
        left, right, front = (_project_rectangle(rect, (lo, hi), camera)
                              for rect in (_geometry.LEFT, _geometry.RIGHT, _geometry.FRONT))
        back = (None, None) if rear is None else _project_rectangle(
            rear, (_geometry.REAR_DEPTH, _geometry.REAR_DEPTH), camera)
        for pixel in _geometry.PIXELS:
            area = _geometry.rectangle_area(pixel)
            values = (_geometry.rectangle_area(left[0], pixel) / area,
                      _geometry.rectangle_area(left[1], pixel) / area,
                      _geometry.rectangle_area(right[0], pixel) / area,
                      _geometry.rectangle_area(right[1], pixel) / area,
                      _geometry.rectangle_difference_area(back[0], front[1], pixel) / area,
                      _geometry.rectangle_difference_area(back[1], front[0], pixel) / area)
            require(all(0 <= values[k] <= values[k + 1] <= 1 for k in (0, 2, 4)),
                    'area: independent rectangle enclosure ordering')
            result.append(values)
    return tuple(result)


@lru_cache(maxsize=16384)
def _interval_area_enclosure(branch, lo, hi, intervals):
    generic = _rectangle_enclosure(branch, lo, hi, intervals)
    qlo, qhi = Q(1) / hi, Q(1) / lo
    qc, radius_q = (qlo + qhi) / 2, (qhi - qlo) / 2
    centered = _a8.area_enclosure(branch, Q(1) / qc, intervals)
    result = []
    for view, (center, interval) in enumerate(zip(CENTERS, intervals)):
        for pixel_number, pixel in enumerate(_geometry.PIXELS):
            index = 48 * view + pixel_number
            values = []
            # Rear depth is fixed. A visibility change costs at most the whole
            # moving FRONT symmetric difference, even for partial occlusion.
            for material, rectangle in enumerate((_geometry.LEFT, _geometry.RIGHT, _geometry.FRONT)):
                delta = rectangle_lipschitz(rectangle, center, interval, pixel) * radius_q
                low = max(Q(0), generic[index][2 * material], centered[index][2 * material] - delta)
                high = min(Q(1), generic[index][2 * material + 1], centered[index][2 * material + 1] + delta)
                require(low <= high, 'area: independently sound enclosures disagree')
                values.extend((low, high))
            result.append(tuple(values))
    return tuple(result)


def interval_area_enclosure(branch, lo, hi, intervals=DEFAULT_CAMERA_INTERVALS):
    require(branch in BRANCHES, 'area: registered branch required')
    lo, hi = _depth_pair(lo, hi)
    return _interval_area_enclosure(branch, lo, hi, _camera_box(intervals))


def rectangle_enclosure(branch, lo, hi, intervals=DEFAULT_CAMERA_INTERVALS):
    require(branch in BRANCHES, 'area: registered branch required')
    lo, hi = _depth_pair(lo, hi)
    return _rectangle_enclosure(branch, lo, hi, _camera_box(intervals))


def _array(value, length, where):
    require(isinstance(value, (tuple, list)) and len(value) == length, where + ': exact array length required')
    return tuple(map(exact, value))


def _prepare_inputs(public, frozen_arm, anchor, arm):
    """Decode only allowed public measurements; IDs/labels never affect rows."""
    use_camera, use_appearance = _a8.enabled(arm)
    require(isinstance(public, dict) and isinstance(frozen_arm, dict), 'inputs: public/calibration dictionaries required')
    photo = {key: public[key] for key in ('observation', 'radius', 'baseline', 'clipped')}
    photo['observation_id'] = 'A9-public-measurements'
    calibration_record = {'calibration_intervals': frozen_arm['calibration_intervals'],
                          'appearance_model': frozen_arm.get('appearance_model', 'bounded'),
                          'drift_bound': frozen_arm.get('drift_bound', encoded(DRIFT))}
    _, image, calibration = _a8.public_inputs(photo, calibration_record, DOMAIN[0], 'none')
    bounds = ((Q(0), Q(1)),) * 9 + calibration + ((Q(0), 2 * BETA),) * 3
    intervals = DEFAULT_CAMERA_INTERVALS
    if use_camera or use_appearance:
        require(isinstance(anchor, dict), 'inputs: enabled public anchor measurements required')
    if use_camera:
        camera = anchor['camera']
        require(isinstance(camera, dict), 'camera: public measurement dictionary required')
        require(tuple(exact(camera[key]) for key in ('Xf', 'Zf', 'survey_bound', 'localization_bound')) ==
                (Q(0), Q(1000), SURVEY, LOCALIZATION), 'camera: frozen survey/localization contract')
        readings = _array(camera['u'], 3, 'camera u')
        residuals = tuple(-center - Q(1000) * u for center, u in zip(CENTERS, readings))
        kmin, kmax = max(-SURVEY, residuals[1] - LOCALIZATION), min(SURVEY, residuals[1] + LOCALIZATION)
        intervals = None
        if kmin <= kmax:
            projected = tuple((Q(0), Q(0)) if view == 1 else
                              (max(-RADIUS, residual - kmax - LOCALIZATION),
                               min(RADIUS, residual - kmin + LOCALIZATION))
                              for view, residual in enumerate(residuals))
            if all(lo <= hi for lo, hi in projected):
                intervals = projected
    appearance_rows = []
    if use_appearance:
        appearance = anchor['appearance']
        require(isinstance(appearance, dict), 'appearance: public measurement dictionary required')
        require(tuple(exact(appearance[key]) for key in ('bias_bound', 'transfer_bound', 'noise_bound')) ==
                (BETA, TRANSFER, ANCHOR_NOISE), 'appearance: frozen independent measurement bounds')
        coupons, references = (_array(appearance[key], size, 'appearance ' + key)
                               for key, size in (('coupons', 9), ('references', 3)))
        require(all(0 <= value <= 1 for value in coupons + references), 'appearance: clipped readings required')
        for index, value in enumerate(coupons):
            coefficients = [Q(0)] * 15
            coefficients[index] = coefficients[12 + index // 3] = Q(1)
            for side, bound in zip(('upper', 'lower'), _a8._bands(value, TRANSFER + ANCHOR_NOISE)):
                if bound is not None:
                    appearance_rows.append((side, tuple(coefficients), bound))
        for view, value in enumerate(references):
            coefficients = [Q(0)] * 15
            coefficients[12 + view] = Q(1)
            for side, bound in zip(('upper', 'lower'), _a8._bands(value, ANCHOR_NOISE)):
                if bound is not None:
                    appearance_rows.append((side, tuple(coefficients), bound - Q(1, 2)))
    return image, calibration, bounds, intervals, tuple(appearance_rows)


def _cell_problem(prepared, branch, lo, hi):
    image, calibration, bounds, intervals, appearance = prepared
    if intervals is None:
        rows = (('upper', (Q(0),) * 15, Q(-1)),)
    else:
        areas = interval_area_enclosure(branch, lo, hi, intervals)
        original = _rows.necessary_rows(image, areas, True, 'joint', calibration, DRIFT)
        rows = tuple((side, tuple(coefficients) + (Q(0),) * 3, bound)
                     for side, coefficients, bound in original)
    return rows + appearance, bounds


def necessary_problem(public, frozen_arm, anchor, arm, branch, lo, hi):
    require(branch in BRANCHES, 'problem: registered branch required')
    lo, hi = _depth_pair(lo, hi)
    return _cell_problem(_prepare_inputs(public, frozen_arm, anchor, arm), branch, lo, hi)


def merged_intervals(intervals):
    require(not isinstance(intervals, (str, bytes, dict)), 'union: closed interval sequence required')
    pairs = []
    for pair in intervals:
        require(isinstance(pair, (tuple, list)) and len(pair) == 2, 'union: closed endpoint pair required')
        pairs.append(_depth_pair(*pair))
    merged = []
    for lo, hi in sorted(pairs):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def _encoded_intervals(intervals):
    return [[encoded(lo), encoded(hi)] for lo, hi in intervals]


def _hull(intervals):
    return None if not intervals else [encoded(intervals[0][0]), encoded(intervals[-1][1])]


def _saved_pair(value, where):
    require(isinstance(value, list) and len(value) == 2, where + ': saved canonical endpoint pair required')
    return tuple(rational(endpoint, where) for endpoint in value)


def _saved_support(record, where):
    require(isinstance(record['outer_intervals'], list), where + ': saved canonical interval array required')
    for value in record['outer_intervals']:
        _saved_pair(value, where + ' interval')
    if record['hull'] is not None:
        _saved_pair(record['hull'], where + ' hull')


def verify_branch(public, frozen_arm, anchor, arm, branch, record):
    """Rebuild every classified whole cell and replay the complete BFS tree."""
    require(branch in BRANCHES and isinstance(record, dict) and
            set(record) in (BRANCH_FIELDS, BRANCH_FIELDS | {'branch'}), 'branch: registered exact result schema')
    if 'branch' in record:
        same(record['branch'], branch, 'branch: exact occlusion branch identity')
    _saved_pair(record['domain'], 'branch domain')
    _saved_support(record, 'branch support')
    same(record['domain'], [encoded(value) for value in DOMAIN], 'branch: root full fixed domain')
    require(type(record['max_nodes']) is int and record['max_nodes'] == MAX_NODES and
            rational(record['min_width']) == MIN_WIDTH and
            finite_seconds(record['branch_seconds'], 'branch seconds limit') == BRANCH_SECONDS,
            'branch: frozen classification budget/tolerance/time limit')
    budget_seconds = finite_seconds(record['search_budget_seconds'], 'branch allocated seconds')
    seconds = finite_seconds(record['seconds'], 'branch elapsed seconds')
    require(budget_seconds <= BRANCH_SECONDS, 'branch: enlarged allocated time')
    trace = record['trace']
    require(isinstance(trace, list) and 0 < len(trace) <= 2 * MAX_NODES + 1,
            'tree: nonempty bounded complete frontier required')
    prepared = _prepare_inputs(public, frozen_arm, anchor, arm)
    queue = deque([(0, None, DOMAIN[0], DOMAIN[1])])
    next_id, classified, pruned, elapsed_classes = 1, 0, 0, 0.0
    retained, retained_ids, states, classification_states = [], [], Counter(), Counter()
    terminal_phase = None
    for node in trace:
        require(queue, 'tree: unexpected unreachable extra node')
        ident, parent, lo, hi = queue.popleft()
        require(isinstance(node, dict) and node.get('status') in NODE_STATES and
                set(node) == NODE_FIELDS | ({'children'} if node['status'] == 'split' else set()),
                'tree: exact registered node schema')
        require(type(node['id']) is int and node['id'] == ident and
                (node['parent'] is None if parent is None else type(node['parent']) is int and node['parent'] == parent),
                'tree: BFS node allocation/parent mismatch')
        require(rational(node['lo']) == lo and rational(node['hi']) == hi and lo < hi,
                'tree: complete midpoint-child closed endpoints required')
        status, classification = node['status'], node['classification']
        states[status] += 1
        if status in ('retained_node_budget', 'retained_deadline'):
            require(classification is None, 'tree: unclassified frontier cannot carry classification')
            if status == 'retained_node_budget':
                require(classified == MAX_NODES and terminal_phase != 'deadline', 'tree: false node-budget frontier')
                terminal_phase = 'node_budget'
            else:
                terminal_phase = 'deadline'
        else:
            require(terminal_phase is None and classified < MAX_NODES and classification is not None,
                    'tree: classification after deadline/node cap or absent classification')
            rows, bounds = _cell_problem(prepared, branch, lo, hi)
            keep = _proof.verify_classification(rows, bounds, classification,
                                                f'whole cell/{branch}/{ident}')
            classified += 1
            elapsed_classes += classification['seconds']
            classification_states[classification['status']] += 1
            if status == 'excluded_certificate':
                require(not keep and classification['status'] == 'CERTIFIED_INFEASIBLE',
                        'tree: exclusion requires exact whole-cell contradiction')
                pruned += 1
            elif status == 'split':
                require(keep and hi - lo > MIN_WIDTH, 'tree: unsupported or unnecessary split')
                require(isinstance(node['children'], list) and node['children'] == [next_id, next_id + 1] and
                        all(type(child) is int for child in node['children']), 'tree: exact complete two-child allocation')
                middle = (lo + hi) / 2
                queue.append((next_id, ident, lo, middle))
                queue.append((next_id + 1, ident, middle, hi))
                next_id += 2
            else:
                require(status == 'retained_tolerance' and keep and hi - lo <= MIN_WIDTH,
                        'tree: tolerance leaf requires retained classification and registered width')
        if status.startswith('retained_'):
            retained.append((lo, hi))
            retained_ids.append(ident)
    require(not queue and len(trace) == next_id, 'tree: missing pending cells/children from complete frontier')
    same(record['retained_ids'], retained_ids, 'tree: every retained leaf, with no certified leaf or gap')
    require(isinstance(record['retained_ids'], list) and all(type(value) is int for value in record['retained_ids']),
            'tree: integer retained node identities required')
    outer = merged_intervals(retained)
    same(record['outer_intervals'], _encoded_intervals(outer), 'union: exact retained closed depth union')
    same(record['hull'], _hull(outer), 'union: exact support hull without deleting gaps')
    same(record['status'], 'OUTER_RETAINED' if outer else 'OUTER_EMPTY', 'branch: outer support status')
    for key, value in (('classified_nodes', classified), ('pruned_nodes', pruned)):
        require(type(record[key]) is int and record[key] == value, 'tree: exact ' + key + ' accounting')
    expected_termination = 'deadline' if states['retained_deadline'] else 'node_budget' if states['retained_node_budget'] else 'tree_complete'
    same(record['termination'], expected_termination, 'tree: complete/deadline/node-budget termination')
    require(elapsed_classes <= seconds, 'branch: classified elapsed time exceeds whole branch elapsed time')
    if states['retained_deadline']:
        require(seconds + 0.000001 >= budget_seconds, 'branch: deadline claimed before allocated window expired')
    return {'branch': branch, 'status': record['status'], 'outer_intervals': _encoded_intervals(outer),
            'hull': _hull(outer), 'intervals': outer, 'classified_nodes': classified, 'pruned_nodes': pruned,
            'retained_nodes': len(retained), 'deadline_leaves': states['retained_deadline'],
            'node_budget_leaves': states['retained_node_budget'],
            'classification_status_counts': dict(sorted(classification_states.items())), 'seconds': seconds}


def aggregate_branches(branches, record=None):
    """Exact existential depth support across all three verified branches."""
    require(isinstance(branches, (tuple, list)) and len(branches) == 3 and
            [value.get('branch') for value in branches] == list(BRANCHES),
            'aggregate: all three branches in registered order required')
    outer = merged_intervals(pair for value in branches for pair in value['outer_intervals'])
    result = {'outer_intervals': _encoded_intervals(outer), 'hull': _hull(outer),
              'status': 'OUTER_RETAINED' if outer else 'OUTER_EMPTY'}
    if record is not None:
        require(isinstance(record, dict), 'aggregate: saved arm record required')
        _saved_support(record, 'aggregate support')
        same(record['outer_intervals'], result['outer_intervals'], 'aggregate: complete three-branch support union')
        same(record['hull'], result['hull'], 'aggregate: exact three-branch hull')
        if 'status' in record:
            same(record['status'], result['status'], 'aggregate: three-branch outer status')
    return result


def verify_selection(incumbent, outer, policy, record):
    """Independently recompute KEEP/HULL/UNION and deterministic low ties."""
    require(policy in POLICIES and isinstance(record, dict) and
            SELECTION_FIELDS <= set(record) <= SELECTION_FIELDS | {'task_id', 'observation_id', 'arm'},
            'selection: exact registered coordinate fields')
    for key in ('task_id', 'observation_id'):
        if key in record:
            require(isinstance(record[key], str), 'selection: string ' + key + ' required')
    if 'arm' in record:
        require(record['arm'] in ARMS, 'selection: registered anchor arm required')
    rational(record['incumbent'], 'selection incumbent')
    rational(record['selected'], 'selection output')
    _saved_support(record, 'selection support')
    incumbent, intervals = exact(incumbent), merged_intervals(outer)
    selected, status = incumbent, 'KEEP'
    if policy != 'KEEP':
        if not intervals:
            status = 'SUPPORT_FAILED'
        else:
            status = 'PROJECTED'
            if policy == 'HULL':
                selected = min(intervals[-1][1], max(intervals[0][0], incumbent))
            else:
                candidates = [min(hi, max(lo, incumbent)) for lo, hi in intervals]
                selected = min(candidates, key=lambda value: (abs(value - incumbent), value))
    expected = {'policy': policy, 'status': status, 'incumbent': encoded(incumbent),
                'selected': encoded(selected), 'moved': selected != incumbent,
                'outer_intervals': _encoded_intervals(intervals), 'hull': _hull(intervals),
                'model_conditional_nonregression': policy in ('KEEP', 'HULL')}
    require(type(record['moved']) is bool and type(record['model_conditional_nonregression']) is bool,
            'selection: exact Boolean movement/conditional claim required')
    same({key: record[key] for key in SELECTION_FIELDS}, expected,
         'selection: exact independent coordinate action and model-conditional claim')
    return expected


def clear_geometry_cache():
    _rectangle_enclosure.cache_clear()
    _interval_area_enclosure.cache_clear()
    _a8.area_enclosure.cache_clear()
    _a8.point_areas.cache_clear()
