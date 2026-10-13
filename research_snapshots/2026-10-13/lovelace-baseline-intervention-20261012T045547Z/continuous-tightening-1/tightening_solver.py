"""A10 endpoint-first tightening of the sealed A9 both closed support.

Only public measurements and the old branch's closed support enter search.
Depth boxes can be removed only when every camera tile has an independently
rechecked exact contradiction. Feasible outer points and UNKNOWN are retained;
neither is a physical forward witness. The unprocessed interior and every
budget/deadline frontier cell remain in the returned closed union.
"""
from fractions import Fraction as Q
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD_ROOT = ROOT.parent / 'continuous-anchors-1'
if str(OLD_ROOT) not in sys.path:
    sys.path.insert(0, str(OLD_ROOT))
import continuous_solver as baseline

if Path(baseline.__file__).resolve() != OLD_ROOT / 'continuous_solver.py':
    raise RuntimeError('wrong frozen A9 solver identity')
anchor, geometry, variation = baseline.anchor, baseline.geometry, baseline.variation
exact, encode = baseline.exact, baseline.encode
classify, certificate_check, witness_check = baseline.classify, baseline.certificate_check, baseline.witness_check
DOMAIN, BRANCHES = baseline.DOMAIN, baseline.BRANCHES
MODES = ('depth', 'camera_tiles')
MIN_WIDTH = Q(15, 256)
MAX_CLASSIFICATIONS = 128
BRANCH_SECONDS = 6.0
select_depth, merge_intervals = baseline.select_depth, baseline.merge_intervals
_now = time.monotonic


def _encoded_pairs(pairs):
    return [[encode(lo), encode(hi)] for lo, hi in pairs]


def _encode_camera(box):
    return None if box is None else _encoded_pairs(box)


def camera_tiles(intervals):
    """Four closed Cartesian midpoint tiles; shared edges are not removed.

    Degenerate left/right intervals still have four indexed tiles. An empty
    anchor camera domain is represented by one explicit None tile, requiring
    a checked constant contradiction instead of a vacuous all-empty test.
    """
    if intervals is None:
        return (None,)
    intervals = baseline._camera_intervals(intervals)
    left, middle, right = intervals
    lm, rm = sum(left) / 2, sum(right) / 2
    return tuple((lpair, middle, rpair)
                 for lpair in ((left[0], lm), (lm, left[1]))
                 for rpair in ((right[0], rm), (rm, right[1])))


def _public_inputs(public, oldarm, anchors, depth, branch):
    photo, calibration, measurements = baseline.strip_method_inputs(public, oldarm, anchors, 'both')
    _z, image, bounds, _enabled = anchor._inputs(photo, calibration, measurements, 'both', depth, branch)
    _survey, intervals = anchor.camera_intervals(measurements)
    return photo, calibration, measurements, image, bounds, intervals


def build_problem(public, oldarm, anchors, lo, hi, branch, camera_box):
    """Same A9 both whole-depth necessary rows on one explicit camera tile.

    All d9, t3 and tA3 bounds/appearance rows remain unchanged. The common
    camera survey correlation remains outer-marginalized exactly as in A9;
    camera subdivision adds no new physical model or measurement.
    """
    lo, hi = baseline._depth_pair(lo, hi)
    _photo, _calibration, measurements, image, bounds, intervals = _public_inputs(public, oldarm, anchors, lo, branch)
    if intervals is None:
        if camera_box is not None:
            raise ValueError('empty camera domain must use an explicit None tile')
        rows = (('upper', (Q(0),) * 15, Q(-1)),)
    else:
        if camera_box is None:
            raise ValueError('a nonempty camera domain requires an explicit closed box')
        camera_box = baseline._camera_intervals(camera_box)
        if any(not alo <= lo_ <= hi_ <= ahi for (lo_, hi_), (alo, ahi) in zip(camera_box, intervals)):
            raise ValueError('camera tile lies outside the original anchor enclosure')
        areas = baseline.interval_area_bounds(branch, lo, hi, camera_box)
        rows12 = variation.necessary_rows(image, areas, 'joint', clipped=True,
                                         drift_bound=anchor.DRIFT_BOUND, beta=anchor.BETA, epsilon=anchor.EPSILON)
        rows = tuple((kind, coefficients + (Q(0),) * 3, bound) for kind, coefficients, bound in rows12)
    return rows + anchor.appearance_rows(measurements), bounds


def clear_geometry_cache():
    """Runner calls this before each arm, without changing frozen globals."""
    baseline.clear_geometry_cache()


def _parameters(mode, deadline, max_classifications, tolerance, branch_seconds):
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    if type(max_classifications) is not int or max_classifications < 0:
        raise ValueError('classification budget must be a nonnegative integer')
    tolerance = exact(tolerance, 'depth tolerance')
    if tolerance <= 0:
        raise ValueError('depth tolerance must be positive')
    if isinstance(branch_seconds, bool) or not isinstance(branch_seconds, (int, float)) or not math.isfinite(branch_seconds) or branch_seconds < 0:
        raise ValueError('branch_seconds must be finite and nonnegative')
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise ValueError('deadline must be finite absolute monotonic time')
    return tolerance


def solve_branch(public, oldarm, anchors, baseline_branch, mode, deadline=None, *,
                 max_classifications=MAX_CLASSIFICATIONS, tolerance=MIN_WIDTH,
                 branch_seconds=BRANCH_SECONDS):
    """Refine only current outer-hull boundaries, alternating left/right.

    Test-only keyword budgets make finite hand fixtures possible; the formal
    runner must use the fixed defaults 128 calls, 15/256 mm and six seconds.
    Each tile classification costs one call, including direct certificates.
    Every processed cell records every tile, with explicit unrun reasons if
    the cap/deadline interrupts. No result is inferred from an unrun tile.
    """
    started = _now()
    tolerance = _parameters(mode, deadline, max_classifications, tolerance, branch_seconds)
    end = started + float(branch_seconds)
    if deadline is not None:
        end = min(end, float(deadline))
    if not isinstance(baseline_branch, dict):
        raise TypeError('sealed A9 branch must be a record')
    branch = baseline_branch['branch']
    if branch not in BRANCHES:
        raise ValueError('unregistered branch')
    source = merge_intervals(baseline_branch['outer_intervals'])
    photo, calibration, measurements, _image, _bounds, intervals = _public_inputs(public, oldarm, anchors, DOMAIN[0], branch)
    tiles = (intervals,) if mode == 'depth' else camera_tiles(intervals)
    nodes, active, roots = [], set(), []

    def new_node(parent, root, lo, hi):
        ident = len(nodes)
        record = {'id': ident, 'parent': parent, 'root': ident if root is None else root,
                  'lo': encode(lo), 'hi': encode(hi), 'status': 'pending',
                  'side': None, 'checks': [], 'children': None}
        nodes.append(record)
        active.add(ident)
        return ident

    for lo, hi in source:
        roots.append(new_node(None, None, lo, hi))
    calls, processed, pruned, splits = 0, 0, 0, 0
    closed, side, schedule = {'left': False, 'right': False}, 'left', []
    termination = None

    def deadline_reached():
        return _now() >= end

    def extrema(which):
        if which == 'left':
            return min(active, key=lambda i: (exact(nodes[i]['lo']), exact(nodes[i]['hi']), i))
        return max(active, key=lambda i: (exact(nodes[i]['hi']), exact(nodes[i]['lo']), i))

    while active and not all(closed.values()):
        if closed[side]:
            side = 'right' if side == 'left' else 'left'
        ident = extrema(side)
        node = nodes[ident]
        if node['status'] == 'retained_tolerance':
            schedule.append({'side': side, 'node_id': ident, 'action': 'close_cached'})
            closed[side] = True
            side = 'right' if side == 'left' else 'left'
            continue
        if node['status'] != 'pending':
            raise RuntimeError('only unprocessed or closed boundary leaves may be active')
        if deadline_reached():
            termination = 'deadline'
            break
        if calls >= max_classifications:
            termination = 'call_budget'
            break
        node['side'] = side
        schedule.append({'side': side, 'node_id': ident, 'action': 'process'})
        processed += 1
        lo, hi = exact(node['lo']), exact(node['hi'])
        interrupted = None
        for tile_id, box in enumerate(tiles):
            check = {'tile_id': tile_id, 'camera_box': _encode_camera(box), 'classification': None, 'reason': None}
            if interrupted is None:
                if deadline_reached():
                    interrupted = 'deadline'
                elif calls >= max_classifications:
                    interrupted = 'call_budget'
            if interrupted is not None:
                check['reason'] = 'not_run_' + interrupted
                node['checks'].append(check)
                continue
            rows, bounds = build_problem(photo, calibration, measurements, lo, hi, branch, box)
            if deadline_reached():
                interrupted = 'deadline'
                check['reason'] = 'not_run_deadline'
                node['checks'].append(check)
                continue
            result = classify(rows, bounds, deadline=end)
            calls += 1
            status = result.get('status')
            if status == 'CERTIFIED_INFEASIBLE':
                if not certificate_check(rows, bounds, result.get('certificate')):
                    raise ArithmeticError('camera/depth tile certificate failed exact recheck')
            elif status == 'FEASIBLE_WITNESS':
                if not witness_check(rows, bounds, result.get('witness')):
                    raise ArithmeticError('outer tile witness failed exact recheck')
            elif status != 'UNKNOWN':
                raise ValueError('unregistered exact tile classification status')
            check.update(classification=result, reason='classified')
            node['checks'].append(check)
        complete_negative = all(check['classification'] is not None and
                                check['classification']['status'] == 'CERTIFIED_INFEASIBLE' for check in node['checks'])
        if complete_negative:
            if len(node['checks']) != len(tiles) or not tiles:
                raise RuntimeError('full nonempty camera coverage required before exclusion')
            node['status'] = 'excluded_certificate'
            active.remove(ident)
            pruned += 1
        elif interrupted is not None:
            node['status'] = 'retained_' + interrupted
            termination = interrupted
            break
        elif hi - lo <= tolerance:
            node['status'] = 'retained_tolerance'
            closed[side] = True
        else:
            midpoint = (lo + hi) / 2
            left = new_node(ident, node['root'], lo, midpoint)
            right = new_node(ident, node['root'], midpoint, hi)
            node.update(status='split', children=[left, right])
            active.remove(ident)
            splits += 1
        side = 'right' if side == 'left' else 'left'
    if termination is None:
        termination = 'empty' if not active else 'boundaries_closed'
    pending_status = {'empty': 'retained_interior', 'boundaries_closed': 'retained_interior',
                      'call_budget': 'retained_call_budget', 'deadline': 'retained_deadline'}[termination]
    for ident in active:
        if nodes[ident]['status'] == 'pending':
            nodes[ident]['status'] = pending_status
    retained = [node['id'] for node in nodes if node['id'] in active]
    outer = merge_intervals((nodes[ident]['lo'], nodes[ident]['hi']) for ident in retained)
    return {'branch': branch, 'mode': mode, 'domain': _encoded_pairs((DOMAIN,))[0],
            'source_outer_intervals': _encoded_pairs(source), 'camera_domain': _encode_camera(intervals),
            'tiles': [_encode_camera(box) for box in tiles], 'roots': roots, 'trace': nodes,
            'retained_ids': retained, 'outer_intervals': _encoded_pairs(outer),
            'hull': None if not outer else [encode(outer[0][0]), encode(outer[-1][1])],
            'status': 'OUTER_RETAINED' if outer else 'OUTER_EMPTY', 'schedule': schedule, 'closed_sides': closed,
            'classification_calls': calls, 'processed_nodes': processed, 'pruned_nodes': pruned, 'split_nodes': splits,
            'max_classifications': max_classifications, 'min_width': encode(tolerance), 'branch_seconds': float(branch_seconds),
            'search_budget_seconds': max(0.0, end - started), 'termination': termination,
            'seconds': max(0.0, _now() - started)}
