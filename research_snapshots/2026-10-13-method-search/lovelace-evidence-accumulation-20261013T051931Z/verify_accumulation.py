#!/usr/bin/env python3
"""Independent A11 cumulative-support proof and HULL checker.

Adapted explicitly from the frozen, independently authored A10 checker:
new target root, arms D_then_D/D_then_DC, tolerance 15/1024, and mandatory
mathematical A10-D source admission before new-tree verification. Only the
independent A10/A9 checker chain is imported, never an inference solver,
runner, evaluation, or private data. The author has prior A9/A10 evaluation
background; implementation independence does not mean fresh scientific blinding.
An outer-relaxation witness is not a complete physical-world witness.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as Q
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode = True
for _accumulation_thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_accumulation_thread_var] = '1'

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-evidence-accumulation-20261013T051931Z')
EXPECTED_PHYSICAL_ROOT = Path('/mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/lovelace-evidence-accumulation-20261013T051931Z')
PYTHON = Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python')
OLD_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/continuous-tightening-1')
A9_ROOT = OLD_ROOT.parent / 'continuous-anchors-1'
_old_path = (OLD_ROOT / 'verify_tightening.py').resolve()
_spec = importlib.util.spec_from_file_location('_accumulation_frozen_independent_tightening', _old_path)
oldcheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oldcheck)
_a9 = oldcheck._a9
_a8, _proof = _a9._a8, _a9._proof
VerificationError = _a9.VerificationError
require, same, exact, enc = _a9.require, _a9.same, _a9.exact, _a9.enc
rational, finite_seconds = _a9.rational, _a9.finite_seconds
merged_intervals, _encoded_intervals, _hull = _a9.merged_intervals, _a9._encoded_intervals, _a9._hull
BRANCHES = _a9.BRANCHES
ARM_MODES = {'D_then_D': 'depth', 'D_then_DC': 'camera_tiles'}
ARMS, MODES = tuple(ARM_MODES), tuple(ARM_MODES.values())
MIN_WIDTH, MAX_CALLS, BRANCH_SECONDS = Q(15, 1024), 128, 6.0
_verified_sources = set()
BRANCH_FIELDS = {'branch', 'mode', 'domain', 'source_outer_intervals', 'camera_domain', 'tiles',
                 'roots', 'trace', 'retained_ids', 'outer_intervals', 'hull', 'status', 'schedule',
                 'closed_sides', 'classification_calls', 'processed_nodes', 'pruned_nodes',
                 'split_nodes', 'max_classifications', 'min_width', 'branch_seconds',
                 'search_budget_seconds', 'termination', 'seconds'}
NODE_FIELDS = {'id', 'parent', 'root', 'lo', 'hi', 'status', 'side', 'checks', 'children'}
NODE_STATES = {'split', 'excluded_certificate', 'retained_tolerance', 'retained_interior',
               'retained_call_budget', 'retained_deadline'}


def check_target():
    # The existing runs symlink was migrated before A11. Verify both names
    # resolve to the registered same object; never call or patch old guards.
    require(socket.gethostname() == 'liekkas' and
            ROOT == EXPECTED_ROOT.resolve() == EXPECTED_PHYSICAL_ROOT.resolve() and
            EXPECTED_ROOT.samefile(EXPECTED_PHYSICAL_ROOT),
            'target: exact logical/physical A11 root on liekkas required')
    require(Path(sys.executable) == PYTHON and sys.flags.dont_write_bytecode,
            'target: existing open3d-019 Python -B required')
    require(Path(oldcheck.__file__).resolve() == (OLD_ROOT / 'verify_tightening.py').resolve() and
            Path(_a9.__file__).resolve() == (A9_ROOT / 'verify_continuous.py').resolve(),
            'source: exact sealed independent A10/A9 checker identities required')
    require(oldcheck.MIN_WIDTH == Q(15, 256) and oldcheck.MAX_CALLS == 128 and
            oldcheck.BRANCH_SECONDS == 6.0,
            'source: unchanged A10 verification contract required')
    require(all(os.environ.get(key) == '1' for key in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
            'target: single-thread CPU required')
    return True


def _source_key(public, frozen_arm, anchor, branch, source):
    require(branch in BRANCHES and isinstance(source, dict) and
            set(source) == oldcheck.BRANCH_FIELDS,
            'source: complete saved A10 branch record required')
    same(source['mode'], 'depth', 'source: only A10-D may seed A11')
    same(source['branch'], branch, 'source: exact A10-D occlusion branch')
    # Exact prepared measurements exclude labels and evaluation-only fields.
    # The full canonical saved record binds this receipt to all proof bytes,
    # not merely its outer union or object identity.
    prepared = _a9._prepare_inputs(public, frozen_arm, anchor, 'both')
    try:
        serialized = json.dumps(source, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise VerificationError('source: canonical JSON A10 record required') from error
    return branch, prepared, serialized


def verify_source_branch(public, frozen_arm, anchor, branch, a9_branch, source):
    """Mathematically admit A10-D under its original independent rules.

    The caller first verifies a9_branch with _a9.verify_branch and aggregates
    all three A9 branches. This function verifies its A10-D refinement, not
    merely a hash. A11 verify_branch then requires the exact admitted source.
    It is sufficient to admit each source once before checking both new arms.
    """
    key = _source_key(public, frozen_arm, anchor, branch, source)
    require(isinstance(a9_branch, dict), 'source: explicit verified A9 branch required')
    same(a9_branch.get('branch'), branch, 'source: A9 branch identity')
    checked = oldcheck.verify_branch(public, frozen_arm, anchor, 'depth', branch, a9_branch, source)
    _verified_sources.add(key)
    return checked


def camera_tiles(intervals, arm):
    """Reconstruct the entire public camera cover, including closed seams."""
    require(arm in MODES, 'tiles: registered refinement mode required')
    if intervals is None:
        return (None,)
    box = _a9._camera_box(intervals)
    if arm == 'depth':
        return (box,)
    left, right = box[0], box[2]
    lm, rm = sum(left) / 2, sum(right) / 2
    return tuple((lp, (Q(0), Q(0)), rp)
                 for lp in ((left[0], lm), (lm, left[1]))
                 for rp in ((right[0], rm), (rm, right[1])))


def encoded_camera_box(box):
    return None if box is None else [[enc(lo), enc(hi)] for lo, hi in box]


def tile_problem(prepared, branch, lo, hi, tile):
    """Rebuild rows using the explicit camera tile, never the parent box."""
    require(branch in BRANCHES, 'problem: registered branch required')
    lo, hi = _a9._depth_pair(lo, hi)
    image, calibration, bounds, parent, appearance = prepared
    if tile is None:
        require(parent is None, 'problem: null tile only for inconsistent camera measurements')
        rows = (('upper', (Q(0),) * 15, Q(-1)),)
    else:
        require(parent is not None, 'problem: camera tile without camera support')
        tile = _a9._camera_box(tile)
        require(all(pl <= lo <= hi <= ph for (lo, hi), (pl, ph) in zip(tile, parent)),
                'problem: tile outside public camera box')
        areas = _a9.interval_area_enclosure(branch, lo, hi, tile)
        original = _a9._rows.necessary_rows(image, areas, True, 'joint', calibration, _a9.DRIFT)
        rows = tuple((side, tuple(coefficients) + (Q(0),) * 3, bound)
                     for side, coefficients, bound in original)
    return rows + appearance, bounds


def necessary_problem(public, frozen_arm, anchor, branch, lo, hi, tile):
    return tile_problem(_a9._prepare_inputs(public, frozen_arm, anchor, 'both'), branch, lo, hi, tile)


def verify_branch(public, frozen_arm, anchor, arm, branch, source_A10_D_branch, record):
    """Verify every new tile proof and moving-boundary tree from admitted D.

    Coverage remains conditional on the independently verified A9 ancestor
    and registered observation model. No source is admitted by hash alone.
    """
    require(arm in ARMS and branch in BRANCHES and isinstance(record, dict) and set(record) == BRANCH_FIELDS,
            'branch: registered exact A11 result schema')
    mode = ARM_MODES[arm]
    same(record['branch'], branch, 'branch: occlusion identity')
    same(record['mode'], mode, 'branch: refinement identity')
    require(_source_key(public, frozen_arm, anchor, branch, source_A10_D_branch) in _verified_sources,
            'source: A10-D must pass verify_source_branch before A11 verification')
    source_value = source_A10_D_branch['outer_intervals']
    source = merged_intervals(source_value)
    for value in record['source_outer_intervals']:
        _a9._saved_pair(value, 'source support')
    same(record['source_outer_intervals'], _encoded_intervals(source), 'source: exact independently admitted A10-D closed union')
    same(record['domain'], list(map(enc, _a9.DOMAIN)), 'branch: frozen global domain')
    for value in record['domain']:
        rational(value, 'branch domain')
    require(type(record['max_classifications']) is int and record['max_classifications'] == MAX_CALLS and
            rational(record['min_width']) == MIN_WIDTH and
            finite_seconds(record['branch_seconds'], 'branch limit') == BRANCH_SECONDS,
            'branch: fixed call cap, tolerance, and time limit')
    allocated = finite_seconds(record['search_budget_seconds'], 'branch allocated seconds')
    seconds = finite_seconds(record['seconds'], 'branch elapsed seconds')
    require(allocated <= BRANCH_SECONDS, 'branch: enlarged time allocation')
    prepared = _a9._prepare_inputs(public, frozen_arm, anchor, 'both')
    tiles = camera_tiles(prepared[3], mode)
    same(record['camera_domain'], encoded_camera_box(prepared[3]), 'tiles: exact public parent camera box')
    same(record['tiles'], [encoded_camera_box(tile) for tile in tiles], 'tiles: complete ordered closed cover')
    for box in [record['camera_domain']] + record['tiles']:
        if box is not None:
            require(isinstance(box, list) and len(box) == 3, 'tiles: three saved camera intervals')
            for pair in box:
                _a9._saved_pair(pair, 'tile camera pair')
    roots = list(range(len(source)))
    require(isinstance(record['roots'], list) and all(type(i) is int for i in record['roots']),
            'tree: canonical integer root IDs')
    same(record['roots'], roots, 'tree: one root per source connected interval')
    trace = record['trace']
    require(isinstance(trace, list) and len(source) <= len(trace) <= len(source) + 2 * MAX_CALLS,
            'tree: bounded complete allocated node set')
    for ident, node in enumerate(trace):
        require(isinstance(node, dict) and set(node) == NODE_FIELDS and node['status'] in NODE_STATES,
                'tree: registered exact node schema')
        require(type(node['id']) is int and node['id'] == ident and type(node['root']) is int,
                'tree: ordered unique integer node identity')
        require(node['parent'] is None or type(node['parent']) is int, 'tree: exact parent identity')
        require(node['side'] in (None, 'left', 'right') and isinstance(node['checks'], list),
                'tree: side and checks schema')
        lo, hi = rational(node['lo']), rational(node['hi'])
        require(_a9.DOMAIN[0] <= lo < hi <= _a9.DOMAIN[1], 'tree: positive closed depth cell in domain')
    expected = {i: (None, i, lo, hi) for i, (lo, hi) in enumerate(source)}
    active, next_id, processed = set(roots), len(roots), set()
    closed, turn = {'left': False, 'right': False}, 'left'
    calls = pruned = splits = 0
    classes, states = Counter(), Counter(node['status'] for node in trace)
    elapsed_classes, terminal_phase = 0.0, None
    schedule = record['schedule']
    require(isinstance(schedule, list) and len(schedule) <= MAX_CALLS + 2, 'schedule: bounded action list')
    for step in schedule:
        require(terminal_phase is None and active and not all(closed.values()),
                'schedule: event after completion or partial classification stop')
        require(isinstance(step, dict) and set(step) == {'side', 'node_id', 'action'} and
                type(step['node_id']) is int, 'schedule: exact action schema')
        if closed[turn]:
            turn = 'right' if turn == 'left' else 'left'
        side = turn
        ident = (min(active, key=lambda i: (expected[i][2], expected[i][3], i)) if side == 'left' else
                 max(active, key=lambda i: (expected[i][3], expected[i][2], i)))
        same((step['side'], step['node_id']), (side, ident), 'schedule: alternating current boundary')
        require(ident < len(trace), 'tree: missing reached node')
        node = trace[ident]
        if ident in processed:
            require(node['status'] == 'retained_tolerance' and step['action'] == 'close_cached',
                    'schedule: only previously tested terminal boundary may close without a call')
            closed[side] = True
        else:
            require(step['action'] == 'process' and calls < MAX_CALLS,
                    'schedule: unexpected action or classification after call cap')
            same(node['side'], side, 'schedule: saved processing side')
            require(len(node['checks']) == len(tiles), 'classification: every camera tile must be recorded')
            parent, root, lo, hi = expected[ident]
            missing_reason, keeps = None, []
            for tile_id, (tile, check) in enumerate(zip(tiles, node['checks'])):
                require(isinstance(check, dict) and set(check) == {'tile_id', 'camera_box', 'classification', 'reason'} and
                        type(check['tile_id']) is int and check['tile_id'] == tile_id,
                        'classification: exact ordered tile entry')
                same(check['camera_box'], encoded_camera_box(tile), 'classification: exact tile camera box')
                if check['camera_box'] is not None:
                    for pair in check['camera_box']:
                        _a9._saved_pair(pair, 'classified camera pair')
                classification = check['classification']
                if classification is None:
                    require(check['reason'] in ('not_run_call_budget', 'not_run_deadline'),
                            'classification: unrun tile must retain a registered reason')
                    if missing_reason is None:
                        missing_reason = check['reason']
                    same(check['reason'], missing_reason, 'classification: consistent unrun suffix')
                    if check['reason'] == 'not_run_call_budget':
                        require(calls == MAX_CALLS, 'classification: falsely claimed call exhaustion')
                    keeps.append(True)
                    continue
                require(missing_reason is None and calls < MAX_CALLS and check['reason'] == 'classified',
                        'classification: call after unrun suffix or cap')
                rows, bounds = tile_problem(prepared, branch, lo, hi, tile)
                keeps.append(_proof.verify_classification(rows, bounds, classification,
                                                         f'A11/{arm}/{branch}/{ident}/tile{tile_id}'))
                calls += 1
                classes[classification['status']] += 1
                elapsed_classes += classification['seconds']
            processed.add(ident)
            if missing_reason is not None:
                terminal_phase = 'deadline' if missing_reason == 'not_run_deadline' else 'call_budget'
                same(node['status'], 'retained_' + terminal_phase, 'tree: incomplete tile cover must remain')
                require(node['children'] is None, 'tree: no children after partial classification stop')
            elif not any(keeps):
                same(node['status'], 'excluded_certificate', 'tree: all tiles certified before deletion')
                require(node['children'] is None, 'tree: excluded node cannot have children')
                active.remove(ident)
                pruned += 1
            elif hi - lo <= MIN_WIDTH:
                same(node['status'], 'retained_tolerance', 'tree: nonexcluded terminal-width boundary retained')
                require(node['children'] is None, 'tree: terminal node cannot have children')
                closed[side] = True
            else:
                same(node['status'], 'split', 'tree: unresolved wide boundary must split')
                require(isinstance(node['children'], list) and node['children'] == [next_id, next_id + 1] and
                        all(type(child) is int for child in node['children']), 'tree: exact two-child allocation')
                midpoint = (lo + hi) / 2
                expected[next_id] = (ident, root, lo, midpoint)
                expected[next_id + 1] = (ident, root, midpoint, hi)
                active.remove(ident)
                active.update((next_id, next_id + 1))
                next_id += 2
                splits += 1
        turn = 'right' if side == 'left' else 'left'
    require(len(trace) == next_id and set(expected) == set(range(next_id)),
            'tree: no missing or extra allocated children')
    for ident, (parent, root, lo, hi) in expected.items():
        node = trace[ident]
        require((node['parent'], node['root'], rational(node['lo']), rational(node['hi'])) ==
                (parent, root, lo, hi), 'tree: exact root and closed midpoint partition')
    if not active:
        expected_termination = 'empty'
    elif all(closed.values()):
        expected_termination = 'boundaries_closed'
    elif terminal_phase is not None:
        expected_termination = terminal_phase
    else:
        require(record['termination'] in ('call_budget', 'deadline'), 'schedule: unexplained early stop')
        expected_termination = record['termination']
    same(record['termination'], expected_termination, 'schedule: actual termination')
    if expected_termination == 'call_budget':
        require(calls == MAX_CALLS, 'schedule: call budget stop before exhaustion')
    if expected_termination == 'deadline':
        require(seconds + 0.000001 >= allocated, 'schedule: deadline stop before allocated window elapsed')
    require(isinstance(record['closed_sides'], dict) and set(record['closed_sides']) == {'left', 'right'} and
            all(type(value) is bool for value in record['closed_sides'].values()), 'schedule: Boolean closed sides')
    same(record['closed_sides'], closed, 'schedule: exact closed boundary accounting')
    for ident in active - processed:
        node = trace[ident]
        expected_status = 'retained_interior' if expected_termination == 'boundaries_closed' else 'retained_' + expected_termination
        same(node['status'], expected_status, 'tree: every unvisited frontier cell retained')
        require(node['side'] is None and node['checks'] == [] and node['children'] is None,
                'tree: unvisited frontier has no fabricated check or children')
    retained_ids = sorted(active)
    require(isinstance(record['retained_ids'], list) and all(type(i) is int for i in record['retained_ids']),
            'tree: exact retained integer IDs')
    same(record['retained_ids'], retained_ids, 'tree: every and only nonexcluded final leaf retained')
    outer = merged_intervals((expected[i][2], expected[i][3]) for i in retained_ids)
    _a9._saved_support(record, 'refined support')
    same(record['outer_intervals'], _encoded_intervals(outer), 'support: exact retained closed union')
    same(record['hull'], _hull(outer), 'support: exact hull')
    require(all(any(sl <= lo <= hi <= sh for sl, sh in source) for lo, hi in outer),
            'support: refinement must be a subset of the A10-D source union')
    same(record['status'], 'OUTER_RETAINED' if outer else 'OUTER_EMPTY', 'support: result status')
    for key, expected_value in (('classification_calls', calls), ('processed_nodes', len(processed)),
                                ('pruned_nodes', pruned), ('split_nodes', splits)):
        require(type(record[key]) is int and record[key] == expected_value, 'accounting: exact ' + key)
    require(elapsed_classes <= seconds + 0.000001, 'accounting: classifier time exceeds branch elapsed time')
    return {'branch': branch, 'arm': arm, 'mode': mode, 'outer_intervals': _encoded_intervals(outer), 'hull': _hull(outer),
            'intervals': outer, 'classification_calls': calls, 'processed_nodes': len(processed),
            'pruned_nodes': pruned, 'split_nodes': splits, 'retained_nodes': len(retained_ids),
            'retained_status_counts': dict(sorted((s, n) for s, n in states.items() if s.startswith('retained_'))),
            'classification_status_counts': dict(sorted(classes.items())), 'termination': expected_termination,
            'seconds': seconds, 'status': record['status']}


def aggregate_branches(branches, record=None):
    return _a9.aggregate_branches(branches, record)


def verify_selection(incumbent, outer, policy, record):
    """Independently reuse sealed coordinate arithmetic, adapting arm identity."""
    require(policy == 'HULL', 'selection: A11 registered HULL rule required')
    require(isinstance(record, dict), 'selection: record required')
    checked = dict(record)
    if 'arm' in checked:
        require(checked['arm'] in ARMS, 'selection: registered refinement arm required')
        del checked['arm']
    return _a9.verify_selection(incumbent, outer, policy, checked)


selection_verify = verify_selection


def clear_geometry_cache():
    """Clear enclosure caches only; mathematical source receipts stay valid."""
    _a9.clear_geometry_cache()

