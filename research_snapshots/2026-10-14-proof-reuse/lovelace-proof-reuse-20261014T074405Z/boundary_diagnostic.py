"""Offline exact four-tile examination of saved final hull boundary cells.

No private truth, incumbent, or task label enters this diagnostic. A feasible
witness certifies only the frozen necessary LP, never the physical renderer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import socket
import sys
import time

ROOT = Path(__file__).resolve().parent
A14 = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z')
PUBLIC = A14 / 'data/confirm/public.json'
SCHEMA = 'a15-boundary-four-tile-diagnostic-v1'
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
sys.dont_write_bytecode = True


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != Path(path).resolve():
        raise RuntimeError('module identity mismatch')
    return module


def load_kernels():
    global JOINT, VERIFY
    if 'JOINT' not in globals():
        JOINT = _load('a15_boundary_original_joint', A14 / 'joint_kernel.py')
        VERIFY = _load('a15_boundary_independent_verify', A14 / 'joint_verify.py')


def qt(value):
    value = Q(value)
    return f'{value.numerator}/{value.denominator}'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boundary_cells(state):
    """All saved cells touching either exact binary64 hull endpoint; dedupe.

    A branch that ties another branch must not be lost. Checked/terminal are
    recorded, not inferred from width or used to filter frontier membership.
    """
    cells = state.get('cells', [])
    if not cells:
        return []
    lo, hi = min(c['lower'] for c in cells), max(c['upper'] for c in cells)
    if lo != state['lower'] or hi != state['upper']:
        raise ValueError('saved hull does not equal retained-cell envelope')
    out = {}
    for cell in cells:
        if not (540 <= cell['lower'] <= cell['upper'] <= 660):
            raise ValueError('invalid saved cell')
        sides = [side for side, yes in (('lower', cell['lower'] == lo),
                                       ('upper', cell['upper'] == hi)) if yes]
        if not sides:
            continue
        key = cell['branch'], qt(Q.from_float(float(cell['lower']))), qt(Q.from_float(float(cell['upper'])))
        if key not in out:
            out[key] = {**cell, 'lo_exact': key[1], 'hi_exact': key[2], 'sides': sides}
        else:
            out[key]['sides'] = sorted(set(out[key]['sides'] + sides))
    return list(out.values())


def exact_witness_check(rows, rhs, bounds, x):
    if len(x) != len(bounds) or any(not isinstance(v, Q) for v in x):
        return False
    return (all(lo <= val <= hi for val, (lo, hi) in zip(x, bounds))
            and all(sum((a*x[j] for j, a in row.items()), Q(0)) <= b
                    for row, b in zip(rows, rhs)))


def rational_basis_candidate(rows, rhs, bounds, x_float, deadline):
    """Recover one exact vertex from approximately tight rows; no guarantee.

    A numerical active-set hint only chooses equations. The resulting rational
    vector is accepted solely by checking every exact row and bound afterward.
    """
    n = len(bounds)
    constraints = [(dict(row), b) for row, b in zip(rows, rhs)]
    for j, (lo, hi) in enumerate(bounds):
        constraints.extend((({j: Q(1)}, hi), ({j: Q(-1)}, -lo)))
    constraints.sort(key=lambda rb: abs(float(rb[1])-sum(float(a)*x_float[j] for j, a in rb[0].items()))
                     / max(1., sum(abs(float(a)) for a in rb[0].values())))
    pivots = {}
    for row, b in constraints:
        if time.perf_counter() >= deadline:
            return None
        eq = [row.get(j, Q(0)) for j in range(n)] + [b]
        for pivot in sorted(pivots):
            factor = eq[pivot]
            if factor:
                eq = [v-factor*w for v, w in zip(eq, pivots[pivot])]
        pivot = next((j for j in range(n) if eq[j]), None)
        if pivot is None:
            continue
        divisor = eq[pivot]
        pivots[pivot] = [v/divisor for v in eq]
        if len(pivots) == n:
            answer = [Q(0)]*n
            for j in reversed(range(n)):
                answer[j] = pivots[j][-1]-sum((pivots[j][k]*answer[k] for k in range(j+1, n)), Q(0))
            return tuple(answer)
    return None


def find_exact_witness(rows, rhs, bounds, deadline):
    """Propose interior and vertex witnesses, accepting exact arithmetic only."""
    import numpy as np
    from scipy.optimize import linprog, OptimizeWarning
    import warnings
    statuses = []
    n = len(bounds)
    matrix = np.zeros((len(rows), n))
    for i, row in enumerate(rows):
        for j, a in row.items():
            matrix[i, j] = float(a)
    # First maximize uniform slack in all rows/bounds (often yields a rational
    # interior witness); then try a zero-objective vertex if the set is thin.
    for mode in ('interior', 'vertex'):
        remain = deadline-time.perf_counter()-.01
        if remain <= .001:
            return None, statuses, 'witness_deadline'
        if mode == 'interior':
            aub = np.zeros((len(rows)+2*n, n+1))
            aub[:len(rows), :n] = matrix
            bub = [float(b) for b in rhs]
            for j, (lo, hi) in enumerate(bounds):
                aub[len(rows)+2*j, j] = 1.
                aub[len(rows)+2*j+1, j] = -1.
                bub.extend((float(hi), -float(lo)))
            aub[:, n] = 1.
            cost = np.zeros(n+1); cost[n] = -1.
            lp_bounds = [(None, None)]*n + [(0., 1.)]
        else:
            aub, bub, cost = matrix, list(map(float, rhs)), np.zeros(n)
            lp_bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=OptimizeWarning)
            proposal = linprog(cost, A_ub=aub, b_ub=np.asarray(bub), bounds=lp_bounds,
                               method='highs', options={'threads': 1, 'time_limit': min(.4, remain),
                               'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
        statuses.append({'mode': mode, 'status': int(proposal.status)})
        if proposal.status != 0 or proposal.x is None:
            continue
        floats = proposal.x[:n]
        raw = tuple(Q.from_float(float(v)) for v in floats)
        proposals = [raw]
        for max_denominator in (10**6, 10**9, 10**12):
            proposals.append(tuple(min(hi, max(lo, v.limit_denominator(max_denominator)))
                                   for v, (lo, hi) in zip(raw, bounds)))
        for candidate in proposals:
            if time.perf_counter() >= deadline:
                return None, statuses, 'witness_deadline'
            if exact_witness_check(rows, rhs, bounds, candidate):
                return candidate, statuses, 'all_rows_and_bounds_exact'
        candidate = rational_basis_candidate(rows, rhs, bounds, floats, deadline)
        if candidate is not None and exact_witness_check(rows, rhs, bounds, candidate):
            return candidate, statuses, 'exact_active_basis_all_rows_and_bounds'
    return None, statuses, 'numeric_results_without_exact_witness'


def independent_verify(observation, cell, tile_id, result):
    """Rebuild constraints using the frozen solver-independent A14 verifier."""
    old = VERIFY.old
    try:
        canonical = old._canonical_observation(observation)
        cameras, _photo, _bounds, _fixed = old._decode_observation(canonical)
        box = VERIFY._tiles(cameras)[tile_id]
        lo, hi = Q(cell['lo_exact']), Q(cell['hi_exact'])
        branch = cell['branch']
        digest = old.observation_sha256(observation)
        if result.get('observation_sha256') != digest or result.get('camera_box') != VERIFY._encode(box):
            raise ValueError('diagnostic target identity mismatch')
        if result['status'] == 'exact_infeasible':
            proof = result['certificate']
            if (proof['observation_sha256'] != digest or proof['branch'] != branch
                    or proof['lo'] != qt(lo) or proof['hi'] != qt(hi)
                    or proof['schema'] != JOINT.TILE_SCHEMA or proof['tile_id'] != tile_id
                    or proof['camera_box'] != VERIFY._encode(box)
                    or proof['variable_order'] != old.VARIABLE_ORDER):
                raise ValueError('certificate identity mismatch')
            weights = {}
            for index, value in proof['multipliers']:
                if type(index) is not int or index < 0 or index in weights:
                    raise ValueError('invalid multiplier index')
                weight = old._fraction(value)
                if weight <= 0:
                    raise ValueError('invalid multiplier')
                weights[index] = weight
            if not weights:
                raise ValueError('empty multiplier')
            rows, bounds, count = VERIFY._rows(canonical, branch, lo, hi, box, set(weights))
            if set(rows) != set(weights) or count != proof['row_count']:
                raise ValueError('constraint layout mismatch')
            totals, beta = [Q(0)]*15, Q(0)
            for index, weight in weights.items():
                row, upper = rows[index]
                beta += weight*upper
                for j, a in row:
                    totals[j] += weight*a
            margin = sum((a*bounds[j][0 if a >= 0 else 1] for j, a in enumerate(totals)), Q(0))-beta
            if margin <= 0 or margin != old._fraction(proof['margin']):
                raise ValueError('no exact contradiction')
            return {'verified': True, 'kind': 'exact_infeasible', 'margin': qt(margin)}
        if result['status'] == 'exact_necessary_lp_feasible':
            _rows, _bounds, count = VERIFY._rows(canonical, branch, lo, hi, box, set())
            rows, bounds, count = VERIFY._rows(canonical, branch, lo, hi, box, set(range(count)))
            x = tuple(old._fraction(v) for v in result['witness'])
            if not exact_witness_check([dict(rows[i][0]) for i in range(count)],
                                       [rows[i][1] for i in range(count)], bounds, x):
                raise ValueError('exact witness violates a row or bound')
            return {'verified': True, 'kind': 'exact_necessary_lp_feasible', 'row_count': count}
        return {'verified': None, 'kind': 'unknown', 'reason': 'no exact claim to validate'}
    except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
        return {'verified': False, 'error': f'{type(exc).__name__}: {exc}'}


def diagnose_tile(observation, cell, tile_id, seconds):
    start, cpu = time.perf_counter(), time.process_time()
    deadline = start+seconds
    context = JOINT.JointContext(observation)
    if context.error is not None:
        raise ValueError(context.error)
    tile = context.contexts[tile_id]
    result = {'tile_id': tile_id, 'status': 'unknown', 'observation_sha256': context.digest,
              'camera_box': JOINT.encode_camera(context.tiles[tile_id]), 'certificate': None,
              'witness': None, 'tile_budget_seconds': seconds}
    # Leave part of the independent cap for finding and checking a witness.
    check = tile.check_cell(cell['branch'], cell['lower'], cell['upper'], min(deadline, start+seconds*.55))
    result['infeasibility_attempt'] = check
    if check['verified']:
        proof = dict(check['certificate'])
        proof.update(schema=JOINT.TILE_SCHEMA, tile_id=tile_id, camera_box=result['camera_box'])
        result.update(status='exact_infeasible', certificate=proof)
    elif time.perf_counter() < deadline:
        rows, rhs, bounds = tile.cell_constraints(cell['branch'], Q(cell['lo_exact']), Q(cell['hi_exact']), deadline)
        witness, statuses, reason = find_exact_witness(rows, rhs, bounds, deadline)
        result.update(witness_lp_statuses=statuses, witness_reason=reason)
        if witness is not None:
            result.update(status='exact_necessary_lp_feasible', witness=[qt(v) for v in witness])
    result['independent_validation'] = independent_verify(observation, cell, tile_id, result)
    if result['independent_validation']['verified'] is False:
        result['rejected_status'] = result['status']; result['status'] = 'unknown'
    result.update(wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu)
    return result


def _child(pipe, observation, cell, tile_id, seconds):
    try:
        pipe.send(diagnose_tile(observation, cell, tile_id, seconds))
    except Exception as exc:
        pipe.send({'tile_id': tile_id, 'status': 'unknown',
                   'error': f'{type(exc).__name__}: {exc}', 'independent_validation': {'verified': None}})
    finally:
        pipe.close()


def bounded_tile(observation, cell, tile_id, seconds):
    """Independent hard wall cap per tile, including child exact verification."""
    ctx = mp.get_context('fork')
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child, args=(child, observation, cell, tile_id, seconds))
    start = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    process.start(); child.close()
    answer = None
    try:
        remaining = max(0., seconds-(time.perf_counter()-start))
        if parent.poll(remaining):
            try:
                answer = parent.recv()
            except EOFError:
                pass
    finally:
        if answer is None:
            process.kill()
        process.join(timeout=max(.05, seconds-(time.perf_counter()-start)))
        if process.is_alive():
            process.kill(); process.join()
        parent.close()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if answer is None:
        answer = {'tile_id': tile_id, 'status': 'unknown', 'reason': 'hard_wall_timeout_or_worker_exit',
                  'independent_validation': {'verified': None}}
    answer.update(worker_exitcode=process.exitcode, hard_cap_seconds=seconds,
                  process_wall_seconds=time.perf_counter()-start,
                  process_cpu_seconds=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    return answer


def classify_cell(tiles):
    if len(tiles) != 4 or [t['tile_id'] for t in tiles] != [0, 1, 2, 3]:
        raise ValueError('all four tile identities required')
    statuses = [t['status'] for t in tiles]
    if any(s == 'exact_necessary_lp_feasible' for s in statuses):
        return 'necessary_relaxation_retains_at_least_one_tile'
    if all(s == 'exact_infeasible' for s in statuses):
        return 'all_four_exactly_infeasible_offline'
    return 'unresolved_exact_status'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--public', type=Path, default=PUBLIC)
    parser.add_argument('--arms', nargs='+', default=None)
    parser.add_argument('--tile-seconds', type=float, default=3.)
    parser.add_argument('--max-cells', type=int, default=None, help='smoke cap; never choose by truth')
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != 'liekkas':
        raise RuntimeError('wrong target host')
    if not 0 < args.tile_seconds <= 3.:
        raise ValueError('per-tile independent hard cap must be in (0,3] seconds')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('output must be a new file inside the A15 working directory')
    started, cpu = time.perf_counter(), time.process_time()
    load_kernels()
    observations = {r['observation_id']: {k: r['observation'][k] for k in ('photo', 'camera', 'appearance')}
                    for r in json.loads(args.public.read_text())['observations']}
    targets, sources, skipped = {}, {}, []
    for batch in args.batch:
        jobs = batch / 'jobs' if (batch / 'jobs').is_dir() else batch
        paths = sorted(jobs.glob('*.result.json'))
        if not paths:
            raise ValueError(f'no saved result jobs in {batch}')
        for path in paths:
            saved = json.loads(path.read_text())
            if args.arms is not None and saved['arm'] not in args.arms:
                continue
            sources[str(path)] = sha(path)
            state, oid = saved['state'], saved['observation_id']
            if oid not in observations:
                raise ValueError(f'public observation not found: {oid}')
            cells = boundary_cells(state)
            if not cells:
                skipped.append({'path': str(path), 'reason': 'no saved retained cells'})
            for cell in cells:
                key = oid, cell['branch'], cell['lo_exact'], cell['hi_exact']
                entry = targets.setdefault(key, {'observation_id': oid, 'cell': cell, 'sources': []})
                entry['sources'].append({'path': str(path), 'arm': saved['arm'],
                    'budget_seconds': saved.get('budget_seconds'), 'hull': [state['lower'], state['upper']],
                    'stop_reason': state.get('reason'), 'checked': cell.get('checked'),
                    'terminal': cell.get('terminal'), 'sides': cell['sides']})
    records = []
    for entry in list(targets.values())[:args.max_cells]:
        tiles = [bounded_tile(observations[entry['observation_id']], entry['cell'], tile_id,
                              args.tile_seconds) for tile_id in range(4)]
        record = {**entry, 'tiles': tiles, 'classification': classify_cell(tiles)}
        records.append(record)
        print(json.dumps({'observation_id': entry['observation_id'], 'cell': entry['cell'],
                          'classification': record['classification'],
                          'tile_statuses': [t['status'] for t in tiles]}), flush=True)
    result = {'schema': SCHEMA, 'host': socket.gethostname(), 'records': records,
              'scope': 'saved-hull offline diagnosis; exact feasibility means necessary LP only',
              'not_proved': ['physical feasibility', 'finer camera boxes will exclude this cell'],
              'public_path': str(args.public), 'public_sha256': sha(args.public),
              'input_result_sha256': sources, 'script_sha256': sha(__file__),
              'unique_target_cells': len(targets), 'diagnosed_cells': len(records), 'skipped': skipped,
              'all_targets_completed': len(records) == len(targets),
              'counts': dict(Counter(r['classification'] for r in records)),
              'tile_counts': dict(Counter(t['status'] for r in records for t in r['tiles'])),
              'process_cpu_seconds': sum(t['process_cpu_seconds'] for r in records for t in r['tiles']),
              'parent_cpu_seconds': time.process_time()-cpu, 'wall_seconds': time.perf_counter()-started,
              'tile_hard_cap_seconds': args.tile_seconds}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
