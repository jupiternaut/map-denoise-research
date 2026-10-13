"""Certified shared-material necessity test, not an exact joint scene solver."""
from collections import deque
from fractions import Fraction as Q
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'lovelace-nuisance-20261011T044500Z'
sys.path.insert(0, str(OLD))
from nuisance_model import ColoredModel, PUBLIC_COLORS, _geometry_bounds
from run_nuisance import union_outer
from continuous_world.interval_solver import encode, decode
sys.path.insert(0, str(ROOT))

EPSILON = Q(1, 100)
TOLERANCE = Q(15, 32)
MAX_BOXES = 255
ROUNDS = 8


def material_rows(observation, geometry, epsilon=EPSILON):
    """Necessary halfspaces in shared d=1-c; coefficients are nonnegative.

    RGB are noisy replicates of a common grayscale, so intersect their bands.
    Rows are upper(L dot d <= b) then lower(U dot d >= b) for every pixel.
    """
    image = tuple(map(Q, observation))
    if len(image) != 3 * len(geometry):
        raise ValueError('RGB layout and area bounds must agree')
    result = []
    for p, bounds in enumerate(geometry):
        ys = image[3*p:3*p+3]
        low = max(1-y-epsilon for y in ys)
        high = min(1-y+epsilon for y in ys)
        result.append(('upper', tuple(bounds[::2]), high))
        result.append(('lower', tuple(bounds[1::2]), low))
    return tuple(result)


def propagate(rows, colors=PUBLIC_COLORS, rounds=ROUNDS):
    """A bounded exact contractor; only an explicit row contradiction excludes.

    This is intentionally incomplete. Feasible/no-contradiction means retain,
    never means that a single depth/pose/color assignment has been found.
    """
    lower = [1-Q(high) for _low, high in colors]
    upper = [1-Q(low) for low, _high in colors]
    if len(lower) != 3 or any(not 0 <= a <= b <= 1 for a, b in zip(lower, upper)):
        raise ValueError('three valid material intervals required')
    steps = []
    for iteration in range(rounds):
        changed = False
        for index, (kind, coefficients, bound) in enumerate(rows):
            if len(coefficients) != 3 or any(a < 0 for a in coefficients):
                raise ValueError('necessary rows must have three nonnegative coefficients')
            values = lower if kind == 'upper' else upper
            total = sum((a*v for a, v in zip(coefficients, values)), Q(0))
            conflict = total > bound if kind == 'upper' else total < bound
            if conflict:
                return {'excluded': True, 'certificate': {'steps': steps,
                        'contradiction': {'row': index}}, 'rounds': iteration+1}
            for j, a in enumerate(coefficients):
                if a == 0:
                    continue
                other = total-a*values[j]
                candidate = (bound-other)/a
                if kind == 'upper' and candidate < upper[j]:
                    upper[j] = candidate
                    steps.append({'row': index, 'variable': j, 'side': 'upper', 'value': encode(candidate)})
                    changed = True
                elif kind == 'lower' and candidate > lower[j]:
                    lower[j] = candidate
                    steps.append({'row': index, 'variable': j, 'side': 'lower', 'value': encode(candidate)})
                    changed = True
        if not changed:
            break
    return {'excluded': False, 'rounds': iteration+1,
            'darkness_bounds': [[encode(a), encode(b)] for a, b in zip(lower, upper)],
            'updates': len(steps)}


def solve_branch(observation, model, max_boxes=MAX_BOXES, tolerance=TOLERANCE):
    """Depth BFS, old interval witnesses first, then shared-material witnesses."""
    image = tuple(map(Q, observation))
    queue = deque([(0, None, Q(540), Q(660))])
    records, retained = [], []
    next_id, evaluated, shared_calls = 1, 0, 0
    while queue:
        node_id, parent, left, right = queue.popleft()
        entry = {'id': node_id, 'parent': parent, 'lo': encode(left), 'hi': encode(right)}
        if evaluated >= max_boxes:
            entry['status'] = 'unresolved_budget'
            records.append(entry)
            retained.append((left, right, node_id))
            continue
        bounds = model.enclosure(left, right)
        evaluated += 1
        conflict = next(((i,a,b) for i,(a,b) in enumerate(bounds)
                         if a > image[i]+EPSILON or b < image[i]-EPSILON), None)
        if conflict is not None:
            i,a,b = conflict
            entry.update(status='excluded', witness_channel=i,
                         prediction_interval=[encode(a), encode(b)])
            records.append(entry)
            continue
        rows = material_rows(image, _geometry_bounds(model.branch, model.camera_radius, left, right))
        contraction = propagate(rows, model.colors)
        shared_calls += 1
        if contraction['excluded']:
            entry.update(status='excluded_shared', certificate=contraction['certificate'])
        elif right-left <= tolerance:
            entry['status'] = 'retained_tolerance'
            retained.append((left,right,node_id))
        else:
            middle = (left+right)/2
            entry.update(status='split', children=[next_id, next_id+1])
            queue.append((next_id,node_id,left,middle))
            queue.append((next_id+1,node_id,middle,right))
            next_id += 2
        records.append(entry)
    hull = None if not retained else [encode(min(a for a,_,_ in retained)), encode(max(b for _,b,_ in retained))]
    return {'domain': [encode(Q(540)), encode(Q(660))], 'epsilon': encode(EPSILON),
            'tolerance': encode(tolerance), 'max_boxes': max_boxes, 'evaluated_boxes': evaluated,
            'trace': records, 'retained_ids': [i for _,_,i in retained], 'hull': hull,
            'shared_tests': shared_calls, 'propagation_rounds': ROUNDS,
            'scope': 'necessary_shared_material_halfspaces_with_conservative_continuous_geometry_bounds'}


def solve_shared(observation, radius, branches, colors=PUBLIC_COLORS):
    started = time.perf_counter()
    outers, seconds = {}, {}
    for branch in branches:
        tick = time.perf_counter()
        outers[branch] = solve_branch(observation, ColoredModel(branch, camera_radius=radius, colors=colors))
        seconds[branch] = time.perf_counter()-tick
    return {'branches': outers, 'outer': union_outer(outers), 'branch_seconds': seconds,
            'outer_seconds': time.perf_counter()-started}
