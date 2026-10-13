"""Exact trace, set-union and action arithmetic; imports no experiment policy."""
from fractions import Fraction as Q

DOMAIN = (Q(540), Q(660))
EPSILON, TOLERANCE, MAX_BOXES = Q(1, 100), Q(15, 32), 255


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def exact(value, where='rational'):
    if isinstance(value, dict):
        require(set(value) == {'numerator', 'denominator'}, f'{where}: invalid rational schema')
        n, d = value['numerator'], value['denominator']
        require(type(n) is int and type(d) is int and d > 0, f'{where}: invalid rational integers')
        q = Q(n, d)
        require(q.numerator == n and q.denominator == d, f'{where}: noncanonical rational')
        return q
    require(type(value) in (str, int) or isinstance(value, Q), f'{where}: exact rational required')
    try:
        return Q(value)
    except (ValueError, ZeroDivisionError) as error:
        raise VerificationError(f'{where}: invalid rational') from error


def encoded(value):
    q = exact(value)
    return {'numerator': q.numerator, 'denominator': q.denominator}


def pair(value, where):
    require(isinstance(value, (list, tuple)) and len(value) == 2, f'{where}: expected endpoint pair')
    low, high = (exact(v, where) for v in value)
    require(low <= high, f'{where}: reversed endpoints')
    return low, high


def indexed(records, key, where):
    require(isinstance(records, list), f'{where}: expected record list')
    result = {}
    for record in records:
        require(isinstance(record, dict), f'{where}: expected record object')
        ident = key(record)
        require(ident not in result, f'{where}: duplicate identity {ident!r}')
        result[ident] = record
    return result


def merged_intervals(intervals):
    """Canonical closed union; adjacent closed intervals are joined."""
    result = []
    for low, high in sorted(intervals):
        require(low <= high, 'union: reversed interval')
        if result and low <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], high))
        else:
            result.append((low, high))
    return result


def verify_partition(intervals, where):
    ordered = sorted(intervals)
    require(ordered and ordered[0][0] == DOMAIN[0] and ordered[-1][1] == DOMAIN[1],
            f'{where}: terminal partition does not span full domain')
    for low, high in ordered:
        require(DOMAIN[0] <= low < high <= DOMAIN[1], f'{where}: invalid leaf interval')
    require(all(first[1] == second[0] for first, second in zip(ordered, ordered[1:])),
            f'{where}: terminal partition gap or interior overlap')


def verify_trace(outer, observation, where):
    """Check one fixed scene branch; empty retained sets are valid outcomes."""
    fields = {'domain', 'epsilon', 'tolerance', 'max_boxes', 'evaluated_boxes', 'trace', 'retained_ids', 'hull', 'scope'}
    require(isinstance(outer, dict) and set(outer) == fields, f'{where}: invalid branch-outer schema')
    require(pair(outer['domain'], where) == DOMAIN and exact(outer['epsilon']) == EPSILON and
            exact(outer['tolerance']) == TOLERANCE, f'{where}: changed domain/noise/tolerance')
    require(type(outer['max_boxes']) is int and outer['max_boxes'] == MAX_BOXES, f'{where}: changed box budget')
    require(outer['scope'] == 'outer_set_for_continuous_depth_and_model_nuisance_contract', f'{where}: changed scope')
    nodes = indexed(outer['trace'], lambda n: n['id'], where)
    require(0 < len(nodes) <= 2 * MAX_BOXES + 1 and
            all(type(n['id']) is int for n in outer['trace']) and
            [n['id'] for n in outer['trace']] == list(range(len(nodes))), f'{where}: invalid BFS node identities')
    require(nodes[0]['parent'] is None, f'{where}: root has parent')
    boxes = {i: (exact(n['lo']), exact(n['hi'])) for i, n in nodes.items()}
    require(boxes[0] == DOMAIN, f'{where}: root is not full depth domain')
    leaves, retained, excluded, referenced = [], [], [], set()
    evaluated, next_child = 0, 1
    for n in outer['trace']:
        ident, status = n['id'], n['status']
        low, high = boxes[ident]
        require(DOMAIN[0] <= low < high <= DOMAIN[1], f'{where}: invalid depth box {ident}')
        extras = {'split': {'children'}, 'excluded': {'witness_channel', 'prediction_interval'},
                  'retained_tolerance': set(), 'unresolved_budget': set()}
        require(status in extras and set(n) == {'id', 'parent', 'lo', 'hi', 'status'} | extras[status],
                f'{where}: invalid node status/schema {ident}')
        if ident:
            require(type(n['parent']) is int and n['parent'] < ident and n['parent'] in nodes,
                    f'{where}: invalid parent {ident}')
        if status == 'unresolved_budget':
            require(evaluated == MAX_BOXES, f'{where}: unresolved leaf before budget exhaustion')
        else:
            require(evaluated < MAX_BOXES, f'{where}: evaluated beyond budget')
            evaluated += 1
        if status == 'split':
            require(high - low > TOLERANCE, f'{where}: split below tolerance')
            children = n['children']
            require(isinstance(children, list) and len(children) == 2 and children[0] != children[1] and
                    all(type(c) is int and c in nodes for c in children), f'{where}: invalid children')
            require(children == [next_child, next_child + 1], f'{where}: changed BFS allocation')
            next_child += 2
            mid = (low + high) / 2
            for child, expected in zip(children, ((low, mid), (mid, high))):
                require(child > ident and child not in referenced and nodes[child]['parent'] == ident and
                        boxes[child] == expected, f'{where}: non-bisection/duplicate child')
                referenced.add(child)
        else:
            leaves.append((low, high))
            if status == 'excluded':
                channel = n['witness_channel']
                require(type(channel) is int and 0 <= channel < len(observation), f'{where}: invalid witness channel')
                a, b = pair(n['prediction_interval'], where)
                require(0 <= a <= b <= 1, f'{where}: witness outside radiance range')
                require(a > observation[channel] + EPSILON or b < observation[channel] - EPSILON,
                        f'{where}: exclusion witness does not conflict with observed noise interval')
                excluded.append(n)
            else:
                require(status != 'retained_tolerance' or high - low <= TOLERANCE,
                        f'{where}: tolerance-retained box too wide')
                retained.append(n)
    require(referenced == set(nodes) - {0}, f'{where}: orphan trace nodes')
    verify_partition(leaves, where)
    require(type(outer['evaluated_boxes']) is int and outer['evaluated_boxes'] == evaluated,
            f'{where}: incorrect evaluated count')
    require(outer['retained_ids'] == [n['id'] for n in retained] and
            all(type(i) is int for i in outer['retained_ids']), f'{where}: incorrect retained IDs')
    intervals = [boxes[n['id']] for n in retained]
    hull = None if not intervals else (min(x[0] for x in intervals), max(x[1] for x in intervals))
    require((outer['hull'] is None and hull is None) or
            (outer['hull'] is not None and hull is not None and pair(outer['hull'], where) == hull),
            f'{where}: incorrect retained hull')
    return {'hull': hull, 'retained': retained, 'intervals': intervals, 'boxes': boxes,
            'excluded': excluded, 'evaluated': evaluated,
            'budget_leaves': sum(n['status'] == 'unresolved_budget' for n in outer['trace'])}


def verify_union(outer, branches, where, prior_only=False):
    fields = {'domain', 'epsilon', 'tolerance', 'max_boxes_per_branch', 'evaluated_boxes',
              'unresolved_budget_leaves', 'source_leaves', 'retained_intervals', 'hull',
              'retained_branches', 'scope'}
    require(isinstance(outer, dict) and set(outer) == fields, f'{where}: invalid union schema')
    require(pair(outer['domain'], where) == DOMAIN and exact(outer['epsilon']) == EPSILON and
            exact(outer['tolerance']) == TOLERANCE and type(outer['max_boxes_per_branch']) is int and
            outer['max_boxes_per_branch'] == MAX_BOXES, f'{where}: changed union contract')
    source = [{'branch': name, 'id': node['id'], 'lo': node['lo'], 'hi': node['hi']}
              for name, checked in branches.items() for node in checked['retained']]
    require(isinstance(outer['source_leaves'], list), f'{where}: invalid source leaves')
    for node in outer['source_leaves']:
        require(isinstance(node, dict) and set(node) == {'branch', 'id', 'lo', 'hi'} and
                type(node['branch']) is str and type(node['id']) is int,
                f'{where}: invalid branch source leaf')
        pair((node['lo'], node['hi']), where)
    intervals = [DOMAIN] if prior_only else merged_intervals(
        interval for checked in branches.values() for interval in checked['intervals'])
    hull = None if not intervals else (intervals[0][0], intervals[-1][1])
    evaluated = sum(checked['evaluated'] for checked in branches.values())
    unresolved = sum(checked['budget_leaves'] for checked in branches.values())
    retained_branches = [name for name, checked in branches.items() if checked['hull'] is not None]
    require(not prior_only or not branches, f'{where}: KEEP unexpectedly solves branches')
    require(outer['source_leaves'] == source, f'{where}: altered or pixel-mixed branch provenance')
    require([pair(value, where) for value in outer['retained_intervals']] == intervals,
            f'{where}: incorrect closed depth union')
    require((outer['hull'] is None and hull is None) or
            (outer['hull'] is not None and hull is not None and pair(outer['hull'], where) == hull),
            f'{where}: incorrect union hull')
    require(type(outer['evaluated_boxes']) is int and outer['evaluated_boxes'] == evaluated and
            type(outer['unresolved_budget_leaves']) is int and outer['unresolved_budget_leaves'] == unresolved and
            outer['retained_branches'] == retained_branches, f'{where}: incorrect union counts')
    require(outer['scope'] == ('prior_only_identity_no_image_solver' if prior_only else
            'union_of_branch_depth_outer_leaves_then_convex_hull'), f'{where}: changed union scope')
    return {'hull': hull, 'intervals': intervals, 'retained_branches': retained_branches,
            'evaluated': evaluated, 'budget_leaves': unresolved}


def endpoint_action(hull, incumbent, proposal):
    a = exact(incumbent)
    if hull is None:
        return {'status': 'INCOMPATIBLE', 'output': a, 'gain_lower': None, 'reason': 'empty_certified_outer_set'}
    left, right = hull
    b = (left + right) / 2 if proposal == 'midpoint' else min(max(a, left), right)
    gain = min((a - left)**2 - (b - left)**2, (a - right)**2 - (b - right)**2)
    move = a != b and gain > 0
    return {'status': 'MOVE' if move else 'KEEP', 'output': b if move else a,
            'gain_lower': gain if move else Q(0),
            'reason': 'strict_gain_over_continuous_outer_hull' if move else 'no_strict_safe_gain_for_proposal'}


def hybrid_action(hull, incumbent):
    midpoint = endpoint_action(hull, incumbent, 'midpoint')
    if midpoint['status'] == 'MOVE':
        return {**midpoint, 'branch': 'midpoint'}
    projected = endpoint_action(hull, incumbent, 'projection')
    return {**projected, 'branch': {'MOVE': 'projection', 'KEEP': 'keep',
                                  'INCOMPATIBLE': 'incompatible'}[projected['status']]}


def verify_decision(decision, incumbent, hull, where):
    method = decision['method']
    expected = ({'status': 'KEEP', 'output': exact(incumbent), 'gain_lower': Q(0),
                 'reason': 'identity'} if method == 'keep' else hybrid_action(hull, incumbent))
    require(set(decision) == {'task_id', 'method'} | set(expected), f'{where}: invalid decision schema')
    for field, value in expected.items():
        actual = decision[field]
        if field in {'output', 'gain_lower'} and actual is not None:
            actual = exact(actual, where)
        require(actual == value, f'{where}: incorrect {field}')
    return exact(decision['output'])
