"""Certified depth outer approximation over continuous model parameters.

Only the model-supplied rigorous enclosure permits exclusion. Midpoint/endpoint
predictions cannot delete a box. All arithmetic is rational and open work stays
in the outer set if a work budget is exhausted.
"""
from collections import deque
from fractions import Fraction as Q


def rational(value):
    if isinstance(value, (float, bool)):
        raise TypeError("exact rational input required")
    return Q(value)


def encode(value):
    q = rational(value)
    return {"numerator": q.numerator, "denominator": q.denominator}


def decode(value):
    return Q(value["numerator"], value["denominator"])


def solve_outer(observation, enclosure, lo=540, hi=660, epsilon=Q(1,100),
                tolerance=Q(15,32), max_boxes=511):
    """Return lossless partition trace and all retained continuous intervals.

    enclosure(lo,hi)->tuple[(lower,upper),...] must cover all model nuisance
    parameters. Its validity is an external mathematical/model contract.
    Both bisection children contain their shared endpoint; duplicate boundary
    membership is safe. An empty retained set is proved incompatible under the
    contract; it is not a vacuous positive-gain certificate.
    """
    obs = tuple(map(rational, observation))
    lo, hi, epsilon, tolerance = map(rational, (lo,hi,epsilon,tolerance))
    if not obs or lo > hi or epsilon < 0 or tolerance <= 0:
        raise ValueError("nonempty observation, ordered domain, valid error/tolerance required")
    if not isinstance(max_boxes,int) or isinstance(max_boxes,bool) or max_boxes < 0:
        raise ValueError("max_boxes must be a nonnegative integer")
    queue = deque([(0, None, lo, hi)])
    records, retained = [], []
    next_id, evaluated = 1, 0
    while queue:
        node_id, parent, left, right = queue.popleft()
        entry = {"id":node_id,"parent":parent,"lo":encode(left),"hi":encode(right)}
        if evaluated >= max_boxes:
            entry["status"] = "unresolved_budget"
            records.append(entry)
            retained.append((left,right,node_id))
            continue
        bounds = tuple((rational(a),rational(b)) for a,b in enclosure(left,right))
        evaluated += 1
        if len(bounds) != len(obs) or any(a>b for a,b in bounds):
            raise ValueError("enclosure must match dimension with ordered intervals")
        exclusion = next(((i,a,b) for i,(a,b) in enumerate(bounds)
                          if a > obs[i]+epsilon or b < obs[i]-epsilon),None)
        if exclusion is not None:
            i,a,b = exclusion
            entry.update(status="excluded", witness_channel=i,
                         prediction_interval=[encode(a),encode(b)])
        elif right-left <= tolerance:
            entry["status"] = "retained_tolerance"
            retained.append((left,right,node_id))
        else:
            middle = (left+right)/2
            entry.update(status="split", children=[next_id,next_id+1])
            queue.append((next_id,node_id,left,middle))
            queue.append((next_id+1,node_id,middle,right))
            next_id += 2
        records.append(entry)
    hull = None if not retained else (min(a for a,_,_ in retained),max(b for _,b,_ in retained))
    return {"domain":[encode(lo),encode(hi)],"epsilon":encode(epsilon),
            "tolerance":encode(tolerance),"max_boxes":max_boxes,"evaluated_boxes":evaluated,
            "trace":records,"retained_ids":[i for _,_,i in retained],
            "hull":None if hull is None else [encode(q) for q in hull],
            "scope":"outer_set_for_continuous_depth_and_model_nuisance_contract"}


def decide(outer, incumbent, proposal=None):
    """Exact endpoint gain certificate; default proposal is outer-hull midpoint."""
    a = rational(incumbent)
    hull = outer["hull"]
    if hull is None:
        return {"status":"INCOMPATIBLE","output":encode(a),"gain_lower":None,
                "reason":"empty_certified_outer_set"}
    left,right = map(decode,hull)
    b = (left+right)/2 if proposal is None else rational(proposal)
    gain = min((a-left)**2-(b-left)**2,(a-right)**2-(b-right)**2)
    if b != a and gain > 0:
        return {"status":"MOVE","output":encode(b),"gain_lower":encode(gain),
                "reason":"strict_gain_over_continuous_outer_hull"}
    return {"status":"KEEP","output":encode(a),"gain_lower":encode(Q(0)),
            "reason":"no_strict_safe_gain_for_proposal"}
