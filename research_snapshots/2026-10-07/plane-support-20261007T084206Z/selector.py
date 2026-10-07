"""Fixed rule on sanitized observations only; no reference/evaluation imports."""
import math
import numpy as np


def max_ncc(obj):
    values = [float(v) for h in obj['hypotheses']
              for v in h['scores']['U11']['ncc'] if v is not None and np.isfinite(v)]
    return max(values, default=-1.0)


def support(objects, intervals, protocol, point=False):
    n = len(intervals)
    tau = protocol['support_tolerance_mm']
    active = [i for i in intervals if i['informative']]
    rows = []
    for o in objects:
        z = o['optical_depth_mm']
        lower = upper = 0
        for item in active:
            lo = hi = item['center_mm']
            if not point:
                lo, hi = item['lo_mm'], item['hi_mm']
            lower += lo >= z - tau and hi <= z + tau
            upper += hi >= z - tau and lo <= z + tau
        rows.append(dict(candidate_id=o['candidate_id'], optical_depth_mm=z,
                         certain=lower / max(n, 1), possible=upper / max(n, 1),
                         certain_count=lower, possible_count=upper, max_single_ncc=max_ncc(o)))
    return rows


def decide(objects, intervals, protocol, baseline_id, point=False, joint=False):
    scores = support(objects, intervals, protocol, point)
    scores.sort(key=lambda r: (-r['certain'], -r['max_single_ncc'], r['candidate_id']))
    winner = scores[0] if scores else None
    incumbent = next((o for o in objects if o['candidate_id'] == -1), None)
    reason = 'SUPPORT_OVERRIDE'
    separation = protocol['separated_depth_mm']
    far = [] if winner is None else [r for r in scores if r['candidate_id'] != winner['candidate_id']
        and abs(r['optical_depth_mm'] - winner['optical_depth_mm']) >= separation]
    margin = None if not far or winner is None else winner['certain'] - max(r['possible'] for r in far)
    if len(intervals) < protocol['minimum_valid_neighbors']:
        reason = 'FALLBACK_FEW_NEIGHBORS'
    elif winner is None or winner['candidate_id'] == -1:
        reason = 'FALLBACK_INCUMBENT_WINNER'
    elif incumbent is not None and abs(winner['optical_depth_mm'] - incumbent['optical_depth_mm']) < separation:
        reason = 'FALLBACK_NEAR_INCUMBENT'
    elif winner['certain'] < protocol['minimum_certain_fraction']:
        reason = 'FALLBACK_WEAK_SUPPORT'
    elif far and margin <= protocol['support_margin']:
        reason = 'FALLBACK_OVERLAPPING_SUPPORT'
    elif joint and winner['max_single_ncc'] < protocol['one_view_ncc_gate']:
        reason = 'FALLBACK_PHOTO_GATE'
    accepted = reason == 'SUPPORT_OVERRIDE'
    return dict(selected_candidate_id=winner['candidate_id'] if accepted else (baseline_id if joint else None),
                winning_candidate_id=None if winner is None else winner['candidate_id'], reason=reason,
                accepted=accepted, far_competitor_ids=[r['candidate_id'] for r in far], robust_margin=margin,
                valid_neighbors=len(intervals), informative_neighbors=sum(i['informative'] for i in intervals),
                scores=scores)


def decisions(objects, intervals, protocol, baseline_id):
    out = {'photo_U11': dict(selected_candidate_id=baseline_id, reason='FROZEN_PHOTO_RULE')}
    for joint in (False, True):
        for point in (True, False):
            name = ('joint_' if joint else 'neighbor_') + ('point' if point else 'interval')
            out[name] = decide(objects, intervals, protocol, baseline_id, point, joint)
    return out
