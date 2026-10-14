"""A15 actual classify-path ablation: hot order and exact tile-weight reuse.

No-switch path preserves A14 calls, four tile order, unknown handling and
boundary search. Independent counters expose actual inner LP costs.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import reuse_kernel as REUSE

A14_ROOT = REUSE.A14_ROOT
_spec = importlib.util.spec_from_file_location('_a15_frozen_a14_search', A14_ROOT/'search_method.py')
OLD = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = OLD
_spec.loader.exec_module(OLD)
FREE = OLD.FREE


def warm_load(*_args):
    OLD.warm_load('base')


def prepare(observation, budget_seconds, tolerance=.001, hot=False, inherit=False):
    if not (0. < float(tolerance) <= 120.):
        raise ValueError('tolerance must be positive and bounded')
    if not hasattr(OLD.OLD, 'CERT'):
        warm_load()
    old_tolerance = FREE.CELL_TOLERANCE
    start = time.perf_counter(); end = start+max(0., float(budget_seconds))
    attempts, exclusions = [], []
    timings = dict(context_seconds=0., floating_proposal_seconds=0.,
                   certification_seconds=0., tile_proposal_seconds=0.,
                   whole_certification_seconds=0., tile_certification_seconds=0.,
                   whole_certificate_search_seconds=0., tile_certificate_search_seconds=0.)
    counts = dict(whole_float_lp=0, tile_float_lp=0, whole_certificate_calls=0,
                  whole_certificate_lp=0, tile_cover_certificate_calls=0,
                  hot_hint_uses=0, hot_hint_updates=0, hot_hint_unknown_returns=0)
    hot_tile, order_trace = {}, []
    ctx = tiled = None
    if budget_seconds > .04:
        t = time.perf_counter()
        ctx = OLD.OLD.CERT.ExactContext(observation)
        tiled = REUSE.JointContext(observation, inherit=inherit)
        timings['context_seconds'] = time.perf_counter()-t
    original = FREE._lp_for_cell

    def classify(encoded, branch, lo, hi, limit):
        t = time.perf_counter()
        ans = original(encoded, branch, lo, hi, limit)
        counts['whole_float_lp'] += 1
        timings['floating_proposal_seconds'] += time.perf_counter()-t
        mode = 'whole'
        if ans.status != 2:
            if ans.status != 0 or tiled is None:
                return ans
            t = time.perf_counter(); statuses = []; tested = []
            preferred = hot_tile.get(branch) if hot else None
            order = ([preferred]+[i for i in range(4) if i != preferred]
                     if preferred is not None else list(range(4)))
            counts['hot_hint_uses'] += int(preferred is not None)
            for tile_id in order:
                cameras = tiled.tiles[tile_id]
                remain = end-time.perf_counter()-.025
                if remain <= .01:
                    statuses.append(4); break
                tile_encoded = (*encoded[:-1], [(float(a), float(b)) for a, b in cameras])
                reply = original(tile_encoded, branch, lo, hi, max(.001, min(.2, remain)))
                counts['tile_float_lp'] += 1
                statuses.append(int(reply.status)); tested.append(tile_id)
                if reply.status != 2:
                    # Only a floating feasible proposal becomes a scheduling hint.
                    # Unknown never becomes proof, and retains A14 whole-result semantics.
                    if hot and reply.status == 0:
                        hot_tile[branch] = tile_id
                        counts['hot_hint_updates'] += 1
                    elif hot:
                        counts['hot_hint_unknown_returns'] += 1
                    break
            timings['tile_proposal_seconds'] += time.perf_counter()-t
            order_trace.append(dict(branch=branch, lower=float(lo), upper=float(hi),
                                    order=tested, statuses=statuses, preferred=preferred))
            if len(statuses) != 4 or any(x != 2 for x in statuses):
                return ans
            mode = 'tiles'
        t = time.perf_counter()
        if ctx is None or t >= end-.025:
            proof = {'verified': False, 'certificate': None, 'reason': 'deadline'}
        else:
            if mode == 'whole':
                counts['whole_certificate_calls'] += 1
            else:
                counts['tile_cover_certificate_calls'] += 1
            proof = (ctx if mode == 'whole' else tiled).check_cell(branch, lo, hi, end-.025)
        elapsed = time.perf_counter()-t
        timings['certification_seconds'] += elapsed
        prefix = 'whole' if mode == 'whole' else 'tile'
        timings[prefix+'_certification_seconds'] += elapsed
        timings[prefix+'_certificate_search_seconds'] += proof.get('certificate_search_seconds', 0.)
        if mode == 'whole':
            counts['whole_certificate_lp'] += int(proof.get('lp_status') is not None)
        entry = {'branch': branch, 'lower': float(lo), 'upper': float(hi), 'mode': mode, **proof}
        attempts.append(entry)
        if proof['verified']:
            exclusions.append(entry)
            return SimpleNamespace(status=2)
        return SimpleNamespace(status=4)

    FREE.CELL_TOLERANCE = float(tolerance); FREE._lp_for_cell = classify
    try:
        state = FREE.prepare(deepcopy(observation), max(0., end-time.perf_counter()))
    finally:
        FREE.CELL_TOLERANCE = old_tolerance; FREE._lp_for_cell = original
    state.update(arm='joint_reuse', tolerance=float(tolerance), hot=bool(hot), inherit=bool(inherit),
        certificate_schema='A14-v1', exclusions=exclusions, certificate_attempts=attempts,
        component_seconds=timings, inner_counts=counts, tile_order_trace=order_trace,
        tile_reuse=tiled.counters() if tiled is not None else {'counts': {}, 'seconds': {}, 'per_tile': []},
        total_prepare_seconds=time.perf_counter()-start,
        original_requested_budget_seconds=float(budget_seconds))
    return state


def decode(state, incumbent, *_args):
    return FREE.decode(state, incumbent)
