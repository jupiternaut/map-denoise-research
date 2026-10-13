"""A14 observation-only search adapters. Ancestor code stays read-only."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

A13_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z')
sys.path.append(str(A13_ROOT))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

OLD = load_module('a14_frozen_a13_search', A13_ROOT/'search_method.py')
FREE = OLD.FREE

def warm_load(arm):
    global JOINT
    if arm == 'legacy':
        OLD.warm_load('legacy')
    else:
        OLD.warm_load('certified')
        if arm == 'joint':
            import joint_kernel as JOINT

def prepare(observation, budget_seconds, arm):
    if arm == 'legacy':
        return OLD.prepare(observation, budget_seconds, 'legacy')
    if arm not in ('base','fine','joint'):
        raise ValueError(arm)
    old_tolerance = FREE.CELL_TOLERANCE
    if arm != 'joint':
        FREE.CELL_TOLERANCE = .001 if arm == 'fine' else .025
        try:
            s = OLD.prepare(observation, budget_seconds, 'certified')
            s['arm'] = arm
            return s
        finally:
            FREE.CELL_TOLERANCE = old_tolerance

    start = time.perf_counter(); end = start + max(0., float(budget_seconds))
    attempts, exclusions = [], []
    timings = dict(context_seconds=0., floating_proposal_seconds=0.,
                   certification_seconds=0., tile_proposal_seconds=0.)
    ctx = tiled = None
    if budget_seconds > .04:
        t = time.perf_counter()
        ctx = OLD.CERT.ExactContext(observation)
        tiled = JOINT.JointContext(observation)
        timings['context_seconds'] = time.perf_counter()-t
    original = FREE._lp_for_cell

    def classify(encoded, branch, lo, hi, limit):
        t = time.perf_counter()
        ans = original(encoded, branch, lo, hi, limit)
        timings['floating_proposal_seconds'] += time.perf_counter()-t
        mode = 'whole'
        if ans.status != 2:
            if ans.status != 0 or tiled is None:
                return ans
            t = time.perf_counter(); statuses=[]
            for cameras in tiled.tiles:
                remain = end-time.perf_counter()-.025
                if remain <= .01:
                    statuses.append(4); break
                tile_encoded = (*encoded[:-1], [(float(a),float(b)) for a,b in cameras])
                reply = original(tile_encoded,branch,lo,hi,max(.001,min(.2,remain)))
                statuses.append(int(reply.status))
                if reply.status != 2:
                    break
            timings['tile_proposal_seconds'] += time.perf_counter()-t
            if len(statuses)!=4 or any(x!=2 for x in statuses):
                return ans
            mode = 'tiles'
        t=time.perf_counter()
        if ctx is None or t >= end-.025:
            proof={'verified':False,'certificate':None,'reason':'deadline'}
        else:
            proof=(ctx if mode=='whole' else tiled).check_cell(branch,lo,hi,end-.025)
        timings['certification_seconds']+=time.perf_counter()-t
        entry={'branch':branch,'lower':float(lo),'upper':float(hi),'mode':mode,**proof}
        attempts.append(entry)
        if proof['verified']:
            exclusions.append(entry); return SimpleNamespace(status=2)
        return SimpleNamespace(status=4)

    FREE.CELL_TOLERANCE=.025; FREE._lp_for_cell=classify
    try:
        state=FREE.prepare(deepcopy(observation),max(0.,end-time.perf_counter()))
    finally:
        FREE.CELL_TOLERANCE=old_tolerance; FREE._lp_for_cell=original
    state.update(arm=arm,certificate_schema='A14-v1',exclusions=exclusions,
        certificate_attempts=attempts,component_seconds=timings,
        total_prepare_seconds=time.perf_counter()-start,
        original_requested_budget_seconds=float(budget_seconds))
    return state

def decode(state, incumbent, arm):
    return OLD.decode(state,incumbent,'legacy') if arm=='legacy' else FREE.decode(state,incumbent)
