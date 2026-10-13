"""A13: keep the frozen search, optionally verify every proposed exclusion."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

PARENT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z')
sys.path.insert(0, str(PARENT / 'task_bundle'))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

FREE = load_module('a13_frozen_free', PARENT / 'arms/free/method.py')

def prepare(observation, budget_seconds, arm):
    if arm == 'legacy':
        # Dependency import is charged by worker initialization via warm_load.
        return LEGACY.prepare_method('accumulated', observation, budget_seconds)
    if arm not in ('numeric', 'certified'):
        raise ValueError(arm)
    start = time.perf_counter()
    deadline = start + max(0., float(budget_seconds))
    attempts, exclusions = [], []
    timings = {'floating_proposal_seconds': 0., 'context_seconds': 0.,
               'certification_seconds': 0.}
    context = None
    if arm == 'certified' and budget_seconds > .04:
        t = time.perf_counter()
        context = CERT.ExactContext(observation)
        timings['context_seconds'] = time.perf_counter() - t
    original = FREE._lp_for_cell

    def classify(encoded, branch, lo, hi, time_limit):
        t = time.perf_counter()
        answer = original(encoded, branch, lo, hi, time_limit)
        timings['floating_proposal_seconds'] += time.perf_counter() - t
        if answer.status != 2:
            return answer
        entry = {'branch': branch, 'lower': float(lo), 'upper': float(hi)}
        if arm == 'numeric':
            entry.update(verified=False, certificate=None, reason='floating_only')
            exclusions.append(entry)
            return answer
        t = time.perf_counter()
        if context is None or t >= deadline - .025:
            proof = {'verified': False, 'certificate': None, 'reason': 'deadline'}
        else:
            proof = context.check_cell(branch, lo, hi, deadline - .025)
        timings['certification_seconds'] += time.perf_counter() - t
        entry.update(proof)
        attempts.append(entry)
        if entry['verified']:
            exclusions.append(entry)
            return answer
        # Same frozen unknown-result path: retain this cell as terminal.
        return SimpleNamespace(status=4)

    FREE._lp_for_cell = classify
    try:
        state = FREE.prepare(deepcopy(observation), max(0., deadline-time.perf_counter()))
    finally:
        FREE._lp_for_cell = original
    state.update(arm=arm, certificate_schema='A13-v1', exclusions=exclusions,
                 certificate_attempts=attempts, component_seconds=timings,
                 total_prepare_seconds=time.perf_counter()-start,
                 original_requested_budget_seconds=float(budget_seconds))
    return state

def warm_load(arm):
    global CERT, LEGACY
    if arm == 'certified':
        import cert_kernel as CERT
    elif arm == 'legacy':
        LEGACY = load_module('a13_frozen_library', PARENT / 'library/methods.py')

def decode(state, incumbent, arm):
    return LEGACY.decode(state, incumbent) if arm == 'legacy' else FREE.decode(state, incumbent)
