"""Midpoint-first composition of two frozen conditional certificates."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
DEPENDENCY_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z')
sys.path.insert(0, str(DEPENDENCY_ROOT))
from challenge import ProfiledResidual, model_for, certificate_projection
from continuous_world.interval_solver import decide, encode, decode, solve_outer


def certificate_hybrid(outer, incumbent):
    """Keep an accepted midpoint; otherwise use a certified projection fallback.

    Selection uses only the containing outer set and incumbent, never truth.
    Validity is conditional on the shared model/noise/coverage contract.
    """
    midpoint = decide(outer, incumbent)
    if midpoint['status'] == 'MOVE':
        return {**midpoint, 'branch': 'midpoint'}
    projected = certificate_projection(outer, incumbent)
    branch = {'MOVE': 'projection', 'KEEP': 'keep', 'INCOMPATIBLE': 'incompatible'}[projected['status']]
    return {**projected, 'branch': branch}
