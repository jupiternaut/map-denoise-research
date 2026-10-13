"""One globally selected, unmodified fixed-library estimator.

Selection is frozen during development, never made from runtime truth,
incumbents, IDs, hidden strata, or results from other candidate methods.
"""
from copy import deepcopy
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / 'library'
if str(LIBRARY) not in sys.path:
    sys.path.insert(0, str(LIBRARY))
import methods as _library

if Path(_library.__file__).resolve() != (LIBRARY / 'methods.py').resolve():
    raise RuntimeError('wrong fixed library identity')

# Single permitted development revision: pooled six-observation MSE chose
# depth. This constant is frozen for both six- and twenty-four-second runs.
SELECTED_METHOD = 'depth'
SELECTION_VERSION = 'global-dev-mse-v1'
_prepare_method = _library.prepare_method


def prepare(observation, budget_seconds):
    budget = float(budget_seconds)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('positive finite preparation budget required')
    # Metadata outside the registered measurement fields cannot select a name
    # or become an implicit estimator input. The library does its own parsing.
    public = deepcopy({key: observation[key] for key in ('photo', 'camera', 'appearance')})
    state = _prepare_method(SELECTED_METHOD, public, budget)
    state['selector'] = {'version': SELECTION_VERSION, 'selected_method': SELECTED_METHOD,
                         'rule': 'one frozen global name; no per-observation fit selection'}
    return state


def decode(state, incumbent):
    value = float(incumbent)
    if not math.isfinite(value):
        raise ValueError('finite incumbent required')
    answer = float(_library.decode(state, value))
    if not math.isfinite(answer):
        raise ValueError('fixed library returned a nonfinite depth')
    return min(660.0, max(540.0, answer))
