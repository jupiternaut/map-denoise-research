"""A11 fixed-budget accumulation on the verified A10 depth closed support.

This is a measurement-only adapter, not a new geometry or certificate engine.
The frozen A10 kernel retains every unresolved cell and excludes a depth cell
only after exact contradictions cover every camera tile. An outer feasible
point is not a physical forward witness. The runner verifies the provenance
of the supplied A10 depth support and places the new arm name outside this
unchanged A10 branch-record schema.
"""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
FROZEN_ROOT = Path(
    '/srv/slam-research/grf/map-denoise/runs/'
    'lovelace-baseline-intervention-20261012T045547Z/continuous-tightening-1'
).resolve(strict=True)
if str(FROZEN_ROOT) not in sys.path:
    sys.path.insert(0, str(FROZEN_ROOT))
import tightening_solver as kernel

# The logical runs path is a symlink. Compare resolved file identities, not
# an unresolved historical spelling against the current physical spelling.
if Path(kernel.__file__).resolve(strict=True) != (FROZEN_ROOT / 'tightening_solver.py').resolve(strict=True):
    raise RuntimeError('wrong frozen A10 solver identity')

ARMS = ('D_then_D', 'D_then_DC')
MODES = {'D_then_D': 'depth', 'D_then_DC': 'camera_tiles'}
BRANCHES = kernel.BRANCHES
MIN_WIDTH = Q(15, 1024)
MAX_CLASSIFICATIONS = 128
BRANCH_SECONDS = 6.0
exact, encode = kernel.exact, kernel.encode
merge_intervals, select_depth = kernel.merge_intervals, kernel.select_depth
clear_geometry_cache = kernel.clear_geometry_cache
_kernel_solve = kernel.solve_branch


def normalize_inputs(public, oldarm, anchors, source_branch):
    """Copy only registered measurements and the inherited depth support.

    IDs, truth, case/unit/eta labels, incumbents, old trace/cost metadata and
    any other unregistered fields are never forwarded to the kernel. The
    selected measurement containers are copied too, so the adapter does not
    expose the caller's mutable lists to the kernel.
    """
    if not isinstance(source_branch, dict):
        raise TypeError('source branch must be a frozen A10 depth record')
    if source_branch.get('mode') != 'depth':
        raise ValueError('A11 requires the A10 depth arm as its common source')
    branch = source_branch['branch']
    if branch not in BRANCHES:
        raise ValueError('unregistered occlusion branch')
    photo, calibration, measurements = kernel.baseline.strip_method_inputs(
        public, oldarm, anchors, 'both')
    support = merge_intervals(source_branch['outer_intervals'])
    source = {'branch': branch,
              'outer_intervals': [[encode(lo), encode(hi)] for lo, hi in support]}
    return deepcopy(photo), deepcopy(calibration), deepcopy(measurements), source


def solve_branch(public, oldarm, anchors, source_branch, arm, deadline=None):
    """Continue A10-D with one registered arm and the same incremental cap.

    Per-arm cache clearing belongs to the runner. No test-budget override is
    exposed here: both arms always use 15/1024 mm, 128 calls and six seconds.
    The absolute shared deadline is forwarded unchanged; kernel deadlines
    are cooperative and do not claim operating-system preemption.
    """
    if arm not in ARMS:
        raise ValueError(f'arm must be one of {ARMS}')
    photo, calibration, measurements, source = normalize_inputs(
        public, oldarm, anchors, source_branch)
    return _kernel_solve(photo, calibration, measurements, source, MODES[arm],
                         deadline=deadline, max_classifications=MAX_CLASSIFICATIONS,
                         tolerance=MIN_WIDTH, branch_seconds=BRANCH_SECONDS)
