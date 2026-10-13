"""A14 four-camera-cover tightening of A13, with rational certificates.

Only the frozen A13 necessary-condition kernel is reused. No historical solver
stack, training inputs, task list, or truth file is read by inference.
"""
from copy import copy
from fractions import Fraction as Q
import importlib.util
import math
from pathlib import Path
import time


A13_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z')
_spec = importlib.util.spec_from_file_location('_a14_frozen_a13_kernel', A13_ROOT/'cert_kernel.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
if Path(base.__file__).resolve() != (A13_ROOT/'cert_kernel.py').resolve():
    raise RuntimeError('wrong frozen kernel identity')

SCHEMA = 'a14-four-camera-cover-v1'
TILE_SCHEMA = 'a14-camera-tile-box-farkas-v1'


def camera_tiles(cameras):
    """Four indexed closed boxes covering both side-camera intervals exactly."""
    if len(cameras) != 3 or cameras[1] != (Q(0), Q(0)):
        raise ValueError('three ordered cameras with fixed middle required')
    if any(lo > hi for lo, hi in cameras):
        raise ValueError('unordered camera interval')
    left, middle, right = cameras
    lm, rm = sum(left)/2, sum(right)/2
    return tuple((lp, middle, rp)
                 for lp in ((left[0], lm), (lm, left[1]))
                 for rp in ((right[0], rm), (rm, right[1])))


def encode_camera(cameras):
    return [[base.fraction_text(lo), base.fraction_text(hi)] for lo, hi in cameras]


class _TileContext(base.ExactContext):
    def __init__(self, context, cameras):
        # All photometric and appearance constraints remain identical. Only
        # the geometric domain is conditioned on this explicit camera tile.
        self.__dict__.update(context.__dict__)
        self.cameras = cameras
        self._cache = None

    def cell_constraints(self, branch, lo, hi, deadline=None):
        key = (branch, Q(lo), Q(hi))
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('deadline before tile geometry')
        if self._cache is None or self._cache[0] != key:
            value = super().cell_constraints(branch, lo, hi, deadline)
            self._cache = key, value
        return self._cache[1]


class JointContext:
    """Observation-only four-tile context, compatible with A13 check_cell.

    propose_cell is optional: it tests four relaxations and stops at the
    first unresolved/feasible one. Only check_cell can authorize exclusion.
    """
    def __init__(self, observation):
        started, cpu = time.perf_counter(), time.process_time()
        self.whole = base.ExactContext(observation)
        self.error = self.whole.error
        self.digest = getattr(self.whole, 'digest', None)
        self.tiles = ()
        self.contexts = ()
        self.float_camera_tiles = ()
        if self.error is None:
            self.tiles = camera_tiles(self.whole.cameras)
            self.contexts = tuple(_TileContext(self.whole, tile) for tile in self.tiles)
            self.float_camera_tiles = tuple(tuple((float(a), float(b)) for a, b in tile)
                                            for tile in self.tiles)
        self.setup_seconds = time.perf_counter()-started
        self.setup_cpu_seconds = time.process_time()-cpu

    def propose_cell(self, encoded, branch, lo, hi, time_limit, lp_solver):
        """Apply supplied frozen FREE._lp_for_cell to four camera domains.

        Argument order follows FREE._lp_for_cell plus the solver callback.
        time_limit bounds the combined relative proposal budget. Returned
        JSON status is 2 only if all four solver statuses are 2. Rounded tile
        endpoints are proposals only: check_cell redoes exact tile geometry.
        """
        started, cpu = time.perf_counter(), time.process_time()
        deadline = started+max(0., float(time_limit))
        result = {'status': 1, 'tile_statuses': [], 'reason': 'unknown'}
        try:
            if self.error is not None:
                raise ValueError(self.error)
            if not math.isfinite(deadline) or len(encoded) != 6:
                raise ValueError('invalid proposal budget or frozen encoding')
            for tile in self.float_camera_tiles:
                remaining = deadline-time.perf_counter()
                if remaining <= .001:
                    raise TimeoutError('numeric tile deadline')
                proposal = lp_solver((*encoded[:5], tile), branch, lo, hi, min(.20, remaining))
                result['tile_statuses'].append(int(proposal.status))
                if proposal.status != 2:
                    result['status'] = 0 if proposal.status == 0 else 1
                    result['reason'] = 'one_tile_retained'
                    break
            else:
                result.update(status=2, reason='all_four_float_infeasible')
            if time.perf_counter() >= deadline:
                result.update(status=1, reason='deadline')
        except Exception as exc:
            result.update(status=1, reason='error_or_deadline', error=f'{type(exc).__name__}: {exc}')
        result['wall_seconds'] = time.perf_counter()-started
        result['cpu_seconds'] = time.process_time()-cpu
        return result

    def check_cell(self, branch, lo, hi, deadline):
        started, cpu = time.perf_counter(), time.process_time()
        result = {'verified': False, 'certificate': None, 'reason': 'unknown',
                  'geometry_seconds': 0., 'certificate_search_seconds': 0.,
                  'exact_check_seconds': 0., 'tile_results': []}
        try:
            if self.error is not None:
                raise ValueError(self.error)
            if not math.isfinite(float(deadline)) or time.perf_counter() >= deadline:
                raise TimeoutError('deadline before tiles')
            qlo, qhi = Q.from_float(float(lo)), Q.from_float(float(hi))
            certificates = []
            for tile_id, (tile, context) in enumerate(zip(self.tiles, self.contexts)):
                check = context.check_cell(branch, lo, hi, deadline)
                for name in ('geometry_seconds', 'certificate_search_seconds', 'exact_check_seconds'):
                    result[name] += check[name]
                result['tile_results'].append({k: v for k, v in check.items() if k != 'certificate'})
                if not check['verified']:
                    result['reason'] = 'one_tile_unverified'
                    break
                cert = copy(check['certificate'])
                cert.update(schema=TILE_SCHEMA, tile_id=tile_id, camera_box=encode_camera(tile))
                certificates.append(cert)
            else:
                if len(certificates) != 4 or time.perf_counter() >= deadline:
                    raise TimeoutError('incomplete cover or deadline after tile checks')
                result.update(verified=True, reason='four_exact_tile_contradictions', certificate={
                    'schema': SCHEMA, 'observation_sha256': self.digest,
                    'branch': branch, 'lo': base.fraction_text(qlo), 'hi': base.fraction_text(qhi),
                    'camera_domain': encode_camera(self.whole.cameras),
                    'tiles': certificates})
        except Exception as exc:
            result.update(verified=False, certificate=None, reason='error_or_deadline',
                          error=f'{type(exc).__name__}: {exc}')
        result['wall_seconds'] = time.perf_counter()-started
        result['cpu_seconds'] = time.process_time()-cpu
        return result


# Explicit alias lets a search adapter select a kernel without changing calls.
ExactContext = JointContext
