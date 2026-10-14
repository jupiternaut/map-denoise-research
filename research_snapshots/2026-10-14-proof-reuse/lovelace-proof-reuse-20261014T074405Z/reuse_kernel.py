"""A15 verified multiplier inheritance, confined to the frozen four tile contexts.

No extra cells are certified. A stored witness is merely a proposal: each child
certificate is rebuilt and checked against the child's exact rational rows.
"""
from copy import deepcopy
from fractions import Fraction as Q
import hashlib
import importlib.util
import math
from pathlib import Path
import time

A14_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z')
_spec = importlib.util.spec_from_file_location('_a15_frozen_a14_joint', A14_ROOT/'joint_kernel.py')
frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen)
if Path(frozen.__file__).resolve() != (A14_ROOT/'joint_kernel.py').resolve():
    raise RuntimeError('incorrect frozen A14 identity')
base = frozen.base
MODEL_SHA256 = hashlib.sha256((A14_ROOT/'joint_kernel.py').read_bytes()
                             + Path(base.__file__).read_bytes()).hexdigest()


class _TileContext(frozen._TileContext):
    """Same exact LP, optional verified ancestor multipliers, no float cache."""
    def __init__(self, context, cameras, inherit=False):
        super().__init__(context, cameras)
        self.inherit = bool(inherit)
        self._verified_parents = []
        self._scope = (self.digest, tuple(cameras), MODEL_SHA256)
        self.reuse_counts = dict(check_calls=0, stored=0, eligible_parents=0,
                                rechecks=0, hits=0, misses=0, invalid=0,
                                deadline=0, fallback_calls=0,
                                certificate_lp_calls=0, strict_descendant_hits=0,
                                repeated_cell_hits=0)
        self.reuse_seconds = dict(parent_lookup=0., child_geometry=0.,
                                  child_exact_check=0., fallback=0.,
                                  certificate_search=0.)

    def _remember(self, result):
        if not self.inherit or not result.get('verified'):
            return
        cert = result['certificate']
        # Only a certificate returned by the rational checker reaches this list.
        self._verified_parents.append((self._scope, deepcopy(cert)))
        self.reuse_counts['stored'] += 1

    def check_cell(self, branch, lo, hi, deadline):
        started, cpu = time.perf_counter(), time.process_time()
        self.reuse_counts['check_calls'] += 1
        inherited_geometry, inherited_exact = 0., 0.
        if self.inherit:
            t = time.perf_counter()
            candidates = []
            try:
                qlo, qhi = Q.from_float(float(lo)), Q.from_float(float(hi))
                if not math.isfinite(float(deadline)) or time.perf_counter() >= deadline:
                    raise TimeoutError('deadline before inherited proof lookup')
                for scope, parent in reversed(self._verified_parents):
                    if (scope == self._scope and parent['observation_sha256'] == self.digest
                            and parent['branch'] == branch
                            and Q(parent['lo']) <= qlo <= qhi <= Q(parent['hi'])):
                        candidates.append(parent)
                self.reuse_counts['eligible_parents'] += len(candidates)
            except TimeoutError:
                self.reuse_counts['deadline'] += 1
            except (ValueError, TypeError, KeyError, ArithmeticError, OverflowError):
                self.reuse_counts['invalid'] += 1
            self.reuse_seconds['parent_lookup'] += time.perf_counter()-t
            for parent in candidates:
                try:
                    t = time.perf_counter()
                    try:
                        rows, rhs, bounds = self.cell_constraints(branch, qlo, qhi, deadline)
                    finally:
                        elapsed = time.perf_counter()-t
                        inherited_geometry += elapsed
                        self.reuse_seconds['child_geometry'] += elapsed
                    if (parent['schema'] != base.SCHEMA
                            or parent['variable_order'] != list(base.VARIABLE_ORDER)
                            or parent['row_count'] != len(rows)):
                        raise ValueError('cached certificate row layout mismatch')
                    weights = tuple((i, Q(w)) for i, w in parent['multipliers'])
                    self.reuse_counts['rechecks'] += 1
                    t = time.perf_counter()
                    try:
                        margin = base.box_margin(rows, rhs, bounds, weights, deadline)
                    finally:
                        elapsed = time.perf_counter()-t
                        inherited_exact += elapsed
                        self.reuse_seconds['child_exact_check'] += elapsed
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('deadline after inherited exact check')
                    if margin <= 0:
                        self.reuse_counts['invalid'] += 1
                        continue
                    certificate = dict(schema=base.SCHEMA, observation_sha256=self.digest,
                        branch=branch, lo=base.fraction_text(qlo), hi=base.fraction_text(qhi),
                        variable_order=list(base.VARIABLE_ORDER), row_count=len(rows),
                        multipliers=[[i, base.fraction_text(w)] for i, w in weights],
                        margin=base.fraction_text(margin))
                    self.reuse_counts['hits'] += 1
                    repeated = (Q(parent['lo']) == qlo and Q(parent['hi']) == qhi)
                    self.reuse_counts['repeated_cell_hits' if repeated else 'strict_descendant_hits'] += 1
                    out = dict(verified=True, certificate=certificate,
                        reason='inherited_weights_positive_child_margin',
                        geometry_seconds=inherited_geometry,
                        certificate_search_seconds=0., exact_check_seconds=inherited_exact,
                        lp_status=None, wall_seconds=time.perf_counter()-started,
                        cpu_seconds=time.process_time()-cpu)
                    self._remember(out)
                    return out
                except TimeoutError:
                    self.reuse_counts['deadline'] += 1
                    break
                except (ValueError, TypeError, KeyError, ArithmeticError, OverflowError):
                    self.reuse_counts['invalid'] += 1
            self.reuse_counts['misses'] += 1
        self.reuse_counts['fallback_calls'] += 1
        t = time.perf_counter()
        out = super().check_cell(branch, lo, hi, deadline)
        self.reuse_seconds['fallback'] += time.perf_counter()-t
        self.reuse_seconds['certificate_search'] += out['certificate_search_seconds']
        self.reuse_counts['certificate_lp_calls'] += int(out.get('lp_status') is not None)
        out['geometry_seconds'] += inherited_geometry
        out['exact_check_seconds'] += inherited_exact
        out['wall_seconds'] = time.perf_counter()-started
        out['cpu_seconds'] = time.process_time()-cpu
        self._remember(out)
        return out


class JointContext(frozen.JointContext):
    def __init__(self, observation, inherit=False):
        # Do not construct discarded contexts; retain A14 initialization order.
        started, cpu = time.perf_counter(), time.process_time()
        self.whole = base.ExactContext(observation)
        self.error = self.whole.error
        self.digest = getattr(self.whole, 'digest', None)
        self.tiles, self.contexts, self.float_camera_tiles = (), (), ()
        if self.error is None:
            self.tiles = frozen.camera_tiles(self.whole.cameras)
            self.contexts = tuple(_TileContext(self.whole, tile, inherit) for tile in self.tiles)
            self.float_camera_tiles = tuple(tuple((float(a), float(b)) for a, b in tile)
                                            for tile in self.tiles)
        self.setup_seconds = time.perf_counter()-started
        self.setup_cpu_seconds = time.process_time()-cpu

    def counters(self):
        counts, seconds = {}, {}
        for context in self.contexts:
            for key, value in context.reuse_counts.items():
                counts[key] = counts.get(key, 0)+value
            for key, value in context.reuse_seconds.items():
                seconds[key] = seconds.get(key, 0.)+value
        return {'counts': counts, 'seconds': seconds,
                'per_tile': [{'counts': dict(x.reuse_counts), 'seconds': dict(x.reuse_seconds)}
                             for x in self.contexts]}


ExactContext = JointContext
