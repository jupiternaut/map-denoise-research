"""Input-only direction replacement in isolated frozen V4 compatible instances."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
INSTRUMENT = PROJECT / 'exploration_v5' / 'slope_pooling.py'
VARIANTS = ('difference', 'pooled_pca')


def fingerprint(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def _load_private():
    spec = importlib.util.spec_from_file_location('_v6_private_slope_instrumentation', INSTRUMENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == INSTRUMENT.resolve()
    assert Path(module._V4.__file__).resolve() == PROJECT / 'exploration_v4' / 'surface_pooling.py'
    assert Path(module._V4._V3.__file__).resolve() == PROJECT / 'exploration_v3' / 'graph_surface.py'
    return module


def _positive_axis(vector):
    vector = np.asarray(vector, dtype=float).copy()
    if vector[np.argmax(abs(vector))] < 0: vector *= -1.
    return vector / np.linalg.norm(vector)


def _pca_basis(points, scan, sigma):
    """Only current centered XYZ; neither a reference nor an injection is read."""
    if len(points) < 3:
        return None, {'reason': 'fewer than three current points'}
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    values, vectors = np.linalg.eigh(covariance)
    diagnostics = {'normal_estimator': 'current observed pooled PCA; not a truth normal',
                   'pca_covariance_eigenvalues_mm2': values.tolist(),
                   'pca_normal_eigen_gap_mm2': float(values[1] - values[0]),
                   'pca_normal_sign_rule': 'largest absolute world component positive; first index wins ties',
                   'pca_tangent_rule': 'largest covariance eigenvector, same positive sign; second=n cross first',
                   'pca_degeneracy_note': 'repeated eigenvalues have no unique physical axis; no quality selector'}
    if values[1] <= 1e-20 * max(float(values[-1]), 1.):
        return None, dict(diagnostics, reason='current point covariance has numerical rank below two')
    normal = _positive_axis(vectors[:, 0])
    tangent = vectors[:, -1] - normal * np.dot(normal, vectors[:, -1])
    tangent = _positive_axis(tangent)
    second = np.cross(normal, tangent)
    second /= np.linalg.norm(second)
    basis = np.column_stack((tangent, second, normal))
    return basis, dict(diagnostics, normal_world=normal.tolist(), basis_world=basis.tolist())


def estimate(xyz_world_m, scan_id, sigma_mm, variant='difference'):
    """Return original-order world metres and JSON-friendly diagnostics."""
    started = time.perf_counter()
    if variant not in VARIANTS: raise ValueError(f'variant must be one of {VARIANTS}')
    original = np.asarray(xyz_world_m)
    original_scans = np.asarray(scan_id)
    if original.ndim != 2 or original.shape[1] != 3 or not len(original) or not np.isfinite(original).all():
        raise ValueError('finite nonempty N by 3 current world coordinates required')
    if original_scans.shape != (len(original),) or original_scans.dtype.kind not in 'iu':
        raise ValueError('one integer scan ID per point required')
    sigma = float(sigma_mm)
    if not np.isfinite(sigma) or sigma <= 0.: raise ValueError('positive finite sigma required')
    before = (fingerprint(original), fingerprint(original_scans))
    world = np.array(original, dtype=float, copy=True)
    frames = np.array(original_scans, dtype=np.int64, copy=True)
    world.flags.writeable = False
    frames.flags.writeable = False
    module = _load_private()
    original_basis = module._V4._V3._basis
    chosen_basis = original_basis if variant == 'difference' else _pca_basis
    trace = []

    def tracked_basis(points, scans, supplied_sigma):
        begin = time.perf_counter()
        basis, details = chosen_basis(points, scans, supplied_sigma)
        trace.append({'point_coordinates_sha256': fingerprint(points),
                      'scan_indices_sha256': fingerprint(scans), 'sigma_mm': float(supplied_sigma),
                      'basis_world': basis.tolist() if basis is not None else None,
                      'details': details, 'seconds': time.perf_counter() - begin})
        return basis, details

    # Only this newly created private instance is altered, never a shared import.
    module._V4._V3._basis = tracked_basis
    result, info = module.estimate(world, frames, sigma, variant='v4_compatible')
    if before != (fingerprint(original), fingerprint(original_scans)):
        raise RuntimeError('caller input was modified')
    if not trace: raise RuntimeError('direction helper was bypassed')
    normal = info.get('normal_world')
    if normal is not None:
        n = np.asarray(normal)
        for entry in trace:
            if entry['basis_world'] is not None:
                if not np.allclose(np.asarray(entry['basis_world'])[:, 2], n, atol=1e-12, rtol=0.):
                    raise RuntimeError('bias and pooling did not use one common direction')
    info.update(method='v6_direction_v4_compatible', direction_variant=variant,
                algorithm='unchanged V4 compatible; isolated upstream direction replacement',
                instrumentation='V5 exact-output v4_compatible mask/state recorder; new predictions unused',
                estimator_input_xyz_sha256=before[0], estimator_input_scan_sha256=before[1],
                basis_calls=trace, basis_call_count=len(trace),
                basis_hook_scope='new private V5/V4/V3 instance for this call; no shared module mutation',
                direction_recomputes=['graph scan bias', 'local coordinates', 'cells', 'associations',
                                      'weights', 'compatible grouping', 'actual output support'],
                output_xyz_sha256=fingerprint(result), input_arrays_unchanged=True,
                total_seconds=time.perf_counter() - started)
    info['seconds'] = info['total_seconds']
    return result, info
