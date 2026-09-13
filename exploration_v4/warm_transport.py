"""Frozen-target V4 transport initialization experiment, CPU reference.

Only current world XYZ (m), integer scan IDs and supplied sigma (mm) enter.
Graph-projected points are discarded. Zero and warm transport build their same
surface elements, normal, masses and pair graph from ORIGINAL observed points.
Warm initialization changes only the source-offset state, not those objects.

graph_bias_only applies the original graph's source translation, not its surface
projection. graph_projected_bias_only applies exactly the projected/gauged/
bounded initial state of warm transport, with zero transport iterations, making
it the strict no-refinement control. Every public pipeline then invokes the same
V3 graph local_only exactly once. Costs include discarded graph projection work.
This is an initialization ablation, not a new OT theory or an accuracy guarantee.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np


V3 = Path(__file__).resolve().parents[1] / 'exploration_v3'


def _load(name, filename):
    path = V3 / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError('wrong frozen estimator source')
    return module


_measure = _load('_warm_v4_frozen_measure', 'measure_surface.py')
_graph = _load('_warm_v4_frozen_graph', 'graph_surface.py')
FROZEN_SOURCE_HASHES = {
    str(V3/name): hashlib.sha256((V3/name).read_bytes()).hexdigest()
    for name in ('measure_surface.py', 'graph_surface.py')
}
VARIANTS = ('graph_bias_only', 'graph_projected_bias_only',
            'zero_balanced_6', 'zero_unbalanced_6',
            'warm_balanced_2', 'warm_unbalanced_2',
            'warm_balanced_3', 'warm_unbalanced_3')


def _fingerprint(elements, normal, edges):
    """Audit the target representation, independent of initialization state."""
    digest = hashlib.sha256()
    for array in (normal, np.asarray(edges, dtype=np.int64)):
        a = np.ascontiguousarray(array)
        digest.update(str((a.shape, a.dtype.str)).encode()); digest.update(a.tobytes())
    for element in elements:
        for key in ('indices', 'points', 'normals', 'planarity', 'mass'):
            a = np.ascontiguousarray(element[key])
            digest.update(key.encode()); digest.update(str((a.shape, a.dtype.str)).encode())
            digest.update(a.tobytes())
        digest.update(np.float64(element['tangent_scale_mm']).tobytes())
    return digest.hexdigest()


def _project_graph_bias(graph_info, ids, counts, normal, edges, sigma):
    """Project physical source vectors, then use the fixed target's gauge."""
    graph_normal = np.asarray(graph_info.get('normal_world', [0., 0., 0.]), dtype=float)
    by_id = {int(sid): float(b) for sid, b in zip(graph_info['scan_ids'], graph_info['bias_mm'])}
    bias = np.asarray([by_id.get(int(sid), 0.) for sid in ids])
    coefficient = float(graph_normal @ normal)
    bias *= coefficient
    projected_before_gauge = bias.copy()
    components = _measure._components(len(ids), edges)
    scale_factors = []
    for group in components:
        index = np.asarray(group)
        if len(group) < 2:
            bias[index] = 0.
            scale_factors.append(0.)
            continue
        bias[index] -= np.average(bias[index], weights=counts[index])
        limit = _measure.PARAMETERS['maximum_bias_sigma_factor']*sigma
        factor = min(1., limit/max(float(np.max(np.abs(bias[index]))), 1e-30))
        bias[index] *= factor
        scale_factors.append(factor)
    return bias, {
        'graph_to_measure_normal_dot': coefficient,
        'projected_before_gauge_bias_mm': projected_before_gauge.tolist(),
        'initial_gauge_components_scan_ids': [[int(ids[s]) for s in group] for group in components],
        'initial_component_limit_scale': scale_factors,
        'initialization_gauge': 'point-count-weighted zero mean per fixed candidate-pair component',
        'initialization_limit': 'same V3 max-bias bound, uniform scale within each component',
    }


def _refine(elements, normal, edges, counts, sigma, initial, variant, iterations):
    """Exact V3 helper order, with explicit initial state and outer budget."""
    bias = initial.copy()
    history, final_details = [], []
    supported = np.zeros(len(counts), dtype=bool)
    for iteration in range(iterations):
        results, final_details = [], []
        for f, g in edges:
            information, target, detail = _measure._pair_measure(
                elements[f], elements[g], normal, bias[f]-bias[g], sigma, variant)
            results.append((f, g, information, target, detail))
            final_details.append({'scan_index_left': f, 'scan_index_right': g,
                                  **{k: v for k, v in detail.items()
                                     if k not in ('left_usable_mass_ratio', 'right_usable_mass_ratio')}})
        bias, supported, graph_info = _measure._solve_graph(
            len(counts), results, counts, bias, sigma)
        history.append({'iteration': iteration+1, 'bias_mm': bias.tolist(),
                        'transport_mass_sum': float(sum(d['transport_mass'] for d in final_details)),
                        'usable_transport_mass_sum': float(sum(d['usable_mass'] for d in final_details)),
                        **graph_info})
    return bias, supported, history, final_details


def _correct(xyz_world_m, scan_id, sigma_mm, variant):
    """Pre-postfilter stage exposed for audit/tests; same-order rigid source moves."""
    started = time.perf_counter()
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}')
    world = np.asarray(xyz_world_m, dtype=float)
    source = np.asarray(scan_id)
    sigma = float(sigma_mm)
    if world.ndim != 2 or world.shape[1] != 3 or source.shape != (len(world),):
        raise ValueError('expected N x 3 coordinates and N scan IDs')
    if source.dtype.kind not in 'iu' or not np.isfinite(world).all():
        raise ValueError('finite coordinates and integer scan IDs are required')
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma_mm must be finite and positive')
    ids, frame = np.unique(source, return_inverse=True)
    counts = np.bincount(frame, minlength=len(ids))
    wants_graph = variant.startswith('warm_') or variant.startswith('graph_')
    wants_measure = variant != 'graph_bias_only'
    iterations = int(variant.rsplit('_', 1)[1]) if variant.startswith(('warm_', 'zero_')) else 0
    transport_variant = 'unbalanced' if 'unbalanced' in variant else 'balanced'
    bias, initial = np.zeros(len(ids)), np.zeros(len(ids))
    info = dict(method='warm_transport_v4', variant=variant, scan_ids=ids.tolist(),
                points_per_scan=counts.tolist(), n_input=len(world), n_output=len(world),
                sigma_mm=sigma, input_units='world metres', point_order_preserved=True,
                outer_iterations=iterations, sinkhorn_iterations=_measure.PARAMETERS['sinkhorn_iterations'],
                transport_parameters={**_measure.PARAMETERS, 'outer_iterations': iterations},
                graph_initialization_seconds=0., normal_preparation_seconds=0., refinement_seconds=0.,
                target_representation_sha256=None, graph_projected_cloud_used=False,
                frozen_source_sha256=dict(FROZEN_SOURCE_HASHES),
                scope='same frozen observed target; only initialization and stated outer budget differ',
                raw_output_scope='one shared normal translation per scan; no per-point projection',
                raw_status='UNCHANGED', graph_initializer=None, outer_history=[], scan_pair_details=[])
    graph_info = None
    if wants_graph and len(world):
        t = time.perf_counter()
        _discarded_projection, graph_info = _graph.estimate(world.copy(), source.copy(), sigma, variant='graph')
        del _discarded_projection
        info['graph_initialization_seconds'] = time.perf_counter()-t
        info['graph_initializer'] = graph_info
    normal = np.zeros(3)
    supported = np.zeros(len(ids), dtype=bool)
    if not wants_measure and graph_info is not None:
        normal = np.asarray(graph_info.get('normal_world', [0., 0., 0.]), dtype=float)
        by_id = {int(s): float(b) for s, b in zip(graph_info['scan_ids'], graph_info['bias_mm'])}
        bias = np.asarray([by_id[int(s)] for s in ids])
        initial = bias.copy()
        for group in graph_info.get('graph_components_scan_ids', []):
            if len(group) >= 2:
                supported[np.isin(ids, group)] = True
        info['normal_source'] = 'original V3 graph normal, without measure-direction projection'
    elif wants_measure and len(ids) >= 2:
        t = time.perf_counter()
        local = (world-world.mean(axis=0))*1000.
        elements, observed_normal, gap, normal_count = _measure._make_elements(local, frame, len(ids), sigma)
        info.update(normal_consensus_gap=gap, usable_local_normal_elements=normal_count,
                    surface_element_counts=[len(e['points']) for e in elements])
        ready = observed_normal is not None and gap >= _measure.PARAMETERS['minimum_normal_consensus_gap']
        if ready:
            normal = observed_normal
            edges = _measure._edges(elements)
            target_hash = _fingerprint(elements, normal, edges)
            info.update(target_representation_sha256=target_hash,
                        normal_source='V3 measure same-scan local PCA from original observations',
                        target_pair_graph=[list(e) for e in edges])
            if graph_info is not None:
                initial, initialization = _project_graph_bias(graph_info, ids, counts, normal, edges, sigma)
                info.update(initialization)
            bias = initial.copy()
            info['normal_preparation_seconds'] = time.perf_counter()-t
            if iterations:
                t = time.perf_counter()
                bias, supported, history, details = _refine(
                    elements, normal, edges, counts, sigma, initial, transport_variant, iterations)
                info['refinement_seconds'] = time.perf_counter()-t
                info.update(outer_history=history, scan_pair_details=details)
            else:
                for group in _measure._components(len(ids), edges):
                    if len(group) >= 2:
                        supported[np.asarray(group)] = True
            if _fingerprint(elements, normal, edges) != target_hash:
                raise RuntimeError('transport modified the fixed target representation')
        else:
            info['normal_preparation_seconds'] = time.perf_counter()-t
            info['raw_reason'] = 'measure direction unsupported; graph projection is not substituted as target'
    else:
        info['raw_reason'] = 'at least two scans required by measure target'
    raw = world-bias[frame, None]*normal[None, :]/1000.
    details = info['scan_pair_details']
    info.update(normal_world=normal.tolist(), initial_bias_mm=initial.tolist(),
                refined_bias_mm=bias.tolist(), bias_mm=bias.tolist(), supported_scans=supported.tolist(),
                bias_rms_mm=float(np.sqrt(np.average(bias**2, weights=counts))) if len(world) else 0.,
                initial_bias_rms_mm=float(np.sqrt(np.average(initial**2, weights=counts))) if len(world) else 0.,
                refinement_bias_change_rms_mm=float(np.sqrt(np.average((bias-initial)**2, weights=counts))) if len(world) else 0.,
                effective_transport_mass=float(np.mean([d['transport_mass'] for d in details])) if details else None,
                usable_transport_mass=float(np.mean([d['usable_mass'] for d in details])) if details else None,
                row_marginal_l1_max=float(max(d['row_marginal_l1'] for d in details)) if details else None,
                column_marginal_l1_max=float(max(d['column_marginal_l1'] for d in details)) if details else None,
                matching_diagnostics_scope='last fixed-budget match before final source-offset update',
                marginal_scope='balanced violations are solver residuals; UOT deviations are allowed, not convergence errors',
                raw_status='APPLY' if np.any(bias != 0.) else 'UNCHANGED',
                raw_correction_seconds=time.perf_counter()-started)
    if not np.isfinite(raw).all():
        raise FloatingPointError('nonfinite raw correction')
    return raw, info


def estimate(xyz_world_m, scan_id, sigma_mm, variant='warm_balanced_2'):
    """Same-order world output and JSON-friendly metadata; one final local filter."""
    started = time.perf_counter()
    raw, info = _correct(xyz_world_m, scan_id, sigma_mm, variant)
    t = time.perf_counter()
    if len(raw):
        output, local_info = _graph.estimate(raw.copy(), np.asarray(scan_id).copy(), sigma_mm, variant='local_only')
    else:
        output, local_info = raw.copy(), {'status': 'UNSUPPORTED', 'reason': 'empty input'}
    info.update(postfilter_seconds=time.perf_counter()-t, local_filter=local_info,
                pipeline='source-bias correction then exactly one unchanged V3 local_only filter',
                postfilter_calls=1 if len(raw) else 0,
                raw_correction_reconstruction='original_world - refined_bias_mm[scan_index] * normal_world / 1000',
                total_seconds=time.perf_counter()-started)
    if output.shape != np.asarray(xyz_world_m).shape or not np.isfinite(output).all():
        raise FloatingPointError('invalid final output')
    return output, info
