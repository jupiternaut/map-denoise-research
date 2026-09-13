"""CPU surface-measure/source-graph prototype, with a matched OT ablation.

Public API: estimate(xyz_world_m, scan_id, sigma_mm, variant='unbalanced').
Only observed coordinates, source identifiers, and a supplied noise scale enter
the estimator. No reference geometry, true normals, labels or injected offsets.

Both variants use identical real-point surface elements, local PCA directions,
sampling-density proxy masses, initialization and iteration budgets. They differ
only in whether transport marginal masses are exact or KL-relaxed. The entropy
reference is the product measure a*b, not counting measure. Matching NEVER
projects points to an OT barycenter: a small source graph estimates one shared
normal translation per scan. All original points of a supported scan receive the
same translation, including poorly matched points; an unsupported scan is kept.

Local covariance is used only to fit a direction/planarity descriptor. It is NOT
a calibrated uncertainty. Density-compensation masses are NOT true surface area,
opacity, occupancy, or physical visibility. Pure-frame layer identity remains
ambiguous; this algorithm does not provide an identifiability or accuracy proof.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp


# Frozen before the root's reserved-seed evaluation. Units below are local mm.
PARAMETERS = {
    'surface_elements_per_scan': 128,
    'local_normal_neighbors': 12,
    'minimum_normal_points': 6,
    'scan_graph_neighbors': 4,
    'outer_iterations': 3,
    'sinkhorn_iterations': 24,
    'entropy_epsilon': 0.6,
    'marginal_kl_penalty': 2.0,
    'normal_distance_sigma_factor': 2.0,
    'tangent_neighbor_scale_factor': 2.0,
    'normal_misalignment_scale': 0.25,
    'maximum_usable_transport_cost': 25.0,
    'minimum_pair_usable_mass': 0.02,
    'minimum_normal_consensus_gap': 0.1,
    'minimum_element_planarity': 0.03,
    'huber_sigma_factor': 3.0,
    'normalized_graph_ridge': 0.03,
    'maximum_bias_sigma_factor': 8.0,
    'maximum_step_sigma_factor': 2.0,
    'update_damping': 0.8,
    'density_mass_clip_low': 0.25,
    'density_mass_clip_high': 4.0,
}
METHODS = ('balanced', 'unbalanced')


def _transport(cost, a, b, variant, epsilon=None, tau=None, iterations=None):
    """Entropic OT / unbalanced OT, solved by fixed-budget log scaling.

Objective = <C,P> + eps KL(P | a*b), plus, in the unbalanced case,
tau [KL(P1 | a) + KL(P.T1 | b)]. Balanced uses exact marginal constraints.
Finite iterations need not meet exact balanced marginals; residuals are exposed.
"""
    if variant not in METHODS:
        raise ValueError(f'variant must be one of {METHODS}')
    epsilon = PARAMETERS['entropy_epsilon'] if epsilon is None else float(epsilon)
    tau = PARAMETERS['marginal_kl_penalty'] if tau is None else float(tau)
    iterations = PARAMETERS['sinkhorn_iterations'] if iterations is None else int(iterations)
    log_a, log_b = np.log(a), np.log(b)
    log_kernel = log_a[:, None] + log_b[None, :] - cost / epsilon
    alpha = 1.0 if variant == 'balanced' else tau / (tau + epsilon)
    log_u = np.zeros_like(a)
    log_v = np.zeros_like(b)
    for _ in range(iterations):
        log_u = alpha * (log_a - logsumexp(log_kernel + log_v[None, :], axis=1))
        log_v = alpha * (log_b - logsumexp(log_kernel + log_u[:, None], axis=0))
    plan = np.exp(log_kernel + log_u[:, None] + log_v[None, :])
    return plan, {
        'transport_mass': float(plan.sum()),
        'row_marginal_l1': float(np.abs(plan.sum(axis=1)-a).sum()),
        'column_marginal_l1': float(np.abs(plan.sum(axis=0)-b).sum()),
        'marginal_relaxation_exponent': float(alpha),
    }


def _sample(indices, local):
    """Deterministic, point-order-independent spatial ordering and capping."""
    xyz = local[indices]
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    limit = PARAMETERS['surface_elements_per_scan']
    if len(order) > limit:
        order = order[np.linspace(0, len(order)-1, limit, dtype=int)]
    return indices[order]


def _make_elements(local, frame, n_frames, sigma):
    elements = []
    normal_vote = np.zeros((3, 3))
    usable_normal_count = 0
    for f in range(n_frames):
        indices = np.flatnonzero(frame == f)
        chosen = _sample(indices, local)
        centers = local[chosen].copy()
        normals = np.zeros_like(centers)
        planarity = np.zeros(len(chosen))
        if len(indices) >= PARAMETERS['minimum_normal_points']:
            k = min(PARAMETERS['local_normal_neighbors'], len(indices))
            _, nearest = cKDTree(local[indices]).query(centers, k=k)
            neighborhoods = local[indices][nearest]
            centered = neighborhoods-neighborhoods.mean(axis=1, keepdims=True)
            covariance = np.einsum('nki,nkj->nij', centered, centered)/k
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            normals = eigenvectors[:, :, 0]
            planarity = np.maximum(eigenvalues[:, 1]-eigenvalues[:, 0], 0.)/np.maximum(eigenvalues[:, 2], 1e-20)
            good = planarity >= PARAMETERS['minimum_element_planarity']
            usable_normal_count += int(good.sum())
            weights = np.where(good, planarity, 0.)
            # Equal total vote per usable scan, so dense scans do not dominate.
            if weights.sum() > 0:
                weights /= weights.sum()
                normal_vote += (normals*weights[:, None]).T @ normals
        elements.append({'indices': chosen, 'points': centers, 'normals': normals,
                         'planarity': planarity, 'n_source_points': len(indices)})
    if usable_normal_count == 0:
        return elements, None, 0., usable_normal_count
    values, vectors = np.linalg.eigh(normal_vote)
    consensus_gap = float((values[-1]-values[-2])/max(values[-1], 1e-20))
    normal = vectors[:, -1]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    for element in elements:
        normals = element['normals']
        bad = element['planarity'] < PARAMETERS['minimum_element_planarity']
        normals[bad] = normal
        normals[(normals @ normal) < 0] *= -1
        centers = element['points']
        tangent = centers-(centers @ normal)[:, None]*normal
        if len(centers) >= 2:
            k = min(5, len(centers))
            distances, _ = cKDTree(tangent).query(tangent, k=k)
            radius = distances[:, -1]
        else:
            radius = np.array([4.*sigma])
        area_proxy = np.maximum(radius**2, sigma**2)
        scale = max(float(np.median(area_proxy)), sigma**2)
        mass = np.clip(area_proxy/scale, PARAMETERS['density_mass_clip_low'], PARAMETERS['density_mass_clip_high'])
        element['mass'] = mass/mass.sum()
        element['tangent_scale_mm'] = max(4.*sigma, PARAMETERS['tangent_neighbor_scale_factor']*float(np.median(radius)))
    return elements, normal, consensus_gap, usable_normal_count


def _edges(elements):
    centers = np.array([e['points'].mean(axis=0) for e in elements])
    count = len(elements)
    k = min(PARAMETERS['scan_graph_neighbors']+1, count)
    _, nearest = cKDTree(centers).query(centers, k=k)
    return sorted({tuple(sorted((f, int(g)))) for f, row in enumerate(np.atleast_2d(nearest)) for g in row if f != g})


def _pair_measure(left, right, normal, bias_difference, sigma, variant):
    original_delta = left['points'][:, None, :]-right['points'][None, :, :]
    delta = original_delta-bias_difference*normal
    normal_delta = delta @ normal
    tangent_squared = np.maximum(np.sum(delta**2, axis=2)-normal_delta**2, 0.)
    tangent_scale = np.sqrt(left['tangent_scale_mm']*right['tangent_scale_mm'])
    normal_scale = PARAMETERS['normal_distance_sigma_factor']*sigma
    dot = np.clip(left['normals'] @ right['normals'].T, -1., 1.)
    cost = normal_delta**2/(normal_scale**2) + tangent_squared/(tangent_scale**2)
    cost += (1.-dot**2)/PARAMETERS['normal_misalignment_scale']
    plan, details = _transport(cost, left['mass'], right['mass'], variant)
    usable_plan = plan*(cost <= PARAMETERS['maximum_usable_transport_cost'])
    usable_mass = float(usable_plan.sum())
    average_normal = left['normals'][:, None, :]+right['normals'][None, :, :]
    average_normal /= np.maximum(np.linalg.norm(average_normal, axis=2, keepdims=True), 1e-15)
    direction_coefficient = average_normal @ normal
    raw_residual = np.sum(original_delta*average_normal, axis=2)
    residual = raw_residual-direction_coefficient*bias_difference
    huber_scale = PARAMETERS['huber_sigma_factor']*sigma
    robust = np.minimum(1., huber_scale/np.maximum(np.abs(residual), 1e-15))
    weight = usable_plan*robust
    information = float(np.sum(weight*direction_coefficient**2))
    rhs = float(np.sum(weight*direction_coefficient*raw_residual))
    details.update({'usable_mass': usable_mass, 'normal_information': information,
                    'transport_weighted_cost': float(np.sum(plan*cost)/max(plan.sum(), 1e-30)),
                    'normal_delta_rms_mm': float(np.sqrt(np.sum(usable_plan*residual**2)/max(usable_mass, 1e-30))),
                    'left_usable_mass_ratio': (usable_plan.sum(axis=1)/left['mass']).tolist(),
                    'right_usable_mass_ratio': (usable_plan.sum(axis=0)/right['mass']).tolist()})
    return information, rhs, details


def _components(n_frames, active_edges):
    neighbors = [set() for _ in range(n_frames)]
    for f, g in active_edges:
        neighbors[f].add(g)
        neighbors[g].add(f)
    unseen = set(range(n_frames))
    result = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        group, frontier = [seed], [seed]
        while frontier:
            f = frontier.pop()
            for g in sorted(neighbors[f] & unseen):
                unseen.remove(g)
                group.append(g)
                frontier.append(g)
        result.append(sorted(group))
    return result


def _solve_graph(n_frames, edge_results, counts, old_bias, sigma):
    hessian = np.zeros((n_frames, n_frames))
    rhs = np.zeros(n_frames)
    active = []
    for f, g, information, target, details in edge_results:
        if details['usable_mass'] < PARAMETERS['minimum_pair_usable_mass'] or information <= 1e-12:
            continue
        hessian[f, f] += information
        hessian[g, g] += information
        hessian[f, g] -= information
        hessian[g, f] -= information
        rhs[f] += target
        rhs[g] -= target
        active.append((f, g))
    components = _components(n_frames, active)
    bias = np.zeros(n_frames)
    supported = np.zeros(n_frames, dtype=bool)
    for group in components:
        if len(group) < 2:
            continue
        idx = np.asarray(group)
        block = hessian[np.ix_(idx, idx)]
        # Normalize total graph information before the fixed ridge. Uniform
        # loss of transport mass alone therefore cannot strengthen the prior.
        information_scale = float(np.trace(block)/len(group))
        solution = np.linalg.solve(block/information_scale + PARAMETERS['normalized_graph_ridge']*np.eye(len(group)),
                                   rhs[idx]/information_scale)
        solution -= np.average(solution, weights=counts[idx])
        step = solution-old_bias[idx]
        step_limit = PARAMETERS['maximum_step_sigma_factor']*sigma
        step *= min(1., step_limit/max(float(np.max(np.abs(step))), 1e-30))
        updated = old_bias[idx]+PARAMETERS['update_damping']*step
        updated -= np.average(updated, weights=counts[idx])
        bias_limit = PARAMETERS['maximum_bias_sigma_factor']*sigma
        updated *= min(1., bias_limit/max(float(np.max(np.abs(updated))), 1e-30))
        bias[idx] = updated
        supported[idx] = True
    return bias, supported, {
        'active_scan_pairs': len(active), 'components': components,
        'normal_information_trace': float(np.trace(hessian)),
        'normal_information_per_scan': np.diag(hessian).tolist(),
        'ridge_scope': 'fixed after mean-diagonal information normalization within each supported component',
    }


def estimate(xyz_world_m, scan_id, sigma_mm, variant='unbalanced'):
    """Return coordinates in the same world frame, point count and input order."""
    started = time.perf_counter()
    xyz = np.asarray(xyz_world_m, dtype=float)
    source = np.asarray(scan_id)
    sigma = float(sigma_mm)
    if variant not in METHODS:
        raise ValueError(f'variant must be one of {METHODS}')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or source.shape != (len(xyz),):
        raise ValueError('Expected finite N x 3 coordinates and N source IDs')
    if not np.isfinite(xyz).all() or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Coordinates must be finite and sigma must be positive')
    if source.dtype.kind in 'fc' and not np.isfinite(source).all():
        raise ValueError('Source IDs must be finite')
    ids, frame = np.unique(source, return_inverse=True)
    counts = np.bincount(frame)
    info = {
        'method': 'surface_measure_source_graph', 'variant': variant, 'parameters': dict(PARAMETERS),
        'status': 'UNCHANGED', 'n_input': len(xyz), 'n_output': len(xyz),
        'point_order_preserved': True, 'n_scans': len(ids), 'scan_ids': ids.tolist(),
        'points_per_scan': counts.tolist(), 'sigma_mm': sigma,
        'normal_source': 'same-scan local PCA with per-scan-normalized axial consensus; no true normals',
        'mass_scope': 'capped same-scan tangent kNN inverse-density proxy, not physical surface area',
        'output_scope': 'one common normal translation for every point of a supported scan; unsupported scan unchanged; no barycentric projection or point deletion',
        'uncertainty_scope': 'local covariance is a geometric normal descriptor, not calibrated measurement uncertainty',
        'information_limit': 'unknown pure-scan layer assignments and weak overlap may remain ambiguous',
        'bias_mm': [0.]*len(ids), 'bias_rms_mm': 0., 'moved_point_fraction': 0.,
        'effective_transport_mass': 0., 'usable_transport_mass': 0., 'supported_scan_fraction': 0.,
        'outer_history': [],
    }
    if len(xyz) == 0 or len(ids) < 2:
        info.update({'reason': 'at least two observed scans are required', 'elapsed_s': time.perf_counter()-started})
        return xyz.copy(), info
    origin = xyz.mean(axis=0)
    local = (xyz-origin)*1000.
    elements, normal, consensus_gap, usable_normals = _make_elements(local, frame, len(ids), sigma)
    info.update({'surface_element_counts': [len(e['points']) for e in elements],
                 'normal_consensus_gap': consensus_gap, 'usable_local_normal_elements': usable_normals})
    if normal is None or consensus_gap < PARAMETERS['minimum_normal_consensus_gap']:
        info.update({'reason': 'insufficient coherent observed local surface directions', 'elapsed_s': time.perf_counter()-started})
        return xyz.copy(), info
    edges = _edges(elements)
    bias = np.zeros(len(ids))
    supported = np.zeros(len(ids), dtype=bool)
    final_pair_details = []
    row_mass_ratios = [np.zeros(len(e['points'])) for e in elements]
    for iteration in range(PARAMETERS['outer_iterations']):
        results = []
        final_pair_details = []
        row_mass_ratios = [np.zeros(len(e['points'])) for e in elements]
        for f, g in edges:
            information, target, detail = _pair_measure(elements[f], elements[g], normal, bias[f]-bias[g], sigma, variant)
            results.append((f, g, information, target, detail))
            row_mass_ratios[f] = np.maximum(row_mass_ratios[f], detail['left_usable_mass_ratio'])
            row_mass_ratios[g] = np.maximum(row_mass_ratios[g], detail['right_usable_mass_ratio'])
            final_pair_details.append({'scan_index_left': f, 'scan_index_right': g,
                                       **{k: v for k, v in detail.items() if k not in ('left_usable_mass_ratio', 'right_usable_mass_ratio')}})
        bias, supported, graph_info = _solve_graph(len(ids), results, counts, bias, sigma)
        info['outer_history'].append({'iteration': iteration+1, 'bias_mm': bias.tolist(),
                                      'transport_mass_sum': float(sum(d['transport_mass'] for d in final_pair_details)),
                                      'usable_transport_mass_sum': float(sum(d['usable_mass'] for d in final_pair_details)),
                                      **graph_info})
    # Physical source constraint: even weakly matched points move with the scan.
    # No per-point mask is applied, so this step cannot split a scan's surfaces.
    output = xyz-bias[frame, None]*normal[None, :]/1000.
    displacement = np.abs(bias[frame])
    weak_matched = [float(np.mean(r < PARAMETERS['minimum_pair_usable_mass'])) for r in row_mass_ratios]
    info.update({'status': 'APPLY' if np.any(displacement > 1e-10) else 'UNCHANGED',
                 'normal_world': normal.tolist(), 'bias_mm': bias.tolist(),
                 'bias_rms_mm': float(np.sqrt(np.average(bias**2, weights=counts))),
                 'supported_scan_fraction': float(supported.mean()), 'supported_scans': supported.tolist(),
                 'moved_point_fraction': float(np.mean(displacement > 1e-10)),
                 'mean_point_displacement_mm': float(displacement.mean()),
                 'max_point_displacement_mm': float(displacement.max()),
                 'effective_transport_mass': float(np.mean([d['transport_mass'] for d in final_pair_details])),
                 'usable_transport_mass': float(np.mean([d['usable_mass'] for d in final_pair_details])),
                 'weakly_matched_element_fraction_per_scan': weak_matched,
                 'weakly_matched_points_of_supported_scan_policy': 'inherit the common scan correction; not individually held fixed',
                 'scan_pair_details': final_pair_details,
                 'matching_diagnostics_scope': 'last fixed-budget matching pass before the last translation update',
                 'elapsed_s': time.perf_counter()-started})
    if not np.isfinite(output).all():
        raise FloatingPointError('Nonfinite output')
    return output, info
