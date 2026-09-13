"""Exploratory local surface/scan incidence model; input coordinates are metres.

Only XYZ, acquisition scan IDs, and the supplied random-noise sigma are used.
The graph variant shares a normal displacement per scan across local surfaces.
The matched local_only ablation fixes those displacements to zero. Neither
variant claims arbitrary-surface reconstruction or independently measured truth.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp


def _basis(points, scan, sigma):
    """Estimate a common normal from within-scan differences, not pooled height."""
    candidates, differences = [], []
    for sid in np.unique(scan):
        cloud = points[scan == sid]
        if len(cloud) < 8:
            continue
        k = min(12, len(cloud))
        indices = cKDTree(cloud).query(cloud, k=k)[1]
        differences.append((cloud[indices[:, 1:min(7, k)]] - cloud[:, None]).reshape(-1, 3))
        chosen = np.linspace(0, len(cloud) - 1, min(16, len(cloud)), dtype=int)
        neighbourhoods = cloud[indices[chosen]]
        centered = neighbourhoods - neighbourhoods.mean(axis=1, keepdims=True)
        values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', centered, centered))
        valid = (values[:, 1] > sigma ** 2) & (values[:, 0] < .35 * values[:, 1])
        candidates.extend(vectors[valid, :, 0])
    if not candidates or not differences:
        return None, {'reason': 'insufficient within-scan planar support'}
    delta = np.concatenate(differences)
    delta = delta[np.linalg.norm(delta, axis=1) > 1e-10]
    normals = np.asarray(candidates)
    # Bounded residual score rejects cross-edge differences rather than fitting
    # their height change as one global ramp. Common scan translation cancels.
    cut = 3. * np.sqrt(2.) * sigma
    residual = delta @ normals.T
    scores = np.mean(np.minimum(residual ** 2, cut ** 2), axis=0)
    normal = normals[np.argmin(scores)].copy()
    for _ in range(5):
        residual = delta @ normal
        weight = np.square(np.maximum(0., 1. - (residual / cut) ** 2))
        if np.count_nonzero(weight) < 12:
            break
        _, vectors = np.linalg.eigh((delta * weight[:, None]).T @ delta)
        update = vectors[:, 0]
        normal = update if np.dot(update, normal) >= 0 else -update
    # A deterministic sign is a coordinate convention, not a reference surface.
    if normal[np.argmax(abs(normal))] < 0:
        normal = -normal
    agreement = float(np.mean(abs(normals @ normal) > np.cos(np.deg2rad(25))))
    if agreement < .35:
        return None, {'reason': 'no common-normal consensus', 'normal_agreement': agreement}
    _, global_axes = np.linalg.eigh(points.T @ points)
    tangent = global_axes[:, -1] - normal * np.dot(global_axes[:, -1], normal)
    tangent /= np.linalg.norm(tangent)
    second = np.cross(normal, tangent)
    return np.column_stack((tangent, second, normal)), {
        'normal_world': normal.tolist(), 'normal_agreement': agreement,
        'normal_candidate_count': len(normals), 'within_scan_difference_count': len(delta),
        'normal_estimator': 'within-scan nearest-neighbour differences and bounded-residual consensus',
    }


def _cells(local):
    """Identical input-derived partitions for graph and local_only."""
    xy = local[:, :2]
    lower, upper = np.min(xy, axis=0), np.max(xy, axis=0)
    extent = np.maximum(upper - lower, 1e-6)
    target = max(1, int(round(len(local) / 96)))
    nx = int(np.clip(round(np.sqrt(target * extent[0] / extent[1])), 1, 6))
    ny = int(np.clip(round(target / nx), 1, 6))
    bins = np.array([nx, ny])
    index = np.minimum(((xy - lower) / extent * bins).astype(int), bins - 1)
    _, labels = np.unique(index, axis=0, return_inverse=True)
    members = [np.flatnonzero(labels == cell) for cell in range(labels.max() + 1)]
    geometry = []
    for take in members:
        center = xy[take].mean(axis=0)
        scale = np.maximum(np.ptp(xy[take], axis=0), 1.)
        geometry.append(((xy[take] - center) / scale, scale))
    return labels, members, geometry, bins


def _weighted_fit(design, values, weight, penalty):
    lhs = design.T @ (design * weight[:, None]) + np.diag(penalty) + 1e-10 * np.eye(design.shape[1])
    return np.linalg.solve(lhs, design.T @ (weight * values))


def _local_model(values, xy, scale, sigma):
    """One/two parallel local planes with shared small tilt and fixed noise."""
    n = len(values)
    slope_sd = np.maximum(2. * sigma, .04 * scale)
    slope_penalty = sigma ** 2 / slope_sd ** 2
    one_design = np.c_[np.ones(n), xy]
    one = _weighted_fit(one_design, values, np.ones(n), np.r_[0., slope_penalty])
    one_residual = values - one_design @ one
    # Initialisation from a robustly fitted plane avoids fitting each cell's
    # absolute global tilt as its layer separation.
    for _ in range(3):
        weights = np.minimum(1., 3. * sigma / np.maximum(abs(one_residual), 1e-12))
        one = _weighted_fit(one_design, values, weights, np.r_[0., slope_penalty])
        one_residual = values - one_design @ one
    candidates = []
    for k in (1, 2):
        if k == 1:
            beta = one.copy()
            responsibility = np.ones((n, 1))
            pi = np.ones(1)
        else:
            levels = np.quantile(one_residual, [.2, .8]) + one[0]
            beta = np.r_[levels, one[1:]]
            pi = np.full(2, .5)
            for _ in range(12):
                predicted = beta[:2][None, :] + (xy @ beta[2:])[:, None]
                logp = -.5 * ((values[:, None] - predicted) / sigma) ** 2 + np.log(pi)
                responsibility = np.exp(logp - logsumexp(logp, axis=1)[:, None])
                design = np.zeros((2 * n, 4))
                design[:, :2] = np.tile(np.eye(2), (n, 1))
                design[:, 2:] = np.repeat(xy, 2, axis=0)
                updated = _weighted_fit(design, np.repeat(values, 2), responsibility.ravel(), np.r_[0., 0., slope_penalty])
                pi = np.maximum(responsibility.mean(axis=0), .03)
                pi /= pi.sum()
                change = np.max(abs(updated - beta))
                beta = updated
                if change < 1e-4:
                    break
        predicted = beta[:k][None, :] + (xy @ beta[k:])[:, None]
        logp = -.5 * ((values[:, None] - predicted) / sigma) ** 2 + np.log(pi)
        ll = logsumexp(logp, axis=1)
        responsibility = np.exp(logp - ll[:, None])
        assignments = np.argmax(responsibility, axis=1)
        hard = predicted[np.arange(n), assignments]
        # Penalise parameters once per cell; this is a local BIC-like model
        # choice, not an identifiability or physical-layer-count certificate.
        score = float(-2. * ll.sum() + (2 * k + 1) * np.log(n) + np.sum(slope_penalty * beta[k:] ** 2) / sigma ** 2)
        if k == 2 and (abs(beta[1] - beta[0]) < 2. * sigma or np.min(np.bincount(assignments, minlength=2)) < 5):
            continue
        candidates.append({'score': score, 'k': k, 'beta': beta, 'assigned': assignments,
                           'prediction': hard, 'confidence': responsibility.max(axis=1),
                           'residual': values - hard, 'slope': xy @ beta[k:]})
    return min(candidates, key=lambda model: model['score'])


def _components(nodes, scans, node_count, scan_count):
    parent = np.arange(node_count + scan_count)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for node, scan in zip(nodes, scans):
        a, b = root(int(node)), root(node_count + int(scan))
        parent[a] = b
    groups = {}
    for scan in range(scan_count):
        groups.setdefault(root(node_count + scan), []).append(scan)
    return list(groups.values())


def _joint_offsets(values, nodes, scans, scan_count, sigma, initial=None, gauge_counts=None):
    """Robust scan/surface bipartite LS, with one explicit gauge per component."""
    _, nodes = np.unique(nodes, return_inverse=True)
    node_count = int(nodes.max()) + 1
    design = np.zeros((len(values), node_count + scan_count))
    design[np.arange(len(values)), nodes] = 1.
    design[np.arange(len(values)), node_count + scans] = 1.
    components = _components(nodes, scans, node_count, scan_count)
    counts = (np.bincount(scans, minlength=scan_count).astype(float)
              if gauge_counts is None else np.asarray(gauge_counts, dtype=float))
    gauges = np.zeros((len(components), design.shape[1]))
    for j, group in enumerate(components):
        if counts[group].sum() > 0:
            gauges[j, node_count + np.array(group)] = counts[group] / counts[group].sum()
    parameters = np.zeros(design.shape[1])
    if initial is not None:
        parameters[node_count:] = initial
    for _ in range(5):
        residual = values - design @ parameters
        weights = np.minimum(1., 3. * sigma / np.maximum(abs(residual), 1e-12))
        lhs = design.T @ (weights[:, None] * design)
        # Gauge penalties establish zero, while the unregularised intercepts
        # allow exactly compensating each component's common translation.
        lhs += len(values) * (gauges.T @ gauges) + np.eye(lhs.shape[0]) * 1e-9
        parameters = np.linalg.solve(lhs, design.T @ (weights * values))
    bias = parameters[node_count:]
    for group in components:
        total = counts[group].sum()
        if total:
            bias[group] -= np.average(bias[group], weights=counts[group])
    return bias, components


def estimate(xyz_world_m, scan_id, sigma_mm, variant='graph'):
    """Return same-order world XYZ and JSON-friendly metadata; no GT inputs."""
    started = time.perf_counter()
    world = np.asarray(xyz_world_m, dtype=float)
    raw_scan = np.asarray(scan_id)
    sigma = float(sigma_mm)
    if variant not in ('graph', 'local_only'):
        raise ValueError('variant must be graph or local_only')
    if world.ndim != 2 or world.shape[1] != 3 or not len(world) or not np.isfinite(world).all():
        raise ValueError('finite nonempty N by 3 world coordinates required')
    if raw_scan.shape != (len(world),) or raw_scan.dtype.kind not in 'iu':
        raise ValueError('one integer scan ID per point required')
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma_mm must be positive and finite')
    ids, scans = np.unique(raw_scan, return_inverse=True)
    gauge_counts = np.bincount(scans, minlength=len(ids))
    center = world.mean(axis=0)
    centered = (world - center) * 1000.
    basis, normal_info = _basis(centered, scans, sigma)
    info = {'method': 'graph_surface', 'variant': variant, 'point_count': len(world),
            'scan_ids': ids.tolist(), 'sigma_mm': sigma, 'input_units': 'm',
            'gauge': 'point-count-weighted zero-mean scan bias within each connected component',
            'scope': 'single common normal; independent local 1/2 parallel tilted surfaces; exploratory',
            **normal_info}
    if basis is None:
        info.update(status='UNSUPPORTED', supported_fraction=0., unchanged_fraction=1.,
                    bias_mm=np.zeros(len(ids)).tolist(), seconds=time.perf_counter() - started)
        return world.copy(), info
    local = centered @ basis
    labels, members, geometry, grid = _cells(local)
    usable = np.array([len(take) >= 24 and len(np.unique(scans[take])) >= 2 for take in members])
    active = usable[labels]
    if np.count_nonzero(active) < 32:
        info.update(status='UNSUPPORTED', reason='insufficient multi-scan local support',
                    supported_fraction=0., unchanged_fraction=1., bias_mm=np.zeros(len(ids)).tolist(),
                    seconds=time.perf_counter() - started)
        return world.copy(), info
    initial = np.zeros(len(ids))
    initial_components = []
    if variant == 'graph':
        initial, initial_components = _joint_offsets(local[active, 2], labels[active], scans[active], len(ids), sigma, gauge_counts=gauge_counts)
    starts = [initial]
    if variant == 'graph' and np.max(abs(initial)) > .05 * sigma:
        starts.append(np.zeros(len(ids)))
    results = []
    for start in starts:
        bias = start.copy()
        for iteration in range(5 if variant == 'graph' else 1):
            models, nodes, slopes, selected = {}, [], [], []
            next_node = 0
            score = 0.
            for cell, take in enumerate(members):
                if not usable[cell]:
                    continue
                xy, scale = geometry[cell]
                model = _local_model(local[take, 2] - bias[scans[take]], xy, scale, sigma)
                models[cell] = model
                score += model['score']
                nodes.extend((next_node + model['assigned']).tolist())
                slopes.extend(model['slope'].tolist())
                selected.extend(take.tolist())
                next_node += model['k']
            if variant == 'local_only':
                break
            selected = np.asarray(selected, dtype=int)
            update, components = _joint_offsets(local[selected, 2] - np.asarray(slopes), np.asarray(nodes), scans[selected], len(ids), sigma, bias, gauge_counts)
            difference = np.max(abs(update - bias))
            bias = update
            if difference < .01 * sigma:
                break
        # Refit local geometry to the last joint bias; both arms use this exact
        # local fitting code, support policy and projection rule.
        models = {}
        score = 0.
        for cell, take in enumerate(members):
            if usable[cell]:
                xy, scale = geometry[cell]
                models[cell] = _local_model(local[take, 2] - bias[scans[take]], xy, scale, sigma)
                score += models[cell]['score']
        results.append((score, bias.copy(), models, iteration + 1))
    score, bias, models, iterations = min(results, key=lambda result: result[0])
    output = local.copy()
    support = np.zeros(len(world), dtype=bool)
    local_k, rejected = {}, {}
    final_nodes, final_scans, next_node = [], [], 0
    for cell, model in models.items():
        take = members[cell]
        final_nodes.extend((next_node + model['assigned']).tolist())
        final_scans.extend(scans[take].tolist())
        next_node += model['k']
        local_k[str(cell)] = model['k']
        xy, scale = geometry[cell]
        slope = model['beta'][model['k']:] / scale
        fit_ok = np.median(abs(model['residual'])) <= 2.5 * sigma and np.linalg.norm(slope) <= .25
        accepted = (abs(model['residual']) <= 4. * sigma) & (model['confidence'] >= .8) if fit_ok else np.zeros(len(take), dtype=bool)
        support[take[accepted]] = True
        output[take[accepted], 2] = model['prediction'][accepted]
        rejected[str(cell)] = int(np.count_nonzero(~accepted))
    # Apply only a scalar normal displacement to preserve tangential coordinates
    # and unsupported rows exactly in the original world-coordinate array.
    result = world.copy()
    result[support] += ((output[support, 2] - local[support, 2]) / 1000.)[:, None] * basis[:, 2]
    final_components = _components(final_nodes, final_scans, next_node, len(ids))
    info.update(status='APPLY' if support.any() else 'UNSUPPORTED', bias_mm=bias.tolist(),
                local_cell_count=len(members), usable_cell_count=int(usable.sum()), grid_shape=grid.tolist(),
                local_k=local_k, rejected_points_by_cell=rejected,
                supported_fraction=float(support.mean()), unchanged_fraction=float(1. - support.mean()),
                graph_components_scan_ids=[[int(ids[s]) for s in component] for component in final_components],
                initial_cell_graph_components_scan_ids=[[int(ids[s]) for s in component] for component in initial_components],
                gauge_point_counts=gauge_counts.tolist(), graph_surface_node_count=next_node,
                graph_connectivity_scope='current fitted assignments; connectivity does not prove unknown layer identifiability',
                objective=float(score), iterations=iterations, starts=len(starts),
                normal_displacement_rms_mm=float(np.sqrt(np.mean((output[:, 2] - local[:, 2]) ** 2))),
                seconds=time.perf_counter() - started)
    return result, info
