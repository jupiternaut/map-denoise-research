"""Reference-free local surface compatibility and finite displacement optimization."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def estimate_normals(points, k=16):
    p = np.asarray(points, float)
    if len(p) < 3:
        return np.tile([0., 0., 1.], (len(p), 1))
    _, ids = cKDTree(p).query(p, k=min(k, len(p)), workers=1)
    q = p[ids] - p[ids].mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', q, q)
    _, vec = np.linalg.eigh(cov)
    normal = vec[:, :, 0]
    sign = np.where(normal[np.arange(len(p)), np.argmax(abs(normal), axis=1)] < 0, -1, 1)
    return normal * sign[:, None]


def build_graph(points, normals, k=8):
    p, n = np.asarray(points, float), np.asarray(normals, float)
    if len(p) <= 1:
        return {'edges': np.empty((0, 2), int), 'weights': np.empty(0), 'normal': np.empty((0, 3)), 'degree': np.zeros(len(p))}
    dist, ids = cKDTree(p).query(p, k=min(k+1, len(p)), workers=1)
    pairs = np.column_stack([np.repeat(np.arange(len(p)), ids.shape[1]-1), ids[:, 1:].ravel()])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    a, b = pairs.T
    dot = np.sum(n[a]*n[b], axis=1)
    average = n[a] + n[b]*np.where(dot < 0, -1, 1)[:, None]
    average /= np.maximum(np.linalg.norm(average, axis=1, keepdims=True), 1e-12)
    scale = max(float(np.median(dist[:, -1])), 1e-6)
    distance = np.linalg.norm(p[a]-p[b], axis=1)
    weight = np.exp(-.5*(distance/scale)**2) * np.clip(abs(dot), 0, 1)**8
    degree = np.bincount(pairs.ravel(), weights=np.repeat(weight, 2), minlength=len(p))
    weight *= 2 / max(float(degree.mean()), 1e-12)
    degree = np.bincount(pairs.ravel(), weights=np.repeat(weight, 2), minlength=len(p))
    return {'edges': pairs, 'weights': weight, 'normal': average, 'degree': degree, 'scale_mm': scale}


def offsets_array(offsets, n):
    offsets = np.asarray(offsets, float)
    return np.broadcast_to(offsets, (n, len(offsets))) if offsets.ndim == 1 else offsets


def closest_minimum(cost, offsets):
    cost = np.asarray(cost, float)
    offsets = offsets_array(offsets, len(cost))
    best = cost.min(axis=1, keepdims=True)
    tie = np.isclose(cost, best, rtol=0, atol=1e-12)
    return np.argmin(np.where(tie, abs(offsets), np.inf), axis=1)


def prepare_cost(aggregate):
    valid = np.asarray(aggregate['valid'], bool)
    cost = np.where(valid, aggregate['cost'], 1e4)
    cost[~valid.any(axis=1)] = 1.
    return cost


def field_energy(points, directions, offsets, cost, graph, choice, lam=.1, pair_sigma=1., trunc=4.):
    p, d = np.asarray(points), np.asarray(directions)
    offset = offsets_array(offsets, len(p))[np.arange(len(p)), choice]
    q = p + d*offset[:, None]
    edges = graph['edges']
    pair = np.sum((q[edges[:, 0]]-q[edges[:, 1]])*graph['normal'], axis=1)
    return float(cost[np.arange(len(p)), choice].sum() + lam*np.sum(graph['weights']*np.minimum((pair/pair_sigma)**2, trunc)))


def solve_field(points, directions, offsets, cost, graph, lam=.1, pair_sigma=1., trunc=4., max_sweeps=12):
    p, d, cost = np.asarray(points, float), np.asarray(directions, float), np.asarray(cost, float)
    offset = offsets_array(offsets, len(p))
    if offset.shape != cost.shape or not np.isfinite(cost).all():
        raise ValueError('finite unary and matching offsets required')
    edges, normals, weights = graph['edges'], graph['normal'], graph['weights']
    adj = [[] for _ in range(len(p))]
    for ei, (i, j) in enumerate(edges):
        adj[i].append((ei, j)); adj[j].append((ei, i))
    # Each node uses the common edge normal; reversing both signs leaves square unchanged.
    local = []
    for i, links in enumerate(adj):
        if not links:
            local.append(None); continue
        ei, js = np.asarray(links, int).T
        nn = normals[ei]
        base = np.einsum('ij,ij->i', p[i]-p[js], nn)
        ai = nn@d[i]
        aj = np.einsum('ij,ij->i', d[js], nn)
        local.append((js, weights[ei], base, ai, aj))
    starts = {'photometric': closest_minimum(cost, offset), 'zero': np.argmin(abs(offset), axis=1)}
    trials = {}
    for name, initial in starts.items():
        choice = initial.copy(); value = offset[np.arange(len(p)), choice]
        history = [field_energy(p, d, offset, cost, graph, choice, lam, pair_sigma, trunc)]
        for sweep in range(max_sweeps):
            changes = 0
            order = range(len(p)) if sweep % 2 == 0 else range(len(p)-1, -1, -1)
            for i in order:
                energy = cost[i].copy()
                if local[i] is not None and lam:
                    js, ww, base, ai, aj = local[i]
                    residual = base[None, :] + offset[i, :, None]*ai[None, :] - value[js][None, :]*aj[None, :]
                    energy += lam*(np.minimum((residual/pair_sigma)**2, trunc)*ww).sum(axis=1)
                best = np.min(energy)
                candidates = np.flatnonzero(np.isclose(energy, best, rtol=0, atol=1e-12))
                proposal = candidates[np.argmin(abs(offset[i, candidates]))]
                if energy[proposal] < energy[choice[i]] - 1e-12:
                    choice[i] = proposal; value[i] = offset[i, proposal]; changes += 1
            current = field_energy(p, d, offset, cost, graph, choice, lam, pair_sigma, trunc)
            if current > history[-1] + 1e-7:
                raise AssertionError(('coordinate descent increased energy', history[-1], current))
            history.append(current)
            if not changes:
                break
        trials[name] = {'choice': choice, 'offset': value, 'energy_history': history}
    winner = min(trials, key=lambda s: trials[s]['energy_history'][-1])
    return {**trials[winner], 'best_start': winner, 'all_start_energy_history': {k:v['energy_history'] for k,v in trials.items()},
            'solver': 'deterministic finite-coordinate descent; no global optimality certificate'}


def interpolation_weights(points, normals, anchors, anchor_normals, k=4):
    distance, ids = cKDTree(anchors).query(points, k=min(k, len(anchors)), workers=1)
    if distance.ndim == 1:
        distance, ids = distance[:, None], ids[:, None]
    similarity = abs(np.sum(normals[:, None, :]*anchor_normals[ids], axis=2))**8
    weight = (similarity + .01) / np.maximum(distance, .05)**2
    exact = distance[:, 0] < 1e-9
    weight[exact] = 0.; weight[exact, 0] = 1.
    weight /= weight.sum(axis=1, keepdims=True)
    return ids, weight


def interpolate(offset, ids, weights):
    return np.sum(np.asarray(offset)[ids]*weights, axis=1)
