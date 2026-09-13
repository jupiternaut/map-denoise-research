"""Physical-distance diagnostics, not a point denoiser or true geometry oracle."""
from __future__ import annotations

import numpy as np


def graph_plane_distances(values, xy_normalized, beta, assignments, scales):
    """Use physical mm slopes after undoing per-coordinate normalization.

    `values` may be conditional on an upstream scan bias. Returned distances
    are then to those conditional planes, not independently known surfaces.
    """
    values = np.asarray(values, float)
    assignments = np.asarray(assignments, int)
    k = len(beta) - 2
    slopes = np.asarray(beta[k:]) / np.asarray(scales)
    predicted = np.asarray(beta[:k])[assignments] + xy_normalized @ beta[k:]
    signed_height = values - predicted
    inverse_cosine = float(np.sqrt(1.0 + slopes @ slopes))
    cosine = 1.0 / inverse_cosine
    normal = np.r_[-slopes, 1.] * cosine
    return dict(height_mm=abs(signed_height), orthogonal_mm=abs(signed_height) * cosine,
                axis_cosine=cosine, axis_condition=inverse_cosine,
                normal_local=normal, physical_slopes=slopes)


def tls_plane(points):
    points = np.asarray(points, float)
    if len(points) < 3:
        raise ValueError('three training points are needed')
    center = points.mean(0)
    centered = points - center
    _, values, vt = np.linalg.svd(centered, full_matrices=False)
    if values[1] <= 1e-10 * max(values[0], 1.):
        raise ValueError('training plane has rank below two')
    normal = vt[-1].copy()
    if normal[np.argmax(abs(normal))] < 0:
        normal *= -1
    return center, normal, int(np.linalg.matrix_rank(centered))


def fit_planes(training_points, k):
    """Small fixed-capacity orthogonal diagnostic, fit only to training rows.

    K=2 uses input PCA median partitions along each of three axes, plus a
    normal-residual partition; hard assignment/refit is not a novel method.
    No test point, scan bias, gate, truth or test-based selector enters here.
    """
    points = np.asarray(training_points, float)
    if k not in (1, 2):
        raise ValueError('fixed capacity must be one or two')
    center, normal, rank = tls_plane(points)
    if k == 1:
        return dict(centers=np.array([center]), normals=np.array([normal]),
                    ranks=[rank], counts=[len(points)], train_sse=float(np.sum(((points-center)@normal)**2)),
                    capacity=1, iterations=1, starts=1, actual_faces=1)
    if len(points) < 12:
        raise ValueError('two-plane diagnostic needs at least 12 training rows')
    _, _, vt = np.linalg.svd(points-center, full_matrices=False)
    best = None
    for direction in list(vt) + [normal]:
        projected = (points-center) @ direction
        assignment = (projected > np.median(projected)).astype(int)
        for iteration in range(12):
            if min(np.bincount(assignment, minlength=2)) < 6:
                break
            try:
                planes = [tls_plane(points[assignment == j]) for j in range(2)]
            except ValueError:
                break
            centers = np.array([p[0] for p in planes])
            normals = np.array([p[1] for p in planes])
            distances = abs(np.einsum('nkd,kd->nk', points[:, None]-centers, normals))
            updated = np.argmin(distances, axis=1)
            if np.array_equal(assignment, updated):
                break
            assignment = updated
        if min(np.bincount(assignment, minlength=2)) < 6:
            continue
        try:
            planes = [tls_plane(points[assignment == j]) for j in range(2)]
        except ValueError:
            continue
        centers = np.array([p[0] for p in planes])
        normals = np.array([p[1] for p in planes])
        residual = np.min(abs(np.einsum('nkd,kd->nk', points[:, None]-centers, normals)), axis=1)
        candidate = dict(centers=centers, normals=normals, ranks=[p[2] for p in planes],
                         counts=np.bincount(assignment, minlength=2).tolist(),
                         train_sse=float(residual @ residual), capacity=2,
                         iterations=iteration+1, starts=4, actual_faces=2)
        if best is None or candidate['train_sse'] < best['train_sse']:
            best = candidate
    if best is None:
        raise ValueError('no two nondegenerate fitted components')
    return best


def plane_errors(points, model, axis):
    """Test assignment is nearest fixed training plane, never a refit."""
    signed = np.einsum('nkd,kd->nk', np.asarray(points)[:, None]-model['centers'], model['normals'])
    assignment = np.argmin(abs(signed), axis=1)
    orthogonal = abs(signed[np.arange(len(points)), assignment])
    cosine = abs(model['normals'][assignment] @ np.asarray(axis))
    axis_distance = np.divide(orthogonal, cosine, out=np.full_like(orthogonal, np.inf), where=cosine>1e-12)
    return orthogonal, axis_distance, cosine, assignment


def spatial_stripe_holdout(points, scans):
    """Predefined per-scan spatial design, not residual-selected row sampling.

    Eight empirical-quantile intervals along each scan's first PCA axis;
    intervals 1 and 5 are test. Equal projected coordinates stay together.
    Full-current coordinates define design only, not any fitted target plane.
    """
    points = np.asarray(points, float)
    scans = np.asarray(scans)
    result = np.zeros(len(points), bool)
    for sid in np.unique(scans):
        take = np.flatnonzero(scans == sid)
        centered = points[take]-points[take].mean(0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        if direction[np.argmax(abs(direction))] < 0:
            direction = -direction
        projection = centered@direction
        edges = np.quantile(projection,np.arange(1,8)/8.)
        stripe = np.searchsorted(edges,projection,side='right')
        result[take] = np.isin(stripe,[1,5])
    return result
