"""Observation-only patchwise one/two-layer inverse-depth quadratic fields.

This pure NumPy module has no file, camera-image or evaluation-reference access.
Each tile shares six coefficients per layer across points. Halo points inform
the fit, while only core points receive that tile's predictions. Layer labels
are local to a tile, never persistent physical surface identifiers.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldResult:
    depth: np.ndarray
    responsibility: np.ndarray
    labels: np.ndarray
    patch_index: np.ndarray
    supported: np.ndarray
    patches: list


def quadratic_basis(uv_normalized):
    u, v = np.asarray(uv_normalized, dtype=np.float64).T
    return np.column_stack((np.ones(len(u)), u, v, u*u, u*v, v*v))


def _softmax_candidates(cost, valid, temperature):
    logits = np.where(valid, -cost / temperature, -np.inf)
    peak = logits.max(axis=1, keepdims=True)
    weight = np.exp(logits - peak)
    return weight / weight.sum(axis=1, keepdims=True)


def _weighted_fit(basis, row_weight, target_sum, ridge_matrix):
    hessian = basis.T @ (row_weight[:, None] * basis) + ridge_matrix
    rhs = basis.T @ target_sum
    # Least squares also defines a finite minimum-norm result for collinear UV.
    return np.linalg.lstsq(hessian, rhs, rcond=None)[0]


def _weighted_quantiles(values, weights, probabilities):
    order = np.argsort(values, kind='stable')
    x, w = values[order], weights[order]
    cumulative = np.cumsum(w)
    if cumulative[-1] <= 0:
        raise ValueError('positive quantile weight required')
    return np.interp(np.asarray(probabilities) * cumulative[-1], cumulative, x)


def _expectation(basis, rho, cost, valid, beta, prior, temperature, sigma):
    prediction = basis @ beta.T
    residual = rho[:, :, None] - prediction[:, None, :]
    logits = (-cost[:, :, None] / temperature
              - residual**2 / (2 * sigma**2)
              + np.log(prior)[None, None, :])
    logits = np.where(valid[:, :, None], logits, -np.inf)
    peak = logits.max(axis=(1, 2), keepdims=True)
    weight = np.exp(logits - peak)
    normalizer = weight.sum(axis=(1, 2), keepdims=True)
    joint = weight / normalizer
    # Uniform candidate prior prevents a row's grid count changing its mass.
    log_evidence = (peak[:, 0, 0] + np.log(normalizer[:, 0, 0])
                    - np.log(valid.sum(axis=1)))
    return joint, log_evidence


def _fit_patch(basis, rho, cost, valid, n_layers, iterations,
               temperature, sigma, ridge):
    n = len(basis)
    penalty = np.array([0., .01, .01, 1., 1., 1.])
    ridge_matrix = np.diag(ridge * n * penalty)
    photo = _softmax_candidates(cost, valid, temperature)
    mean_rho = (photo * rho).sum(axis=1)
    base = _weighted_fit(basis, np.ones(n), mean_rho, ridge_matrix)
    if n_layers == 1:
        initializers = [('single', base[None, :])]
    else:
        residual = rho - (basis @ base)[:, None]
        offsets = _weighted_quantiles(residual[valid], photo[valid], [.2, .8])
        split = np.tile(base, (2, 1)); split[:, 0] += offsets
        initializers = [('residual_quantiles', split)]
        for column, name in ((1, 'spatial_u'), (2, 'spatial_v')):
            left = basis[:, column] <= np.median(basis[:, column])
            halves = []
            for mask in (left, ~left):
                halves.append(_weighted_fit(basis, mask.astype(float),
                    mean_rho * mask, ridge_matrix) if mask.any() else base.copy())
            initializers.append((name, np.stack(halves)))
    fits = []
    for name, initial in initializers:
        beta = initial.copy()
        prior = np.full(n_layers, 1. / n_layers)
        for _ in range(iterations):
            joint, _ = _expectation(basis, rho, cost, valid, beta, prior,
                                     temperature, sigma)
            for k in range(n_layers):
                weight = joint[:, :, k]
                beta[k] = _weighted_fit(basis, weight.sum(axis=1),
                                        (weight * rho).sum(axis=1), ridge_matrix)
            prior = np.maximum(joint.sum(axis=(0, 1)), 1e-12)
            prior /= prior.sum()
        joint, log_evidence = _expectation(basis, rho, cost, valid, beta, prior,
                                         temperature, sigma)
        objective = float(-log_evidence.sum()
            + .5 * np.sum(beta**2 * np.diag(ridge_matrix)[None, :]) / sigma**2)
        fits.append((objective, name, beta, prior, joint))
    chosen = min(fits, key=lambda entry: entry[0])
    return chosen, [dict(initializer=f[1], objective=f[0]) for f in fits]


def fit_local_fields(uv, candidate_depth, photo_cost, valid, *, input_depth=None,
                     n_layers=1, view_count=None, tile_size=32., halo=16.,
                     min_points=12, iterations=12, temperature=.15, sigma=.75,
                     ridge=1e-5):
    """Fit shared inverse-depth fields from ray proposals and photometric costs.

    uv: (N,2) reference-image pixels. candidate_depth / photo_cost / valid:
    (N,C) arrays. A True validity entry must already mean at least two valid
    source views; optional view_count applies that requirement explicitly.
    input_depth defaults to candidate_depth[:,0]. A row without any valid
    candidate contributes no fitting weight and is returned unchanged.

    Depth is physical positive camera Z in mm. For each patch centre depth zc,
    fit rho = zc**2 * (1/z - 1/zc), also measured in mm. Prediction inversion is
    exact (the interpretation rho approximately -delta_z is only local).
    No output clipping to candidate ranges or old A/B segments is performed;
    the integrating operator applies the declared 6mm ray-motion limit.
    """
    uv = np.asarray(uv, dtype=np.float64)
    depths = np.asarray(candidate_depth, dtype=np.float64)
    cost = np.asarray(photo_cost, dtype=np.float64)
    good = np.asarray(valid, dtype=bool).copy()
    if (uv.ndim != 2 or uv.shape[1] != 2 or depths.ndim != 2
            or depths.shape[0] != len(uv) or cost.shape != depths.shape
            or good.shape != depths.shape or depths.shape[1] == 0):
        raise ValueError('expected uv[N,2] and matching depth/cost/valid[N,C]')
    if not np.isfinite(uv).all():
        raise ValueError('uv must be finite')
    if n_layers not in (1, 2) or tile_size <= 0 or halo < 0:
        raise ValueError('one/two layers and positive tile size required')
    if min_points < 1 or iterations < 1 or temperature <= 0 or sigma <= 0 or ridge < 0:
        raise ValueError('invalid fit parameters')
    original = depths[:, 0].copy() if input_depth is None else np.asarray(input_depth, dtype=float)
    if original.shape != (len(uv),) or not np.isfinite(original).all() or np.any(original <= 0):
        raise ValueError('input_depth must be finite positive [N]')
    good &= np.isfinite(depths) & (depths > 0) & np.isfinite(cost)
    if view_count is not None:
        views = np.asarray(view_count)
        if views.shape != depths.shape:
            raise ValueError('view_count must match candidate_depth')
        good &= views >= 2
    # Mask invalid numerics before EM: an invalid candidate never gets weight.
    clean_cost = np.where(good, cost, 0.)
    clean_depth = np.where(good, depths, original[:, None])
    row_valid = good.any(axis=1)
    depth_out = np.repeat(original[:, None], n_layers, axis=1)
    responsibility = np.zeros((len(uv), n_layers), dtype=float)
    labels = np.full(len(uv), -1, dtype=np.int16)
    patch_index = np.full(len(uv), -1, dtype=np.int32)
    supported = np.zeros(len(uv), dtype=bool)
    tiles = np.floor(uv / tile_size).astype(np.int64)
    patches = []
    for tile in np.unique(tiles, axis=0):
        core = np.all(tiles == tile, axis=1)
        core_ids = np.flatnonzero(core)
        centre = (tile + .5) * tile_size
        fit_mask = (np.all(np.abs(uv-centre) <= tile_size*.5+halo, axis=1)
                    & row_valid)
        fit_ids = np.flatnonzero(fit_mask)
        pid = len(patches)
        patch_index[core_ids] = pid
        record = dict(patch=pid, tile=tile.tolist(), centre_uv=centre.tolist(),
                      n_core=int(core.sum()), n_fit=len(fit_ids), layers=n_layers)
        if len(fit_ids) < min_points:
            record.update(status='input_fallback', reason='fewer_than_min_points')
            patches.append(record)
            continue
        zc = float(np.median(original[fit_ids]))
        rho = zc**2 * (1. / clean_depth[fit_ids] - 1. / zc)
        basis = quadratic_basis((uv[fit_ids]-centre) / tile_size)
        fit, initialization_scores = _fit_patch(basis, rho, clean_cost[fit_ids],
            good[fit_ids], n_layers, iterations, temperature, sigma, ridge)
        objective, name, beta, prior, joint = fit
        active_core = np.flatnonzero(core & row_valid)
        output_basis = quadratic_basis((uv[active_core]-centre) / tile_size)
        denominator = 1. / zc + (output_basis @ beta.T) / zc**2
        finite_prediction = np.all(np.isfinite(denominator) & (denominator > 0), axis=1)
        valid_core = active_core[finite_prediction]
        prediction = 1. / denominator[finite_prediction]
        depth_out[valid_core] = prediction
        # fit_ids are ascending and contain every active core row.
        fit_rows = np.searchsorted(fit_ids, valid_core)
        responsibility[valid_core] = joint[fit_rows].sum(axis=1)
        labels[valid_core] = responsibility[valid_core].argmax(axis=1)
        supported[valid_core] = True
        record.update(status='fit', z_center_mm=zc, coefficients=beta.tolist(),
                      layer_prior=prior.tolist(), objective=objective,
                      initializer=name, initialization_scores=initialization_scores,
                      iterations=iterations, n_supported_core=len(valid_core),
                      temperature=temperature, sigma_rho_mm=sigma, ridge=ridge)
        patches.append(record)
    return FieldResult(depth_out, responsibility, labels, patch_index, supported, patches)
