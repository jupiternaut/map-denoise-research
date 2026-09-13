"""Conditional Gaussian selection-noise compensation, without evaluator inputs.

This estimates a conditional mean of noise, not its unknown per-point draw.
The fitted finite mixture and supplied sigma are working assumptions, not GT.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtr

PROJECT = Path(__file__).resolve().parents[1]

def interval_noise_mean(means, prior, lower, upper, sigma_mm):
    """E[epsilon | lower < mu_H+epsilon < upper] in a fixed Gaussian mixture.

    Row-wise component means/prior, scalar common sigma in mm. Returns the
    conditional noise mean, event probability, and per-component event mass.
    Outside-tail CDF subtraction uses symmetry to avoid cancellation at +inf.
    """
    means, prior = np.asarray(means, float), np.asarray(prior, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    sigma = float(sigma_mm)
    if means.ndim != 2 or means.shape != prior.shape or not means.shape[1]:
        raise ValueError('matching nonempty N by H means and prior required')
    if lower.shape != (len(means),) or upper.shape != lower.shape:
        raise ValueError('one interval per row required')
    if not np.isfinite(means).all() or not np.isfinite(prior).all() or np.any(prior < 0):
        raise ValueError('finite means and nonnegative priors required')
    if not np.allclose(prior.sum(1), 1., atol=1e-12, rtol=0):
        raise ValueError('prior rows must sum to one')
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower >= upper):
        raise ValueError('nonempty intervals required')
    if not np.isfinite(sigma) or sigma <= 0: raise ValueError('positive sigma required')
    a, b = (lower[:, None]-means)/sigma, (upper[:, None]-means)/sigma
    probability = np.where(a >= 0, ndtr(-a)-ndtr(-b), ndtr(b)-ndtr(a))
    density_a = np.exp(-.5*a*a)/np.sqrt(2.*np.pi)
    density_b = np.exp(-.5*b*b)/np.sqrt(2.*np.pi)
    component_mass = prior*np.maximum(probability, 0.)
    mass = component_mass.sum(1)
    first = sigma*np.sum(prior*(density_a-density_b), axis=1)
    mean = np.divide(first, mass, out=np.zeros_like(first), where=mass > 1e-14)
    if not np.isfinite(mean).all(): raise FloatingPointError('nonfinite conditional mean')
    return mean, mass, component_mass

def selection_bias(design, coefficients, candidate_mask, assigned_group, sigma_mm):
    """Analytic noise mean for uniform-prior nearest-plane-height selection.

    At each fixed tangent coordinate, MAP regions are 1-D Voronoi intervals.
    Coincident candidate heights share one geometric interval, while all prior
    component masses are retained. No approximate surface merge is performed.
    """
    x, beta = np.asarray(design, float), np.asarray(coefficients, float)
    mask, groups = np.asarray(candidate_mask, bool), np.asarray(assigned_group, int)
    if x.ndim != 2 or x.shape[1] != 3 or beta.ndim != 2 or beta.shape[1] != 3:
        raise ValueError('affine plane design and coefficients required')
    if mask.shape != (len(x), len(beta)) or groups.shape != (len(x),):
        raise ValueError('candidate/group shape mismatch')
    if not np.isfinite(x).all() or not np.isfinite(beta).all(): raise ValueError('finite input required')
    if np.any(groups < 0) or np.any(groups >= len(beta)) or not mask.any(1).all():
        raise ValueError('valid groups and candidates required')
    if not mask[np.arange(len(x)), groups].all(): raise ValueError('inaccessible chosen plane')
    means = x@beta.T
    selected = means[np.arange(len(x)), groups]
    # Exact equal heights have equal likelihood for every possible noisy value.
    below = mask & (means < selected[:, None])
    above = mask & (means > selected[:, None])
    prev = np.max(np.where(below, means, -np.inf), axis=1)
    nxt = np.min(np.where(above, means, np.inf), axis=1)
    lower, upper = .5*(prev+selected), .5*(nxt+selected)
    prior = mask.astype(float)/mask.sum(1)[:, None]
    noise, mass, component_mass = interval_noise_mean(means, prior, lower, upper, sigma_mm)
    return noise, dict(selection_lower_mm=lower, selection_upper_mm=upper,
         modeled_selection_probability=mass, selection_component_mass=component_mass,
         underflow_fallback=mass <= 1e-14, candidate_mean_mm=means, candidate_prior=prior)

def _fit_and_project(state, groups, corrected_values):
    active, order = state['active'], state['order']
    x, weights = state['design'][active], state['weights'][active]
    ids = np.unique(groups)
    predicted = state['local'][:, 2].copy()
    coefficients, ranks = [], []
    for group in ids:
        rows = groups == group
        root_w = np.sqrt(weights[rows])
        coefficient, _, rank, _ = np.linalg.lstsq(x[rows]*root_w[:, None],
                                                 corrected_values[rows]*root_w, rcond=1e-12)
        predicted[active[rows]] = x[rows]@coefficient
        coefficients.append(coefficient); ranks.append(int(rank))
    output = state['world'].copy()
    supported = np.flatnonzero(state['support'])
    original = order[supported]
    output[original] += (predicted[supported]-state['local'][supported, 2])[:, None]*state['normal']/1000.
    return output, np.asarray(coefficients), ids, ranks

def compensate_frozen(state, artifacts, sigma_mm, mode='conditional', random_seed=913801):
    """Apply a new output action to frozen legal V7 state, no truth parameter.

    Modes: conditional (working-model noise mean), none (old hard WLS replay),
    random (independent same-scale noise subtraction, negative control).
    Upstream, observed MAP groups, fitting rows, point weights and support stay fixed.
    """
    started = time.perf_counter()
    if mode not in ('conditional', 'none', 'random'): raise ValueError('unknown mode')
    sigma = float(sigma_mm)
    if not np.isfinite(sigma) or sigma <= 0: raise ValueError('positive sigma required')
    active, order, world = state['active'], state['order'], state['world']
    world_rows = order[active]
    groups = np.asarray(artifacts['group_ids'])[world_rows]
    n = len(world)
    support = np.zeros(n, bool); support[order] = state['support']
    if not len(active):
        return world.copy(), dict(mode=mode, status='UNSUPPORTED', truth_fields_used=[]), {
            'support_mask': support, 'group_ids': np.full(n,-1,int),
            'subtracted_noise_mm': np.zeros(n)}
    x, beta = state['design'][active], artifacts['coefficients']
    mask = np.asarray(artifacts['candidate_mask'])[world_rows]
    if mode == 'conditional':
        noise, details = selection_bias(x, beta, mask, groups, sigma)
    else:
        noise = np.zeros(len(active)) if mode == 'none' else np.random.default_rng(random_seed).normal(0.,sigma,len(active))
        details = {}
    output, fitted, ids, ranks = _fit_and_project(state, groups, state['corrected'][active]-noise)
    if not np.isfinite(output).all(): raise FloatingPointError('nonfinite output')
    np.testing.assert_array_equal(output[~support], world[~support])
    noise_world = np.zeros(n); noise_world[world_rows] = noise
    result = dict(support_mask=support, group_ids=np.asarray(artifacts['group_ids']).copy(),
                  subtracted_noise_mm=noise_world, compensated_coefficients=fitted,
                  compensated_coefficient_ids=ids, active_original_indices=world_rows.copy())
    result.update(details)
    info = dict(method='conditional_selection_noise_compensation_v8', mode=mode,
         sigma_mm=sigma, point_count=n, active_count=len(active), group_count=len(ids),
         fit_parameter_count=3*len(ids), fit_rank=sum(ranks), group_fit_ranks=ranks,
         supported_fraction=float(support.mean()), truth_fields_used=[],
         mean_absolute_subtracted_noise_mm=float(np.mean(abs(noise))),
         max_absolute_subtracted_noise_mm=float(np.max(abs(noise))),
         interpretation='conditional fitted-mixture correction, not recovered noise realization',
         sigma_scope='supplied existing scale, no blind noise estimation',
         generative_model='uniform over accessible saved soft-M planes; no assumed single true plane',
         selection_scope='nearest fixed height at observed tangent coordinate; numerical MAP ties have measure-zero idealization',
         uncertainty_scope='does not model parameter-estimation, association or upstream frame-bias error',
         low_probability_fallbacks=int(np.sum(details.get('underflow_fallback', []))),
         actual_changed_fraction=float(np.mean(np.any(output != world, axis=1))),
         seconds=time.perf_counter()-started)
    return output, info, result

def estimate(xyz_world_m, scan_id, sigma_mm, budget=6, mode='conditional'):
    """Convenience legal-input estimator. Includes old V7 search, then new action.

    This wrapper currently also computes/discards V7 hard output; no efficiency
    claim is based on cached-action timing.
    """
    spec = importlib.util.spec_from_file_location('_v8_legal_v7', PROJECT/'exploration_v7/algorithm/reassociation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    state = module.freeze(xyz_world_m, scan_id, sigma_mm)
    _, _, artifacts = module.fit_frozen(state, variant='reassociate', budget=budget, sharing='independent')
    return compensate_frozen(state, artifacts, sigma_mm, mode)
