"""Frozen selection-mixture moment update; measured inputs only.

This is a one-step working-model estimate, not physical identification or a
novelty claim. Candidate probabilities and noise means are frozen at the seed.
Observation-dependent inherited weights need not preserve population moments.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    '_v9_unmix_v8', Path(__file__).resolve().parents[1]/'exploration_v8/compensation.py')
_V8 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V8)
RCOND = 1e-8
MASS_THRESHOLD = 1e-14


def unmix_frozen(state, artifacts, sigma_mm, mode='offset'):
    """Estimate candidate corrections from group-event mixture moments.

For event G_i and candidate h, A_ih=P(H=h | x_i,G_i). The regression is
    corrected_y_i - b_i = sum_h A_ih x_i^T beta_h + residual_i.
Offset mode changes candidate intercepts only; affine mode changes all three
coefficients. Solve for the minimum-norm delta around beta_seed, retaining the
seed's unidentifiable right-nullspace component under the fixed SVD cutoff.
Final output projects to the unchanged selected candidate, never the mixture
mean. Low-mass rows are excluded from fitting and leave their input unchanged.
"""
    started = time.perf_counter()
    if mode not in ('offset', 'affine'):
        raise ValueError('mode must be offset or affine')
    sigma = float(sigma_mm)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('positive finite sigma required')
    world = np.asarray(state['world'], float)
    order = np.asarray(state['order'], int)
    active = np.asarray(state['active'], int)
    world_rows = order[active]
    support = np.zeros(len(world), bool)
    support[order] = np.asarray(state['support'], bool)
    groups_world = np.asarray(artifacts['group_ids'], int)
    beta_seed = np.asarray(artifacts['coefficients'], float)
    if groups_world.shape != (len(world),) or beta_seed.ndim != 2 or beta_seed.shape[1] != 3:
        raise ValueError('group/coefficient dimensions invalid')
    base_info = dict(method='frozen_selection_mixture_moment_update_v9', mode=mode,
                     sigma_mm=sigma, rcond=RCOND, truth_fields_used=[],
                     interpretation='one frozen working-model moment update; not an identified physical solution',
                     weight_scope='inherited observation-dependent weights, no unbiased WLS guarantee',
                     output_action='fixed selected candidate projection, no reassociation or hard-group refit',
                     low_probability_action='exclude fitting row and keep its measured input unchanged',
                     point_count=len(world), active_count=len(active),
                     supported_fraction=float(support.mean()))
    if not len(active):
        return world.copy(), dict(base_info, status='UNSUPPORTED', fit_rank=0,
                                  low_probability_rows=0, seconds=time.perf_counter()-started), {
            'support_mask': support, 'group_ids': groups_world.copy(),
            'coefficients': beta_seed.copy(), 'coefficient_delta': np.zeros_like(beta_seed),
            'active_original_indices': world_rows.copy(), 'fit_row_mask': np.zeros(len(world), bool),
            'singular_values': np.empty(0)}
    x = np.asarray(state['design'], float)[active]
    y = np.asarray(state['corrected'], float)[active]
    weights = np.asarray(state['weights'], float)[active]
    groups = groups_world[world_rows]
    mask = np.asarray(artifacts['candidate_mask'], bool)[world_rows]
    if not np.isfinite(y).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('finite response and nonnegative finite weights required')
    noise, details = _V8.selection_bias(x, beta_seed, mask, groups, sigma)
    mass = details['modeled_selection_probability']
    low = mass <= MASS_THRESHOLD
    mixture = np.divide(details['selection_component_mass'], mass[:, None],
                        out=np.zeros_like(details['selection_component_mass']),
                        where=mass[:, None] > MASS_THRESHOLD)
    use = ~low & (weights > 0)
    baseline = np.sum(mixture*details['candidate_mean_mm'], axis=1)
    response = y-noise-baseline
    matrix = mixture if mode == 'offset' else (mixture[:, :, None]*x[:, None, :]).reshape(len(x), -1)
    root = np.sqrt(weights[use])
    weighted_matrix = matrix[use]*root[:, None]
    weighted_response = response[use]*root
    if np.any(use):
        delta, _, rank, singular = np.linalg.lstsq(weighted_matrix, weighted_response, rcond=RCOND)
    else:
        delta = np.zeros(matrix.shape[1]); rank = 0; singular = np.empty(0)
    change = np.zeros_like(beta_seed)
    if mode == 'offset':
        change[:, 0] = delta
    else:
        change[:] = delta.reshape(beta_seed.shape)
    beta_new = beta_seed+change
    predicted = np.einsum('ij,ij->i', x, beta_new[groups])
    output = world.copy()
    local_z = np.asarray(state['local'], float)[active, 2]
    movable = ~low & support[world_rows]
    output[world_rows[movable]] += ((predicted[movable]-local_z[movable])/1000.)[:, None]*state['normal']
    if not np.isfinite(output).all() or not np.isfinite(beta_new).all():
        raise FloatingPointError('nonfinite moment-update output')
    np.testing.assert_array_equal(output[~support], world[~support])
    np.testing.assert_array_equal(output[world_rows[low]], world[world_rows[low]])
    fit_world = np.zeros(len(world), bool); fit_world[world_rows] = use
    low_world = np.zeros(len(world), bool); low_world[world_rows] = low
    noise_world = np.zeros(len(world)); noise_world[world_rows] = noise
    mixture_world = np.zeros((len(world), len(beta_seed))); mixture_world[world_rows] = mixture
    before_residual = weighted_response
    after_residual = weighted_response-weighted_matrix@delta
    info = dict(base_info, status='APPLY' if movable.any() else 'NO_SUPPORTED_EVENT',
                candidate_count=len(beta_seed), fit_parameter_count=matrix.shape[1],
                fit_rows=int(use.sum()), fit_rank=int(rank),
                numerical_nullity=int(matrix.shape[1]-rank),
                singular_values=singular.tolist(),
                effective_condition=float(singular[0]/singular[rank-1]) if rank else None,
                low_probability_rows=int(low.sum()), zero_weight_rows=int(np.sum(weights == 0)),
                weighted_moment_sse_before_mm2=float(before_residual@before_residual),
                weighted_moment_sse_after_mm2=float(after_residual@after_residual),
                coefficient_delta_l2=float(np.linalg.norm(change)),
                maximum_abs_intercept_change_mm=float(np.max(abs(change[:, 0]))),
                maximum_abs_normalized_slope_change=float(np.max(abs(change[:, 1:]))),
                input_edit_rms_mm=float(np.sqrt(np.mean(np.sum((output-world)**2, axis=1))))*1000.,
                actual_changed_fraction=float(np.mean(np.any(output != world, axis=1))),
                seconds=time.perf_counter()-started)
    result = dict(support_mask=support, group_ids=groups_world.copy(),
                  coefficients=beta_new, seed_coefficients=beta_seed.copy(),
                  coefficient_delta=change, coefficient_ids=np.arange(len(beta_seed)),
                  active_original_indices=world_rows.copy(), fit_row_mask=fit_world,
                  low_probability_mask=low_world, subtracted_noise_mm=noise_world,
                  conditional_component_probability=mixture_world,
                  singular_values=singular.copy(),
                  conditional_event_probability=mass.copy())
    return output, info, result
