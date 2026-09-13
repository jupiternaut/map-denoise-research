"""Fixed-group decomposition of V8's model-based selection correction.

The group-weighted constant and zero-group-mean (spatial) components are
attribution actions, not independent estimates of physical noise. All upstream
state, hard labels, fitting rows, weights, and output support are unchanged.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_v9_action_v8", PROJECT / "exploration_v8/compensation.py")
V8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V8)
MODES = ("none", "constant", "slope", "full")


def action_frozen(state, artifacts, sigma_mm, mode="constant"):
    """Return (world XYZ in metres, info, artifacts) from legal frozen inputs.

    constant: subtract each hard group's weighted mean V8 correction, leaving
      that group's original fitted slope coefficients exactly unchanged.
    slope: subtract the remaining point-varying, weighted-zero-mean correction.
      This preserves the group's weighted fitted mean, not necessarily the
      affine intercept at a fixed global origin.
    full: V8's complete conditional-mean correction and hard WLS refit.
    none: original hard WLS replay.

    All modes use fixed coefficients/candidates for the correction model.
    Rank-deficient groups retain least-squares predictions; constant applies an
    explicit intercept shift to preserve the chosen base slope representation.
    """
    started = time.perf_counter()
    if mode not in MODES:
        raise ValueError("unknown action mode")
    sigma = float(sigma_mm)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive finite sigma_mm required")
    active = np.asarray(state["active"], int)
    order = np.asarray(state["order"], int)
    world = np.asarray(state["world"], float)
    world_rows = order[active]
    support = np.zeros(len(world), bool)
    support[order] = state["support"]
    world_groups = np.asarray(artifacts["group_ids"], int).copy()
    groups = world_groups[world_rows]
    result = dict(support_mask=support, group_ids=world_groups,
                  active_original_indices=world_rows.copy(),
                  subtracted_noise_mm=np.zeros(len(world)),
                  full_selection_noise_mm=np.zeros(len(world)),
                  group_constant_noise_mm=np.zeros(len(world)))
    if not len(active):
        result.update(coefficients=np.empty((0, 3)), coefficient_ids=np.empty(0, int),
                      compensated_coefficients=np.empty((0, 3)),
                      compensated_coefficient_ids=np.empty(0, int))
        return world.copy(), dict(mode=mode, status="UNSUPPORTED", truth_fields_used=[],
                                  seconds=time.perf_counter()-started), result
    weights = np.asarray(state["weights"])[active]
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("finite nonnegative weights required")
    # This uses fitted candidate surfaces and sigma, never true layer labels.
    full, details = V8.selection_bias(state["design"][active], artifacts["coefficients"],
        np.asarray(artifacts["candidate_mask"])[world_rows], groups, sigma)
    constant = np.empty_like(full)
    ids = np.unique(groups)
    constants = []
    for group in ids:
        rows = groups == group
        mass = float(weights[rows].sum())
        if mass <= 0:
            raise ValueError("every fitted group requires positive weight mass")
        value = float(np.dot(weights[rows], full[rows])/mass)
        constant[rows] = value
        constants.append(value)
    correction = {"none": np.zeros_like(full), "constant": constant,
                  "slope": full-constant, "full": full}[mode]
    if mode == "constant":
        output, fitted, fit_ids, ranks = V8._fit_and_project(
            state, groups, state["corrected"][active])
        fitted[:, 0] -= np.asarray(constants)
        use = np.asarray(state["support"])[active]
        output[world_rows[use]] -= constant[use, None]*state["normal"]/1000.
    else:
        output, fitted, fit_ids, ranks = V8._fit_and_project(
            state, groups, state["corrected"][active]-correction)
    if not np.isfinite(output).all():
        raise FloatingPointError("nonfinite action output")
    np.testing.assert_array_equal(output[~support], world[~support])
    result["subtracted_noise_mm"][world_rows] = correction
    result["full_selection_noise_mm"][world_rows] = full
    result["group_constant_noise_mm"][world_rows] = constant
    result.update(coefficients=fitted, coefficient_ids=fit_ids,
                  compensated_coefficients=fitted.copy(), compensated_coefficient_ids=fit_ids.copy(),
                  group_constant_ids=ids, group_constant_values_mm=np.asarray(constants))
    result.update(details)
    info = dict(method="fixed_group_selection_action_v9", mode=mode, status="APPLY",
        sigma_mm=sigma, point_count=len(world), active_count=len(active), group_count=len(ids),
        fit_parameter_count=3*len(ids), fit_rank=sum(ranks), group_fit_ranks=ranks,
        supported_fraction=float(support.mean()), truth_fields_used=[],
        mean_absolute_subtracted_noise_mm=float(np.mean(abs(correction))),
        max_absolute_subtracted_noise_mm=float(np.max(abs(correction))),
        low_probability_fallbacks=int(np.sum(details["underflow_fallback"])),
        actual_changed_fraction=float(np.mean(np.any(output != world, axis=1))),
        assumption="same fitted Gaussian mixture and supplied sigma as V8",
        decomposition="per-hard-group weighted mean plus weighted-zero-mean spatial variation",
        slope_mode_scope="preserves weighted group mean, not intercept at global tangent origin",
        timing_scope="cached action including compensation calculation, excludes upstream search",
        seconds=time.perf_counter()-started)
    return output, info, result
