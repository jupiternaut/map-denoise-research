"""Additional evaluator-only diagnostics. Never imported by estimators."""
from __future__ import annotations
import numpy as np


def structure_metrics(output_m, evaluation):
    if not bool(evaluation.get('identifiable', True)):
        return {}
    out = np.asarray(output_m, float) * 1000
    ref = np.asarray(evaluation['gt_clean_xyz_world'], float) * 1000
    labels = np.asarray(evaluation['gt_layer'])
    delta = out-ref
    aligned = delta-delta.mean(axis=0)
    result = {
        'global_translation_only_aligned_rms_mm': float(np.sqrt(np.mean(np.sum(aligned**2, axis=1)))),
        'output_edit_from_reference_centroid_mm': float(np.linalg.norm(delta.mean(axis=0))),
    }
    fits = {}; tilts = []
    for label in np.unique(labels):
        p = out[labels == label]
        if len(p) < 6:
            continue
        A = np.c_[p[:, :2], np.ones(len(p))]
        if np.linalg.matrix_rank(A) < 3:
            continue
        coeff = np.linalg.lstsq(A, p[:, 2], rcond=None)[0]
        tilt = float(np.degrees(np.arctan(np.linalg.norm(coeff[:2]))))
        fits[int(label)] = coeff
        tilts.append((tilt, len(p)))
    if tilts:
        result['source_surface_tilt_mean_deg'] = float(np.average([a for a,n in tilts], weights=[n for a,n in tilts]))
        result['source_surface_tilt_max_deg'] = max(a for a,n in tilts)
    if 0 in fits and 1 in fits:
        gap_at_origin = float(fits[1][2]-fits[0][2])
        result['fitted_gap_at_same_xy_mm'] = gap_at_origin
        result['fitted_gap_at_same_xy_error_mm'] = abs(gap_at_origin-float(evaluation['true_gap_mm']))
        result['fitted_gap_note'] = 'GT source-group affine fits evaluated at x=y=0; may extrapolate, not physical layer count'
    return result


def subset_evaluation(ev, indices, original_n):
    return {k: (v[indices].copy() if k in ('gt_clean_xyz_world', 'gt_layer', 'gt_eps_mm')
                 and isinstance(v, np.ndarray) and len(v)==original_n else v)
            for k,v in ev.items()}


def sampling_indices(xyz_world, scan_id, variant):
    """Observed-coordinate selection only; no GT labels or surface geometry."""
    p = np.asarray(xyz_world)
    frame = np.asarray(scan_id)
    keep = np.zeros(len(p), dtype=bool)
    for j, sid in enumerate(np.unique(frame)):
        ind = np.flatnonzero(frame == sid)
        x = p[ind, 0]
        if variant == 'density':
            # Retain sparse-side points at deterministic 1-in-4, alternating sides.
            dense_side = x <= np.median(x) if j % 2 == 0 else x >= np.median(x)
            sparse = np.flatnonzero(~dense_side)
            dense_side[sparse[::4]] = True
            keep[ind[dense_side]] = True
        elif variant == 'partial_overlap':
            mask = x <= np.quantile(x, .70) if j % 2 == 0 else x >= np.quantile(x, .30)
            keep[ind[mask]] = True
        else:
            raise ValueError(variant)
    return np.flatnonzero(keep)
