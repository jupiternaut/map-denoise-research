"""Observation-only selective revision. No evaluation labels are accepted."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter


@dataclass(frozen=True)
class CurveFeatures:
    contrast: np.ndarray
    width_mm: np.ndarray
    secondary_prominence: np.ndarray
    margin: np.ndarray
    primary_boundary: np.ndarray


def curve_features(curves, grid):
    c = np.asarray(curves, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if c.ndim != 2 or c.shape[1] != len(grid) or not np.isfinite(c).all():
        raise ValueError('expected finite curves on a shared grid')
    if len(grid) < 5 or not np.all(np.diff(grid) > 0):
        raise ValueError('grid must be increasing with at least five samples')
    contrast = np.ptp(c, axis=1)
    width = np.empty(len(c))
    prom = np.zeros(len(c))
    margin = np.full(len(c), np.inf)
    boundary = np.zeros(len(c), dtype=bool)
    step = float(np.min(np.diff(grid)))
    for i, row in enumerate(c):
        primary = int(np.argmin(row))
        peaks, properties = find_peaks(-row, prominence=0.0)
        j = np.flatnonzero(peaks == primary)
        if len(j):
            w_index = peak_widths(-row, [primary], rel_height=.5)[0][0]
            # The frozen E0 grid is uniform. Validate it rather than silently assume.
            if not np.allclose(np.diff(grid), step, rtol=1e-9, atol=1e-12):
                raise ValueError('width estimator requires a uniform grid')
            width[i] = max(2 * step, w_index * step / (2 * np.sqrt(2 * np.log(2))))
        else:
            boundary[i] = True
            low = grid[row <= np.quantile(row, .1)]
            width[i] = max(2 * step, np.ptp(low) / (2 * np.sqrt(2 * np.log(2))))
        far = np.abs(grid[peaks] - grid[primary]) >= 2 * width[i]
        secondary = peaks[far]
        if len(secondary):
            prom[i] = float(properties['prominences'][far].max())
            margin[i] = float((row[secondary].min() - row[primary]) / max(contrast[i], 1e-12))
    return CurveFeatures(contrast, width, prom, margin, boundary)


def calibration_roughness(curves, strata):
    """Observed high-frequency residual scale; not an iid-noise estimator."""
    c = np.asarray(curves, dtype=float)
    residual = c - savgol_filter(c, window_length=41, polyorder=3, axis=1)
    centered = residual - np.median(residual, axis=1, keepdims=True)
    scales = np.median(np.abs(centered), axis=1) / .6744897501960817
    return np.array([np.median(scales[strata == k]) if np.sum(strata == k) >= 50
                     else np.median(scales) for k in range(3)])


def veto_mask(features, sigma_hf, rho=.25):
    threshold = np.maximum(rho * features.contrast, 3 * np.asarray(sigma_hf))
    # Positive prominence prevents an exactly flat zero-threshold curve being vetoed.
    veto = (features.secondary_prominence > 0) & (features.secondary_prominence >= threshold)
    return veto, threshold


def choose_outputs(old, project, curves, strata, calibration, sigma_by_stratum, features=None):
    """Only public evidence and explicitly labelled calibration enter this API."""
    s, m, argmin, _ = old.scores(curves)
    dec = old.decide(s, m, strata, calibration)
    eligible = (m <= dec['qabs_i']) & (dec['n_comp'] == 1) & ~dec['L'][:, old.IDX0]
    test = eligible & (dec['p'] <= .1)
    bh = dec['out'] == 'MOVE'
    proposal = project(old, s, dec, eligible)
    feat = features if features is not None else curve_features(curves, old.TGRID)
    veto, threshold = veto_mask(feat, np.asarray(sigma_by_stratum)[strata])
    margin_veto = feat.margin <= .25
    outputs = {
        'identity': np.zeros(len(curves)),
        'argmin': argmin.copy(),
        'gated1': np.where(np.abs(argmin) > 1., argmin, 0.),
        'original_bh_projection': np.where(bh, proposal, 0.),
        'no_bh_projection': proposal,
        'test_projection': np.where(test, proposal, 0.),
        'test_veto_projection': np.where(test & ~veto, proposal, 0.),
        'test_veto_argmin': np.where(test & ~veto, argmin, 0.),
        'test_margin_projection': np.where(test & ~margin_veto, proposal, 0.),
    }
    diagnostics = dict(p=dec['p'], eligible=eligible, test=test, bh=bh,
                       veto=veto, margin_veto=margin_veto, threshold=threshold,
                       absolute_bad=m > dec['qabs_i'], n_comp=dec['n_comp'],
                       q_tgt=dec['qtgt_i'], sigma_hf=np.asarray(sigma_by_stratum)[strata],
                       width_mm=feat.width_mm, secondary_prominence=feat.secondary_prominence,
                       normalized_margin=feat.margin, primary_boundary=feat.primary_boundary,
                       contrast=feat.contrast,
                       research_acquire=(dec['n_comp'] >= 2) | veto)
    return outputs, diagnostics
