"""Observation-only anchored ambiguity veto, composed with frozen E0-E proposals."""
import numpy as np
from scipy.signal import find_peaks


def anchored_prominence(curves, grid, primary_width, radius_mm=1.0):
    c = np.asarray(curves, dtype=float)
    g = np.asarray(grid, dtype=float)
    widths = np.asarray(primary_width, dtype=float)
    if (c.ndim != 2 or c.shape[1] != len(g) or len(g) < 5
            or widths.shape != (len(c),) or not np.isfinite(c).all()
            or not np.isfinite(g).all() or not np.all(np.diff(g)>0)
            or not np.isfinite(widths).all() or np.any(widths <= 0)
            or not np.isfinite(radius_mm) or radius_mm <= 0):
        raise ValueError('finite curves, increasing grid, positive widths/radius required')
    result = np.zeros(len(c))
    locations = np.full(len(c), np.nan)
    for i, row in enumerate(c):
        primary = int(np.argmin(row))
        peaks, prop = find_peaks(-row, prominence=0.)
        valid = ((np.abs(g[peaks]-g[primary]) >= 2*widths[i])
                 & (np.abs(g[peaks]) <= radius_mm + 1e-12))
        if valid.any():
            ids = np.flatnonzero(valid)
            j = ids[np.argmax(prop['prominences'][ids])]
            result[i] = prop['prominences'][j]
            locations[i] = g[peaks[j]]
    return result, locations


def construct(curves, grid, features, diagnostics, existing):
    """No evaluation truth or mechanism label accepted or inferred from point IDs."""
    outputs = {k: existing[k].copy() for k in
               ('identity','test_projection','test_veto_projection')}
    masks = {}
    public = {}
    for radius in (1., 2.):
        name = f'anchor{int(radius)}'
        prominence, location = anchored_prominence(curves, grid, features.width_mm, radius)
        veto = (prominence > 0) & (prominence >= diagnostics['threshold'])
        # A local subset cannot veto a curve not vetoed by the identical global rule.
        assert not np.any(veto & ~diagnostics['veto'])
        mask = diagnostics['test'] & ~veto
        masks[name] = veto
        public[name+'_prominence'] = prominence
        public[name+'_location'] = location
        outputs[name+'_projection'] = np.where(mask, existing['test_projection'], 0.)
        if radius == 1.:
            outputs[name+'_argmin'] = np.where(mask, existing['argmin'], 0.)
    return outputs, masks, public

