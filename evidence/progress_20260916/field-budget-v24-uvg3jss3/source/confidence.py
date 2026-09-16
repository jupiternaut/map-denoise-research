"""Descriptive alpha ranking against independently fixed probe actions.

Inputs must already share one frozen evaluation mask. Probe actions must not use
alpha to determine their own movement. This module reports association, not
calibrated probabilities, causal attribution, or independent-point inference.
"""
from collections.abc import Mapping
import numpy as np


def _vector(value, name):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite numeric vector') from exc
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a nonempty finite one-dimensional vector')
    return array


def _average_ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts - 1) / 2.)[inverse]


def _spearman(first, second):
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    x, y = _average_ranks(first), _average_ranks(second)
    x, y = x-x.mean(), y-y.mean()
    return float(np.clip(np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)), -1., 1.))


def _summary(alpha, gain, mask):
    a, g = alpha[mask], gain[mask]
    if not len(a):
        return None
    return dict(n=int(len(a)), mean_alpha=float(a.mean()),
                min_alpha=float(a.min()), max_alpha=float(a.max()),
                mean_signed_gain=float(g.mean()),
                beneficial_fraction=float(np.mean(g > 0)),
                harmful_fraction=float(np.mean(g < 0)),
                unchanged_fraction=float(np.mean(g == 0)))


def diagnose(alpha, baseline_errors, probe_errors):
    """Compare alpha ranks to ``baseline_errors - probe_errors[name]``.

    Quartile cuts use observed alpha values (linear sample quantiles), never row
    ranks: equal alpha values always enter the same bin. Duplicate boundaries and
    empty bins are omitted. For low/high quartile comparisons, values satisfying
    both tail definitions are excluded from both groups; a missing group yields
    null. Errors and gains retain the caller's units. No absolute-gain correlation,
    p-values, point-independence confidence intervals or calibration claims.
    """
    a = _vector(alpha, 'alpha')
    baseline = _vector(baseline_errors, 'baseline_errors')
    if a.shape != baseline.shape:
        raise ValueError('alpha and baseline_errors must have the same shape')
    if np.any((a < 0) | (a > 1)):
        raise ValueError('alpha must lie in [0, 1]')
    if not isinstance(probe_errors, Mapping) or not probe_errors:
        raise ValueError('probe_errors must be a nonempty mapping of names to vectors')
    probes = {}
    for name, errors in probe_errors.items():
        if not isinstance(name, str) or not name:
            raise ValueError('probe names must be nonempty strings')
        value = _vector(errors, f'probe_errors[{name}]')
        if value.shape != a.shape:
            raise ValueError(f'probe_errors[{name}] must match alpha shape')
        with np.errstate(over='ignore', invalid='ignore'):
            gain = baseline-value
        if not np.isfinite(gain).all():
            raise ValueError(f'probe_errors[{name}] produces nonfinite signed gains')
        probes[name] = gain
    quartiles = np.quantile(a, [.25, .5, .75])
    cuts = np.unique(quartiles)
    # Boundary ties go into the lower bin; no random or order-based tie splitting.
    bin_ids = np.searchsorted(cuts, a, side='left')
    low, high = a <= quartiles[0], a >= quartiles[2]
    overlapping = low & high
    low, high = low & ~overlapping, high & ~overlapping
    results = {}
    for name, gain in probes.items():
        bins = [dict(bin_id=int(index), **_summary(a, gain, bin_ids == index))
                for index in np.unique(bin_ids)]
        lower, upper = _summary(a, gain, low), _summary(a, gain, high)
        comparison = (dict(low=lower, high=upper,
                           high_minus_low_mean_signed_gain=float(
                               upper['mean_signed_gain']-lower['mean_signed_gain']))
                      if lower is not None and upper is not None else None)
        results[name] = dict(spearman_rank_correlation=_spearman(a, gain),
                             overall=_summary(a, gain, np.ones(len(a), dtype=bool)),
                             bins=bins, high_low_comparison=comparison)
    return dict(n=int(len(a)), alpha_quantiles=dict(q25=float(quartiles[0]),
                q50=float(quartiles[1]), q75=float(quartiles[2])),
                bin_boundaries=cuts.tolist(),
                tie_policy='Equal alpha values remain together; boundary ties go to the lower bin.',
                high_low_rule='alpha <= q25 versus alpha >= q75, removing overlapping ties from both groups',
                high_low_counts=dict(low=int(low.sum()), high=int(high.sum()),
                                     overlapping_ties_excluded=int(overlapping.sum())),
                high_low_missing_reason=('a tail is empty after excluding overlapping ties'
                                         if not low.any() or not high.any() else None),
                signed_gain_definition='baseline_error - fixed_probe_error; positive means improvement',
                scope='Descriptive ranking for fixed probe actions on a shared evaluation mask; '
                      'not calibration, not a causal effect, not independent-point inference.',
                probes=results)
