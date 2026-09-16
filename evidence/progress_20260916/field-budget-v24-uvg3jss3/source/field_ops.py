"""Input-only displacement-budget experiments; no reference geometry accepted."""
import numpy as np

FIELDS = ('local_plane64', 'quadratic64', 'multiscale_full')
BUDGETS = (0.025, 0.05, 0.1)
SHUFFLE_SEEDS = tuple(range(92401, 92406))
EPS_RMS = 1e-12  # millimetres, used to reject degenerate fields, not divide by them


def rms(field):
    a = np.asarray(field, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
        raise ValueError('field must be finite nonempty N x 3')
    return float(np.sqrt(np.mean(np.sum(a*a, axis=1))))


def to_budget(field, budget):
    native = rms(field)
    if not np.isfinite(budget) or budget < 0:
        raise ValueError('budget must be finite and nonnegative')
    if budget == 0:
        return np.zeros_like(field, dtype=float), 0.
    if native <= EPS_RMS:
        return None, None
    factor = float(budget/native)
    return np.asarray(field, dtype=float)*factor, factor


def construct(points, cached_outputs, alpha):
    points = np.asarray(points, dtype=float)
    rms(points)
    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (len(points),) or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError('alpha must be finite N-vector in [0,1]')
    fields = {}
    for name in FIELDS:
        output = np.asarray(cached_outputs[name], dtype=float)
        if output.shape != points.shape:
            raise ValueError('cached output shape mismatch')
        fields[name] = output-points
    native_rms = {name: rms(d) for name, d in fields.items()}
    cap = min(native_rms.values())
    if cap <= EPS_RMS:
        cap = 0.
    records = []
    seen_budgets = set()
    for requested in BUDGETS:
        budget = min(requested, cap)
        repeated = budget in seen_budgets
        seen_budgets.add(budget)
        weighted = [('uniform', None, fields['multiscale_full']),
                    ('alpha', None, alpha[:, None]*fields['multiscale_full'])]
        for seed in SHUFFLE_SEEDS:
            shuffled = np.random.default_rng(seed).permutation(alpha)
            weighted.append(('shuffle', seed, shuffled[:, None]*fields['multiscale_full']))
        specifications = [(name, 'uniform', None, fields[name]) for name in FIELDS[:2]]
        specifications += [('multiscale_full', allocation, seed, d) for allocation, seed, d in weighted]
        for name, allocation, seed, d in specifications:
            move, factor = to_budget(d, budget)
            label = f'budget{requested:g}__{name}__{allocation}' + (f'_{seed}' if seed is not None else '')
            records.append(dict(method=label, kind='budget', field=name, allocation=allocation,
                                seed=seed, requested_budget_mm=requested, actual_budget_mm=budget,
                                duplicate_budget=repeated, direction_informative=budget > 0,
                                native_field_rms_mm=rms(d), normalization_factor=factor,
                                status='OK' if move is not None else 'INFEASIBLE',
                                output=points+move if move is not None else None))
    for t in (0.5, 1.):
        records.append(dict(method=f'probe_t{t:g}', kind='probe', field='multiscale_full',
                            allocation='fixed_step', step=t, status='OK',
                            output=points+t*fields['multiscale_full']))
    return records, dict(native_field_rms_mm=native_rms, shared_cap_mm=cap)
