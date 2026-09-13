"""A stronger optimizer for the SAME constant-prior parallel-plane mixture.

Input: design matrix, measured corrected heights, supplied sigma. No truth.
Optimization in standardized coordinates is a reparameterization, not new geometry.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit, logsumexp, log_expit
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exploration_v11'))
from spatial_filter import solve as legacy_solve, geometry


def objective(t, x, y):
    """Fixed-unit-noise negative log likelihood and exact gradient."""
    pred = t[:2][None, :] + (x @ t[2:4])[:, None]
    residual = pred - y[:, None]
    lp = -.5 * residual**2 + np.array([log_expit(-t[4]), log_expit(t[4])])
    norm = logsumexp(lp, axis=1)
    r = np.exp(lp - norm[:, None])
    gr = r * residual
    gradient = np.r_[gr.sum(0), x.T @ gr.sum(1), np.sum(expit(t[4]) - r[:, 1])]
    return float(-norm.sum()), gradient


def solve_constant(x, y, sigma, starts=64, maxiter=324):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != (len(y), 3) or sigma <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite N x 3 affine design and positive sigma required')
    xc = x[:, 1:].mean(0)
    xs = np.maximum(x[:, 1:].std(0), 1e-8)
    xx = (x[:, 1:] - xc) / xs
    ym = float(y.mean())
    yy = (y - ym) / sigma
    design = np.c_[np.ones(len(y)), xx]
    old = legacy_solve(x, y, sigma, spatial=False, force_k=2, iterations=324)
    initial = np.r_[(old['means'] + old['slope'] @ xc - ym)/sigma,
                    old['slope']*xs/sigma, old['gate'][0]]
    seeds = [initial]
    one = np.linalg.lstsq(design, yy, rcond=1e-12)[0]
    residual = yy - design @ one
    rng = np.random.default_rng(9150001)
    for j in range(starts - 1):
        if j < 9:
            prob = expit((residual - np.quantile(residual, (j+1)/10)) * (1 + j % 3))
        else:
            coef = rng.normal(size=3)
            feature = np.c_[residual, xx] @ coef
            feature = (feature - np.quantile(feature, rng.uniform(.1, .9))) / max(feature.std(), 1e-6)
            prob = expit(feature * rng.choice([1., 3., 8.]))
        r = np.c_[1-prob, prob]
        means, slope = geometry(design, yy, r, None)
        p = np.clip(prob.mean(), 1e-8, 1-1e-8)
        seeds.append(np.r_[means, slope, np.log(p/(1-p))])
    best, best_value = initial.copy(), objective(initial, xx, yy)[0]
    initial_value = best_value
    diagnostics = []
    for start in seeds:
        # Include starts themselves: a line-search failure may not discard a good incumbent.
        val = objective(start, xx, yy)[0]
        if val < best_value:
            best, best_value = start.copy(), val
        res = minimize(objective, start, args=(xx, yy), jac=True, method='L-BFGS-B',
                       bounds=[(None, None)]*4 + [(-30., 30.)],
                       options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8, maxls=40))
        value, grad = objective(res.x, xx, yy)
        diagnostics.append(dict(success=bool(res.success), nit=int(res.nit), nfev=int(res.nfev),
                                value=value, gradient_inf=float(np.max(abs(grad))), message=str(res.message)))
        if np.isfinite(value) and value < best_value:
            best, best_value = res.x.copy(), value
    means = best[:2]*sigma + ym - (best[2:4]*sigma/xs) @ xc
    slope = best[2:4]*sigma/xs
    pred = means[None, :] + (x[:, 1:] @ slope)[:, None]
    lp = -.5*((y[:, None]-pred)/sigma)**2 + np.array([log_expit(-best[4]), log_expit(best[4])])
    groups = np.argmax(lp, axis=1)
    score = 2*best_value + 5*np.log(len(y))
    one_world = np.linalg.lstsq(x, y, rcond=1e-12)[0]
    one_pred = x @ one_world
    single_score = float(np.sum(((y-one_pred)/sigma)**2) + 3*np.log(len(y)))
    k = 2 if score < single_score else 1
    return dict(k=k, means=means if k == 2 else one_world[:1],
                slope=slope if k == 2 else one_world[1:],
                prediction=pred[np.arange(len(y)), groups] if k == 2 else one_pred,
                groups=groups if k == 2 else np.zeros(len(y), int),
                score=float(min(score, single_score)), gate=np.array([best[4]]) if k == 2 else np.zeros(1),
                initial_two_plane_nll=initial_value, final_two_plane_nll=best_value,
                solver_diagnostics=diagnostics, starts=len(seeds), maxiter=maxiter)


def filter_frozen(state, sigma):
    model = solve_constant(state['design'], state['corrected'], sigma)
    take = np.flatnonzero(state['support'])
    rows = state['order'][take]
    out = state['world'].copy()
    out[rows] += ((model['prediction'][take]-state['local'][take, 2])/1000)[:, None]*state['normal']
    mask = np.zeros(len(out), bool)
    mask[rows] = True
    info = {k: v for k, v in model.items() if k not in ('prediction', 'groups')}
    info.update(truth_fields_used=[], parameter_count=5, geometry='two intercepts and shared two slopes',
                optimizer='64-start L-BFGS-B; legacy 324-iteration incumbent retained; not global certificate')
    return out, info, mask
