"""Read-only V15 estimators plus explicitly known-slope diagnostic adapters.

No generator or evaluator is imported here. Normal API takes only X, h, sigma.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit, logsumexp, log_expit
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'exploration_v15'), str(ROOT/'exploration_v11')]
from constant_solver import solve_constant
from spatial_filter import solve as original_spatial, features, logistic, RIDGE

METHODS = ('constant_free', 'spatial_free', 'constant_oracle_slope', 'spatial_oracle_slope')


def fixed_geometry(y, r):
    return (r.T @ y) / np.maximum(r.sum(0), np.finfo(float).tiny)


def fixed_em(x, y, sigma, spatial, iterations=36, force_k=None):
    """The legacy EM schedule with fixed slope removed from y; fit intercepts."""
    n = len(y)
    z = features(x[:, 1:], spatial)
    pdim = z.shape[1]
    single = dict(k=1, means=np.array([y.mean()]), slope=np.zeros(2),
                  prediction=np.full(n, y.mean()), groups=np.zeros(n, int), gate=np.zeros(pdim),
                  score=float(np.sum(((y-y.mean())/sigma)**2) + np.log(n)))
    residual = y - y.mean()
    starts = [expit((residual-np.quantile(residual, q))/sigma) for q in (.35, .65)]
    for axis in (0, 1):
        coord = x[:, axis+1]
        coord = (coord-np.median(coord))/max(np.std(coord), 1e-6)
        starts.extend([expit(3*coord), expit(-3*coord)])
    candidates = []
    for probability in starts:
        r = np.c_[1-probability, probability]
        gate = logistic(z, probability, np.zeros(pdim))
        levels = fixed_geometry(y, r)
        for _ in range(iterations):
            p = np.clip(expit(z@gate), 1e-9, 1-1e-9)
            lp = -.5*((y[:, None]-levels)/sigma)**2 + np.log(np.c_[1-p, p])
            r = np.exp(lp-logsumexp(lp, axis=1)[:, None])
            levels = fixed_geometry(y, r)
            gate = logistic(z, r[:, 1], gate)
        p = np.clip(expit(z@gate), 1e-9, 1-1e-9)
        lp = -.5*((y[:, None]-levels)/sigma)**2 + np.log(np.c_[1-p, p])
        g = np.argmax(lp, axis=1)
        score = float(-2*logsumexp(lp, axis=1).sum() + RIDGE*np.sum(gate[1:]**2)
                      + (2+pdim)*np.log(n))
        candidates.append(dict(k=2, means=levels, slope=np.zeros(2), prediction=levels[g],
                               groups=g, gate=gate, score=score))
    best = min(candidates, key=lambda m: m['score'])
    return best if force_k == 2 or best['score'] < single['score'] else single


def fixed_objective(t, y):
    residual = t[:2][None, :] - y[:, None]
    lp = -.5*residual**2 + np.array([log_expit(-t[2]), log_expit(t[2])])
    norm = logsumexp(lp, axis=1)
    r = np.exp(lp-norm[:, None])
    grad = np.r_[(r*residual).sum(0), np.sum(expit(t[2])-r[:, 1])]
    return float(-norm.sum()), grad


def fixed_constant(x, y, sigma, starts=64, maxiter=324):
    ym = float(y.mean())
    yy = (y-ym)/sigma
    xx = (x[:, 1:]-x[:, 1:].mean(0))/np.maximum(x[:, 1:].std(0), 1e-8)
    old = fixed_em(x, y, sigma, False, iterations=324, force_k=2)
    initial = np.r_[(old['means']-ym)/sigma, old['gate'][0]]
    seeds = [initial]
    rng = np.random.default_rng(9150001)
    for j in range(starts-1):
        if j < 9:
            prob = expit((yy-np.quantile(yy, (j+1)/10))*(1+j % 3))
        else:
            feature = np.c_[yy, xx] @ rng.normal(size=3)
            feature = (feature-np.quantile(feature, rng.uniform(.1, .9)))/max(feature.std(), 1e-6)
            prob = expit(feature*rng.choice([1., 3., 8.]))
        means = fixed_geometry(yy, np.c_[1-prob, prob])
        p = np.clip(prob.mean(), 1e-8, 1-1e-8)
        seeds.append(np.r_[means, np.log(p/(1-p))])
    best = initial.copy()
    initial_value = best_value = fixed_objective(initial, yy)[0]
    diagnostics = []
    for start in seeds:
        value = fixed_objective(start, yy)[0]
        if value < best_value:
            best, best_value = start.copy(), value
        opt = minimize(fixed_objective, start, args=(yy,), jac=True, method='L-BFGS-B',
                       bounds=[(None, None), (None, None), (-30., 30.)],
                       options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8, maxls=40))
        value, grad = fixed_objective(opt.x, yy)
        diagnostics.append(dict(success=bool(opt.success), nit=int(opt.nit), nfev=int(opt.nfev),
                                value=value, gradient_inf=float(np.max(abs(grad))), message=str(opt.message)))
        if np.isfinite(value) and value < best_value:
            best, best_value = opt.x.copy(), value
    means = best[:2]*sigma+ym
    lp = -.5*((y[:, None]-means)/sigma)**2 + np.array([log_expit(-best[2]), log_expit(best[2])])
    g = np.argmax(lp, axis=1)
    score = 2*best_value+3*np.log(len(y))
    single_score = float(np.sum(((y-ym)/sigma)**2)+np.log(len(y)))
    k = 2 if score < single_score else 1
    return dict(k=k, means=means if k == 2 else np.array([ym]), slope=np.zeros(2),
                prediction=means[g] if k == 2 else np.full(len(y), ym),
                groups=g if k == 2 else np.zeros(len(y), int),
                score=float(min(score, single_score)), gate=np.array([best[2]]) if k == 2 else np.zeros(1),
                initial_two_plane_nll=initial_value, final_two_plane_nll=best_value,
                solver_diagnostics=diagnostics, starts=len(seeds), maxiter=maxiter)


def attach_posterior(model, x, y, sigma, spatial):
    """Canonical component order from fitted intercepts only; never pointwise GT."""
    m = dict(model)
    k = m['k']
    if k == 1:
        m['posterior'] = np.ones((len(y), 1))
        return m
    pred = m['means'][None, :] + (x[:, 1:]@m['slope'])[:, None]
    if spatial:
        p = np.clip(expit(features(x[:, 1:])@m['gate']), 1e-9, 1-1e-9)
        logprior = np.log(np.c_[1-p, p])
    else:
        logprior = np.array([log_expit(-m['gate'][0]), log_expit(m['gate'][0])])
    lp = -.5*((y[:, None]-pred)/sigma)**2+logprior
    posterior = np.exp(lp-logsumexp(lp, axis=1)[:, None])
    order = np.argsort(m['means'], kind='stable')
    inverse = np.argsort(order)
    m['means'] = m['means'][order]
    m['groups'] = inverse[m['groups']]
    m['posterior'] = posterior[:, order]
    if order[0] == 1:
        m['gate'] = -m['gate']
    np.testing.assert_allclose(m['prediction'], m['means'][m['groups']]+x[:, 1:]@m['slope'], rtol=0, atol=1e-10)
    return m


def fit(x, y, sigma, method, *, oracle_slope=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (method not in METHODS or x.shape != (len(y), 3) or len(y) < 8 or sigma <= 0
            or not np.isfinite(sigma) or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('valid method, finite affine design/height and positive sigma required')
    spatial = method.startswith('spatial_')
    oracle = method.endswith('oracle_slope')
    if oracle != (oracle_slope is not None):
        raise ValueError('Only explicitly labelled oracle methods may receive a fixed slope')
    if oracle:
        slope = np.asarray(oracle_slope, float)
        if slope.shape != (2,) or not np.isfinite(slope).all():
            raise ValueError('finite two-dimensional known slope required')
        adjusted = y-x[:, 1:]@slope
        m = fixed_em(x, adjusted, sigma, True) if spatial else fixed_constant(x, adjusted, sigma)
        m['slope'] = slope.copy()
        m['prediction'] = m['prediction']+x[:, 1:]@slope
    else:
        m = original_spatial(x, y, sigma, spatial=True, iterations=36) if spatial else solve_constant(x, y, sigma)
    m = attach_posterior(m, x, y, sigma, spatial)
    m['method'] = method
    m['oracle_fields'] = ['shared_slope'] if oracle else []
    return m
