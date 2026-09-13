"""Cross-fitted spatial priors, full-data mixture geometry, deterministic MAP.

The V18 family is frozen. Only each row's gate training excludes that row;
family selection and final geometry estimation are not cross-fitted.
"""
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exploration_v17"))
import v17_operator as base

FOLDS = 3
GATE_STARTS = 12
MAX_ITERATIONS = 324
PRIOR_CLIP = 1e-8
METHODS = ("crossfit_gate_map", "insample_gate_map")


def validate(x, y, sigma):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or x.shape != (len(y), 3) or y.ndim != 1 or len(y) < 24:
        raise ValueError("N>=24, N x 3 affine design, N-vector heights")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("all observations must be finite")
    if not np.isfinite(sigma) or sigma <= 0 or not np.all(x[:, 0] == 1):
        raise ValueError("positive supplied sigma and affine intercept exactly 1")
    return x, y


def train_gate(x, y, sigma, family):
    """Joint Gaussian-mixture fit from these training observations only."""
    xc = x[:, 1:].mean(0)
    xs = np.maximum(x[:, 1:].std(0), 1e-8)
    ym = float(y.mean())
    center = np.median(x[:, 1:], axis=0)
    scale = np.maximum(np.ptp(x[:, 1:], axis=0) / 2, 1e-6)
    seeds = base.training_seeds(x, y, sigma, xc, xs, ym)[:GATE_STARTS]
    model = base.solve_family(x, y, sigma, family, seeds, xc, xs, ym, center, scale)
    return model


def prior_at(model, x):
    features = base.feature_matrix(x[:, 1:], model["family"],
                                   model["feature_center"], model["feature_scale"])
    p = np.clip(expit(features @ model["gate"]), PRIOR_CLIP, 1 - PRIOR_CLIP)
    return np.c_[1 - p, p]


def gate_summary(model, train, valid):
    records = model["solver_diagnostics"]
    return dict(train_indices=np.flatnonzero(train), validation_indices=np.flatnonzero(valid),
                family=model["family"], means=model["means"], slope=model["slope"],
                gate=model["gate"], feature_center=model["feature_center"],
                feature_scale=model["feature_scale"], nll=model["nll"], starts=model["starts"],
                nonconverged_starts=sum(not d["success"] for d in records),
                best_gradient_inf=min(d["gradient_inf"] for d in records))


def estimate_priors(x, y, sigma, family, crossfit):
    """No full-input geometry or posterior is passed to gate training."""
    folds = base.folds_from_x(x, FOLDS) if crossfit else np.zeros(len(y), int)
    prior = np.empty((len(y), 2))
    models, records = [], []
    for fold in range(FOLDS if crossfit else 1):
        valid = folds == fold
        train = ~valid if crossfit else np.ones(len(y), bool)
        model = train_gate(x[train], y[train], sigma, family)
        prior[valid] = prior_at(model, x[valid])
        models.append(model)
        records.append(gate_summary(model, train, valid))
    return prior, models, dict(folds=folds, training=records)


def fixed_prior_objective(t, xx, yy, log_prior):
    """Exact homoscedastic mixture NLL, with nonnegative gap coordinate.

    t = [midpoint, gap, slope_x, slope_y] in noise-standardized units.
    Constants shared by all candidate fits are omitted.
    """
    means = t[0] + np.array([-.5, .5]) * t[1]
    residual = means[None, :] + (xx @ t[2:])[:, None] - yy[:, None]
    lp = -.5 * residual ** 2 + log_prior
    ld = logsumexp(lp, axis=1)
    responsibility = np.exp(lp - ld[:, None])
    gr = responsibility * residual
    gradient = np.r_[gr.sum(), .5 * (gr[:, 1].sum() - gr[:, 0].sum()),
                     xx.T @ gr.sum(axis=1)]
    return float(-ld.sum()), gradient


def fit_geometry(x, y, sigma, prior, initial_model, gate_models):
    """Fit all heights with fixed prior. No hard trimming or subset refit."""
    xc = x[:, 1:].mean(0)
    xs = np.maximum(x[:, 1:].std(0), 1e-8)
    ym = float(y.mean())
    xx = (x[:, 1:] - xc) / xs
    yy = (y - ym) / sigma
    log_prior = np.log(prior)

    def convert(means, slope):
        m = (np.asarray(means) + np.asarray(slope) @ xc - ym) / sigma
        return np.r_[m.mean(), max(float(m[1] - m[0]), 0.), np.asarray(slope) * xs / sigma]

    starts = [convert(initial_model["means"], initial_model["slope"])]
    means = np.median(np.array([m["means"] for m in gate_models]), axis=0)
    slope = np.median(np.array([m["slope"] for m in gate_models]), axis=0)
    starts.append(convert(means, slope))
    plane = np.linalg.lstsq(np.c_[np.ones(len(y)), xx], yy, rcond=1e-12)[0]
    starts.extend(np.r_[plane[0], gap, plane[1:]] for gap in (.5, 1., 2., 4.))
    best, best_value = None, np.inf
    diagnostics = []
    for start in starts:
        value, _ = fixed_prior_objective(start, xx, yy, log_prior)
        if value < best_value:
            best, best_value = start.copy(), value
        opt = minimize(fixed_prior_objective, start, args=(xx, yy, log_prior), jac=True,
                       method="L-BFGS-B", bounds=[(None, None), (0., None),
                                                   (None, None), (None, None)],
                       options=dict(maxiter=MAX_ITERATIONS, ftol=1e-12, gtol=1e-8, maxls=40))
        value, grad = fixed_prior_objective(opt.x, xx, yy, log_prior)
        diagnostics.append(dict(success=bool(opt.success), iterations=int(opt.nit),
                                objective=value, gradient_inf=float(np.max(abs(grad))),
                                message=str(opt.message)))
        if np.isfinite(value) and value < best_value:
            best, best_value = opt.x.copy(), value
    slope = best[2:] * sigma / xs
    means = (best[0] + np.array([-.5, .5]) * best[1]) * sigma + ym - slope @ xc
    levels = means[None, :] + (x[:, 1:] @ slope)[:, None]
    lp = -.5 * ((y[:, None] - levels) / sigma) ** 2 + log_prior
    ld = logsumexp(lp, axis=1)
    posterior = np.exp(lp - ld[:, None])
    groups = np.argmax(posterior, axis=1)
    return dict(k=2, means=means, slope=slope, posterior=posterior,
                prediction=levels[np.arange(len(y)), groups], groups=groups,
                gate_prior=prior, log_density=ld, nll=float(-ld.sum()),
                geometry_solver_diagnostics=diagnostics,
                geometry_starts=len(starts), sigma_supplied_mm=float(sigma))


def run(x, y, sigma, baseline_model):
    """Return one candidate and one in-sample mechanism control.

    Parameters are observed affine XY design, measured normal heights in mm,
    supplied measurement sigma in mm, and the V18-selected model. No GT input.
    Output is row-preserving deterministic normal-coordinate projection.
    """
    x, y = validate(x, y, sigma)
    family = baseline_model.get("family", "S" if baseline_model["k"] == 1 else None)
    if family not in base.FAMILIES:
        raise ValueError("baseline must carry frozen V18 S/C/L/R family")
    output = {}
    for name, crossfit in zip(METHODS, (True, False)):
        started = time.perf_counter()
        if int(baseline_model["k"]) == 1:
            model = dict(baseline_model)
            model["groups"] = np.zeros(len(y), int)
            model["posterior"] = np.ones((len(y), 1))
            diagnostic = dict(single_surface_passthrough=True)
        else:
            prior, gates, diagnostic = estimate_priors(x, y, sigma, family, crossfit)
            model = fit_geometry(x, y, sigma, prior, baseline_model, gates)
        model.update(family=family, method=name,
                     metadata=dict(truth_fields_used=[], gate_training_crossfit=crossfit,
                                   whole_pipeline_crossfit=False, family_source="V18 frozen selection",
                                   gate_diagnostics=diagnostic, seconds=time.perf_counter() - started,
                                   input_fields=["measured_design", "measured_height", "supplied_sigma",
                                                 "baseline_selected_family", "baseline_geometry_initialization"]))
        output[name] = model
    return output
