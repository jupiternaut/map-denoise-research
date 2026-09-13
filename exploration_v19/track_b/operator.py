"""Fixed-fit posterior surface representations and mass-constrained exports.

Only measured design/heights, supplied sigma, and a frozen fitted model enter
this module. No evaluator or true labels are imported. A weighted measure is
NOT an N-point prediction: one observation's K atoms share one unit of mass.
"""
import numpy as np

METHODS = ("map", "mean", "weighted_measure", "quota_map", "posterior_draw")
DRAW_SEED = 9193101


def _validate(x, y, sigma, model):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != (len(y), 3) or not len(y) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("finite, nonempty N x 3 design and N heights required")
    if not np.all(x[:, 0] == 1) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("affine design with intercept one and positive sigma required")
    k = int(model["k"])
    if k not in (1, 2):
        raise ValueError("this export implements one or two parallel surfaces")
    means = np.asarray(model["means"], float)
    slope = np.asarray(model["slope"], float)
    posterior = np.asarray(model["posterior"], float)
    if means.shape != (k,) or slope.shape != (2,) or posterior.shape != (len(y), k):
        raise ValueError("model means, slope, or posterior shape mismatch")
    if not np.isfinite(means).all() or not np.isfinite(slope).all() or not np.isfinite(posterior).all():
        raise ValueError("nonfinite frozen model")
    if np.any(posterior < 0) or np.any(posterior > 1) or not np.allclose(posterior.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("posterior must already contain normalized probabilities")
    if k == 2 and means[1] < means[0]:
        raise ValueError("ordered lower/upper fitted surfaces required")
    return x, posterior, means[None, :] + (x[:, 1:] @ slope)[:, None]


def quota_assignment(upper_posterior):
    """Highest posterior first, subject to rounded expected upper-layer count.

    Among assignments with this fixed count, this minimizes the sum of posterior
    expected 0-1 label errors (and maximizes the factorized posterior likelihood).
    It does not make the supplied posterior true or infer an external layer mass.
    Exact posterior ties use source row order; permutation equivariance is not
    claimed for the tied subset. Half counts round UP, never bankers' rounding.
    """
    upper = np.asarray(upper_posterior, float)
    if upper.ndim != 1 or not np.isfinite(upper).all() or np.any((upper < 0) | (upper > 1)):
        raise ValueError("finite upper-layer probabilities in [0, 1] required")
    expected = float(np.sum(upper, dtype=np.float64))
    target = int(np.floor(expected + .5))
    order = np.argsort(-upper, kind="stable")
    groups = np.zeros(len(upper), dtype=np.int64)
    groups[order[:target]] = 1
    return groups, target, expected


def run(x, y, sigma, baseline_model):
    """Return five outputs of the SAME supplied fit, without changing its fit.

    Weighted support_weights sum to one per observation; aggregate empirical
    measure weights are support_weights / N. They must not be counted as K*N
    independent observations or as an unweighted point-cloud expansion.
    """
    x, posterior, levels = _validate(x, y, sigma, baseline_model)
    n, k = posterior.shape
    map_groups = np.argmax(posterior, axis=1)
    mean = np.sum(posterior * levels, axis=1)
    if k == 2:
        quota_groups, target, expected = quota_assignment(posterior[:, 1])
        draw_groups = (np.random.default_rng(DRAW_SEED).random(n) < posterior[:, 1]).astype(np.int64)
    else:
        quota_groups = draw_groups = np.zeros(n, dtype=np.int64)
        target = 0
        expected = 0.
    common = dict(baseline_model, truth_fields_used=[], fit_changed=False,
                  posterior=np.array(posterior, copy=True),
                  fitted_layer_mass=posterior.mean(0), source_point_count=n,
                  competing_model_scores_interpretation="scores retained if supplied; not converted into probabilities")
    outputs = {}
    for method, groups in (("map", map_groups), ("quota_map", quota_groups), ("posterior_draw", draw_groups)):
        prediction = levels[np.arange(n), groups]
        outputs[method] = dict(common, representation="points", export_policy=method,
                               prediction=prediction, groups=groups,
                               output_distance_to_fitted_surface_mm=np.min(abs(prediction[:, None] - levels), axis=1))
    outputs["mean"] = dict(common, representation="points", export_policy="mean", groups=None,
                           prediction=mean,
                           output_distance_to_fitted_surface_mm=np.min(abs(mean[:, None] - levels), axis=1))
    outputs["weighted_measure"] = dict(common, representation="weighted", export_policy="weighted_measure",
                                       prediction=None, groups=None, support_heights=levels,
                                       support_weights=posterior.copy(), total_measure_mass=1.,
                                       aggregate_weight_normalizer=n)
    outputs["quota_map"].update(quota_upper_count=target, expected_upper_count=expected,
                                 quota_rounding="floor(expected_count + 0.5)",
                                 quota_tie_rule="descending posterior, stable original row order")
    outputs["posterior_draw"]["draw_seed"] = DRAW_SEED
    return {name: outputs[name] for name in METHODS}
