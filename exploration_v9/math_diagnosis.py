"""Known-model diagnostic only; not a deployable oracle or scene benchmark.

Constructs its own symmetric two-plane population. No project truth, candidate
state, or experiment output is loaded. Checks the selection-compensation target
with analytic moments, independent numerical integration, and Monte Carlo.
"""
from __future__ import annotations

import json
from math import exp, pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def known_model(a: float, sigma: float = 1.) -> dict:
    """S in {-a,+a}, equiprobable; Y=S+epsilon; group=sign(Y)."""
    phi = lambda z: exp(-.5*z*z)/sqrt(2.*pi)
    kappa = float(2.*ndtr(a/sigma)-1.)
    noise_mean = 2.*sigma*phi(a/sigma)
    hard_center = a*kappa + noise_mean
    density = lambda y: (phi((y-a)/sigma)+phi((y+a)/sigma))/(2.*sigma)
    # The mixed signed first noise moment in the positive observation cell.
    noise_density = lambda y: ((y-a)*phi((y-a)/sigma)
                              + (y+a)*phi((y+a)/sigma))/(2.*sigma)
    upper = a+12.*sigma
    probability = quad(density, 0., upper, epsabs=1e-12)[0]
    moment = quad(lambda y:y*density(y), 0., upper, epsabs=1e-12)[0]
    noise_moment = quad(noise_density, 0., upper, epsabs=1e-12)[0]
    np.testing.assert_allclose(probability, .5, atol=1e-11, rtol=0)
    np.testing.assert_allclose(moment/probability, hard_center, atol=1e-11, rtol=0)
    np.testing.assert_allclose(noise_moment/probability, noise_mean, atol=1e-11, rtol=0)
    modes = {}
    for name, center in (("hard_fit", hard_center),
                         ("exact_noise_compensation_then_hard_fit", a*kappa),
                         ("known_candidate_MAP_projection", a),
                         ("independent_noise_then_hard_fit_population", hard_center)):
        modes[name] = dict(
            positive_output_group_center_mm=center,
            output_group_center_gap_mm=2.*center,
            true_source_mean_output_gap_mm=2.*center*kappa,
            nearest_true_surface_mae_mm=abs(center-a),
            point_correspondence_mse_mm2=a*a+center*center-2.*a*center*kappa)
    return dict(true_gap_mm=2.*a, sigma_mm=sigma, kappa=kappa,
                wrong_group_probability=float(ndtr(-a/sigma)),
                positive_group_conditional_noise_mean_mm=noise_mean,
                quadrature_max_error=float(max(abs(probability-.5),
                    abs(moment/probability-hard_center),
                    abs(noise_moment/probability-noise_mean))), modes=modes)


def simulate(a: float, sigma: float = 1., count: int = 400_000) -> dict:
    rng = np.random.default_rng(912909+int(10*a))
    source = np.where(rng.random(count) < .5, -1., 1.)
    truth = source*a
    epsilon = rng.normal(0., sigma, count)
    eta = rng.normal(0., sigma, count)
    observed = truth+epsilon
    group = np.where(observed >= 0., 1., -1.)
    theory = known_model(a, sigma)
    b = theory['positive_group_conditional_noise_mean_mm']
    results = {}
    for name, response in (("hard_fit", observed),
                           ("exact_noise_compensation_then_hard_fit", observed-b*group),
                           ("known_candidate_MAP_projection", None),
                           ("independent_noise_then_hard_fit_population", observed-eta)):
        if response is None:
            centers = np.array([-a, a])
        else:
            centers = np.array([response[group < 0].mean(), response[group > 0].mean()])
        output = centers[(group > 0).astype(int)]
        got = dict(output_group_center_gap_mm=float(centers[1]-centers[0]),
                   true_source_mean_output_gap_mm=float(output[source > 0].mean()-output[source < 0].mean()),
                   nearest_true_surface_mae_mm=float(np.mean(abs(abs(output)-a))),
                   point_correspondence_mse_mm2=float(np.mean((output-truth)**2)))
        expected = theory['modes'][name]
        # Monte Carlo is an auxiliary check, with a loose stated sampling bound;
        # exact numerical integration above is the deterministic formula check.
        errors = {key:abs(value-expected[key]) for key,value in got.items()}
        tolerance = .03
        assert max(errors.values()) < tolerance, (name, errors)
        results[name] = dict(**got, max_absolute_theory_discrepancy=max(errors.values()),
                             diagnostic_sampling_tolerance=tolerance)
    raw_difference_noise_variance = float(np.var(observed-eta-truth))
    assert abs(raw_difference_noise_variance-2.*sigma*sigma) < .04*sigma*sigma
    return dict(sample_count=count, independent_noise_raw_error_variance_mm2=raw_difference_noise_variance,
                expected_independent_noise_raw_error_variance_mm2=2.*sigma*sigma, modes=results)


def constant_shift_check() -> dict:
    rng = np.random.default_rng(912910)
    x = np.column_stack([np.ones(180), rng.normal(size=(180,2))])
    w = rng.uniform(.1, 1.5, len(x))
    y = x@np.array([.2, .03, -.04])+rng.normal(size=len(x))
    root = np.sqrt(w)
    fit = lambda v:np.linalg.lstsq(x*root[:, None], v*root, rcond=1e-12)[0]
    before, after = fit(y), fit(y-.7)
    expected = before-np.array([.7, 0., 0.])
    np.testing.assert_allclose(after, expected, atol=1e-12, rtol=0)
    return dict(max_coefficient_identity_error=float(np.max(abs(after-expected))),
                scope='each fixed full-rank fitted group; not a GT-source-wide slope guarantee')


def main() -> None:
    result = dict(status='PASS', role='known-model diagnostic, no deployable oracle comparison',
                  input_files_read=[], output_files_written=[],
                  constant_shift_identity=constant_shift_check(), cases=[])
    for a in (1.,2.,4.):
        result['cases'].append(dict(theory_and_quadrature=known_model(a), simulation=simulate(a)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
