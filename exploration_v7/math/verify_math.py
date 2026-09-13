"""Small algebra checks, separate from V7 data and algorithm performance.

Only NumPy is needed. This does not import the project estimator, read its
datasets, or establish that an association posterior is physically calibrated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _check_problem(cost, prior):
    cost = np.asarray(cost, dtype=float)
    prior = np.asarray(prior, dtype=float)
    if cost.ndim != 2 or prior.shape != cost.shape:
        raise ValueError("cost and prior must be matching N by G arrays")
    if not np.isfinite(cost).all() or not np.isfinite(prior).all():
        raise ValueError("finite arrays required; zero prior encodes an absent candidate")
    if (prior < 0).any() or not np.allclose(prior.sum(axis=1), 1., atol=1e-14, rtol=0):
        raise ValueError("each prior row must be a probability vector")
    return cost, prior


def posterior(cost, prior):
    """Exact entropy-regularized row update; no prescribed column marginal."""
    cost, prior = _check_problem(cost, prior)
    log_prior = np.full_like(prior, -np.inf)
    np.log(prior, out=log_prior, where=prior > 0)
    logits = log_prior - cost
    row_max = logits.max(axis=1, keepdims=True)
    shifted = np.exp(logits - row_max)
    normalizer = shifted.sum(axis=1, keepdims=True)
    return shifted / normalizer, (row_max + np.log(normalizer)).ravel()


def free_energy(responsibilities, cost, prior, weights):
    """Sum_i w_i Sum_g r_ig [cost_ig + log(r_ig / prior_ig)]."""
    cost, prior = _check_problem(cost, prior)
    r = np.asarray(responsibilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if r.shape != cost.shape or weights.shape != (len(cost),):
        raise ValueError("invalid responsibilities or weights shape")
    if not np.isfinite(r).all() or not np.isfinite(weights).all():
        raise ValueError("finite responsibilities and weights required")
    if (r < 0).any() or (weights < 0).any():
        raise ValueError("nonnegative responsibilities and weights required")
    if not np.allclose(r.sum(axis=1), 1., atol=1e-14, rtol=0):
        raise ValueError("responsibility rows must conserve mass")
    use = r > 0
    if np.any(use & (prior == 0)):
        return float("inf")
    terms = np.zeros_like(r)
    terms[use] = r[use] * (cost[use] + np.log(r[use] / prior[use]))
    return float(weights @ terms.sum(axis=1))


def plane_cost(design, values, coefficients, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive finite sigma required")
    return (values[:, None] - design @ coefficients.T) ** 2 / (2. * sigma ** 2)


def refit_original_measurements(design, values, weights, r, coefficients):
    """Least-squares M step, retaining an empty component without a GT reset."""
    fitted = coefficients.copy()
    ranks = []
    for group in range(r.shape[1]):
        root = np.sqrt(weights * r[:, group])
        if not np.any(root > 0):
            ranks.append(0)
            continue
        fitted[group], _, rank, _ = np.linalg.lstsq(
            design * root[:, None], values * root, rcond=1e-12
        )
        ranks.append(int(rank))
    return fitted, ranks


def monotonic_toy(rank_deficient=False):
    rng = np.random.default_rng(913107)
    design = np.c_[np.ones(160), rng.normal(size=(160, 2))]
    if rank_deficient:
        design[:, 2] = design[:, 1]
    values = np.where(np.arange(160) % 2, -2., 2.) + .1 * design[:, 1]
    values += rng.normal(scale=.6, size=160)
    weights = rng.uniform(.4, 1., size=160)
    prior = np.full((len(design), 2), .5)
    r = prior.copy()
    beta = np.array([[-1.5, 0., 0.], [1.5, 0., 0.]])
    sigma = 1.
    trace = []
    rank_trace = []
    for _ in range(8):
        cost = plane_cost(design, values, beta, sigma)
        trace.append(free_energy(r, cost, prior, weights))
        r, _ = posterior(cost, prior)
        trace.append(free_energy(r, cost, prior, weights))
        beta, ranks = refit_original_measurements(design, values, weights, r, beta)
        rank_trace.append(ranks)
    trace.append(free_energy(r, plane_cost(design, values, beta, sigma), prior, weights))
    change = np.diff(trace)
    tolerance = 1e-10 * max(1., max(abs(np.asarray(trace))))
    if change.max() > tolerance:
        raise AssertionError("a fixed-problem E or M half-step increased free energy")
    return {"seed": 913107, "rank_deficient_design": rank_deficient,
            "objective_half_steps": trace, "rank_by_iteration": rank_trace,
            "maximum_objective_increase": float(change.max()),
            "numerical_tolerance": tolerance,
            "maximum_row_mass_error": float(abs(r.sum(axis=1) - 1.).max())}


def duplicate_representation_toy():
    original, _ = posterior(np.zeros((1, 2)), np.array([[.5, .5]]))
    changed, _ = posterior(np.zeros((1, 3)), np.array([[1/3, 1/3, 1/3]]))
    refined, _ = posterior(np.zeros((1, 3)), np.array([[.25, .25, .5]]))
    return {"original_aggregate": original[0].tolist(),
            "cloned_uniform_aggregate": [float(changed[0, :2].sum()), float(changed[0, 2])],
            "cloned_split_prior_aggregate": [float(refined[0, :2].sum()), float(refined[0, 2])],
            "interpretation": "uniform normalization changes the prior measure; splitting prior mass does not"}


def decision_toy():
    planes = np.array([-2., 2.])
    probability = np.array([.5, .5])
    mean = float(probability @ planes)
    hard = float(planes[0])
    return {"planes_mm": planes.tolist(), "posterior": probability.tolist(),
            "mean_mm": mean, "hard_mm": hard,
            "mean_distance_to_plane_union_mm": float(abs(planes - mean).min()),
            "hard_distance_to_plane_union_mm": float(abs(planes - hard).min()),
            "mean_expected_correspondence_mse_mm2": float(probability @ (planes - mean) ** 2),
            "hard_expected_correspondence_mse_mm2": float(probability @ (planes - hard) ** 2),
            "interpretation": "different losses favor different actions; no physical calibration is implied"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"scope": "independent algebra toys; not V7 estimator/data evidence",
              "monotonic_full_rank": monotonic_toy(),
              "monotonic_rank_deficient": monotonic_toy(True),
              "duplicate_representation": duplicate_representation_toy(),
              "decision": decision_toy()}
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
