#!/usr/bin/env python3
"""Finite, CPU-only witnesses for THEORY.md. No input data, labels, or tuning.

Run: python3 theory/test_theory.py
All constants are part of the analytic examples; output is JSON on stdout.
"""

import json
import math

import numpy as np


TAU = 0.6
EDGE_THRESHOLD = 20.0
PAD_METERS = 0.001


def descriptor(values):
    centered = np.asarray(values, dtype=float) - np.mean(values)
    norm = np.linalg.norm(centered)
    if norm <= 1e-12:
        raise ValueError("constant patch has no NCC descriptor")
    return centered / norm


def ncc(a, b):
    return float(descriptor(a) @ descriptor(b))


def connected_center(reference):
    """The exact stated 4-neighbor, intensity-difference ownership hypothesis."""
    size = reference.shape[0]
    center = (size // 2, size // 2)
    selected = {center}
    pending = [center]
    while pending:
        row, col = pending.pop()
        for rr, cc in ((row - 1, col), (row + 1, col),
                       (row, col - 1), (row, col + 1)):
            if not (0 <= rr < size and 0 <= cc < size):
                continue
            if (rr, cc) in selected:
                continue
            if abs(reference[row, col] - reference[rr, cc]) <= EDGE_THRESHOLD:
                selected.add((rr, cc))
                pending.append((rr, cc))
    result = np.zeros(reference.shape, dtype=bool)
    for row, col in selected:
        result[row, col] = True
    return result


def support_intervals(grid, accepted):
    """Each accepted run gets half a grid step + 1 mm; merge overlaps."""
    half_step = float(grid[1] - grid[0]) / 2
    indices = np.flatnonzero(accepted)
    if len(indices) == 0:
        return []
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    intervals = []
    for run in runs:
        low = max(float(grid[0]), float(grid[run[0]]) - half_step - PAD_METERS)
        high = min(float(grid[-1]), float(grid[run[-1]]) + half_step + PAD_METERS)
        if intervals and low <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], high)
        else:
            intervals.append([low, high])
    return intervals


def interval_mean(intervals):
    total = sum(high - low for low, high in intervals)
    if total > 0:
        return sum((high - low) * (low + high) / 2
                   for low, high in intervals) / total
    return float(np.mean(sorted({low for low, _ in intervals})))


def choose_p(candidates, incumbent, mean_depth):
    target = np.array([0.0, 0.0, mean_depth])
    squared = np.sum((candidates - target) ** 2, axis=1)
    best = int(np.argmin(squared))
    gain = float(squared[incumbent] - squared[best])
    selected = best if gain > 0 else incumbent
    return selected, gain, target


def exact_and_perturbed_owned_surface():
    # A three-phase sinusoid makes centering and descriptor norm exact on 3x3.
    x = np.tile(np.arange(-1, 2, dtype=float), 3)
    omega = 2 * math.pi / 3
    amplitude, mean = 10.0, 128.0
    beta = 10 * math.pi
    z_star = 1.0
    reference = mean + amplitude * np.cos(omega * x)
    assert np.std(reference) >= 3.0
    grid = np.linspace(0.92, 1.08, 161)
    max_warp_error, max_photo_error = 0.001, 0.01
    sigma = amplitude / math.sqrt(2)
    descriptor_error = 2 * (amplitude * omega * max_warp_error + max_photo_error) / sigma
    threshold_distance = math.sqrt(2 * (1 - TAU))
    q_radius = 2 / beta * math.asin((threshold_distance + descriptor_error) / 2)
    ideal_depth_bounds = [1 / (1 + q_radius), 1 / (1 - q_radius)]
    pad = (grid[1] - grid[0]) / 2 + PAD_METERS
    target_error_bound = max(z_star - ideal_depth_bounds[0],
                             ideal_depth_bounds[1] - z_star) + pad
    exact_error, normalized_error = 0.0, 0.0
    scores = []
    for index, depth in enumerate(grid):
        per_view = []
        for view_sign in (-1, 1):
            phase = omega * x + view_sign * beta * (1 / z_star - 1 / depth)
            ideal = mean + amplitude * np.cos(phase)
            delta_x = max_warp_error * np.sin(np.arange(9) + index * 0.73 + view_sign)
            noise = max_photo_error * np.cos(np.arange(9) * 0.37 + index + view_sign)
            observed = mean + amplitude * np.cos(phase + omega * delta_x) + noise
            assert np.std(observed) >= 3.0
            exact_error = max(exact_error, abs(ncc(reference, ideal) - math.cos(beta * (1 / depth - 1))))
            normalized_error = max(normalized_error, float(np.linalg.norm(descriptor(observed) - descriptor(ideal))))
            per_view.append(ncc(reference, observed))
        scores.append(per_view)
    scores = np.asarray(scores)
    accepted = np.all(scores >= TAU, axis=1)
    accepted_depths = grid[accepted]
    assert exact_error < 1e-12
    assert normalized_error <= descriptor_error + 1e-12
    assert accepted[np.argmin(abs(grid - z_star))]
    assert np.all(accepted_depths >= ideal_depth_bounds[0] - 1e-12)
    assert np.all(accepted_depths <= ideal_depth_bounds[1] + 1e-12)
    assert max(abs(beta * (1 / grid - 1))) < math.pi  # No alias in this example.
    intervals = support_intervals(grid, accepted)
    mean_depth = interval_mean(intervals)
    assert abs(mean_depth - z_star) <= target_error_bound
    candidates = np.array([[0, 0, 1.12], [0, 0, 1], [0, 0, 0.88],
                           [0.12, 0, 1], [0, 0.12, 1]], dtype=float)
    chosen, observed_gain, target = choose_p(candidates, 0, mean_depth)
    truth = np.array([0.0, 0.0, z_star])
    true_gain = float(np.sum((candidates[0] - truth) ** 2) - np.sum((candidates[chosen] - truth) ** 2))
    step_length = float(np.linalg.norm(candidates[chosen] - candidates[0]))
    certified_gain = observed_gain - 2 * step_length * target_error_bound
    assert chosen == 1
    assert true_gain >= certified_gain > 0
    for index in (0, 2, 3, 4):
        pairwise_gap = float(np.sum((candidates[index] - truth) ** 2))
        assert pairwise_gap > 2 * np.linalg.norm(candidates[index] - truth) * target_error_bound
    return {
        "grid_count": len(grid), "accepted_count": int(np.sum(accepted)),
        "max_exact_ncc_formula_error": exact_error,
        "observed_descriptor_perturbation_max": normalized_error,
        "descriptor_perturbation_bound": descriptor_error,
        "derived_inverse_depth_radius": q_radius,
        "derived_unpadded_depth_bounds_m": ideal_depth_bounds,
        "accepted_depth_extrema_m": [float(min(accepted_depths)), float(max(accepted_depths))],
        "padded_intervals_m": intervals,
        "mean_depth_m": mean_depth,
        "target_position_error_bound_m": target_error_bound,
        "selected_candidate": chosen,
        "observed_squared_gain_m2": observed_gain,
        "guaranteed_squared_gain_lower_bound_m2": certified_gain,
        "actual_squared_gain_m2": true_gain,
    }


def mixed_surface_counterexample():
    # The center owns depth 1; the 72-pixel ring owns depth 2/3.
    # All reference neighbor differences are <=15, so connected9 takes all 81.
    yy, xx = np.meshgrid(np.arange(-4, 5), np.arange(-4, 5), indexing="ij")
    owned = (abs(xx) <= 1) & (abs(yy) <= 1)
    reference = 128 + 10 * np.cos((2 * math.pi / 3) * xx)
    graph_mask = connected_center(reference)
    assert int(np.sum(graph_mask)) == 81
    truth_depth = 1.0
    ring_depth = 2.0 / 3
    actual_inverse_depth = np.where(owned, 1 / truth_depth, 1 / ring_depth)
    beta = 2 * math.pi
    grid = np.linspace(0.60, 1.24, 641)
    masks = {"center3": owned, "full9": np.ones((9, 9), dtype=bool),
             "connected9": graph_mask}
    candidates = np.array([[0, 0, 0.94], [0, 0, truth_depth], [0, 0, ring_depth]])
    results = {}
    for name, mask in masks.items():
        scores = []
        for depth in grid:
            per_view = []
            for sign in (-1, 1):
                source = 128 + 10 * np.cos((2 * math.pi / 3) * xx + sign * beta * (actual_inverse_depth - 1 / depth))
                per_view.append(ncc(reference[mask], source[mask]))
            scores.append(per_view)
        scores = np.asarray(scores)
        accepted = np.all(scores >= TAU, axis=1)
        intervals = support_intervals(grid, accepted)
        mean_depth = interval_mean(intervals)
        chosen, gain, _ = choose_p(candidates, 0, mean_depth)
        exact_at_truth, exact_at_ring = [], []
        for depth in (truth_depth, ring_depth):
            source = 128 + 10 * np.cos((2 * math.pi / 3) * xx + beta * (actual_inverse_depth - 1 / depth))
            value = ncc(reference[mask], source[mask])
            (exact_at_truth if depth == truth_depth else exact_at_ring).append(value)
        expected = 1 if name == "center3" else 2
        assert chosen == expected
        if name != "center3":
            assert abs(exact_at_truth[0] + 7 / 9) < 1e-12
            assert abs(exact_at_ring[0] - 7 / 9) < 1e-12
            assert not accepted[np.argmin(abs(grid - truth_depth))]
        selected_error = float(abs(candidates[chosen, 2] - truth_depth))
        results[name] = {
            "mask_size": int(np.sum(mask)), "ncc_at_center_truth": exact_at_truth[0],
            "ncc_at_ring_depth": exact_at_ring[0], "accepted_count": int(np.sum(accepted)),
            "padded_intervals_m": intervals, "mean_depth_m": mean_depth,
            "selected_candidate": chosen, "selected_depth_m": float(candidates[chosen, 2]),
            "selected_queried_point_error_m": selected_error,
            "incumbent_queried_point_error_m": 0.06,
            "observed_squared_gain_m2": gain,
        }
    return results


def positive_gain_does_not_certify_true_gain():
    candidates = np.array([[0, 0, 1], [0, 0, 1.02]], dtype=float)
    chosen, observed_gain, _ = choose_p(candidates, 0, 1.02)
    true_gain = -(0.02 ** 2)
    lower_bound = observed_gain - 2 * 0.02 * 0.02
    assert chosen == 1 and observed_gain > 0 and true_gain < 0
    assert abs(true_gain - lower_bound) < 1e-15  # The gain bound is sharp.
    return {"observed_squared_gain_m2": observed_gain,
            "true_squared_gain_m2": true_gain,
            "bound_m2": lower_bound}


def point_gain_is_not_surface_gain():
    # True surface is the plane z=1; the first candidate lies on that plane.
    truth = np.array([0.0, 0.0, 1.0])
    candidates = np.array([[0.1, 0, 1], [0, 0, 1.01]], dtype=float)
    chosen, gain, _ = choose_p(candidates, 0, 1.0)
    assert chosen == 1 and gain > 0
    before_surface = abs(candidates[0, 2] - 1)
    after_surface = abs(candidates[chosen, 2] - 1)
    assert after_surface > before_surface
    return {"queried_point_squared_gain_m2": gain,
            "surface_distance_before_m": float(before_surface),
            "surface_distance_after_m": float(after_surface)}


def padding_is_not_photometric_certification():
    grid = np.array([0.99, 1.00, 1.01])
    accepted = np.array([False, True, False])
    intervals = support_intervals(grid, accepted)
    assert np.allclose(intervals, [[0.994, 1.006]], atol=1e-14, rtol=0)
    # A continuous NCC curve can pass at 1.000 and fail inside its fill cell.
    narrow_ncc = lambda z: math.exp(-((z - 1.0) / 0.0001) ** 2)
    assert narrow_ncc(1.0) >= TAU and narrow_ncc(1.001) < TAU
    return {"padded_interval_m": intervals[0],
            "ncc_at_accepted_sample": narrow_ncc(1.0),
            "ncc_at_filled_unsampled_depth": narrow_ncc(1.001)}


def main():
    results = {
        "status": "PASS",
        "scope": "analytic finite examples; no real images, evaluation labels, or GT files read",
        "owned_surface_theorem": exact_and_perturbed_owned_surface(),
        "ownership_counterexample": mixed_surface_counterexample(),
        "positive_gain_counterexample": positive_gain_does_not_certify_true_gain(),
        "point_vs_surface_counterexample": point_gain_is_not_surface_gain(),
        "padding_counterexample": padding_is_not_photometric_certification(),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
