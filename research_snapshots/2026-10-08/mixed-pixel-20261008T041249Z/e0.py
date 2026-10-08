"""E0 analytic boundary sanity checks, independent of the E1 renderer.

No generator, predictor, truth file, or historical implementation is imported.
The half-plane/pixel intersection is analytic polygon clipping; the reference
quadrature is an independent uniform midpoint sampler, with convergence checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def halfplane_pixel_area(normal, threshold, center=(0., 0.)):
    """Area of n dot (x,y) <= threshold inside a unit sensor pixel."""
    normal = np.asarray(normal, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    vertices = center + np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    clipped = []
    previous = vertices[-1]
    previous_distance = float(previous @ normal - threshold)
    for current in vertices:
        current_distance = float(current @ normal - threshold)
        if (current_distance <= 0.) != (previous_distance <= 0.):
            fraction = previous_distance / (previous_distance - current_distance)
            clipped.append(previous + fraction * (current - previous))
        if current_distance <= 0.:
            clipped.append(current)
        previous, previous_distance = current, current_distance
    if len(clipped) < 3:
        return 0.
    polygon = np.asarray(clipped)
    # Translating before shoelace reduces cancellation at large pixel indices.
    polygon -= center
    area = .5 * abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                           - polygon[:, 1] * np.roll(polygon[:, 0], -1)))
    return float(np.clip(area, 0., 1.))


def midpoint_area(normal, threshold, resolution, center=(0., 0.)):
    """Independent dense sampler; no polygon-clipping or area formulas."""
    offset = (np.arange(resolution, dtype=np.float64) + .5) / resolution - .5
    x = center[0] + offset
    y = center[1] + offset
    count = 0
    # Bound temporary memory independently of resolution.
    for start in range(0, resolution, 64):
        inside = normal[0] * x[None, :] + normal[1] * y[start:start + 64, None] <= threshold
        count += int(np.count_nonzero(inside))
    return count / float(resolution * resolution)


def analytic_checks():
    normals = [(1., .37), (1., -.53), (1., .71)]
    thresholds = [-.43, -.13, 0., .17, .49]
    resolutions = [32, 128, 512, 2048]
    cases, maximum_errors, rms_errors = [], [], []
    exact = np.array([halfplane_pixel_area(n, t) for n in normals for t in thresholds])
    for resolution in resolutions:
        sampled = np.array([midpoint_area(n, t, resolution) for n in normals for t in thresholds])
        errors = np.abs(sampled - exact)
        maximum_errors.append(float(errors.max()))
        rms_errors.append(float(np.sqrt(np.mean(errors ** 2))))
    for normal in normals:
        for threshold in thresholds:
            area = halfplane_pixel_area(normal, threshold)
            complement = halfplane_pixel_area(-np.asarray(normal), -threshold)
            cases.append(dict(normal=list(normal), threshold=threshold, area=area,
                              complement_sum=area + complement))
    assert all(0. <= row["area"] <= 1. for row in cases)
    assert max(abs(row["complement_sum"] - 1.) for row in cases) < 1e-12
    assert halfplane_pixel_area((1., 0.), 0.) == .5
    assert halfplane_pixel_area((1., 0.), -1.) == 0.
    assert halfplane_pixel_area((1., 0.), 1.) == 1.
    assert maximum_errors[-1] < 2e-4
    assert rms_errors[-1] < rms_errors[0]
    return dict(cases=cases, resolutions=resolutions, maximum_absolute_error=maximum_errors,
                rms_error=rms_errors, passed=True)


def mixed_color_depth_checks():
    # A reference half-plane x + slope*y <= boundary is lifted to each depth.
    # A rectified source with Cx=60 has x_src=x_ref-f*Cx/z. This known camera
    # geometry alone determines candidate coverage; observed intensities do not.
    focal, baseline, slope = 160., 60., .37
    candidates = np.array([450., 540., 600., 660., 900.])
    true_depth, incumbent = 600., 540.
    boundary = 64. + slope * 64. + .11
    pixels = np.array([(x, y) for y in (63, 64, 65) for x in range(44, 57)], dtype=float)
    coverage = np.array([[halfplane_pixel_area((1., slope), boundary - focal * baseline / z, pixel)
                         for pixel in pixels] for z in candidates])
    assert np.all((coverage >= 0.) & (coverage <= 1.))
    true_index = int(np.flatnonzero(candidates == true_depth)[0])
    incumbent_index = int(np.flatnonzero(candidates == incumbent)[0])
    foreground, background = 190., 40.
    prediction = coverage * foreground + (1. - coverage) * background
    observed = prediction[true_index]
    dynamic_loss = np.mean((prediction - observed) ** 2, axis=1)
    fixed_prediction = np.repeat(prediction[incumbent_index][None, :], len(candidates), axis=0)
    fixed_loss = np.mean((fixed_prediction - observed) ** 2, axis=1)
    assert dynamic_loss[true_index] == 0.
    assert np.all(np.delete(dynamic_loss, true_index) > 0.)
    assert np.ptp(fixed_loss) == 0.
    equal_prediction = coverage * 127. + (1. - coverage) * 127.
    equal_loss = np.mean((equal_prediction - 127.) ** 2, axis=1)
    assert np.max(np.abs(equal_prediction - 127.)) < 3e-14
    assert np.max(equal_loss) < 1e-24
    assert np.ptp(equal_loss) < 1e-24
    reconstruction_error = float(np.max(np.abs(observed - (
        coverage[true_index] * foreground + (1. - coverage[true_index]) * background))))
    assert reconstruction_error == 0.
    return dict(candidates=candidates.tolist(), true_depth=true_depth, incumbent=incumbent,
                camera=dict(focal=focal, baseline=baseline), pixels=pixels.tolist(),
                normal=[1., slope], reference_boundary=boundary,
                dynamic_coverage=coverage.tolist(), dynamic_prediction=prediction.tolist(),
                observed=observed.tolist(), dynamic_loss=dynamic_loss.tolist(),
                fixed_loss=fixed_loss.tolist(), equal_color_loss=equal_loss.tolist(),
                equal_color_max_prediction_error=float(np.max(np.abs(equal_prediction - 127.))),
                reconstruction_max_error=reconstruction_error, passed=True)


def fixed_coverage_texture_check():
    # Nine subrays; an incumbent assigns the left column to foreground. Holding
    # those exact three ownership bits fixed preserves a candidate-dependent
    # foreground warp. Linear texture makes all expected numbers hand-checkable.
    offsets = np.array([(x, y) for y in (-1 / 3, 0., 1 / 3) for x in (-1 / 3, 0., 1 / 3)])
    ownership = offsets[:, 0] < 0.
    depth = np.array([540., 600., 660.])
    source_x, focal, baseline = 48., 160., 60.
    warped_x = source_x + offsets[None, :, 0] + focal * baseline / depth[:, None]
    foreground = 100. + 2. * (warped_x - 64.)
    background = 40.
    prediction = np.mean(np.where(ownership[None, :], foreground, background), axis=1)
    expected = (1 / 3) * (100. + 2. * (source_x - 1 / 3 + focal * baseline / depth - 64.)) + (2 / 3) * background
    assert np.max(np.abs(prediction - expected)) < 2e-14
    assert np.ptp(prediction) > 1.
    return dict(candidates=depth.tolist(), frozen_subray_ownership=ownership.astype(int).tolist(),
                frozen_alpha=float(ownership.mean()), prediction=prediction.tolist(),
                hand_computed=expected.tolist(), candidate_prediction_range=float(np.ptp(prediction)),
                passed=True)


def run(output_dir):
    root = Path(output_dir).resolve()
    if (root / "RESULTS.json").exists():
        raise FileExistsError("E0 output already exists; retain the previous result")
    root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    analytic = analytic_checks()
    mixed = mixed_color_depth_checks()
    texture = fixed_coverage_texture_check()
    result = dict(stage="E0", passed=True, analytic_boundary=analytic,
                  mixed_color=mixed, fixed_coverage_texture=texture,
                  errors=[], elapsed_seconds=time.perf_counter() - start,
                  scope="Independent analytic/numerical sanity, not E1 algorithm performance",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / "RESULTS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    command_summary = dict(passed=result["passed"], elapsed_seconds=result["elapsed_seconds"],
                           analytic_final_max_error=analytic["maximum_absolute_error"][-1],
                           contrast_dynamic_loss=mixed["dynamic_loss"],
                           contrast_fixed_loss=mixed["fixed_loss"],
                           equal_color_loss=mixed["equal_color_loss"],
                           fixed_coverage_texture=texture["prediction"])
    text = json.dumps(command_summary, indent=2, allow_nan=False)
    (root / "COMMAND_OUTPUT.txt").write_text(text + "\n")
    print(text)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "e0"))
    args = parser.parse_args()
    run(args.output)
