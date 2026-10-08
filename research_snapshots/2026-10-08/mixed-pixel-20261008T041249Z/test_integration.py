"""Independent E0-to-predictor integration contracts; no full E1 generation.

Direct execution writes a new E0 integration result. The original E0 analytic
result is not changed. All observations exist only in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import unittest

import numpy as np

import fixtures
import predictor


GRID = np.array([450., 540., 600., 660., 900.])


def _snapshot(aux):
    return {key: np.asarray(value).copy() for key, value in aux.items()}


def _unchanged(before, after):
    assert set(before) == set(after)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def independent_fixture_contract():
    # A single contrast world is enough to check the rotated supplied cameras,
    # independent ray/plane generator, oracle array contract, and both modes.
    world = fixtures.make_world("flat_contrast", 1.65, 1103)
    cameras = fixtures.cameras()
    images = fixtures.render_world(world, cameras)
    aux = fixtures.oracle_aux(world)
    before = _snapshot(aux)
    folds = []
    for view in (1, 2):
        dynamic = predictor.predict_pixels(aux, cameras[0], cameras[view], GRID, 540.)
        fixed = predictor.predict_pixels(aux, cameras[0], cameras[view], GRID, 540., mode="fixed")
        dynamic_curve = predictor.predict_curve(aux, cameras[0], cameras[view], images[view], GRID, 540.)
        fixed_curve = predictor.predict_curve(aux, cameras[0], cameras[view], images[view], GRID, 540., mode="fixed")
        assert dynamic_curve["valid"] and fixed_curve["valid"]
        assert not dynamic_curve["sigma_valid"]  # Raw curves remain available.
        np.testing.assert_array_equal(dynamic["roi"], fixed["roi"])
        np.testing.assert_array_equal(dynamic_curve["roi"], fixed_curve["roi"])
        assert dynamic_curve["pixel_count"] == fixed_curve["pixel_count"] == len(dynamic["roi"])
        true_index = 2
        roi = dynamic["roi"]
        generated_pixels = images[view, roi[:, 1], roi[:, 0]]
        max_error = float(np.max(np.abs(dynamic["prediction"][true_index] - generated_pixels)))
        assert max_error < 1e-12
        assert dynamic_curve["loss"][true_index] < 1e-24
        assert np.all(np.delete(dynamic_curve["loss"], true_index) > 1.)
        assert np.ptp(fixed_curve["loss"]) < 1e-12
        assert fixed_curve["loss"][true_index] > 1.
        expected_loss = np.mean((dynamic["prediction"] - generated_pixels[None]) ** 2, axis=1)
        np.testing.assert_allclose(dynamic_curve["loss"], expected_loss, atol=1e-12)
        for depth in range(1, len(GRID)):
            np.testing.assert_array_equal(fixed["ownership"][0], fixed["ownership"][depth])
        folds.append(dict(view=view, pixel_count=len(roi), max_true_pixel_error=max_error,
                          dynamic_loss=dynamic_curve["loss"].tolist(),
                          fixed_loss=fixed_curve["loss"].tolist(),
                          raw_valid_without_sigma=True))
    _unchanged(before, aux)
    return dict(passed=True, folds=folds, auxiliaries_unchanged=True)


def full_image_equal_color_contract():
    world = fixtures.make_world("flat_equal", 1.65, 2207)
    cameras = fixtures.cameras()
    images = fixtures.render_world(world, cameras)
    aux = fixtures.oracle_aux(world)
    assert np.ptp(images) == 0.
    curves = []
    for view in (1, 2):
        for incumbent in (540., 600., 660.):
            for mode in ("dynamic", "fixed"):
                pixels = predictor.predict_pixels(aux, cameras[0], cameras[view], GRID, incumbent, mode=mode)
                curve = predictor.predict_curve(aux, cameras[0], cameras[view], images[view], GRID, incumbent, mode=mode)
                assert curve["valid"] and curve["flat"]
                assert np.ptp(pixels["prediction"]) < 6e-14
                assert np.max(curve["loss"]) < 1e-24
                curves.append(dict(view=view, incumbent=incumbent, mode=mode,
                                   loss=curve["loss"].tolist(), flat=bool(curve["flat"])))
    return dict(passed=True, image_dynamic_range=float(np.ptp(images)), curves=curves)


def _manual_halfplane_aux(textured=False):
    # Polygon is the known half-plane clipped to a large reference square. Its
    # other three edges are far away from every tested ray and cannot contribute.
    slope = .37
    boundary = 64. + slope * 64. + .11
    polygon = np.array([[0., 0.], [boundary, 0.],
                        [boundary - slope * 127., 127.], [0., 127.]])
    yy, xx = np.indices((128, 128), dtype=float)
    foreground = 100. + .8 * (xx - 64.) if textured else np.full((128, 128), 190.)
    aux = dict(mode="two", foreground=foreground, background=np.full((128, 128), 40.),
               mask=(xx + slope * yy <= boundary).astype(float), polygon=polygon,
               background_depth=1500., valid=True, sigma_valid=False)
    K = np.array([[160., 0., 64.], [0., 160., 64.], [0., 0., 1.]])
    reference = dict(K=K.copy(), R=np.eye(3), C=np.zeros(3))
    source = dict(K=K.copy(), R=np.eye(3), C=np.array([60., 0., 0.]))
    return aux, reference, source, slope, boundary


def _manual_rays(roi):
    # Written independently of predictor's private offset implementation.
    coords = np.arange(3, dtype=float) / 3. - 1 / 3
    offsets = np.array([[x, y] for y in coords for x in coords])
    return roi[:, None, :] + offsets[None, :, :]


def analytic_halfplane_contract():
    aux, reference, source, slope, boundary = _manual_halfplane_aux()
    dynamic = predictor.predict_pixels(aux, reference, source, GRID, 540.)
    fixed = predictor.predict_pixels(aux, reference, source, GRID, 540., mode="fixed")
    np.testing.assert_array_equal(dynamic["roi"], fixed["roi"])
    rays = _manual_rays(dynamic["roi"])
    reference_x = rays[None, :, :, 0] + 160. * 60. / GRID[:, None, None]
    reference_y = rays[None, :, :, 1]
    expected_ownership = reference_x + slope * reference_y <= boundary
    expected_alpha = expected_ownership.mean(axis=-1)
    expected = 190. * expected_alpha + 40. * (1. - expected_alpha)
    np.testing.assert_array_equal(dynamic["ownership"], expected_ownership)
    np.testing.assert_array_equal(dynamic["alpha"], expected_alpha)
    np.testing.assert_allclose(dynamic["prediction"], expected, rtol=0., atol=3e-14)
    np.testing.assert_array_equal(fixed["ownership"], np.broadcast_to(expected_ownership[1], expected_ownership.shape))
    # Observations are generated by a direct rectified-camera expression, not
    # by sampling predictor output or its alpha at the true candidate.
    yy, xx = np.indices((128, 128), dtype=float)
    sensor = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    source_rays = _manual_rays(sensor)
    true_owned = source_rays[..., 0] + 160. * 60. / 600. + slope * source_rays[..., 1] <= boundary
    image = np.where(true_owned, 190., 40.).mean(axis=-1).reshape(128, 128)
    curve = predictor.predict_curve(aux, reference, source, image, GRID, 540.)
    assert curve["valid"] and curve["loss"][2] < 1e-24
    assert np.all(np.delete(curve["loss"], 2) > 1.)
    return dict(passed=True, pixel_count=len(dynamic["roi"]),
                max_prediction_error=float(np.max(np.abs(dynamic["prediction"] - expected))),
                dynamic_loss=curve["loss"].tolist(), exact_subray_ownership=True)


def fixed_subray_texture_contract():
    aux, reference, source, slope, boundary = _manual_halfplane_aux(textured=True)
    before = _snapshot(aux)
    fixed = predictor.predict_pixels(aux, reference, source, GRID, 540., mode="fixed")
    rays = _manual_rays(fixed["roi"])
    incumbent_x = rays[..., 0] + 160. * 60. / 540.
    expected_ownership = incumbent_x + slope * rays[..., 1] <= boundary
    expected_ownership = np.broadcast_to(expected_ownership, fixed["ownership"].shape)
    np.testing.assert_array_equal(fixed["ownership"], expected_ownership)
    warped_x = rays[None, :, :, 0] + 160. * 60. / GRID[:, None, None]
    # All warped coordinates are within the reference array, so order1 sampling
    # of the known linear texture agrees exactly with this closed-form value.
    foreground = 100. + .8 * (warped_x - 64.)
    expected = np.mean(np.where(expected_ownership, foreground, 40.), axis=-1)
    error = float(np.max(np.abs(fixed["prediction"] - expected)))
    assert error < 1e-12
    variation = np.ptp(fixed["prediction"], axis=0)
    assert np.max(variation) > 1.
    mixed = (fixed["alpha"][0] > 0.) & (fixed["alpha"][0] < 1.)
    assert np.any(mixed) and np.max(variation[mixed]) > .1
    pixel_index = int(np.flatnonzero(mixed)[0])
    _unchanged(before, aux)
    return dict(passed=True, max_hand_calculation_error=error,
                max_pixel_depth_variation=float(np.max(variation)),
                example_pixel=fixed["roi"][pixel_index].tolist(),
                example_fixed_alpha=float(fixed["alpha"][0, pixel_index]),
                example_predictions=fixed["prediction"][:, pixel_index].tolist(),
                ownership_identical_across_candidates=True, auxiliaries_unchanged=True)


CHECKS = {
    "independent_rotated_camera_fixture": independent_fixture_contract,
    "full_image_equal_color": full_image_equal_color_contract,
    "analytic_slanted_halfplane": analytic_halfplane_contract,
    "fixed_subray_texture_warp": fixed_subray_texture_contract,
}


class IntegrationContracts(unittest.TestCase):
    def test_independent_rotated_camera_fixture(self):
        self.assertTrue(independent_fixture_contract()["passed"])

    def test_full_image_equal_color(self):
        self.assertTrue(full_image_equal_color_contract()["passed"])

    def test_analytic_slanted_halfplane(self):
        self.assertTrue(analytic_halfplane_contract()["passed"])

    def test_fixed_subray_texture_warp(self):
        self.assertTrue(fixed_subray_texture_contract()["passed"])


def run(output_file):
    output = Path(output_file).resolve()
    if output.exists():
        raise FileExistsError("Preserve prior integration evidence; choose a new output filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checks, errors = {}, []
    for name, check in CHECKS.items():
        try:
            checks[name] = check()
        except Exception as error:
            checks[name] = dict(passed=False, error=f"{type(error).__name__}: {error}")
            errors.append(dict(check=name, error=f"{type(error).__name__}: {error}"))
    sources = [Path(__file__), Path(fixtures.__file__), Path(predictor.__file__)]
    result = dict(stage="E0 integration", passed=not errors, checks=checks, errors=errors,
                  elapsed_seconds=time.perf_counter() - started,
                  source_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sources},
                  generated_full_e1=False,
                  scope="Numerical contracts from isolated hand-built and independent-renderer examples")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "e0" / "INTEGRATION.json"))
    args = parser.parse_args()
    raise SystemExit(run(args.output))
