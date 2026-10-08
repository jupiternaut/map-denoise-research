"""Small fixture/E0 contract tests; never generate or evaluate full E1."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import e0
import fixtures


class FixtureContracts(unittest.TestCase):
    def test_design_counts_and_background_pair_target_identity(self):
        worlds = fixtures.world_specs()
        self.assertEqual(len(worlds), 36)
        groups = {}
        for world in worlds:
            key = (world["mechanism"], world["half_width"], world["seed"])
            groups.setdefault(key, []).append(world)
        self.assertEqual(len(groups), 24)
        self.assertEqual(sum(len(group) == 2 for group in groups.values()), 12)
        probe = np.array([[60.1, 61.2], [64., 64.], [67.2, 64.3]])
        for group in groups.values():
            if len(group) == 2:
                first, second = group
                np.testing.assert_array_equal(fixtures.reference_polygon(first), fixtures.reference_polygon(second))
                np.testing.assert_array_equal(fixtures.radiance(first, probe, "foreground"),
                                              fixtures.radiance(second, probe, "foreground"))
                self.assertGreater(float(np.max(np.abs(fixtures.radiance(first, probe, "background")
                                                        - fixtures.radiance(second, probe, "background")))), 0.)

    def test_cameras_match_the_supplied_historical_convention(self):
        records = fixtures.cameras()
        historical = Path("/srv/slam-research/grf/map-denoise/runs/footprint-support-20261008T025757Z/OBS_INPUTS.json")
        supplied = json.loads(historical.read_text())[0]["cameras"]
        for actual, expected in zip(records, supplied):
            for key in ("K", "R", "C"):
                np.testing.assert_allclose(actual[key], expected[key], rtol=0., atol=2e-16)
            np.testing.assert_allclose(actual["R"] @ actual["R"].T, np.eye(3), atol=1e-15)

    def test_equal_color_has_no_image_geometry(self):
        for seed in fixtures.SEEDS:
            world = fixtures.make_world("flat_equal", 1.65, seed)
            images, diagnostic = fixtures.render_world(world, return_diagnostics=True)
            self.assertEqual(images.shape, (3, 128, 128))
            self.assertEqual(images.dtype, np.float64)
            self.assertEqual(float(np.ptp(images)), 0.)
            self.assertGreater(float(np.ptp(diagnostic["source_alpha"])), 0.)
            np.testing.assert_allclose(images, world["equal_level"], rtol=0., atol=3e-14)

    def test_oracle_interface_has_no_target_depth_or_source_ownership(self):
        for mechanism in fixtures.MECHANISMS:
            aux = fixtures.oracle_aux(fixtures.make_world(mechanism, 2.65, 2207))
            self.assertNotIn("true_depth", aux)
            self.assertNotIn("target_depth", aux)
            self.assertNotIn("source_alpha", aux)
            self.assertNotIn("seed", aux)
            self.assertNotIn("phases", aux)
            self.assertNotIn("mechanism", aux)
            self.assertFalse(bool(aux["sigma_valid"]))
            self.assertTrue(bool(aux["auxiliary_valid"]))
            self.assertEqual(aux["foreground"].shape, (128, 128))
            self.assertEqual(aux["background"].shape, (128, 128))
            if mechanism == "textured_single":
                self.assertEqual(str(aux["mode"]), "single")
                self.assertTrue(np.all(aux["mask"] == 1.))
                self.assertEqual(aux["polygon"].shape, (0, 2))
            else:
                self.assertEqual(aux["polygon"].shape, (4, 2))

    def test_paired_images_change_only_where_background_contributes(self):
        first = fixtures.make_world("textured_boundary", 2.65, 1103, 0)
        second = fixtures.make_world("textured_boundary", 2.65, 1103, 1)
        image_a, diagnostic = fixtures.render_world(first, return_diagnostics=True)
        image_b = fixtures.render_world(second)
        fully_foreground = diagnostic["source_alpha"] == 1.
        np.testing.assert_array_equal(image_a[fully_foreground], image_b[fully_foreground])
        self.assertGreater(float(np.mean(np.abs(image_a - image_b))), 1.)

    def test_mini_serialization_separates_observed_and_truth_and_refuses_overwrite(self):
        # One tiny unit world only; the full E1 remains behind main's source lock.
        world = fixtures.make_world("flat_equal", 1.65, 1103)
        with tempfile.TemporaryDirectory(prefix="fixture-unit-", dir=Path(__file__).resolve().parent) as directory:
            with patch.object(fixtures, "world_specs", return_value=[world]):
                summary = fixtures.create_e1(directory)
                self.assertEqual(summary["worlds"], 1)
                root = Path(directory)
                inputs = json.loads((root / "observed" / "inputs.json").read_text())
                self.assertEqual(set(inputs[0]), {"id", "cameras", "image_file", "sha256"})
                with np.load(inputs[0]["image_file"], allow_pickle=False) as record:
                    self.assertEqual(record.files, ["images"])
                self.assertEqual(fixtures.sha256(inputs[0]["image_file"]), inputs[0]["sha256"])
                with self.assertRaises(FileExistsError):
                    fixtures.create_e1(directory)


class AnalyticE0(unittest.TestCase):
    def test_analytic_area_invariance_and_complement(self):
        normal, threshold = np.array([1., .37]), .14
        area = e0.halfplane_pixel_area(normal, threshold)
        translated = e0.halfplane_pixel_area(normal, threshold + np.array([64., 31.]) @ normal,
                                             (64., 31.))
        self.assertAlmostEqual(area, translated, places=13)
        self.assertAlmostEqual(area + e0.halfplane_pixel_area(-normal, -threshold), 1., places=14)
        self.assertAlmostEqual(area, e0.midpoint_area(normal, threshold, 512), places=4)

    def test_geometry_and_fixed_texture_controls(self):
        self.assertTrue(e0.mixed_color_depth_checks()["passed"])
        self.assertTrue(e0.fixed_coverage_texture_check()["passed"])


if __name__ == "__main__":
    unittest.main()
