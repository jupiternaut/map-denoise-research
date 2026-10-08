"""Mechanism tests only; no support scores, decisions, or evaluation run."""

import json
import unittest

import numpy as np

from oracle import (HISTORICAL_RENDERER, _appearance_key, _cached_components,
                    _cached_fields, _renderer, ownership_fields, sample_fraction)


def explicit_four_by_nine(scene, camera, uv, with_numerator=False):
    """Independent composition of 4 pixels × 9 original renderer calls."""
    uv = np.asarray(uv)
    lo = np.floor(uv)
    fraction = uv - lo
    out = np.zeros(uv.shape[:-1])
    numerator = np.zeros(uv.shape[:-1])
    for by in (0, 1):
        for bx in (0, 1):
            wx = fraction[..., 0] if bx else 1 - fraction[..., 0]
            wy = fraction[..., 1] if by else 1 - fraction[..., 1]
            pixel = lo + np.array([bx, by])
            for oy in (-1 / 3, 0., 1 / 3):
                for ox in (-1 / 3, 0., 1 / 3):
                    hit = _renderer().intersect(
                        scene, camera, pixel + np.array([ox, oy])
                    )
                    owned = hit["owner"] == scene["target_id"]
                    out += wx * wy * owned / 9
                    numerator += wx * wy * np.where(owned, hit["value"], 0.) / 9
    return (out, numerator) if with_numerator else out


class FullFootprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.loads(
            (HISTORICAL_RENDERER.parent / "FIXTURES.json").read_text()
        )

    def test_matches_explicit_36_rays_all_30_fixtures_and_cameras(self):
        rng = np.random.default_rng(818)
        offsets = rng.uniform(-4.5, 4.5, size=(3, 11, 2))
        offsets[0, :3] = [[0, 0], [1, 0], [-1, 2]]
        renderer = _renderer()
        for meta in self.fixtures:
            fields = ownership_fields(meta)
            self.assertEqual(fields["center"].shape, (3, 128, 128))
            self.assertEqual(fields["area"].shape, (3, 128, 128))
            self.assertEqual(fields["numerator"].shape, (3, 128, 128))
            self.assertTrue(np.all((fields["area"] >= 0) & (fields["area"] <= 1)))
            # The historical fixture is an independent stored image generated
            # by render(), which averages values before component separation.
            numerator, background = _cached_components(_appearance_key(meta), 128, 128)
            with np.load(HISTORICAL_RENDERER.parent / "fixtures" / meta["arrays"]) as fixture:
                image = fixture["images"]
            np.testing.assert_allclose(numerator + background, image,
                                       atol=1e-12, rtol=0)
            np.testing.assert_array_equal(fields["numerator"], numerator)
            for i, raw_camera in enumerate(meta["actual_cameras"]):
                camera = renderer.deserialize_cam(raw_camera)
                center = renderer.project(camera, np.array([0., 0., 600.]))
                uv = center + offsets
                actual = sample_fraction(fields["area"][i], uv)
                expected, expected_numerator = explicit_four_by_nine(
                    meta["scene"], camera, uv, with_numerator=True
                )
                with self.subTest(fixture=meta["id"], camera=i):
                    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
                    np.testing.assert_allclose(
                        sample_fraction(fields["numerator"][i], uv),
                        expected_numerator, atol=1e-12, rtol=0
                    )
                # Center labels use the original scene's physical first hit.
                yy, xx = np.mgrid[59:70, 59:70]
                integer_uv = np.stack((xx, yy), axis=-1)
                hit = renderer.intersect(meta["scene"], camera, integer_uv)
                np.testing.assert_array_equal(
                    fields["center"][i, 59:70, 59:70],
                    hit["owner"] == meta["scene"]["target_id"]
                )

    def test_integer_center_and_area_differ_at_source_boundary(self):
        meta = next(f for f in self.fixtures if f["name"] == "ring9_flat")
        fields = ownership_fields(meta)
        area, center = fields["area"], fields["center"]
        mixed = (area > 0) & (area < 1)
        self.assertGreater(int(mixed[1:].sum()), 0)
        self.assertTrue(np.any(center[mixed] != area[mixed]))
        for i in (1, 2):
            y, x = np.argwhere(mixed[i])[0]
            camera = _renderer().deserialize_cam(meta["actual_cameras"][i])
            expected = explicit_four_by_nine(meta["scene"], camera, [x, y])
            np.testing.assert_allclose(sample_fraction(area[i], [x, y]), expected,
                                       atol=1e-14, rtol=0)

    def test_cache_reuses_geometry_across_background_and_seed(self):
        a = next(f for f in self.fixtures if f["id"] == "ring9_flat_seed11")
        b = next(f for f in self.fixtures if f["id"] == "ring9_textured_seed47")
        fa, fb = ownership_fields(a), ownership_fields(b)
        self.assertIs(fa["area"], fb["area"])
        self.assertIs(fa["center"], fb["center"])
        self.assertFalse(fa["area"].flags.writeable)
        self.assertFalse(fa["center"].flags.writeable)
        self.assertFalse(fa["numerator"].flags.writeable)
        self.assertGreater(np.max(np.abs(fa["numerator"] - fb["numerator"])), 0)
        self.assertGreater(_cached_fields.cache_info().hits, 0)

    def test_batch_scalar_empty_and_invalid_domain(self):
        meta = self.fixtures[0]
        fields = ownership_fields(meta)
        uv = np.array([[[63.2, 64.8], [64., 64.]], [[0., 0.], [127., 127.]]])
        values = sample_fraction(fields, uv)
        self.assertEqual(values.shape, (3, 2, 2))
        np.testing.assert_allclose(values, 1., atol=1e-14, rtol=0)
        self.assertEqual(sample_fraction(fields["area"][0], [64., 64.]).shape, ())
        self.assertEqual(sample_fraction(fields, np.empty((0, 2))).shape, (3, 0))
        invalid = [[-.01, 64], [128, 64], [64, 128], [np.nan, 64], [64, np.inf]]
        self.assertTrue(np.isnan(sample_fraction(fields, invalid)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
