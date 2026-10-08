"""Handcrafted numerical checks; never generates registered48-object datasets."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import new_fixtures as f


def rectified_camera(baseline=0.):
    return {"K": np.array([[70., 0., 64.], [0., 70., 64.], [0., 0., 1.]]),
            "R": np.eye(3), "C": np.array([baseline, 0., 0.])}


def handcrafted(mechanism="flat_contrast"):
    return {"mechanism": mechanism, "true_depth": 350., "background_depth": 700.,
            "center": [64., 64.], "half_width": 1.2, "half_height": 8., "angle": 0.,
            "foreground_level": 170., "background_level": 30.,
            "foreground_texture": None, "background_texture": None, "equal_level": 117.25}


class FixtureTests(unittest.TestCase):
    def test_camera_contract(self):
        records = f.cameras()
        self.assertEqual(len(records), 3)
        np.testing.assert_array_equal(records[0]["R"], np.eye(3))
        for sign, record in zip((-1., 1.), records[1:]):
            a, b = sign*.07, sign*.11
            ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            rz = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
            np.testing.assert_allclose(record["R"], ry@rz, atol=1e-15)
            np.testing.assert_array_equal(record["C"], [sign*60, 0, 0])
            np.testing.assert_allclose(record["R"]@record["R"].T, np.eye(3), atol=1e-15)

    def test_homography_matches_independent_direct_3d_projection(self):
        records = f.cameras()
        reference_pixels = np.array([[20.25, 30.75, 1.], [64., 64., 1.], [100., 88.25, 1.]])
        depth = 573.4
        points = np.column_stack(((reference_pixels[:, 0]-64.)*depth/160.,
                                  (reference_pixels[:, 1]-64.)*depth/160., np.full(3, depth)))
        for camera in records:
            camera_points = (points-camera["C"]) @ camera["R"].T
            expected = np.column_stack((160*camera_points[:, 0]/camera_points[:, 2]+64.,
                                        160*camera_points[:, 1]/camera_points[:, 2]+64.))
            hom = f.plane_homography(depth, records[0], camera)
            mapped = reference_pixels @ hom.T
            np.testing.assert_allclose(mapped[:, :2]/mapped[:, 2:3], expected, atol=3e-14)

    def test_equal_control_exact_and_depth_independent(self):
        first = handcrafted("flat_equal")
        second = dict(first, true_depth=681.123, background_depth=985., half_width=3.1)
        a, b = f.render_object(first), f.render_object(second)
        self.assertEqual(a.dtype, np.float64)
        self.assertEqual(a.shape, (3, 128, 128))
        np.testing.assert_array_equal(a, 117.25)
        np.testing.assert_array_equal(a, b)

    def test_independent_midpoint_count_and_disparity(self):
        spec = handcrafted()
        cams = [rectified_camera(), rectified_camera(20.)]
        actual = f.render_object(spec, cams)
        offsets = [-3/7, -2/7, -1/7, 0., 1/7, 2/7, 3/7]
        expected = []
        for x in range(60, 68):
            count = sum(abs(x+offset-64.) <= 1.2 for offset in offsets)
            expected.append(30.+140.*count/7.)
        np.testing.assert_allclose(actual[0, 64, 60:68], expected, atol=3e-14)
        # f*C/Z=70*20/350=4 pixels: source target moves exactly left4.
        np.testing.assert_allclose(actual[1, 60:69, 55:72], actual[0, 60:69, 59:76], atol=1e-12)

    def test_linear_plane_pixel_average_and_warp(self):
        spec = handcrafted("textured_single")
        cams = [rectified_camera(), rectified_camera(20.)]
        def independent_linear(specification, layer, x, y):
            return 60.+.4*x+.2*y
        with patch.object(f, "_radiance", independent_linear):
            images = f.render_object(spec, cams, row_chunk=13)
        yy, xx = np.indices((128, 128))
        np.testing.assert_allclose(images[0], 60+.4*xx+.2*yy, atol=6e-14)
        np.testing.assert_allclose(images[1], 60+.4*(xx+4)+.2*yy, atol=6e-14)

    def test_slanted_rectangle_tiling_invariance(self):
        spec = handcrafted()
        spec.update({"center": [64.35, 63.8], "angle": .17})
        cams = [rectified_camera(), rectified_camera(20.)]
        a = f.render_object(spec, cams, row_chunk=1)
        b = f.render_object(spec, cams, row_chunk=32)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.any((a > 30.) & (a < 170.)))

    def test_bounded_handcrafted_texture(self):
        spec = handcrafted("textured_single")
        spec["foreground_texture"] = {"amplitude": 17., "frequencies": [[.1, 0.], [0., .14], [.1, .1], [-.09, .04]],
                                      "phases": [.1, .4, .8, 1.7]}
        image = f.render_object(spec, [rectified_camera()])
        self.assertGreaterEqual(float(np.min(image)), 153.)
        self.assertLessEqual(float(np.max(image)), 187.)
        self.assertGreater(float(np.std(image)), 1.)

    def test_no_old_or_predictor_dependency(self):
        tree = ast.parse(Path(f.__file__).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertEqual(set(imports), {"__future__", "hashlib", "json", "pathlib", "numpy"})


if __name__ == "__main__":
    unittest.main()
