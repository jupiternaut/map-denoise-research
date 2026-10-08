"""Small synthetic mechanism tests; no fixture truth or experiment scores."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from estimator import estimate_fields


class EstimatorTests(unittest.TestCase):
    def test_input_boundary_excludes_hidden_truth_and_files(self):
        self.assertEqual(list(inspect.signature(estimate_fields).parameters), ["images", "xy"])
        images = np.full((3, 9, 9), 80.0)
        hidden = np.full((3, 9, 9), 1234.0)
        for forbidden_name in ("truth", "labels", "depth", "candidates", "scene", "cameras"):
            with self.assertRaises(TypeError):
                estimate_fields(images, xy=(4, 4), **{forbidden_name: hidden})
        with self.assertRaises((ValueError, TypeError)):
            estimate_fields({"images": images, "truth": hidden}, xy=(4, 4))
        with patch("builtins.open", side_effect=AssertionError("unexpected file read")), \
                patch("numpy.load", side_effect=AssertionError("unexpected array read")):
            result = estimate_fields(images, xy=(4, 4))
        self.assertEqual(result["metadata"]["prototype"], 80.0)

    def test_shape_binary_range_and_input_preservation(self):
        images = np.full((3, 9, 11), 80.0)
        original = images.copy()
        result = estimate_fields(images, xy=(5, 4))
        for key in ("center", "footprint"):
            self.assertEqual(result[key].shape, images.shape)
            self.assertEqual(result[key].dtype, np.dtype("float64"))
            self.assertTrue(np.isin(result[key], (0.0, 1.0)).all())
        self.assertTrue((result["footprint"] <= result["center"]).all())
        np.testing.assert_array_equal(images, original)

    def test_clear_boundary_is_eroded_by_one_pixel(self):
        images = np.full((3, 9, 9), 150.0)
        images[:, 2:7, 2:7] = 80.0
        result = estimate_fields(images, xy=(4, 4))
        expected_center = np.zeros((3, 9, 9))
        expected_center[:, 2:7, 2:7] = 1.0
        expected_footprint = np.zeros((3, 9, 9))
        expected_footprint[:, 3:6, 3:6] = 1.0
        np.testing.assert_array_equal(result["center"], expected_center)
        np.testing.assert_array_equal(result["footprint"], expected_footprint)
        self.assertEqual(result["metadata"]["center_counts"], [25, 25, 25])
        self.assertEqual(result["metadata"]["footprint_counts"], [9, 9, 9])

    def test_constant_image_interior_and_unknown_exterior(self):
        result = estimate_fields(np.full((3, 9, 11), 72.0), xy=(5, 4))
        np.testing.assert_array_equal(result["center"], np.ones((3, 9, 11)))
        expected = np.zeros((3, 9, 11))
        expected[:, 1:-1, 1:-1] = 1.0
        np.testing.assert_array_equal(result["footprint"], expected)

    def test_reference_connectivity_but_no_source_anchor(self):
        images = np.full((3, 11, 11), 150.0)
        images[0, 4:7, 4:7] = 80.0
        images[0, 1, 1] = 80.0  # Disconnected reference appearance match.
        images[1:, 1:4, 1:4] = 80.0  # Source target need not be at (5,5).
        result = estimate_fields(images, xy=(5, 5))
        self.assertEqual(result["center"][0, 1, 1], 0.0)
        self.assertEqual(result["center"][1, 2, 2], 1.0)
        self.assertEqual(result["footprint"][1, 2, 2], 1.0)
        self.assertEqual(result["center"][1, 1, 1], 1.0)
        self.assertEqual(result["footprint"][1, 1, 1], 0.0)
        self.assertEqual(result["center"][1, 5, 5], 0.0)

    def test_fixed_inclusive_threshold(self):
        images = np.full((3, 9, 9), 80.0)
        images[1, 2, 2] = 100.0
        images[1, 2, 3] = 100.000001
        images[2, 2, 2] = 60.0
        images[2, 2, 3] = 59.999999
        result = estimate_fields(images, xy=(4, 4))
        np.testing.assert_array_equal(result["center"][1:, 2, 2], [1, 1])
        np.testing.assert_array_equal(result["center"][1:, 2, 3], [0, 0])
        self.assertEqual(result["metadata"]["intensity_tolerance"], 20.0)

    def test_failed_anchor_and_empty_sources_are_not_repaired(self):
        images = np.full((3, 9, 9), 150.0)
        images[0] = 80.0
        result = estimate_fields(images, xy=(4, 4))
        self.assertEqual(result["center"][1:].sum(), 0.0)
        self.assertEqual(result["footprint"][1:].sum(), 0.0)
        images[0, 4, 4] = 150.0
        result = estimate_fields(images, xy=(4, 4))
        self.assertEqual(result["center"][0].sum(), 0.0)
        self.assertFalse(result["metadata"]["anchor_eligible"])

    def test_invalid_images_and_anchor(self):
        for bad in (np.zeros((2, 9, 9)), np.zeros((3, 9, 9, 3)),
                    np.zeros((3, 2, 9)), np.full((3, 9, 9), np.nan)):
            with self.assertRaises(ValueError):
                estimate_fields(bad, xy=(4, 4))
        for bad_xy in ((4.5, 4), (0, 4), (9, 4), (np.nan, 4), (4,)):
            with self.assertRaises(ValueError):
                estimate_fields(np.zeros((3, 9, 9)), xy=bad_xy)


if __name__ == "__main__":
    unittest.main()
