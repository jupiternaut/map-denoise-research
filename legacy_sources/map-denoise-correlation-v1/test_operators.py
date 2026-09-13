"""Small deterministic adapter tests; no dataset or evaluator inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

import operators


def direct_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    old_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_flag
    return module


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.warmup_info = operators.warmup()
        cls.legacy = direct_module(operators.LEGACY_SOURCE, "_test_direct_frozen_legacy")
        cls.fast = direct_module(operators.FAST_SOURCE, "_test_direct_frozen_fast")
        rng = np.random.default_rng(911073)
        cls.frame = np.repeat(np.arange(8), 24)
        xy = rng.uniform(-40, 40, (len(cls.frame), 2))
        z = 6.0 * rng.integers(0, 2, len(cls.frame))
        z += rng.normal(0, 1.5, 8)[cls.frame] + rng.normal(0, 1.0, len(cls.frame))
        cls.xyz = np.column_stack((xy, z))

    def test_six_methods_match_direct_frozen_calls_and_preserve_inputs(self):
        self.assertEqual(len(operators.METHODS), 6)
        xyz_before, frame_before = self.xyz.copy(), self.frame.copy()
        for method in operators.METHODS:
            with self.subTest(method=method):
                inp = self.legacy.Input(self.xyz.copy(), self.frame.copy(), np.zeros(len(self.xyz), int), 1.0, 8.0)
                backend = self.fast if method == "fast" else self.legacy
                backend_method = "scalar_profile_hard" if method == "fast" else method
                expected, bias, details = backend.estimate(inp, backend_method)
                actual, info = operators.estimate(method, self.xyz, self.frame, 1.0)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
                np.testing.assert_allclose(info["bias_mm"], bias, rtol=1e-12, atol=1e-10)
                self.assertEqual(info["backend_info"], details)
                self.assertEqual(actual.shape, self.xyz.shape)
                self.assertTrue(np.isfinite(actual).all())
                self.assertFalse(np.shares_memory(actual, self.xyz))
                np.testing.assert_array_equal(self.xyz, xyz_before)
                np.testing.assert_array_equal(self.frame, frame_before)
                json.dumps(info, allow_nan=False)

    def test_readonly_arrays_and_sparse_frame_ids(self):
        xyz = self.xyz.copy()
        frame = 101 + self.frame * 7
        xyz.flags.writeable = False
        frame.flags.writeable = False
        for method in ("identity", "joint_forced", "fast"):
            with self.subTest(method=method):
                expected, _ = operators.estimate(method, self.xyz, self.frame, 1.25)
                actual, info = operators.estimate(method, xyz, frame, 1.25)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(info["frame_ids"], np.unique(frame).tolist())
                self.assertEqual(info["bias_bound_mm"], 10.0)

    def test_invalid_input_and_unknown_method(self):
        with self.assertRaises(KeyError):
            operators.estimate("not_a_method", self.xyz, self.frame, 1.0)
        invalid = (
            (np.empty((0, 3)), np.empty(0, int), 1.0),
            (self.xyz[:, :2], self.frame, 1.0),
            (self.xyz * np.nan, self.frame, 1.0),
            (self.xyz, self.frame[:-1], 1.0),
            (self.xyz, self.frame.astype(float), 1.0),
            (self.xyz, self.frame, 0.0),
            (self.xyz, self.frame, float("nan")),
            (self.xyz, self.frame, [1.0]),
        )
        for xyz, frame, sigma in invalid:
            with self.subTest(shape=xyz.shape, sigma=sigma):
                with self.assertRaises(ValueError):
                    operators.estimate("identity", xyz, frame, sigma)

    def test_hashes_and_import_prewarm(self):
        hashes = operators.source_hashes()
        self.assertEqual(len(hashes), 3)
        for name, expected in hashes.items():
            self.assertTrue(Path(name).is_absolute())
            self.assertEqual(expected, hashlib.sha256(Path(name).read_bytes()).hexdigest())
        self.assertIsNotNone(self.warmup_info["open3d_version"])
        self.assertGreaterEqual(self.warmup_info["open3d_import_s"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
