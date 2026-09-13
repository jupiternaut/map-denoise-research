import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = Path("/home/grf/Documents/Codex/2026-09-11/map-denoise-correlation-v1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OLD))

from adapt import build_adapter, from_operator_mm, to_operator_mm
from generate_synthetics import make_ghost
import operators


class EstimatorIsolationTests(unittest.TestCase):
    def test_signature_has_no_gt(self):
        sig = inspect.signature(operators.estimate)
        self.assertEqual(list(sig.parameters), ["method", "xyz_mm", "frame", "sigma_mm"])

    def test_call_does_not_need_eval_fields(self):
        points, meta, ev = make_ghost(912101, 4.0)
        adapter = build_adapter(points["xyz_world"])
        xyz_mm = to_operator_mm(points["xyz_world"], adapter)
        output, info = operators.estimate("identity", xyz_mm, points["scan_id"], 1.0)
        self.assertEqual(len(output), len(xyz_mm))
        self.assertTrue(np.isfinite(output).all())
        world = from_operator_mm(output, adapter)
        np.testing.assert_allclose(world, points["xyz_world"], atol=1e-12)
        blob = str(info).lower() + str(meta) 
        self.assertNotIn("inverse_extra", blob)
        # evaluation dict stays local to the test, not passed in
        self.assertIn("gt_layer", ev)

    def test_forbidden_keys_absent_from_operator_input(self):
        source = Path(operators.__file__).read_text()
        self.assertNotIn("evaluation/", source)
        self.assertNotIn("gt_layer", inspect.getsource(operators.estimate))


if __name__ == "__main__":
    unittest.main()
