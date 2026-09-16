import inspect
import os
import sys
import unittest

os.chdir("/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration")
sys.path.insert(0, os.getcwd())

from src.v25 import regenerate  # noqa: E402
from src.v25.paths import EVAL_ONLY  # noqa: E402


class IsolationTests(unittest.TestCase):
    def test_constructor_signatures_have_no_gt(self):
        forbidden = ("laser", "stl", "obs", "mask", "gt", "reference")
        for name, fn in regenerate.ARMS.items():
            source = inspect.getsource(fn)
            args = inspect.signature(fn).parameters
            for key in args:
                self.assertTrue(all(tok not in key.lower() for tok in forbidden), f"{name} arg {key}")
            self.assertNotIn("EVAL_ONLY", source)
            self.assertNotIn("stl024", source)
            self.assertNotIn("stl037", source)

    def test_eval_paths_exist_but_are_not_imported_by_regenerate(self):
        source = inspect.getsource(regenerate)
        self.assertNotIn("evaluation.metrics", source)
        self.assertNotIn("stl024_total", source)
        for spec in EVAL_ONLY.values():
            self.assertTrue(spec["laser"].is_file())


if __name__ == "__main__":
    unittest.main()
