import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extract_patches import overlap_boxes


class ExtractBoxTests(unittest.TestCase):
    def test_same_axis_fallback_stays_on_wall(self):
        rng = np.random.default_rng(12)
        y = rng.uniform(-4.0, 4.0, 5000)
        z = rng.uniform(0.0, 3.0, 5000)
        x = rng.normal(0.0, 0.02, 5000)
        xyz = np.column_stack([x, y, z])
        boxes, extras = overlap_boxes(xyz, n=3)
        self.assertEqual(len(boxes), 3)
        self.assertGreaterEqual(len(extras), 1)
        self.assertEqual(boxes[1]["role"], "same_axis_second_slab")
        wall = boxes[0]
        junction = boxes[1]
        self.assertLess(wall["hi"][0] - wall["lo"][0], 0.4)
        self.assertLess(junction["hi"][1] - junction["lo"][1], 3.3)
        self.assertGreater(junction["hi"][1] - junction["lo"][1], 0.7)
        # Second slab must sit on the same thin wall, not a room-scale XY box.
        self.assertLess(junction["hi"][0] - junction["lo"][0], 0.4)
        n_j = int(np.sum(np.all((xyz >= junction["lo"]) & (xyz <= junction["hi"]), axis=1)))
        self.assertGreaterEqual(n_j, 300)

    def test_orthogonal_peak_uses_crossing(self):
        rng = np.random.default_rng(13)
        wall_n = 3500
        cross_n = 1800
        wall = np.column_stack(
            (
                rng.normal(0.0, 0.02, wall_n),
                rng.uniform(-4.0, 4.0, wall_n),
                rng.uniform(0.0, 3.0, wall_n),
            )
        )
        cross = np.column_stack(
            (
                rng.uniform(-2.0, 2.0, cross_n),
                rng.normal(3.5, 0.02, cross_n),
                rng.uniform(0.0, 3.0, cross_n),
            )
        )
        xyz = np.concatenate([wall, cross])
        boxes, extras = overlap_boxes(xyz, n=3)
        self.assertEqual(len(boxes), 3)
        self.assertEqual(boxes[1]["role"], "orthogonal_crossing")
        self.assertEqual(extras, [])


if __name__ == "__main__":
    unittest.main()
