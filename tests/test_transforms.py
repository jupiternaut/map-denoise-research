import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transforms import (
    apply_pose,
    as_matrix44,
    check_homogeneous,
    estimate_patch_frame,
    invert_pose,
    local_mm_to_world,
    local_world_roundtrip,
    world_to_local_mm,
)


class TransformTests(unittest.TestCase):
    def test_roundtrip_and_units(self):
        rng = np.random.default_rng(7)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = 0.4
        k = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        rot = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
        matrix = np.eye(4)
        matrix[:3, :3] = rot
        matrix[:3, 3] = (1.2, -0.4, 3.1)
        points = rng.normal(size=(20, 3))
        check = check_homogeneous(matrix)
        self.assertTrue(check["ok"])
        trip = local_world_roundtrip(points, matrix, atol=1e-12)
        self.assertTrue(trip["ok"])
        world = apply_pose(points, matrix)
        back = apply_pose(world, invert_pose(matrix))
        np.testing.assert_allclose(back, points, atol=1e-12)

    def test_mm_world_roundtrip(self):
        xyz = np.array([
            [1.0, 2.0, 3.0],
            [1.01, 2.0, 3.002],
            [0.99, 2.02, 3.001],
            [1.02, 1.98, 2.999],
        ])
        frame = estimate_patch_frame(xyz)
        local = world_to_local_mm(xyz, frame["origin_m"], frame["basis"])
        world = local_mm_to_world(local, frame["origin_m"], frame["basis"])
        np.testing.assert_allclose(world, xyz, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(frame["basis"]), 1.0, places=12)

    def test_pose_not_applied_twice_helper(self):
        points = np.array([[1.0, 0.0, 0.0]])
        matrix = as_matrix44(np.eye(4))
        matrix[:3, 3] = (10.0, 0, 0)
        once = apply_pose(points, matrix)
        twice = apply_pose(once, matrix)
        self.assertAlmostEqual(once[0, 0], 11.0)
        self.assertAlmostEqual(twice[0, 0], 21.0)
        self.assertGreater(abs(twice[0, 0] - once[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
