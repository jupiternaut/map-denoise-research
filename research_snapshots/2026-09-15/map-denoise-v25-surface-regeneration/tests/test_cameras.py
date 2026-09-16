import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.chdir("/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration")
sys.path.insert(0, os.getcwd())

from src.v25.cameras import load_scene, project_matrix, unproject_colmap_depth  # noqa: E402


class CameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = {24: load_scene(24, load_mesh=True), 37: load_scene(37, load_mesh=True)}

    def test_reprojection_matches_v23_order(self):
        s24 = self.scenes[24].provenance
        s37 = self.scenes[37].provenance
        self.assertLess(s24["sparse_reprojection_median_px"], 1.0)
        self.assertLess(s24["sparse_reprojection_p95_px"], 5.0)
        self.assertLess(s37["sparse_reprojection_median_px"], 1.0)
        self.assertLess(s37["sparse_reprojection_p95_px"], 5.0)
        self.assertAlmostEqual(s24["sparse_reprojection_median_px"], 0.356, places=2)
        self.assertAlmostEqual(s37["sparse_reprojection_median_px"], 0.441, places=2)

    def test_roundtrip_unproject_project(self):
        scene = self.scenes[24]
        view = scene.view_by_name("0022.png")
        sample = scene.mesh_vertices_phys[:: max(1, len(scene.mesh_vertices_phys) // 2000)][:200]
        from src.v25.cameras import physical_to_colmap_depth

        depth = physical_to_colmap_depth(view, sample, scene.scale)
        uv, z = project_matrix(sample, view.P_phys)
        keep = (
            np.isfinite(uv).all(1)
            & (z > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < view.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < view.height)
        )
        recovered = unproject_colmap_depth(view, uv[keep], depth[keep], scene.scale)
        err = np.linalg.norm(recovered - sample[keep], axis=1)
        self.assertLess(float(np.median(err)), 0.05)

    def test_image_files_exist(self):
        for scene in self.scenes.values():
            self.assertTrue(Path(scene.image_dir / "0000.png").is_file())
            self.assertEqual(len(scene.views), 49)


if __name__ == "__main__":
    unittest.main()
