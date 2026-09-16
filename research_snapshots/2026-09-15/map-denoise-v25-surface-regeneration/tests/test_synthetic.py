import os
import sys
import unittest

import numpy as np

os.chdir("/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration")
sys.path.insert(0, os.getcwd())

from src.v25.cameras import unproject_colmap_depth  # noqa: E402
from src.v25.evidence import _masked_zncc, _project, _warp_source  # noqa: E402
from src.v25.synthetic import textured_step_images  # noqa: E402


def gray(image):
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]) / 255.0


class SyntheticSweepTests(unittest.TestCase):
    def test_front_plane_is_localized(self):
        views, images, planes = textured_step_images(128)
        ref, srcs = views[0], views[1:]
        ref_g = gray(images[0])
        src_g = [gray(im) for im in images[1:]]
        yy, xx = np.mgrid[0:128, 0:128]
        pixels = np.column_stack([xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)])
        depths = np.linspace(60.0, 120.0, 31)
        identity = np.eye(4)
        best = np.full((128, 128), np.nan)
        best_s = np.full((128, 128), -1.0)
        for depth in depths:
            xyz = unproject_colmap_depth(ref, pixels, np.full(len(pixels), depth), identity)
            acc = np.zeros((128, 128))
            n = np.zeros((128, 128))
            for view, img in zip(srcs, src_g):
                uv, z = _project(xyz, view.P_phys)
                uv[z <= 0] = np.nan
                warped, valid = _warp_source(img, uv, (128, 128))
                valid &= z.reshape(128, 128) > 0
                score = _masked_zncc(ref_g, warped, valid, 7)
                good = np.isfinite(score)
                acc = np.where(good, acc + score, acc)
                n += good
            mean = np.where(n >= 1, acc / np.maximum(n, 1), np.nan)
            take = np.isfinite(mean) & (mean > best_s)
            best_s[take] = mean[take]
            best[take] = depth
        from scipy.ndimage import uniform_filter

        local_var = uniform_filter(ref_g * ref_g, 7) - uniform_filter(ref_g, 7) ** 2
        textured = local_var > 1e-3
        recovered = best[textured]
        recovered = recovered[np.isfinite(recovered)]
        self.assertGreater(len(recovered), 80)
        near80 = np.mean(np.abs(recovered - 80) < 6)
        self.assertGreater(float(near80), 0.5)
        self.assertLess(float(np.median(np.abs(recovered - 80))), 6.0)


if __name__ == "__main__":
    unittest.main()
