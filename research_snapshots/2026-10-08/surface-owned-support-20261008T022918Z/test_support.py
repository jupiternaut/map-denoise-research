import unittest
import numpy as np
from support import center_mask, score_support, support_intervals, intersection, rays, project


class SupportTests(unittest.TestCase):
    def test_masks(self):
        p = np.zeros((9, 9)); p[3:6, 3:6] = 100
        self.assertEqual(center_mask(p, 'connected9').sum(), 9)
        self.assertEqual(center_mask(p, 'center3').sum(), 9)
        self.assertEqual(center_mask(p, 'full9').sum(), 81)

    def test_chain_leak_is_not_segmentation(self):
        p = np.tile(np.arange(9)*10., (9, 1))
        self.assertEqual(center_mask(p, 'connected9').sum(), 81)

    def test_camera_roundtrip(self):
        c = dict(K=np.array([[50., 0, 40], [0, 50., 40], [0, 0, 1]]), R=np.eye(3), C=np.zeros(3))
        xy = np.array([[20., 21.], [42., 48.]])
        uv, z = project(c, 100*rays(c, xy))
        np.testing.assert_allclose(uv, xy)
        np.testing.assert_allclose(z, 100)

    def test_self_and_flat(self):
        im = np.random.default_rng(1).normal(128, 8, (80, 80))
        c = dict(K=np.array([[50., 0, 40], [0, 50., 40], [0, 0, 1]]), R=np.eye(3), C=np.zeros(3))
        for warp in ('translation', 'plane'):
            for mask in ('full9', 'center3', 'connected9'):
                s = score_support(im, im, c, c, [40,40], [99,100,101], warp, mask)
                np.testing.assert_allclose(s['scores'], 1, atol=1e-12)
        s = score_support(np.zeros_like(im), im, c, c, [40,40], [99,100], 'plane', 'full9')
        self.assertTrue(np.isnan(s['scores']).all())

    def test_shared_samples_not_padded_overlap(self):
        g = np.arange(10.)
        a = g == 3; b = g == 5
        self.assertTrue(intersection(support_intervals(g,a), support_intervals(g,b)))
        self.assertEqual(support_intervals(g, a & b), [])

    def test_components_retained(self):
        self.assertEqual(support_intervals(np.arange(10.), np.isin(np.arange(10),[0,1,8,9]), 0), [[0.,1.5],[7.5,9.]])


if __name__ == '__main__':
    unittest.main()
