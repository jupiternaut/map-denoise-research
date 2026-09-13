import unittest
import numpy as np
from geometry import graph_plane_distances, tls_plane, fit_planes, plane_errors, spatial_stripe_holdout


class GeometryTests(unittest.TestCase):
    def test_physical_units_undo_normalization(self):
        xy = np.array([[0., 0.], [1., 1.]])
        beta = np.array([10., 600., 800.])
        predicted = 10.+xy@beta[1:]
        got = graph_plane_distances(predicted+13., xy, beta, np.zeros(2, int), [100., 100.])
        np.testing.assert_allclose(got['height_mm'], 13.)
        np.testing.assert_allclose(got['orthogonal_mm'], 13./np.sqrt(101.))
        self.assertAlmostEqual(got['axis_condition'], np.sqrt(101.))

    def test_two_intercepts_same_distance_metric(self):
        got = graph_plane_distances(np.array([3., 14.]), np.zeros((2, 2)),
                                    np.array([1., 10., 0., 0.]), [0, 1], [1., 1.])
        np.testing.assert_array_equal(got['orthogonal_mm'], [2., 4.])

    def test_tls_rotation_equivariance(self):
        rng = np.random.default_rng(912701)
        xy = rng.normal(size=(100, 2))
        points = np.c_[xy, xy[:, 0]*2.+1.]
        model = fit_planes(points, 1)
        ortho, _, _, _ = plane_errors(points, model, [0., 0., 1.])
        self.assertLess(ortho.max(), 1e-12)
        q, _ = np.linalg.qr(rng.normal(size=(3,3)))
        rotated = points@q+np.array([7.,2.,-1.])
        transformed_model = fit_planes(rotated,1)
        rotated_orth, _, _, _ = plane_errors(rotated,transformed_model,np.array([0.,0.,1.])@q)
        np.testing.assert_allclose(rotated_orth,ortho,atol=1e-12)
        self.assertAlmostEqual(abs(float((model['normals'][0]@q)@transformed_model['normals'][0])),1.)

    def test_axis_parallel_infinite_not_clipped(self):
        model = dict(centers=np.zeros((1, 3)), normals=np.array([[1., 0., 0.]]))
        ortho, delta, cosine, _ = plane_errors(np.array([[2., 0., 0.]]), model, [0., 0., 1.])
        self.assertEqual(ortho[0], 2.)
        self.assertTrue(np.isinf(delta[0]))
        self.assertEqual(cosine[0], 0.)

    def test_fit_does_not_use_or_mutate_heldout(self):
        rng = np.random.default_rng(912702)
        points = rng.normal(size=(60, 3)); points[:, 2] *= .01
        before = points.copy()
        model = fit_planes(points, 2)
        frozen = model['normals'].copy()
        plane_errors(np.ones((20, 3))*1e6, model, [0., 0., 1.])
        np.testing.assert_array_equal(points, before)
        np.testing.assert_array_equal(model['normals'], frozen)

    def test_degenerate_rejected(self):
        with self.assertRaises(ValueError):
            tls_plane(np.zeros((20, 3)))

    def test_scan_stripes_are_balanced_and_permutation_equivariant(self):
        x = np.arange(80.)
        points = np.vstack((np.c_[x,x*0,x*0], np.c_[x,x*0+2.,x*0]))
        scans = np.repeat([1,2],80)
        test = spatial_stripe_holdout(points,scans)
        self.assertEqual(test[:80].sum(),20)
        self.assertEqual(test[80:].sum(),20)
        perm = np.random.default_rng(913).permutation(len(points))
        got = spatial_stripe_holdout(points[perm],scans[perm])
        np.testing.assert_array_equal(got,test[perm])


if __name__ == '__main__':
    unittest.main()
