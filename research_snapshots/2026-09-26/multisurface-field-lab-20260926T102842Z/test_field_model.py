"""Mechanism tests; all synthetic truth stays in this test module."""

import inspect
import unittest

import numpy as np

import field_model
from field_model import fit_local_fields, quadratic_basis


def points(seed=0, count=160):
    return np.random.default_rng(seed).uniform(4., 28., size=(count, 2))


def evidence(truth, offsets=(-1., 0., 1.)):
    offsets = np.asarray(offsets)
    z = truth[:, None] + offsets
    cost = np.broadcast_to(10. * offsets**2, z.shape).copy()
    return z, cost, np.ones_like(z, dtype=bool)


class FieldTests(unittest.TestCase):
    def test_inverse_depth_tilted_plane(self):
        uv = points()
        basis = quadratic_basis((uv-16.)/32.)
        rho = basis @ np.array([.7, 1.4, -.9, 0., 0., 0.])
        truth = 1. / (1./1000. + rho/1000.**2)
        z, cost, valid = evidence(truth)
        result = fit_local_fields(uv, z, cost, valid, input_depth=truth+1.)
        self.assertTrue(result.supported.all())
        self.assertLess(np.max(np.abs(result.depth[:, 0]-truth)), 1e-5)
        self.assertEqual(np.asarray(result.patches[0]['coefficients']).shape, (1, 6))
        np.testing.assert_allclose(result.responsibility.sum(axis=1), 1.)

    def test_two_parallel_layers_and_unequal_counts(self):
        uv = points(8, 200)
        truth = np.full(200, 1000.)
        truth[::5] = 1004.
        z, cost, valid = evidence(truth)
        result = fit_local_fields(uv, z, cost, valid, input_depth=truth+.4, n_layers=2)
        sorted_depth = np.sort(result.depth, axis=1)
        self.assertLess(np.max(np.abs(sorted_depth[:, 0]-1000.)), .02)
        self.assertLess(np.max(np.abs(sorted_depth[:, 1]-1004.)), .02)
        np.testing.assert_allclose(sorted(result.patches[0]['layer_prior']), [.2, .8], atol=.002)
        selected = result.depth[np.arange(len(uv)), result.labels]
        self.assertLess(np.max(np.abs(selected-truth)), .02)

    def test_two_competing_candidates_per_row(self):
        uv = points(33)
        z = np.broadcast_to([999., 1003.], (len(uv), 2)).copy()
        result = fit_local_fields(uv, z, np.zeros_like(z), np.ones_like(z, bool),
                                  input_depth=np.full(len(uv), 1001.), n_layers=2)
        np.testing.assert_allclose(np.sort(result.depth, axis=1), z, atol=.01)

    def test_curved_inverse_depth_surface(self):
        uv = points(4)
        basis = quadratic_basis((uv-16.)/32.)
        rho = basis @ np.array([-.2, .4, -.6, 2.5, -1.8, 1.5])
        truth = 1. / (1./900. + rho/900.**2)
        z, cost, valid = evidence(truth)
        result = fit_local_fields(uv, z, cost, valid, input_depth=truth+.2)
        self.assertLess(np.max(np.abs(result.depth[:, 0]-truth)), .003)

    def test_row_permutation_invariance(self):
        uv = points(15)
        truth = 1000. + np.where(np.arange(len(uv)) % 3 == 0, 4., 0.)
        z, cost, valid = evidence(truth)
        original = truth + .5
        order = np.random.default_rng(17).permutation(len(uv))
        inverse = np.argsort(order)
        a = fit_local_fields(uv, z, cost, valid, input_depth=original, n_layers=2)
        b = fit_local_fields(uv[order], z[order], cost[order], valid[order],
                             input_depth=original[order], n_layers=2)
        # Labels are arbitrary within a patch; compare unordered layer geometry.
        np.testing.assert_allclose(np.sort(a.depth, axis=1),
                                   np.sort(b.depth[inverse], axis=1), atol=1e-8)

    def test_invalid_evidence_cannot_be_good_evidence(self):
        uv = points(21)
        truth = np.full(len(uv), 1000.)
        z, cost, valid = evidence(truth)
        original = np.full(len(uv), 1002.)
        valid[:20] = False
        z[:20] = np.nan; cost[:20] = -1e100
        views = np.full(z.shape, 2)
        views[20:40] = 1
        result = fit_local_fields(uv, z, cost, valid, input_depth=original,
                                  view_count=views)
        np.testing.assert_array_equal(result.depth[:40, 0], original[:40])
        self.assertFalse(result.supported[:40].any())
        self.assertTrue(result.supported[40:].all())
        np.testing.assert_allclose(result.depth[40:, 0], 1000., atol=1e-9)
        self.assertTrue((result.labels[:40] == -1).all())

    def test_insufficient_patch_falls_back_to_input(self):
        uv = points(count=11)
        original = np.full(11, 1002.)
        z, cost, valid = evidence(np.full(11, 1000.))
        result = fit_local_fields(uv, z, cost, valid, input_depth=original, n_layers=2)
        np.testing.assert_array_equal(result.depth, np.repeat(original[:, None], 2, axis=1))
        self.assertFalse(result.supported.any())

    def test_halo_context_but_core_ownership(self):
        uv = np.array([[x, y] for x in np.linspace(20., 44., 25)
                       for y in np.linspace(6., 25., 12)])
        truth = 1000. + .03*uv[:, 0]
        z, cost, valid = evidence(truth)
        result = fit_local_fields(uv, z, cost, valid, input_depth=truth+.5)
        self.assertEqual(len(result.patches), 2)
        self.assertTrue(result.supported.all())
        self.assertTrue(np.all(result.patch_index[uv[:, 0] < 32.] == 0))
        self.assertTrue(np.all(result.patch_index[uv[:, 0] >= 32.] == 1))
        self.assertTrue(all(p['n_fit'] > p['n_core'] for p in result.patches))

    def test_shared_coefficients_reproduce_predicted_geometry(self):
        uv = points(71)
        truth = 1000. + .01*uv[:, 0] + .02*uv[:, 1]
        z, cost, valid = evidence(truth)
        result = fit_local_fields(uv, z, cost, valid, input_depth=truth+.5)
        patch = result.patches[0]
        basis = quadratic_basis((uv-np.asarray(patch['centre_uv']))/32.)
        rho = basis @ np.asarray(patch['coefficients']).T
        reconstructed = 1./(1./patch['z_center_mm'] + rho/patch['z_center_mm']**2)
        np.testing.assert_allclose(result.depth, reconstructed, rtol=0, atol=0)

    def test_no_reference_or_file_interface(self):
        parameters = inspect.signature(fit_local_fields).parameters
        self.assertFalse({'gt', 'truth', 'reference', 'support', 'path', 'condition'} & set(parameters))
        source = inspect.getsource(field_model)
        for prohibited in ('np.load(', 'open(', 'read_points(', 'import os', 'import pathlib'):
            self.assertNotIn(prohibited, source)


if __name__ == '__main__':
    unittest.main()
