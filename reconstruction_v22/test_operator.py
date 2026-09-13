"""Small analytic/synthetic correctness checks, not real-data tuning."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("v22_reconstruction_operator", Path(__file__).with_name("operator.py"))
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


def curved_sample(seed=17, n=256, noise=0.):
    rng = np.random.default_rng(seed)
    uv = rng.uniform(-1., 1., (n, 2))
    height = .10*uv[:, 0]**2 + .06*uv[:, 1]**2 + .02*uv[:, 0]*uv[:, 1]
    return np.column_stack((uv, height + rng.normal(0., noise, n)))


class OperatorTests(unittest.TestCase):
    def test_input_validation(self):
        with self.assertRaises(ValueError):
            operator.construct(np.zeros((8, 3)))
        with self.assertRaises(ValueError):
            operator.construct(np.full((32, 3), np.nan))

    def test_plane_consistency_and_leave_self_out(self):
        rng = np.random.default_rng(3)
        uv = rng.uniform(-1., 1., (192, 2))
        points = np.column_stack((uv, .3*uv[:, 0]-.2*uv[:, 1]+.7))
        outputs, d = operator.construct(points)
        for output in outputs.values():
            np.testing.assert_allclose(output, points, atol=2e-12, rtol=0.)
        for i, row in enumerate(d["neighbour_ids"]):
            self.assertNotIn(i, row)

    def test_rigid_motion_equivariance(self):
        points = curved_sample(noise=.008)
        rng = np.random.default_rng(25)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        shift = np.array([2.3, -4.1, .8])
        original, _ = operator.construct(points)
        moved, _ = operator.construct(points @ q.T + shift)
        for name in original:
            np.testing.assert_allclose(moved[name], original[name] @ q.T + shift, atol=2e-10, rtol=1e-10)

    def test_shapes_finite_and_no_tangent_update(self):
        points = curved_sample(noise=.01)
        outputs, d = operator.construct(points)
        for name, output in outputs.items():
            self.assertEqual(output.shape, points.shape)
            self.assertTrue(np.isfinite(output).all())
            normal = d["plane64"]["normal"] if name == "local_plane64" else d["quadratics"][64]["normal"]
            delta = output-points
            tangential = delta-np.sum(delta*normal, axis=1)[:, None]*normal
            self.assertLess(np.max(np.abs(tangential)), 2e-15)
        self.assertTrue(np.all((d["consensus"]["alpha"] >= 0.) & (d["consensus"]["alpha"] <= 1.)))

    def test_curved_noise_smoke(self):
        points = curved_sample(noise=.01)
        outputs, _ = operator.construct(points)
        def residual_rms(p):
            surface = .10*p[:, 0]**2 + .06*p[:, 1]**2 + .02*p[:, 0]*p[:, 1]
            return np.sqrt(np.mean((p[:, 2]-surface)**2))
        # A single known smooth generative family smoke test, not evidence of
        # arbitrary-geometry generalization or a tuning objective.
        self.assertLess(residual_rms(outputs["quadratic64"]), residual_rms(points))
        self.assertLess(residual_rms(outputs["multiscale_consensus"]), residual_rms(points))

    def test_duplicate_coordinates_are_finite(self):
        points = np.repeat(curved_sample(n=20), 2, axis=0)
        outputs, d = operator.construct(points)
        self.assertTrue(all(np.isfinite(p).all() for p in outputs.values()))
        for i, row in enumerate(d["neighbour_ids"]):
            self.assertNotIn(i, row)
        identical = np.ones((16, 3))
        outputs, _ = operator.construct(identical)
        for output in outputs.values():
            np.testing.assert_array_equal(output, identical)


if __name__ == "__main__":
    unittest.main()
