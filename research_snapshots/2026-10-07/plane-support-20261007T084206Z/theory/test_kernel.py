"""Seeded CPU checks of conditional bounds; no real observations or GT input."""
import itertools
import math
import sys
import unittest

sys.dont_write_bytecode = True
import numpy as np

from kernel import (anchor_error_from_depth, certain_possible_support,
                    optical_ray, pixel_ray_error, plane_depth_interval,
                    select_candidate, support_scores)


def ball(rng, radius):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v) * radius * rng.random() ** (1 / 3)


def perturbed_normal(rng, n, theta):
    v = rng.normal(size=3)
    v -= (v @ n) * n
    v /= np.linalg.norm(v)
    angle = rng.uniform(-theta, theta)
    return n * math.cos(angle) + v * math.sin(angle)


def interval(lo, hi, radius=None):
    return {"status": "ok", "low": lo, "high": hi,
            "radius": (hi - lo) / 2 if radius is None else radius}


class KernelTests(unittest.TestCase):
    def test_optical_z_not_range(self):
        K = [[500., 0, 320.], [0, 400., 240.], [0, 0, 1.]]
        r = optical_ray(K, [620., 400.])
        np.testing.assert_allclose(r, [0.6, 0.4, 1.])
        self.assertNotAlmostEqual(np.linalg.norm(r), 1.)
        got = plane_depth_interval([0, 0, 700], [0, 0, 7], r,
                                   normal_angle_rad=0.)
        self.assertEqual(got["status"], "ok")
        self.assertAlmostEqual(got["center"], 700.)
        self.assertAlmostEqual(got["radius"], 0.)
        self.assertAlmostEqual(pixel_ray_error(K, .25), .25 / 400)

    def test_bounded_joint_monte_carlo(self):
        rng = np.random.default_rng(20261007)
        checked = skipped = 0
        worst_ratio = 0.
        for _ in range(12000):
            r = np.r_[rng.uniform(-.8, .8, 2), 1.]
            n = rng.normal(size=3)
            n /= np.linalg.norm(n)
            ray_j = r + np.r_[rng.uniform(-.06, .06, 2), 0.]
            C = rng.uniform(-1000, 1000, 3)
            X = C + rng.uniform(100, 900) * ray_j
            theta = rng.uniform(0., .12)
            eX, eC, er, kappa = .4, .15, .001, .25
            got = plane_depth_interval(X, n, r, camera_center=C,
                                       normal_angle_rad=theta, anchor_error=eX,
                                       camera_error=eC, ray_error=er,
                                       residual_error=kappa)
            if got["status"] != "ok":
                skipped += 1
                continue
            nt = perturbed_normal(rng, n, theta)
            # Correlated directions are allowed, not assumed independent.
            u = ball(rng, 1.)
            Xt, Ct = X + eX * u, C - eC * u
            rt = r + er * u
            residual = rng.uniform(-kappa, kappa)
            zt = (nt @ (Xt - Ct) + residual) / (nt @ rt)
            if zt < 0:
                skipped += 1
                continue
            self.assertLessEqual(got["low"] - 1e-9, zt)
            self.assertGreaterEqual(got["high"] + 1e-9, zt)
            ratio = abs(zt - got["center"]) / max(got["radius"], 1e-15)
            worst_ratio = max(worst_ratio, ratio)
            if got["chord_radius"] is not None:
                self.assertLessEqual(got["radius"], got["chord_radius"] + 1e-8)
            checked += 1
        self.assertGreater(checked, 10000)
        print(f"bounded_joint_draws={checked}; uninformative_or_negative={skipped}; "
              f"max_error_over_radius={worst_ratio:.8f}")

    def test_common_camera_translation_and_rigid_invariance(self):
        r = np.array([.2, -.1, 1.])
        n = np.array([.3, .1, .9])
        C = np.array([300., -20., 1000.])
        X = C + np.array([170., -65., 700.])
        kwargs = dict(normal_angle_rad=.03, relative_anchor_error=.5,
                      ray_error=.001, residual_error=.2)
        a = plane_depth_interval(X, n, r, camera_center=C, **kwargs)
        alpha = .7
        Q = np.array([[math.cos(alpha), 0, math.sin(alpha)], [0, 1, 0],
                      [-math.sin(alpha), 0, math.cos(alpha)]])
        t = np.array([.1e6, -3.1e4, .7e5])
        b = plane_depth_interval(Q @ X + t, Q @ n, Q @ r,
                                 camera_center=Q @ C + t, **kwargs)
        for key in ["center", "radius", "low", "high", "denominator_margin"]:
            self.assertAlmostEqual(a[key], b[key], places=8)

    def test_joint_cancellation_option(self):
        rng = np.random.default_rng(45)
        r = np.array([.2, .1, 1.])
        n = np.array([0., 0., 1.])
        A = np.array([141., 73., 700.])
        got = plane_depth_interval(A, n, r, normal_angle_rad=.03,
                                   relative_anchor_error=.7, ray_error=.001,
                                   joint_residual_error=0.)
        for _ in range(300):
            dr = ball(rng, .001)
            dA = got["center"] * dr  # Exactly justified joint cancellation.
            nt = perturbed_normal(rng, n, .03)
            zt = nt @ (A + dA) / (nt @ (r + dr))
            self.assertLessEqual(got["low"] - 1e-9, zt)
            self.assertGreaterEqual(got["high"] + 1e-9, zt)

    def test_anchor_error_product_term(self):
        rng = np.random.default_rng(32)
        zj, rj, ez, er = 700., np.array([.5, .1, 1.]), 1., .0005
        bound = anchor_error_from_depth(zj, rj, ez, er)
        for _ in range(1000):
            zt = zj + rng.uniform(-ez, ez)
            rt = rj + ball(rng, er)
            self.assertLessEqual(np.linalg.norm(zt * rt - zj * rj), bound + 1e-12)

    def test_near_parallel_no_finite_guarantee(self):
        for n in ([1., 0., 0.], [1., 0., .001]):
            result = plane_depth_interval([1., 0., 700.], n, [0, 0, 1.],
                                          normal_angle_rad=.01)
            self.assertEqual(result["status"], "uninformative")
            self.assertIsNone(result["radius"])
            self.assertTrue(math.isinf(result["high"]))
        n = np.array([1., 0., .001])
        n /= np.linalg.norm(n)
        z1 = n @ np.array([1., 0., 700.]) / n[2]
        n2 = np.array([1., 0., 1e-9])
        n2 /= np.linalg.norm(n2)
        z2 = n2 @ np.array([1., 0., 700.]) / n2[2]
        self.assertGreater(z2, z1 * 1e5)

    def test_curvature_residual_exact_boundary(self):
        # h(u)=H*u^2/2 at u=l saturates the stated Taylor remainder.
        H, ell = .002, 20.
        kappa = H * ell ** 2 / 2
        got = plane_depth_interval([0, 0, 700.], [0, 0, 1.], [0, 0, 1.],
                                   normal_angle_rad=0., residual_error=kappa)
        self.assertLessEqual(got["low"], 700. - kappa)
        self.assertGreaterEqual(got["high"], 700. + kappa)

    def test_candidate_geometry_and_contamination_threshold(self):
        zs = [680., 700., 720.]
        good = [interval(699., 701.) for _ in range(5)]
        bad = [interval(679., 681.) for _ in range(2)]
        result = select_candidate(zs, good + bad, max_bad=2)
        self.assertEqual(result["selected_index"], 1)
        self.assertEqual(result["counts"], [2, 5, 0])
        self.assertGreater(result["pairwise_separating_counts"][1][0], 4)
        # At 50% contamination, rival layers are indistinguishable.
        tie = select_candidate([680., 700.], good[:2] + bad, max_bad=2)
        self.assertIsNone(tie["selected_index"])
        self.assertEqual(tie["status"], "ambiguous")

    def test_tolerance_candidates_and_broad_intervals(self):
        result = select_candidate([699.5, 720.], [interval(700., 700.)] * 5,
                                  max_bad=1, tolerance=.5)
        self.assertEqual(result["selected_index"], 0)
        broad = select_candidate([680., 700., 720.], [interval(670., 730.)] * 5,
                                 max_bad=1)
        self.assertEqual(broad["status"], "ambiguous")
        self.assertEqual(broad["pairwise_separating_counts"], [[0, 0, 0]] * 3)

    def test_common_wrong_layer_and_pool_failure_are_not_certified(self):
        wrong = select_candidate([680., 700., 720.], [interval(719., 721.)] * 7,
                                 max_bad=1)
        self.assertEqual(wrong["selected_index"], 2)
        # If truth=700, all seven intervals violate the b=1 premise.
        self.assertNotEqual([680., 700., 720.][wrong["selected_index"]], 700.)
        # Pool coverage also matters: a broad correct interval alone permits
        # the only offered (but wrong) candidate, without a correct candidate.
        missing = select_candidate([720.], [interval(699., 721.)] * 7, max_bad=1)
        self.assertEqual(missing["selected_index"], 0)

    def test_certain_possible_for_all_endpoint_realizations(self):
        zs = np.array([700., 712.])
        records = [interval(699., 701.), interval(702., 708.), interval(711., 713.),
                   interval(680., 740.), {"status": "uninformative"}]
        result = certain_possible_support(zs, records, tolerance=5., max_radius=5.)
        self.assertEqual(result["denominator"], 5)
        self.assertEqual(result["active_indices"], [0, 1, 2])
        L, U = np.array(result["certain"]), np.array(result["possible"])
        for values in itertools.product(*[(i["low"], (i["low"]+i["high"])/2, i["high"])
                                         for i in records[:3]]):
            V = (np.abs(np.array(values)[:, None] - zs[None, :]) <= 5).sum(axis=0) / 5
            self.assertTrue(np.all(L <= V + 1e-12))
            self.assertTrue(np.all(V <= U + 1e-12))
            # Replace one active latent value arbitrarily: +/-1/N bounds.
            corrupted = np.array(values)
            corrupted[1] = 712.
            Vb = (np.abs(corrupted[:, None] - zs[None, :]) <= 5).sum(axis=0) / 5
            self.assertTrue(np.all(L - 1/5 <= Vb + 1e-12))
            self.assertTrue(np.all(Vb <= U + 1/5 + 1e-12))

    def test_input_rejection(self):
        with self.assertRaises(ValueError):
            plane_depth_interval([0, 0, 700], [0, 0, 0], [0, 0, 1], normal_angle_rad=0.)
        with self.assertRaises(ValueError):
            plane_depth_interval([0, 0, 700], [0, 0, 1], [0, 0, 1], normal_angle_rad=math.pi/2)
        with self.assertRaises(ValueError):
            certain_possible_support([700], [interval(699, 701)], tolerance=1, denominator=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
