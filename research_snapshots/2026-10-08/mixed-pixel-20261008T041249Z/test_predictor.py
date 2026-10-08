"""Hand-built array checks only; never loads experiment observations or truth."""
import unittest
import numpy as np
import predictor as p


def camera(cx=0.):
    return {"K": np.array([[80., 0., 64.], [0., 80., 64.], [0., 0., 1.]]),
            "R": np.eye(3), "C": np.array([cx, 0., 0.])}


def auxiliary(textured=False, equal=False):
    yy, xx = np.indices((128, 128))
    fg = 100.+.5*xx if textured else np.full((128, 128), 100.)
    bg = fg.copy() if equal else np.full((128, 128), 20.)
    return {"mode": "two", "foreground": fg, "background": bg,
            "polygon": np.array([[62.3, 56.], [65.7, 56.], [65.7, 72.], [62.3, 72.]]),
            "mask": np.zeros((128, 128)), "background_depth": np.inf,
            "valid": True, "sigma": np.nan, "sigma_valid": False}


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.ref, self.eval = camera(), camera(20.)
        self.grid = np.array([200., 400., 800.])

    def test_projection_inverse_and_fixed_roi(self):
        xy = np.array([[62.4, 60.2], [66., 70.]])
        uv, front = p.project_reference_pixels(xy, [400.], self.ref, self.eval)
        np.testing.assert_allclose(uv[0], xy-[4., 0.], atol=1e-12)
        self.assertTrue(front.all())
        inv, _, ok = p._inverse_plane(uv[0, :, None], [400.], self.ref, self.eval)
        np.testing.assert_allclose(inv[0, :, 0], xy, atol=1e-12)
        self.assertTrue(ok.all())
        d = p.predict_pixels(auxiliary(), self.ref, self.eval, self.grid, 400.)
        f = p.predict_pixels(auxiliary(), self.ref, self.eval, self.grid, 200., mode="fixed")
        np.testing.assert_array_equal(d["roi"], f["roi"])
        self.assertTrue(d["depth_valid"].all())
        self.assertTrue(np.all((d["alpha"] >= 0) & (d["alpha"] <= 1)))

    def test_fixed_ownership_is_per_ray_but_texture_keeps_warp(self):
        a = auxiliary(textured=True)
        f = p.predict_pixels(a, self.ref, self.eval, self.grid, 400., mode="fixed")
        np.testing.assert_array_equal(f["ownership"][0], f["ownership"][1])
        np.testing.assert_array_equal(f["ownership"][1], f["ownership"][2])
        expected_delta = f["alpha"][0] * .5 * (80*20/200-80*20/800)
        np.testing.assert_allclose(f["prediction"][0]-f["prediction"][2], expected_delta, atol=1e-12)
        self.assertGreater(float(np.max(np.abs(expected_delta))), 0.)
        d = p.predict_pixels(a, self.ref, self.eval, self.grid, 400.)
        self.assertTrue(np.any(d["ownership"][0] != d["ownership"][2]))

    def test_equal_constant_has_no_depth_signal(self):
        a = auxiliary(equal=True)
        observed = np.full((128, 128), 100.)
        for mode in ("fixed", "dynamic"):
            result = p.predict_curve(a, self.ref, self.eval, observed, self.grid, 400., mode=mode)
            self.assertTrue(result["valid"])
            self.assertFalse(result["sigma_valid"])
            self.assertTrue(result["flat"])
            np.testing.assert_array_equal(result["loss"], 0.)

    def test_manual_sensor_integration_and_raw_loss_without_sigma(self):
        # Independent parallel-camera formula, independent ownership expression.
        yy, xx = np.indices((128, 128))
        image = np.zeros_like(xx, float)
        for ox in (-1/3, 0., 1/3):
            for oy in (-1/3, 0., 1/3):
                rx, ry = xx+ox+4., yy+oy
                owned = (rx >= 62.3) & (rx <= 65.7) & (ry >= 56.) & (ry <= 72.)
                image += np.where(owned, 100., 20.)/9
        result = p.predict_curve(auxiliary(), self.ref, self.eval, image, self.grid, 400.)
        self.assertTrue(result["valid"])
        self.assertFalse(result["sigma_valid"])
        self.assertEqual(int(np.argmin(result["loss"])), 1)
        self.assertLess(result["loss"][1], 1e-20)
        self.assertGreater(result["loss"][0], 1.)

    def test_background_occludes_deeper_candidate(self):
        a = auxiliary()
        a["background_depth"] = 500.
        d = p.predict_pixels(a, self.ref, self.eval, self.grid, 400.)
        self.assertFalse(d["ownership"][-1].any())
        np.testing.assert_allclose(d["prediction"][-1], 20.)

    def test_estimator_constant_is_single_and_scale_is_held_out(self):
        image = np.full((128, 128), 37.)
        a = p.estimate_aux(image, image, self.ref, self.eval)
        self.assertEqual(a["mode"], "single")
        self.assertTrue(a["valid"])
        self.assertTrue(a["sigma_valid"])
        self.assertEqual(a["sigma"], 1.)
        self.assertGreaterEqual(a["metadata"]["sigma_foreground"]["blocks"], 4)
        result = p.predict_curve(a, self.ref, self.eval, image, self.grid, 400.)
        np.testing.assert_allclose(result["loss"], 0.)

    def test_narrow_target_reports_insufficient_scale_but_retains_aux(self):
        image = np.full((128, 128), 25.)
        image[58:71, 63:66] = 120.
        a = p.estimate_aux(image, image, self.ref, self.eval)
        self.assertEqual(a["mode"], "two")
        self.assertTrue(a["valid"])
        self.assertFalse(a["sigma_valid"])
        self.assertTrue(np.isposinf(a["background_depth"]))
        self.assertFalse(a["background_depth_valid"])
        self.assertEqual(a["background_geometry"], "constant_background")
        result = p.predict_curve(a, self.ref, self.eval, image, self.grid, 400.)
        self.assertTrue(result["valid"])
        self.assertTrue(np.all(np.isfinite(result["loss"])))

    def test_background_depth_comes_from_training_ncc(self):
        yy, xx = np.indices((128, 128))
        texture = lambda x, y: 40+8*np.sin(x/6)+6*np.cos(y/7)+3*np.sin((x+y)/3)
        reference = texture(xx, yy)
        training = texture(xx+80*20/760, yy)
        distance = np.maximum(np.abs(xx-64), np.abs(yy-64))
        use = (distance >= 12) & (distance <= 20)
        xy = np.column_stack((xx[use], yy[use]))
        depth, info = p._estimate_background(reference, training, xy, self.ref, self.eval, p._params({"background_grid": np.arange(500., 1001., 2.)}))
        self.assertTrue(info["valid"])
        self.assertLessEqual(abs(depth-760), 10.)
        self.assertGreater(info["best_ncc"], .99)

    def test_invalid_projection_never_partially_scores_candidates(self):
        a = auxiliary()
        a["valid"] = False
        result = p.predict_curve(a, self.ref, self.eval, np.zeros((128, 128)), self.grid, 400.)
        self.assertFalse(result["valid"])
        self.assertTrue(np.isnan(result["loss"]).all())


if __name__ == "__main__":
    unittest.main()
