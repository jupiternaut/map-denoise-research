"""Small exact fixtures; no confirmation batch or saved science outputs."""
from fractions import Fraction as Q
import inspect
import unittest
from unittest.mock import patch

import structured_solver as s


def rgb(*values):
    return tuple(channel for value in values for channel in (value,) * 3)


class StructuredSolverTests(unittest.TestCase):
    def test_raw_bands_preserve_both_sides_outside_unit_interval(self):
        self.assertEqual(s.pixel_bands(rgb(Q(-1, 100), Q(101, 100)), clipped=False),
                         ((Q(101, 100), Q(103, 100)), (Q(-1, 100), Q(1, 100))))

    def test_clip_endpoints_are_one_sided(self):
        self.assertEqual(s.pixel_bands(rgb(Q(0), Q(1), Q(1, 2)), clipped=True),
                         ((Q(1), None), (None, Q(1, 50)), (Q(1, 2), Q(13, 25))))
        self.assertEqual(s.pixel_bands((Q(0), Q(1, 100), Q(1, 50)), clipped=True),
                         ((Q(1), Q(1)),))

    def test_clip_zero_retains_negative_bias_saturated_witness(self):
        area = ((Q(1), Q(1), Q(0), Q(0), Q(0), Q(0)),)
        rows = s.necessary_rows(rgb(Q(0)), area, "joint", clipped=True, epsilon=Q(0))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "lower")
        # f=0,b=-.01,e=0 produces preclip -.01. Treating clip0 as a raw
        # equality would incorrectly impose d+t<=1.01 and exclude this witness.
        witness = (Q(1), Q(0), Q(0), Q(1, 50), Q(0), Q(0))
        self.assertGreaterEqual(sum(a*x for a, x in zip(rows[0][1], witness)), rows[0][2])
        bounds = tuple((x, x) for x in witness)
        self.assertFalse(s.propagate(rows, bounds=bounds)["excluded"])
        raw_rows = s.necessary_rows(rgb(Q(0)), area, "joint", clipped=False, epsilon=Q(0))
        self.assertTrue(s.propagate(raw_rows, bounds=bounds)["excluded"])

    def test_clip_one_retains_positive_bias_saturated_witness(self):
        area = ((Q(0),) * 6,)
        rows = s.necessary_rows(rgb(Q(1)), area, "joint", clipped=True, epsilon=Q(0))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "upper")
        bounds = ((Q(0), Q(0)),) * 6  # f=1,b=+.01,e=0 -> preclip1.01
        self.assertFalse(s.propagate(rows, bounds=bounds)["excluded"])
        raw_rows = s.necessary_rows(rgb(Q(1)), area, "joint", clipped=False, epsilon=Q(0))
        self.assertTrue(s.propagate(raw_rows, bounds=bounds)["excluded"])

    def test_background_uses_every_branch_and_full_domain(self):
        mask = s.background_mask(Q(1, 5))
        self.assertEqual(len(mask), 144)
        self.assertEqual([sum(mask[v*48:(v+1)*48]) for v in range(3)], [18, 12, 18])
        full = [s.nuisance_model._geometry_bounds(branch, Q(1, 5), Q(540), Q(660))
                for branch in s.BRANCHES]
        self.assertTrue(all(all(all(a == 0 for a in g[p][1::2]) for g in full)
                            for p, flag in enumerate(mask) if flag))
        # A pixel that is background only under none must not enter the mask.
        none_only = [p for p in range(144) if all(a == 0 for a in full[0][p][1::2])
                     and any(a > 0 for g in full[1:] for a in g[p][1::2])]
        self.assertTrue(none_only)
        self.assertTrue(all(not mask[p] for p in none_only))

    def test_background_mask_not_hardcoded_to_registered_radius(self):
        blank = [(Q(0),) * 6 for _ in range(144)]
        altered = list(blank)
        altered[0] = (Q(0), Q(0), Q(0), Q(0), Q(0), Q(1, 2))
        calls = []

        def geometry(branch, radius, lo, hi):
            calls.append((branch, radius, lo, hi))
            return tuple(altered if branch == "full" else blank)

        with patch.object(s.nuisance_model, "_geometry_bounds", side_effect=geometry):
            mask = s.background_mask(Q(3, 7))
        self.assertFalse(mask[0])
        self.assertTrue(all(mask[1:]))
        self.assertEqual(calls, [(branch, Q(3, 7), Q(540), Q(660)) for branch in s.BRANCHES])

    def test_calibration_raw_bias_and_clip_one_ambiguity(self):
        mask = tuple(p in (0, 48, 96) for p in range(144))
        image = rgb(*(Q(99, 100) if p < 48 else Q(1) if p < 96 else Q(101, 100)
                      for p in range(144)))
        cal = s.calibration_bounds(image, mask, clipped=False, epsilon=Q(0))
        self.assertFalse(cal["excluded"])
        self.assertEqual([[s.decode(x) for x in pair] for pair in cal["intervals"]],
                         [[Q(1, 50), Q(1, 50)], [Q(1, 100), Q(1, 100)], [Q(0), Q(0)]])
        clipped = tuple(min(Q(1), max(Q(0), x)) for x in image)
        cal = s.calibration_bounds(clipped, mask, clipped=True, epsilon=Q(0))
        self.assertEqual([[s.decode(x) for x in pair] for pair in cal["intervals"]],
                         [[Q(1, 50), Q(1, 50)], [Q(0), Q(1, 100)], [Q(0), Q(1, 100)]])

    def test_calibration_crossing_is_explicit_incompatible(self):
        mask = tuple(p == 0 for p in range(144))
        cal = s.calibration_bounds(rgb(*([Q(0)] * 144)), mask, clipped=True)
        self.assertTrue(cal["excluded"])
        contradiction = cal["certificate"]["contradiction"]
        self.assertEqual((contradiction["kind"], contradiction["pixel"], contradiction["view"]),
                         ("interval", 0, 0))
        self.assertGreater(s.decode(contradiction["lower"]), s.decode(contradiction["upper"]))
        result = s.solve_structured(rgb(*([Q(0)] * 144)), Q(1, 5), "joint", clipped=True)
        self.assertTrue(result["calibration"]["excluded"])
        self.assertIsNone(result["outer"]["hull"])
        self.assertEqual(result["outer"]["retained_intervals"], [])
        self.assertTrue(all(b["trace"][0]["status"] == "excluded_calibration"
                            and b["evaluated_boxes"] == 0 for b in result["branches"].values()))

    def test_calibrated_rows_marginalize_frozen_interval(self):
        area = ((Q(1, 4), Q(1, 2), Q(0), Q(0), Q(0), Q(0)),)
        ts = ((Q(1, 200), Q(3, 200)), (Q(0), Q(1, 50)), (Q(0), Q(1, 50)))
        rows = s.necessary_rows(rgb(Q(1, 2)), area, "calibrated", clipped=False,
                                calibration_intervals=ts)
        self.assertEqual(rows, (("upper", (Q(1, 4), Q(0), Q(0)), Q(103, 200)),
                                ("lower", (Q(1, 2), Q(0), Q(0)), Q(97, 200))))

    def test_shared_bias_rejects_two_pixels_independent_bias_retains(self):
        # No shared material can absorb the discrepancy: background pixels
        # demand t=.005 and t=.015 in the same view, each inside the prior.
        area = ((Q(0),) * 6,) * 2
        image = rgb(Q(201, 200), Q(199, 200))
        joint = s.necessary_rows(image, area, "joint", clipped=False, epsilon=Q(0))
        prior = s.MATERIAL_BOUNDS + ((Q(0), Q(1, 50)),) * 3
        result = s.propagate(joint, bounds=prior)
        self.assertTrue(result["excluded"])
        self.assertTrue(result["certificate"]["steps"])
        self.assertEqual(result["certificate"]["contradiction"]["kind"], "row")
        for pixel in range(2):
            separate = s.necessary_rows(image[3*pixel:3*pixel+3], area[:1], "joint",
                                        clipped=False, epsilon=Q(0))
            self.assertFalse(s.propagate(separate, bounds=prior)["excluded"])
        cal = s.necessary_rows(image, area, "calibrated", clipped=False,
                               calibration_intervals=((Q(0), Q(1, 50)),) * 3, epsilon=Q(0))
        self.assertFalse(s.propagate(cal)["excluded"])

    def test_initial_crossing_certificate_and_invalid_inputs(self):
        crossed = ((Q(1, 2), Q(1, 3)), (Q(0), Q(1)), (Q(0), Q(1)))
        result = s.propagate((), bounds=crossed, rounds=0)
        self.assertTrue(result["excluded"])
        self.assertEqual(result["certificate"]["contradiction"]["kind"], "initial_interval")
        self.assertEqual(result["certificate"]["steps"], [])
        for bounds in ((((Q(-1), Q(1)),) * 3), (((Q(0), Q(1)),) * 4)):
            with self.assertRaises(ValueError):
                s.propagate((), bounds=bounds)
        with self.assertRaises(ValueError):
            s.propagate((("upper", (Q(-1), Q(0), Q(0)), Q(0)),))
        with self.assertRaises(TypeError):
            s.pixel_bands((0.5,) * 3, clipped=False)
        with self.assertRaises(ValueError):
            s.pixel_bands(rgb(Q(101, 100)), clipped=True)
        with self.assertRaises(ValueError):
            s.solve_structured((Q(1),) * 3, Q(1, 5), "joint", clipped=False)

    def test_contractor_trace_replay_and_zero_rounds(self):
        rows = (("upper", (Q(1), Q(1), Q(0)), Q(3, 4)),
                ("lower", (Q(1), Q(0), Q(0)), Q(4, 5)))
        result = s.propagate(rows)
        self.assertTrue(result["excluded"])
        certificate = result["certificate"]
        endpoints = [[s.decode(x) for x in pair] for pair in certificate["initial_bounds"]]
        for step in certificate["steps"]:
            kind, coefficients, bound = rows[step["row"]]
            side = 0 if kind == "upper" else 1
            values = [pair[side] for pair in endpoints]
            variable = step["variable"]
            candidate = (bound - sum(a*x for i, (a, x) in enumerate(zip(coefficients, values))
                                      if i != variable)) / coefficients[variable]
            self.assertEqual(s.decode(step["value"]), candidate)
            endpoints[variable][1 if step["side"] == "upper" else 0] = candidate
        witness = certificate["contradiction"]
        kind, coefficients, bound = rows[witness["row"]]
        values = [pair[0 if kind == "upper" else 1] for pair in endpoints]
        total = sum(a*x for a, x in zip(coefficients, values))
        self.assertEqual(s.decode(witness["total"]), total)
        self.assertTrue(total > bound if kind == "upper" else total < bound)
        untouched = s.propagate(rows, rounds=0)
        self.assertFalse(untouched["excluded"])
        self.assertEqual(untouched["bounds"], s._encoded_pairs(s.MATERIAL_BOUNDS))

    def test_feasible_truth_is_retained_raw_and_clip(self):
        radius, truth = Q(1, 5), Q(600)
        model = s.historical.ColoredModel("partial", camera_radius=radius)
        clean = model.point_prediction(truth, colors=(Q(0), Q(1, 3), Q(1)))
        bias = (Q(-1, 100), Q(0), Q(1, 100))
        raw = tuple(y + bias[channel // 144] for channel, y in enumerate(clean))
        for clipped in (False, True):
            image = tuple(min(Q(1), max(Q(0), y)) for y in raw) if clipped else raw
            for method in s.METHODS:
                result = s.solve_structured(image, radius, method, clipped=clipped,
                                            max_boxes=3, rounds=3)
                self.assertTrue(any(s.decode(lo) <= truth <= s.decode(hi)
                                    for lo, hi in result["outer"]["retained_intervals"]))
                t = tuple(Q(1, 100) - b for b in bias)
                self.assertTrue(all(s.decode(pair[0]) <= value <= s.decode(pair[1])
                                    for value, pair in zip(t, result["calibration_intervals"])))
                darkness = (Q(1), Q(2, 3), Q(0))
                for node in result["branches"]["partial"]["trace"]:
                    if s.decode(node["lo"]) <= truth <= s.decode(node["hi"]):
                        self.assertIn(node["status"], ("split", "retained_tolerance", "unresolved_budget"))
                        self.assertTrue(all(s.decode(pair[0]) <= value <= s.decode(pair[1])
                                            for value, pair in zip(darkness, node["darkness_bounds"])))
                        self.assertTrue(all(s.decode(pair[0]) <= value <= s.decode(pair[1])
                                            for value, pair in zip(t, node["t_bounds"])))
                self.assertEqual(result["outer"]["mode"], method)
                self.assertEqual(result["outer"]["channel_precheck"], "none")

    def test_public_interface_and_budget_keep_all_branches(self):
        signature = inspect.signature(s.solve_structured)
        self.assertEqual(tuple(signature.parameters),
                         ("observation", "radius", "method", "clipped", "beta", "epsilon",
                          "rounds", "max_boxes", "tolerance"))
        result = s.solve_structured(rgb(*([Q(1)] * 144)), Q(1, 5), "joint",
                                    clipped=True, max_boxes=0)
        self.assertEqual(result["outer"]["retained_branches"], list(s.BRANCHES))
        self.assertEqual(result["outer"]["unresolved_budget_leaves"], 3)
        self.assertEqual(result["outer"]["hull"], [s.encode(Q(540)), s.encode(Q(660))])
        self.assertTrue(all(branch["trace"][0]["status"] == "unresolved_budget"
                            for branch in result["branches"].values()))
        self.assertEqual(result["background_mask"], list(s.background_mask(Q(1, 5))))
        self.assertEqual(result["calibration_intervals"], result["calibration"]["intervals"])
        self.assertTrue(all(branch["initial_bounds"][3:] == result["calibration_intervals"]
                            for branch in result["branches"].values()))
        self.assertGreaterEqual(result["outer_seconds"], result["preparation_seconds"])


if __name__ == "__main__":
    unittest.main()
