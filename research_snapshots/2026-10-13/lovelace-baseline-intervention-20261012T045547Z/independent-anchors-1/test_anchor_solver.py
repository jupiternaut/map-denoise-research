"""Handwritten A8 fixtures only: no real observation or private files read."""
from copy import deepcopy
from fractions import Fraction as Q
import time
import unittest
from unittest.mock import patch

import numpy as np
import anchor_solver as anchor


def fixture(branch="partial", depth=Q(600), offsets=(Q(-1, 10), Q(0), Q(1, 10)),
            darkness=(Q(1, 2),) * 9, ts=(Q(1, 50), Q(1, 100), Q(0)),
            ta=(Q(0), Q(1, 100), Q(1, 50)), photo_noise=None,
            k=Q(1, 200), camera_noise=(Q(-1, 200), Q(0), Q(1, 200)),
            transfer=(Q(0),) * 9, coupon_noise=(Q(0),) * 9, reference_noise=(Q(0),) * 3):
    photo_noise = (Q(0),) * 432 if photo_noise is None else photo_noise
    raw = anchor.pose._prediction(depth, branch, offsets, darkness, ts)
    public = {"observation_id": "manual-A8-only", "baseline": anchor.encode(Q(80)),
              "radius": anchor.encode(Q(1, 5)), "clipped": True,
              "observation": [anchor.encode(anchor._clip(value + error)) for value, error in zip(raw, photo_noise)]}
    oldarm = {"appearance_model": "bounded", "drift_bound": anchor.encode(Q(1, 10)),
              "calibration_intervals": [[anchor.encode(t), anchor.encode(t)] for t in ts]}
    readings = tuple((-center - e - k - n) / 1000
                     for center, e, n in zip((-Q(80), Q(0), Q(80)), offsets, camera_noise))
    coupons = tuple(anchor._clip(Q(1) - d + anchor.BETA - ta[index // 3] + q + n)
                    for index, (d, q, n) in enumerate(zip(darkness, transfer, coupon_noise)))
    refs = tuple(anchor._clip(Q(1, 2) + anchor.BETA - t + n) for t, n in zip(ta, reference_noise))
    anchors = {"observation_id": public["observation_id"],
               "camera": {"u": list(map(anchor.encode, readings)), "Xf": anchor.encode(0),
                          "Zf": anchor.encode(1000), "survey_bound": anchor.encode(Q(1, 100)),
                          "localization_bound": anchor.encode(Q(1, 100))},
               "appearance": {"coupons": list(map(anchor.encode, coupons)),
                              "references": list(map(anchor.encode, refs)),
                              "bias_bound": anchor.encode(Q(1, 100)),
                              "transfer_bound": anchor.encode(Q(1, 200)),
                              "noise_bound": anchor.encode(Q(1, 200))}}
    witness = {"offsets": list(map(anchor.encode, offsets)), "darkness": list(map(anchor.encode, darkness)),
               "t": list(map(anchor.encode, ts)), "noise": list(map(anchor.encode, photo_noise)),
               "tA": list(map(anchor.encode, ta)), "camera_k": anchor.encode(k),
               "camera_noise": list(map(anchor.encode, camera_noise)),
               "coupon_transfer": list(map(anchor.encode, transfer)),
               "coupon_noise": list(map(anchor.encode, coupon_noise)),
               "reference_noise": list(map(anchor.encode, reference_noise))}
    return public, oldarm, anchors, witness


def for_arm(witness, name):
    result = deepcopy(witness)
    if name not in ("camera", "both"):
        result.update(camera_k=None, camera_noise=None)
    if name not in ("appearance", "both"):
        result.update(coupon_transfer=None, coupon_noise=None, reference_noise=None)
    return result


def old_witness(witness):
    return {key: witness[key] for key in anchor.pose.WITNESS_FIELDS}


def unknown(*args, **kwargs):
    return {"status": "UNKNOWN", "witness": None, "lp_status": None, "lp_seconds": 0.0}


class AnchorSolverTests(unittest.TestCase):
    def test_four_arms_have_fixed_15_bounds_and_exact_forward(self):
        public, arm, anchors, witness = fixture()
        self.assertEqual(anchor.ARM_NAMES, ("none", "camera", "appearance", "both"))
        frozen = anchor.pose._inputs(public, arm, 600, "partial")[2]
        for name in anchor.ARM_NAMES:
            with self.subTest(name=name):
                rows, bounds = anchor.build_problem(public, arm, anchors, name, 600, "partial")
                self.assertEqual(bounds, frozen + ((Q(0), Q(1, 50)),) * 3)
                self.assertTrue(all(len(coefficients) == 15 for _, coefficients, _ in rows))
                w = for_arm(witness, name)
                self.assertTrue(anchor.check_forward_witness(public, arm, anchors, name, 600, "partial", w))
                coordinates = w["darkness"] + w["t"] + w["tA"]
                self.assertTrue(anchor.witness_check(rows, bounds, coordinates))

    def test_fixed_pose_rows_recover_original_photo_with_rational_pose(self):
        offsets = (Q(-3, 25), Q(0), Q(1, 11))
        public, arm, anchors, witness = fixture(offsets=offsets)
        rows, bounds = anchor.build_problem(public, arm, anchors, "both", 600, "partial", offsets)
        photo, frozen = anchor.pose.exact_rows(public, arm, 600, "partial", offsets)
        padded = tuple((kind, coeff + (Q(0),) * 3, bound) for kind, coeff, bound in photo)
        self.assertEqual(rows[:len(photo)], padded)
        self.assertEqual(bounds[:12], frozen)
        coordinates = witness["darkness"] + witness["t"] + witness["tA"]
        self.assertTrue(anchor.witness_check(rows, bounds, coordinates))
        rebuilt = anchor._reconstruct_witness(public, arm, anchors, "both", Q(600), "partial", offsets, coordinates)
        self.assertIsNotNone(rebuilt)
        self.assertTrue(anchor.check_forward_witness(public, arm, anchors, "both", 600, "partial", rebuilt))

    def test_camera_nonzero_center_and_common_survey(self):
        public, arm, anchors, witness = fixture()
        survey, intervals = anchor.camera_intervals(anchors)
        self.assertEqual(survey, (Q(-1, 200), Q(1, 100)))
        self.assertEqual(intervals, ((Q(-3, 25), Q(-17, 200)), (Q(0), Q(0)),
                                     (Q(9, 100), Q(1, 8))))
        self.assertEqual(anchor.camera_k_interval(anchors, witness["offsets"]), (Q(0), Q(1, 100)))
        # Both coordinates lie in marginal I, but require incompatible shared k.
        incompatible = (intervals[0][0], Q(0), intervals[2][1])
        self.assertIsNone(anchor.camera_k_interval(anchors, incompatible))
        rows, bounds = anchor.build_problem(public, arm, anchors, "camera", 600, "partial", incompatible)
        result = anchor.classify(rows, bounds, deadline=time.monotonic() + 1)
        self.assertEqual(result["status"], "CERTIFIED_INFEASIBLE")
        self.assertTrue(anchor.certificate_check(rows, bounds, result["certificate"]))

    def test_empty_camera_is_exact_outer_certificate(self):
        public, arm, anchors, _ = fixture()
        anchors["camera"]["u"][1] = anchor.encode(Q(-3, 100000))
        self.assertEqual(anchor.camera_intervals(anchors), (None, None))
        rows, bounds = anchor.build_problem(public, arm, anchors, "camera", 600, "partial")
        result = anchor.classify(rows, bounds, deadline=time.monotonic() + 1)
        self.assertEqual(result["status"], "CERTIFIED_INFEASIBLE")
        self.assertTrue(anchor.certificate_check(rows, bounds, result["certificate"]))
        bad = deepcopy(result["certificate"])
        bad["margin"] = anchor.encode(0)
        self.assertFalse(anchor.certificate_check(rows, bounds, bad))

    def test_exact_narrow_area_enclosure_includes_affine_breakpoints(self):
        for branch in anchor.BRANCHES:
            for depth in (Q(540), Q(577), Q(1231, 2), Q(660)):
                intervals = ((Q(-121, 1000), Q(-119, 1000)), (Q(0), Q(0)),
                             (Q(89, 1000), Q(91, 1000)))
                bounds = anchor.area_bounds(branch, depth, intervals)
                broad = anchor.area_bounds(branch, depth, ((-anchor.RADIUS, anchor.RADIUS),
                                                           (Q(0), Q(0)), (-anchor.RADIUS, anchor.RADIUS)))
                for weights in ((0, 0), (1, 2), (1, 1), (2, 1), (1, 0)):
                    a, b = weights
                    offsets = tuple((lo + hi) / 2 if a + b == 0 else (a * lo + b * hi) / (a + b)
                                    for lo, hi in intervals)
                    areas = anchor.geometry._point_areas(branch, depth, offsets, Q(80))
                    for triple, enclosure, whole in zip(areas, bounds, broad):
                        for index, value in enumerate(triple):
                            self.assertLessEqual(whole[2 * index], enclosure[2 * index])
                            self.assertLessEqual(enclosure[2 * index], value)
                            self.assertLessEqual(value, enclosure[2 * index + 1])
                            self.assertLessEqual(enclosure[2 * index + 1], whole[2 * index + 1])

    def test_anchor_band_interiors_and_clipped_endpoints(self):
        for darkness in ((Q(0),) * 9, (Q(1),) * 9, (Q(1, 2),) * 9):
            errors = (Q(-1, 200), Q(0), Q(1, 200)) * 3
            public, arm, anchors, witness = fixture(darkness=darkness, transfer=errors, coupon_noise=errors)
            rows, bounds = anchor.build_problem(public, arm, anchors, "appearance", 600, "partial", witness["offsets"])
            self.assertTrue(anchor.witness_check(rows, bounds, witness["darkness"] + witness["t"] + witness["tA"]))
            self.assertTrue(anchor.check_forward_witness(public, arm, anchors, "appearance", 600, "partial", for_arm(witness, "appearance")))
            rebuilt = anchor.revalidate_old_witness(public, arm, anchors, "appearance", 600, "partial", old_witness(witness))
            self.assertIsNotNone(rebuilt)
            for y, q, n in zip(anchors["appearance"]["coupons"], rebuilt["coupon_transfer"], rebuilt["coupon_noise"]):
                y, residual = anchor.exact(y), anchor.exact(q) + anchor.exact(n)
                if y in (0, 1):
                    self.assertEqual(residual, Q(-1, 100) if y == 0 else Q(1, 100))

    def test_independent_appearance_bias_is_not_photo_bias(self):
        public, arm, anchors, witness = fixture()
        self.assertNotEqual(witness["t"], witness["tA"])
        self.assertTrue(anchor.check_forward_witness(public, arm, anchors, "both", 600, "partial", witness))
        wrong = deepcopy(witness)
        wrong["tA"] = wrong["t"]
        self.assertFalse(anchor.check_forward_witness(public, arm, anchors, "both", 600, "partial", wrong))
        ref_rows = anchor.appearance_rows(anchors)[18:]
        self.assertTrue(all(not any(coeff[:12]) for _, coeff, _ in ref_rows))

    def test_shared_tA_cannot_be_fitted_per_coupon(self):
        public, arm, anchors, witness = fixture(ta=(Q(1, 100),) * 3)
        anchors["appearance"]["coupons"][0] = anchor.encode(Q(53, 100))
        anchors["appearance"]["coupons"][1] = anchor.encode(Q(47, 100))
        self.assertIsNone(anchor.revalidate_old_witness(public, arm, anchors, "appearance", 600, "partial", old_witness(witness)))
        self.assertIsNone(anchor._ta_intervals(anchors, (Q(1, 2),) * 9))

    def test_disabled_reading_invariance_in_rows_search_and_witness(self):
        public, arm, anchors, witness = fixture()
        for name, disabled in (("none", ("camera", "appearance")),
                               ("camera", ("appearance",)), ("appearance", ("camera",))):
            changed = deepcopy(anchors)
            for key in disabled:
                changed[key] = {"invalid": "must never be decoded"}
            expected = anchor.build_problem(public, arm, anchors, name, 600, "partial")
            self.assertEqual(anchor.build_problem(public, arm, changed, name, 600, "partial"), expected)
            self.assertTrue(anchor.check_forward_witness(public, arm, changed, name, 600, "partial", for_arm(witness, name)))
            self.assertEqual(anchor.revalidate_old_witness(public, arm, changed, name, 600, "partial", old_witness(witness)),
                             anchor.revalidate_old_witness(public, arm, anchors, name, 600, "partial", old_witness(witness)))
            with patch.object(anchor, "_phase_one", return_value=(1.0, "ok")), \
                 patch.object(anchor, "classify", side_effect=unknown), patch.object(anchor, "minimize"):
                before = anchor.complete_branch(public, arm, anchors, name, 600, "partial")
                after = anchor.complete_branch(public, arm, changed, name, 600, "partial")
            for key in ("numeric_evaluations", "numeric_lp_calls", "skipped_camera_candidates",
                        "exact_attempts", "best_phase_slack", "termination"):
                self.assertEqual(before[key], after[key])

    def test_strict_shape_exact_values_gauge_and_instrument_bounds(self):
        public, arm, anchors, witness = fixture()
        for field, index, value in (("offsets", 1, Q(1, 100)), ("offsets", 0, Q(201, 1000)),
                                     ("tA", 0, Q(-1, 1000)), ("tA", 2, Q(21, 1000)),
                                     ("camera_noise", 0, Q(11, 1000)),
                                     ("coupon_transfer", 0, Q(6, 1000)),
                                     ("coupon_noise", 0, Q(6, 1000)),
                                     ("reference_noise", 0, Q(6, 1000))):
            wrong = deepcopy(witness)
            wrong[field][index] = anchor.encode(value)
            self.assertFalse(anchor.check_forward_witness(public, arm, anchors, "both", 600, "partial", wrong))
        for value in (anchor.encode(Q(11, 1000)), 0.0, False):
            wrong = deepcopy(witness)
            wrong["camera_k"] = value
            self.assertFalse(anchor.check_forward_witness(public, arm, anchors, "both", 600, "partial", wrong))
        for field in anchor.WITNESS_FIELDS - {"camera_k"}:
            wrong = deepcopy(witness)
            wrong[field] = wrong[field][:-1]
            self.assertFalse(anchor.check_forward_witness(public, arm, anchors, "both", 600, "partial", wrong))
        wrong = deepcopy(witness)
        wrong["extra"] = None
        self.assertFalse(anchor.check_forward_witness(public, arm, anchors, "both", 600, "partial", wrong))
        wrong = deepcopy(anchors)
        wrong["observation_id"] = "other"
        self.assertFalse(anchor.check_forward_witness(public, arm, wrong, "both", 600, "partial", witness))

    def test_old_witness_failure_is_not_depth_exclusion(self):
        public, arm, anchors, witness = fixture()
        anchors["appearance"]["coupons"][0] = anchor.encode(Q(1, 5))
        self.assertIsNone(anchor.revalidate_old_witness(public, arm, anchors, "appearance", 600, "partial", old_witness(witness)))
        classification = {"status": "CERTIFIED_INFEASIBLE", "witness": None, "lp_status": None, "lp_seconds": 0.0}
        with patch.object(anchor, "_phase_one", return_value=(1.0, "ok")), \
             patch.object(anchor, "classify", return_value=classification), patch.object(anchor, "minimize"):
            result = anchor.complete_branch(public, arm, anchors, "appearance", 600, "partial")
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["center_status"], "UNKNOWN")

    def test_numeric_matrix_contains_all_15_exact_anchor_rows(self):
        public, arm, anchors, witness = fixture()
        offsets = tuple(map(anchor.exact, witness["offsets"]))
        rows, bounds = anchor.build_problem(public, arm, anchors, "both", 600, "partial", offsets)
        bands = anchor.variation.frozen.pixel_bands(tuple(map(anchor.exact, public["observation"])), clipped=True)
        matrix, rhs = anchor._numeric_rows(anchor.pose._float_areas("partial", 600, offsets), bands,
                                           anchor.appearance_rows(anchors))
        normalized, _ = anchor.point.polytope._prepare(rows, bounds)
        np.testing.assert_allclose(matrix, np.asarray([entry[0] for entry in normalized], dtype=float), atol=5e-14, rtol=0)
        np.testing.assert_allclose(rhs, np.asarray([entry[1] for entry in normalized], dtype=float), atol=0, rtol=0)

    def test_complete_midpoint_forward_and_counters(self):
        public, arm, anchors, witness = fixture(darkness=(Q(0),) * 9, ts=(Q(1, 100),) * 3,
                                                ta=(Q(1, 100),) * 3)
        with patch.object(anchor, "minimize") as optimizer:
            result = anchor.complete_branch(public, arm, anchors, "both", 600, "partial", time.monotonic() + 10)
        self.assertEqual(result["status"], "FORWARD_WITNESS")
        self.assertEqual(result["center_status"], "FORWARD_WITNESS")
        self.assertEqual(result["exact_attempts"], 1)
        self.assertEqual(result["numeric_lp_calls"], result["numeric_evaluations"]
                         - result["skipped_camera_candidates"] + result["classifier_lp_calls"])
        self.assertTrue(anchor.check_forward_witness(public, arm, anchors, "both", 600, "partial", result["witness"]))
        optimizer.assert_not_called()

    def test_deadline_lp_and_powell_budgets(self):
        public, arm, anchors, _ = fixture()
        before = anchor.complete_branch(public, arm, anchors, "none", 600, "partial", time.monotonic() - 1)
        self.assertEqual(before["termination"], "whole_deadline")
        self.assertEqual(before["numeric_lp_calls"], 0)
        def many_calls(objective, start, **kwargs):
            for _ in range(1000):
                objective(start)
        classification = {"status": "UNKNOWN", "witness": None, "lp_status": 0, "lp_seconds": .01}
        with patch.object(anchor, "_phase_one", return_value=(1.0, "ok")), \
             patch.object(anchor, "classify", return_value=classification), \
             patch.object(anchor, "minimize", side_effect=many_calls), patch.object(anchor, "MAX_LP_CALLS", 40):
            result = anchor.complete_branch(public, arm, anchors, "none", 600, "partial")
        self.assertEqual(result["numeric_lp_calls"], 40)
        self.assertEqual(result["skipped_camera_candidates"], 0)
        self.assertEqual(result["numeric_lp_calls"], result["numeric_evaluations"]
                         - result["skipped_camera_candidates"] + result["classifier_lp_calls"])
        self.assertEqual(result["termination"], "lp_budget")
        self.assertLessEqual(result["exact_attempts"], 20)
        self.assertLessEqual(result["powell_starts"], 2)
        self.assertTrue(all(value <= 60 for value in result["powell_evaluations"]))

    def test_recovery_after_deadline_is_not_returned(self):
        public, arm, anchors, witness = fixture()
        clock = [100.0]
        positive = {"status": "FEASIBLE_WITNESS", "witness": witness["darkness"] + witness["t"] + witness["tA"],
                    "lp_status": 0, "lp_seconds": .01}
        def delayed(*args):
            clock[0] = 104.0
            return for_arm(witness, "none")
        with patch.object(anchor.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(anchor, "_phase_one", return_value=(0.0, "ok")), \
             patch.object(anchor, "classify", return_value=positive), \
             patch.object(anchor, "_reconstruct_witness", side_effect=delayed):
            result = anchor.complete_branch(public, arm, anchors, "none", 600, "partial", 200.0)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["center_status"], "UNKNOWN")
        self.assertEqual(result["termination"], "branch_timeout")

    def test_exact_recovery_and_two_powell_start_limits(self):
        public, arm, anchors, _ = fixture()
        def distinct_calls(objective, start, **kwargs):
            for index in range(1000):
                objective((-.19 + index / 1000, .19 - index / 1000))
        with patch.object(anchor, "_phase_one", return_value=(0.0, "ok")), \
             patch.object(anchor, "classify", side_effect=unknown), \
             patch.object(anchor, "minimize", side_effect=distinct_calls):
            result = anchor.complete_branch(public, arm, anchors, "none", 600, "partial")
        self.assertEqual(result["exact_attempts"], 20)
        self.assertEqual(result["powell_starts"], 2)
        self.assertEqual(result["powell_evaluations"], [60, 60])
        self.assertEqual(result["numeric_evaluations"], 129)

    def test_candidate_skips_common_k_inconsistent_grid(self):
        public, arm, anchors, _ = fixture()
        seen = []
        def phase(areas, bands, bounds, extra, deadline):
            seen.append(areas)
            return 1.0, "ok"
        with patch.object(anchor, "_phase_one", side_effect=phase), \
             patch.object(anchor, "classify", side_effect=unknown), patch.object(anchor, "minimize"):
            result = anchor.complete_branch(public, arm, anchors, "camera", 600, "partial")
        self.assertEqual(result["numeric_evaluations"], 9)
        self.assertGreater(result["numeric_failures"], 0)
        self.assertGreater(result["skipped_camera_candidates"], 0)
        self.assertEqual(len(seen), result["numeric_evaluations"] - result["skipped_camera_candidates"])
        self.assertEqual(result["numeric_lp_calls"], result["numeric_evaluations"]
                         - result["skipped_camera_candidates"] + result["classifier_lp_calls"])

    def test_clip_order_in_original_photo_is_preserved(self):
        public, arm, anchors, witness = fixture(darkness=(Q(0),) * 9, ts=(Q(0),) * 3)
        public["observation"] = [anchor.encode(Q(199, 200))] * 432
        witness["noise"] = [anchor.encode(Q(-1, 200))] * 432
        self.assertFalse(anchor.check_forward_witness(public, arm, anchors, "both", 600, "partial", witness))
        self.assertIsNone(anchor.revalidate_old_witness(public, arm, anchors, "both", 600, "partial", old_witness(witness)))

    def test_no_private_files_or_opaque_old_artifacts_are_opened(self):
        public, arm, anchors, witness = fixture()
        with patch("builtins.open", side_effect=AssertionError("input file access forbidden")), \
             patch.object(anchor.Path, "read_text", side_effect=AssertionError("input file access forbidden")):
            self.assertIsNotNone(anchor.revalidate_old_witness(public, arm, anchors, "both", 600, "partial", old_witness(witness)))
            self.assertTrue(anchor.check_forward_witness(public, arm, anchors, "both", 600, "partial", witness))


if __name__ == "__main__":
    unittest.main()
