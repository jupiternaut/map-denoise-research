"""Handwritten nonformal A7 fixtures; no old observation/private files loaded."""
from copy import deepcopy
from fractions import Fraction as Q
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np
import pose_solver as pose


def fixture(branch="partial", depth=Q(600), offsets=(Q(0), Q(0), Q(0)),
            darkness=(Q(1, 2),) * 9, ts=(Q(1, 50), Q(1, 100), Q(0)), noise=None):
    noise = (Q(0),) * 432 if noise is None else noise
    raw = pose._prediction(depth, branch, offsets, darkness, ts)
    image = tuple(min(Q(1), max(Q(0), value + error)) for value, error in zip(raw, noise))
    public = {"observation_id": "manual-A7-only", "baseline": pose.encode(Q(80)),
              "radius": pose.encode(Q(1, 5)), "clipped": True,
              "observation": list(map(pose.encode, image))}
    arm = {"appearance_model": "bounded", "drift_bound": pose.encode(Q(1, 10)),
           "calibration_intervals": [[pose.encode(t), pose.encode(t)] for t in ts]}
    witness = {"offsets": list(map(pose.encode, offsets)), "darkness": list(map(pose.encode, darkness)),
               "t": list(map(pose.encode, ts)), "noise": list(map(pose.encode, noise))}
    return public, arm, witness


def unknown_classification(*args, **kwargs):
    return {"status": "UNKNOWN", "witness": None, "lp_status": None, "lp_seconds": 0.0}


class PoseSolverTests(unittest.TestCase):
    def test_exact_rows_use_point_areas_and_global_not_leaf_bounds(self):
        public, arm, witness = fixture(offsets=(Q(-1, 7), Q(0), Q(1, 9)))
        arm["branches"] = {"partial": {"mode": "joint", "trace": [
            {"darkness_bounds": [[pose.encode(0), pose.encode(0)]] * 9,
             "t_bounds": [[pose.encode(0), pose.encode(0)]] * 3}]}}
        offsets = tuple(map(pose.exact, witness["offsets"]))
        rows, bounds = pose.exact_rows(public, arm, Q(600), "partial", offsets)
        areas = pose.geometry._point_areas("partial", Q(600), offsets, Q(80))
        degenerate = tuple(tuple(value for area in triple for value in (area, area)) for triple in areas)
        expected = pose.variation.necessary_rows(tuple(map(pose.exact, public["observation"])),
                                               degenerate, "joint", clipped=True,
                                               drift_bound=Q(1, 10))
        self.assertEqual(rows, expected)
        self.assertEqual(bounds[:9], ((Q(0), Q(1)),) * 9)
        self.assertTrue(pose.point.witness_check(rows, bounds, witness["darkness"] + witness["t"]))

    def test_clip_once_endpoints_and_reconstruction(self):
        noise = tuple(-pose.EPSILON if i // 144 == 0 else pose.EPSILON for i in range(432))
        public, arm, witness = fixture("full", darkness=(Q(1),) * 9, noise=noise)
        image = tuple(map(pose.exact, public["observation"]))
        self.assertIn(Q(0), image)
        self.assertIn(Q(1), image)
        self.assertTrue(pose.check_forward_witness(public, arm, 600, "full", witness))
        reconstructed = pose._reconstruct_witness(public, arm, Q(600), "full", (Q(0),) * 3,
                                                  witness["darkness"] + witness["t"])
        self.assertIsNotNone(reconstructed)
        for y, error in zip(image, map(pose.exact, reconstructed["noise"])):
            if y == 0:
                self.assertEqual(error, -pose.EPSILON)
            elif y == 1:
                self.assertEqual(error, pose.EPSILON)

    def test_interior_not_distance_from_already_clipped_prediction(self):
        public, arm, witness = fixture(darkness=(Q(0),) * 9, ts=(Q(0),) * 3)
        public["observation"] = [pose.encode(Q(199, 200))] * 432
        witness["noise"] = [pose.encode(Q(-1, 200))] * 432
        # clip(1+.01)=1 is within .01 of .995, but legal preclip noise cannot
        # reach .995. The exact raw residual is -.015, outside the noise box.
        self.assertLess(abs(Q(1) - Q(199, 200)), pose.EPSILON)
        self.assertFalse(pose.check_forward_witness(public, arm, 600, "partial", witness))
        self.assertIsNone(pose._reconstruct_witness(public, arm, Q(600), "partial", (Q(0),) * 3,
                                                  witness["darkness"] + witness["t"]))

    def test_bias_is_beta_minus_t_and_noise_has_original_rgb_order(self):
        noise = tuple((Q(-1), Q(0), Q(1))[i % 3] * pose.EPSILON for i in range(432))
        public, arm, witness = fixture(noise=noise)
        self.assertTrue(pose.check_forward_witness(public, arm, 600, "partial", witness))
        wrong = deepcopy(witness)
        wrong["t"] = list(reversed(wrong["t"]))
        arm["calibration_intervals"] = [[pose.encode(0), pose.encode(Q(1, 50))]] * 3
        self.assertFalse(pose.check_forward_witness(public, arm, 600, "partial", wrong))
        wrong = deepcopy(witness)
        wrong["noise"][1] = pose.encode(pose.EPSILON)
        self.assertFalse(pose.check_forward_witness(public, arm, 600, "partial", wrong))

    def test_gauge_radius_calibration_drift_and_exact_types(self):
        public, arm, witness = fixture()
        for field, index, value in (("offsets", 1, Q(1, 100)), ("offsets", 0, Q(201, 1000)),
                                     ("t", 0, Q(0)), ("noise", 0, Q(11, 1000)),
                                     ("darkness", 3, Q(601, 1000)), ("darkness", 0, Q(2))):
            with self.subTest(field=field, value=value):
                wrong = deepcopy(witness)
                wrong[field][index] = pose.encode(value)
                self.assertFalse(pose.check_forward_witness(public, arm, 600, "partial", wrong))
        for wrong_value in (0.0, False, {"numerator": 0, "denominator": 1, "extra": 0}):
            wrong = deepcopy(witness)
            wrong["offsets"][0] = wrong_value
            self.assertFalse(pose.check_forward_witness(public, arm, 600, "partial", wrong))

    def test_all_pair_drift_is_enforced_even_unused_material(self):
        public, arm, witness = fixture("none")
        wrong = deepcopy(witness)
        wrong["darkness"][2] = pose.encode(Q(2, 5))
        wrong["darkness"][5] = pose.encode(Q(1, 2))
        wrong["darkness"][8] = pose.encode(Q(3, 5))
        # Each side is within .1 of middle; first-to-last spread is .2.
        self.assertFalse(pose.check_forward_witness(public, arm, 600, "none", wrong))

    def test_large_exact_offset_denominator_is_physically_legal(self):
        offsets = (Q(1, 1_000_000_007), Q(0), Q(-1, 1_000_000_009))
        public, arm, witness = fixture(offsets=offsets)
        self.assertTrue(pose.check_forward_witness(public, arm, 600, "partial", witness))
        rows, bounds = pose.exact_rows(public, arm, 600, "partial", offsets)
        self.assertTrue(pose.point.witness_check(rows, bounds, witness["darkness"] + witness["t"]))

    def test_strict_witness_shapes_and_tampering(self):
        public, arm, witness = fixture()
        for field in pose.WITNESS_FIELDS:
            for length_change in (-1, 1):
                wrong = deepcopy(witness)
                wrong[field] = wrong[field][:-1] if length_change == -1 else wrong[field] + [pose.encode(0)]
                self.assertFalse(pose.check_forward_witness(public, arm, 600, "partial", wrong))
        wrong = deepcopy(witness)
        wrong["extra"] = "not accepted"
        self.assertFalse(pose.check_forward_witness(public, arm, 600, "partial", wrong))

    def test_float_areas_match_original_exact_geometry(self):
        for branch in pose.BRANCHES:
            for depth in (Q(540), Q(577, 1), Q(1231, 2), Q(660)):
                for offsets in ((Q(-1, 5), Q(0), Q(1, 5)), (Q(0),) * 3,
                                (Q(-3, 25), Q(0), Q(1, 11))):
                    exact = np.asarray(pose.geometry._point_areas(branch, depth, offsets, Q(80)), dtype=float)
                    np.testing.assert_allclose(pose._float_areas(branch, depth, offsets), exact, atol=5e-14, rtol=0)

    def test_numeric_and_exact_rows_have_matching_matrix(self):
        public, arm, witness = fixture(offsets=(Q(-1, 5), Q(0), Q(1, 5)))
        offsets = tuple(map(pose.exact, witness["offsets"]))
        rows, bounds = pose.exact_rows(public, arm, 600, "partial", offsets)
        bands = pose.variation.frozen.pixel_bands(tuple(map(pose.exact, public["observation"])), clipped=True)
        matrix, rhs = pose._numeric_rows(pose._float_areas("partial", 600, offsets), bands)
        normalized, _ = pose.point.polytope._prepare(rows, bounds)
        np.testing.assert_allclose(matrix, np.asarray([entry[0] for entry in normalized], dtype=float), atol=5e-14, rtol=0)
        np.testing.assert_allclose(rhs, np.asarray([entry[1] for entry in normalized], dtype=float), atol=0, rtol=0)

    def test_center_exact_completion_stops_first_witness(self):
        public, arm, _ = fixture(darkness=(Q(0),) * 9, ts=(Q(1, 100),) * 3)
        with patch.object(pose, "minimize") as optimizer:
            result = pose.complete_branch(public, arm, 600, "partial", time.monotonic() + 10)
        self.assertEqual(result["status"], "FORWARD_WITNESS")
        self.assertEqual(result["center_status"], "FORWARD_WITNESS")
        self.assertEqual(result["center_witness"], result["witness"])
        self.assertEqual(result["exact_attempts"], 1)
        self.assertEqual(result["numeric_evaluations"], 1)
        self.assertEqual(result["termination"], "first_exact_witness")
        self.assertTrue(pose.check_forward_witness(public, arm, 600, "partial", result["witness"]))
        optimizer.assert_not_called()

    def test_grid_candidate_can_become_first_exact_noncenter_witness(self):
        public, arm, witness = fixture(offsets=(Q(-1, 5), Q(0), Q(-1, 5)))
        positive = {"status": "FEASIBLE_WITNESS", "witness": witness["darkness"] + witness["t"],
                    "lp_status": 0, "lp_seconds": .01}
        with patch.object(pose, "_phase_one", return_value=(0.0, "ok")), \
             patch.object(pose.point, "classify", side_effect=[unknown_classification(), positive]), \
             patch.object(pose, "minimize") as optimizer:
            result = pose.complete_branch(public, arm, 600, "partial")
        self.assertEqual(result["status"], "FORWARD_WITNESS")
        self.assertEqual(result["center_status"], "UNKNOWN")
        self.assertEqual(result["witness"]["offsets"], witness["offsets"])
        self.assertEqual(result["exact_attempts"], 2)
        self.assertEqual(result["numeric_evaluations"], 2)
        self.assertEqual(result["numeric_lp_calls"], 3)
        self.assertEqual(result["powell_evaluations"], [])
        optimizer.assert_not_called()

    def test_exact_completion_after_candidate_deadline_is_not_returned(self):
        public, arm, witness = fixture()
        clock = [100.0]
        positive = {"status": "FEASIBLE_WITNESS", "witness": witness["darkness"] + witness["t"],
                    "lp_status": 0, "lp_seconds": .01}
        def delayed_reconstruction(*args):
            clock[0] = 104.0
            return witness
        with patch.object(pose.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(pose, "_phase_one", return_value=(0.0, "ok")), \
             patch.object(pose.point, "classify", return_value=positive), \
             patch.object(pose, "_reconstruct_witness", side_effect=delayed_reconstruction):
            result = pose.complete_branch(public, arm, 600, "partial", 200.0)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["center_status"], "UNKNOWN")
        self.assertEqual(result["termination"], "branch_timeout")
        self.assertEqual(result["search_budget_seconds"], 3.0)

    def test_exact_fixed_pose_infeasibility_never_becomes_negative(self):
        public, arm, _ = fixture()
        classification = {"status": "CERTIFIED_INFEASIBLE", "witness": None,
                          "lp_status": None, "lp_seconds": 0.0}
        with patch.object(pose, "_phase_one", return_value=(1.0, "ok")), \
             patch.object(pose.point, "classify", return_value=classification):
            result = pose.complete_branch(public, arm, 600, "partial")
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["center_status"], "UNKNOWN")
        self.assertIsNone(result["witness"])
        self.assertEqual(result["termination"], "search_exhausted")

    def test_lp_exception_and_bad_numeric_results_are_unknown(self):
        public, arm, _ = fixture()
        for outcome in (RuntimeError("manual LP failure"), SimpleNamespace(fun=float("nan"), status=0, success=True)):
            context = patch.object(pose, "linprog", side_effect=outcome) if isinstance(outcome, Exception) else patch.object(pose, "linprog", return_value=outcome)
            with context, patch.object(pose.point, "classify", side_effect=unknown_classification):
                result = pose.complete_branch(public, arm, 600, "partial")
            self.assertEqual(result["status"], "UNKNOWN")
            self.assertEqual(result["termination"], "numeric_failure")
            self.assertEqual(result["numeric_failures"], 9)
            self.assertEqual(result["exact_attempts"], 1)

    def test_total_lp_budget_includes_a5_calls(self):
        public, arm, _ = fixture()
        def many_calls(objective, start, **kwargs):
            for _ in range(1000):
                objective(start)
        classification = {"status": "UNKNOWN", "witness": None, "lp_status": 0, "lp_seconds": .01}
        with patch.object(pose, "_phase_one", return_value=(1.0, "ok")), \
             patch.object(pose.point, "classify", return_value=classification), \
             patch.object(pose, "minimize", side_effect=many_calls), \
             patch.object(pose, "MAX_LP_CALLS", 40):
            result = pose.complete_branch(public, arm, 600, "partial")
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["termination"], "lp_budget")
        self.assertEqual(result["numeric_lp_calls"], 40)
        self.assertEqual(result["numeric_evaluations"], 39)
        self.assertEqual(result["classifier_lp_calls"], 1)
        self.assertEqual(len(result["powell_evaluations"]), result["powell_starts"])
        self.assertTrue(all(value <= 60 for value in result["powell_evaluations"]))

    def test_exceptional_a5_lp_with_null_status_still_counts(self):
        public, arm, _ = fixture()
        classification = {"status": "UNKNOWN", "witness": None, "lp_status": None, "lp_seconds": .01}
        with patch.object(pose, "_phase_one", return_value=(1.0, "ok")), \
             patch.object(pose.point, "classify", return_value=classification):
            result = pose.complete_branch(public, arm, 600, "partial")
        self.assertEqual(result["classifier_lp_calls"], 1)
        self.assertEqual(result["numeric_lp_calls"], result["numeric_evaluations"] + 1)

    def test_expired_deadline_and_outside_domain_do_no_search(self):
        public, arm, _ = fixture()
        with patch.object(pose, "linprog") as lp, patch.object(pose.point, "classify") as classifier:
            result = pose.complete_branch(public, arm, 600, "partial", time.monotonic() - 1)
            self.assertEqual(result["termination"], "whole_deadline")
            self.assertEqual(result["numeric_lp_calls"], 0)
            result = pose.complete_branch(public, arm, 700, "partial")
            self.assertEqual(result["termination"], "outside_domain")
            self.assertEqual(result["numeric_lp_calls"], 0)
        lp.assert_not_called()
        classifier.assert_not_called()

    def test_exact_attempt_cap_and_powell_configuration(self):
        public, arm, _ = fixture()
        def explore(objective, start, **kwargs):
            self.assertEqual(kwargs["method"], "Powell")
            self.assertEqual(kwargs["options"]["maxfev"], 60)
            for i in range(60):
                objective((-.19 + i / 200, .17 - i / 200))
        with patch.object(pose, "_phase_one", return_value=(0.0, "ok")), \
             patch.object(pose.point, "classify", side_effect=unknown_classification), \
             patch.object(pose, "minimize", side_effect=explore):
            result = pose.complete_branch(public, arm, 600, "partial")
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["exact_attempts"], 20)
        self.assertEqual(result["powell_starts"], 2)
        self.assertEqual(result["powell_evaluations"], [60, 60])
        self.assertLessEqual(result["numeric_lp_calls"], 200)


if __name__ == "__main__":
    unittest.main()
