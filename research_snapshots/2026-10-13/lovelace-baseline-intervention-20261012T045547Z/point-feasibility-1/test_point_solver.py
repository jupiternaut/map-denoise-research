"""Handwritten A6 fixtures only: no observation/task/result/private file loads."""

from copy import deepcopy
from fractions import Fraction as Q
import json
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np
import point_solver as point


def encoded_pairs(pairs):
    return [[point.encode(lo), point.encode(hi)] for lo, hi in pairs]


def public_fixture():
    return {"observation_id": "manual-only", "radius": point.encode(Q(1, 10)),
            "baseline": point.encode(Q(80)), "clipped": True,
            "observation": [point.encode(Q(1, 2))] * point.variation.RGB_COUNT}


def arm_fixture(method="calibrated", nodes=()):
    return {"appearance_model": "bounded", "drift_bound": point.encode(Q(1, 10)),
            "calibration_intervals": encoded_pairs(((Q(0), Q(1, 50)),) * 3),
            "branches": {branch: {"mode": method, "trace": list(nodes) if branch == "none" else [],
                                  "retained_ids": [node["id"] for node in nodes] if branch == "none" else []}
                         for branch in point.BRANCHES}}


def node_fixture(index=3, lo=Q(599), hi=Q(601), status="retained_tolerance"):
    return {"id": index, "status": status, "lo": point.encode(lo), "hi": point.encode(hi),
            "darkness_bounds": encoded_pairs(((Q(1, 4), Q(3, 4)),) * 9),
            "t_bounds": encoded_pairs(((Q(1, 200), Q(3, 200)),) * 3)}


def manual_area_intervals(active=True):
    pixel = (Q(1), Q(1), Q(0), Q(0), Q(0), Q(0)) if active else (Q(0),) * 6
    return (pixel,) * point.variation.PIXEL_COUNT


class PointSolverTests(unittest.TestCase):
    def test_zero_width_depth_uses_interval_geometry_and_preserves_pose_radius(self):
        public = public_fixture()
        arm = arm_fixture()
        depth = Q(600)
        with patch.object(point.geometry, "geometry_bounds", wraps=point.geometry.geometry_bounds) as geometry:
            rows, bounds = point.build_problem(public, arm, "calibrated", "none", depth)
        geometry.assert_called_once_with("none", Q(1, 10), depth, depth, Q(80))
        self.assertEqual(len(bounds), 9)
        self.assertEqual(bounds, ((Q(0), Q(1)),) * 9)
        self.assertTrue(rows)
        areas = point.geometry.geometry_bounds("none", Q(1, 10), depth, depth, Q(80))
        self.assertTrue(any(lo < hi for pixel in areas for lo, hi in zip(pixel[::2], pixel[1::2])))

    def test_point_area_intervals_are_nested_in_containing_depth_box(self):
        for branch in point.BRANCHES:
            with self.subTest(branch=branch):
                outer = point.geometry.geometry_bounds(branch, Q(1, 10), Q(590), Q(610), Q(80))
                inner = point.geometry.geometry_bounds(branch, Q(1, 10), Q(600), Q(600), Q(80))
                self.assertEqual(len(outer), point.variation.PIXEL_COUNT)
                self.assertEqual(len(inner), len(outer))
                for outer_pixel, inner_pixel in zip(outer, inner):
                    for low, high, sub_low, sub_high in zip(outer_pixel[::2], outer_pixel[1::2],
                                                          inner_pixel[::2], inner_pixel[1::2]):
                        self.assertLessEqual(low, sub_low)
                        self.assertLessEqual(sub_low, sub_high)
                        self.assertLessEqual(sub_high, high)

    def test_calibrated_same_leaf_uses_original_material_and_arm_calibration(self):
        public = public_fixture()
        node = node_fixture()
        arm = arm_fixture(nodes=(node,))
        areas = manual_area_intervals()
        with patch.object(point.geometry, "geometry_bounds", return_value=areas):
            global_rows, global_bounds = point.build_problem(public, arm, "calibrated", "none", Q(600))
            local_rows, local_bounds = point.build_problem(public, arm, "calibrated", "none", Q(600), node)
        calibration = ((Q(0), Q(1, 50)),) * 3
        expected = point.variation.necessary_rows((Q(1, 2),) * point.variation.RGB_COUNT,
                                                areas, "calibrated", clipped=True,
                                                drift_bound=Q(1, 10), calibration_intervals=calibration,
                                                beta=point.variation.BETA, epsilon=point.variation.EPSILON)
        wrong = point.variation.necessary_rows((Q(1, 2),) * point.variation.RGB_COUNT,
                                             areas, "calibrated", clipped=True,
                                             drift_bound=Q(1, 10),
                                             calibration_intervals=((Q(1, 200), Q(3, 200)),) * 3)
        self.assertEqual(global_rows, expected)
        self.assertEqual(local_rows, expected)
        self.assertNotEqual(local_rows, wrong)
        self.assertEqual(global_bounds, ((Q(0), Q(1)),) * 9)
        self.assertEqual(local_bounds, ((Q(1, 4), Q(3, 4)),) * 9)

    def test_joint_same_leaf_uses_node_bias_and_global_uses_arm_bias(self):
        node = node_fixture()
        arm = arm_fixture("joint", (node,))
        with patch.object(point.geometry, "geometry_bounds", return_value=manual_area_intervals()):
            global_rows, global_bounds = point.build_problem(public_fixture(), arm, "joint", "none", Q(600))
            local_rows, local_bounds = point.build_problem(public_fixture(), arm, "joint", "none", Q(600), node)
        self.assertEqual(global_rows, local_rows)
        self.assertEqual(len(global_bounds), 12)
        self.assertEqual(global_bounds[:9], ((Q(0), Q(1)),) * 9)
        self.assertEqual(global_bounds[9:], ((Q(0), Q(1, 50)),) * 3)
        self.assertEqual(local_bounds[:9], ((Q(1, 4), Q(3, 4)),) * 9)
        self.assertEqual(local_bounds[9:], ((Q(1, 200), Q(3, 200)),) * 3)

    def test_both_closed_boundary_leaves_are_classified_and_counted(self):
        nodes = (node_fixture(7, Q(599), Q(600)), node_fixture(8, Q(600), Q(601)))
        arm = arm_fixture(nodes=nodes)
        with patch.object(point.geometry, "geometry_bounds", return_value=manual_area_intervals()), \
             patch.object(point, "classify", side_effect=[{"status": "FEASIBLE_WITNESS"},
                                                        {"status": "UNKNOWN"}]) as classify:
            records = [point.classify_problem(public_fixture(), arm, "calibrated", "none", Q(600), node)
                       for node in nodes]
        self.assertEqual(classify.call_count, 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(point.summarize(record["status"] for record in records), "FEASIBLE_WITNESS")

    def test_budget_leaf_and_zero_width_leaf_are_valid_original_sources(self):
        for status in point.RETAINED_STATUSES:
            node = node_fixture(lo=Q(600), hi=Q(600), status=status)
            arm = arm_fixture(nodes=(node,))
            with patch.object(point.geometry, "geometry_bounds", return_value=manual_area_intervals()):
                rows, bounds = point.build_problem(public_fixture(), arm, "calibrated", "none", Q(600), node)
            self.assertTrue(rows)
            self.assertEqual(bounds, ((Q(1, 4), Q(3, 4)),) * 9)

    def test_wrong_leaf_identity_status_or_depth_is_rejected(self):
        node = node_fixture()
        arm = arm_fixture(nodes=(node,))
        with self.assertRaises(ValueError):
            point.build_problem(public_fixture(), arm, "calibrated", "none", Q(602), node)
        bad = deepcopy(node)
        bad["darkness_bounds"][0][1] = point.encode(Q(1))
        with self.assertRaises(ValueError):
            point.build_problem(public_fixture(), arm, "calibrated", "none", Q(600), bad)
        with self.assertRaises(ValueError):
            point.build_problem(public_fixture(), arm, "calibrated", "partial", Q(600), node)
        for status in ("split", "excluded_shared", "excluded_calibration"):
            bad = node_fixture(status=status)
            with self.assertRaises(ValueError):
                point.build_problem(public_fixture(), arm_fixture(nodes=(bad,)), "calibrated", "none", Q(600), bad)

    def test_joint_node_bias_must_remain_inside_arm_calibration(self):
        node = node_fixture()
        node["t_bounds"][0][1] = point.encode(Q(1, 10))
        with self.assertRaises(ValueError):
            point.build_problem(public_fixture(), arm_fixture("joint", (node,)), "joint", "none", Q(600), node)

    def test_classifier_and_exact_checkers_are_reused_intact(self):
        self.assertIs(point.original, point.variation)
        self.assertIs(point.contractor, point.polytope)
        self.assertIs(point.classify, point.polytope.classify)
        self.assertIs(point.certificate_check, point.polytope.certificate_check)
        self.assertIs(point.witness_check, point.polytope.witness_check)
        public, arm = public_fixture(), arm_fixture()
        with patch.object(point.geometry, "geometry_bounds", return_value=manual_area_intervals()):
            rows, bounds = point.build_problem(public, arm, "calibrated", "none", Q(600))
            result = point.classify_problem(public, arm, "calibrated", "none", Q(600))
        self.assertEqual(result["status"], "FEASIBLE_WITNESS")
        self.assertTrue(point.witness_check(rows, bounds, result["witness"]))
        json.dumps(result, allow_nan=False)
        with patch.object(point.geometry, "geometry_bounds", return_value=manual_area_intervals(False)):
            rows, bounds = point.build_problem(public, arm, "calibrated", "none", Q(600))
            result = point.classify_problem(public, arm, "calibrated", "none", Q(600))
        self.assertEqual(result["status"], "CERTIFIED_INFEASIBLE")
        self.assertTrue(point.certificate_check(rows, bounds, result["certificate"]))

    def test_zero_margin_float_candidate_is_unknown_not_excluded(self):
        rows = (("upper", (Q(1),), Q(1)), ("lower", (Q(1),), Q(1)))
        bounds = ((Q(0), Q(1)),)
        fake = SimpleNamespace(status=0, success=True, fun=1e-12, x=np.array([0.0, 1e-12]),
                               ineqlin=SimpleNamespace(marginals=np.array([-0.5, -0.5])))
        with patch.object(point, "build_problem", return_value=(rows, bounds)), \
             patch.object(point.polytope, "linprog", return_value=fake):
            result = point.classify_problem(public_fixture(), arm_fixture(), "calibrated", "none", Q(600))
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(result["certificate"])
        self.assertIsNone(result["witness"])

    def test_deadline_is_forwarded_without_reset_or_retry(self):
        rows, bounds = (), ((Q(0), Q(1)),)
        deadline = time.monotonic() - 1
        with patch.object(point, "build_problem", return_value=(rows, bounds)), \
             patch.object(point, "classify", wraps=point.classify) as classify:
            result = point.classify_problem(public_fixture(), arm_fixture(), "calibrated", "none", Q(600),
                                            deadline=deadline)
        classify.assert_called_once_with(rows, bounds, deadline=deadline)
        self.assertEqual(result["status"], "UNKNOWN")

    def test_global_exact_witness_passes_containing_original_rows_and_leaf_bounds(self):
        public = public_fixture()
        public["observation"] = [point.encode(Q(1))] * point.variation.RGB_COUNT
        lo, depth, hi = Q(4799, 8), Q(600), Q(4801, 8)
        calibration = ((Q(0), Q(1, 50)),) * 3
        for method in point.METHODS:
            for branch in point.BRANCHES:
                with self.subTest(method=method, branch=branch):
                    areas = point.geometry.geometry_bounds(branch, Q(1, 10), lo, hi, Q(80))
                    old_rows = point.variation.necessary_rows(
                        (Q(1),) * point.variation.RGB_COUNT, areas, method, clipped=True,
                        drift_bound=Q(1, 10), calibration_intervals=calibration)
                    initial = ((Q(0), Q(1)),) * 9 + (calibration if method == "joint" else ())
                    contraction = point.variation.propagate(old_rows, initial, rounds=8)
                    self.assertFalse(contraction["excluded"])
                    node = node_fixture(lo=lo, hi=hi)
                    node["darkness_bounds"] = contraction["bounds"][:9]
                    node["t_bounds"] = (contraction["bounds"][9:] if method == "joint"
                                        else encoded_pairs(calibration))
                    arm = arm_fixture(method)
                    arm["branches"][branch]["trace"] = [node]
                    arm["branches"][branch]["retained_ids"] = [node["id"]]
                    global_result = point.classify_problem(public, arm, method, branch, depth)
                    self.assertEqual(global_result["status"], "FEASIBLE_WITNESS")
                    self.assertTrue(point.witness_check(old_rows, contraction["bounds"],
                                                        global_result["witness"]))
                    local_rows, local_bounds = point.build_problem(public, arm, method, branch, depth, node)
                    self.assertTrue(point.witness_check(local_rows, local_bounds, global_result["witness"]))

    def test_aggregation_unknown_empty_and_different_problem_states(self):
        self.assertEqual(point.summarize([]), "SOURCE_EXCLUDED")
        self.assertEqual(point.summarize([], empty_status="NO_ORIGINAL_LEAF"), "NO_ORIGINAL_LEAF")
        self.assertEqual(point.summarize(["CERTIFIED_INFEASIBLE"] * 2), "CERTIFIED_INFEASIBLE")
        self.assertEqual(point.summarize(["CERTIFIED_INFEASIBLE", "UNKNOWN"]), "UNKNOWN")
        self.assertEqual(point.summarize(["UNKNOWN"]), "UNKNOWN")
        self.assertEqual(point.summarize(["CERTIFIED_INFEASIBLE", "FEASIBLE_WITNESS", "UNKNOWN"]),
                         "FEASIBLE_WITNESS")

    def test_aggregation_consumes_all_entries_and_rejects_hidden_invalid_state(self):
        seen = []
        def statuses():
            for status in ("FEASIBLE_WITNESS", "UNKNOWN", "CERTIFIED_INFEASIBLE"):
                seen.append(status)
                yield status
        self.assertEqual(point.summarize(statuses()), "FEASIBLE_WITNESS")
        self.assertEqual(len(seen), 3)
        with self.assertRaises(ValueError):
            point.summarize(["FEASIBLE_WITNESS", "BAD"])
        with self.assertRaises(ValueError):
            point.summarize([{"status": "FEASIBLE_WITNESS", "certificate": {}, "witness": []}])
        with self.assertRaises(TypeError):
            point.summarize("UNKNOWN")

    def test_public_inputs_and_frozen_arm_parameters_are_validated(self):
        self.assertEqual(point.exact("3/7"), Q(3, 7))
        self.assertEqual(point.exact(point.encode(Q(3, 7))), Q(3, 7))
        public, arm = public_fixture(), arm_fixture()
        bad = deepcopy(public)
        bad["baseline"] = point.encode(Q(70))
        with self.assertRaises(ValueError):
            point.build_problem(bad, arm, "calibrated", "none", Q(600))
        bad = deepcopy(arm)
        bad["drift_bound"] = point.encode(Q(0))
        with self.assertRaises(ValueError):
            point.build_problem(public, bad, "calibrated", "none", Q(600))
        with self.assertRaises(ValueError):
            point.build_problem(public, arm, "joint", "none", Q(600))
        with self.assertRaises(TypeError):
            point.build_problem(public, arm, "calibrated", "none", 600.0)
        bad = deepcopy(public)
        bad["observation"] = bad["observation"][:3]
        with self.assertRaises(ValueError):
            point.build_problem(bad, arm, "calibrated", "none", Q(600))

    def test_cache_clear_delegates_to_original_function(self):
        with patch.object(point.variation, "clear_geometry_cache") as clear:
            point.clear_geometry_cache()
        clear.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
