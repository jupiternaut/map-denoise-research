"""Handwritten A9 geometry, whole-tree, and coordinate-selection fixtures."""
from copy import deepcopy
from fractions import Fraction as Q
import time
import unittest
from unittest.mock import patch

import continuous_solver as solver


def fixture(branch="partial", depth=Q(600), offsets=(Q(-3, 25), Q(0), Q(1, 11)),
            darkness=(Q(1, 2),) * 9, ts=(Q(1, 50), Q(1, 100), Q(0)),
            ta=(Q(0), Q(1, 100), Q(1, 50))):
    a = solver.anchor
    raw = a.pose._prediction(depth, branch, offsets, darkness, ts)
    public = {"observation_id": "handwritten-photo", "radius": a.encode(Q(1, 5)),
              "baseline": a.encode(Q(80)), "clipped": True,
              "observation": [a.encode(a._clip(value)) for value in raw]}
    oldarm = {"appearance_model": "bounded", "drift_bound": a.encode(Q(1, 10)),
              "calibration_intervals": [[a.encode(t), a.encode(t)] for t in ts]}
    k, noise = Q(1, 200), (Q(-1, 200), Q(0), Q(1, 200))
    u = tuple((-center - offset - k - error) / 1000
              for center, offset, error in zip((-Q(80), Q(0), Q(80)), offsets, noise))
    anchors = {"observation_id": "handwritten-photo", "camera": {
        "u": list(map(a.encode, u)), "Xf": a.encode(0), "Zf": a.encode(1000),
        "survey_bound": a.encode(Q(1, 100)), "localization_bound": a.encode(Q(1, 100))},
        "appearance": {"coupons": [a.encode(a._clip(Q(1) - d + a.BETA - ta[index // 3]))
                                    for index, d in enumerate(darkness)],
                       "references": [a.encode(Q(1, 2) + a.BETA - t) for t in ta],
                       "bias_bound": a.encode(Q(1, 100)), "transfer_bound": a.encode(Q(1, 200)),
                       "noise_bound": a.encode(Q(1, 200))}}
    coordinates = list(map(a.encode, darkness + ts + ta))
    return public, oldarm, anchors, coordinates


def unknown(*args, **kwargs):
    return {"status": "UNKNOWN", "certificate": None, "witness": None,
            "lp_status": None, "lp_seconds": 0.0}


def decode_pair(pair):
    return tuple(map(solver.exact, pair))


def check_tree_cover(test, result):
    records = {record["id"]: record for record in result["trace"]}
    test.assertEqual(len(records), len(result["trace"]))
    test.assertEqual(decode_pair(result["domain"]), solver.DOMAIN)
    test.assertEqual(records[0]["parent"], None)
    test.assertEqual((solver.exact(records[0]["lo"]), solver.exact(records[0]["hi"])), solver.DOMAIN)
    retained = []
    for record in result["trace"]:
        lo, hi = solver.exact(record["lo"]), solver.exact(record["hi"])
        if record["status"] == "split":
            left, right = (records[index] for index in record["children"])
            midpoint = (lo + hi) / 2
            test.assertEqual((solver.exact(left["lo"]), solver.exact(left["hi"])), (lo, midpoint))
            test.assertEqual((solver.exact(right["lo"]), solver.exact(right["hi"])), (midpoint, hi))
            test.assertEqual(left["parent"], record["id"])
            test.assertEqual(right["parent"], record["id"])
        elif record["status"].startswith("retained_"):
            retained.append(record["id"])
        else:
            test.assertEqual(record["status"], "excluded_certificate")
            test.assertIsNotNone(record["classification"]["certificate"])
    test.assertEqual(result["retained_ids"], retained)
    union = solver.merge_intervals((records[index]["lo"], records[index]["hi"]) for index in retained)
    test.assertEqual(tuple(map(decode_pair, result["outer_intervals"])), union)


class ContinuousSolverTests(unittest.TestCase):
    def test_frozen_domain_tolerance_source_and_budget(self):
        self.assertEqual(solver.DOMAIN, (Q(540), Q(660)))
        self.assertEqual(solver.MIN_WIDTH, Q(15, 32))
        self.assertEqual(solver.MAX_NODES, 127)
        self.assertEqual(solver.BRANCH_SECONDS, 3.0)
        self.assertEqual(solver.ARM_NAMES, ("none", "camera", "appearance", "both"))

    def test_whole_depth_camera_enclosures_at_rational_interior_worlds(self):
        boxes = (((Q(-1, 5), Q(1, 5)), (Q(0), Q(0)), (Q(-1, 5), Q(1, 5))),
                 ((Q(-13, 100), Q(-11, 100)), (Q(0), Q(0)), (Q(9, 100), Q(11, 100))))
        cells = ((Q(540), Q(660)), (Q(577), Q(1231, 2)), (Q(5999, 10), Q(6001, 10)))
        for branch in solver.BRANCHES:
            for intervals in boxes:
                for lo, hi in cells:
                    enclosure = solver.interval_area_bounds(branch, lo, hi, intervals)
                    generic = solver._rectangle_bounds(branch, lo, hi, intervals)
                    for numerator in range(5):
                        z = lo + (hi - lo) * Q(numerator, 4)
                        for weight in (Q(0), Q(1, 3), Q(1, 2), Q(1)):
                            offsets = tuple(elo + (ehi - elo) * weight for elo, ehi in intervals)
                            points = solver.geometry._point_areas(branch, z, offsets, solver.anchor.BASELINE)
                            for triple, bounds, crude in zip(points, enclosure, generic):
                                for index, area in enumerate(triple):
                                    self.assertLessEqual(crude[2 * index], bounds[2 * index])
                                    self.assertLessEqual(bounds[2 * index], area)
                                    self.assertLessEqual(area, bounds[2 * index + 1])
                                    self.assertLessEqual(bounds[2 * index + 1], crude[2 * index + 1])

    def test_zero_width_is_exact_A8_all_camera_envelope(self):
        intervals = ((Q(-121, 1000), Q(-119, 1000)), (Q(0), Q(0)), (Q(89, 1000), Q(91, 1000)))
        for branch in solver.BRANCHES:
            for depth in (Q(540), Q(577), Q(1231, 2), Q(660)):
                self.assertEqual(solver.interval_area_bounds(branch, depth, depth, intervals),
                                 solver.anchor.area_bounds(branch, depth, intervals))

    def test_narrow_bounds_are_no_looser_than_rectangle_and_strictly_improve(self):
        intervals = solver.DEFAULT_CAMERA_INTERVALS
        lo, hi = Q(5999, 10), Q(6001, 10)
        generic = solver._rectangle_bounds("partial", lo, hi, intervals)
        tightened = solver.interval_area_bounds("partial", lo, hi, intervals)
        self.assertTrue(any(first != second for first, second in zip(generic, tightened)))
        for first, second in zip(generic, tightened):
            for material in range(3):
                self.assertLessEqual(first[2 * material], second[2 * material])
                self.assertLessEqual(second[2 * material + 1], first[2 * material + 1])

    def test_edge_strip_lipschitz_bounds_actual_area_change(self):
        branch, lo, hi = "partial", Q(577), Q(1231, 2)
        qc = (Q(1) / lo + Q(1) / hi) / 2
        zc = Q(1) / qc
        intervals = ((Q(-13, 100), Q(-11, 100)), (Q(0), Q(0)), (Q(9, 100), Q(11, 100)))
        for view, (center, interval) in enumerate(zip(solver.geometry.camera_centers(Q(80)), intervals)):
            for offset in (interval[0], sum(interval) / 2, interval[1]):
                offsets = tuple(offset if index == view else Q(0) for index in range(3))
                midpoint = solver.geometry._point_areas(branch, zc, offsets, Q(80))
                for z in (lo, (lo + hi) / 2, hi):
                    points = solver.geometry._point_areas(branch, z, offsets, Q(80))
                    for pixel_index, (pixel, _area) in enumerate(solver.geometry.PIXELS):
                        index = view * solver.geometry.PIXELS_PER_VIEW + pixel_index
                        for material, rect in enumerate((solver.geometry.LEFT_RECT, solver.geometry.RIGHT_RECT,
                                                         solver.geometry.FRONT_RECT)):
                            bound = solver.rectangle_lipschitz(rect, center, interval, pixel) * abs(Q(1) / z - qc)
                            self.assertLessEqual(abs(points[index][material] - midpoint[index][material]), bound)

    def test_15dim_whole_cell_rows_admit_handwritten_generating_world(self):
        for branch in solver.BRANCHES:
            public, arm, anchors, coordinates = fixture(branch=branch)
            for name in solver.ARM_NAMES:
                rows, bounds = solver.build_interval_problem(public, arm, anchors, name, Q(590), Q(610), branch)
                self.assertEqual(len(bounds), 15)
                self.assertEqual(bounds[-3:], ((Q(0), Q(1, 50)),) * 3)
                self.assertTrue(all(len(coefficients) == 15 for _, coefficients, _ in rows))
                self.assertTrue(solver.witness_check(rows, bounds, coordinates))
                exact_rows, exact_bounds = solver.build_interval_problem(public, arm, anchors, name, 600, 600, branch)
                a8_rows, a8_bounds = solver.anchor.build_problem(public, arm, anchors, name, 600, branch)
                self.assertEqual((exact_rows, exact_bounds), (a8_rows, a8_bounds))

    def test_input_labels_IDs_and_disabled_sections_do_not_affect_method(self):
        public, arm, anchors, _ = fixture()
        changed_public, changed_arm, changed_anchors = deepcopy(public), deepcopy(arm), deepcopy(anchors)
        changed_public.update(observation_id="s0-case-eta", case="winner", eta="secret", s0=True,
                              tasks=[540, 600, 660], truth="private")
        changed_arm["branches"] = {"partial": {"mode": "calibrated", "trace": "ignore"}}
        changed_anchors.update(observation_id="deliberately-unmatched", case="hidden")
        for name in solver.ARM_NAMES:
            expected = solver.build_interval_problem(public, arm, anchors, name, 590, 610, "partial")
            self.assertEqual(solver.build_interval_problem(changed_public, changed_arm, changed_anchors,
                                                          name, 590, 610, "partial"), expected)
        for name, disabled in (("none", ("camera", "appearance")), ("camera", ("appearance",)),
                               ("appearance", ("camera",))):
            bad = deepcopy(changed_anchors)
            for key in disabled:
                bad[key] = {"never_decode": object()}
            self.assertEqual(solver.build_interval_problem(public, arm, bad, name, 590, 610, "partial"),
                             solver.build_interval_problem(public, arm, anchors, name, 590, 610, "partial"))

    def test_BFS_budget_retains_complete_pending_frontier(self):
        public, arm, anchors, _ = fixture()
        with patch.object(solver, "MAX_NODES", 3), patch.object(solver, "classify", side_effect=unknown):
            result = solver.solve_interval_branch(public, arm, anchors, "both", "partial")
        self.assertEqual(result["classified_nodes"], 3)
        self.assertEqual(len(result["trace"]), 7)
        self.assertEqual(result["retained_ids"], [3, 4, 5, 6])
        self.assertEqual(result["termination"], "node_budget")
        self.assertEqual(tuple(map(decode_pair, result["outer_intervals"])), (solver.DOMAIN,))
        check_tree_cover(self, result)

    def test_unknowns_split_until_registered_tolerance(self):
        public, arm, anchors, _ = fixture()
        with patch.object(solver, "MAX_NODES", 1000), patch.object(solver, "MIN_WIDTH", Q(30)), \
             patch.object(solver, "classify", side_effect=unknown):
            result = solver.solve_interval_branch(public, arm, anchors, "none", "partial")
        self.assertEqual(result["classified_nodes"], 7)
        self.assertEqual(result["termination"], "tree_complete")
        self.assertEqual([record["status"] for record in result["trace"][-4:]], ["retained_tolerance"] * 4)
        check_tree_cover(self, result)

    def test_outer_feasible_witness_never_claims_forward_completion(self):
        public, arm, anchors, coordinates = fixture()
        classification = {"status": "FEASIBLE_WITNESS", "certificate": None, "witness": coordinates}
        with patch.object(solver, "MAX_NODES", 1), patch.object(solver, "classify", return_value=classification):
            result = solver.solve_interval_branch(public, arm, anchors, "both", "partial")
        self.assertEqual(result["status"], "OUTER_RETAINED")
        self.assertEqual(result["trace"][0]["status"], "split")
        self.assertNotIn("FORWARD_WITNESS", repr(result))
        check_tree_cover(self, result)

    def test_exact_camera_contradiction_prunes_whole_domain(self):
        public, arm, anchors, _ = fixture()
        anchors["camera"]["u"][1] = solver.encode(Q(-3, 100000))
        result = solver.solve_interval_branch(public, arm, anchors, "camera", "partial")
        self.assertEqual(result["status"], "OUTER_EMPTY")
        self.assertEqual(len(result["trace"]), 1)
        self.assertEqual(result["outer_intervals"], [])
        rows, bounds = solver.build_interval_problem(public, arm, anchors, "camera", 540, 660, "partial")
        self.assertTrue(solver.certificate_check(rows, bounds, result["trace"][0]["classification"]["certificate"]))
        check_tree_cover(self, result)

    def test_invalid_whole_cell_certificate_cannot_prune(self):
        public, arm, anchors, _ = fixture()
        bad = {"status": "CERTIFIED_INFEASIBLE", "certificate": {"invalid": True}, "witness": None}
        with patch.object(solver, "classify", return_value=bad):
            with self.assertRaises(ArithmeticError):
                solver.solve_interval_branch(public, arm, anchors, "both", "partial")

    def test_expired_deadline_retains_root_without_geometry_or_classifier(self):
        public, arm, anchors, _ = fixture()
        with patch.object(solver, "build_interval_problem") as build, patch.object(solver, "classify") as classify:
            result = solver.solve_interval_branch(public, arm, anchors, "both", "partial", time.monotonic() - 1)
        self.assertEqual(result["classified_nodes"], 0)
        self.assertEqual(result["trace"][0]["status"], "retained_deadline")
        self.assertEqual(tuple(map(decode_pair, result["outer_intervals"])), (solver.DOMAIN,))
        build.assert_not_called()
        classify.assert_not_called()

    def test_geometry_hitting_deadline_retains_current_and_remaining_queue(self):
        public, arm, anchors, _ = fixture()
        clock, calls = [100.0], [0]
        def build(*args):
            calls[0] += 1
            clock[0] = 101.0 if calls[0] == 1 else 104.0
            return (), ((Q(0), Q(1)),) * 15
        with patch.object(solver.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(solver, "build_interval_problem", side_effect=build), \
             patch.object(solver, "classify", side_effect=unknown):
            result = solver.solve_interval_branch(public, arm, anchors, "both", "partial", 200.0)
        self.assertEqual(result["classified_nodes"], 1)
        self.assertEqual(result["retained_ids"], [1, 2])
        self.assertEqual([record["status"] for record in result["trace"]],
                         ["split", "retained_deadline", "retained_deadline"])
        self.assertEqual(result["search_budget_seconds"], 3.0)
        check_tree_cover(self, result)

    def test_merge_touching_closed_intervals_and_union_tie(self):
        pairs = ((Q(650), Q(660)), (Q(540), Q(545)), (Q(545), Q(550)))
        self.assertEqual(solver.merge_intervals(pairs), ((Q(540), Q(550)), (Q(650), Q(660))))
        result = solver.select_depth(pairs, 600, "UNION")
        self.assertEqual(solver.exact(result["selected"]), Q(550))
        self.assertFalse(result["model_conditional_nonregression"])
        hull = solver.select_depth(pairs, 600, "HULL")
        self.assertEqual(solver.exact(hull["selected"]), Q(600))
        self.assertTrue(hull["model_conditional_nonregression"])
        # True depth 650 is covered, but nearest UNION moves 600 to 550:
        # error grows from 50 to 100 despite exact outer support membership.
        self.assertGreater(abs(Q(650) - solver.exact(result["selected"])), abs(Q(650) - Q(600)))

    def test_hull_projection_conditional_absolute_and_squared_nonregression(self):
        intervals = ((Q(570), Q(580)), (Q(620), Q(630)))
        for incumbent in (Q(500), Q(570), Q(600), Q(630), Q(700)):
            selected = solver.exact(solver.select_depth(intervals, incumbent, "HULL")["selected"])
            for truth in (Q(570), Q(575), Q(620), Q(630)):
                self.assertLessEqual(abs(selected - truth), abs(incumbent - truth))
                self.assertLessEqual((selected - truth) ** 2, (incumbent - truth) ** 2)
            self.assertEqual(solver.exact(solver.select_depth(intervals, incumbent, "KEEP")["selected"]), incumbent)

    def test_empty_support_keeps_incumbent_and_reports_failure(self):
        for policy in ("HULL", "UNION"):
            result = solver.select_depth([], Q(601, 1), policy)
            self.assertEqual(result["status"], "SUPPORT_FAILED")
            self.assertEqual(solver.exact(result["selected"]), Q(601))
            self.assertFalse(result["moved"])

    def test_exact_input_validation_and_no_file_access(self):
        public, arm, anchors, _ = fixture()
        for lo, hi in ((539, 600), (600, 661), (601, 600), (540.0, 660), (True, 660)):
            with self.assertRaises((TypeError, ValueError)):
                solver.build_interval_problem(public, arm, anchors, "both", lo, hi, "partial")
        with self.assertRaises(ValueError):
            solver.interval_area_bounds("partial", 540, 660, ((Q(0), Q(0)), (Q(1, 100), Q(1, 100)), (Q(0), Q(0))))
        with patch("builtins.open", side_effect=AssertionError("file read forbidden")), \
             patch.object(solver.Path, "read_text", side_effect=AssertionError("file read forbidden")):
            rows, bounds = solver.build_interval_problem(public, arm, anchors, "both", 590, 610, "partial")
            self.assertTrue(solver.witness_check(rows, bounds, fixture()[3]))


if __name__ == "__main__":
    unittest.main()
