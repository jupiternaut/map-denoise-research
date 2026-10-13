"""Nonconfirmation tests of grouping, exact witnesses and complete budgets."""
from fractions import Fraction as Q
import importlib.util
import inspect
import random
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import grouped_solver as solver


UNIT_GEOMETRY = (Q(1), Q(1), Q(0), Q(0), Q(0), Q(0))
ZERO_GEOMETRY = (Q(0),) * 6


def contains(outer, truth):
    return any(solver.decode(lo) <= truth <= solver.decode(hi)
               for lo, hi in outer["retained_intervals"])


class PublicFakeModel:
    """Small exact enclosure fixture; no generating metadata enters a solve."""
    branch, camera_radius = "none", Q(1, 5)
    depth_range = (Q(540), Q(660))
    colors = solver.PUBLIC_COLORS

    def __init__(self, pixels=2):
        self.observation_size = 3 * pixels

    def enclosure(self, _lo, _hi):
        return ((Q(0), Q(1)),) * self.observation_size


class GroupedSolverTests(unittest.TestCase):
    def test_registered_partition_and_gray_row_invariants(self):
        self.assertEqual(solver.PIXELS_PER_VIEW, 48)
        self.assertEqual(solver.nuisance_model.GRID.width * solver.nuisance_model.GRID.height, 48)
        geometry = (UNIT_GEOMETRY,) * 144
        image = tuple(y for pixel in range(144) for y in
                      (Q(pixel, 1000), Q(pixel, 1000) + Q(1, 100), Q(pixel, 1000) + Q(2, 100)))
        global_rows = solver.grouped_rows(image, geometry, "global_gray")
        pixel_rows = solver.grouped_rows(image, geometry, "pixel_gray")
        view_rows = solver.grouped_rows(image, geometry, "view_gray")
        self.assertEqual(tuple(global_rows), ("global",))
        self.assertEqual(tuple(pixel_rows), tuple(f"p{p}" for p in range(144)))
        self.assertEqual(tuple(view_rows), ("v0", "v1", "v2"))
        self.assertEqual({len(rows) for rows in pixel_rows.values()}, {2})
        self.assertEqual({len(rows) for rows in view_rows.values()}, {96})
        for groups in (pixel_rows, view_rows):
            self.assertEqual(tuple(row for rows in groups.values() for row in rows), global_rows["global"])
        self.assertEqual(view_rows["v1"][0], pixel_rows["p48"][0])
        self.assertEqual(view_rows["v2"][0], pixel_rows["p96"][0])
        for upper, lower in zip(global_rows["global"][::2], global_rows["global"][1::2]):
            self.assertEqual((upper[0], lower[0]), ("upper", "lower"))
            self.assertEqual(upper[1], UNIT_GEOMETRY[::2])
            self.assertEqual(lower[1], UNIT_GEOMETRY[1::2])
            self.assertEqual(upper[2], lower[2])
            self.assertTrue(all(isinstance(value, Q) for row in (upper, lower) for value in (*row[1], row[2])))

    def test_global_rgb_has_three_separate_channel_blocks(self):
        image = tuple(y for _ in range(144) for y in (Q(1, 5), Q(1, 2), Q(4, 5)))
        rows = solver.grouped_rows(image, (UNIT_GEOMETRY,) * 144, "global_rgb")
        self.assertEqual(tuple(rows), ("c0", "c1", "c2"))
        self.assertEqual({len(group) for group in rows.values()}, {288})
        for channel, y in enumerate((Q(1, 5), Q(1, 2), Q(4, 5))):
            self.assertEqual(rows[f"c{channel}"][0][2], 1 - y + solver.EPSILON)
            self.assertEqual(rows[f"c{channel}"][1][2], 1 - y - solver.EPSILON)
            self.assertFalse(solver.propagate(rows[f"c{channel}"])["excluded"])
        for mode in ("global_gray", "pixel_gray", "view_gray"):
            gray = solver.grouped_rows(image, (UNIT_GEOMETRY,) * 144, mode)
            self.assertTrue(any(solver.propagate(group)["excluded"] for group in gray.values()))

    def test_sharing_conflict_disappears_when_pixels_or_views_separate(self):
        geometry = tuple(UNIT_GEOMETRY if pixel in (0, 48) else ZERO_GEOMETRY for pixel in range(49))
        image = tuple(y for pixel in range(49) for y in
                      ((Q(3, 4),) * 3 if pixel == 0 else (Q(1, 4),) * 3 if pixel == 48 else (Q(1),) * 3))
        for mode in ("global_gray", "global_rgb"):
            self.assertTrue(any(solver.propagate(rows)["excluded"]
                                for rows in solver.grouped_rows(image, geometry, mode, epsilon=Q(0)).values()))
        for mode in ("pixel_gray", "view_gray"):
            self.assertFalse(any(solver.propagate(rows)["excluded"]
                                 for rows in solver.grouped_rows(image, geometry, mode, epsilon=Q(0)).values()))

    def test_shared_trace_saves_group_and_local_row_certificate(self):
        image = (Q(3, 4),) * 3 + (Q(1, 4),) * 3
        with patch.object(solver.nuisance_model, "_geometry_bounds", return_value=(UNIT_GEOMETRY,) * 2):
            outer = solver.solve_branch(image, PublicFakeModel(), mode="global_gray", max_boxes=1,
                                        rounds=8, tolerance=Q(120), epsilon=Q(0))
            independent = solver.solve_branch(image, PublicFakeModel(), mode="pixel_gray", max_boxes=1,
                                              rounds=8, tolerance=Q(120), epsilon=Q(0))
        node = outer["trace"][0]
        self.assertEqual((node["status"], node["group_id"]), ("excluded_shared", "global"))
        self.assertEqual(node["certificate"]["contradiction"], {"row": 3})
        self.assertEqual(node["certificate"]["steps"],
                         [{"row": 0, "variable": 0, "side": "upper", "value": solver.encode(Q(1, 4))},
                          {"row": 1, "variable": 0, "side": "lower", "value": solver.encode(Q(1, 4))}])
        self.assertEqual(independent["trace"][0]["status"], "retained_tolerance")
        self.assertEqual(outer["shared_tests"], 1)
        self.assertEqual(independent["shared_tests"], 2)

    def test_incomplete_contractor_and_zero_rounds_retain(self):
        rows = (("lower", (Q(1), Q(1), Q(0)), Q(1)),
                ("lower", (Q(1), Q(0), Q(1)), Q(1)),
                ("lower", (Q(0), Q(1), Q(1)), Q(1)),
                ("upper", (Q(1), Q(1), Q(1)), Q(1)))
        self.assertFalse(solver.propagate(rows)["excluded"])
        conflict = (("upper", (Q(1), Q(0), Q(0)), Q(1, 4)),
                    ("lower", (Q(1), Q(0), Q(0)), Q(3, 4)))
        result = solver.propagate(conflict, rounds=0)
        self.assertEqual((result["excluded"], result["rounds"], result["updates"]), (False, 0, 0))
        self.assertTrue(solver.propagate(conflict)["excluded"])

    def test_propagator_matches_historical_exact_algorithm_without_mutation(self):
        path = solver.ROOT.parent / "lovelace-shared-appearance-20261011T052300Z/shared_solver.py"
        spec = importlib.util.spec_from_file_location("read_only_shared_reference", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        rng = random.Random(982)
        colors = [[Q(0), Q(1)] for _ in range(3)]
        original = [pair[:] for pair in colors]
        for _ in range(20):
            darkness = tuple(Q(rng.randrange(101), 100) for _ in range(3))
            rows = []
            for _ in range(8):
                coefficients = tuple(Q(rng.randrange(101), 100) for _ in range(3))
                value = sum((a * d for a, d in zip(coefficients, darkness)), Q(0))
                margin = Q(rng.randrange(20), 100)
                rows.extend((("upper", coefficients, value + margin), ("lower", coefficients, value - margin)))
            self.assertEqual(solver.propagate(rows, colors=colors, rounds=8),
                             reference.propagate(rows, colors=colors, rounds=8))
            self.assertFalse(solver.propagate(rows)["excluded"])
        self.assertEqual(colors, original)
        sys.path.insert(0, str(solver.ROOT))

    def test_truth_inclusion_every_mode_small_budget_and_old_action(self):
        from run_nuisance import select_action
        model = solver.ColoredModel("partial", camera_radius=Q(1, 5))
        truth = Q(601)
        image = model.point_prediction(truth, offsets=(Q(1, 10), Q(0), Q(-1, 5)),
                                       colors=(Q(1, 4), Q(1, 2), Q(3, 4)))
        image = tuple(value + ((index % 3) - 1) * solver.EPSILON for index, value in enumerate(image))
        for mode in solver.MODES:
            with self.subTest(mode=mode):
                entry = solver.solve_arm(image, Q(1, 5), mode, max_boxes=7)
                self.assertEqual(tuple(entry["branches"]), solver.BRANCHES)
                self.assertTrue(contains(entry["outer"], truth))
                self.assertEqual(entry["outer"]["max_boxes_per_branch"], 7)
                self.assertEqual(select_action(entry["outer"], truth)["status"], "KEEP")
                decision = select_action(entry["outer"], truth + Q(3, 4))
                self.assertLessEqual(abs(solver.decode(decision["output"]) - truth), Q(3, 4))

    def test_real_full_negative_complete_511_tree_has_no_unresolved(self):
        from run_nuisance import select_action
        model = solver.ColoredModel("full")
        image = model.point_prediction(Q(600), colors=(Q(1, 5),) * 3)
        for mode in solver.MODES:
            with self.subTest(mode=mode):
                branch = solver.solve_branch(image, model, mode=mode, max_boxes=511)
                self.assertEqual(branch["evaluated_boxes"], 511)
                self.assertEqual(len(branch["trace"]), 511)
                self.assertEqual(len(branch["retained_ids"]), 256)
                self.assertFalse(any(node["status"].startswith("excluded") or
                                     node["status"] == "unresolved_budget" for node in branch["trace"]))
                self.assertEqual(branch["hull"], branch["domain"])
                outer = solver.union_outer({"full": branch}, mode=mode)
                self.assertEqual(outer["unresolved_budget_leaves"], 0)
                self.assertEqual(outer["max_boxes_per_branch"], 511)
                self.assertEqual(select_action(outer, Q(600) + Q(3))["status"], "KEEP")
        old_budget = solver.solve_branch(image, model, mode="channel", max_boxes=255)
        self.assertEqual(sum(node["status"] == "unresolved_budget" for node in old_budget["trace"]), 256)

    def test_zero_budget_preserves_all_modes_and_dynamic_union(self):
        image = (Q(1, 2),) * 432
        for mode in solver.MODES:
            entry = solver.solve_arm(image, Q(1, 5), mode, max_boxes=0, rounds=3,
                                     tolerance=Q(1, 2), epsilon=Q(1, 50))
            self.assertEqual(entry["outer"]["hull"], entry["outer"]["domain"])
            self.assertEqual(entry["outer"]["unresolved_budget_leaves"], 3)
            self.assertEqual(entry["outer"]["evaluated_boxes"], 0)
            self.assertEqual(entry["outer"]["max_boxes_per_branch"], 0)
            self.assertEqual(entry["outer"]["epsilon"], solver.encode(Q(1, 50)))
            self.assertEqual(entry["outer"]["tolerance"], solver.encode(Q(1, 2)))
            for branch in entry["branches"].values():
                self.assertEqual((branch["mode"], branch["rounds"], branch["budget"]), (mode, 3, 0))
                self.assertEqual(branch["trace"][0]["status"], "unresolved_budget")
                self.assertEqual(branch["retained_ids"], [0])

    def test_channel_trace_is_old_solver_with_explicit_parameters(self):
        model = PublicFakeModel()
        image = (Q(1, 2),) * 6
        with patch.object(solver, "solve_outer", wraps=solver.solve_outer) as old:
            result = solver.solve_branch(image, model, mode="channel", max_boxes=9, rounds=3,
                                         tolerance=Q(30), epsilon=Q(1, 7))
        self.assertEqual(old.call_args.args, (image, model.enclosure))
        self.assertEqual(old.call_args.kwargs, {"lo": Q(540), "hi": Q(660), "epsilon": Q(1, 7),
                                               "tolerance": Q(30), "max_boxes": 9})
        reference = solver.solve_outer(image, model.enclosure, lo=Q(540), hi=Q(660), epsilon=Q(1, 7),
                                       tolerance=Q(30), max_boxes=9)
        for field, expected in reference.items():
            self.assertEqual(result[field], expected)
        self.assertEqual(result["shared_tests"], 0)

    def test_arm_explicit_parameters_and_no_private_metadata(self):
        names = list(inspect.signature(solver.solve_arm).parameters)
        self.assertEqual(names, ["observation", "radius", "mode", "max_boxes", "rounds", "tolerance", "epsilon"])
        self.assertEqual(inspect.signature(solver.solve_arm).parameters["max_boxes"].default, 511)
        image = (Q(1, 2),) * 432
        with patch.object(solver, "solve_branch", wraps=solver.solve_branch) as branch:
            entry = solver.solve_arm(image, Q(1, 5), "view_gray", max_boxes=0, rounds=3,
                                     tolerance=Q(1, 2), epsilon=Q(1, 50))
        self.assertEqual(branch.call_count, 3)
        for call in branch.call_args_list:
            self.assertEqual(call.args[0], image)
            self.assertEqual(call.args[1].colors, solver.PUBLIC_COLORS)
            self.assertEqual(call.args[1].camera_radius, Q(1, 5))
            self.assertEqual(call.kwargs, {"mode": "view_gray", "max_boxes": 0, "rounds": 3,
                                         "tolerance": Q(1, 2), "epsilon": Q(1, 50)})
        self.assertEqual(set(entry), {"branches", "outer", "branch_seconds", "outer_seconds"})
        self.assertEqual(set(entry["branch_seconds"]), set(solver.BRANCHES))

    def test_grouped_branch_forwards_error_and_round_limit(self):
        image = (Q(1, 2),) * 6
        with patch.object(solver.nuisance_model, "_geometry_bounds", return_value=(UNIT_GEOMETRY,) * 2), \
             patch.object(solver, "grouped_rows", wraps=solver.grouped_rows) as rows, \
             patch.object(solver, "propagate", wraps=solver.propagate) as contractor:
            result = solver.solve_branch(image, PublicFakeModel(), mode="global_rgb", max_boxes=1,
                                         rounds=2, tolerance=Q(120), epsilon=Q(1, 50))
        self.assertEqual(rows.call_args.args, (image, (UNIT_GEOMETRY,) * 2, "global_rgb"))
        self.assertEqual(rows.call_args.kwargs, {"epsilon": Q(1, 50)})
        self.assertEqual(contractor.call_count, 3)
        self.assertTrue(all(call.kwargs == {"colors": solver.PUBLIC_COLORS, "rounds": 2}
                            for call in contractor.call_args_list))
        self.assertEqual(result["trace"][0]["status"], "retained_tolerance")

    def test_cache_clear_covers_geometry_models_and_projection_data(self):
        model = solver.ColoredModel("partial")
        model.enclosure(Q(590), Q(610))
        model.point_prediction(Q(600), colors=(Q(1, 4), Q(1, 2), Q(3, 4)))
        self.assertGreater(solver.nuisance_model._geometry_bounds.cache_info().currsize, 0)
        self.assertGreater(solver.ColoredModel._enclosure_cached.cache_info().currsize, 0)
        solver.clear_geometry_cache()
        for cached in (solver.nuisance_model._front_boxes, solver.nuisance_model._front_area_bounds,
                       solver.nuisance_model._rear_boxes, solver.nuisance_model._geometry_bounds,
                       solver.nuisance_model._point_geometry, solver.ColoredModel._enclosure_cached,
                       solver.layered_model.LayeredModel._point_prediction_cached,
                       solver.layered_model.LayeredModel._enclosure_cached,
                       solver.interval_model.Model._point_prediction_cached,
                       solver.interval_model.Model._triangle_camera_data,
                       solver.interval_model.Model._enclosure_cached):
            self.assertEqual(cached.cache_info().currsize, 0)

    def test_exact_parameter_and_layout_validation(self):
        image = (Q(1, 2),) * 432
        for kwargs in ({"max_boxes": -1}, {"max_boxes": True}, {"rounds": -1},
                       {"tolerance": Q(0)}, {"epsilon": Q(-1)}, {"mode": "missing"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solver.solve_arm(image, Q(1, 5), **{"mode": "channel", **kwargs})
        with self.assertRaises(TypeError):
            solver.solve_arm(image, 0.2, "channel", max_boxes=0)
        with self.assertRaises(TypeError):
            solver.grouped_rows((0.5,) * 3, (UNIT_GEOMETRY,), "pixel_gray")
        with self.assertRaises(ValueError):
            solver.solve_arm((Q(1, 2),) * 3, Q(1, 5), "channel", max_boxes=0)
        with self.assertRaises(ValueError):
            solver.grouped_rows((Q(1, 2),) * 3, (), "global_gray")
        with self.assertRaises(ValueError):
            solver.propagate((("invalid", (Q(1), Q(0), Q(0)), Q(1)),))


if __name__ == "__main__":
    unittest.main()
