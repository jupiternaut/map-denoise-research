"""Nonconfirmation runner, information-boundary and checkpoint tests."""
from contextlib import redirect_stdout
import csv
from fractions import Fraction as Q
import hashlib
import inspect
import io
import json
from pathlib import Path
import random
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import run_nuisance as runner


def branch_fixture(intervals):
    trace = [{"id": i, "parent": None, "lo": runner.encode(lo), "hi": runner.encode(hi), "status": "retained_tolerance"}
             for i, (lo, hi) in enumerate(intervals)]
    hull = None if not intervals else [runner.encode(min(lo for lo, _hi in intervals)), runner.encode(max(hi for _lo, hi in intervals))]
    return {"domain": [runner.encode(Q(540)), runner.encode(Q(660))], "epsilon": runner.encode(runner.EPSILON),
            "tolerance": runner.encode(runner.TOLERANCE), "max_boxes": runner.MAX_BOXES,
            "evaluated_boxes": len(intervals), "trace": trace, "retained_ids": list(range(len(trace))), "hull": hull}


class FakeModel:
    def __init__(self, branch, camera_radius=runner.RADIUS, colors=((Q(0), Q(1)),) * 3):
        self.branch, self.camera_radius, self.colors = branch, camera_radius, colors

    def point_prediction(self, depth, offsets=(Q(0), Q(0), Q(0)), colors=None):
        return (Q(1, 4), Q(1, 2), Q(3, 4))

    def enclosure(self, lo, hi):
        return ((Q(0), Q(1)),) * 3


class RunNuisanceTests(unittest.TestCase):
    def test_registered_configuration_and_all_incumbents_inside_prior(self):
        protocol = runner.configuration()
        self.assertEqual(runner.ROOT, Path(__file__).resolve().parent)
        self.assertEqual(protocol["expected_observations"], 54)
        self.assertEqual(protocol["expected_tasks"], 486)
        self.assertEqual(protocol["expected_decisions"], 2430)
        self.assertEqual(protocol["truths_evaluation_only"], ["6199/11", "7853/13", "8999/14"])
        self.assertEqual(protocol["seeds_evaluation_only"], (51001, 51002, 51003))
        self.assertEqual(protocol["random_tape_prefix"], "nuisance-v1")
        self.assertEqual(protocol["methods"], ("keep", "frozen_nominal", "expanded_unanchored", "calibrated_isolated", "calibrated_union"))
        self.assertEqual((protocol["camera_radius_mm"], protocol["epsilon"], protocol["tolerance_mm"], protocol["max_boxes_per_branch"]),
                         ("1/5", "1/100", "15/32", 255))
        self.assertEqual((protocol["cpu_threads"], protocol["gpu_count"], protocol["wall_time_limit_seconds"]), (1, 0, 1800))
        for truth in runner.TRUTHS:
            for offset in runner.OFFSETS:
                self.assertTrue(Q(540) <= truth + offset <= Q(660))
        self.assertEqual(len(runner.CASES) * len(runner.TRUTHS) * len(runner.SEEDS) * (len(runner.OFFSETS) - 1), 432)

    def test_public_observation_and_private_truth_random_tape(self):
        case, radius, truth, seed = "unit_scene", Q(1, 5), Q(600), 99001
        colors = (Q(1, 4), Q(1, 2), Q(3, 4))
        with patch.object(runner, "ColoredModel", FakeModel):
            public, private = runner.generate_observation(case, "partial", colors, truth, seed, prefix="nuisance-unit-v1")
        self.assertEqual(set(public), {"radius", "observation", "calibration"})
        forbidden = {"case", "kind", "rear_branch", "colors", "truth", "seed", "random_tape", "actual_offsets", "branch"}
        self.assertFalse(forbidden.intersection(public))
        tape = f"nuisance-unit-v1:{case}:{radius}:{truth}:{seed}"
        self.assertEqual(private["random_tape"], tape)
        rng = random.Random(tape)
        offsets = (radius * rng.choice(runner.WEIGHTS), Q(0), radius * rng.choice(runner.WEIGHTS))
        image_noise = tuple(runner.EPSILON * rng.choice(runner.WEIGHTS) for _ in range(3))
        calibration_noise = tuple(runner.EPSILON * rng.choice(runner.WEIGHTS) for _ in range(3))
        self.assertEqual(tuple(map(runner.decode, private["actual_offsets"])), offsets)
        self.assertEqual(tuple(map(runner.decode, private["image_noise"])), image_noise)
        self.assertEqual(tuple(map(runner.decode, private["calibration_noise"])), calibration_noise)
        self.assertEqual(tuple(map(runner.decode, public["observation"])), tuple(c + n for c, n in zip(colors, image_noise)))
        self.assertEqual(tuple(item["source"] for item in public["calibration"]), runner.SAMPLE_SOURCES)
        self.assertEqual(tuple(runner.decode(item["value"]) for item in public["calibration"]), tuple(c + n for c, n in zip(colors, calibration_noise)))

    def test_actual_generator_model_and_calibration_contract_nonconfirmation_scene(self):
        colors = (Q(1, 4), Q(1, 2), Q(3, 4))
        public, private = runner.generate_observation("unit_scene", "partial", colors, Q(600), 99001, prefix="nuisance-unit-v1")
        self.assertEqual(len(public["observation"]), 432)
        expected = runner.ColoredModel("partial").point_prediction(Q(600), tuple(map(runner.decode, private["actual_offsets"])), colors=colors)
        self.assertEqual(tuple(map(runner.decode, public["observation"])), tuple(x + runner.decode(n) for x, n in zip(expected, private["image_noise"])))
        for actual, (lo, hi) in zip(colors, runner.calibration_intervals(public["calibration"])):
            self.assertLessEqual(lo, actual)
            self.assertLessEqual(actual, hi)

    def test_material_interval_clipping_and_source_labels(self):
        calibration = [{"source": source, "value": runner.encode(v)} for source, v in zip(runner.SAMPLE_SOURCES, (Q(-1, 100), Q(1, 2), Q(101, 100)))]
        self.assertEqual(runner.calibration_intervals(calibration), ((Q(0), Q(0)), (Q(49, 100), Q(51, 100)), (Q(1), Q(1))))
        wrong_order = list(reversed(calibration))
        with self.assertRaises(ValueError):
            runner.calibration_intervals(wrong_order)
        invalid = [{"source": source, "value": runner.encode(Q(2))} for source in runner.SAMPLE_SOURCES]
        with self.assertRaises(ValueError):
            runner.calibration_intervals(invalid)

    def test_fixed_information_and_model_budget_by_arm(self):
        calibration = [{"source": source, "value": runner.encode(Q(1, 3))} for source in runner.SAMPLE_SOURCES]
        specs = runner.arm_specifications(calibration)
        self.assertEqual(specs["frozen_nominal"][0], ("none",))
        self.assertEqual(specs["frozen_nominal"][1][:2], ((Q(0), Q(0)),) * 2)
        self.assertEqual(specs["expanded_unanchored"], (runner.BRANCHES, ((Q(0), Q(1)),) * 3))
        self.assertEqual(specs["calibrated_isolated"][0], ("none",))
        self.assertEqual(specs["calibrated_union"][0], runner.BRANCHES)
        self.assertEqual(specs["calibrated_union"][1], specs["calibrated_isolated"][1])
        self.assertEqual(list(inspect.signature(runner.solve_arm).parameters), ["observation", "radius", "allowed_branches", "color_intervals"])
        self.assertEqual(list(inspect.signature(runner.select_action).parameters), ["outer", "incumbent", "keep"])

    def test_union_merges_branch_leaves_and_keeps_disjoint_gaps(self):
        branches = {"none": branch_fixture(((Q(550), Q(560)),)),
                    "partial": branch_fixture(((Q(555), Q(570)), (Q(600), Q(610)))),
                    "full": branch_fixture(())}
        result = runner.union_outer(branches)
        intervals = [[runner.decode(x) for x in pair] for pair in result["retained_intervals"]]
        self.assertEqual(intervals, [[Q(550), Q(570)], [Q(600), Q(610)]])
        self.assertEqual(list(map(runner.decode, result["hull"])), [Q(550), Q(610)])
        self.assertEqual(result["retained_branches"], ["none", "partial"])
        self.assertEqual(len(result["source_leaves"]), 3)
        self.assertFalse(runner.contains_truth(result, Q(580)))
        self.assertTrue(runner.contains_truth(result, Q(570)))
        empty = runner.union_outer({"none": branch_fixture(())})
        self.assertIsNone(empty["hull"])
        self.assertEqual(empty["retained_intervals"], [])
        self.assertEqual(empty["retained_branches"], [])

    def test_solve_arm_preserves_full_branch_records_and_calls_frozen_hybrid(self):
        with patch.object(runner, "ColoredModel", FakeModel), \
             patch.object(runner, "solve_outer", side_effect=lambda obs, enc, **kwargs: branch_fixture(((Q(590), Q(610)),))) as solve:
            record = runner.solve_arm((Q(0),) * 3, runner.RADIUS, runner.BRANCHES, ((Q(0), Q(1)),) * 3)
        self.assertEqual(tuple(record["branches"]), runner.BRANCHES)
        self.assertEqual(solve.call_count, 3)
        for call in solve.call_args_list:
            self.assertEqual(call.kwargs, {"epsilon": runner.EPSILON, "tolerance": runner.TOLERANCE, "max_boxes": 255})
        self.assertEqual(set(record["branch_seconds"]), set(runner.BRANCHES))
        self.assertEqual(runner.select_action(record["outer"], Q(589)), runner.certificate_hybrid(record["outer"], Q(589)))
        self.assertEqual(runner.select_action(record["outer"], Q(600), keep=True)["status"], "KEEP")

    def test_real_model_interval_solver_hybrid_integration_with_tiny_test_budget(self):
        public, _ = runner.generate_observation("unit_scene", "none", (Q(1, 4), Q(1, 2), Q(3, 4)), Q(600), 99001, prefix="nuisance-unit-v1")
        with patch.object(runner, "MAX_BOXES", 1):
            record = runner.solve_arm(tuple(map(runner.decode, public["observation"])), runner.RADIUS,
                                      ("none",), runner.calibration_intervals(public["calibration"]))
        self.assertEqual(record["branches"]["none"]["evaluated_boxes"], 1)
        self.assertTrue(runner.contains_truth(record["outer"], Q(600)))
        self.assertEqual(runner.select_action(record["outer"], Q(600))["status"], "KEEP")

    def test_source_lock_covers_new_and_all_imported_historical_sources(self):
        files = runner.input_sources()
        for p, h in files.items():
            self.assertTrue(Path(p).is_absolute())
            self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(), h)
        for p in (runner.ROOT / "run_nuisance.py", runner.ROOT / "test_run_nuisance.py", runner.ROOT / "nuisance_model.py", runner.ROOT / "PROTOCOL.md",
                  runner.HYBRID_ROOT / "hybrid.py", runner.HYBRID_ROOT / "PROTOCOL.md", runner.DEPENDENCY_ROOT / "run_challenge.py",
                  runner.DEPENDENCY_ROOT / "source_snapshot/continuous_world/interval_solver.py",
                  runner.DEPENDENCY_ROOT / "source_snapshot/finite_world/assets/lovelace_patch.json"):
            self.assertIn(str(p), files)
        for module in list(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                p = Path(filename).resolve()
                if p.suffix in (".py", ".json") and any(p.is_relative_to(r) for r in (runner.HYBRID_ROOT, runner.DEPENDENCY_ROOT)):
                    self.assertIn(str(p), files)
        self.assertFalse(any(Path(p).is_relative_to(r / "outputs") for p in files for r in (runner.HYBRID_ROOT, runner.DEPENDENCY_ROOT)))
        import run_challenge
        self.assertEqual(run_challenge.ROOT, runner.DEPENDENCY_ROOT)

    def test_fresh_output_guard_never_overwrites_existing_or_historical_path(self):
        with tempfile.TemporaryDirectory(prefix="runner-test-", dir=runner.ROOT) as temporary:
            output = Path(temporary)
            with self.assertRaises(FileExistsError):
                runner.run_with_deadline(output)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(runner.validate_output(output / "fresh"), output / "fresh")
        for root in (runner.HYBRID_ROOT, runner.DEPENDENCY_ROOT, runner.ROOT):
            target = root if root == runner.ROOT else root / "outputs/forbidden-nuisance"
            with self.assertRaises(ValueError):
                runner.validate_output(target)

    def miniature_run(self, output, interrupt=False):
        source_lock = runner.input_sources()
        real_evaluate = runner.evaluate
        def evaluate(*args):
            self.assertEqual(json.loads((output / "decisions.json").read_text()), args[4])
            self.assertEqual(len(args[4]), 10)
            return real_evaluate(*args)
        with patch.multiple(runner, CASES=(("unit_fixture", "none", (Q(1, 4), Q(1, 2), Q(3, 4))),),
                            TRUTHS=(Q(600),), SEEDS=(99001,), OFFSETS=(Q(0), Q(1, 4)), MAX_BOXES=0), \
             patch.object(runner, "ColoredModel", FakeModel), \
             patch.object(runner, "input_sources", return_value=source_lock), \
             patch.object(runner, "evaluate", side_effect=evaluate), redirect_stdout(io.StringIO()):
            if interrupt:
                with patch.object(runner, "solve_arm", side_effect=TimeoutError("synthetic deadline")):
                    return runner.run_with_deadline(output)
            return runner.run_with_deadline(output)

    def test_miniature_recording_flow_scoring_counts_samples_and_public_fields(self):
        before = {p: h for p, h in runner.input_sources().items() if not Path(p).is_relative_to(runner.ROOT)}
        with tempfile.TemporaryDirectory(prefix="runner-test-", dir=runner.ROOT) as temporary:
            output = Path(temporary) / "fresh"
            summary = self.miniature_run(output)
            self.assertEqual((summary["observation_configurations"], summary["task_count"], summary["decision_count"]), (1, 2, 10))
            status = json.loads((output / "RUN_STATUS.json").read_text())
            self.assertEqual((status["status"], status["observations_complete"], status["decisions_complete"]), ("COMPLETE", 1, 10))
            observations = json.loads((output / "observations.json").read_text())
            self.assertEqual(set(observations[0]), {"observation_id", "radius", "observation", "calibration"})
            tasks = json.loads((output / "tasks.json").read_text())
            self.assertTrue(all(set(t) == {"task_id", "observation_id", "incumbent"} for t in tasks))
            rows = json.loads((output / "evaluation/ROWS.json").read_text())
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(r["covered"] for r in rows))
            self.assertEqual({r["state_group"] for r in rows}, {"correct", "small"})
            self.assertFalse(any({"colors", "truth", "rear_branch", "random_tape", "actual_offsets"}.intersection(r) for r in rows))
            self.assertEqual(summary["sample_counts"]["auxiliary_samples_total"], 3)
            self.assertEqual(summary["sample_counts"]["auxiliary_samples_used_by_method"],
                             {"keep": 0, "frozen_nominal": 0, "expanded_unanchored": 0, "calibrated_isolated": 3, "calibrated_union": 3})
            self.assertEqual(set(summary["per_method_timing"]), set(runner.METHODS))
            self.assertEqual(len(json.loads((output / "evaluation/COVERAGE.json").read_text())), 5)
            with (output / "evaluation/SUMMARY.csv").open() as stream:
                exported_summary = list(csv.DictReader(stream))
            self.assertEqual(len(exported_summary), len(summary["rows"]))
            self.assertTrue({"covered_observations", "improved", "harm", "same", "mae_mm", "mse_mm2",
                             "mean_hull_width_mm", "incompatible", "unresolved_budget_leaves", "auxiliary_samples_used"}.issubset(exported_summary[0]))
            with (output / "evaluation/TIMING.csv").open() as stream:
                exported_timing = list(csv.DictReader(stream))
            self.assertEqual({r["method"] for r in exported_timing}, set(runner.METHODS))
            for p in output.rglob("*.json"):
                json.loads(p.read_text())
            lock = json.loads((output / "SOURCE_LOCK.json").read_text())
            self.assertFalse(lock["historical_outputs_used"])
            self.assertEqual(lock["new_experiment_root"], str(runner.ROOT))
        after = {p: h for p, h in runner.input_sources().items() if not Path(p).is_relative_to(runner.ROOT)}
        self.assertEqual(before, after)

    def test_timeout_keeps_valid_empty_checkpoint_and_never_scores_partial_observation(self):
        with tempfile.TemporaryDirectory(prefix="runner-test-", dir=runner.ROOT) as temporary:
            output = Path(temporary) / "fresh"
            with self.assertRaises(TimeoutError):
                self.miniature_run(output, interrupt=True)
            status = json.loads((output / "RUN_STATUS.json").read_text())
            self.assertEqual((status["status"], status["observations_complete"], status["decisions_complete"]), ("INCOMPLETE", 0, 0))
            self.assertEqual(json.loads((output / "decisions.json").read_text()), [])
            self.assertEqual(json.loads((output / "outers.json").read_text()), [])
            self.assertFalse((output / "evaluation/ROWS.json").exists())
            self.assertFalse((output / "evaluation/SUMMARY.json").exists())

    def test_signal_during_checkpoint_is_delivered_after_all_json_is_complete(self):
        handler, original = signal.getsignal(signal.SIGALRM), Path.write_text
        injected = False
        def write(path, data, *args, **kwargs):
            nonlocal injected
            if path.name == "RUN_STATUS.json" and not injected:
                injected = True
                original(path, "", *args, **kwargs)
                signal.raise_signal(signal.SIGALRM)
            return original(path, data, *args, **kwargs)
        with tempfile.TemporaryDirectory(prefix="runner-test-", dir=runner.ROOT) as temporary:
            output = Path(temporary)
            signal.signal(signal.SIGALRM, runner.deadline_reached)
            try:
                with patch.object(Path, "write_text", write), self.assertRaises(TimeoutError):
                    runner.save_progress(output, [{"done": True}], [{"done": True}], [], time.perf_counter())
                self.assertEqual(json.loads((output / "RUN_STATUS.json").read_text())["observations_complete"], 1)
                self.assertEqual(json.loads((output / "decisions.json").read_text()), [{"done": True}])
                self.assertEqual(json.loads((output / "outers.json").read_text()), [{"done": True}])
            finally:
                signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    unittest.main()
