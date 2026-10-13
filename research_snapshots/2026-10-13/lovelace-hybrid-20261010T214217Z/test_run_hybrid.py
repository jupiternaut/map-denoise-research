"""Focused runner checks; synthetic fixtures never run confirmation."""
from contextlib import redirect_stdout
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

import numpy as np

import run_hybrid as runner


def fixture_outer(left=Q(599), right=Q(601)):
    return {"hull": [runner.encode(left), runner.encode(right)],
            "evaluated_boxes": 0, "trace": []}


class FakeBaseline:
    def __init__(self, kind, events=None):
        if events is not None:
            events.append(("compile", kind))
        self.compile_seconds = 0.0
        self.depth_grid = (Q(600),)
        self.forecasts = np.zeros((1, 1, 3, 1))

    def grid_losses(self, observation, radius):
        return np.zeros(1)

    def select(self, observation, radius, incumbent, grid_losses):
        return {"status": "KEEP", "output": runner.encode(incumbent),
                "gain_lower": None, "reason": "finite_pose_profiled_residual",
                "incumbent_residual": 0.0, "minimum_residual": 0.0}


class RunHybridTests(unittest.TestCase):
    def test_registered_new_configuration(self):
        protocol = runner.configuration()
        self.assertEqual(runner.ROOT, Path(__file__).resolve().parent)
        self.assertNotEqual(runner.ROOT, runner.DEPENDENCY_ROOT)
        self.assertEqual(protocol["truths_evaluation_only"], ["6217/11", "7864/13", "8441/13"])
        self.assertEqual(protocol["seeds"], (41001, 41002, 41003))
        self.assertEqual(protocol["random_tape_prefix"], "hybrid-v1")
        self.assertEqual(protocol["expected_observations"], 135)
        self.assertEqual(protocol["expected_tasks"], 1215)
        self.assertEqual(protocol["expected_decisions"], 6075)
        self.assertEqual(protocol["methods"], ("keep", "certificate_midpoint", "certificate_projection",
                                               "certificate_hybrid", "pose_profiled_residual"))
        self.assertEqual(protocol["wall_time_limit_seconds"], 1800)
        self.assertEqual(protocol["cpu_threads"], 1)
        self.assertEqual(protocol["gpu_count"], 0)
        self.assertEqual((protocol["epsilon"], protocol["tolerance_mm"], protocol["max_boxes"]), ("1/100", "15/32", 511))

    def test_random_tape_uses_new_prefix_and_exact_registered_weights(self):
        class Model:
            def point_prediction(self, truth, offsets):
                return (Q(2, 3), Q(4, 7), Q(5, 9))
        truth, radius, seed = runner.TRUTHS[0], Q(1, 5), runner.SEEDS[0]
        rng = random.Random(f"hybrid-v1:rectangle:{radius}:{truth}:{seed}")
        expected_offsets = (radius * rng.choice(runner.WEIGHTS), Q(0), radius * rng.choice(runner.WEIGHTS))
        prediction = Model().point_prediction(truth, expected_offsets)
        expected_obs = tuple(x + runner.EPSILON * rng.choice(runner.WEIGHTS) for x in prediction)
        with patch.object(runner, "model_for", return_value=Model()):
            _, offsets, image, observed = runner.observation_for("rectangle", radius, truth, seed)
        self.assertEqual(offsets, expected_offsets)
        self.assertEqual(image, prediction)
        self.assertEqual(observed, expected_obs)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(image, observed)), runner.EPSILON)

    def test_source_lock_is_absolute_and_covers_all_imported_old_sources(self):
        files = runner.input_sources()
        for path in files:
            self.assertTrue(Path(path).is_absolute())
            self.assertEqual(files[path], hashlib.sha256(Path(path).read_bytes()).hexdigest())
        required = (runner.ROOT / "run_hybrid.py", runner.ROOT / "hybrid.py",
                    runner.ROOT / "test_run_hybrid.py", runner.ROOT / "PROTOCOL.md",
                    runner.DEPENDENCY_ROOT / "challenge.py", runner.DEPENDENCY_ROOT / "run_challenge.py",
                    runner.DEPENDENCY_ROOT / "layered_model.py",
                    runner.DEPENDENCY_ROOT / "source_snapshot/continuous_world/interval_solver.py",
                    runner.DEPENDENCY_ROOT / "source_snapshot/continuous_world/interval_model.py",
                    runner.DEPENDENCY_ROOT / "source_snapshot/finite_world/assets/lovelace_patch.json")
        for path in required:
            self.assertIn(str(path.resolve()), files)
        for module in list(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if path.is_relative_to(runner.DEPENDENCY_ROOT) and path.suffix in (".py", ".json"):
                    self.assertIn(str(path), files)
        self.assertFalse(any(Path(p).is_relative_to(runner.DEPENDENCY_ROOT / "outputs") for p in files))

    def test_old_helpers_keep_their_original_root(self):
        import run_challenge
        self.assertEqual(run_challenge.ROOT, runner.DEPENDENCY_ROOT)
        self.assertIs(runner.aggregates, run_challenge.aggregates)
        self.assertIs(runner.save_progress, run_challenge.save_progress)
        self.assertIs(runner.deadline_reached, run_challenge.deadline_reached)

    def test_output_requires_fresh_path_in_exact_new_root(self):
        with tempfile.TemporaryDirectory(prefix="runner-test-", dir=runner.ROOT) as temporary:
            existing = Path(temporary)
            marker = existing / "marker"
            marker.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                runner.run(existing)
            self.assertEqual(marker.read_bytes(), b"preserve")
            fresh = existing / "fresh"
            self.assertEqual(runner.validate_output(fresh), fresh)
            self.assertFalse(fresh.exists())
        with self.assertRaises(ValueError):
            runner.validate_output(runner.DEPENDENCY_ROOT / "outputs/new-hybrid-results")
        with self.assertRaises(ValueError):
            runner.validate_output(runner.ROOT)

    def test_selector_has_no_evaluation_inputs_and_integrates_all_hybrid_branches(self):
        self.assertEqual(list(inspect.signature(runner.select_task).parameters),
                         ["outer", "incumbent", "observation", "radius", "baseline", "grid_losses"])
        outer = fixture_outer(Q(600), Q(602))
        for incumbent, expected in ((Q(597), "midpoint"), (Q(59975, 100), "projection"), (Q(601), "keep")):
            actions, timing = runner.select_task(outer, incumbent, (Q(0),), Q(0), FakeBaseline("rectangle"), np.zeros(1))
            self.assertEqual(tuple(actions), runner.METHODS)
            self.assertEqual(set(timing), set(runner.METHODS))
            hybrid = actions["certificate_hybrid"]
            self.assertEqual(hybrid["branch"], expected)
            self.assertEqual(hybrid, runner.certificate_hybrid(outer, incumbent))
            self.assertEqual((hybrid["status"] == "MOVE"), (actions["certificate_projection"]["status"] == "MOVE"))
            if expected == "midpoint":
                self.assertEqual({k: v for k, v in hybrid.items() if k != "branch"}, actions["certificate_midpoint"])
        incompatible = {**outer, "hull": None}
        actions, _ = runner.select_task(incompatible, Q(600), (Q(0),), Q(0), FakeBaseline("rectangle"), np.zeros(1))
        self.assertEqual(actions["certificate_hybrid"]["branch"], "incompatible")

    def test_alarm_during_record_write_leaves_valid_complete_json(self):
        handler = signal.getsignal(signal.SIGALRM)
        original_write = Path.write_text
        injected = False
        def interrupted_write(path, data, *args, **kwargs):
            nonlocal injected
            if path.name == "RUN_STATUS.json" and not injected:
                injected = True
                original_write(path, "", *args, **kwargs)
                signal.raise_signal(signal.SIGALRM)
            return original_write(path, data, *args, **kwargs)
        with tempfile.TemporaryDirectory(prefix="runner-test-", dir=runner.ROOT) as temporary:
            output = Path(temporary)
            signal.signal(signal.SIGALRM, runner.deadline_reached)
            try:
                with patch.object(Path, "write_text", interrupted_write), self.assertRaises(TimeoutError):
                    runner.save_progress(output, [{"completed": True}], [{"completed": True}], [], time.perf_counter())
                self.assertEqual(json.loads((output / "RUN_STATUS.json").read_text())["observations_complete"], 1)
                self.assertEqual(json.loads((output / "decisions.json").read_text()), [{"completed": True}])
                self.assertEqual(json.loads((output / "outers.json").read_text()), [{"completed": True}])
                self.assertEqual(json.loads((output / "TIMING_PARTIAL.json").read_text()), [])
            finally:
                signal.signal(signal.SIGALRM, handler)

    def miniature_run(self, output, events, stop_on_selection=False):
        class Model:
            enclosure = staticmethod(lambda left, right: ((Q(0), Q(1)),))
        def observation(kind, radius, truth, seed):
            events.append(("observation", kind))
            return Model(), (Q(0), Q(0), Q(0)), (Q(0),), (Q(0),)
        real_evaluate = runner.evaluate
        def evaluate(*args):
            recorded = json.loads((output / "decisions.json").read_text())
            self.assertEqual(len(recorded), 10)
            self.assertEqual(recorded, args[4])
            events.append(("evaluation", len(recorded)))
            return real_evaluate(*args)
        with patch.multiple(runner, KINDS=("rectangle",), RADII=(Q(0),), TRUTHS=(Q(600),),
                            SEEDS=(41001,), OFFSETS=(Q(0), Q(1, 4))), \
             patch.object(runner, "ProfiledResidual", side_effect=lambda kind: FakeBaseline(kind, events)), \
             patch.object(runner, "observation_for", side_effect=observation), \
             patch.object(runner, "model_for", return_value=Model()), \
             patch.object(runner, "solve_outer", return_value=fixture_outer()), \
             patch.object(runner, "evaluate", side_effect=evaluate), redirect_stdout(io.StringIO()):
            if stop_on_selection:
                with patch.object(runner, "select_task", side_effect=TimeoutError("synthetic deadline")):
                    return runner.run_with_deadline(output)
            return runner.run(output)

    def test_miniature_recording_flow_compiles_first_saves_then_evaluates(self):
        before = {p: h for p, h in runner.input_sources().items() if Path(p).is_relative_to(runner.DEPENDENCY_ROOT)}
        events = []
        with tempfile.TemporaryDirectory(prefix="runner-test-", dir=runner.ROOT) as temporary:
            output = Path(temporary) / "fresh"
            summary = self.miniature_run(output, events)
            self.assertEqual(events, [("compile", "rectangle"), ("observation", "rectangle"), ("evaluation", 10)])
            self.assertEqual((summary["observation_configurations"], summary["task_count"], summary["decision_count"]), (1, 2, 10))
            status = json.loads((output / "RUN_STATUS.json").read_text())
            self.assertEqual((status["status"], status["observations_complete"], status["decisions_complete"]), ("COMPLETE", 1, 10))
            rows = json.loads((output / "evaluation/ROWS.json").read_text())
            self.assertEqual(set(rows[0]), {"task_id", "observation_id", "kind", "radius", "method", "offset",
                "state_group", "prior_position", "incumbent", "output", "truth", "status", "initial_ae_mm", "ae_mm", "actual_gain_mm2"})
            self.assertTrue(all("branch" not in row for row in rows))
            decisions = json.loads((output / "decisions.json").read_text())
            self.assertTrue(all(("branch" in d) == (d["method"] == "certificate_hybrid") for d in decisions))
            self.assertEqual(summary["hybrid_branch_counts"], {"keep": 2})
            self.assertEqual(set(summary["information_strata"]), {"informative", "hidden"})
            for filename in ("PROTOCOL.json", "SOURCE_LOCK.json", "FORECAST_INFO.json", "TIMING.json", "TIMING_PARTIAL.json"):
                json.loads((output / filename).read_text())
            lock = json.loads((output / "SOURCE_LOCK.json").read_text())
            self.assertEqual(lock["new_experiment_root"], str(runner.ROOT))
            self.assertFalse(lock["historical_outputs_used"])
        after = {p: h for p, h in runner.input_sources().items() if Path(p).is_relative_to(runner.DEPENDENCY_ROOT)}
        self.assertEqual(before, after)

    def test_timeout_records_incomplete_and_does_not_evaluate_partial_observation(self):
        with tempfile.TemporaryDirectory(prefix="runner-test-", dir=runner.ROOT) as temporary:
            output = Path(temporary) / "fresh"
            with self.assertRaises(TimeoutError):
                self.miniature_run(output, [], stop_on_selection=True)
            status = json.loads((output / "RUN_STATUS.json").read_text())
            self.assertEqual((status["status"], status["observations_complete"], status["decisions_complete"]), ("INCOMPLETE", 0, 0))
            self.assertFalse((output / "evaluation/ROWS.json").exists())
            self.assertFalse((output / "evaluation/SUMMARY.json").exists())
            self.assertFalse((output / "decisions.json").exists())


if __name__ == "__main__":
    unittest.main()
