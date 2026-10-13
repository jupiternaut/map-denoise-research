"""Fast nonconfirmation checks of shared-run registration and recording."""
from contextlib import redirect_stdout
import csv
from fractions import Fraction as Q
import hashlib
import inspect
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import run_shared as runner


def fixture_entry():
    branches = {branch: runner.old.solve_outer((Q(0),), lambda lo, hi: ((Q(0), Q(1)),), max_boxes=0)
                for branch in runner.BRANCHES}
    return {"branches": branches, "outer": runner.union_outer(branches),
            "branch_seconds": {b: 0.0 for b in branches}, "outer_seconds": 0.0}


class RunSharedTests(unittest.TestCase):
    def test_new_registration_four_arms_and_complete_counts(self):
        protocol = runner.configuration()
        self.assertEqual(runner.ROOT, Path(__file__).resolve().parent)
        self.assertEqual(protocol["truths_evaluation_only"], ["6203/11", "7861/13", "9007/14"])
        self.assertEqual(protocol["seeds_evaluation_only"], (61001, 61002, 61003))
        self.assertEqual(protocol["random_tape_prefix"], "shared-appearance-v1")
        self.assertEqual(protocol["methods"], ("keep", "channelwise_union", "shared_union", "calibrated_union"))
        self.assertEqual((protocol["expected_observations"], protocol["expected_tasks"], protocol["expected_decisions"]), (54, 486, 1944))
        self.assertEqual((protocol["expected_correct_tasks"], protocol["expected_small_tasks"]), (54, 432))
        self.assertEqual((protocol["max_boxes_per_branch"], protocol["shared_constraint_max_rounds"]), (255, 8))
        self.assertEqual((protocol["cpu_threads"], protocol["gpu_count"], protocol["wall_time_limit_seconds"]), (1, 0, 1800))
        for truth in runner.TRUTHS:
            for offset in runner.OFFSETS:
                self.assertTrue(Q(540) <= truth + offset <= Q(660))

    def test_public_raw_arms_use_same_images_priors_and_no_calibration(self):
        image = (Q(1, 2),) * 3
        samples = [{"source": source, "value": runner.encode(Q(1, 4))} for source in runner.SAMPLE_SOURCES]
        with patch.object(runner, "solve_channelwise", side_effect=lambda *a, **k: fixture_entry()) as channel, \
             patch.object(runner, "solve_shared", side_effect=lambda *a, **k: fixture_entry()) as shared:
            entries = runner.public_arm_outers(image, runner.RADIUS, samples)
        self.assertEqual(tuple(entries), runner.METHODS)
        self.assertEqual(channel.call_args_list[0].args, (image, runner.RADIUS, runner.BRANCHES, runner.PUBLIC_COLORS))
        self.assertEqual(channel.call_args_list[1].args, (image, runner.RADIUS, runner.BRANCHES, runner.calibration_intervals(samples)))
        self.assertEqual(shared.call_args.args, (image, runner.RADIUS, runner.BRANCHES))
        self.assertEqual(shared.call_args.kwargs, {"colors": runner.PUBLIC_COLORS})
        self.assertEqual(list(inspect.signature(runner.public_arm_outers).parameters), ["observation", "radius", "calibration"])
        self.assertIs(runner.select_action, runner.old.select_action)

    def test_old_globals_remain_old_and_paths_target_new_experiment(self):
        self.assertEqual(runner.old.ROOT, runner.NUISANCE_ROOT)
        self.assertEqual(runner.old.RANDOM_TAPE_PREFIX, "nuisance-v1")
        self.assertEqual(runner.old.SEEDS, (51001, 51002, 51003))
        self.assertEqual(runner.old.METHODS, ("keep", "frozen_nominal", "expanded_unanchored", "calibrated_isolated", "calibrated_union"))
        self.assertNotEqual(runner.ROOT, runner.old.ROOT)
        with tempfile.TemporaryDirectory(prefix="runner-test-", dir=runner.ROOT) as temporary:
            existing = Path(temporary)
            with self.assertRaises(FileExistsError):
                runner.run_with_deadline(existing)
            self.assertEqual(list(existing.iterdir()), [])
        for root in (runner.NUISANCE_ROOT, runner.HYBRID_ROOT, runner.DEPENDENCY_ROOT):
            with self.assertRaises(ValueError):
                runner.validate_output(root / "outputs/forbidden-shared")

    def test_source_lock_is_absolute_covers_science_and_excludes_outputs(self):
        files = runner.input_sources()
        for path, digest in files.items():
            self.assertTrue(Path(path).is_absolute())
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
        for path in (runner.ROOT / "run_shared.py", runner.ROOT / "shared_solver.py", runner.ROOT / "test_run_shared.py",
                     runner.ROOT / "test_shared_solver.py", runner.ROOT / "PROTOCOL.md", runner.NUISANCE_ROOT / "run_nuisance.py",
                     runner.NUISANCE_ROOT / "nuisance_model.py", runner.HYBRID_ROOT / "hybrid.py",
                     runner.DEPENDENCY_ROOT / "run_challenge.py", runner.DEPENDENCY_ROOT / "source_snapshot/finite_world/assets/lovelace_patch.json"):
            self.assertIn(str(path), files)
        for name in ("shared_verification.py", "test_shared_verification.py"):
            if (runner.ROOT / name).exists():
                self.assertIn(str(runner.ROOT / name), files)
        self.assertFalse(any(Path(p).is_relative_to(root / "outputs") for p in files
                             for root in (runner.ROOT, runner.NUISANCE_ROOT, runner.HYBRID_ROOT, runner.DEPENDENCY_ROOT)))
        self.assertFalse(any(Path(p).name.startswith(("analy", "report")) for p in files if Path(p).parent == runner.ROOT))

    def miniature_run(self, output, interrupt=False):
        source_lock = runner.input_sources()
        real_evaluate = runner.evaluate
        generated = []
        actual_generate = runner.generate_observation
        def generate(*args, **kwargs):
            generated.append(kwargs)
            return actual_generate(*args, **kwargs)
        def evaluate(*args):
            self.assertEqual(json.loads((output / "decisions.json").read_text()), args[4])
            self.assertEqual(len(args[4]), 8)
            return real_evaluate(*args)
        with patch.multiple(runner, CASES=(("unit_fixture", "partial", (Q(1, 4), Q(1, 2), Q(3, 4))),),
                            TRUTHS=(Q(600),), SEEDS=(99011,), OFFSETS=(Q(0), Q(1, 4))), \
             patch.object(runner, "generate_observation", side_effect=generate), \
             patch.object(runner, "solve_channelwise", side_effect=lambda *a, **k: fixture_entry()), \
             patch.object(runner, "solve_shared", side_effect=TimeoutError("synthetic deadline") if interrupt else lambda *a, **k: fixture_entry()), \
             patch.object(runner, "input_sources", return_value=source_lock), \
             patch.object(runner, "evaluate", side_effect=evaluate), redirect_stdout(io.StringIO()):
            summary = runner.run_with_deadline(output)
        self.assertEqual(generated, [{"radius": runner.RADIUS, "prefix": "shared-appearance-v1"}])
        return summary

    def test_miniature_flow_records_new_tape_all_actions_then_scores(self):
        before = {p: h for p, h in runner.input_sources().items() if not Path(p).is_relative_to(runner.ROOT)}
        with tempfile.TemporaryDirectory(prefix="runner-test-", dir=runner.ROOT) as temporary:
            output = Path(temporary) / "fresh"
            summary = self.miniature_run(output)
            self.assertEqual((summary["observation_configurations"], summary["task_count"], summary["decision_count"]), (1, 2, 8))
            self.assertEqual(json.loads((output / "RUN_STATUS.json").read_text())["status"], "COMPLETE")
            private = json.loads((output / "evaluation/truth.json").read_text())[0]
            self.assertTrue(private["random_tape"].startswith("shared-appearance-v1:unit_fixture:"))
            public = json.loads((output / "observations.json").read_text())[0]
            self.assertEqual(set(public), {"observation_id", "radius", "observation", "calibration"})
            self.assertTrue(all(set(t) == {"task_id", "observation_id", "incumbent"} for t in json.loads((output / "tasks.json").read_text())))
            rows = json.loads((output / "evaluation/ROWS.json").read_text())
            self.assertEqual({r["seed"] for r in rows}, {99011})
            self.assertFalse(any({"truth", "colors", "rear_branch", "random_tape", "actual_offsets"}.intersection(r) for r in rows))
            self.assertEqual(summary["sample_counts"]["auxiliary_samples_used_by_method"],
                             {"keep": 0, "channelwise_union": 0, "shared_union": 0, "calibrated_union": 3})
            self.assertTrue({"overall", "by_case", "by_information", "by_seed", "by_case_seed"}.issubset({r["grouping"] for r in summary["rows"]}))
            self.assertEqual(set(summary["per_method_timing"]), set(runner.METHODS))
            for name in ("ROWS.csv", "COVERAGE.csv", "SUMMARY.csv", "TIMING.csv", "PAIRED.csv"):
                with (output / "evaluation" / name).open() as stream:
                    self.assertTrue(list(csv.DictReader(stream)))
            for path in output.rglob("*.json"):
                json.loads(path.read_text())
            self.assertEqual(len(json.loads((output / "outers.json").read_text())[0]["arms"]), 4)
        after = {p: h for p, h in runner.input_sources().items() if not Path(p).is_relative_to(runner.ROOT)}
        self.assertEqual(before, after)

    def test_timeout_checkpoint_never_scores_an_unfinished_observation(self):
        with tempfile.TemporaryDirectory(prefix="runner-test-", dir=runner.ROOT) as temporary:
            output = Path(temporary) / "fresh"
            with self.assertRaises(TimeoutError):
                self.miniature_run(output, interrupt=True)
            status = json.loads((output / "RUN_STATUS.json").read_text())
            self.assertEqual((status["status"], status["observations_complete"], status["decisions_complete"]), ("INCOMPLETE", 0, 0))
            self.assertEqual(json.loads((output / "decisions.json").read_text()), [])
            self.assertEqual(json.loads((output / "outers.json").read_text()), [])
            self.assertFalse((output / "evaluation/ROWS.json").exists())

    def test_alarm_during_checkpoint_preserves_readable_record(self):
        old_handler, original_write = signal.getsignal(signal.SIGALRM), Path.write_text
        injected = False
        def write(path, data, *args, **kwargs):
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
                with patch.object(Path, "write_text", write), self.assertRaises(TimeoutError):
                    runner.save_progress(output, [{"done": True}], [{"done": True}], [], time.perf_counter())
                self.assertEqual(json.loads((output / "RUN_STATUS.json").read_text())["observations_complete"], 1)
                self.assertEqual(json.loads((output / "decisions.json").read_text()), [{"done": True}])
                self.assertEqual(json.loads((output / "outers.json").read_text()), [{"done": True}])
            finally:
                signal.signal(signal.SIGALRM, old_handler)

    def test_smoke_uses_only_nonconfirmation_parameters_and_both_raw_arms(self):
        real_generator = runner.generate_observation
        with patch.object(runner, "generate_observation", wraps=real_generator) as generate, \
             patch.object(runner, "solve_channelwise", side_effect=lambda *a, **k: fixture_entry()) as channel, \
             patch.object(runner, "solve_shared", side_effect=lambda *a, **k: fixture_entry()) as shared, redirect_stdout(io.StringIO()):
            runner.smoke()
        self.assertEqual(generate.call_args.args, ("smoke_only", "partial", (Q(1, 4), Q(1, 2), Q(3, 4)), Q(600), 99011))
        self.assertEqual(generate.call_args.kwargs, {"radius": runner.RADIUS, "prefix": "shared-appearance-smoke-v1"})
        self.assertEqual(channel.call_count, 1)
        self.assertEqual(shared.call_count, 1)


if __name__ == "__main__":
    unittest.main()
