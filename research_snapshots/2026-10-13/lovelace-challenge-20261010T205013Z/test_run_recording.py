"""Small checks of actual runner checkpoint and deadline paths, not full runs."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from run_challenge import ROOT, deadline_reached, input_sources, save_progress


class RunRecordingTests(unittest.TestCase):
    def test_completed_checkpoint_is_readable_and_counts_agree(self):
        decisions = [{"task_id": "diagnostic", "method": "keep"}]
        outers = [{"observation_id": "diagnostic", "outer": {"hull": None}}]
        timing = [{"observation_id": "diagnostic", "selector_seconds_by_method": {"keep": .01}}]
        with tempfile.TemporaryDirectory(prefix="lovelace-recording-test-") as temporary:
            output = Path(temporary)
            save_progress(output, decisions, outers, timing, time.perf_counter())
            for filename, expected in (("decisions.json", decisions), ("outers.json", outers),
                                       ("TIMING_PARTIAL.json", timing)):
                self.assertEqual(json.loads((output / filename).read_text()), expected)
            status = json.loads((output / "RUN_STATUS.json").read_text())
            self.assertEqual(status["observations_complete"], 1)
            self.assertEqual(status["decisions_complete"], 1)
            self.assertEqual(status["status"], "RUNNING")

    def test_alarm_handler_really_interrupts_work(self):
        code = ("import signal,time; from run_challenge import deadline_reached; "
                "signal.signal(signal.SIGALRM,deadline_reached); signal.alarm(1); time.sleep(3)")
        answer = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=5)
        self.assertNotEqual(answer.returncode, 0)
        self.assertIn("registered 1800-second CPU wall-time limit reached", answer.stderr)

    def test_signal_during_truncate_is_delivered_after_complete_checkpoint(self):
        original_handler = signal.getsignal(signal.SIGALRM)
        original_write = Path.write_text
        injected = False
        def interrupted_write(path, data, *args, **kwargs):
            nonlocal injected
            if path.name == "RUN_STATUS.json" and not injected:
                injected = True
                original_write(path, "", *args, **kwargs)
                signal.raise_signal(signal.SIGALRM)
            return original_write(path, data, *args, **kwargs)
        with tempfile.TemporaryDirectory(prefix="lovelace-recording-test-") as temporary:
            output = Path(temporary)
            signal.signal(signal.SIGALRM, deadline_reached)
            try:
                with patch.object(Path, "write_text", interrupted_write), self.assertRaises(TimeoutError):
                    save_progress(output, [{"completed": True}], [{"completed": True}], [], time.perf_counter())
                status = json.loads((output / "RUN_STATUS.json").read_text())
                self.assertEqual(status["observations_complete"], 1)
                self.assertEqual(len(json.loads((output / "decisions.json").read_text())), 1)
                self.assertEqual(len(json.loads((output / "outers.json").read_text())), 1)
                self.assertEqual(json.loads((output / "TIMING_PARTIAL.json").read_text()), [])
            finally:
                signal.signal(signal.SIGALRM, original_handler)

    def test_scientific_input_list_covers_runner_models_and_protocol(self):
        names = input_sources()
        self.assertIn("challenge.py", names)
        self.assertIn("layered_model.py", names)
        self.assertIn("run_challenge.py", names)
        self.assertIn("PROTOCOL.md", names)
        self.assertIn("source_snapshot/continuous_world/interval_solver.py", names)


if __name__ == "__main__":
    unittest.main()
