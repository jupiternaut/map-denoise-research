from pathlib import Path
import unittest
from unittest.mock import patch
import execute_appearance as e


class EntryTests(unittest.TestCase):
    def test_existing_output_refused_without_run_or_failure_writes(self):
        with patch.object(Path,'exists',return_value=True), patch.object(e.runner,'run') as run, patch.object(e.runner,'save') as save:
            with self.assertRaises(RuntimeError):
                e.execute(e.ROOT/'outputs/existing')
            run.assert_not_called()
            save.assert_not_called()

    def test_wrong_output_refused_without_failure_writes(self):
        with patch.object(e.runner,'run') as run, patch.object(e.runner,'save') as save:
            with self.assertRaises(RuntimeError):
                e.execute(Path('/tmp/not-the-target'))
            run.assert_not_called()
            save.assert_not_called()

    def test_concurrent_existing_target_rejection_has_no_ownership_or_writes(self):
        with patch.object(Path,'exists',return_value=False), patch.object(e.runner,'run',side_effect=RuntimeError('existing run')), patch.object(e.runner,'save') as save:
            with self.assertRaises(RuntimeError):
                e.execute(e.ROOT/'outputs/not-owned')
            save.assert_not_called()


if __name__=='__main__':
    unittest.main()
