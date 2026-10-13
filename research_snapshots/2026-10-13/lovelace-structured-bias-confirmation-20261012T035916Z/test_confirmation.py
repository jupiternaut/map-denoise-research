"""Non-confirmation tests: no seed61004 generation, labels or solver batch."""
import sys
sys.dont_write_bytecode = True
import copy
from fractions import Fraction as Q
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_confirmation as prep
import run_confirmation as run


class BoundaryTests(unittest.TestCase):
    def actions(self):
        return [{'task_id': f'c{i:03d}s{s}', 'method': m, 'reused': False}
            for i in range(1, 19) for m in run.METHODS for s in range(9)]

    def test_complete_saved_gate(self):
        rows = self.actions()
        with tempfile.TemporaryDirectory(prefix='.test-', dir=run.ROOT) as directory:
            out = Path(directory)
            run.save(out/'decisions.json', rows)
            run.complete_saved_actions(out, rows)
            with self.assertRaises(RuntimeError):
                run.complete_saved_actions(out, rows[:-1])
            dup = copy.deepcopy(rows)
            dup[-1] = dup[-2]
            with self.assertRaises(RuntimeError):
                run.complete_saved_actions(out, dup)
            altered = copy.deepcopy(rows)
            altered[-1]['reused'] = True
            with self.assertRaises(RuntimeError):
                run.complete_saved_actions(out, altered)

    def test_method_counts_not_just_population(self):
        rows = self.actions()
        for r in rows:
            if r['method'] == 'joint':
                r['method'] = 'other'
        with patch.object(run, 'load', return_value=rows):
            with self.assertRaises(RuntimeError):
                run.complete_saved_actions(run.ROOT, rows)

    def test_saved_readback_not_in_memory_only(self):
        rows = self.actions()
        with patch.object(run, 'load', return_value=[]):
            with self.assertRaises(RuntimeError):
                run.complete_saved_actions(run.ROOT, rows)

    def test_public_solver_wrapper_only_four_arguments(self):
        with patch.object(run.frozen, 'execute_new', return_value={'test': True}) as fn:
            self.assertEqual(run.execute_new(['1/2'], '1/5', 'joint', True), {'test': True})
            fn.assert_called_once_with(['1/2'], '1/5', 'joint', True)

    def test_bias_then_single_clip(self):
        image = [run.encode(Q(1,2)) for _ in range(432)]
        image[288] = run.encode(Q(-1,100))
        result = prep.transform(image, run.helpers.decode, run.encode)
        self.assertEqual(run.exact(result[288]), 0)
        self.assertEqual(run.exact(result[0]), Q(49,100))
        self.assertEqual(run.exact(result[144]), Q(1,2))

    def test_path_and_progress_limits(self):
        with self.assertRaises(ValueError):
            run.save(run.ROOT.parent/'not-allowed.json', {})
        with self.assertRaises(ValueError):
            run.save(run.ROOT/'summary.json', {}, replace=True)

    def test_no_repair_can_be_claimed_from_equal_errors(self):
        rows = [{'task_id': 'c001s1', 'method': m, 'initial_ae_mm': '1/4',
                 'ae_mm': '1/4', 'case': 'test', 'seed': -1, 'state_group': 'small'} for m in run.METHODS]
        pairs = run.paired_rows(rows)
        counts = run.paired_claims(rows, pairs, {})
        self.assertEqual((counts['better'], counts['same'], counts['worse'], counts['new_repairs']), (0,1,0,0))


if __name__ == '__main__':
    unittest.main()
