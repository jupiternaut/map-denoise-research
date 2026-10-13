import unittest
from fractions import Fraction as Q
from pathlib import Path
from unittest.mock import patch
import run_ablation as run

class RunnerTests(unittest.TestCase):
    def test_fixed_contract_and_separate_inputs(self):
        config=run.configuration()
        self.assertEqual(config['expected_decisions'],2916)
        self.assertEqual(config['expected_observations'],54)
        self.assertEqual(config['max_boxes_per_branch'],511)
        self.assertEqual(len(config['methods']),6)
        self.assertNotIn('calibrated_union',config['methods'])
    def test_parameters_explicit_and_cache_each_arm(self):
        with patch.object(run,'clear_geometry_cache') as clear,patch.object(run,'solve_arm',return_value={}) as solver:
            run.execute_arm([run.encode(Q(1,2))],str(Q(1,5)),'global_rgb511')
        clear.assert_called_once()
        args,kwargs=solver.call_args
        self.assertEqual(args,( (Q(1,2),),Q(1,5),'global_rgb'))
        self.assertEqual(kwargs,{'max_boxes':511,'rounds':8,'tolerance':Q(15,32),'epsilon':Q(1,100)})
    def test_identity_budget_and_no_solver(self):
        entry=run.identity_arm()
        self.assertEqual(entry['outer']['max_boxes_per_branch'],511)
        self.assertEqual(entry['branches'],{})
        self.assertTrue(run.covered(entry['outer'],Q(600)))
    def test_output_identity_and_no_overwrite(self):
        with self.assertRaises(ValueError): run.validate_output('/tmp/substitute')
        with self.assertRaises(ValueError): run.validate_output(run.ROOT)
        with self.assertRaises(FileExistsError): run.validate_output(run.ROOT/'refine-logs')
        self.assertEqual(run.validate_output(run.ROOT/'outputs/test-new'),run.ROOT/'outputs/test-new')
    def test_historical_input_identity(self):
        self.assertEqual(run.DATA,Path('/srv/slam-research/grf/map-denoise/runs/lovelace-shared-appearance-20261011T052300Z/outputs/confirmation-v1'))
        self.assertEqual(len(run.input_lock()),8)
        self.assertEqual(len(run.load(run.DATA/'tasks.json')),486)

if __name__=='__main__': unittest.main()
