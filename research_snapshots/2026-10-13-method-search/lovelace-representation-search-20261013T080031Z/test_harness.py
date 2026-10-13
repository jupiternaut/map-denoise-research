"""Pre-freeze development-only harness checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from run_experiment import ROOT,run_job,aggregate

class HarnessTests(unittest.TestCase):
    def test_cold_process_and_single_incumbent(self):
        data=json.loads((ROOT/'task_bundle/dev.json').read_text())
        p=data['observations'][0]
        tasks=[{'task_id':t['task_id'],'incumbent':t['incumbent']} for t in data['tasks']
               if t['observation_id']==p['observation_id']]
        original=deepcopy(p['observation'])
        with tempfile.TemporaryDirectory(prefix='a12-smoke-',dir=ROOT) as d:
            for i,arm in enumerate(('fixed','library','free')):
                job={'arm':arm,'budget_seconds':.3,'observation_id':p['observation_id'],
                     'observation':p['observation'],'tasks':tasks}
                receipt=run_job(job,Path(d),i)
                self.assertEqual(receipt['returncode'],0,receipt['stderr'])
                result=json.loads((ROOT/receipt['result']).read_text())
                self.assertIsNone(result['prepare_failure'])
                self.assertEqual(len(result['predictions']),7)
                self.assertTrue(all(540<=r['selected']<=660 for r in result['predictions']))
                self.assertTrue(all(r['failure'] is None for r in result['predictions']))
            self.assertEqual(original,p['observation'])

    def test_arithmetic(self):
        rows=[dict(observation_id='a',initial_ae=2.,initial_se=4.,final_ae=1.,final_se=1.,failure=False),
              dict(observation_id='a',initial_ae=0.,initial_se=0.,final_ae=0.,final_se=0.,failure=False)]
        s=aggregate(rows)
        self.assertEqual(s['mse'],.5);self.assertEqual(s['mse_improvement_percent'],75.)
        self.assertEqual((s['improved'],s['same'],s['harmed'],s['worlds']),(1,1,0,1))

if __name__=='__main__': unittest.main(verbosity=2)
