"""Non-confirmation integration checks, without registered seed generation."""
import sys
sys.dont_write_bytecode = True
from fractions import Fraction as Q
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_baseline as prepare
import run_baseline as runner
import verify_baseline as verifier

class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = prepare.module('baseline_test_generator',prepare.SHARED/'run_shared.py')
        cls.observations,cls.tasks,cls.truths = prepare.build_population(cls.shared,1901)

    def test_population_and_common_noise(self):
        self.assertEqual((len(self.observations),len(self.tasks),len(self.truths)),(36,324,36))
        self.assertEqual(len({t['task_id'] for t in self.tasks}),324)
        self.assertEqual(len({t['base_task_id'] for t in self.tasks}),162)
        for a,b in zip(self.truths[::2],self.truths[1::2]):
            for key in ('truth','colors','actual_offsets','image_noise','actual_view_bias','random_tape'):
                self.assertEqual(a[key],b[key])
        for item in self.observations:
            self.assertNotIn('calibration',item)

    def test_b70_generator_identity(self):
        for item,private in zip(self.observations[::2],self.truths[::2]):
            branch,truth,colors = private['rear_branch'],self.shared.decode(private['truth']),tuple(map(self.shared.decode,private['colors']))
            public,_ = self.shared.generate_observation(private['case'],branch,colors,truth,1901,radius=Q(1,5),prefix=prepare.PREFIX)
            expected = [self.shared.encode(min(Q(1),max(Q(0),self.shared.decode(y)+(-Q(1,100),Q(0),Q(1,100))[i//144])))
                        for i,y in enumerate(public['observation'])]
            self.assertEqual(item['observation'],expected)

    def test_independent_paired_regeneration(self):
        images = {o['observation_id']:o for o in self.observations}
        tasks = {t['task_id']:t for t in self.tasks}
        with patch.object(verifier,'SEED',1901), patch.object(verifier,'load',return_value=self.truths):
            facts,witnesses = verifier.verify_private_population(images,tasks)
        self.assertEqual(list(facts.values()),self.truths)
        self.assertEqual(len(witnesses),36)

    def test_independent_source_audit(self):
        self.assertEqual(verifier.audit_runner_source()['status'],'PASS')

    def test_complete_actions_gate_counts_and_readback(self):
        decisions = [{'task_id':t['task_id'],'method':m,'reused':False} for t in self.tasks for m in runner.METHODS]
        with patch.object(runner,'load',return_value=decisions):
            runner.complete_saved_actions(Path('/unused'),decisions)
        with patch.object(runner,'load',return_value=decisions[:-1]):
            with self.assertRaises(RuntimeError):
                runner.complete_saved_actions(Path('/unused'),decisions)
        with self.assertRaises(RuntimeError):
            runner.complete_saved_actions(Path('/unused'),decisions[:-1])

    def test_private_parse_after_saved_actions(self):
        import ast
        tree = ast.parse(Path(runner.__file__).read_text())
        body = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run')
        calls = [(n.lineno,n.func.id) for n in ast.walk(body) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        gate = min(n for n,name in calls if name=='complete_saved_actions')
        parse = min(n for n,name in calls if name=='private_truths')
        self.assertLess(gate,parse)
        self.assertEqual([name for _,name in calls].count('private_truths'),1)

    def test_baseline_pair_statistics(self):
        rows = []
        facts = {}
        for unit,error in zip(runner.UNITS,('2','1')):
            oid = 'p001b'+unit[-2:]
            facts[oid] = {'truth':'9007/14'}
            for method in runner.METHODS:
                rows.append({'task_id':oid+'s1','base_task_id':'p001s1','observation_id':oid,'unit_id':unit,
                    'method':method,'case':'uniform_isolated','seed':1901,'state_group':'small','ae_mm':error,'initial_ae_mm':'2'})
        pairs = runner.baseline_pairs(rows)
        result = runner.paired_baseline_claims(rows,pairs,facts)
        self.assertEqual(result['better'],3)
        self.assertEqual(result['new_repairs'],3)
        self.assertEqual(result['deep_informative']['joint']['total'],1)
        self.assertEqual(pairs[0]['mse_gain_mm2'],'3')

    def test_independent_metric_reconstruction_and_negative_preserved(self):
        facts = {r['observation_id']:r for r in self.truths}
        checked = {'hull':(Q(540),Q(660)),'intervals':[(Q(540),Q(660))],
                   'retained_branches':[],'evaluated':0,'budget_leaves':0}
        rows = []
        for task in self.tasks:
            private = facts[task['observation_id']]
            for method in runner.METHODS:
                action = {'task_id':task['task_id'],'method':method,'reused':False,'unit_id':private['unit_id'],
                          'status':'KEEP','output':task['incumbent']}
                if method=='joint' and task['task_id']=='p001b80s1':
                    action['status'],action['output'] = 'MOVE',runner.encode(runner.exact(private['truth'])+4)
                rows.append(verifier.make_row(task,private,action,checked))
        self.assertEqual(runner.aggregate_rows(rows,self.truths),verifier.aggregate_rows(rows,facts))
        self.assertEqual(runner.method_pairs(rows),verifier.paired_method_rows(rows))
        self.assertEqual(runner.baseline_pairs(rows),verifier.paired_baseline_rows(rows))
        pairs = runner.baseline_pairs(rows)
        self.assertEqual(runner.paired_baseline_claims(rows,pairs,facts),verifier.paired_baseline_summary(rows,pairs,facts))
        self.assertEqual(sum(p['worse'] for p in pairs),1)
        self.assertGreater(runner.exact(next(r for r in rows if r['task_id']=='p001b80s1' and r['method']=='joint')['ae_mm']),Q(1,4))

if __name__=='__main__':
    unittest.main()

