"""Hand fixtures for immutable pipeline math and information order."""
import ast
from fractions import Fraction as Q
import unittest
from a11_io_contract import ROOT,target
import run_accumulation as run
import audit_accumulation as audit
class PipelineTests(unittest.TestCase):
    def test_exact_target(self):target()
    def test_counterbalance(self):
        self.assertEqual(run.arm_order(0),tuple(reversed(run.arm_order(1))))
    def test_fixed_hull(self):
        sets=[dict(observation_id='hand',arm=a,outer_intervals=[['601','602'],['604','605']]) for a in run.solver.ARMS]
        tasks=[dict(task_id=str(x),observation_id='hand',incumbent=str(x)) for x in (600,603,606)]
        out=run.selections(tasks,sets)
        self.assertEqual([audit.check.exact(r['selected']) for r in out],[601,601,603,603,605,605])
        self.assertEqual(audit.pairs([[{'numerator':600,'denominator':1},{'numerator':601,'denominator':1}]]),((Q(600),Q(601)),))
    def test_nesting_not_only_endpoints(self):
        self.assertFalse(audit.nested([['1','5']],[['1','2'],['4','5']]))
        self.assertTrue(audit.nested([['1','2']],[['0','3']]))
    def test_groups(self):
        f={'case':'plain','truth':'9007/14'};p={'variation_index':1}
        self.assertEqual(audit.groups(f,p,Q(0)),['all','correct'])
        self.assertIn('primary',audit.groups(f,p,Q(1)))
    def test_mse_external_denominator(self):
        rows=[dict(arm='a',unit_id='u',groups=['all'],initial_ae_mm='2',final_ae_mm='1')]
        cell=next(r for r in audit.summarize(rows) if r['unit']=='ALL')
        self.assertEqual(cell['initial_MSE_mm2'],'4');self.assertEqual(cell['final_MSE_mm2'],'1')
        self.assertEqual(cell['MSE_reduction_percent'],75)
    def test_static_parse_order(self):
        source=(ROOT/'run_accumulation.py').read_text()
        self.assertLess(source.index("save(OUT/'sets.json'"),source.index("tasks=load(OLD9/'tasks.json')"))
        source=(ROOT/'audit_accumulation.py').read_text()
        self.assertLess(source.index("save(OUT/'verification/PUBLIC_GATE.json'"),source.index('facts={'))
        self.assertNotIn('accumulation_solver',source)
        ast.parse(source)
if __name__=='__main__':unittest.main()
