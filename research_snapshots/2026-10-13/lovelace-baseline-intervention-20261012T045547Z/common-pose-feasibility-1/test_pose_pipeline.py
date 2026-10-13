import unittest
from unittest.mock import patch
import run_pose as p

class PipelineTests(unittest.TestCase):
    def test_aggregate_mixed(self):
        self.assertEqual(p.aggregate(['UNKNOWN','FORWARD_WITNESS','INHERITED_INFEASIBLE']),'FORWARD_WITNESS')
        self.assertEqual(p.aggregate(['UNKNOWN','INHERITED_INFEASIBLE','INHERITED_INFEASIBLE']),'UNKNOWN')
        self.assertEqual(p.aggregate(['INHERITED_INFEASIBLE']*3),'INHERITED_INFEASIBLE')
    def test_aggregate_rejects_invalid_even_with_witness(self):
        for v in (['FORWARD_WITNESS','bad','UNKNOWN'],['UNKNOWN']*2,['UNKNOWN']*4):
            with self.assertRaises(ValueError):p.aggregate(v)
    def test_complete_public_gate(self):
        tasks=[{'task_id':str(i)} for i in range(324)]
        records=[{'task_id':t['task_id'],'status':'UNKNOWN','branches':[{'branch':b,'status':'UNKNOWN'} for b in p.BRANCHES]} for t in tasks]
        p.check_records(records,tasks)
        with self.assertRaises(RuntimeError):p.check_records(records[:-1],tasks)
        records[0]['branches']=records[0]['branches'][::-1]
        with self.assertRaises(RuntimeError):p.check_records(records,tasks)
    def test_truth_parse_occurs_only_after_gate_in_run_source(self):
        import inspect
        s=inspect.getsource(p.run)
        self.assertLess(s.index('save(output/\'diagnoses.json\''),s.index('check_records(records,ts)'))
        self.assertLess(s.index('check_records(records,ts)'),s.index('evaluate(output,ps,ts,records)'))
        self.assertNotIn('private',s)
    def test_summary_denominators(self):
        rows=[{'unit_id':'eta0','initial_ae_mm':'0' if i==0 else '1','case':'base','truth_mm':'600',
            'a6_status':'FEASIBLE_WITNESS','status':'FORWARD_WITNESS','nominal_status':'UNKNOWN','seed_status':'UNKNOWN'} for i in range(9)]
        ss=p.summary_rows(rows)
        self.assertEqual(next(r['n'] for r in ss if r['group']=='all'),9)
        self.assertEqual(next(r['n'] for r in ss if r['group']=='shifted'),8)
        self.assertEqual(next(r['additional_verified_over_nominal'] for r in ss if r['group']=='all'),9)

if __name__=='__main__':unittest.main()
