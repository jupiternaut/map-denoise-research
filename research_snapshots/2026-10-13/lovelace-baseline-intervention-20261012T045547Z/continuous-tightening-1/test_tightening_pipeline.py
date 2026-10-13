import unittest
from fractions import Fraction as Q
from unittest.mock import patch
import io_contract as io
import run_tightening as run
import audit_tightening as audit
class PipelineTests(unittest.TestCase):
    def test_public_probe_selection(self):
        record={'branches':[{'branch':'partial','trace':[
            {'id':0,'lo':'1','hi':'2','status':'excluded_certificate'},
            {'id':1,'lo':'2','hi':'3','status':'retained_tolerance'},
            {'id':2,'lo':'3','hi':'9','status':'retained_interior'}]}]}
        self.assertEqual([p['depth'] for p in io.probe_positions(record,Q)],['5/2','6'])
    def test_hull_preserves_correct_and_clamps(self):
        tasks=[{'task_id':'a','observation_id':'o','incumbent':'604'}, {'task_id':'b','observation_id':'o','incumbent':'608'}]
        sets=[{'observation_id':'o','arm':a,'outer_intervals':[['603','606']]} for a in ('depth','camera_tiles')]
        result=run.selections(tasks,sets)
        self.assertEqual([run.solver.exact(r['selected']) for r in result],[Q(604),Q(604),Q(606),Q(606)])
    def test_sources_include_transitive_and_all_local(self):
        with patch.object(io,'load',return_value={'files':{'old':'digest'}}),patch.object(io,'sha',return_value='x'):
            s=io.sources()
        self.assertEqual(s['old'],'digest'); self.assertEqual(len(s),len(io.SCIENCE_NAMES)+1)
    def test_metric_against_truth(self):
        rows=[{'arm':'depth','policy':'HULL','unit_id':'u','groups':['shifted'],'initial_ae_mm':'2',
            'final_ae_mm':'1','moved':True,'displacement_mm':'1'}]
        allrow=next(r for r in audit.summarize(rows) if r['unit']=='ALL')
        self.assertEqual(allrow['MSE_reduction_percent'],75)
    def test_no_private_method_read(self):
        import inspect
        src=inspect.getsource(run)
        self.assertNotIn('private/',src)
        self.assertLess(src.index("save(OUT/'sets.json'"),src.index("tasks=load(OLD/'tasks.json')"))
        self.assertLess(src.index("PREDICTION_RECEIPT.json"),src.index('complete_branch('))
if __name__=='__main__':unittest.main()
