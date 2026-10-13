"""Hand-built nonconfirmation fixtures only; no formal data reads."""
from fractions import Fraction as Q
import importlib, json
from pathlib import Path
import tempfile, unittest
from unittest.mock import patch
import prepare_variation as prep
import run_variation as runner
import execute_variation as entry
import analyze_variation as analysis
class GeneratorTests(unittest.TestCase):
    def test_mean_physical_and_spread(self):
        for d in (Q(0),Q(1,4),Q(1,2),Q(3,4),Q(1)):
            for eta in (Q(0),Q(1,10),Q(1)):
                values=prep.view_darkness([1-d]*3,eta)
                self.assertTrue(all(0<=x<=1 for x in values))
                for k in range(3):
                    column=values[k::3]
                    self.assertEqual(sum(column),3*d)
                    self.assertLessEqual(max(column)-min(column),eta)
    def test_nonzero_material_specific_and_endpoints(self):
        values=prep.view_darkness([Q(1,2)]*3,Q(1,10))
        self.assertNotEqual(values[0]-Q(1,2),values[1]-Q(1,2))
        for color in (Q(0),Q(1)):
            self.assertEqual(prep.view_darkness([color]*3,Q(1,10)),(1-color,)*9)
    def test_zero_forward_identity_and_single_clip(self):
        fact={'colors':['1','1','1'],'actual_offsets':['0']*3,'truth':'600','baseline':'80',
              'rear_branch':'none','image_noise':['-1/100']*432,'actual_view_bias':['0','0','1/100']}
        raw=(Q(1),)*432
        with patch.object(prep.geometry,'point_rgb',return_value=raw):
            image,dark=prep.render_views(fact,Q(0))
        self.assertEqual(prep.exact(image[0]),Q(99,100))
        self.assertEqual(prep.exact(image[288]),Q(1))
        self.assertEqual(dark,(Q(0),)*9)
    def test_reject_bad_parameters(self):
        for eta in (-1,2):
            with self.assertRaises(ValueError):prep.view_darkness([1]*3,eta)
class MatrixTests(unittest.TestCase):
    def records(self):
        tasks=[{'task_id':f'p{i}e{v}s{s}'} for v in range(2) for i in range(18) for s in range(9)]
        records=[{'task_id':t['task_id'],'appearance_model':a,'method':m,
                  'reused':('e0s' in t['task_id'] and a!='bounded')}
                 for t in tasks for a in runner.ARMS for m in runner.METHODS]
        return tasks,records
    def test_complete_count_and_tamper(self):
        tasks,records=self.records();runner.check_actions(records,tasks)
        with self.assertRaises(RuntimeError):runner.check_actions(records[:-1],tasks)
        records[-1]=records[0]
        with self.assertRaises(RuntimeError):runner.check_actions(records,tasks)
    def test_empty_outer_does_not_move(self):
        d=runner.select(None,600,'hybrid')
        self.assertEqual(d['status'],'INCOMPATIBLE')
        self.assertEqual(d['output'],600)
    def test_keep_uniqueness_and_identity(self):
        tasks=[{'task_id':f't{i}','incumbent':prep.encode(600)} for i in range(324)]
        records=[{'task_id':t['task_id'],'method':'keep','status':'KEEP','output':t['incumbent']} for t in tasks]
        runner.check_keep(records,tasks)
        records[-1]=records[0]
        with self.assertRaises(RuntimeError):runner.check_keep(records,tasks)
    def test_existing_output_refusal_never_writes(self):
        with tempfile.TemporaryDirectory(prefix='a4v-test-') as tmp:
            root=Path(tmp);out=root/'outputs/old';out.mkdir(parents=True)
            marker=out/'keep.txt';marker.write_text('fixture')
            with patch.object(entry,'ROOT',root),patch.object(entry.runner,'EXPECTED',root),patch.object(entry.socket,'gethostname',return_value='liekkas'),patch.object(entry.runner,'save') as save:
                with self.assertRaises(RuntimeError):entry.execute(out)
                save.assert_not_called()
            self.assertEqual(marker.read_text(),'fixture')
    def test_wrong_output_refusal(self):
        with patch.object(entry.runner,'save') as save:
            with self.assertRaises(RuntimeError):entry.execute('/tmp/substitute/variation-v1')
            save.assert_not_called()
class AnalysisTests(unittest.TestCase):
    def row(self,key,initial,error,case='texture'):
        return {'base_task_id':key,'initial_ae_mm':str(initial),'ae_mm':str(error),
                'se_mm2':str(Q(error)**2),'case':case,'depth_mm':'9007/14'}
    def test_exact_metrics_and_paired_sign(self):
        old=[self.row('a',2,2),self.row('b',2,1)]
        new=[self.row('b',2,2),self.row('a',2,0)]
        self.assertEqual(analysis.metrics(new)['mse_mm2'],'2')
        value=analysis.paired(old,new)
        self.assertEqual((value['mse_decrease_mm2'],value['better'],value['worse']),('1/2',1,1))
    def test_paired_duplicate_rejected(self):
        row=self.row('a',2,0)
        with self.assertRaises(ValueError):analysis.paired([row,row],[row,row])
    def test_primary_excludes_correct_and_negative(self):
        rows=[self.row('a',2,0),self.row('b',0,0),self.row('c',2,0,analysis.NEGATIVE)]
        self.assertEqual([r['base_task_id'] for r in analysis.subset(rows,'primary_depth')],['a'])
if __name__=='__main__':unittest.main(verbosity=2)
