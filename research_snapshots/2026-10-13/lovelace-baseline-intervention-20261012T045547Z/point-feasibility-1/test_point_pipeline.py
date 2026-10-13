"""A6 hand-built control tests. No formal observations or private data."""
import ast, time, unittest
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
from unittest.mock import patch
import run_point as run

def arm_fixture():
    e=run.encode
    branches={}
    for b in run.BRANCHES:
        nodes=[{'id':0,'lo':e(590),'hi':e(600),'status':'retained_tolerance',
                'darkness_bounds':[[e(0),e(1)]]*9,'t_bounds':[[e(0),e(Q(1,50))]]*3},
               {'id':1,'lo':e(600),'hi':e(610),'status':'retained_tolerance',
                'darkness_bounds':[[e(0),e(1)]]*9,'t_bounds':[[e(0),e(Q(1,50))]]*3}]
        branches[b]={'trace':nodes,'retained_ids':[0,1]}
    return {'branches':branches,'calibration_intervals':[[e(0),e(Q(1,50))]]*3,
            'outer':{'hull':[e(590),e(610)]}}

class PipelineTests(unittest.TestCase):
    def test_interval_feasible_fixed_point_excluded_not_just_witness_failure(self):
        box=(('upper',(Q(0),),Q(1,2)),('lower',(Q(1),),Q(1,2)))
        point=(('upper',(Q(0),),Q(1,2)),('lower',(Q(0),),Q(1,2)))
        bounds=((Q(0),Q(1)),)
        a=run.solver.contractor.classify(box,bounds)
        b=run.solver.contractor.classify(point,bounds)
        self.assertEqual(a['status'],'FEASIBLE_WITNESS')
        self.assertEqual(b['status'],'CERTIFIED_INFEASIBLE')
        run.check_classification(box,bounds,a);run.check_classification(point,bounds,b)

    def test_closed_boundary_every_leaf_all_branches(self):
        pairs=run.containing(arm_fixture(),Q(600))
        self.assertEqual([(b,n['id']) for b,n in pairs],[(b,i) for b in run.BRANCHES for i in (0,1)])

    def test_union_gap_differs_from_hull(self):
        a=arm_fixture()
        for v in a['branches'].values():
            v['trace'][0]['hi']=run.encode(599)
            v['trace'][1]['lo']=run.encode(601)
        self.assertEqual(run.containing(a,Q(600)),[])
        self.assertTrue(run.exact(a['outer']['hull'][0])<600<run.exact(a['outer']['hull'][1]))

    def test_deadline_prevents_solving(self):
        with patch.object(run.solver,'build_problem') as f:
            with self.assertRaises(TimeoutError):run.classify({}, {}, 'joint','none',Q(600),None,time.time()-1)
        f.assert_not_called()

    def test_unknown_retained_and_invalid_certificate_not_accepted(self):
        rows=(('upper',(Q(1),),Q(1)),);bounds=((Q(0),Q(1)),)
        unknown=run.solver.contractor.classify(rows,bounds,deadline=time.monotonic()-1)
        self.assertEqual(unknown['status'],'UNKNOWN')
        run.check_classification(rows,bounds,unknown)
        with self.assertRaises(RuntimeError):
            run.check_classification(rows,bounds,{'status':'CERTIFIED_INFEASIBLE','certificate':None})

    def test_every_global_branch_evaluated_with_witness(self):
        old=arm_fixture();before=deepcopy(old)
        result=run.solver.contractor.classify((('upper',(Q(1),),Q(1)),),((Q(0),Q(1)),))
        prior={'polytope_records':{b:[{'node_id':i,'classification':result} for i in (0,1)] for b in run.BRANCHES}}
        calls=[]
        def classify(p,a,m,b,z,n,deadline):
            calls.append((b,None if n is None else n['id']));return deepcopy(result)
        def build(*a,**k):return (('upper',(Q(1),),Q(1)),),((Q(0),Q(1)),)
        task={'task_id':'manuals1','incumbent':run.encode(600)}
        public={'observation_id':'manual','unit_id':'manual','variation_index':0}
        action={'status':'KEEP','output':run.encode(600)}
        with patch.object(run,'classify',side_effect=classify),patch.object(run.solver,'build_problem',side_effect=build):
            d=run.diagnose(task,public,old,prior,'calibrated',action,action,time.time()+30)
        self.assertEqual(len(calls),9)
        self.assertEqual([b for b,n in calls if n is None],list(run.BRANCHES))
        self.assertEqual(d['global_status'],'FEASIBLE_WITNESS')
        self.assertEqual(d['local_status'],'FEASIBLE_WITNESS')
        self.assertEqual(old,before)

    def test_full648_record_gate_and2924_counts(self):
        tasks=[{'task_id':str(i)} for i in range(324)]
        records=[]
        for m,total in [('calibrated',494),('joint',486)]:
            for i in range(324):
                count=2 if i<total-324 else 1
                records.append({'task_id':str(i),'method':m,'global':[{'branch':b} for b in run.BRANCHES],
                    'local':[{'branch':'none','node_id':n} for n in range(count)]})
        run.check_records(records,tasks)
        for bad in (records[:-1],records[:-1]+[records[0]]):
            with self.assertRaises(RuntimeError):run.check_records(bad,tasks)
        bad=deepcopy(records);bad[0]['global']=bad[0]['global'][:2]
        with self.assertRaises(RuntimeError):run.check_records(bad,tasks)

    def test_diagnostic_does_not_generate_geometry_output(self):
        record={'method':'joint','source_hull_contains':True,'source_union_contains':True,
                'box_status':'FEASIBLE_WITNESS','local_status':'CERTIFIED_INFEASIBLE','global_status':'CERTIFIED_INFEASIBLE',
                'local':[],'a4_action':{'status':'KEEP','output':run.encode(601)},
                'a5_action':{'status':'KEEP','output':run.encode(601)}}
        row=run.evaluation_row(record,{'task_id':'m','incumbent':run.encode(601)},
             {'observation_id':'p','pair_id':'p','unit_id':'u','variation_index':1},
             {'truth':run.encode(600),'case':'manual'})
        self.assertEqual(row['initial_ae_mm'],'1')
        self.assertEqual(row['a4_output_mm'],'601')
        self.assertEqual(row['a5_output_mm'],'601')
        self.assertNotIn('new_output',row)

    def test_public_readback_full_gate_precedes_only_evaluate_call(self):
        tree=ast.parse((run.ROOT/'run_point.py').read_text())
        f={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        trial=next(n for n in f['run'].body if isinstance(n,ast.Try))
        funcs=[]
        for node in trial.body:
            calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
            funcs.extend((n.func.id,n.lineno) for n in calls if n.func.id in ('save','load','check_records','evaluate'))
        gate=next(l for name,l in funcs if name=='check_records')
        evaluation=next(l for name,l in funcs if name=='evaluate')
        readback=[l for name,l in funcs if name=='load']
        self.assertTrue(any(l<gate for l in readback))
        self.assertLess(gate,evaluation)
        truth_functions=[fn.name for fn in f.values() if any(isinstance(n,ast.Constant) and n.value=='private/truth.json' for n in ast.walk(fn))]
        self.assertEqual(truth_functions,['evaluate'])
if __name__=='__main__':unittest.main()

