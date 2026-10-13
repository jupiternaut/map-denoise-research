"""Hand-built orchestration checks; never read the confirmatory population."""
import ast
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import run_polytope as run
import execute_polytope as execute


def fixture_arm():
    branches = {}
    for index, branch in enumerate(run.original.BRANCHES):
        node = {'id': index, 'lo': run.encode(index), 'hi': run.encode(index+1),
                'status': 'retained_tolerance', 'darkness_bounds': [[run.encode(0),run.encode(1)]]*9,
                't_bounds': [[run.encode(0),run.encode(1)]]*3}
        branches[branch] = {'trace':[node], 'retained_ids':[index], 'hull':[node['lo'],node['hi']]}
    return {'branches':branches, 'outer':{'hull':[run.encode(0),run.encode(3)]},
            'calibration_intervals':[[run.encode(0),run.encode(1)]]*3}


class PipelineTests(unittest.TestCase):
    def test_filter_only_certified_deletion_preserves_trace_and_unknown(self):
        old = fixture_arm();before = deepcopy(old)
        bounds = ((Q(0),Q(1)),(Q(0),Q(1)))
        contradictory = (('upper',(Q(1),Q(1)),Q(99,100)),
                         ('lower',(Q(1),Q(1)),Q(101,100)))
        feasible = (('upper',(Q(1),Q(1)),Q(1)),)
        rows = [contradictory,feasible,feasible]
        results = [run.contractor.classify(rows[0],bounds),run.contractor.classify(rows[1],bounds),
                   run.contractor.classify(rows[2],bounds,deadline=time.monotonic()-1)]
        self.assertEqual([x['status'] for x in results],['CERTIFIED_INFEASIBLE','FEASIBLE_WITNESS','UNKNOWN'])
        seen=[]
        def problem(public,arm,branch,node,method):
            seen.append((branch,node['id'],method));return rows[node['id']],bounds
        def union(branches,**kwargs):
            ids=[i for tree in branches.values() for i in tree['retained_ids']]
            return {'hull':None if not ids else [run.encode(min(ids)),run.encode(max(ids)+1)],'mode':kwargs['method']}
        with patch.object(run,'leaf_rows',side_effect=problem),patch.object(run.contractor,'classify',side_effect=results) as calls,\
             patch.object(run.original.frozen,'union_outer',side_effect=union):
            new=run.filter_arm({'clipped':False},old,'calibrated',time.time()+60)
        self.assertEqual(old,before)
        self.assertEqual([new['branches'][b]['retained_ids'] for b in run.original.BRANCHES],[[],[1],[2]])
        self.assertEqual(new['outer']['hull'],[run.encode(1),run.encode(3)])
        for branch in run.original.BRANCHES:self.assertEqual(new['branches'][branch]['trace'],old['branches'][branch]['trace'])
        self.assertEqual(new['polytope_stats']['original_leaves'],3)
        self.assertEqual(new['polytope_stats']['budget_leaves_rebuilt'],0)
        self.assertEqual(len(seen),3)
        self.assertTrue(all(call.kwargs['deadline']>time.monotonic() for call in calls.call_args_list))

    def test_certificate_failure_is_not_deletion(self):
        old=fixture_arm()
        bad={'status':'CERTIFIED_INFEASIBLE','certificate':None}
        with patch.object(run,'leaf_rows',return_value=((),((Q(0),Q(1)),))),\
             patch.object(run.contractor,'classify',return_value=bad):
            with self.assertRaisesRegex(RuntimeError,'unverified exclusion'):run.filter_arm({'clipped':False},old,'joint',time.time()+60)

    def test_shared_deadline_prevents_leaf_calls(self):
        with patch.object(run,'leaf_rows') as rows:
            with self.assertRaises(TimeoutError):run.filter_arm({'clipped':False},fixture_arm(),'joint',time.time()-1)
        rows.assert_not_called()

    def test_full_action_matrix_and_duplicate_detection(self):
        ts=[{'task_id':str(i),'incumbent':run.encode(i)} for i in range(324)]
        ds=[{'task_id':t['task_id'],'appearance_model':a,'method':m,'reused':a=='box'} for t in ts for a in run.ARMS for m in run.METHODS]
        run.check_actions(ds,ts)
        for bad in (ds[:-1],ds[:-1]+[ds[0]]):
            with self.assertRaises(RuntimeError):run.check_actions(bad,ts)
        ds[0]['reused']=False
        with self.assertRaises(RuntimeError):run.check_actions(ds,ts)
        ks=[{'task_id':t['task_id'],'method':'keep','status':'KEEP','output':t['incumbent']} for t in ts]
        run.check_keep(ks,ts)
        ks[-1]['output']=run.encode(-1)
        with self.assertRaises(RuntimeError):run.check_keep(ks,ts)

    def test_private_parse_is_below_all_persist_readback_gates(self):
        tree=ast.parse(Path(run.__file__).read_text())
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        private_strings=[(name,n.lineno) for name,node in functions.items() for n in ast.walk(node)
                         if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value=='private/truth.json']
        self.assertEqual([name for name,_ in private_strings],['evaluate'])
        body=functions['run'];calls=[n for n in ast.walk(body) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        evaluation=next(n.lineno for n in calls if n.func.id=='evaluate')
        for name in ('check_actions','check_keep'):
            self.assertLess(next(n.lineno for n in calls if n.func.id==name),evaluation)
        loads=[n.lineno for n in calls if n.func.id=='load' and any(isinstance(a,ast.BinOp) and isinstance(a.right,ast.Constant)
               and a.right.value in ('decisions.json','keep/decisions.json','outers.json') for a in n.args)]
        self.assertGreaterEqual(len(loads),4)
        self.assertTrue(all(line<evaluation for line in loads))

    def test_write_boundary_rejects_old_target(self):
        with self.assertRaises(ValueError):run.save(run.OLD/'should-never-write.json',{})
        with self.assertRaises(ValueError):run.save(run.ROOT/'PROTOCOL.md',{},replace=True)

    def test_executor_uses_remaining_shared_budget(self):
        output=run.ROOT/'outputs/hand-fixture-not-written'
        saved={}
        def load(path):
            if Path(path).name=='REVIEW_GATE.json':return {'status':'PASS','files':{}}
            if Path(path).name=='VERIFY.json':return {'status':'PASS','output':str(output)}
            return saved.get(str(path),{})
        def save(path,value,**kwargs):saved[str(path)]=deepcopy(value)
        with patch.object(Path,'exists',return_value=False),patch.object(Path,'mkdir'),\
             patch.object(run,'load',side_effect=load),patch.object(run,'save',side_effect=save),\
             patch.object(run,'sources',return_value={}),patch.object(run,'run') as main,\
             patch.object(execute.time,'time',side_effect=[100,107,109,110]),\
             patch.object(execute.subprocess,'run') as verify:
            execute.execute(output)
        main.assert_called_once_with(output)
        self.assertEqual(saved[str(output/'BUDGET.json')]['deadline_epoch'],3700)
        self.assertEqual(verify.call_args.kwargs['timeout'],3593)
        self.assertEqual(saved[str(output/'STAGE_STATUS.json')]['elapsed_main_plus_verification_seconds'],10)


if __name__=='__main__':unittest.main()
