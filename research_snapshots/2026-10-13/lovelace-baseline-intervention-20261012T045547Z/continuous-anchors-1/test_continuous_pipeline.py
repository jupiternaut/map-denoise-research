"""Manual pipeline/interface and exact endpoint readout tests; no formal inputs."""
import ast
from copy import deepcopy
from fractions import Fraction as Q
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import continuous_solver as solver
import run_continuous as runner
import audit_continuous as audit
from test_continuous_solver import fixture
class PipelineTests(unittest.TestCase):
    def test_registration_budgets_and_denominators(self):
        r=json.loads((runner.ROOT/"REGISTRATION.json").read_text())
        self.assertEqual(r["arms"],list(solver.ARM_NAMES));self.assertEqual(r["branches"],list(solver.BRANCHES))
        self.assertEqual(r["domain"],["540","660"]);self.assertEqual(r["node_cap"],solver.MAX_NODES)
        self.assertEqual(Q(r["min_width"]),solver.MIN_WIDTH);self.assertEqual(r["branch_seconds"],solver.BRANCH_SECONDS)
        self.assertEqual(r["sets"],r["observations"]*4)
        self.assertEqual(r["predictions"],r["tasks"]*4*2)
        self.assertEqual(r["whole_seconds"],r["main_seconds"]+r["verification_reserve_seconds"])
        self.assertEqual(runner.SCIENCE_NAMES,audit.SCIENCE_NAMES)
    def test_method_photo_and_calibration_whitelists(self):
        public,oldarm,anchor,point=fixture()
        public.update(truth=600,tasks=[590,600,610],case="bad",variation_index=1)
        oldarm.update(saved_witness=point)
        self.assertEqual(set(runner.photo_payload(public)),{"observation","radius","baseline","clipped"})
        self.assertEqual(set(runner.calibration_payload(oldarm)),{"calibration_intervals","appearance_model","drift_bound"})
    def test_single_incumbent_cannot_change_other_output_or_set(self):
        sets=[{"observation_id":"opaque","arm":name,"outer_intervals":[[solver.encode(590),solver.encode(610)]]}
              for name in solver.ARM_NAMES]
        before=deepcopy(sets)
        tasks=[{"task_id":"a","observation_id":"opaque","incumbent":solver.encode(580)},
               {"task_id":"b","observation_id":"opaque","incumbent":solver.encode(620)}]
        first=runner.selections(tasks,sets)
        changed=deepcopy(tasks);changed[0]["incumbent"]=solver.encode(600)
        second=runner.selections(changed,sets)
        self.assertEqual(sets,before)
        self.assertEqual([r for r in first if r["task_id"]=="b"],[r for r in second if r["task_id"]=="b"])
        relabeled=deepcopy(tasks);relabeled[0]["task_id"]="s0secret";relabeled[1]["task_id"]="s8secret"
        third=runner.selections(relabeled,sets)
        keys=("selected","incumbent","moved","policy","arm")
        self.assertEqual([{k:r[k] for k in keys} for r in first],[{k:r[k] for k in keys} for r in third])
    def test_exact_summary_and_union_damage_is_not_hidden(self):
        rows=[]
        for old,new in ((Q(5421,10),Q(543)),(Q(538),Q(540)),(Q(541),Q(541))):
            rows.append({"arm":"both","policy":"UNION","unit_id":"manual","groups":["all"],
                 "initial_ae_mm":str(abs(old-541)),"final_ae_mm":str(abs(new-541)),
                 "displacement_mm":str(abs(old-new)),"moved":old!=new})
        allrow=next(r for r in audit.summarize(rows) if r["unit"]=="ALL")
        self.assertEqual((allrow["improved"],allrow["equal"],allrow["worsened"]),(1,1,1))
        self.assertEqual(Q(allrow["max_AE_increase_mm"]),Q(9,10))
        self.assertEqual(Q(allrow["initial_MAE_mm"]),Q(41,30))
    def test_independent_rows_match_on_handwritten_continuous_cells(self):
        public,oldarm,anchor,point=fixture()
        for name in solver.ARM_NAMES:
            for branch in solver.BRANCHES:
                actual=solver.build_interval_problem(public,oldarm,anchor,name,590,610,branch)
                checked=audit.independent.necessary_problem(public,oldarm,anchor,name,branch,Q(590),Q(610))
                self.assertEqual(actual,checked)
    def test_real_handwritten_tree_and_actions_pass_independent_checker(self):
        public,oldarm,anchor,point=fixture()
        record=solver.solve_interval_branch(public,oldarm,anchor,"both","partial")
        checked=audit.independent.verify_branch(public,oldarm,anchor,"both","partial",record)
        self.assertEqual(checked["outer_intervals"],record["outer_intervals"])
        for policy in ("HULL","UNION"):
            decision=solver.select_depth(record["outer_intervals"],Q(590),policy)
            audit.independent.verify_selection(Q(590),record["outer_intervals"],policy,decision)
    def test_closed_coverage_nesting_and_disjoint_gaps(self):
        self.assertTrue(audit.contains([[540,541],[543,544]],Q(541)))
        self.assertFalse(audit.contains([[540,541],[543,544]],Q(542)))
        self.assertTrue(audit.covered_by([[540,541],[543,544]],[[540,544]]))
        self.assertFalse(audit.covered_by([[540,544]],[[540,541],[543,544]]))
    def test_no_private_parse_or_new_method_import_in_independent_checker(self):
        tree=ast.parse((runner.ROOT/"verify_continuous.py").read_text())
        names={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        names.update(a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names)
        self.assertNotIn("continuous_solver",names)
        self.assertNotIn("run_continuous",names)
        run_tree=ast.parse((runner.ROOT/"run_continuous.py").read_text())
        for node in ast.walk(run_tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="load":
                self.assertNotIn("/private/",ast.unparse(node))
        source=(runner.ROOT/"audit_continuous.py").read_text()
        self.assertLess(source.index('"verification/PUBLIC_GATE.json"'),source.index('facts = {f['))
if __name__=="__main__":unittest.main()
