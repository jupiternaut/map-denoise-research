from fractions import Fraction as Q
import unittest
import run_appearance as r


class ActionTests(unittest.TestCase):
    def test_exact_encoded_boundary(self):
        self.assertEqual(r.exact(r.encode(Q(13,7))),Q(13,7))
        for bad in (0.1,True,{'numerator':1,'denominator':0}):
            with self.assertRaises(ValueError):
                r.exact(bad)

    def test_same_hybrid_rule(self):
        self.assertEqual(r.select((Q(0),Q(2)),Q(5))['branch'],'midpoint')
        self.assertEqual(r.select((Q(0),Q(2)),Q(-1,10))['branch'],'projection')
        self.assertEqual(r.select(None,Q(1))['status'],'INCOMPATIBLE')
        self.assertEqual(r.select((Q(0),Q(2)),Q(1))['status'],'KEEP')

    def test_complete_actions_reject_duplicates_and_scope(self):
        tasks=[{'task_id':'fixture'}]
        records=[{'task_id':'fixture','method':method,'appearance_scope':'per_view','reused':False}
                 for method in r.METHODS]
        r.check_actions(records,tasks)
        with self.assertRaises(RuntimeError):
            r.check_actions(records[:1]*2,tasks)
        records[0]['appearance_scope']='global_shared'
        with self.assertRaises(RuntimeError):
            r.check_actions(records,tasks)

    def test_public_action_has_no_private_input(self):
        task={'task_id':'fixture','incumbent':r.encode(Q(5))}
        public={'unit_id':'baseline_80'}
        arm={'outer':{'mode':'joint','hull':[r.encode(Q(0)),r.encode(Q(2))]}}
        decision=r.action(task,public,arm)
        self.assertEqual(decision['output'],r.encode(Q(1)))
        self.assertEqual(decision['appearance_scope'],'per_view')

    def test_coverage_requires_all_nine_material_intervals(self):
        bounds=[[r.encode(Q(0)),r.encode(Q(1))] for _ in range(9)]
        node={'id':0,'lo':r.encode(Q(0)),'hi':r.encode(Q(2)),
              'darkness_bounds':bounds,'t_bounds':[[r.encode(Q(0)),r.encode(Q(1,50))] for _ in range(3)]}
        arm={'outer':{'hull':[r.encode(Q(0)),r.encode(Q(2))],'retained_branches':['none']},
             'branches':{'none':{'trace':[node],'retained_ids':[0]}}}
        private={'truth':r.encode(Q(1)),'rear_branch':'none','colors':[r.encode(Q(1,2))]*3,
                 'actual_view_bias':[r.encode(Q(0))]*3,'case':'fixture','seed':0}
        public={'observation_id':'fixture','unit_id':'baseline_80'}
        self.assertTrue(r.coverage(public,private,'joint',arm)['latent_material_covered'])
        node['darkness_bounds']=bounds[:3]
        self.assertFalse(r.coverage(public,private,'joint',arm)['latent_material_covered'])


if __name__=='__main__':
    unittest.main()
