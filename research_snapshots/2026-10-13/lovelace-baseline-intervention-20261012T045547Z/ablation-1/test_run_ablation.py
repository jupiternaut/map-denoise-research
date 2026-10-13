from fractions import Fraction as Q
import unittest
import run_ablation as r


class ActionTests(unittest.TestCase):
    def test_encoded_boundary(self):
        encoded=r.encode(Q(13,7))
        self.assertEqual(r.exact(encoded),Q(13,7))
        with self.assertRaises(ValueError):
            r.exact(0.1)

    def test_hybrid_fallback_and_empty(self):
        self.assertEqual(r.select((Q(0),Q(2)),Q(5))['branch'],'midpoint')
        self.assertEqual(r.select((Q(0),Q(2)),Q(-1,10))['branch'],'projection')
        self.assertEqual(r.select(None,Q(1))['status'],'INCOMPATIBLE')
        self.assertEqual(r.select((Q(0),Q(2)),Q(1))['status'],'KEEP')

    def test_nested_projection(self):
        for a in (Q(-5),Q(1),Q(3),Q(8)):
            large=r.select((Q(0),Q(6)),a,'projection')['output']
            small=r.select((Q(2),Q(4)),a,'projection')['output']
            for truth in (Q(2),Q(3),Q(4)):
                self.assertLessEqual(abs(small-truth),abs(large-truth))

    def test_complete_action_matrix_rejects_duplicate(self):
        tasks=[{'task_id':'fixture'}]
        records=[{'task_id':'fixture','method':m,'mask_name':mask,'reused':False} for m in r.METHODS for mask in r.MASKS]
        r.check_actions(records,tasks,r.MASKS,False)
        with self.assertRaises(RuntimeError):
            r.check_actions(records[:-1]+records[:1],tasks,r.MASKS,False)


if __name__=='__main__':
    unittest.main()
