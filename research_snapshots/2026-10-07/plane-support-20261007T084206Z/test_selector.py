import copy
import unittest
import numpy as np
from experiment import load, ROOT, sanitize
from selector import decisions, support


def obj(cid,z,ncc=.8):
    return dict(candidate_id=cid,optical_depth_mm=z,
                hypotheses=[dict(scores={'U11':{'ncc':[ncc]}})])


def interval(center,radius=1,active=True):
    return dict(center_mm=center,lo_mm=center-radius,hi_mm=center+radius,informative=active)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.p=load(ROOT/'PROTOCOL.json')
        self.objects=[obj(-1,60),obj(0,100),obj(1,140)]

    def test_clear_consensus_moves(self):
        d=decisions(self.objects,[interval(100) for _ in range(20)],self.p,None)
        self.assertEqual(d['joint_interval']['selected_candidate_id'],0)
        self.assertEqual(d['neighbor_point']['selected_candidate_id'],0)

    def test_overlapping_intervals_do_not_certify(self):
        # Large-radius abstentions cannot be silently removed from denominator.
        values=[interval(100) for _ in range(3)]+[interval(100,20,False) for _ in range(17)]
        d=decisions(self.objects,values,self.p,1)
        self.assertEqual(d['joint_interval']['selected_candidate_id'],1)
        self.assertEqual(d['joint_interval']['reason'],'FALLBACK_WEAK_SUPPORT')

    def test_two_layers_ambiguous(self):
        values=[interval(100) for _ in range(10)]+[interval(140) for _ in range(10)]
        d=decisions(self.objects,values,self.p,None)
        self.assertIsNone(d['joint_interval']['selected_candidate_id'])

    def test_no_movement_inside_incumbent_scale(self):
        d=decisions([obj(-1,97),obj(0,100)], [interval(100) for _ in range(20)],self.p,None)
        self.assertIsNone(d['joint_interval']['selected_candidate_id'])

    def test_joint_gate_and_neighbor_are_separate(self):
        self.objects[1]=obj(0,100,.5)
        d=decisions(self.objects,[interval(100) for _ in range(20)],self.p,None)
        self.assertEqual(d['neighbor_interval']['selected_candidate_id'],0)
        self.assertIsNone(d['joint_interval']['selected_candidate_id'])

    def test_every_interval_realization_obeys_support_bounds(self):
        values=[interval(100,2),interval(104,2),interval(141,1),interval(90,20,False)]
        scores=support(self.objects,values,self.p)
        for frac in np.linspace(0,1,17):
            for s in scores:
                latent=[v['lo_mm']+frac*(v['hi_mm']-v['lo_mm']) for v in values if v['informative']]
                actual=sum(abs(x-s['optical_depth_mm'])<=5 for x in latent)/len(values)
                self.assertLessEqual(s['certain'],actual)
                self.assertLessEqual(actual,s['possible'])

    def test_sanitizer_discards_reference_like_extra_fields(self):
        raw={'rows':[dict(scene=118,roi='test',query=0,pixel_xy=[1,2],primary=True,
             candidates=[dict(candidate_id=0,xyz_mm=[1,2,3],hypotheses=self.objects[1]['hypotheses'])],
             incumbent=None,decisions={'U11_bestpair':{'selected_candidate_id':None}})]}
        original=sanitize(raw)
        raw['rows'][0]['official_diagnostic']={'depth':123,'gt':999}
        raw['rows'][0]['candidates'][0]['official_normal_diagnostic']={'gt':0}
        self.assertEqual(original,sanitize(raw))


if __name__=='__main__':
    unittest.main(verbosity=2)
