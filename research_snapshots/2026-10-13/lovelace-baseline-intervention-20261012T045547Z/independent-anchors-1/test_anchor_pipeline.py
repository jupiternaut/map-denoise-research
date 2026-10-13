"""Manual fixtures only: generation, 2x2 integration, no old private data."""
import copy, json, random, sys, unittest
from fractions import Fraction as Q
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import anchor_solver as s
import prepare_anchors as g
import run_anchors as r
import verify_anchors as v

def fixture():
    z=Q(9007,14);es=(Q(3,100),Q(0),Q(-7,100))
    ds=(Q(1,5),Q(3,5),Q(2,5))*3;ts=(Q(1,100),)*3
    areas=s.geometry._point_areas('partial',z,es,Q(80))
    photo=[]
    for i,a in enumerate(areas):
        gray=1-sum(a[k]*ds[3*(i//48)+k] for k in range(3))
        photo.extend([s.encode(gray)]*3)
    p={'observation_id':'fixture','radius':'1/5','baseline':'80','clipped':True,'observation':photo}
    arm={'appearance_model':'bounded','drift_bound':'1/10','calibration_intervals':[['0','1/50']]*3}
    fact={'observation_id':'fixture','actual_offsets':list(map(s.encode,es)),
          'view_colors':list(map(s.encode,(1-d for d in ds)))}
    t=g.tape(random.Random(61006));a=g.acquire(fact,t)
    w={'offsets':list(map(s.encode,es)),'darkness':list(map(s.encode,ds)),
       't':list(map(s.encode,ts)),'noise':[s.encode(0)]*432}
    return p,arm,a,z,w

class PipelineTests(unittest.TestCase):
    def test_all_arms_true_manual_forward(self):
        p,arm,a,z,w=fixture()
        for name in s.ARM_NAMES:
            x=s.revalidate_old_witness(p,arm,a,name,z,'partial',w)
            self.assertIsNotNone(x,name)
            self.assertTrue(s.check_forward_witness(p,arm,a,name,z,'partial',x))
            self.assertTrue(v.verify_forward(p,arm,a,z,'partial',name,x))
            rows,bounds=s.build_problem(p,arm,a,name,z,'partial')
            vr,vb=v.necessary_problem(p,arm,a,z,'partial',name)
            self.assertEqual(rows,vr);self.assertEqual(bounds,vb)
            coordinates=tuple(map(s.exact,x['darkness']+x['t']+x['tA']))
            self.assertTrue(s.witness_check(rows,bounds,list(map(s.encode,coordinates))))
    def test_paired_measurement_tape(self):
        p,arm,a,z,w=fixture();t=g.tape(random.Random(61006))
        fact={'observation_id':'second','actual_offsets':w['offsets'],
              'view_colors':[s.encode(Q(3,10))]*9}
        b=g.acquire(fact,t)
        self.assertEqual(a['camera'],b['camera']);self.assertNotEqual(a['appearance'],b['appearance'])
    def test_disabled_readings_not_used(self):
        p,arm,a,z,w=fixture()
        for name,disabled in [('none',('camera','appearance')),('camera',('appearance',)),('appearance',('camera',))]:
            bad=copy.deepcopy(a)
            for section in disabled:bad[section]={'ignored':'arbitrary unavailable data'}
            self.assertEqual(s.build_problem(p,arm,a,name,z,'partial'),s.build_problem(p,arm,bad,name,z,'partial'))
            self.assertEqual(s.revalidate_old_witness(p,arm,a,name,z,'partial',w),
                             s.revalidate_old_witness(p,arm,bad,name,z,'partial',w))
    def test_union_and_population_contract(self):
        self.assertEqual(r.aggregate(['INHERITED_INFEASIBLE']*3),'CERTIFIED_INFEASIBLE')
        self.assertEqual(r.aggregate(['CERTIFIED_INFEASIBLE','UNKNOWN','INHERITED_INFEASIBLE']),'UNKNOWN')
        self.assertEqual(r.aggregate(['CERTIFIED_INFEASIBLE','FORWARD_WITNESS','UNKNOWN']),'FORWARD_WITNESS')
        with self.assertRaises(ValueError):r.aggregate(['FORWARD_WITNESS'])
        reg=json.loads((ROOT/'REGISTRATION.json').read_text())
        self.assertEqual(reg['total_records'],reg['tasks_per_arm']*len(reg['arms']))
        self.assertEqual(reg['total_branches'],reg['total_records']*3)
    def test_explicit_source_allowlist(self):
        self.assertIn('test_anchor_pipeline.py',r.SCIENCE_NAMES)
        self.assertNotIn('analysis.py',r.SCIENCE_NAMES)
        self.assertEqual(len(r.SCIENCE_NAMES),len(set(r.SCIENCE_NAMES)))
if __name__=='__main__':unittest.main(verbosity=2)
