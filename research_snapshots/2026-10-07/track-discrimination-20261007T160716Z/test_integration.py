import unittest
import numpy as np
from experiment import choose
from theory.kernel import gain_bounds_world


class Tests(unittest.TestCase):
    def test_empty_and_missing(self):
        objs=[dict(candidate_id=-1,xyz_mm=[0,0,2]),dict(candidate_id=0,xyz_mm=[0,0,4])]
        self.assertEqual(choose(objs,-1,[0,0,0],[0,0,1],[])['selected_candidate_id'],-1)
        self.assertEqual(choose(objs,None,[0,0,0],[0,0,1],[[4,5]])['reason'],'NO_CURRENT_OUTPUT')

    def test_move_and_overlap(self):
        objs=[dict(candidate_id=-1,xyz_mm=[0,0,2]),dict(candidate_id=0,xyz_mm=[0,0,4])]
        self.assertEqual(choose(objs,-1,[0,0,0],[0,0,1],[[3.5,4.5]])['selected_candidate_id'],0)
        self.assertEqual(choose(objs,-1,[0,0,0],[0,0,1],[[2,2.5],[4,4.5]])['selected_candidate_id'],-1)

    def test_world_against_direct(self):
        rng=np.random.default_rng(7041)
        for _ in range(500):
            a,b,C,r=rng.normal(size=(4,3));bounds=gain_bounds_world(a,b,C,r,[[1,2],[4,8]])
            z=np.array([1.,2.,4.,8.]);x=C+z[:,None]*r
            g=((a-x)**2).sum(1)-((b-x)**2).sum(1)
            np.testing.assert_allclose(bounds,[g.min(),g.max()],atol=1e-10,rtol=1e-10)

    def test_out_of_ray_candidate(self):
        self.assertEqual(gain_bounds_world([0,0,2],[100,0,4],[0,0,0],[0,0,1],[[4,4]]),(-9996.,-9996.))

    def test_unchanged_and_tie(self):
        objs=[dict(candidate_id=-1,xyz_mm=[0,0,2]),dict(candidate_id=1,xyz_mm=[0,0,4]),dict(candidate_id=0,xyz_mm=[0,0,4])]
        self.assertEqual(choose(objs,-1,[0,0,0],[0,0,1],[[3,3]])['selected_candidate_id'],-1)
        self.assertEqual(choose(objs,-1,[0,0,0],[0,0,1],[[4,4]])['selected_candidate_id'],0)


if __name__=='__main__':unittest.main(verbosity=2)
