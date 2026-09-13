import unittest
import numpy as np
from experiment import Input, identifiability_probe, generate, estimate, joint_fit


class Tests(unittest.TestCase):
    def test_exact_ambiguity_and_graph_rank(self):
        result=identifiability_probe()
        self.assertTrue(result["indistinguishable_exactly"])
        self.assertEqual(result["known_association_disconnected_nullity"],2)
        self.assertEqual(result["known_association_crossed_nullity"],1)

    def test_generated_pair_has_identical_legal_input(self):
        a,ta=generate("single_ghost_no_anchor",0)
        b,tb=generate("thin_segregated_no_anchor",0)
        np.testing.assert_array_equal(a.xyz_mm,b.xyz_mm)
        np.testing.assert_array_equal(a.frame,b.frame)
        self.assertFalse(np.array_equal(ta["clean_xyz_mm"],tb["clean_xyz_mm"]))

    def test_guard_keeps_ambiguous_pair(self):
        a,_=generate("thin_segregated_no_anchor",0)
        out,b,info=estimate(a,"joint_guarded")
        self.assertEqual(info["status"],"WAIT")
        np.testing.assert_array_equal(out,a.xyz_mm)

    def test_crossed_layers_survive(self):
        a,_=generate("crossed_balanced",0)
        out,b,info=estimate(a,"joint_guarded")
        self.assertEqual(info["status"],"APPLY")
        self.assertEqual(info["k"],2)
        self.assertGreater(np.ptp(out[:,2]),6.)

    def test_anchor_removes_single_ghost(self):
        a,_=generate("single_ghost_anchor",0)
        out,b,info=estimate(a,"joint_guarded")
        self.assertEqual(info["status"],"APPLY")
        self.assertEqual(info["k"],1)
        self.assertLess(np.std(out[a.roi==0,2]),.1)

    def test_source_inputs_are_not_modified(self):
        a,_=generate("crossed_balanced",0);before=a.xyz_mm.copy()
        estimate(a,"joint_forced")
        np.testing.assert_array_equal(before,a.xyz_mm)

    def test_frame_center_can_destroy_segregated_thin_layer(self):
        a,truth=generate("thin_segregated_no_anchor",0)
        out,_,_=estimate(a,"frame_center_then_xyz")
        labels=truth["layer"]
        self.assertLess(abs(out[labels==1,2].mean()-out[labels==0,2].mean()),.5)


if __name__=="__main__":unittest.main()
