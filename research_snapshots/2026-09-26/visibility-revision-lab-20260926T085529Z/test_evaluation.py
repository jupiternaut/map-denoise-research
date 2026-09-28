"""Metric contract checks independent of replay labels."""
import unittest
import numpy as np
from evaluate import measure, route_count_random, valid_route, summarize, add_ratios, NUMERIC, SCENES, CONDS


class EvaluationTests(unittest.TestCase):
    def test_route_counts_preserve_support_groups(self):
        route=np.array([0,0,1,2,2,0,1,1,1,2],np.uint8)
        support=np.array([True]*5+[False]*5)
        result=route_count_random(route,support,40)
        np.testing.assert_array_equal(result,route_count_random(route,support,40))
        for group in (False,True):
            np.testing.assert_array_equal(np.bincount(result[support==group],minlength=3),np.bincount(route[support==group],minlength=3))

    def test_gross_increment_is_not_net_oracle_headroom(self):
        errors=dict(d0=np.array([3.,3.]),dA=np.array([1.,4.]),dB=np.array([2.,1.]),
                    support=np.ones(2,bool),move2_A=np.ones(2),move2_B=np.ones(2))
        m=measure(errors,np.array([2,2],np.uint8))
        add_ratios(m,9.)
        self.assertEqual(m['B_captured_incremental_MSE_mm2'],4.)
        self.assertEqual(m['B_net_incremental_MSE_mm2'],2.5)
        self.assertEqual(m['B_incremental_harm_MSE_mm2'],1.5)
        self.assertEqual(m['source_MSE_mm2'],2.5)
        self.assertEqual(m['joint_oracle_MSE_mm2'],1.)
        self.assertEqual(m['oracle_headroom_capture_fraction'],6.5/8.)

    def test_no_movement_despite_nonzero_route(self):
        errors=dict(d0=np.ones(2),dA=np.ones(2),dB=np.ones(2),support=np.ones(2,bool),
                    move2_A=np.zeros(2),move2_B=np.zeros(2))
        m=measure(errors,np.array([1,2],np.uint8));add_ratios(m,1.)
        self.assertEqual(m['accepted_fraction'],1.)
        self.assertEqual(m['moved_fraction'],0.)
        self.assertIsNone(m['oracle_headroom_capture_fraction'])

    def test_invalid_routes_rejected(self):
        for route in (np.array([0,1],bool),np.array([0,3],np.uint8),np.array([[0,1]],np.uint8)):
            with self.assertRaises(AssertionError):valid_route(route,2)

    def test_equal_ROI_scene_means_and_descriptive_best(self):
        rows=[];primary='locked';arms=['identity',primary,'other']+[f'random_route_count__{primary}__seed{i}' for i in range(10)]
        for index,scene in enumerate(SCENES,1):
            for roi in range(4):
                for condition in CONDS:
                    for arm in arms:
                        m=dict.fromkeys(NUMERIC,0.)
                        m.update(source_MSE_mm2=index*(10. if arm=='identity' else 8. if arm=='other' else 9.),
                                 joint_oracle_MSE_mm2=index*5.,B_incremental_MSE_mm2=index*(roi+1),B_captured_incremental_MSE_mm2=1.)
                        rows.append(dict(scene=scene,roi=f'r{roi}',condition=condition,arm=arm,n_source=100**roi,**m))
        result=summarize(rows,primary,{primary,'other'})
        cell=result['exposed_replay']['native'][primary]
        self.assertEqual(cell['source_MSE_mm2'],18.)
        self.assertEqual(cell['MSE_gain_percent'],10.)
        self.assertEqual(cell['oracle_headroom_capture_fraction'],.2)
        self.assertEqual(cell['B_incremental_capture_fraction'],.2)
        self.assertEqual(cell['wins'],12)
        self.assertEqual(result['descriptive_best_locked_arm']['native']['arm'],'other')

    def test_negative_gain_not_clipped(self):
        cell=dict(source_MSE_mm2=12.,joint_oracle_MSE_mm2=5.,B_incremental_MSE_mm2=1.,B_captured_incremental_MSE_mm2=.2)
        add_ratios(cell,10.)
        self.assertEqual(cell['oracle_headroom_capture_fraction'],-.4)
        self.assertEqual(cell['MSE_gain_percent'],-20.)


if __name__=='__main__':unittest.main()
