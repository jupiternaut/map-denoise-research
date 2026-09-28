"""Small independent counterexamples for joint-route metric semantics."""
import unittest
import numpy as np
from evaluate import measure, route_count_random, valid_route, summarize, NUMERIC, PRIMARY, SCENES, CONDS


class JointEvaluationTests(unittest.TestCase):
    def test_route_count_matches_separately_and_is_deterministic(self):
        route = np.array([0,0,1,2,2, 0,1,1,1,2],dtype=np.uint8)
        support = np.array([True]*5+[False]*5)
        random = route_count_random(route,support,20260926)
        np.testing.assert_array_equal(random,route_count_random(route,support,20260926))
        for group in (False,True):
            np.testing.assert_array_equal(np.bincount(random[support==group],minlength=3),
                                          np.bincount(route[support==group],minlength=3))

    def test_joint_regret_and_incremental_capture_are_different(self):
        errors = dict(d0=np.array([3.,3.,3.,100.]), dA=np.array([1.,4.,2.,200.]),
                      dB=np.array([2.,1.,1.,0.]),support=np.array([True,True,True,False]),
                      move2_A=np.array([1.,4.,9.,16.]),move2_B=np.array([4.,9.,16.,25.]))
        route = np.array([1,2,0,2],dtype=np.uint8)
        m = measure(errors,route)
        self.assertAlmostEqual(m['source_MSE_mm2'],11/3)
        self.assertAlmostEqual(m['source_MAE_mm'],5/3)
        self.assertAlmostEqual(m['joint_oracle_regret_MSE_mm2'],8/3)
        self.assertAlmostEqual(m['B_incremental_MSE_mm2'],11/3)
        self.assertAlmostEqual(m['B_captured_incremental_MSE_mm2'],8/3)
        self.assertAlmostEqual(m['B_incremental_capture_fraction'],8/11)
        self.assertAlmostEqual(m['B_net_incremental_MSE_mm2'],8/3)
        self.assertEqual(m['B_selected_oracle_regret_MSE_mm2'],0.)
        self.assertAlmostEqual(m['B_fraction'],.5)
        self.assertAlmostEqual(m['B_support_fraction'],1/3)
        self.assertAlmostEqual(m['accepted_fraction'],.75)
        self.assertAlmostEqual(m['move_RMS_mm'],np.sqrt(35/4))
        self.assertEqual(m['harmed_fraction'],0.)

    def test_gross_capture_does_not_hide_B_incremental_harm(self):
        errors = dict(d0=np.array([3.,3.]),dA=np.array([1.,4.]),dB=np.array([2.,1.]),
                      support=np.ones(2,bool),move2_A=np.ones(2),move2_B=np.ones(2))
        m = measure(errors,np.array([2,2],dtype=np.uint8))
        self.assertEqual(m['B_captured_incremental_MSE_mm2'],4.)
        self.assertEqual(m['B_net_incremental_MSE_mm2'],2.5)
        self.assertEqual(m['B_incremental_harm_MSE_mm2'],1.5)
        self.assertEqual(m['B_selected_oracle_regret_MSE_mm2'],1.5)

    def test_acceptance_is_not_actual_edit(self):
        errors = dict(d0=np.array([1.,1.]),dA=np.array([1.,1.]),dB=np.array([1.,1.]),
                      support=np.ones(2,bool),move2_A=np.zeros(2),move2_B=np.zeros(2))
        m = measure(errors,np.array([1,2],dtype=np.uint8))
        self.assertEqual(m['accepted_fraction'],1.)
        self.assertEqual(m['moved_fraction'],0.)
        self.assertEqual(m['joint_oracle_regret_MSE_mm2'],0.)
        self.assertIsNone(m['B_incremental_capture_fraction'])

    def test_invalid_route_rejected(self):
        for route in (np.array([0,1],dtype=bool),np.array([0,3],dtype=np.uint8),
                      np.array([[0,1]],dtype=np.uint8)):
            with self.assertRaises(AssertionError):
                valid_route(route,2)

    def test_equal_scene_roi_means_and_ratio_after_aggregation(self):
        rows = []
        arms = ['identity',PRIMARY] + [f'random_route_count__{PRIMARY}__seed{i}' for i in range(10)]
        for scene_index,scene in enumerate(SCENES,1):
            for roi in range(4):
                for condition in CONDS:
                    for arm in arms:
                        metric = dict.fromkeys(NUMERIC,0.)
                        metric.update(source_MSE_mm2=scene_index*(10. if arm == 'identity' else 9.),
                                      B_incremental_MSE_mm2=scene_index*(roi+1),
                                      B_captured_incremental_MSE_mm2=1.)
                        rows.append(dict(scene=scene,roi=f'roi{roi}',condition=condition,arm=arm,
                                         n_source=100**roi,**metric))
        summary = summarize(rows)['exposed_replay']['native'][PRIMARY]
        self.assertEqual(summary['source_MSE_mm2'],18.)
        self.assertAlmostEqual(summary['MSE_gain_percent'],10.)
        self.assertAlmostEqual(summary['B_incremental_capture_fraction'],.2)
        self.assertEqual(summary['wins'],12)


if __name__ == '__main__':
    unittest.main()
