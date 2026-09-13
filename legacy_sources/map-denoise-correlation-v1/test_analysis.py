import unittest
import numpy as np
from run_experiment import evaluate, selected_k


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 6.], [1., 1., 6.]])
        self.truth = {'clean_xyz_mm': self.clean, 'labels': np.array([0, 0, 1, 1])}
        self.frame = np.array([0, 1, 0, 1])

    def test_perfect_and_collapse_are_distinct(self):
        perfect = evaluate(self.clean, self.clean, self.truth, 6., self.frame)
        collapsed = self.clean.copy(); collapsed[:, 2] = 3.
        bad = evaluate(collapsed, self.clean, self.truth, 6., self.frame)
        self.assertEqual(perfect['normal_mae_mm'], 0.)
        self.assertEqual(perfect['gap_retention'], 1.)
        self.assertEqual(bad['normal_mae_mm'], 3.)
        self.assertEqual(bad['layer_gap_hat_mm'], 0.)
        self.assertTrue(bad['layer_separation_lost'])

    def test_common_translation_not_hidden(self):
        out = self.clean + [0., 0., 2.]
        score = evaluate(out, self.clean, self.truth, 6., self.frame)
        self.assertEqual(score['normal_mae_mm'], 2.)
        self.assertEqual(score['centered_normal_mae_mm'], 0.)
        self.assertEqual(score['layer_gap_error_mm'], 0.)

    def test_single_surface_and_k_parsing(self):
        truth = {'clean_xyz_mm': self.clean * [1., 1., 0.], 'labels': np.zeros(4, int)}
        score = evaluate(truth['clean_xyz_mm'], self.clean, truth, 0., self.frame)
        self.assertIsNone(score['layer_gap_hat_mm'])
        self.assertEqual(selected_k({'k': 2}), 2)
        self.assertEqual(selected_k({'fit': {'0': {'k': 1}}}), 1)
        self.assertIsNone(selected_k({'status': 'UNCHANGED'}))


if __name__ == '__main__':
    unittest.main()
