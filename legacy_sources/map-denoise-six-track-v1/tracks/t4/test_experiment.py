import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiment import metric, metric_probes, support_graph


class EvaluationTests(unittest.TestCase):
    def test_paired_worlds_same_input_opposite_correct_output(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = metric_probes(Path(folder))
        self.assertEqual(len(rows), 4)
        self.assertEqual([round(r['metrics']['plane_distance_mm'], 6) for r in rows], [0., 4., 4., 0.])

    def test_exact_identity(self):
        p = np.array([[0., 0., 0.], [1., 0., 0.]])
        result = metric(p, p, [0.])
        self.assertEqual(result['chamfer_mm'], 0.)
        self.assertTrue(all(x['f1'] == 1. for x in result['prf']))

    def test_frame_relabeling_does_not_change_support(self):
        rng = np.random.default_rng(99)
        points = rng.normal(size=(100, 3))
        frames = np.repeat(np.arange(10), 10)
        a = support_graph(points, frames, np.array([0., 0., 1.]))
        b = support_graph(points, 123 - 7 * frames, np.array([0., 0., 1.]))
        for key in a:
            self.assertAlmostEqual(a[key], b[key])

    def test_pure_between_frame_offset(self):
        p = np.zeros((20, 3)); p[10:, 2] = .008
        frames = np.repeat([0, 1], 10)
        info = support_graph(p, frames, np.array([0., 0., 1.]))
        self.assertAlmostEqual(info['between_frame_variance_fraction'], 1.)
        self.assertEqual(info['frames_min3_each_median_side'], 0)


if __name__ == '__main__':
    unittest.main()
