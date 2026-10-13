"""Small non-confirmation checks; never generates the formal holdout seeds."""
from collections import Counter
from fractions import Fraction as Q
import math
from pathlib import Path
import random
import tempfile
import unittest

import data_builder as b


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public, cls.facts, cls.tasks = b.build_population('dev', seeds=(919191,))

    def test_complete_new_continuous_worlds_and_independent_forward(self):
        self.assertEqual(len(self.public), 6)
        self.assertEqual(len(self.tasks), 42)
        self.assertEqual(Counter((f['rear_branch'], f['stratum']) for f in self.facts),
                         Counter((branch, stratum) for branch in b.BRANCHES for stratum in b.STRATA))
        for p, f in zip(self.public, self.facts):
            z = b.exact(f['truth'])
            self.assertTrue(550 < z < 650 and z not in b.OLD_DEPTHS)
            self.assertTrue(f['independent_exact_verified'])
            self.assertLessEqual(f['float_forward_max_abs_error'], 1e-12)
            self.assertLessEqual(f['float_observation_max_abs_error'], 1e-12)
            packet = p['observation']
            self.assertEqual(set(packet), {'photo', 'camera', 'appearance'})
            self.assertEqual(len(packet['photo']), 432)
            self.assertTrue(all(math.isfinite(y) and 0 <= y <= 1 for y in packet['photo']))
            self.assertEqual(set(packet['camera']), {'u', 'Xf', 'Zf', 'survey_bound', 'localization_bound'})
            self.assertEqual(set(packet['appearance']), {'coupons', 'references', 'bias_bound', 'transfer_bound', 'noise_bound'})

    def test_task_correctness_bounds_no_center_leak(self):
        for f in self.facts:
            z = b.exact(f['truth'])
            ts = [t for t in self.tasks if t['observation_id'] == f['observation_id']]
            errors = [b.exact(t['incumbent_exact']) - z for t in ts]
            self.assertEqual(errors.count(0), 1)
            self.assertTrue(all(Q(1, 10) <= abs(x) <= 8 for x in errors if x))
            self.assertNotEqual(sum(errors, Q(0)), 0)
            self.assertNotEqual(sorted(errors)[3], 0)
            self.assertEqual(len(set(abs(x) for x in errors if x)), 6)
            self.assertTrue(all(540 <= t['incumbent'] <= 660 for t in ts))

    def test_equal_full_control_and_valid_parameter_bounds(self):
        controls = [f for f in self.facts if f['negative_control']]
        self.assertEqual(len(controls), 1)
        f = controls[0]
        self.assertEqual(f['rear_branch'], 'full')
        self.assertEqual(b.exact(f['eta']), 0)
        self.assertEqual(len(set(map(b.exact, f['view_colors']))), 1)
        verifier = b.dependencies()[2]
        first = verifier.independent_rgb_per_view('full', Q(555), f['actual_offsets'], f['view_colors'])
        last = verifier.independent_rgb_per_view('full', Q(645), f['actual_offsets'], f['view_colors'])
        self.assertEqual(first, last)
        for f in self.facts:
            self.assertLessEqual(b.exact(f['actual_material_spread']), Q(1, 10))
            self.assertEqual(b.exact(f['actual_offsets'][1]), 0)
            self.assertTrue(all(abs(b.exact(e)) <= Q(1, 5) for e in f['actual_offsets']))
            self.assertTrue(all(abs(b.exact(e)) <= Q(1, 100) for e in f['image_noise'] + f['actual_view_bias']))

    def test_repeatability_of_nonconfirmation_fixture(self):
        self.assertEqual((self.public, self.facts, self.tasks), b.build_population('dev', seeds=(919191,)))

    def test_holdout_freeze_and_no_overwrite_guards(self):
        with self.assertRaises(ValueError):
            b.check_freeze(None)
        with self.assertRaises(ValueError):
            b.generate('dev', b.ROOT)
        with self.assertRaises(FileExistsError):
            b.generate('dev', b.ROOT / 'task_bundle')


if __name__ == '__main__':
    unittest.main()
