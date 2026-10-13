"""Interface fixtures; no development or hidden data loaded by these tests."""
import importlib.util
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('a12_library_method_test_target', Path(__file__).with_name('method.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class LibrarySelectorTests(unittest.TestCase):
    def test_single_name_one_call_both_budgets_and_measurement_only(self):
        public = {'photo': [1.0] * 432, 'camera': {'u': [.08, 0, -.08]},
                  'appearance': {'coupons': [.5] * 9}, 'truth': object(),
                  'observation_id': object(), 'incumbents': object()}
        for budget in (6, 24):
            with patch.object(m, '_prepare_method', return_value={'hull': [580., 620.]}) as prepare:
                state = m.prepare(public, budget)
            prepare.assert_called_once()
            name, forwarded, seconds = prepare.call_args.args
            self.assertEqual(name, m.SELECTED_METHOD)
            self.assertEqual(set(forwarded), {'photo', 'camera', 'appearance'})
            self.assertEqual(seconds, budget)
            self.assertIsNot(forwarded['photo'], public['photo'])
            self.assertEqual(state['selector']['selected_method'], m.SELECTED_METHOD)
            json.dumps(state, allow_nan=False)

    def test_hull_and_identity_decode_preserve_correct_inside(self):
        for value, expected in ((570., 580.), (600., 600.), (630., 620.)):
            self.assertEqual(m.decode({'hull': [580., 620.]}, value), expected)
        for hull in (None, []):
            self.assertEqual(m.decode({'hull': hull}, 611.25), 611.25)
        self.assertEqual(m.decode({'hull': None}, 700.), 660.)
        self.assertEqual(m.decode({'hull': None}, 500.), 540.)

    def test_invalid_budgets_and_incumbents(self):
        for value in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                m.prepare({}, value)
        for value in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                m.decode({'hull': None}, value)

    def test_fixed_name_and_source_identity(self):
        self.assertIn(m.SELECTED_METHOD, ('coarse', 'depth', 'accumulated'))
        self.assertEqual(Path(m._library.__file__).resolve(), (m.LIBRARY / 'methods.py').resolve())


if __name__ == '__main__':
    unittest.main()
