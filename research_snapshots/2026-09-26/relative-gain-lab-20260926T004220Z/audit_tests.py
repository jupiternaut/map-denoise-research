"""Focused tests of independent audit math and confound detection."""
import unittest
import numpy as np
from audit_array_checks import metrics, verify_matching, threshold_rows, pick_threshold


class AuditMathTests(unittest.TestCase):
    def test_error_reduction_and_actual_movement_are_distinct(self):
        result = metrics([4, 1, 9], [1, 4, 9], [True, False, True], [2, 3, 0], [True, True, False])
        self.assertEqual(result["source_MSE_mm2"], 1)
        self.assertEqual(result["benefit_sum_mm2"], 3)
        self.assertEqual(result["harm_sum_mm2"], 0)
        self.assertEqual(result["accepted_fraction"], 2 / 3)
        self.assertEqual(result["moved_fraction"], 1 / 3)

    def test_fixed_support_prevents_output_crop_escape(self):
        result = metrics([1, 1], [100, 0], [True, True], [4, 1], [True, False])
        self.assertEqual(result["source_MSE_mm2"], 100)
        self.assertEqual(result["harm_sum_mm2"], 99)

    def test_matching_detects_support_imbalance(self):
        with self.assertRaises(AssertionError):
            verify_matching([True, False], [False, True], [1, 1], [True, False])

    def test_matching_detects_displacement_imbalance(self):
        with self.assertRaises(AssertionError):
            verify_matching([True, False], [False, True], [0.2, 2], [True, True], [0.5, 1])

    def test_strict_threshold_ties_prefer_keep(self):
        rows = threshold_rows([0, 1, 1], [1, 4, 4], [1, 1, 1], [0, 1, 2],
                              ["native", "minus3", "plus3"], [1, 1, 1], [True] * 3, [-1, 0, 1])
        best = pick_threshold(rows)
        self.assertEqual(best["threshold"], 0)
        self.assertTrue(best["native_feasible"])

    def test_native_priority_can_be_infeasible(self):
        rows = threshold_rows([1, 1, 1], [1, 4, 4], [2, 1, 1], [0, 1, 2],
                              ["native", "minus3", "plus3"], [1, 1, 1], [True] * 3, [0, 1])
        self.assertEqual(pick_threshold(rows, native_priority=True)["status"], "infeasible_KEEP")


if __name__ == "__main__":
    unittest.main()
