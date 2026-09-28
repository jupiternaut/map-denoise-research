"""Pure synthetic calibration contracts; no fitting or experiment I/O.

Only the two public calibration functions are imported from train_experiment.
The fixture contains deliberately unequal case sizes so a pooled-row metric
cannot accidentally pass the case-equal weighting check.
"""

import unittest

import numpy as np

from train_experiment import calibrate, threshold_statistics


def synthetic_cases(specifications):
    """Build only the arrays read by calibration, never a feature/label file."""
    values = {key: [] for key in ("case_id", "condition", "e0", "e1")}
    for case_id, (condition, count, e0, e1) in enumerate(specifications):
        values["case_id"].extend([case_id] * count)
        values["condition"].extend([condition] * count)
        values["e0"].extend([e0] * count)
        values["e1"].extend([e1] * count)
    data = {key: np.asarray(value) for key, value in values.items()}
    data["calibration_support"] = np.ones(len(data["e0"]), dtype=bool)
    return data


class TrainingContractTests(unittest.TestCase):
    def test_explicit_all_remains_all_across_score_ranges(self):
        data = synthetic_cases([
            ("native", 2, 4.0, 1.0),
            ("minus3", 5, 4.0, 1.0),
            ("plus3", 3, 4.0, 1.0),
        ])
        scores = np.arange(len(data["e0"]), dtype=float)
        choices, rows = calibrate(data, scores, default=None)
        all_choice = choices["balanced"]
        self.assertEqual(all_choice["action"], "all")
        self.assertIsNone(all_choice["threshold"])
        self.assertEqual(sum(row["action"] == "all" for row in rows), 1)
        self.assertEqual(sum(row["action"] == "keep" for row in rows), 1)
        self.assertTrue(choices["native_priority"]["feasible"])
        for shifted in (scores - 1e12, scores + 1e12, np.full(len(scores), -1e30)):
            with self.subTest(score_min=float(shifted.min())):
                statistics = threshold_statistics(
                    data, shifted, all_choice["threshold"], all_choice["action"]
                )
                self.assertEqual(statistics["accepted_fraction"], 1.0)
                self.assertEqual(statistics["objective_relative_MSE"], 0.25)

    def test_explicit_keep_remains_keep_across_score_ranges(self):
        data = synthetic_cases([
            ("native", 2, 1.0, 2.0),
            ("minus3", 5, 1.0, 2.0),
            ("plus3", 3, 1.0, 2.0),
        ])
        scores = np.arange(len(data["e0"]), dtype=float)
        choices, _ = calibrate(data, scores, default=None)
        keep_choice = choices["balanced"]
        self.assertEqual(keep_choice["action"], "keep")
        self.assertIsNone(keep_choice["threshold"])
        for shifted in (scores - 1e12, scores + 1e12, np.full(len(scores), 1e30)):
            with self.subTest(score_max=float(shifted.max())):
                statistics = threshold_statistics(
                    data, shifted, keep_choice["threshold"], keep_choice["action"]
                )
                self.assertEqual(statistics["accepted_fraction"], 0.0)
                self.assertEqual(statistics["objective_relative_MSE"], 1.0)

    def test_infeasible_native_constraint_returns_explicit_keep(self):
        # Identical scores permit only ALL or KEEP. ALL damages native, and
        # KEEP cannot achieve the required 5% injected-error improvement.
        data = synthetic_cases([
            ("native", 2, 1.0, 1.1),
            ("minus3", 7, 4.0, 1.0),
            ("plus3", 3, 4.0, 1.0),
        ])
        scores = np.zeros(len(data["e0"]))
        choices, rows = calibrate(data, scores, default=0.0)
        native = choices["native_priority"]
        self.assertFalse(native["feasible"])
        self.assertTrue(native["keep_all"])
        self.assertEqual(native["action"], "keep")
        self.assertIsNone(native["threshold"])
        self.assertIn("reason", native)
        self.assertFalse(any(
            row["native_MSE"] <= row["native_identity_MSE"] + 1e-12
            and row["injected_relative_MSE"] <= 0.95 + 1e-12
            for row in rows
        ))
        after_shift = threshold_statistics(
            data, scores + 1e20, native["threshold"], native["action"]
        )
        self.assertEqual(after_shift["accepted_fraction"], 0.0)

    def test_statistics_use_case_equal_weights_with_unequal_row_counts(self):
        data = synthetic_cases([
            ("native", 1, 2.0, 1.0),
            ("minus3", 8, 10.0, 20.0),
            ("plus3", 3, 1.0, 0.5),
        ])
        scores = np.array([1.0] + [1.0] * 4 + [-1.0] * 4 + [-1.0] * 3)
        statistics = threshold_statistics(data, scores, 0.0)
        # Case relative MSEs are 0.5, 1.5, 1; acceptance is 1, 0.5, 0.
        self.assertAlmostEqual(statistics["objective_relative_MSE"], 1.0)
        self.assertAlmostEqual(statistics["accepted_fraction"], 0.5)
        self.assertAlmostEqual(statistics["native_MSE"], 1.0)
        self.assertAlmostEqual(statistics["native_identity_MSE"], 2.0)
        self.assertAlmostEqual(statistics["injected_relative_MSE"], 1.25)
        self.assertNotAlmostEqual(statistics["accepted_fraction"], float(np.mean(scores > 0)))

    def test_excluded_rows_do_not_change_metrics_or_threshold_grid(self):
        data = synthetic_cases([
            ("native", 2, 2.0, 1.0),
            ("minus3", 5, 4.0, 1.0),
            ("plus3", 3, 4.0, 1.0),
        ])
        scores = np.linspace(-1, 1, len(data["e0"]))
        base_choices, base_rows = calibrate(data, scores, default=None)
        extra = {
            "case_id": np.array([0, 1, 2]),
            "condition": np.array(["native", "minus3", "plus3"]),
            "e0": np.full(3, 1e6),
            "e1": np.full(3, 1e12),
            "calibration_support": np.zeros(3, dtype=bool),
        }
        extended = {key: np.concatenate([value, extra[key]]) for key, value in data.items()}
        extended_scores = np.concatenate([scores, [-1e25, 1e25, 1e20]])
        extra_choices, extra_rows = calibrate(extended, extended_scores, default=None)
        self.assertEqual(base_choices, extra_choices)
        self.assertEqual(base_rows, extra_rows)

    def test_threshold_is_strict_and_natural_threshold_is_preserved(self):
        data = synthetic_cases([
            ("native", 2, 1.0, 0.5),
            ("minus3", 3, 1.0, 0.5),
            ("plus3", 4, 1.0, 0.5),
        ])
        scores = np.full(len(data["e0"]), 0.5)
        choices, _ = calibrate(data, scores, default=0.5)
        self.assertEqual(choices["natural"]["action"], "threshold")
        self.assertEqual(choices["natural"]["threshold"], 0.5)
        self.assertEqual(choices["natural"]["accepted_fraction"], 0.0)
        self.assertEqual(choices["balanced"]["action"], "all")


if __name__ == "__main__":
    unittest.main()
