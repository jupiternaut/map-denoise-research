"""Posthoc reporting regression: unchanged coordinates are not KEEP status."""
import unittest
from summarize import summarize


class SummaryStatusTest(unittest.TestCase):
    def test_incompatible_is_not_keep(self):
        common = {"observation_id": "fixture", "covered": False,
                  "case": "equal_full_negative", "status": "INCOMPATIBLE"}
        rows = [{**common, "initial_ae_mm": "1", "ae_mm": "1",
                 "output": "2", "incumbent": "2"},
                {**common, "initial_ae_mm": "0", "ae_mm": "0",
                 "output": "1", "incumbent": "1"}]
        result = summarize(rows)
        self.assertEqual(result["correct_kept"], 0)
        self.assertEqual(result["correct_output_unchanged"], 1)
        self.assertEqual(result["correct_incompatible"], 1)
        self.assertEqual(result["negative_offset_kept"], 0)
        self.assertEqual(result["negative_output_unchanged"], 1)
        self.assertEqual(result["n_incompatible"], 2)

    def test_keep_and_incompatible_have_equal_error_but_different_status(self):
        common = {"observation_id": "fixture", "covered": True,
                  "case": "equal_full_negative", "status": "KEEP"}
        rows = [{**common, "initial_ae_mm": "1", "ae_mm": "1",
                 "output": "2", "incumbent": "2"},
                {**common, "initial_ae_mm": "0", "ae_mm": "0",
                 "output": "1", "incumbent": "1"}]
        result = summarize(rows)
        self.assertEqual(result["correct_kept"], 1)
        self.assertEqual(result["negative_offset_kept"], 1)
        self.assertEqual(result["n_incompatible"], 0)


if __name__ == "__main__":
    unittest.main()
