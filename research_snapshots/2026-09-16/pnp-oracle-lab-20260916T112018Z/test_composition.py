import unittest

import composition
import planning


class CompositionTests(unittest.TestCase):
    def row(self, target=0xAAAA, prediction=0xAAAA):
        return {"family": "test", "target_id": "test0", "target_mask": str(target),
                "regime": "uniform", "m": "8", "rep": "0", "method": "exact",
                "pred_mask": str(prediction), "expression": "test"}

    def test_target_changes_score_not_deployed_policy(self):
        _, a = composition.evaluate_one(self.row(target=0xAAAA, prediction=0xAAAA), 2, "adaptive")
        _, b = composition.evaluate_one(self.row(target=0x5555, prediction=0xAAAA), 2, "adaptive")
        self.assertEqual(a["deployed_policy"], b["deployed_policy"])
        self.assertEqual(a["row"]["actual_error"], 0)
        self.assertEqual(b["row"]["actual_error"], 1)

    def test_full_sensing_does_not_repair_wrong_model(self):
        for model in (0, 0xAAAA, 0xCAFE, 65535):
            result, _ = composition.evaluate_one(self.row(target=0x1234, prediction=model), 4, "adaptive")
            self.assertEqual(result["actual_error"], (model ^ 0x1234).bit_count() / 16)

    def test_unavailable_sensor_has_irreducible_error(self):
        result = planning.solve_adaptive(0xFF00, 4, allowed_bits=(0, 1, 2))
        self.assertEqual(result["error"], 0.5)
        for x in range(8):
            self.assertEqual(planning.replay(result["policy"], x),
                             planning.replay(result["policy"], x + 8))

    def test_same_class_oracle_cannot_be_beaten(self):
        for target in (0xCAFE, 0xAAAA, 0x6666):
            for pred in (0, 0xFF00, 0x6666):
                for budget in (0, 1, 2, 3, 4):
                    for kind in ("adaptive", "batch"):
                        result, _ = composition.evaluate_one(self.row(target, pred), budget, kind)
                        self.assertGreaterEqual(result["actual_error"], result["oracle_error"])


if __name__ == "__main__":
    unittest.main()

