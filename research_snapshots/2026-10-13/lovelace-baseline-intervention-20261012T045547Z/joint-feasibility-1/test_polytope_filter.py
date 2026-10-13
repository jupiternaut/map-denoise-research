"""Only small, hand-built non-confirmatory fixtures; no frozen data reads."""

from copy import deepcopy
from fractions import Fraction as Q
import json
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np
import polytope_filter as polytope


BOX = ((Q(0), Q(1)), (Q(0), Q(1)))
CORRELATED = (("upper", (Q(1), Q(1)), Q(99, 100)),
              ("lower", (Q(1), Q(1)), Q(101, 100)))
RESULT_FIELDS = {"status", "certificate", "witness", "lp_status", "seconds", "lp_seconds",
                 "single_row_contradiction", "single_row_contradiction_rows",
                 "row_count", "dimension", "reason"}


def eight_sweep_interval_contract(rows, bounds):
    """The frozen signed-row snapshot rule, on an independent two-variable fixture."""
    lower, upper = [pair[0] for pair in bounds], [pair[1] for pair in bounds]
    for _ in range(8):
        for kind, coefficients, bound in rows:
            values = tuple((lower[j] if a >= 0 else upper[j]) if kind == "upper"
                           else (upper[j] if a >= 0 else lower[j])
                           for j, a in enumerate(coefficients))
            total = sum((a * value for a, value in zip(coefficients, values)), Q(0))
            if (total > bound if kind == "upper" else total < bound):
                return True
            for j, a in enumerate(coefficients):
                if not a:
                    continue
                candidate = (bound - (total - a * values[j])) / a
                if (kind == "upper" and a > 0) or (kind == "lower" and a < 0):
                    upper[j] = min(upper[j], candidate)
                else:
                    lower[j] = max(lower[j], candidate)
                if lower[j] > upper[j]:
                    return True
    return False


def fake_lp(x, marginals, *, status=0, success=True, fun=0.0):
    return SimpleNamespace(status=status, success=success, fun=fun, x=np.asarray(x),
                           ineqlin=SimpleNamespace(marginals=np.asarray(marginals)))


class PolytopeFilterTests(unittest.TestCase):
    def assert_json_record(self, record):
        self.assertEqual(set(record), RESULT_FIELDS)
        self.assertIn(record["status"], polytope.STATUSES)
        self.assertGreaterEqual(record["seconds"], 0)
        self.assertGreaterEqual(record["lp_seconds"], 0)
        json.dumps(record, allow_nan=False)

    def test_combination_rejects_where_eight_sweeps_do_not(self):
        self.assertFalse(eight_sweep_interval_contract(CORRELATED, BOX))
        result = polytope.classify(CORRELATED, BOX)
        self.assert_json_record(result)
        self.assertEqual(result["status"], "CERTIFIED_INFEASIBLE")
        self.assertFalse(result["single_row_contradiction"])
        self.assertEqual(result["single_row_contradiction_rows"], [])
        self.assertTrue(polytope.certificate_check(CORRELATED, BOX, result["certificate"]))
        self.assertIsNone(result["witness"])
        self.assertEqual(len(result["certificate"]["sparse_weights"]), 2)
        self.assertEqual(result["certificate"]["aggregate_coefficients"],
                         [polytope.encode(Q(0)), polytope.encode(Q(0))])

    def test_feasible_boundary_is_never_deleted(self):
        rows = (("upper", (Q(1), Q(1)), Q(1)),
                ("lower", (Q(1), Q(1)), Q(1)))
        result = polytope.classify(rows, BOX)
        self.assert_json_record(result)
        self.assertIn(result["status"], ("FEASIBLE_WITNESS", "UNKNOWN"))
        self.assertIsNone(result["certificate"])
        if result["status"] == "FEASIBLE_WITNESS":
            self.assertTrue(polytope.witness_check(rows, BOX, result["witness"]))

    def test_negative_coefficients_with_nonzero_aggregate(self):
        rows = (("upper", (Q(-1), Q(0)), Q(-3, 4)),
                ("upper", (Q(1), Q(-1)), Q(-3, 10)))
        result = polytope.classify(rows, BOX)
        self.assertEqual(result["status"], "CERTIFIED_INFEASIBLE")
        self.assertFalse(result["single_row_contradiction"])
        self.assertTrue(polytope.certificate_check(rows, BOX, result["certificate"]))
        self.assertNotEqual(result["certificate"]["aggregate_coefficients"],
                            [polytope.encode(Q(0)), polytope.encode(Q(0))])
        widened = (BOX[0], (Q(0), Q(2)))
        self.assertFalse(polytope.certificate_check(rows, widened, result["certificate"]))

    def test_single_row_diagnostic_and_pinned_variable(self):
        rows = (("upper", (Q(1), Q(0)), Q(1, 4)),)
        bounds = ((Q(1, 3), Q(1, 3)), BOX[1])
        with patch.object(polytope, "linprog") as solve:
            result = polytope.classify(rows, bounds)
        solve.assert_not_called()
        self.assertEqual(result["status"], "CERTIFIED_INFEASIBLE")
        self.assertTrue(result["single_row_contradiction"])
        self.assertEqual(result["single_row_contradiction_rows"], [0])
        self.assertIsNone(result["lp_status"])
        self.assertTrue(polytope.certificate_check(rows, bounds, result["certificate"]))

    def test_pinned_feasible_coordinate_is_exact(self):
        rows = (("lower", (Q(1), Q(1)), Q(1)),
                ("upper", (Q(1), Q(1)), Q(1)))
        bounds = ((Q(1, 3), Q(1, 3)), BOX[1])
        result = polytope.classify(rows, bounds)
        self.assertEqual(result["status"], "FEASIBLE_WITNESS")
        self.assertTrue(polytope.witness_check(rows, bounds, result["witness"]))
        self.assertEqual(result["witness"][0], polytope.encode(Q(1, 3)))

    def test_zero_duplicate_signed_rows_and_negative_box(self):
        rows = (("upper", (Q(0), Q(0)), Q(0)),
                ("lower", (Q(0), Q(0)), Q(0)),
                ("upper", (Q(-1), Q(1)), Q(0)),
                ("upper", (Q(-1), Q(1)), Q(0)),
                ("lower", (Q(-1), Q(1)), Q(0)))
        box = ((Q(-2), Q(-1)), (Q(-2), Q(-1)))
        result = polytope.classify(rows, box)
        self.assertEqual(result["status"], "FEASIBLE_WITNESS")
        self.assertTrue(polytope.witness_check(rows, box, result["witness"]))

    def test_zero_dimension_and_empty_rows(self):
        result = polytope.classify((("lower", (), Q(0)),), ())
        self.assertEqual(result["status"], "FEASIBLE_WITNESS")
        self.assertEqual(result["witness"], [])
        impossible = (("upper", (), Q(-1)),)
        result = polytope.classify(impossible, ())
        self.assertEqual(result["status"], "CERTIFIED_INFEASIBLE")
        self.assertTrue(polytope.certificate_check(impossible, (), result["certificate"]))
        result = polytope.classify((), BOX)
        self.assertTrue(polytope.witness_check((), BOX, result["witness"]))

    def test_certificate_tampering_and_changed_lower_row(self):
        certificate = polytope.classify(CORRELATED, BOX)["certificate"]
        bad = deepcopy(certificate)
        bad["sparse_weights"][0]["weight"] = polytope.encode(Q(-1))
        self.assertFalse(polytope.certificate_check(CORRELATED, BOX, bad))
        bad = deepcopy(certificate)
        bad["sparse_weights"][0]["weight"] = polytope.encode(Q(0))
        self.assertFalse(polytope.certificate_check(CORRELATED, BOX, bad))
        bad = deepcopy(certificate)
        bad["aggregate_coefficients"][0] = polytope.encode(Q(1))
        self.assertFalse(polytope.certificate_check(CORRELATED, BOX, bad))
        bad = deepcopy(certificate)
        bad["box_min"] = polytope.encode(Q(1))
        self.assertFalse(polytope.certificate_check(CORRELATED, BOX, bad))
        bad = deepcopy(certificate)
        bad["margin"] = polytope.encode(Q(0))
        self.assertFalse(polytope.certificate_check(CORRELATED, BOX, bad))
        bad = deepcopy(certificate)
        bad["sparse_weights"].append(deepcopy(bad["sparse_weights"][0]))
        self.assertFalse(polytope.certificate_check(CORRELATED, BOX, bad))
        feasible = (CORRELATED[0], ("lower", (Q(1), Q(1)), Q(98, 100)))
        self.assertFalse(polytope.certificate_check(feasible, BOX, certificate))

    def test_witness_checks_all_rows_and_bounds(self):
        rows = (("lower", (Q(1), Q(1)), Q(1)),)
        self.assertFalse(polytope.witness_check(rows, BOX, [polytope.encode(Q(0))] * 2))
        self.assertFalse(polytope.witness_check(rows, BOX,
                                             [polytope.encode(Q(2)), polytope.encode(Q(0))]))
        self.assertFalse(polytope.witness_check(rows, BOX, [0.5, 0.5]))

    def test_fake_phase_one_success_is_not_a_proof(self):
        rows = (("upper", (Q(1), Q(1)), Q(1)),
                ("lower", (Q(1), Q(1)), Q(1)))
        candidate = fake_lp([0, 0, 0], [-0.5, -0.5], fun=0.0)
        with patch.object(polytope, "linprog", return_value=candidate):
            result = polytope.classify(rows, BOX)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(result["certificate"])
        self.assertIsNone(result["witness"])

    def test_near_critical_float_margin_cannot_delete(self):
        rows = (("upper", (Q(1), Q(1)), Q(1)),
                ("lower", (Q(1), Q(1)), Q(1) - Q(1, 10**12)))
        candidate = fake_lp([0, 0, 1e-12], [-0.5, -0.5000000001], fun=1e-12)
        with patch.object(polytope, "linprog", return_value=candidate):
            result = polytope.classify(rows, BOX)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(result["certificate"])

    def test_active_exact_reconstruction_exceeds_denominator_limit(self):
        exact = Q(1, 10**12 + 39)
        rows = (("upper", (Q(1),), exact), ("lower", (Q(1),), exact))
        bounds = ((Q(0), Q(1)),)
        candidate = fake_lp([float(exact), 0], [0, 0])
        with patch.object(polytope, "linprog", return_value=candidate):
            result = polytope.classify(rows, bounds)
        self.assertEqual(result["status"], "FEASIBLE_WITNESS")
        self.assertEqual(result["reason"], "exact_active_witness")
        self.assertEqual(result["witness"], [polytope.encode(exact)])
        self.assertTrue(polytope.witness_check(rows, bounds, result["witness"]))

    def test_unsuccessful_timeout_or_numerical_status_is_unknown(self):
        for status in (1, 2, 4):
            with self.subTest(status=status):
                candidate = fake_lp([0, 1, 0.01], [-0.5, -0.5], status=status,
                                    success=False, fun=0.01)
                with patch.object(polytope, "linprog", return_value=candidate) as solve:
                    result = polytope.classify(CORRELATED, BOX)
                self.assertEqual(result["status"], "UNKNOWN")
                self.assertEqual(result["lp_status"], status)
                self.assertIsNone(result["certificate"])
                self.assertIsNone(result["witness"])
                solve.assert_called_once()

    def test_nan_marginal_or_point_or_objective_is_unknown(self):
        candidates = (fake_lp([0, 1, 0.01], [np.nan, -0.5], fun=0.01),
                      fake_lp([np.nan, 1, 0.01], [-0.5, -0.5], fun=0.01),
                      fake_lp([0, 1, 0.01], [-0.5, -0.5], fun=np.inf),
                      fake_lp([0, 1], [-0.5, -0.5], fun=0.01))
        for candidate in candidates:
            with patch.object(polytope, "linprog", return_value=candidate):
                result = polytope.classify(CORRELATED, BOX)
            self.assertEqual(result["status"], "UNKNOWN")
            self.assertIsNone(result["certificate"])
            self.assert_json_record(result)

    def test_lp_exception_is_unknown_without_retry(self):
        with patch.object(polytope, "linprog", side_effect=RuntimeError("fixture")) as solve:
            result = polytope.classify(CORRELATED, BOX)
        solve.assert_called_once()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["reason"], "lp_exception:RuntimeError")

    def test_nonfinite_float_input_is_unknown(self):
        rows = (("upper", (Q(10**1000),), Q(10**1000)),)
        with patch.object(polytope, "linprog") as solve:
            result = polytope.classify(rows, ((Q(0), Q(1)),))
        solve.assert_not_called()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["reason"], "nonfinite_lp_input")

    def test_deadline_and_single_thread_phase_one_options(self):
        candidate = fake_lp([0, 1, 0.01], [-0.5, -0.5], fun=0.01)
        deadline = time.monotonic() + 0.25
        with patch.object(polytope, "linprog", return_value=candidate) as solve:
            result = polytope.classify(CORRELATED, BOX, deadline=deadline)
        self.assertEqual(result["status"], "CERTIFIED_INFEASIBLE")
        arguments = solve.call_args.kwargs
        self.assertEqual(arguments["method"], "highs")
        self.assertEqual(arguments["options"]["threads"], 1)
        self.assertGreater(arguments["options"]["time_limit"], 0)
        self.assertLessEqual(arguments["options"]["time_limit"], 0.25)
        np.testing.assert_array_equal(arguments["A_ub"][:, -1], [-1, -1])
        np.testing.assert_array_equal(solve.call_args.args[0], [0, 0, 1])
        self.assertEqual(arguments["bounds"][-1], (0.0, None))
        with patch.object(polytope, "linprog") as solve:
            result = polytope.classify(CORRELATED, BOX, deadline=time.monotonic() - 1)
        solve.assert_not_called()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["reason"], "deadline")
        with patch.object(polytope, "linprog", return_value=candidate) as solve:
            polytope.classify(CORRELATED, BOX)
        self.assertEqual(solve.call_args.kwargs["options"]["time_limit"], 5.0)

    def test_exact_input_validation(self):
        with self.assertRaises(TypeError):
            polytope.classify((("upper", (0.5,), Q(1)),), ((Q(0), Q(1)),))
        with self.assertRaises(ValueError):
            polytope.classify((), ((Q(1), Q(0)),))
        with self.assertRaises(ValueError):
            polytope.classify((("other", (Q(1),), Q(1)),), ((Q(0), Q(1)),))
        with self.assertRaises(ValueError):
            polytope.classify((), BOX, deadline=float("inf"))
        with self.assertRaises(TypeError):
            polytope.classify((), BOX, deadline=True)
        self.assertFalse(polytope.certificate_check(CORRELATED, BOX, {}))


if __name__ == "__main__":
    unittest.main()
