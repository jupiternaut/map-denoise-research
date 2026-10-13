"""Small in-memory checks: no solver run, GPU, network or output-directory writes."""
import copy
from collections import deque
from fractions import Fraction as Q
import random
import unittest

from verification import verify_results as verifier


def node(ident, parent, low, high, status, **extra):
    return {"id": ident, "parent": parent, "lo": verifier.encoded(low), "hi": verifier.encoded(high), "status": status, **extra}


def simple_outer():
    # The right leaf is excluded by its [0,0] image-channel interval against 1.
    return {"domain": list(map(verifier.encoded, (540, 660))), "epsilon": verifier.encoded(Q(1, 100)),
            "tolerance": verifier.encoded(Q(15, 32)), "max_boxes": 511, "evaluated_boxes": 3,
            "trace": [node(0, None, 540, 660, "split", children=[1, 2]),
                      node(1, 0, 540, 600, "excluded", witness_channel=0,
                           prediction_interval=list(map(verifier.encoded, (0, 0)))),
                      node(2, 0, 600, 660, "excluded", witness_channel=0,
                           prediction_interval=list(map(verifier.encoded, (0, 0))))],
            "retained_ids": [], "hull": None,
            "scope": "outer_set_for_continuous_depth_and_model_nuisance_contract"}


def complete_outer():
    """A full 511-node, 256-leaf, tolerance-limited closed partition."""
    outer = simple_outer()
    pending = deque([(0, None, Q(540), Q(660))])
    trace, retained, next_id = [], [], 1
    while pending:
        ident, parent, low, high = pending.popleft()
        if high - low > Q(15, 32):
            children = [next_id, next_id + 1]
            next_id += 2
            trace.append(node(ident, parent, low, high, "split", children=children))
            middle = (low + high) / 2
            pending.extend([(children[0], ident, low, middle), (children[1], ident, middle, high)])
        else:
            trace.append(node(ident, parent, low, high, "retained_tolerance"))
            retained.append(ident)
    outer.update(trace=trace, retained_ids=retained, evaluated_boxes=511,
                 hull=list(map(verifier.encoded, (540, 660))))
    return outer


class PartitionTests(unittest.TestCase):
    def test_complete_trace_covers_truth_and_shared_boundaries(self):
        result = verifier.verify_outer(complete_outer(), (Q(1),), Q(600), "complete")
        self.assertEqual(result["hull"], (Q(540), Q(660)))
        self.assertEqual(result["evaluated"], 511)

    def test_truth_in_hull_but_excluded_leaf_is_rejected(self):
        outer = complete_outer()
        truth = Q(6001, 10)
        leaf = next(n for n in outer["trace"] if n["status"] == "retained_tolerance" and
                    verifier.exact(n["lo"]) < truth < verifier.exact(n["hi"]))
        leaf.update(status="excluded", witness_channel=0,
                    prediction_interval=list(map(verifier.encoded, (0, 0))))
        outer["retained_ids"].remove(leaf["id"])
        with self.assertRaisesRegex(verifier.VerificationError, "outside every retained leaf"):
            verifier.verify_outer(outer, (Q(1),), truth, "truth in hull only")

    def test_shared_closed_boundaries_are_a_complete_partition(self):
        verifier.verify_partition([(Q(600), Q(660)), (Q(540), Q(570)), (Q(570), Q(600))])

    def test_gap_overlap_empty_and_missing_domain_end_fail(self):
        for intervals in ([], [(Q(540), Q(599)), (Q(600), Q(660))],
                          [(Q(540), Q(601)), (Q(600), Q(660))], [(Q(541), Q(660))]):
            with self.subTest(intervals=intervals), self.assertRaises(verifier.VerificationError):
                verifier.verify_partition(intervals)

    def test_false_exclusion_witness_fails_before_truth_coverage(self):
        outer = simple_outer()
        outer["trace"][1]["prediction_interval"] = list(map(verifier.encoded, (Q(99, 100), Q(101, 100))))
        with self.assertRaisesRegex(verifier.VerificationError, "radiance range"):
            verifier.verify_outer(outer, (Q(1),), Q(610), "fixture")
        outer["trace"][1]["prediction_interval"] = list(map(verifier.encoded, (Q(99, 100), Q(1))))
        with self.assertRaisesRegex(verifier.VerificationError, "does not conflict"):
            verifier.verify_outer(outer, (Q(1),), Q(610), "fixture")

    def test_child_gap_is_rejected(self):
        outer = simple_outer()
        outer["trace"][2]["lo"] = verifier.encoded(601)
        with self.assertRaisesRegex(verifier.VerificationError, "midpoint bisection"):
            verifier.verify_outer(outer, (Q(1),), Q(610), "fixture")

    def test_empty_outer_cannot_pass_registered_truth(self):
        with self.assertRaisesRegex(verifier.VerificationError, "empty certified outer"):
            verifier.verify_outer(simple_outer(), (Q(1),), Q(610), "fixture")


class ArithmeticTests(unittest.TestCase):
    def test_shared_model_sample_catches_forged_trace_interval(self):
        class FakeModel:
            def point_prediction(self, truth, offsets):
                return (Q(0),)

            def enclosure(self, low, high):
                return ((Q(0), Q(0)),)

        rng = random.Random("challenge-v1:rectangle:0:600:31001")
        rng.choice(verifier.WEIGHTS)
        rng.choice(verifier.WEIGHTS)
        noise = Q(1, 100) * rng.choice(verifier.WEIGHTS)
        observation = {"kind": "rectangle", "radius": "0", "observation": [verifier.encoded(noise)]}
        truth = {"truth": verifier.encoded(600), "seed": 31001,
                 "actual_offsets": list(map(verifier.encoded, (0, 0, 0))),
                 "noise_linf": verifier.encoded(abs(noise))}
        forged = node(1, 0, 540, 600, "excluded", witness_channel=0,
                      prediction_interval=list(map(verifier.encoded, (1, 1))))
        with self.assertRaisesRegex(verifier.VerificationError, "forged exclusion interval"):
            verifier.model_spot_checks(FakeModel(), observation, truth,
                                      {"excluded": [forged], "boxes": {1: (Q(540), Q(600))}}, "sample", 2)

    def fixture_row(self):
        # Projection moves 539 -> 540; truth 600, so error 61 -> 60, gain 121.
        task = {"task_id": "fixture", "observation_id": "o000", "incumbent": verifier.encoded(539)}
        observation = {"kind": "rectangle", "radius": "0"}
        decision = {"task_id": "fixture", "method": "certificate_projection", "status": "MOVE",
                    "output": verifier.encoded(540), "gain_lower": verifier.encoded(1),
                    "reason": "strict_gain_over_continuous_outer_hull"}
        expected = verifier.expected_row(task, observation, Q(600), decision, Q(540))
        return task, decision, expected

    def test_tampered_error_and_state_rows_fail(self):
        _, _, expected = self.fixture_row()
        verifier.verify_row(expected, expected, "fixture")
        self.assertEqual(expected["ae_mm"], "60")
        self.assertEqual(expected["actual_gain_mm2"], "121")
        for field, value in (("ae_mm", "0"), ("actual_gain_mm2", "999"),
                             ("prior_position", "inside_prior"), ("state_group", "correct")):
            row = copy.deepcopy(expected)
            row[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(verifier.VerificationError, field):
                verifier.verify_row(row, expected, "tampered ROWS")

    def test_endpoint_strict_gain_clamp_and_midpoint_keep(self):
        task, decision, _ = self.fixture_row()
        verifier.verify_decision(decision, Q(539), (Q(540), Q(660)), "projection")
        projection = verifier.certificate_action((Q(540), Q(660)), Q(539), "certificate_projection")
        midpoint = verifier.certificate_action((Q(540), Q(660)), Q(539), "certificate_midpoint")
        self.assertEqual((projection["output"], projection["gain_lower"]), (Q(540), Q(1)))
        self.assertEqual((midpoint["status"], midpoint["output"]), ("KEEP", Q(539)))
        for a in (Q(540), Q(600), Q(660)):
            self.assertEqual(verifier.certificate_action((Q(540), Q(660)), a, "certificate_projection")["status"], "KEEP")
        # A zero endpoint gain is insufficient: a=0, hull=[1,3], midpoint=2.
        self.assertEqual(verifier.certificate_action((Q(1), Q(3)), Q(0), "certificate_midpoint")["status"], "KEEP")
        tampered = copy.deepcopy(decision)
        tampered["gain_lower"] = verifier.encoded(121)
        with self.assertRaisesRegex(verifier.VerificationError, "gain_lower"):
            verifier.verify_decision(tampered, Q(539), (Q(540), Q(660)), "tampered certificate")

    def test_empty_certificate_is_incompatible_not_positive_gain(self):
        result = verifier.certificate_action(None, Q(600), "certificate_projection")
        self.assertEqual(result["status"], "INCOMPATIBLE")
        self.assertIsNone(result["gain_lower"])

    def test_tampered_aggregate_count_fails(self):
        _, _, row = self.fixture_row()
        expected = verifier.recompute_summary([row])
        summary = [{"grouping": key[0], "state_group": key[1], "method": key[2], "condition": list(key[3]), **value}
                   for key, value in expected.items()]
        verifier.verify_summary_rows(summary, [row])
        summary[0]["n"] = 2
        with self.assertRaisesRegex(verifier.VerificationError, "incorrect n"):
            verifier.verify_summary_rows(summary, [row])

    def test_empty_confirmation_is_not_success(self):
        empty = {name: ([] if name != "summary" else {}) for name in verifier.REQUIRED}
        with self.assertRaisesRegex(verifier.VerificationError, "expected 135 nonempty"):
            verifier.verify_bundle(empty, models={})

    def test_exact_inputs_reject_float_bool_and_noncanonical_encoding(self):
        for value in (0.1, True, {"numerator": 2, "denominator": 2}, {"numerator": 1, "denominator": -2}):
            with self.subTest(value=value), self.assertRaises(verifier.VerificationError):
                verifier.exact(value)


if __name__ == "__main__":
    unittest.main()
