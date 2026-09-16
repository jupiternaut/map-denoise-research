import unittest
from dataclasses import FrozenInstanceError

import learning


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = learning.build_library()

    def test_all_programs_independently_evaluate_within_grammar(self):
        for program in self.library:
            self.assertEqual(program.mask, learning.truth_mask(program.tree))
            self.assertEqual(program.gates, program.tree.gate_count())
            self.assertLessEqual(program.gates, 3)
        self.assertEqual(len(self.library), len({p.mask for p in self.library}))
        self.assertEqual(tuple(sorted(self.library, key=learning.canonical_key)), self.library)

    def test_canonical_minimality_against_independent_two_gate_enumeration(self):
        # Includes redundant subexpressions instead of using minimal semantic levels.
        leaves = learning.build_library(0)
        expected = {p.mask: (p.gates, p.expression) for p in leaves}
        one_gate = []
        for left in leaves:
            for right in leaves:
                a, b = sorted((left, right), key=lambda p: p.expression)
                for op in ("&", "|", "^"):
                    tree = learning.Expr(op, left=a.tree, right=b.tree)
                    mask = learning.truth_mask(tree)
                    expression = f"({a.expression}{op}{b.expression})"
                    one_gate.append(learning.Program(mask, 1, expression, tree))
                    expected[mask] = min(expected.get(mask, (99, "")), (1, expression))
        for left in leaves:
            for right in one_gate:
                a, b = sorted((left, right), key=lambda p: p.expression)
                for op in ("&", "|", "^"):
                    tree = learning.Expr(op, left=a.tree, right=b.tree)
                    mask = learning.truth_mask(tree)
                    expression = f"({a.expression}{op}{b.expression})"
                    expected[mask] = min(expected.get(mask, (99, "")), (2, expression))
        actual = {p.mask: (p.gates, p.expression) for p in learning.build_library(2)}
        self.assertEqual(expected, actual)

    def test_all_affine_functions_are_present(self):
        masks = learning.affine_masks()
        self.assertEqual(len(masks), 32)
        self.assertTrue(masks <= {p.mask for p in self.library})

    def test_known_unseen_named_expression_is_synthesized_from_labels(self):
        # Target is constructed here, independently of library expression names.
        target = sum(((((x >> 0) & 1) & ((x >> 1) & 1)) ^ (((x >> 2) & 1) | ((x >> 3) & 1))) << x for x in range(16))
        samples = tuple((x, (target >> x) & 1) for x in range(16))
        fitted = learning.fit(samples, self.library)
        self.assertTrue(fitted.consistent)
        self.assertEqual(fitted.program.mask, target)
        self.assertEqual(learning.truth_mask(fitted.program.tree), target)

    def test_iid_draws_retain_multiplicity(self):
        inputs = learning.iid_inputs(9100, 32, tuple(range(8)))
        self.assertEqual(len(inputs), 32)
        self.assertLess(len(set(inputs)), 32)
        self.assertEqual(inputs, learning.iid_inputs(9100, 32, tuple(range(8))))
        constants = tuple(p for p in self.library if p.expression in ("0", "1"))
        fitted = learning.fit(((0, 0), (1, 1), (1, 1)), constants)
        self.assertEqual(fitted.program.expression, "1")
        self.assertEqual(fitted.train_errors, 1)

    def test_hidden_label_invariance_and_cheat_detection(self):
        samples = ((0, 0), (1, 1), (4, 1), (6, 0))
        observed = sum(1 << x for x, y in samples)
        target_a = sum(y << x for x, y in samples)
        target_b = target_a ^ (learning.FULL_MASK ^ observed)

        def audit(learner):
            return learner(samples, target_a) == learner(samples, target_b)

        self.assertTrue(audit(lambda observations, hidden: learning.fit(observations, self.library).program.mask))
        self.assertFalse(audit(lambda observations, hidden: hidden))

    def test_loss_matches_naive_weighted_erm_including_conflicting_labels(self):
        samples = ((0, 0), (0, 1), (0, 1), (5, 0), (5, 0), (12, 1))
        fitted = learning.fit(samples, self.library)
        expected = min(self.library, key=lambda p: (sum(p.evaluate(x) != y for x, y in samples), p.gates, p.expression))
        self.assertEqual(fitted.program, expected)
        self.assertEqual(fitted.train_errors, sum(expected.evaluate(x) != y for x, y in samples))
        self.assertFalse(fitted.consistent)

    def test_tie_rule_is_order_invariant(self):
        samples = ((0, 1), (15, 0))
        self.assertEqual(learning.fit(samples, self.library).program, learning.fit(samples, reversed(self.library)).program)

    def test_immutable_program_and_empty_candidates(self):
        with self.assertRaises(FrozenInstanceError):
            self.library[0].mask = 42
        with self.assertRaisesRegex(ValueError, "candidate set is empty"):
            learning.fit((), ())
        self.assertTrue(learning.fit((), self.library).consistent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
