"""Small exact in-memory verifier tests; no confirmation or old writes."""
import copy
from fractions import Fraction as Q
import unittest

import shared_verification as verifier


def proof_fixture():
    rows = (('lower', (Q(1), Q(0), Q(0)), Q(3, 4)),
            ('upper', (Q(1), Q(1), Q(0)), Q(1, 2)))
    proof = {'steps': [{'row': 0, 'variable': 0, 'side': 'lower', 'value': verifier.encoded(Q(3, 4))}],
             'contradiction': {'row': 1}}
    return rows, proof


def trace_fixture():
    # Pixel 0: d0 >= 3/4. Pixel 1: d0 <= 1/2. All individual bands
    # are compatible with [0,1], while their shared material is impossible.
    geometry = ((Q(1), Q(1), Q(0), Q(0), Q(0), Q(0)),) * 2
    image = (Q(6, 25),) * 3 + (Q(51, 100),) * 3
    rows = verifier.necessary_rows(image, geometry)
    proof = {'steps': [{'row': 0, 'variable': 0, 'side': 'upper', 'value': verifier.encoded(Q(77, 100))},
                       {'row': 1, 'variable': 0, 'side': 'lower', 'value': verifier.encoded(Q(3, 4))}],
             'contradiction': {'row': 2}}
    outer = {'domain': list(map(verifier.encoded, (540, 660))), 'epsilon': verifier.encoded(Q(1, 100)),
             'tolerance': verifier.encoded(Q(15, 32)), 'max_boxes': 255, 'evaluated_boxes': 1,
             'trace': [{'id': 0, 'parent': None, 'lo': verifier.encoded(540), 'hi': verifier.encoded(660),
                        'status': 'excluded_shared', 'certificate': proof}], 'retained_ids': [], 'hull': None,
             'shared_tests': 1, 'propagation_rounds': 8, 'scope': verifier.SHARED_SCOPE}
    return geometry, image, rows, outer


class CertificateTests(unittest.TestCase):
    def test_rgb_intersection_and_coefficient_order(self):
        rows = verifier.necessary_rows((Q(1, 5), Q(21, 100), Q(19, 100)),
                                      ((Q(1, 4), Q(1, 2), Q(1, 5), Q(2, 5), Q(0), Q(1, 10)),))
        self.assertEqual(rows, (('upper', (Q(1, 4), Q(1, 5), Q(0)), Q(4, 5)),
                                ('lower', (Q(1, 2), Q(2, 5), Q(1, 10)), Q(4, 5))))

    def test_exact_propagation_proof_and_immediate_contradiction(self):
        rows, proof = proof_fixture()
        self.assertEqual(verifier.verify_certificate(rows, proof, 'valid'), {'rounds': 1, 'updates': 1, 'row': 1})
        self.assertEqual(verifier.verify_certificate((('upper', (Q(0),) * 3, Q(-1, 100)),),
                         {'steps': [], 'contradiction': {'row': 0}}, 'immediate')['updates'], 0)

    def test_forged_bound_omitted_step_and_false_row_fail(self):
        rows, proof = proof_fixture()
        changes = []
        bad = copy.deepcopy(proof)
        bad['steps'][0]['value'] = verifier.encoded(Q(1, 2))
        changes.append(bad)
        changes.append({'steps': [], 'contradiction': {'row': 1}})
        bad = copy.deepcopy(proof)
        bad['contradiction']['row'] = 0
        changes.append(bad)
        for index, bad in enumerate(changes):
            with self.subTest(index=index), self.assertRaises(verifier.VerificationError):
                verifier.verify_certificate(rows, bad, 'tampered proof')

    def test_feasible_rows_do_not_get_vacuous_certificate(self):
        with self.assertRaisesRegex(verifier.VerificationError, 'no genuine row contradiction'):
            verifier.verify_certificate((('upper', (Q(1), Q(0), Q(0)), Q(1)),),
                                        {'steps': [], 'contradiction': {'row': 0}}, 'feasible')

    def test_shared_tree_proof_checked_and_empty_branch_preserved(self):
        geometry, image, _, outer = trace_fixture()
        checked = verifier.verify_shared_trace(outer, image, 'none', lambda *args: geometry, 'valid tree')
        self.assertIsNone(checked['hull'])
        self.assertEqual(checked['proof_updates'], 2)
        bad = copy.deepcopy(outer)
        bad['trace'][0]['certificate']['steps'][0]['value'] = verifier.encoded(Q(1))
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_shared_trace(bad, image, 'none', lambda *args: geometry, 'forged tree')

    def test_partition_gap_and_wrong_work_count_fail(self):
        with self.assertRaises(verifier.VerificationError):
            verifier._arithmetic.verify_partition([(Q(540), Q(599)), (Q(600), Q(660))], 'gap')
        geometry, image, _, outer = trace_fixture()
        outer['shared_tests'] = 0
        with self.assertRaisesRegex(verifier.VerificationError, 'wrong shared test count'):
            verifier.verify_shared_trace(outer, image, 'none', lambda *args: geometry, 'count')


class EvidenceTests(unittest.TestCase):
    def test_regenerated_image_tampering_is_detected(self):
        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def point_prediction(self, *args, **kwargs):
                return (Q(1, 2),) * 432

        public, private = verifier.regenerate(*verifier.CASES[0], verifier.TRUTHS[0], verifier.SEEDS[0], FakeModel)
        self.assertEqual(private['random_tape'], 'shared-appearance-v1:uniform_isolated:1/5:6203/11:61001')
        bad = copy.deepcopy(public)
        bad['observation'][0] = verifier.encoded(Q(0))
        with self.assertRaisesRegex(verifier.VerificationError, 'independent reconstruction'):
            verifier.same(bad, public, 'tampered observation')

    def fixture_rows(self):
        checked = {'hull': (Q(590), Q(610)), 'intervals': [(Q(590), Q(610))],
                   'retained_branches': ['none'], 'evaluated': 5, 'budget_leaves': 0}
        task = {'task_id': 'fixture', 'observation_id': 'o000', 'incumbent': verifier.encoded(Q(603))}
        private = {'truth': verifier.encoded(Q(600)), 'case': 'textured_partial', 'seed': 61001}
        rows = []
        for method, value in (('channelwise_union', Q(603)), ('shared_union', Q(601))):
            decision = {'method': method, 'status': 'KEEP' if value == 603 else 'MOVE', 'output': verifier.encoded(value)}
            rows.append(verifier.make_row(task, private, decision, checked))
        return rows

    def test_actual_error_rows_and_summary_tampering_fail(self):
        rows = self.fixture_rows()
        self.assertEqual((rows[1]['ae_mm'], rows[1]['se_mm2'], rows[1]['actual_gain_mm2']), ('1', '1', '8'))
        bad_rows = copy.deepcopy(rows)
        bad_rows[1]['ae_mm'] = '0'
        with self.assertRaises(verifier.VerificationError):
            verifier.same(bad_rows, rows, 'tampered ROWS')
        summary = verifier.reconstruct_aggregates(rows)
        bad = copy.deepcopy(summary)
        bad[0]['n'] += 1
        with self.assertRaises(verifier.VerificationError):
            verifier.same(bad, summary, 'tampered SUMMARY')
        paired = verifier.paired_relations(rows)
        self.assertTrue(all(row['shared_lower_ae'] == 1 and row['shared_mae_mm'] == '1' and row['channelwise_mae_mm'] == '3' for row in paired))

    def test_empty_population_never_succeeds(self):
        with self.assertRaisesRegex(verifier.VerificationError, '54 nonempty observations'):
            verifier.verify_population([], [], [], [], [])

    def test_complete_group_and_paired_denominators(self):
        checked = {'hull': verifier.DOMAIN, 'intervals': [verifier.DOMAIN], 'retained_branches': ['full'],
                   'evaluated': 255, 'budget_leaves': 256}
        rows, index = [], 0
        for case, _branch, _colors in verifier.CASES:
            for truth in verifier.TRUTHS:
                for seed in verifier.SEEDS:
                    oid = f'o{index:03d}'
                    private = {'truth': verifier.encoded(truth), 'case': case, 'seed': seed}
                    for j, offset in enumerate(verifier.OFFSETS):
                        task = {'task_id': f'{oid}-s{j}', 'observation_id': oid, 'incumbent': verifier.encoded(truth + offset)}
                        for method in verifier.METHODS:
                            rows.append(verifier.make_row(task, private,
                                        {'method': method, 'status': 'KEEP', 'output': task['incumbent']}, checked))
                    index += 1
        self.assertEqual(len(rows), 1944)
        groups = verifier.reconstruct_aggregates(rows)
        self.assertEqual(len(groups), 360)
        informative = next(group for group in groups if group['grouping'] == 'by_information' and
                           group['condition'] == 'informative' and group['state_group'] == 'small' and group['method'] == 'shared_union')
        self.assertEqual((informative['n'], informative['observations'], informative['harm'], informative['same']), (360, 45, 0, 360))
        paired = verifier.paired_relations(rows)
        self.assertEqual(len(paired), 9)
        self.assertEqual(next(row['n'] for row in paired if row['information'] == 'overall' and row['state_group'] == 'small'), 432)
        self.assertEqual(next(row['n'] for row in paired if row['information'] == 'negative_control' and row['state_group'] == 'small'), 72)

    def test_source_lock_rejects_hash_or_inventory_tampering(self):
        lock = {'utc': '2026-10-11T05:23:00+00:00', 'source_commit': 'bd8d05aad65e2c55cd9cd288beae4ef949606984',
                'new_experiment_root': str(verifier.ROOT),
                'read_only_dependency_roots': list(map(str, (verifier.NUISANCE_ROOT, verifier.HYBRID_ROOT, verifier.CHALLENGE_ROOT))),
                'path_format': 'absolute', 'historical_outputs_used': False,
                'files': verifier.scientific_files(),
                'lock_scope': 'scientific implementation/tests/verifier/protocol; post-confirmation reports and analysis excluded'}
        verifier.verify_source_lock(lock)
        bad = copy.deepcopy(lock)
        bad['files'][str(verifier.ROOT / 'shared_verification.py')] = '0' * 64
        with self.assertRaisesRegex(verifier.VerificationError, 'inventory and hashes'):
            verifier.verify_source_lock(bad)
        bad = copy.deepcopy(lock)
        del bad['files'][str(verifier.ROOT / 'test_shared_verification.py')]
        with self.assertRaisesRegex(verifier.VerificationError, 'inventory and hashes'):
            verifier.verify_source_lock(bad)

    def test_duplicate_json_keys_and_noncanonical_bounds_fail(self):
        with self.assertRaises(verifier.VerificationError):
            verifier.unique_object([('n', 1), ('n', 2)])
        with self.assertRaises(verifier.VerificationError):
            verifier.exact({'numerator': 2, 'denominator': 2})


if __name__ == '__main__':
    unittest.main()
