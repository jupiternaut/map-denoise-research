"""Independent in-memory branch and tamper checks; no confirmation is run."""
import copy
from collections import deque
from fractions import Fraction as Q
import random
import unittest

from verification import verify_results as v


def node(ident, parent, low, high, status, **extra):
    return {'id': ident, 'parent': parent, 'lo': v.encoded(low), 'hi': v.encoded(high),
            'status': status, **extra}


def complete_outer():
    pending, trace, retained = deque([(0, None, Q(540), Q(660))]), [], []
    next_id = 1
    while pending:
        ident, parent, low, high = pending.popleft()
        if high - low > v.TOLERANCE:
            children = [next_id, next_id + 1]
            next_id += 2
            trace.append(node(ident, parent, low, high, 'split', children=children))
            mid = (low + high) / 2
            pending.extend([(children[0], ident, low, mid), (children[1], ident, mid, high)])
        else:
            trace.append(node(ident, parent, low, high, 'retained_tolerance'))
            retained.append(ident)
    return {'domain': list(map(v.encoded, v.DOMAIN)), 'epsilon': v.encoded(v.EPSILON),
            'tolerance': v.encoded(v.TOLERANCE), 'max_boxes': 511, 'evaluated_boxes': 511,
            'trace': trace, 'retained_ids': retained, 'hull': list(map(v.encoded, v.DOMAIN)),
            'scope': 'outer_set_for_continuous_depth_and_model_nuisance_contract'}


def decision(method, hull, a, tid='fixture'):
    result = v.certificate_action(hull, a, method)
    return {'task_id': tid, 'method': method, **{k: (v.encoded(value) if k in {'output', 'gain_lower'}
            and value is not None else value) for k, value in result.items()}}


def row_and_decision(kind, tid):
    a, truth, hull = Q(539), Q(599), v.DOMAIN
    task = {'task_id': tid, 'observation_id': tid + '-obs', 'incumbent': v.encoded(a)}
    observation = {'kind': kind, 'radius': '0'}
    rows, decisions = [], {}
    for method in v.METHODS:
        if method in v.CERTIFICATES:
            d = decision(method, hull, a, tid)
        elif method == 'keep':
            d = {'task_id': tid, 'method': method, 'status': 'KEEP', 'output': v.encoded(a),
                 'gain_lower': v.encoded(0), 'reason': 'identity'}
        else:
            d = {'task_id': tid, 'method': method, 'status': 'KEEP', 'output': v.encoded(a),
                 'gain_lower': None, 'reason': 'finite_pose_profiled_residual',
                 'incumbent_residual': 0.01, 'minimum_residual': 0.01}
        decisions[(tid, method)] = d
        rows.append(v.expected_row(task, observation, truth, d, v.exact(d['output'])))
    return rows, decisions


def summary_rows(rows):
    return [{'grouping': key[0], 'state_group': key[1], 'method': key[2], 'condition': list(key[3]), **value}
            for key, value in v.recompute_summary(rows).items()]


class BranchTests(unittest.TestCase):
    def test_all_four_branches_and_strict_zero_gain(self):
        cases = (((Q(599), Q(601)), Q(595), 'midpoint', 'MOVE', Q(600)),
                 (v.DOMAIN, Q(539), 'projection', 'MOVE', Q(540)),
                 (v.DOMAIN, Q(600), 'keep', 'KEEP', Q(600)),
                 (None, Q(600), 'incompatible', 'INCOMPATIBLE', Q(600)))
        for hull, a, branch, status, output in cases:
            with self.subTest(branch=branch):
                d = decision('certificate_hybrid', hull, a)
                self.assertEqual((d['branch'], d['status'], v.exact(d['output'])), (branch, status, output))
                self.assertEqual(v.verify_decision(d, a, hull, branch), output)
        # The midpoint proposal 2 has zero gain at endpoint 1, so only
        # the projection 1 is accepted. Equality is not a positive certificate.
        d = v.certificate_action((Q(1), Q(3)), Q(0), 'certificate_hybrid')
        self.assertEqual((d['branch'], d['output'], d['gain_lower']), ('projection', Q(1), Q(1)))

    def test_missing_wrong_branch_and_forged_gain_are_rejected(self):
        original = decision('certificate_hybrid', v.DOMAIN, Q(539))
        for field, value in (('branch', 'midpoint'), ('output', v.encoded(541)), ('gain_lower', v.encoded(121))):
            d = copy.deepcopy(original)
            d[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(v.VerificationError, field):
                v.verify_decision(d, Q(539), v.DOMAIN, 'tampered hybrid')
        del original['branch']
        with self.assertRaisesRegex(v.VerificationError, 'schema'):
            v.verify_decision(original, Q(539), v.DOMAIN, 'missing branch')

    def test_an_accepted_midpoint_cannot_be_replaced_by_projection(self):
        hull, a = (Q(599), Q(601)), Q(595)
        original = decision('certificate_hybrid', hull, a)
        projected = decision('certificate_projection', hull, a)
        tampered = {**projected, 'method': 'certificate_hybrid', 'branch': 'projection'}
        self.assertNotEqual(original['output'], projected['output'])
        with self.assertRaises(v.VerificationError):
            v.verify_decision(tampered, a, hull, 'accepted midpoint replaced')

    def test_per_task_paired_error_and_move_membership(self):
        _, decisions = row_and_decision('rectangle', 'paired')
        v.verify_paired('paired', decisions, Q(599))
        decisions[('paired', 'certificate_hybrid')]['output'] = v.encoded(660)
        with self.assertRaisesRegex(v.VerificationError, 'error exceeds'):
            v.verify_paired('paired', decisions, Q(599))
        _, decisions = row_and_decision('rectangle', 'paired')
        decisions[('paired', 'certificate_hybrid')]['status'] = 'KEEP'
        with self.assertRaisesRegex(v.VerificationError, 'MOVE set'):
            v.verify_paired('paired', decisions, Q(599))

    def test_paired_preservation_checks_output_not_just_error(self):
        hull, a, tid = (Q(599), Q(601)), Q(595), 'accepted'
        decisions = {(tid, m): decision(m, hull, a, tid) for m in v.CERTIFICATES}
        # At truth 599, replacing midpoint 600 by projection 599 improves the
        # error, but violates the explicitly registered preservation contract.
        decisions[(tid, 'certificate_hybrid')]['output'] = v.encoded(599)
        with self.assertRaisesRegex(v.VerificationError, 'midpoint output changed'):
            v.verify_paired(tid, decisions, Q(599))


class TraceAndNoiseTests(unittest.TestCase):
    def test_complete_partition_and_tampered_bisection(self):
        outer = complete_outer()
        checked = v.verify_outer(outer, (Q(1),), Q(600), 'complete')
        self.assertEqual(checked['hull'], v.DOMAIN)
        self.assertEqual(checked['evaluated'], 511)
        outer['trace'][2]['lo'] = v.encoded(601)
        with self.assertRaisesRegex(v.VerificationError, 'midpoint bisection'):
            v.verify_outer(outer, (Q(1),), Q(600), 'tampered partition')

    def test_truth_in_hull_but_excluded_leaf_is_rejected(self):
        outer, truth = complete_outer(), Q(6001, 10)
        leaf = next(n for n in outer['trace'] if n['status'] == 'retained_tolerance' and
                    v.exact(n['lo']) < truth < v.exact(n['hi']))
        leaf.update(status='excluded', witness_channel=0, prediction_interval=list(map(v.encoded, (0, 0))))
        outer['retained_ids'].remove(leaf['id'])
        with self.assertRaisesRegex(v.VerificationError, 'outside every retained leaf'):
            v.verify_outer(outer, (Q(1),), truth, 'hull without coverage')

    def test_exclusion_must_conflict_with_noise_bound(self):
        outer = complete_outer()
        leaf = next(n for n in outer['trace'] if n['status'] == 'retained_tolerance')
        leaf.update(status='excluded', witness_channel=0,
                    prediction_interval=list(map(v.encoded, (Q(99, 100), Q(1)))))
        outer['retained_ids'].remove(leaf['id'])
        with self.assertRaisesRegex(v.VerificationError, 'does not conflict'):
            v.verify_outer(outer, (Q(1),), Q(600), 'forged noise witness')

    def seed_fixture(self):
        truth, radius, seed = v.TRUTHS[0], Q(1, 5), v.SEEDS[0]
        rng = random.Random(f'hybrid-v1:rectangle:{radius}:{truth}:{seed}')
        offsets = (radius * rng.choice(v.WEIGHTS), Q(0), radius * rng.choice(v.WEIGHTS))
        noise = v.EPSILON * rng.choice(v.WEIGHTS)
        observed = {'kind': 'rectangle', 'radius': str(radius), 'observation': [v.encoded(noise)]}
        record = {'truth': v.encoded(truth), 'seed': seed, 'actual_offsets': list(map(v.encoded, offsets)),
                  'noise_linf': v.encoded(abs(noise))}
        return observed, record

    def test_new_seed_noise_and_shared_witness_tampering(self):
        class ZeroModel:
            def point_prediction(self, truth, offsets):
                return (Q(0),)

            def enclosure(self, low, high):
                return ((Q(0), Q(0)),)

        observed, record = self.seed_fixture()
        outer = {'excluded': [], 'boxes': {}}
        self.assertEqual(v.model_spot_checks(ZeroModel(), observed, record, outer, 'new tape', 2), 0)
        observed['observation'][0] = v.encoded(v.exact(observed['observation'][0]) + Q(1, 1000))
        with self.assertRaisesRegex(v.VerificationError, 'hybrid-v1 seed|exceeds bound'):
            v.model_spot_checks(ZeroModel(), observed, record, outer, 'tampered tape', 2)
        observed, record = self.seed_fixture()
        outer = {'excluded': [node(1, 0, 540, 600, 'excluded', witness_channel=0,
                                  prediction_interval=list(map(v.encoded, (1, 1))))],
                 'boxes': {1: (Q(540), Q(600))}}
        with self.assertRaisesRegex(v.VerificationError, 'forged exclusion interval'):
            v.model_spot_checks(ZeroModel(), observed, record, outer, 'forged model', 2)


class EvidenceTests(unittest.TestCase):
    def test_absolute_source_inventory_hash_and_target_tampering(self):
        current = {'/registered/source.py': 'a' * 64}
        lock = {'source_commit': v.SOURCE_COMMIT, 'files': current.copy(),
                'new_experiment_root': str(v.ROOT.resolve()), 'read_only_dependency_root': str(v.DEPENDENCY_ROOT.resolve()),
                'path_format': 'absolute', 'historical_outputs_used': False}
        self.assertEqual(v.verify_source_lock(lock, current), 1)
        lock['files']['/registered/source.py'] = 'b' * 64
        with self.assertRaisesRegex(v.VerificationError, 'source changed'):
            v.verify_source_lock(lock, current)
        lock['files'] = {'relative/source.py': 'a' * 64}
        with self.assertRaisesRegex(v.VerificationError, 'must be absolute'):
            v.verify_source_lock(lock, current)
        lock['files'] = current.copy()
        lock['new_experiment_root'] = str(v.DEPENDENCY_ROOT)
        with self.assertRaisesRegex(v.VerificationError, 'target/dependency'):
            v.verify_source_lock(lock, current)

    def test_row_error_state_and_csv_tampering(self):
        rows, _ = row_and_decision('rectangle', 'csv')
        row = rows[0]
        csv_rows = [{k: str(value) for k, value in r.items()} for r in rows]
        self.assertEqual(v.verify_csv_rows(list(row), csv_rows, rows), 5)
        for field, value in (('ae_mm', '0'), ('actual_gain_mm2', '999'), ('state_group', 'correct')):
            tampered = dict(row, **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(v.VerificationError, field):
                v.verify_row(tampered, row, 'tampered row')
        csv_rows[0]['ae_mm'] = '0'
        with self.assertRaisesRegex(v.VerificationError, 'differ from JSON'):
            v.verify_csv_rows(list(row), csv_rows, rows)

    def test_all_six_summary_families_and_paired_strata_tampering(self):
        first_rows, first_decisions = row_and_decision('rectangle', 'informative')
        second_rows, second_decisions = row_and_decision('layered_hidden', 'hidden')
        rows, decisions = first_rows + second_rows, {**first_decisions, **second_decisions}
        expected = summary_rows(rows)
        checked = v.verify_summary_rows(expected, rows)
        self.assertEqual({key[0] for key in checked},
                         {'overall', 'by_kind', 'by_radius', 'by_offset', 'by_prior', 'by_kind_radius'})
        summary = {'information_strata': {'informative': summary_rows(first_rows), 'hidden': summary_rows(second_rows)},
                   'hybrid_branch_counts': {'projection': 2}, 'paired_relations': v.paired_statistics(rows, decisions)}
        v.verify_extended_summary(summary, rows, decisions)
        summary['paired_relations'][0]['n'] += 1
        with self.assertRaisesRegex(v.VerificationError, 'paired strata'):
            v.verify_extended_summary(summary, rows, decisions)
        expected[0]['n'] += 1
        with self.assertRaisesRegex(v.VerificationError, 'incorrect n'):
            v.verify_summary_rows(expected, rows)

    def test_empty_confirmation_is_not_success(self):
        bundle = {name: [] for name in v.REQUIRED}
        with self.assertRaisesRegex(v.VerificationError, 'expected 135 nonempty'):
            v.verify_bundle(bundle, models={})

    def test_helpers_loaded_from_exact_read_only_dependency(self):
        self.assertEqual(v._old.ROOT, v.DEPENDENCY_ROOT)
        self.assertEqual(v._old.TRUTHS, (Q(3965, 7), Q(4238, 7), Q(4541, 7)))
        self.assertNotEqual(v.TRUTHS, v._old.TRUTHS)
        self.assertNotIn('certificate_hybrid', v._old.METHODS)
        self.assertEqual(v.ROOT.name, 'lovelace-hybrid-20261010T214217Z')


if __name__ == '__main__':
    unittest.main()
