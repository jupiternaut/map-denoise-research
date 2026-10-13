"""Hand-built nonconfirmation A7 independent-verifier adversarial fixtures."""
from copy import deepcopy
from fractions import Fraction as Q
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import verify_pose as v


def enc(values):
    return [v.encoded(x) for x in values]


def fixture(branch='partial', darkness=None, ts=None, offsets=None, noise=None):
    depth = Q(640)
    darkness = [Q(1, 3), Q(2, 3), Q(1, 2)] * 3 if darkness is None else list(darkness)
    ts = [Q(1, 100)] * 3 if ts is None else list(ts)
    offsets = [Q(1, 10), Q(0), Q(-1, 10)] if offsets is None else list(offsets)
    noise = [Q(0)] * 432 if noise is None else list(noise)
    image = []
    for pixel, areas in enumerate(v.point_areas(branch, depth, tuple(offsets))):
        view = pixel // 48
        value = 1 - sum((a * d for a, d in zip(areas, darkness[3 * view:3 * view + 3])), Q(0)) + v.BETA - ts[view]
        image.extend(min(Q(1), max(Q(0), value + noise[3 * pixel + c])) for c in range(3))
    public = {'observation_id': 'fixture', 'clipped': True, 'baseline': v.encoded(80), 'radius': v.encoded(v.RADIUS),
              'observation': enc(image)}
    arm = {'appearance_model': 'bounded', 'drift_bound': v.encoded(v.DRIFT),
           'calibration_intervals': [enc((0, Q(1, 50))) for _ in range(3)]}
    witness = {'offsets': enc(offsets), 'darkness': enc(darkness), 't': enc(ts), 'noise': enc(noise)}
    return public, arm, depth, branch, witness


def necessary_classification(public, arm, depth, branch, witness=None):
    rows, bounds = v._point.point_problem(public, arm, 'joint', branch, depth)
    normalized, box = v._proof.normalized_problem(rows, bounds)
    singles = v._proof.single_row_contradictions(rows, bounds)
    certificate = None
    if witness is None:
        if not singles:
            raise AssertionError('fixture requires a single-row exact contradiction')
        row = singles[0]
        coefficients, bound = normalized[row]
        minimum = v._proof.minimum_over_box(coefficients, box)
        certificate = {'sparse_weights': [{'row': row, 'weight': v.encoded(1)}],
                       'aggregate_coefficients': enc(coefficients), 'weighted_bound': v.encoded(bound),
                       'box_min': v.encoded(minimum), 'margin': v.encoded(minimum - bound)}
    return {'status': 'FEASIBLE_WITNESS' if witness is not None else 'CERTIFIED_INFEASIBLE', 'certificate': certificate,
            'witness': witness, 'lp_status': None, 'seconds': 0.0, 'lp_seconds': 0.0,
            'single_row_contradiction': bool(singles), 'single_row_contradiction_rows': singles,
            'row_count': len(rows), 'dimension': 12, 'reason': 'hand_fixture'}


def unknown_search():
    return {'status': 'UNKNOWN', 'witness': None, 'center_status': 'UNKNOWN', 'center_witness': None,
            'numeric_evaluations': 0, 'classifier_lp_calls': 0, 'numeric_lp_calls': 0, 'exact_attempts': 0,
            'numeric_failures': 0, 'powell_starts': 0, 'powell_evaluations': [], 'best_phase_slack': None,
            'seconds': 0.0, 'termination': 'whole_deadline', 'search_budget_seconds': 0.0, 'limits': dict(v.LIMITS)}


class ForwardTests(unittest.TestCase):
    def test_complete_physical_witness_each_branch(self):
        for branch in v.BRANCHES:
            args = fixture(branch)
            self.assertTrue(v.verify_forward(*args))

    def test_pose_radius_and_gauge_tampering(self):
        args = fixture()
        for index, value in ((0, Q(201, 1000)), (1, Q(1, 1000))):
            bad = deepcopy(args[-1]); bad['offsets'][index] = v.encoded(value)
            with self.assertRaises(v.VerificationError):
                v.verify_forward(*args[:-1], bad)

    def test_wrong_inbounds_pose_rejects_pixel_mismatch(self):
        args = fixture()
        bad = deepcopy(args[-1]); bad['offsets'][0] = v.encoded(-v.RADIUS)
        with self.assertRaises(v.VerificationError):
            v.verify_forward(*args[:-1], bad)

    def test_noise_shape_limit_and_tampering(self):
        args = fixture()
        for kind in ('missing', 'limit', 'changed'):
            bad = deepcopy(args[-1])
            if kind == 'missing': bad['noise'].pop()
            else: bad['noise'][0] = v.encoded(Q(11, 1000) if kind == 'limit' else Q(-1, 1000))
            with self.assertRaises(v.VerificationError):
                v.verify_forward(*args[:-1], bad)

    def test_darkness_and_pairwise_drift(self):
        args = fixture()
        for value in (Q(-1, 10), Q(1)):
            bad = deepcopy(args[-1]); bad['darkness'][0] = v.encoded(value)
            with self.assertRaises(v.VerificationError):
                v.verify_forward(*args[:-1], bad)

    def test_calibration_and_bias_sign(self):
        args = fixture(ts=[Q(0), Q(1, 100), Q(1, 50)])
        bad = deepcopy(args[-1]); bad['t'] = enc([Q(1, 50), Q(1, 100), Q(0)])
        with self.assertRaises(v.VerificationError):
            v.verify_forward(*args[:-1], bad)
        bad = deepcopy(args[-1]); bad['t'][0] = v.encoded(Q(21, 1000))
        with self.assertRaises(v.VerificationError):
            v.verify_forward(*args[:-1], bad)

    def test_clip_before_noise_false_positive(self):
        # Gray1+bias.01+noise(-.01) clips to1. Clipping before noise gives.99.
        args = fixture(darkness=[Q(0)] * 9, ts=[Q(0)] * 3, noise=[-v.EPSILON] * 432)
        self.assertTrue(v.verify_forward(*args))
        bad_public = deepcopy(args[0]); bad_public['observation'] = enc([Q(99, 100)] * 432)
        with self.assertRaises(v.VerificationError):
            v.verify_forward(bad_public, *args[1:])

    def test_strict_encoded_rationals_and_candidate_cap(self):
        args = fixture(darkness=[Q(0)] * 9)
        bad = deepcopy(args[-1]); bad['noise'][0] = 0.0
        with self.assertRaises(v.VerificationError):
            v.verify_forward(*args[:-1], bad)
        physical = deepcopy(args[-1]); physical['offsets'][0] = v.encoded(Q(1, 1000000001))
        self.assertTrue(v.verify_forward(*args[:-1], physical))
        with self.assertRaises(v.VerificationError):
            v.verify_forward(*args[:-1], physical, candidate=True)


class SeedAndBranchTests(unittest.TestCase):
    def test_exact_seed_partition_lift_and_closed_cells(self):
        public, arm, depth, branch, witness = fixture(darkness=[Q(0)] * 9)
        seed = witness['darkness'] + witness['t']
        record = v.seed_intervals(public, arm, depth, branch, seed)
        self.assertEqual(record['status'], 'SEED_LIFTED')
        self.assertEqual(record['views'][1]['breakpoints'], enc([0]))
        for view in record['views']:
            for cell in view['cells']:
                self.assertEqual(cell['cell'], cell['feasible'])
        self.assertIsNotNone(v.verify_seed(public, arm, depth, branch, seed, record))

    def test_seed_wrong_interval_and_changed_fixed_bias(self):
        public, arm, depth, branch, witness = fixture(darkness=[Q(0)] * 9)
        seed = witness['darkness'] + witness['t']
        record = v.seed_intervals(public, arm, depth, branch, seed)
        bad = deepcopy(record); bad['views'][0]['cells'][0]['feasible'] = None
        with self.assertRaises(v.VerificationError):
            v.verify_seed(public, arm, depth, branch, seed, bad)
        bad = deepcopy(record); bad['views'][0]['breakpoints'][0] = v.encoded(-Q(1, 10))
        with self.assertRaises(v.VerificationError):
            v.verify_seed(public, arm, depth, branch, seed, bad)

    def test_fixed_seed_negative_does_not_certify_depth(self):
        public, arm, depth, branch, witness = fixture(darkness=[Q(0)] * 9)
        public['observation'][144] = v.encoded(Q(1, 2))
        seed = witness['darkness'] + witness['t']
        record = v.seed_intervals(public, arm, depth, branch, seed)
        self.assertEqual(record['status'], 'SEED_NOT_LIFTABLE')
        self.assertIsNone(v.verify_seed(public, arm, depth, branch, seed, record))
        self.assertEqual(v.aggregate(['UNKNOWN', 'INHERITED_INFEASIBLE', 'UNKNOWN']), 'UNKNOWN')

    def test_branch_chosen_seed_is_independently_checked(self):
        public, arm, depth, branch, witness = fixture(darkness=[Q(0)] * 9)
        coords = witness['darkness'] + witness['t']
        source = {'branch': branch, 'point_classification': necessary_classification(public, arm, depth, branch, coords)}
        seed = v.seed_intervals(public, arm, depth, branch, coords)
        chosen = v.seed_forward(public, depth, branch, coords, seed['offsets'])
        record = {'branch': branch, 'source_status': 'FEASIBLE_WITNESS', 'status': 'FORWARD_WITNESS',
                  'seed': seed, 'search': unknown_search(), 'witness': chosen}
        self.assertEqual(v.verify_branch(record, source, public, arm, {'incumbent': v.encoded(depth)})['forward'], 1)
        bad = deepcopy(record); bad['witness']['noise'][0] = v.encoded(-Q(1, 100))
        with self.assertRaises(v.VerificationError):
            v.verify_branch(bad, source, public, arm, {'incumbent': v.encoded(depth)})

    def test_inherited_certificate_checked_and_no_new_negative(self):
        public, arm, depth, branch, witness = fixture('none')
        public['observation'] = enc([0] * 432)
        source = {'branch': branch, 'point_classification': necessary_classification(public, arm, depth, branch)}
        record = {'branch': branch, 'source_status': 'CERTIFIED_INFEASIBLE', 'status': 'INHERITED_INFEASIBLE',
                  'seed': None, 'search': None, 'witness': None}
        self.assertEqual(v.verify_branch(record, source, public, arm, {'incumbent': v.encoded(depth)})['inherited'], 1)
        bad_source = deepcopy(source)
        bad_source['point_classification']['certificate']['margin'] = v.encoded(0)
        with self.assertRaises(v.VerificationError):
            v.verify_branch(record, bad_source, public, arm, {'incumbent': v.encoded(depth)})
        bad = deepcopy(record); bad['status'] = 'CERTIFIED_INFEASIBLE'
        with self.assertRaises(v.VerificationError):
            v.verify_branch(bad, source, public, arm, {'incumbent': v.encoded(depth)})


class AccountingAndGateTests(unittest.TestCase):
    def test_real_solver_fixture_search_schema(self):
        # Integration is only in this synthetic test; verifier module never
        # imports the new numerical solver or uses it to check a proof.
        import pose_solver
        public, arm, depth, branch, _ = fixture(darkness=[Q(0)] * 9, offsets=[Q(0)] * 3)
        record = pose_solver.complete_branch(public, arm, depth, branch)
        v.verify_search(public, arm, depth, branch, record)
        self.assertEqual(record['status'], 'FORWARD_WITNESS')

    def test_search_caps_and_false_positive_center(self):
        args = fixture()
        public, arm, depth, branch = args[:-1]
        good = unknown_search()
        v.verify_search(public, arm, depth, branch, good)
        for field, value in (('numeric_lp_calls', 201), ('exact_attempts', 21), ('powell_starts', 3),
                             ('search_budget_seconds', 3.1), ('seconds', float('nan'))):
            bad = deepcopy(good); bad[field] = value
            with self.assertRaises(v.VerificationError):
                v.verify_search(public, arm, depth, branch, bad)
        bad = deepcopy(good); bad.update(powell_starts=1, powell_evaluations=[61], numeric_evaluations=61, numeric_lp_calls=61)
        with self.assertRaises(v.VerificationError):
            v.verify_search(public, arm, depth, branch, bad)
        bad = deepcopy(good); bad.update(status='INHERITED_INFEASIBLE')
        with self.assertRaises(v.VerificationError):
            v.verify_search(public, arm, depth, branch, bad)

    def test_population_missing_duplicate_and_union(self):
        observations = {f'o{i}': {} for i in range(36)}
        tasks = {f'o{i}s{s}': {'task_id': f'o{i}s{s}', 'observation_id': f'o{i}', 'incumbent': v.encoded(640)}
                 for i in range(36) for s in range(9)}
        records = [{'task_id': t['task_id'], 'observation_id': t['observation_id'], 'incumbent': t['incumbent'],
                    'a6_status': 'FEASIBLE_WITNESS', 'status': 'UNKNOWN',
                    'branches': [{'branch': b, 'status': 'UNKNOWN'} for b in v.BRANCHES]} for t in tasks.values()]
        v.verify_record_population(records, observations, tasks)
        for kind in ('missing', 'duplicate', 'branch', 'union'):
            bad = deepcopy(records)
            if kind == 'missing': bad.pop()
            elif kind == 'duplicate': bad[-1] = bad[0]
            elif kind == 'branch': bad[0]['branches'].pop()
            else: bad[0]['status'] = 'FORWARD_WITNESS'
            with self.assertRaises(v.VerificationError):
                v.verify_record_population(bad, observations, tasks)

    def test_source_hash_mismatch(self):
        output = Path('/fixture-output')
        sources, inputs = {'/registered/source.py': 'actual'}, {'/registered/input.json': 'input'}
        def mocked_load(path):
            if Path(path).name == 'SOURCE_LOCK.json':
                return {'source_root': str(v.ROOT), 'files': {'/registered/source.py': 'tampered'}}
            return {'files': inputs}
        with patch.object(v, 'scientific_files', return_value=sources), patch.object(v, 'input_files', return_value=inputs), patch.object(v, 'load', side_effect=mocked_load):
            with self.assertRaises(v.VerificationError):
                v.verify_locks(output)

    def test_saved_gate_and_shared_executor_source_audits(self):
        self.assertEqual(v.audit_runner_source()['status'], 'PASS')
        self.assertEqual(v.audit_executor_source()['status'], 'PASS')
        source = (v.ROOT / 'run_pose.py').read_text()
        with self.assertRaises(v.VerificationError):
            v.audit_runner_source(source.replace('check_records(records,ts)', 'pass'))
        with self.assertRaises(v.VerificationError):
            v.audit_runner_source(source.replace('!=972', '!=971'))
        executor = (v.ROOT / 'execute_pose.py').read_text()
        with self.assertRaises(v.VerificationError):
            v.audit_executor_source(executor.replace('remaining=deadline-time.time()', 'remaining=3600'))

    def test_verifier_imports_no_new_solver_lift_runner(self):
        tree = ast.parse(Path(v.__file__).read_text())
        imports = [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
        imports += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(name in ('pose_solver', 'pose_lift', 'run_pose') for name in imports))
        self.assertFalse(any(isinstance(n, ast.Constant) and n.value in ('pose_solver.py', 'pose_lift.py', 'run_pose.py')
                             for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'spec_from_file_location'))

    def test_budget_scope_and_no_reset(self):
        with patch.object(v.time, 'time', return_value=100):
            budget = {'start_epoch': 0.0, 'deadline_epoch': 3600.0, 'limit_seconds': 3600, 'scope': v.BUDGET_SCOPE}
            self.assertEqual(v.verify_budget(budget), 3600)
            bad = deepcopy(budget); bad['deadline_epoch'] = 3601.0
            with self.assertRaises(v.VerificationError):
                v.verify_budget(bad)


if __name__ == '__main__':
    unittest.main()
