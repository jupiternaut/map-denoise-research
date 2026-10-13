"""Independent counterexamples for observation bridges and proof contracts."""
from collections import deque
from copy import deepcopy
import csv
from fractions import Fraction as Q
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_test_independent_bridge_verifier', ROOT / 'verify_bridge.py')
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)
encode = verify.encoded


def complete_tree(unit):
    """Build a BFS fixture without importing any experimental search code."""
    queue = deque([(0, None, *verify.DOMAIN)])
    trace, retained, next_id = [], [], 1
    while queue:
        ident, parent, low, high = queue.popleft()
        node = {'id': ident, 'parent': parent, 'lo': encode(low), 'hi': encode(high)}
        if high-low <= verify.TOLERANCE:
            node['status'] = 'retained_tolerance'
            retained.append(ident)
        else:
            middle = (low+high)/2
            node.update(status='split', children=[next_id, next_id+1])
            queue.extend(((next_id, ident, low, middle), (next_id+1, ident, middle, high)))
            next_id += 2
        trace.append(node)
    rounds = verify.UNIT_CONFIGS[unit]['rounds']
    return {'domain': list(map(encode, verify.DOMAIN)), 'epsilon': encode(Q(verify.UNIT_CONFIGS[unit]['bound'])),
            'tolerance': encode(verify.TOLERANCE), 'max_boxes': 511, 'evaluated_boxes': 511, 'trace': trace,
            'retained_ids': retained, 'hull': list(map(encode, verify.DOMAIN)), 'scope': verify._old.GROUP_SCOPE,
            'mode': 'global_gray', 'rounds': rounds, 'budget': 511, 'shared_tests': 511, 'propagation_rounds': rounds}


def replay_fixture():
    base_obs = verify.indexed(verify.load(verify.BASE_OUTPUT / 'observations.json'), lambda row: row['observation_id'], 'fixture observations')
    base_tasks = verify.indexed(verify.load(verify.BASE_OUTPUT / 'tasks.json'), lambda row: row['task_id'], 'fixture tasks')
    base_truth = verify.indexed(verify.load(verify.BASE_OUTPUT / 'evaluation/truth.json'), lambda row: row['observation_id'], 'fixture facts')
    observations, tasks, truths = verify.reconstruct_population(base_obs, base_tasks, base_truth)
    om, tm, pm = ({row['observation_id']: row for row in observations}, {row['task_id']: row for row in tasks},
                  {row['observation_id']: row for row in truths})
    rows, coverage = [], []
    checked = {'hull': verify.DOMAIN, 'intervals': [verify.DOMAIN], 'retained_branches': [], 'evaluated': 0, 'budget_leaves': 0}
    for public in observations:
        oid, unit = public['observation_id'], public['unit_id']
        for method in verify.METHODS:
            coverage.append({**{key: public[key] for key in verify.META}, 'reused': public['reused'] and method == 'shared',
                             'method': method, 'covered': True})
            for state in range(9):
                task = tm[f'{oid}-s{state}']
                action = {'task_id': task['task_id'], 'method': method, 'status': 'KEEP', 'output': task['incumbent'],
                          'reused': public['reused'] and method == 'shared'}
                rows.append(verify.make_row(task, pm[oid], action, checked))
    return observations, tasks, truths, rows, coverage, om, tm, base_obs, base_tasks


class ImageBridgeTests(unittest.TestCase):
    def test_raw_out_of_range_remains_raw(self):
        raw = (Q(101, 100),) * 432
        for unit in ('R0epsilon', 'R0delta'):
            self.assertEqual(verify.transformed_image(raw, unit), raw)
        for unit in ('C0epsilon', 'C0delta'):
            self.assertEqual(verify.transformed_image(raw, unit), (Q(1),) * 432)
        biased = verify.transformed_image(raw, 'RBdelta')
        self.assertEqual((biased[0], biased[144], biased[288]), (Q(1), Q(101, 100), Q(51, 50)))
        self.assertGreater(max(biased), 1)

    def test_direct_clipping_differs_from_clipping_before_bias(self):
        raw = [Q(1, 2)] * 432
        raw[0] = Q(201, 200)
        direct = verify.transformed_image(raw, 'CBdelta')
        self.assertEqual(direct[0], Q(199, 200))
        self.assertNotEqual(direct[0], Q(1) - Q(1, 100))
        self.assertEqual(direct, verify.transformed_image(raw, 'CBdelta4'))
        self.assertEqual(direct, verify.transformed_image(raw, 'CBdelta16'))
        self.assertEqual((direct[143], direct[144], direct[287], direct[288]),
                         (Q(49, 100), Q(1, 2), Q(1, 2), Q(51, 100)))

    def test_unknown_unit_or_layout_rejected(self):
        with self.assertRaises(verify.VerificationError):
            verify.transformed_image((Q(1, 2),) * 432, 'CBdelta9')
        with self.assertRaises(verify.VerificationError):
            verify.transformed_image((Q(1, 2),) * 431, 'R0epsilon')

    def test_actual_truth_residual_for_unclipped_diagnostic(self):
        class Prediction:
            def __init__(self, *_args, **_kwargs):
                pass
            def point_prediction(self, *_args, **_kwargs):
                return (Q(1),) * 432
        raw = (Q(101, 100),) * 432
        private = {'rear_branch': 'full', 'truth': encode(Q(600)),
                   'actual_offsets': [encode(Q(0))] * 3, 'colors': [encode(Q(1, 5))] * 3}
        image = verify.transformed_image(raw, 'RBdelta')
        self.assertEqual(verify.validate_residual(raw, image, private, 'RBdelta', Prediction, 'raw bias'), Q(1, 50))
        changed = list(image)
        changed[288] = Q(103, 100)
        with self.assertRaises(verify.VerificationError):
            verify.validate_residual(raw, changed, private, 'RBdelta', Prediction, 'wrong residual')


class ExplicitProofContractTests(unittest.TestCase):
    def test_global_bound_and_round_constants_restore(self):
        before = (verify._old.EPSILON, verify._old.ROUNDS, verify._historical.EPSILON,
                  verify._historical.ROUNDS, verify._old.verify_group_certificate)
        with self.assertRaises(RuntimeError):
            with verify.historical_contract(Q(1, 50), 16):
                self.assertEqual((verify._old.ROUNDS, verify._historical.ROUNDS), (16, 16))
                self.assertEqual((verify._old.EPSILON, verify._historical.EPSILON), (Q(1, 50), Q(1, 50)))
                raise RuntimeError('fixture')
        self.assertEqual((verify._old.EPSILON, verify._old.ROUNDS, verify._historical.EPSILON,
                          verify._historical.ROUNDS, verify._old.verify_group_certificate), before)

    def test_rounds_not_silently_defaulted_to_eight(self):
        # An exact inconsistent pair whose box contractor needs >8 rounds.
        eta = Q(1, 1000000)
        rows = (('upper', (Q(1), Q(1), Q(0)), Q(1)),
                ('lower', (Q(1, 2), Q(1), Q(0)), Q(1) + eta / 2))
        lower, upper, steps, certificate = [Q(0), Q(0)], [Q(1), Q(1)], [], None
        for iteration in range(16):
            for row_id, (side, coefficients, bound) in enumerate(rows):
                values = lower if side == 'upper' else upper
                extreme = sum(coefficient * value for coefficient, value in zip(coefficients, values))
                if (extreme > bound if side == 'upper' else extreme < bound):
                    certificate = {'steps': steps, 'contradiction': {'row': row_id}}
                    break
                for variable, coefficient in enumerate(coefficients[:2]):
                    candidate = (bound - extreme + coefficient * values[variable]) / coefficient
                    target = upper if side == 'upper' else lower
                    if (candidate < target[variable] if side == 'upper' else candidate > target[variable]):
                        target[variable] = candidate
                        steps.append({'row': row_id, 'variable': variable, 'side': side, 'value': encode(candidate)})
            if certificate is not None:
                break
        self.assertIsNotNone(certificate)
        self.assertGreater(iteration + 1, 8)
        with verify.historical_contract(Q(1, 50), 16):
            checked = verify._historical.verify_certificate(rows, certificate, 'sixteen rounds')
            self.assertGreater(checked['rounds'], 8)
        with verify.historical_contract(Q(1, 50), 8), self.assertRaises(verify.VerificationError):
            verify._historical.verify_certificate(rows, certificate, 'eight rounds cannot certify')

    def test_unit_contract_rejects_forged_action_metadata(self):
        task = {'task_id': 'CBdelta4-o000-s1', 'incumbent': encode(Q(601))}
        core = verify._arithmetic.hybrid_action(None, Q(601))
        decision = {'task_id': task['task_id'], 'method': 'shared', 'unit_id': 'CBdelta4', 'bound': '1/50',
                    'rounds': 4, 'reused': False, **{key: encode(value) if isinstance(value, Q) else value for key, value in core.items()}}
        checked = {'hull': None}
        self.assertEqual(verify.verify_decision(decision, task, checked, 'CBdelta4', 'incompatible'), Q(601))
        for field, value in (('rounds', 8), ('bound', '1/100'), ('unit_id', 'CBdelta'), ('reused', True), ('status', 'KEEP')):
            changed = deepcopy(decision)
            changed[field] = value
            with self.assertRaises(verify.VerificationError):
                verify.verify_decision(changed, task, checked, 'CBdelta4', 'forged metadata')

    def test_independent_rows_use_declared_bound(self):
        geometry = ((Q(1), Q(1), Q(0), Q(0), Q(0), Q(0)),) * 144
        image = [Q(1)] * 432
        image[1] = image[2] = Q(0)
        certificate = {'steps': [{'row': 0, 'variable': 0, 'side': 'upper', 'value': encode(Q(1, 50))}],
                       'contradiction': {'row': 1}}
        with verify.historical_contract(Q(1, 50), 4):
            self.assertEqual(verify._old.verify_group_certificate(image, geometry, 'global_gray', 'global', certificate, 'bound')['row'], 1)
            changed = deepcopy(certificate)
            changed['steps'][0]['value'] = encode(Q(1, 100))
            with self.assertRaises(verify.VerificationError):
                verify._old.verify_group_certificate(image, geometry, 'global_gray', 'global', changed, 'old default')


class HistoricalReuseTests(unittest.TestCase):
    def test_reuse_identity_and_exact_source_body(self):
        source = verify.reuse_source('R0epsilon', 'o000')
        self.assertEqual(source['output_root'], str(verify.ABLATION_OUTPUT))
        self.assertEqual(source['method'], 'global_gray511')
        self.assertEqual(verify.reuse_source('CBdelta', 'o000')['observation_id'], 'b1-o000')
        image = verify.load(verify.BASE_OUTPUT / 'observations.json')[0]['observation']
        arm = {'sentinel': 'exact frozen source body', 'outer_seconds': 1.25}
        records = {'R0epsilon': {'o000': {'arms': {'global_gray511': arm}}}}
        self.assertEqual(verify.verify_reused_arm(deepcopy(arm), 'R0epsilon', 'o000', records, image), source)
        changed = deepcopy(arm)
        changed['outer_seconds'] = 0
        with self.assertRaises(verify.VerificationError):
            verify.verify_reused_arm(changed, 'R0epsilon', 'o000', records, image)
        changed_image = list(image)
        changed_image[0] = encode(Q(0))
        with self.assertRaises(verify.VerificationError):
            verify.verify_reused_arm(arm, 'R0epsilon', 'o000', records, changed_image)
        with self.assertRaises(verify.VerificationError):
            verify.reuse_source('CBdelta4', 'o000')

    def test_provenance_hash_task_mapping_and_reused_action_body(self):
        observations, tasks, _truths, _rows, _coverage, om, tm, base_obs, base_tasks = replay_fixture()
        histories, images, histories_tasks, source_actions, times = {}, {}, {}, {}, {}
        outer_records, decisions, expected = {}, {}, []
        for unit in ('R0epsilon', 'CBdelta'):
            histories[unit], images[unit], histories_tasks[unit], source_actions[unit], times[unit] = [], [], [], [], []
        for public in observations:
            oid, unit = public['observation_id'], public['unit_id']
            if not public['reused']:
                outer_records[oid] = {'reused_source': None}
                continue
            source = verify.reuse_source(unit, public['base_observation_id'])
            original_image = (base_obs[source['observation_id']] if unit == 'R0epsilon' else
                              {'observation_id': source['observation_id'], 'base_observation_id': public['base_observation_id'],
                               'beta': '1/100', 'radius': '1/5', 'observation': public['observation']})
            arm = {'sentinel': oid, 'outer_seconds': 1.25}
            histories[unit].append({'observation_id': source['observation_id'], 'arms': {source['method']: arm}})
            images[unit].append(original_image)
            times[unit].append({'observation_id': source['observation_id'], 'outer_seconds_by_method': {source['method']: 1.25},
                                'selector_seconds_by_method': {source['method']: .125}})
            outer_records[oid] = {'reused_source': source, 'arms': {'shared': deepcopy(arm)}}
            mappings = []
            for state in range(9):
                task = tm[f'{oid}-s{state}']
                source_tid = task['base_task_id'] if unit == 'R0epsilon' else 'b1-'+task['base_task_id']
                original_task = {**task, 'task_id': source_tid, 'observation_id': source['observation_id']}
                histories_tasks[unit].append(original_task)
                old_action = {'task_id': source_tid, 'method': source['method'], 'status': 'KEEP', 'output': task['incumbent']}
                source_actions[unit].append(old_action)
                decisions[task['task_id'], 'shared'] = {'task_id': task['task_id'], 'method': 'shared', 'unit_id': unit,
                                                      'bound': public['bound'], 'rounds': 8, 'reused': True,
                                                      'status': 'KEEP', 'output': task['incumbent']}
                mappings.append({'task_id': task['task_id'], 'source_task_id': source_tid, 'action_sha256': verify.json_hash(old_action)})
            expected.append({**{key: public[key] for key in verify.META}, 'source': source,
                             'observation_sha256': verify.json_hash(original_image), 'outer_sha256': verify.json_hash(arm),
                             'source_selector_seconds': .125, 'tasks': mappings})
        output = ROOT / 'author-in-memory-fixture'
        evidence = {output / 'REUSE.json': expected}
        for unit, source_root in (('R0epsilon', verify.ABLATION_OUTPUT), ('CBdelta', verify.BIAS_OUTPUT)):
            evidence.update({source_root / 'outers.json': histories[unit], source_root / 'decisions.json': source_actions[unit],
                             source_root / 'TIMING.json': {'observations': times[unit]}})
        evidence[verify.BIAS_OUTPUT / 'observations.json'] = images['CBdelta']
        evidence[verify.BIAS_OUTPUT / 'tasks.json'] = histories_tasks['CBdelta']
        with patch.object(verify, 'load', side_effect=lambda path: evidence[Path(path)]):
            self.assertEqual(verify.verify_provenance(output, om, tm, outer_records, decisions, base_obs, base_tasks), 108)
            for alteration in ('image_hash', 'outer_hash', 'action_hash', 'task_mapping', 'source'):
                changed = deepcopy(expected)
                if alteration == 'image_hash':
                    changed[0]['observation_sha256'] = '0'*64
                elif alteration == 'outer_hash':
                    changed[0]['outer_sha256'] = '0'*64
                elif alteration == 'action_hash':
                    changed[0]['tasks'][0]['action_sha256'] = '0'*64
                elif alteration == 'task_mapping':
                    changed[0]['tasks'][0]['source_task_id'] = 'o001-s0'
                else:
                    changed[0]['source']['method'] = 'channel511'
                evidence[output / 'REUSE.json'] = changed
                with self.assertRaises(verify.VerificationError):
                    verify.verify_provenance(output, om, tm, outer_records, decisions, base_obs, base_tasks)
            evidence[output / 'REUSE.json'] = expected
            decisions['R0epsilon-o000-s0', 'shared']['status'] = 'INCOMPATIBLE'
            with self.assertRaises(verify.VerificationError):
                verify.verify_provenance(output, om, tm, outer_records, decisions, base_obs, base_tasks)

    def test_all26_frozen_input_hashes_and_false_digest(self):
        expected = {'source_root': str(verify.BASE_OUTPUT), 'files': verify.historical_inputs()}
        self.assertEqual(len(verify.verify_inputs(expected)), 26)
        changed = deepcopy(expected)
        path = str(verify.BASE_OUTPUT / 'evaluation/truth.json')
        changed['files'][path] = '0'*64
        with self.assertRaises(verify.VerificationError):
            verify.verify_inputs(changed)
        changed = deepcopy(expected)
        del changed['files'][str(verify.BIAS_OUTPUT / 'observations.json')]
        with self.assertRaises(verify.VerificationError):
            verify.verify_inputs(changed)


class CompleteProofTests(unittest.TestCase):
    def test_complete511_tree_for_every_registered_round(self):
        class WideKernel:
            def enclosure(self, *_args):
                return ((Q(0), Q(1)),) * 432
        for unit in ('R0epsilon', 'CBdelta4', 'CBdelta16'):
            checked, calls, witnesses = verify.verify_branch(complete_tree(unit), (Q(1, 2),) * 432,
                unit, 'full', lambda *_args: None, WideKernel(), 'complete full tree')
            self.assertEqual((checked['evaluated'], len(checked['retained']), checked['budget_leaves'], calls, witnesses),
                             (511, 256, 0, 511, 0))

    def test_wide_kernel_checker_does_not_default_to_old_epsilon(self):
        class UnitKernel:
            def enclosure(self, *_args):
                return ((Q(1), Q(1)),) * 432
        # Residual .015 is lawful under .02, but is excluded under old .01.
        _, calls, witnesses = verify.verify_branch(complete_tree('RBdelta'), (Q(203, 200),) * 432,
            'RBdelta', 'full', lambda *_args: None, UnitKernel(), 'wide raw witness')
        self.assertEqual((calls, witnesses), (511, 0))

    def test_forged_tree_metadata_counter_partition_or_budget_leaf(self):
        original = complete_tree('CBdelta4')
        for field, value in (('epsilon', encode(Q(1, 100))), ('rounds', 8), ('propagation_rounds', 8),
                             ('budget', 255), ('shared_tests', 510), ('evaluated_boxes', True)):
            changed = deepcopy(original)
            changed[field] = value
            with verify.historical_contract(Q(1, 50), 4), self.assertRaises(verify.VerificationError):
                verify._old.verify_trace(changed, (Q(1, 2),) * 432, 'forged tree', mode='global_gray',
                                         branch='full', geometry_bounds=lambda *_args: None)
        for alteration in ('parent', 'endpoint', 'budget_leaf'):
            changed = deepcopy(original)
            if alteration == 'parent':
                changed['trace'][1]['parent'] = 1
            elif alteration == 'endpoint':
                changed['trace'][1]['hi'] = encode(Q(601))
            else:
                changed['trace'][-1]['status'] = 'unresolved_budget'
            with verify.historical_contract(Q(1, 50), 4), self.assertRaises(verify.VerificationError):
                verify._old.verify_trace(changed, (Q(1, 2),) * 432, 'bad partition', mode='global_gray',
                                         branch='full', geometry_bounds=lambda *_args: None)

    def test_false_shared_certificate_or_group_rejected_in_tree(self):
        outer = complete_tree('CBdelta4')
        image = [Q(1)] * 432
        image[1] = image[2] = Q(0)
        geometry = ((Q(1), Q(1), Q(0), Q(0), Q(0), Q(0)),) * 144
        certificate = {'steps': [{'row': 0, 'variable': 0, 'side': 'upper', 'value': encode(Q(1, 50))}],
                       'contradiction': {'row': 1}}
        outer.update(trace=[{'id': 0, 'parent': None, 'lo': encode(Q(540)), 'hi': encode(Q(660)),
                             'status': 'excluded_shared', 'group_id': 'global', 'certificate': certificate}],
                     evaluated_boxes=1, shared_tests=1, retained_ids=[], hull=None)
        with verify.historical_contract(Q(1, 50), 4):
            checked = verify._old.verify_trace(outer, image, 'valid certificate', mode='global_gray', branch='full',
                                              geometry_bounds=lambda *_args: geometry)
        self.assertEqual(checked['proof_updates'], 1)
        for alteration in ('group', 'row', 'value', 'steps'):
            changed = deepcopy(outer)
            if alteration == 'group':
                changed['trace'][0]['group_id'] = 'v0'
            elif alteration == 'row':
                changed['trace'][0]['certificate']['contradiction']['row'] = 0
            elif alteration == 'value':
                changed['trace'][0]['certificate']['steps'][0]['value'] = encode(Q(1, 100))
            else:
                changed['trace'][0]['certificate']['steps'] = []
            with verify.historical_contract(Q(1, 50), 4), self.assertRaises(verify.VerificationError):
                verify._old.verify_trace(changed, image, 'forged certificate', mode='global_gray', branch='full',
                                         geometry_bounds=lambda *_args: geometry)


class IndependentAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations, cls.tasks, cls.truths, cls.rows, cls.coverage, *_rest = replay_fixture()

    def test_full_population_pairs_aggregates_and_keep_origin(self):
        self.assertEqual((len(self.observations), len(self.tasks), len(self.rows), len(self.coverage)), (432, 3888, 7776, 864))
        pairs, interactions = verify.paired_rows(self.rows)
        self.assertEqual((len(pairs), len(interactions)), (8262, 486))
        self.assertTrue(all(row['same'] and row['mae_gain_mm'] == '0' for row in pairs))
        aggregates = verify.reconstruct_aggregates(self.rows)
        self.assertEqual(len(aggregates), 1440)
        self.assertEqual(len(verify.group_mean_sd(aggregates)), 96)
        self.assertTrue(all(not row['reused'] for row in aggregates if row['method'] == 'keep'))
        claims = verify.safety_claims(self.rows, self.coverage)
        self.assertEqual(claims['contract_safety']['CBdelta']['keep']['unresolved_budget_leaves'], 0)

    def test_missing_task_or_duplicate_arm_rejected(self):
        for bad_rows in (self.rows[:-1], self.rows[:-1] + [self.rows[0]]):
            with self.assertRaises(verify.VerificationError):
                verify.paired_rows(bad_rows)
        with self.assertRaises(verify.VerificationError):
            verify.safety_claims(self.rows[:-1], self.coverage)
        truncated = {row['task_id']: row for row in self.tasks[:-1]}
        with self.assertRaises(verify.VerificationError):
            verify.reconstruct_population({row['base_observation_id']: row for row in self.observations[:54]}, truncated,
                                          {row['base_observation_id']: row for row in self.truths[:54]})

    def test_interaction_sign_and_keep_incompatible_are_distinct(self):
        rows = deepcopy(self.rows)
        tid = rows[1]['base_task_id']
        for row in rows:
            if row['base_task_id'] == tid and row['method'] == 'shared':
                row['ae_mm'] = {'R0delta': '4', 'C0delta': '3', 'RBdelta': '6', 'CBdelta': '2'}.get(row['unit_id'], row['ae_mm'])
        _, interactions = verify.paired_rows(rows)
        result = next(row for row in interactions if row['base_task_id'] == tid)
        self.assertEqual((result['clipping_error_change_no_bias_mm'], result['clipping_error_change_with_bias_mm'],
                          result['difference_in_error_changes_mm']), ('-1', '-4', '-3'))
        rows = deepcopy(self.rows)
        changed = next(row for row in rows if row['unit_id'] == 'R0epsilon' and row['method'] == 'shared' and
                       row['state_group'] == 'small' and row['case'] != 'equal_full_negative')
        changed['status'] = 'INCOMPATIBLE'
        counts = verify.safety_claims(rows, self.coverage)['contract_safety']['R0epsilon']['shared']
        self.assertEqual((counts['kept'], counts['incompatible']), (485, 1))
        changed['state_group'] = 'correct'
        with self.assertRaises(verify.VerificationError):
            verify.safety_claims(rows, self.coverage)

    def test_actual_truth_not_selected_output_and_csv_every_cell(self):
        row = deepcopy(self.rows[1])
        row['ae_mm'] = '0'
        self.assertNotEqual(row['ae_mm'], self.rows[1]['ae_mm'])
        regenerated = replay_fixture()[3][1]
        self.assertEqual(regenerated, self.rows[1])
        records = [{'hull': None, 'retained_intervals': [[encode(Q(540)), encode(Q(660))]], 'covered': True}]
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='verify-bridge-author-') as directory:
            path = Path(directory) / 'coverage.csv'
            with path.open('x', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerow({'hull': None, 'retained_intervals': json.dumps(records[0]['retained_intervals']), 'covered': True})
            verify.verify_csv(path, records, ('hull', 'retained_intervals'))
            forged = deepcopy(records)
            forged[0]['covered'] = False
            with self.assertRaises(verify.VerificationError):
                verify.verify_csv(path, forged, ('hull', 'retained_intervals'))

    def test_evaluation_entry_rejects_counterfeit_scores_or_summary(self):
        output = ROOT / 'author-in-memory-evaluation'
        pairs, interactions = verify.paired_rows(self.rows)
        aggregates = verify.reconstruct_aggregates(self.rows)
        summary = {'observation_configurations': 432, 'task_count': 3888, 'decision_count': 7776,
                   'new_shared_actions': 2916, 'reused_shared_actions': 972, 'keep_actions': 3888,
                   'rows': aggregates, 'group_mean_sd': verify.group_mean_sd(aggregates), 'per_unit_method_timing': [],
                   'elapsed_seconds': 1.0, 'sampling_note': verify.SAMPLING_NOTE, 'timing_scope': verify.TIMING_SCOPE}
        evidence = {output / f'evaluation/{name}.json': value for name, value in
                    (('ROWS', self.rows), ('COVERAGE', self.coverage), ('PAIRED', pairs), ('INTERACTION', interactions), ('SUMMARY', summary))}
        with patch.object(verify, 'load', side_effect=lambda path: evidence[Path(path)]), patch.object(verify, 'verify_csv') as csv_check, \
             patch.object(verify, 'verify_timing') as timing_check:
            result = verify.verify_evaluation(output, self.rows, self.coverage, {}, {'elapsed_seconds': 2.0}, {})
            self.assertEqual(tuple(map(len, result)), (1440, 8262, 486))
            self.assertEqual(csv_check.call_count, 5)
            timing_check.assert_called_once()
            wrong_scores = deepcopy(self.rows)
            wrong_scores[1]['ae_mm'] = '0'
            evidence[output / 'evaluation/ROWS.json'] = wrong_scores
            with self.assertRaises(verify.VerificationError):
                verify.verify_evaluation(output, self.rows, self.coverage, {}, {'elapsed_seconds': 2.0}, {})
            evidence[output / 'evaluation/ROWS.json'] = self.rows
            for field, value in (('keep_actions', 3887), ('group_mean_sd', []), ('rows', aggregates[:-1])):
                evidence[output / 'evaluation/SUMMARY.json'] = {**summary, field: value}
                with self.assertRaises(verify.VerificationError):
                    verify.verify_evaluation(output, self.rows, self.coverage, {}, {'elapsed_seconds': 2.0}, {})


class SourceFenceAndDeadlineTests(unittest.TestCase):
    def test_actual_source_and_private_order_corruptions(self):
        source = (ROOT / 'run_bridge.py').read_text(encoding='utf-8')
        self.assertEqual(verify.audit_runner_source(source)['status'], 'PASS')
        corruptions = [source.replace('    truths = private_truths(observations)\n', '').replace(
                           "    save(output / 'outers.json', outers)\n", "    truths = private_truths(observations)\n    save(output / 'outers.json', outers)\n"),
                       source.replace('len(decisions) != 7776', 'len(decisions) != 7775', 1),
                       source.replace("('shared', True): 972", "('shared', True): 971", 1),
                       source.replace('def execute_shared(observation, radius, epsilon, rounds, max_boxes, tolerance):',
                                      'def execute_shared(observation, radius, epsilon, rounds, max_boxes, tolerance, truth):'),
                       source.replace("load(BASE / 'evaluation/truth.json')", "load(BASE / 'evaluation/fake_truth.json')")]
        for changed in corruptions:
            with self.assertRaises(verify.VerificationError):
                verify.audit_runner_source(changed)

    def test_absolute_budget_and_alarm_restore(self):
        budget = {'started_unix': 100, 'deadline_unix': 200, 'limit_seconds': 5400,
                  'scope': 'main plus independent verification; verification must not reset deadline'}
        with patch.object(verify.time, 'time', return_value=150):
            self.assertEqual(verify.verify_budget(budget), 200)
            for field, value in (('deadline_unix', 5501), ('limit_seconds', 5401), ('scope', 'fresh verification window')):
                changed = {**budget, field: value}
                with self.assertRaises(verify.VerificationError):
                    verify.verify_budget(changed)
        with patch.object(verify.time, 'time', return_value=200), self.assertRaises(TimeoutError):
            verify.verify_budget(budget)
        before = verify.signal.getsignal(verify.signal.SIGALRM)
        with patch.object(verify, 'load', return_value=budget), patch.object(verify.time, 'time', return_value=150):
            with verify.deadline_guard(ROOT):
                self.assertGreater(verify.signal.getitimer(verify.signal.ITIMER_REAL)[0], 0)
        self.assertEqual(verify.signal.getsignal(verify.signal.SIGALRM), before)
        self.assertEqual(verify.signal.getitimer(verify.signal.ITIMER_REAL)[0], 0)

    def test_partial_evidence_failure_and_verification_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='verify-bridge-author-') as directory:
            output = Path(directory)
            with patch.object(verify, 'verify') as full_pipeline, patch('builtins.print'):
                self.assertEqual(verify.main(['--output', str(output)]), 1)
                full_pipeline.assert_not_called()
                report = verify.load(output / 'verification/VERIFY.json')
                self.assertEqual((report['status'], report['mechanical_verification']), ('FAIL', 'FAIL'))
                original = (output / 'verification/VERIFY.json').read_bytes()
                self.assertEqual(verify.main(['--output', str(output)]), 2)
                self.assertEqual((output / 'verification/VERIFY.json').read_bytes(), original)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='verify-bridge-author-') as directory:
            output = Path(directory)
            budget = {'started_unix': 100, 'deadline_unix': 200, 'limit_seconds': 5400,
                      'scope': 'main plus independent verification; verification must not reset deadline'}
            with (output / 'BUDGET.json').open('x', encoding='utf-8') as stream:
                json.dump(budget, stream)
            with patch.object(verify, 'verify') as full_pipeline, patch('builtins.print'):
                self.assertEqual(verify.main(['--output', str(output)]), 1)
                full_pipeline.assert_not_called()
                self.assertEqual(verify.load(output / 'verification/VERIFY.json')['status'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
