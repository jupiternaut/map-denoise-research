"""Author tests; synthetic fixtures and read-only source validation, no full run."""
import copy
from fractions import Fraction as Q
import inspect
import io
from pathlib import Path
import signal
import time
import unittest
from unittest.mock import patch

import run_bridge as run


def base_fixture(count=54):
    observations, tasks, truths = [], [], []
    cases = ('case_a', 'case_b', 'case_c', 'case_d', 'case_e', 'equal_full_negative')
    offsets = (Q(0), Q(1, 4), Q(-1, 4), Q(1, 2), Q(-1, 2), Q(1), Q(-1), Q(3), Q(-3))
    for index in range(count):
        oid, truth = f'o{index:03}', Q(570+30*((index//3) % 3))
        observations.append({'observation_id': oid, 'radius': '1/5',
            'observation': [run.encode(Q(1, 2)) for _ in range(432)], 'calibration': 'must not leak'})
        truths.append({'observation_id': oid, 'case': cases[index//9], 'seed': 61001+index%3,
                       'truth': run.encode(truth)})
        for state, offset in enumerate(offsets):
            tasks.append({'task_id': f'{oid}-s{state}', 'observation_id': oid,
                          'incumbent': run.encode(truth+offset), 'truth': 'must not leak'})
    return observations, tasks, truths


def shared_fixture(bound=Q(1, 100), rounds=8):
    entry = run.helpers.identity_arm()
    entry['outer'].update(epsilon=run.encode(bound), mode='global_gray', rounds=rounds, budget=511)
    entry['branches'] = {branch: {'epsilon': run.encode(bound), 'tolerance': run.encode(run.TOLERANCE),
        'mode': 'global_gray', 'rounds': rounds, 'budget': 511, 'max_boxes': 511,
        'propagation_rounds': rounds, 'trace': [], 'retained_ids': []} for branch in run.old.BRANCHES}
    entry['branch_seconds'] = {branch: 0.0 for branch in run.old.BRANCHES}
    return entry


def memory_reuse(observations, tasks):
    result = {}
    for item in observations:
        if not item['reused']:
            continue
        entry = shared_fixture(Q(item['bound']), item['rounds'])
        actions = {task['task_id']: run.old.select_action(entry['outer'], run.decode(task['incumbent']))
                   for task in tasks if task['observation_id'] == item['observation_id']}
        source, method = run.REUSE_SOURCES[item['unit_id']]
        result[item['observation_id']] = {'entry': entry, 'actions': actions,
            'source': {'output_root': str(source), 'observation_id': item['base_observation_id'], 'method': method}}
    return result


class PublicInterfaceTests(unittest.TestCase):
    def test_layout_and_bias_for_all_rgb_channels(self):
        source = [run.encode(Q(1, 2)) for _ in range(432)]
        image = list(map(run.decode, run.transform_image(source, 'biased_raw')))
        self.assertEqual(image, [Q(49, 100)]*144+[Q(1, 2)]*144+[Q(51, 100)]*144)
        for index in (142, 143, 144, 145, 286, 287, 288, 289):
            self.assertEqual(image[index]-Q(1, 2), (-run.BETA, Q(0), run.BETA)[index//144])
        for size in (0, 431, 433):
            with self.assertRaises(ValueError):
                run.transform_image(source[:size] if size < 432 else source+[source[0]], 'raw')

    def test_clip_after_bias_not_preclip(self):
        source = [run.encode(Q(1, 2)) for _ in range(432)]
        source[0], source[288] = run.encode(Q(101, 100)), run.encode(Q(-1, 100))
        raw = run.transform_image(source, 'raw')
        self.assertEqual(raw, source)
        correct = run.transform_image(source, 'biased_clip')
        wrong = run.transform_image(run.transform_image(source, 'clip'), 'biased_clip')
        self.assertEqual(run.decode(correct[0]), Q(1))
        self.assertEqual(run.decode(wrong[0]), Q(99, 100))
        self.assertEqual(run.decode(correct[288]), Q(0))
        self.assertEqual(run.decode(wrong[288]), Q(1, 100))
        unclipped = run.transform_image([run.encode(Q(101, 100))]*432, 'biased_raw')
        self.assertEqual(run.decode(unclipped[-1]), Q(51, 50))

    def test_public_only_population_and_frozen_matrix(self):
        base, base_tasks, _ = base_fixture()
        observations, tasks = run.generate_public(base, base_tasks)
        self.assertEqual((len(observations), len(tasks), len(tasks)*len(run.METHODS)), (432, 3888, 7776))
        self.assertEqual(sum(not item['reused'] for item in observations), 324)
        self.assertEqual(sum(item['reused'] for item in observations), 108)
        self.assertEqual(len({item['observation_id'] for item in observations}), 432)
        self.assertEqual(len({item['task_id'] for item in tasks}), 3888)
        self.assertEqual(set(observations[0]), set(run.META) | {'radius', 'observation'})
        self.assertEqual(set(tasks[0]), {'task_id', 'observation_id', 'base_task_id', 'incumbent'})
        self.assertEqual(observations[0]['observation_id'], 'R0epsilon-o000')
        self.assertEqual(tasks[0]['task_id'], 'R0epsilon-o000-s0')
        self.assertEqual(run.UNIT['CBdelta16']['rounds'], 16)
        with self.assertRaises(ValueError):
            run.generate_public(base, base_tasks[:-1])

    def test_explicit_bound_rounds_and_raw_values_reach_solver(self):
        self.assertEqual(tuple(inspect.signature(run.execute_shared).parameters),
                         ('observation', 'radius', 'epsilon', 'rounds', 'max_boxes', 'tolerance'))
        raw = [run.encode(Q(-1, 100)), run.encode(Q(51, 50))] + [run.encode(Q(1, 2))]*430
        for bound in (run.EPSILON, run.DELTA):
            for rounds in (4, 8, 16):
                with patch.object(run, 'solve_arm', return_value='sentinel') as solver, patch.object(run, 'clear_geometry_cache') as clear:
                    self.assertEqual(run.execute_shared(raw, '1/5', bound, rounds, 511, '15/32'), 'sentinel')
                    clear.assert_called_once_with()
                    solver.assert_called_once_with(tuple(map(run.decode, raw)), Q(1, 5), 'global_gray',
                        epsilon=bound, rounds=rounds, max_boxes=511, tolerance=Q(15, 32))
        for epsilon, rounds, boxes, tolerance in (('1/200', 8, 511, '15/32'), ('1/50', 8.0, 511, '15/32'),
                ('1/50', True, 511, '15/32'), ('1/50', 8, 511.0, '15/32'), ('1/50', 8, 255, '15/32'),
                ('1/50', 8, 511, '1/4')):
            with self.assertRaises((ValueError, TypeError)), patch.object(run, 'solve_arm') as solver:
                run.execute_shared(raw, '1/5', epsilon, rounds, boxes, tolerance)
            solver.assert_not_called()

    def test_exact_values_and_public_radius(self):
        with self.assertRaises(TypeError):
            run.exact(.01)
        with self.assertRaises(TypeError):
            run.exact(True)
        with self.assertRaises(ValueError):
            run.execute_shared([run.encode(Q(1, 2))]*432, '-1/5', '1/50', 8, 511, '15/32')


class ReuseTests(unittest.TestCase):
    def fixture(self):
        base, base_tasks, _ = base_fixture(1)
        observations, tasks = run.generate_public(base, base_tasks)
        inputs = {str(run.BASE/rel): 'fixed-sha-'+rel for rel in run.helpers.INPUT_RELATIVE}
        documents = {run.BASE/'verification/VERIFY.json': {'status': 'PASS'}}
        for unit, (source, method) in run.REUSE_SOURCES.items():
            item = next(row for row in observations if row['unit_id'] == unit)
            source_oid = 'o000' if unit == 'R0epsilon' else 'b1-o000'
            entry = shared_fixture(Q(item['bound']), item['rounds'])
            source_tasks = []
            source_decisions = []
            for task in base_tasks:
                tid = task['task_id'] if unit == 'R0epsilon' else 'b1-'+task['task_id']
                source_tasks.append({'task_id': tid, 'observation_id': source_oid,
                    'base_task_id': task['task_id'], 'incumbent': task['incumbent']})
                source_decisions.append({'task_id': tid, 'method': method,
                    **run.old.select_action(entry['outer'], run.decode(task['incumbent']))})
            documents.update({source/'verification/VERIFY.json': {'status': 'PASS', 'host': 'liekkas', 'output': str(source)},
                source/'RUN_STATUS.json': {'status': 'COMPLETE'},
                source/'INPUTS.json': {'source_root': str(run.BASE), 'files': copy.deepcopy(inputs)},
                source/'PROTOCOL.json': {'max_boxes_per_branch': 511, 'tolerance_mm': '15/32',
                    'allowed_branches': list(run.old.BRANCHES), 'epsilon': '1/100',
                    'method_configs': {method: {'mode': 'global_gray', 'rounds': 8, 'max_boxes': 511,
                                               'epsilon_by_beta': {'1/100': '1/50'}}}},
                source/'observations.json': [{'observation_id': source_oid, 'base_observation_id': 'o000',
                    'observation': copy.deepcopy(item['observation']), 'radius': '1/5', 'beta': '1/100'}],
                source/'tasks.json': source_tasks,
                source/'outers.json': [{'observation_id': source_oid, 'arms': {method: entry}}],
                source/'decisions.json': source_decisions,
                source/'TIMING.json': {'observations': [{'observation_id': source_oid,
                    'outer_seconds_by_method': {method: 0.0}, 'selector_seconds_by_method': {method: .001}}]}})
        return observations, tasks, base, base_tasks, inputs, documents

    def test_reuse_identity_and_action_provenance(self):
        observations, tasks, base, base_tasks, inputs, documents = self.fixture()
        with patch.object(run, 'load', side_effect=lambda path: copy.deepcopy(documents[Path(path)])):
            reused, provenance = run.prepare_reuse(observations, tasks, base, base_tasks, inputs)
        self.assertEqual(set(reused), {'R0epsilon-o000', 'CBdelta-o000'})
        self.assertEqual(sum(len(row['tasks']) for row in provenance), 18)
        self.assertEqual(provenance[1]['source']['observation_id'], 'b1-o000')
        self.assertEqual(provenance[1]['source']['method'], 'global_shared_wide')
        self.assertTrue(all(len(row['outer_sha256']) == 64 for row in provenance))

    def test_reject_changed_reuse_inputs_parameters_actions_and_target(self):
        def change_image(d): d[run.BIAS/'observations.json'][0]['observation'][0] = run.encode(Q(9, 10))
        def change_incumbent(d): d[run.BIAS/'tasks.json'][0]['incumbent'] = run.encode(Q(601))
        def change_lineage(d): d[run.ABLATION/'INPUTS.json']['files'][str(run.BASE/'observations.json')] = 'different'
        def change_epsilon(d): d[run.BIAS/'outers.json'][0]['arms']['global_shared_wide']['outer']['epsilon'] = run.encode(run.EPSILON)
        def change_action(d): d[run.ABLATION/'decisions.json'][0]['output'] = run.encode(Q(600))
        def change_host(d): d[run.BIAS/'verification/VERIFY.json']['host'] = 'another-host'
        def change_budget(d): d[run.ABLATION/'outers.json'][0]['arms']['global_gray511']['outer']['budget'] = 255
        def duplicate_outer(d): d[run.ABLATION/'outers.json'].append(copy.deepcopy(d[run.ABLATION/'outers.json'][0]))
        for mutate in (change_image, change_incumbent, change_lineage, change_epsilon, change_action, change_host, change_budget, duplicate_outer):
            with self.subTest(mutation=mutate.__name__):
                observations, tasks, base, base_tasks, inputs, documents = self.fixture()
                mutate(documents)
                with patch.object(run, 'load', side_effect=lambda path: copy.deepcopy(documents[Path(path)])), self.assertRaises(RuntimeError):
                    run.prepare_reuse(observations, tasks, base, base_tasks, inputs)

    def test_read_only_actual_historical_reuse_108(self):
        run.check_target()
        inputs, sources = run.input_lock(), run.science_lock()
        base = run.load(run.BASE/'observations.json')
        base_tasks = run.load(run.BASE/'tasks.json')
        observations, tasks = run.generate_public(base, base_tasks)
        reused, provenance = run.prepare_reuse(observations, tasks, base, base_tasks, inputs)
        self.assertEqual((len(inputs), len(reused), sum(len(row['tasks']) for row in provenance)), (26, 108, 972))
        self.assertGreater(len(sources), 58)
        self.assertEqual(sum(len(row['actions']) for row in reused.values()), 972)


class PersistenceTests(unittest.TestCase):
    def test_guards_no_write_outside_root_or_existing_target(self):
        run.check_target()
        with self.assertRaises(ValueError): run.validate_output(run.ROOT)
        with self.assertRaises(ValueError): run.validate_output(run.BASE/'must-not-create')
        with self.assertRaises(FileExistsError): run.validate_output(run.ROOT/'PROTOCOL.md')
        with self.assertRaises(ValueError): run.save(run.BASE/'must-not-create.json', {})
        with self.assertRaises(FileExistsError): run.save(run.ROOT/'PROTOCOL.md', {})
        with self.assertRaises(ValueError): run.save(run.ROOT/'PROTOCOL.md', {}, replace=True)
        with patch.object(run.socket, 'gethostname', return_value='not-liekkas'), self.assertRaises(RuntimeError):
            run.validate_output(run.ROOT/'outputs/never-created')

    def test_absolute_shared_deadline_and_no_reset(self):
        output = run.ROOT/'outputs/never-created-deadline-test'
        with patch.object(run, 'validate_output', return_value=output), patch.object(run.time, 'time', return_value=1000), \
                patch.object(run.signal, 'getitimer', return_value=(0, 0)), patch.object(run.signal, 'signal'), \
                patch.object(run.signal, 'setitimer') as timer, patch.object(run, '_run', return_value='complete') as main:
            self.assertEqual(run.run(output, 1300), 'complete')
            budget = main.call_args.args[2]
            self.assertEqual(budget['deadline_unix'], 1300)
            self.assertEqual(budget['limit_seconds'], 5400)
            self.assertEqual(timer.call_args_list[0].args, (signal.ITIMER_REAL, 300))
            self.assertEqual(timer.call_args_list[-1].args, (signal.ITIMER_REAL, 0))
            for deadline in (999, 1000, 6401, float('nan'), float('inf')):
                with self.assertRaises(ValueError): run.run(output, deadline)
        with self.assertRaises(TimeoutError): run.deadline_reached(None, None)

    def _exercise_action_gate(self, corrupt=False):
        base, base_tasks, _ = base_fixture()
        observations, tasks = run.generate_public(base, base_tasks)
        reused = memory_reuse(observations, tasks)
        output = run.ROOT/'outputs/in-memory-only'
        saved, events = {}, []
        def fake_save(path, value, **kwargs):
            # This MUST copy: retaining list references would hide a missing flush.
            saved[Path(path)] = copy.deepcopy(value)
            if Path(path).name == 'decisions.json': events.append(('saved', len(value)))
        def fake_load(path):
            path = Path(path)
            if path == run.BASE/'observations.json': return copy.deepcopy(base)
            if path == run.BASE/'tasks.json': return copy.deepcopy(base_tasks)
            if path == output/'decisions.json':
                events.append(('read', len(saved[path])))
                if corrupt:
                    return copy.deepcopy(saved[path][:-1])
            if path == run.BASE/'evaluation/truth.json': self.fail('private truth parsed before explicit gate')
            return copy.deepcopy(saved[path])
        def private(public):
            self.assertEqual(events[-1], ('read', 7776))
            self.assertEqual(len(saved[output/'decisions.json']), 7776)
            events.append(('truth', 7776))
            return []
        with patch.object(run, 'input_lock', return_value={'fixed': 'hash'}), \
                patch.object(run, 'science_lock', return_value={'code': 'hash'}), \
                patch.object(run, 'load', side_effect=fake_load), patch.object(run, 'save', side_effect=fake_save), \
                patch.object(run, 'prepare_reuse', return_value=(reused, [])), \
                patch.object(run, 'execute_shared', side_effect=lambda image, radius, bound, rounds, boxes, tol: shared_fixture(Q(bound), rounds)) as solver, \
                patch.object(run, 'private_truths', side_effect=private) as truth_reader, \
                patch.object(run, 'evaluate', return_value={'checked': True}) as evaluator, \
                patch.object(Path, 'mkdir'), patch('sys.stdout', new=io.StringIO()):
            state = {}
            if corrupt:
                with self.assertRaisesRegex(RuntimeError, 'saved and read back before truth parsing'):
                    run._run(output, time.perf_counter(), {'deadline_unix': time.time()+1000}, state)
                truth_reader.assert_not_called()
                evaluator.assert_not_called()
                self.assertEqual(saved[output/'RUN_STATUS.json']['status'], 'RUNNING')
                return
            result = run._run(output, time.perf_counter(), {'deadline_unix': time.time()+1000}, state)
        self.assertEqual(result, {'checked': True})
        self.assertEqual(solver.call_count, 324)
        truth_reader.assert_called_once()
        evaluator.assert_called_once()
        self.assertTrue(state['owns_output'])
        self.assertEqual(saved[output/'RUN_STATUS.json']['status'], 'COMPLETE')
        self.assertEqual(events[-1], ('truth', 7776))
        decisions = saved[output/'decisions.json']
        self.assertEqual(sum(row['method'] == 'keep' and row['reused'] for row in decisions), 0)
        self.assertEqual(sum(row['method'] == 'shared' and row['reused'] for row in decisions), 972)
        self.assertEqual(len(saved[output/'outers.json']), 432)

    def test_all_actions_deepcopy_readback_before_truth_and_evaluation(self):
        self._exercise_action_gate()

    def test_missing_saved_action_blocks_private_truth_and_evaluation(self):
        self._exercise_action_gate(corrupt=True)


class EvaluationTests(unittest.TestCase):
    def test_full_denominators_pairs_interaction_and_keep_metadata(self):
        base, base_tasks, private_base = base_fixture()
        observations, tasks = run.generate_public(base, base_tasks)
        private = {row['observation_id']: row for row in private_base}
        truths = [{**private[item['base_observation_id']], **{key: item[key] for key in run.META}} for item in observations]
        by_oid = {item['observation_id']: item for item in observations}
        decisions, outers, timings = [], [], []
        for task in tasks:
            item = by_oid[task['observation_id']]
            for method in run.METHODS:
                decisions.append({'task_id': task['task_id'], 'method': method,
                    'unit_id': item['unit_id'], 'bound': item['bound'], 'rounds': item['rounds'],
                    'reused': item['reused'] and method == 'shared',
                    **run.old.select_action(run.helpers.identity_arm()['outer'], run.decode(task['incumbent']), keep=method == 'keep')})
        for item in observations:
            metadata = {key: item[key] for key in run.META}
            outers.append({**metadata, 'arms': {method: run.helpers.identity_arm() for method in run.METHODS}})
            timings.append({**metadata, 'new_solver_seconds_by_method': dict.fromkeys(run.METHODS, 0.),
                'selector_seconds_by_method': dict.fromkeys(run.METHODS, 0.),
                'historical_outer_seconds_by_method': dict.fromkeys(run.METHODS, 0.)})
        saved = {}
        def fake_save(path, value, **kwargs): saved[Path(path).name] = copy.deepcopy(value)
        with patch.object(run, 'save', side_effect=fake_save), patch.object(run, 'csv_save'):
            summary = run.evaluate(run.ROOT/'outputs/in-memory-only', observations, tasks, truths,
                                   decisions, outers, timings, time.perf_counter())
        self.assertEqual((len(saved['ROWS.json']), len(saved['COVERAGE.json']), len(saved['PAIRED.json']),
                          len(saved['INTERACTION.json'])), (7776, 864, 8262, 486))
        self.assertEqual((len(summary['rows']), len(summary['group_mean_sd']), len(summary['per_unit_method_timing'])), (1440, 96, 16))
        self.assertEqual((summary['new_shared_actions'], summary['reused_shared_actions'], summary['keep_actions']), (2916, 972, 3888))
        self.assertTrue(all(not row['reused'] for row in summary['rows'] if row['method'] == 'keep'))
        for row in summary['rows']:
            if row['grouping'] == 'overall':
                self.assertEqual(row['n'], {'all': 486, 'correct': 54, 'small': 432}[row['state_group']])
            if row['grouping'] == 'by_information' and row['condition'] == 'negative_control' and row['state_group'] == 'small':
                self.assertEqual(row['n'], 72)
        self.assertTrue(all(row['difference_in_error_changes_mm'] == '0' for row in saved['INTERACTION.json']))
        self.assertTrue(all(row['same'] for row in saved['PAIRED.json']))

    def test_incompatible_is_not_keep_and_pair_signs(self):
        outer = run.helpers.identity_arm()['outer']
        outer['hull'], outer['retained_intervals'] = None, []
        action = run.old.select_action(outer, Q(600))
        self.assertEqual(action['status'], 'INCOMPATIBLE')
        self.assertEqual(run.decode(action['output']), Q(600))
        rows = []
        errors = {'R0delta': Q(5), 'C0delta': Q(4), 'RBdelta': Q(3), 'CBdelta': Q(1)}
        for unit in run.UNIT:
            for method in run.METHODS:
                rows.append({'unit_id': unit, 'base_task_id': 'one', 'method': method, 'case': 'fixture',
                    'seed': 99991, 'state_group': 'small', 'ae_mm': str(errors.get(unit, Q(6)) if method == 'shared' else Q(10))})
        pairs, interaction = run.make_pairs(rows)
        pair = next(row for row in pairs if row['left_unit'] == 'R0delta' and row['right_unit'] == 'C0delta')
        self.assertEqual((pair['mae_gain_mm'], pair['mse_gain_mm2'], pair['better']), ('1', '9', True))
        self.assertEqual(interaction[0]['difference_in_error_changes_mm'], '-1')


if __name__ == '__main__':
    unittest.main()
