"""Author contracts; mocks exercise orchestration without a confirmation solve."""
from collections import Counter
from contextlib import redirect_stdout
from fractions import Fraction as Q
import inspect
import io
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import run_view_bias as run


class ViewBiasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_observations = run.load(run.DATA / 'observations.json')
        cls.base_tasks = run.load(run.DATA / 'tasks.json')

    def test_view_major_rgb_layout(self):
        actual = list(map(run.decode, run.biased_observation([run.encode(Q(1, 2))] * 432, '1/100')))
        self.assertEqual(actual, [Q(49, 100)] * 144 + [Q(1, 2)] * 144 + [Q(51, 100)] * 144)
        for pixel in range(144):
            self.assertEqual(len(set(actual[3*pixel:3*pixel+3])), 1)

    def test_one_clip_after_addition_and_boundaries(self):
        image = [run.encode(Q(1, 2))] * 432
        image[0], image[1], image[144], image[288] = map(run.encode, (Q(201, 200), Q(0), Q(101, 100), Q(199, 200)))
        actual = list(map(run.decode, run.biased_observation(image, '1/100')))
        self.assertEqual(actual[0], Q(199, 200))  # preclipping would incorrectly yield 99/100
        self.assertEqual(actual[1], 0)
        self.assertEqual(actual[144], 1)
        self.assertEqual(actual[288], 1)

    def test_clip_preserves_declared_wide_error_bound(self):
        for beta in run.BETAS:
            for truth in (Q(0), Q(1, 200), Q(1, 4), Q(1, 2), Q(199, 200), Q(1)):
                for noise in (-run.EPSILON, Q(0), run.EPSILON):
                    image = [run.encode(truth+noise)] * 432
                    actual = list(map(run.decode, run.biased_observation(image, beta)))
                    self.assertTrue(all(abs(value-truth) <= run.EPSILON+beta for value in actual))

    def test_epsilon_explicit_for_each_method_and_beta(self):
        image = [run.encode(Q(1, 2))] * 432
        for beta in run.BETAS:
            for method in run.METHODS[1:]:
                with patch.object(run, 'clear_geometry_cache') as clear, patch.object(run, 'solve_arm', return_value={}) as solver:
                    run.execute_arm(image, '1/5', str(beta), method)
                clear.assert_called_once_with()
                args, kwargs = solver.call_args
                self.assertEqual(args, ((Q(1, 2),) * 432, Q(1, 5), run.MODES[method]))
                expected = run.EPSILON if method == 'shared_strict' else run.EPSILON+beta
                self.assertEqual(kwargs, {'epsilon': expected, 'max_boxes': 511, 'rounds': 8, 'tolerance': Q(15, 32)})

    def test_identity_never_calls_solver(self):
        with patch.object(run, 'clear_geometry_cache') as clear, patch.object(run, 'solve_arm') as solver:
            entry = run.execute_arm([run.encode(Q(1, 2))] * 432, '1/5', '1/200', 'keep')
        clear.assert_called_once_with()
        solver.assert_not_called()
        self.assertEqual(entry['branches'], {})
        self.assertEqual(entry['outer']['max_boxes_per_branch'], 511)
        self.assertTrue(run.covered(entry['outer'], Q(600)))

    def test_solver_interface_is_public_only_and_does_not_read(self):
        self.assertEqual(tuple(inspect.signature(run.execute_arm).parameters), ('observation', 'radius', 'beta', 'method'))
        with patch.object(run, 'load', side_effect=AssertionError('solver attempted file read')):
            with patch.object(run, 'solve_arm', return_value={}), patch.object(run, 'clear_geometry_cache'):
                run.execute_arm([run.encode(Q(1, 2))] * 432, '1/5', '1/50', 'global_shared_wide')

    def test_public_fields_and_full_population(self):
        observations, tasks = run.generate_public(self.base_observations, self.base_tasks)
        self.assertEqual((len(observations), len(tasks)), (162, 1458))
        self.assertEqual(len(tasks) * len(run.METHODS), 5832)
        self.assertEqual(Counter(item['beta'] for item in observations), {str(b): 54 for b in run.BETAS})
        self.assertEqual(len({item['observation_id'] for item in observations}), 162)
        self.assertEqual(len({item['task_id'] for item in tasks}), 1458)
        self.assertEqual(set(Counter(task['observation_id'] for task in tasks).values()), {9})
        original = {task['task_id']: task for task in self.base_tasks}
        for item in observations:
            self.assertEqual(set(item), {'observation_id', 'base_observation_id', 'beta', 'radius', 'observation'})
            self.assertEqual(len(item['observation']), 432)
            self.assertTrue(all(0 <= run.decode(value) <= 1 for value in item['observation']))
        for task in tasks:
            self.assertEqual(set(task), {'task_id', 'observation_id', 'base_task_id', 'incumbent'})
            self.assertEqual(task['incumbent'], original[task['base_task_id']]['incumbent'])
        self.assertEqual(observations[0]['observation_id'], 'b0-o000')
        self.assertEqual(observations[-1]['observation_id'], 'b2-o053')
        self.assertEqual(tasks[0]['task_id'], 'b0-o000-s0')

    def test_generation_cannot_copy_calibration_or_labels(self):
        first = {**self.base_observations[0], 'case': 'PRIVATE', 'seed': 999,
                 'truth': 'PRIVATE', 'calibration': ['PRIVATE'], 'actual_view_bias': ['PRIVATE']}
        tasks = [task for task in self.base_tasks if task['observation_id'] == first['observation_id']]
        with patch.object(run, 'load', side_effect=AssertionError('generation read private file')):
            generated, _ = run.generate_public([first], tasks)
        self.assertNotIn('PRIVATE', repr(generated))

    def test_invalid_layout_beta_radius_and_duplicate_ids(self):
        image = [run.encode(Q(1, 2))] * 432
        for value in ([], image[:-1], image+[image[0]]):
            with self.assertRaises(ValueError): run.biased_observation(value, '1/100')
        with self.assertRaises(ValueError): run.biased_observation(image, '1/1000')
        with self.assertRaises(TypeError): run.biased_observation(image, 0.01)
        with self.assertRaises(ValueError): run.execute_arm(image, '-1', '1/100', 'keep')
        with self.assertRaises(ValueError): run.execute_arm(image, '1/5', '1/100', 'oracle')
        with self.assertRaises(ValueError): run.execute_arm([run.encode(Q(2))] * 432, '1/5', '1/100', 'keep')
        with self.assertRaises(ValueError): run.generate_public([self.base_observations[0]] * 2, [])
        with self.assertRaises(ValueError): run.generate_public(self.base_observations, self.base_tasks[:-1])

    def test_output_target_and_overwrite_guards(self):
        with self.assertRaises(ValueError): run.validate_output('/tmp/substitute')
        with self.assertRaises(ValueError): run.validate_output(run.ROOT)
        with self.assertRaises(FileExistsError): run.validate_output(run.ROOT / 'refine-logs')
        with self.assertRaises(ValueError): run.validate_output(run.ROOT / '../outside')
        self.assertEqual(run.validate_output(run.ROOT / 'outputs/unit-not-created'), run.ROOT / 'outputs/unit-not-created')
        with patch.object(run.socket, 'gethostname', return_value='other-host'):
            with self.assertRaises(RuntimeError): run.validate_output(run.ROOT / 'outputs/unit-not-created')
        with self.assertRaises(ValueError): run.save('/tmp/forbidden.json', {})
        with self.assertRaises(ValueError): run.save(run.ROOT / 'PROTOCOL.md', {}, replace=True)
        with self.assertRaises(FileExistsError): run.save(run.ROOT / 'PROTOCOL.md', {})

    def test_frozen_inputs_and_registered_configuration(self):
        inputs = run.input_lock()
        self.assertEqual(len(inputs), 11)
        self.assertIn(str(run.PRIOR_OUTPUT / 'SOURCE_LOCK.json'), inputs)
        self.assertIn(str(run.PRIOR_OUTPUT / 'verification/VERIFY.json'), inputs)
        self.assertIn(str(run.PRIOR_OUTPUT / 'PROTOCOL.json'), inputs)
        config = run.configuration()
        self.assertEqual(config['expected_per_beta_per_method'],
                         {'observations': 54, 'correct': 54, 'small': 432, 'negative_control_small': 72})
        self.assertEqual(config['wall_time_limit_seconds'], 5400)
        self.assertFalse(config['method_configs']['shared_strict']['contract_valid'])
        self.assertEqual(config['method_configs']['global_shared_wide']['epsilon_by_beta'],
                         {'1/200': '3/200', '1/100': '1/50', '1/50': '3/100'})

    def test_all_decisions_saved_before_private_truth_is_read(self):
        memory, output = {}, run.ROOT / 'outputs/unit-never-created'
        def fake_load(path):
            path = Path(path)
            if path in memory: return memory[path]
            if path.name == 'VERIFY.json': return {'status': 'PASS'}
            if path == run.DATA / 'observations.json': return self.base_observations
            if path == run.DATA / 'tasks.json': return self.base_tasks
            raise AssertionError('unexpected input read: ' + str(path))
        def fake_save(path, value, **kwargs): memory[Path(path)] = value
        def private(observations):
            self.assertEqual(len(memory[output / 'decisions.json']), 5832)
            self.assertIn(output / 'outers.json', memory)
            self.assertEqual(len(observations), 162)
            return []
        with patch.object(run, 'load', side_effect=fake_load), patch.object(run, 'save', side_effect=fake_save), \
             patch.object(run, 'input_lock', return_value={}), patch.object(run, 'science_lock', return_value={}), \
             patch.object(Path, 'mkdir'), patch.object(run, 'execute_arm', side_effect=lambda *args: run.identity_arm()), \
             patch.object(run, 'private_truths', side_effect=private) as private_call, \
             patch.object(run, 'evaluate', return_value={'ok': True}), redirect_stdout(io.StringIO()):
            result = run._run(output, time.perf_counter(), {})
        private_call.assert_called_once()
        self.assertEqual(result, {'ok': True})
        self.assertEqual(memory[output / 'RUN_STATUS.json']['status'], 'COMPLETE')
        self.assertEqual(sum(path.parent.name == 'checkpoint' for path in memory), 162)
        self.assertFalse(output.exists())

    def test_real_deadline_wraps_programmatic_run_and_resets(self):
        output = run.ROOT / 'outputs/unit-never-created'
        with patch.object(run.signal, 'getitimer', return_value=(0., 0.)), \
             patch.object(run.signal, 'signal', return_value=run.signal.SIG_DFL), \
             patch.object(run.signal, 'setitimer') as timer, \
             patch.object(run, '_run', side_effect=TimeoutError('test deadline')):
            with self.assertRaises(TimeoutError): run.run(output)
        self.assertEqual(timer.call_args_list[0].args, (run.signal.ITIMER_REAL, 5400))
        self.assertEqual(timer.call_args_list[-1].args, (run.signal.ITIMER_REAL, 0))

    def test_strict_diagnostic_failure_is_retained_by_evaluation(self):
        observations, tasks = run.generate_public(self.base_observations, self.base_tasks)
        truths = run.private_truths(observations)
        outers, timing = [], []
        for item in observations:
            arms = {method: run.identity_arm() for method in run.METHODS}
            arms['shared_strict']['outer'].update(hull=None, retained_intervals=[])
            outers.append({**{key: item[key] for key in ('observation_id', 'base_observation_id', 'beta')}, 'arms': arms})
            timing.append({'observation_id': item['observation_id'], 'beta': item['beta'],
                'outer_seconds_by_method': dict.fromkeys(run.METHODS, 0.),
                'selector_seconds_by_method': dict.fromkeys(run.METHODS, 0.)})
        decisions = [{'task_id': task['task_id'], 'method': method,
                      'output': run.encode(run.decode(task['incumbent'])+(1 if method == 'shared_strict' else 0)),
                      'status': 'MOVE' if method == 'shared_strict' else 'KEEP'}
                     for task in tasks for method in run.METHODS]
        memory = {}
        with patch.object(run, 'save', side_effect=lambda path, value: memory.update({Path(path).name: value})), \
             patch.object(run, 'csv_save'):
            summary = run.evaluate(run.ROOT / 'outputs/unit-never-created', observations, tasks, truths,
                                   decisions, outers, timing, time.perf_counter())
        self.assertEqual(len(memory['ROWS.json']), 5832)
        self.assertEqual(len(memory['COVERAGE.json']), 648)
        self.assertEqual(len(memory['PAIRED.json']), 5832)
        strict = [row for row in summary['rows'] if row['method'] == 'shared_strict'
                  and row['grouping'] == 'overall' and row['state_group'] == 'correct']
        self.assertEqual(len(strict), 3)
        self.assertTrue(all(row['harm'] == 54 and row['wrongly_excluded_observations'] == 54
                            and not row['contract_valid'] for row in strict))
        negative = [row for row in summary['rows'] if row['grouping'] == 'by_information'
                    and row['condition'] == 'negative_control' and row['state_group'] == 'small']
        self.assertEqual(len(negative), 12)
        self.assertTrue(all(row['n'] == 72 for row in negative))


if __name__ == '__main__':
    unittest.main()
