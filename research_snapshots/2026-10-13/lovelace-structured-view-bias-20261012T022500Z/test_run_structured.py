"""Runner contracts, provenance and public-only forwarding; no formal batch."""
import ast
from collections import Counter
from fractions import Fraction as Q
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_structured as r


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations, cls.tasks = r.population()

    def test_exact_target(self):
        r.check_target()
        with patch.object(r.socket, 'gethostname', return_value='not-liekkas'):
            with self.assertRaises(RuntimeError):
                r.check_target()

    def test_population_preserves_images_and_nine_initial_states(self):
        self.assertEqual(len(self.observations), 108)
        self.assertEqual(len(self.tasks), 972)
        self.assertEqual(Counter(t['observation_id'] for t in self.tasks),
                         {o['observation_id']: 9 for o in self.observations})
        source = {o['observation_id']: o for o in r.load(r.DATA / 'observations.json')}
        for o in self.observations:
            self.assertEqual(o['observation'], source[o['source_observation_id']]['observation'])
            self.assertEqual(o['clipped'], o['unit_id'] == 'clip')
            self.assertEqual(len(o['observation']), 432)
        self.assertEqual(Counter(o['unit_id'] for o in self.observations), {'raw': 54, 'clip': 54})

    def test_all_wide_actions_and_source_body_hashes(self):
        reused, provenance = r.prepare_reuse(self.observations, self.tasks)
        self.assertEqual(len(reused), 108)
        self.assertEqual(sum(len(v['actions']) for v in reused.values()), 972)
        images = {o['observation_id']: o for o in r.load(r.DATA / 'observations.json')}
        actions = {(a['task_id'], a['method']): a for a in r.load(r.DATA / 'decisions.json')}
        for ref in provenance:
            self.assertEqual(ref['outer_sha256'], r.json_hash(reused[ref['observation_id']]['entry']))
            self.assertEqual(ref['observation_sha256'], r.json_hash(images[ref['source_observation_id']]))
            for task in ref['tasks']:
                self.assertEqual(task['action_sha256'], r.json_hash(actions[task['source_task_id'], 'shared']))

    def test_public_only_solver_forwarding(self):
        calls = []
        def solve(*args, **kwargs):
            calls.append((args, kwargs))
            return {'sentinel': True}
        image = ['1'] * 432
        with patch.object(r.helpers, 'clear_geometry_cache'), patch.object(r, 'solve_structured', side_effect=solve):
            self.assertEqual(r.execute_new(image, '1/5', 'joint', True), {'sentinel': True})
        args, kwargs = calls[0]
        self.assertEqual(args, (tuple([Q(1)] * 432), Q(1, 5), 'joint'))
        self.assertEqual(kwargs, {'clipped': True, 'beta': Q(1, 100), 'epsilon': Q(1, 100),
                                 'rounds': 8, 'max_boxes': 511, 'tolerance': Q(15, 32)})
        self.assertEqual(tuple(inspect.signature(r.execute_new).parameters),
                         ('observation', 'radius', 'method', 'clipped'))
        with self.assertRaises(ValueError):
            r.execute_new(image, '1/5', 'wide', True)

    def test_saved_full_action_fences_precede_private_parse(self):
        tree = ast.parse(Path(r.__file__).read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        calls = [(f.name, n.lineno) for f in functions.values() for n in ast.walk(f)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'private_truths']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], '_run')
        private_line = calls[0][1]
        gates = [n for n in functions['_run'].body if isinstance(n, ast.If)
                 and any(isinstance(c, ast.Raise) for c in n.body)]
        self.assertEqual(len(gates), 3)
        self.assertLess(gates[0].end_lineno, private_line)
        self.assertLess(gates[1].end_lineno, private_line)
        self.assertIn('3888', ast.unparse(gates[0].test))
        self.assertIn("load(output / 'decisions.json') != decisions", ast.unparse(gates[0].test))
        self.assertIn('972', ast.unparse(gates[1].test))

    def test_exact_input_and_source_inventory(self):
        inputs, sources = r.input_lock(), r.science_lock()
        self.assertIn(str(r.DATA / 'verification/VERIFY.json'), inputs)
        self.assertIn(str(r.ROOT / 'structured_solver.py'), sources)
        self.assertIn(str(r.ROOT / 'PROTOCOL.md'), sources)
        self.assertEqual(inputs, r.input_lock())
        self.assertEqual(sources, r.science_lock())

    def test_output_is_owned_exclusive_and_progress_only(self):
        with tempfile.TemporaryDirectory(prefix='runner-unit-', dir=r.ROOT) as folder:
            path = Path(folder) / 'RUN_STATUS.json'
            r.save(path, {'status': 'one'})
            with self.assertRaises(FileExistsError):
                r.save(path, {'status': 'bad'})
            r.save(path, {'status': 'two'}, replace=True)
            self.assertEqual(r.load(path), {'status': 'two'})
            with self.assertRaises(ValueError):
                r.save(Path(folder) / 'outers.json', [], replace=True)
        with self.assertRaises(ValueError):
            r.save(r.PREV / 'forbidden.json', {})

    def test_counts_and_single_budget_window(self):
        config = r.configuration()
        self.assertEqual(config['expected_decisions'], 3888)
        self.assertEqual(config['expected_new_solver_calls'], 216)
        self.assertEqual(config['expected_new_method_actions'], 1944)
        self.assertEqual(config['wall_time_limit_seconds_main_plus_verification'], 5400)
        with self.assertRaises(ValueError):
            r.run(r.ROOT)
        with self.assertRaises(ValueError):
            r.run(r.ROOT / 'never-created-test-output', deadline_unix=0)

    def test_pairing_compares_same_initial_and_uses_exact_errors(self):
        rows = [{'unit_id': unit, 'base_task_id': 'o000-s0', 'method': method,
                 'case': 'fixture', 'seed': 0, 'state_group': 'small', 'ae_mm': str(index)}
                for unit in r.UNITS for index, method in enumerate(r.METHODS)]
        pairs = r.paired(rows)
        self.assertEqual(len(pairs), 12)
        self.assertTrue(all(p['worse'] for p in pairs))
        first = pairs[0]
        self.assertEqual((first['mae_gain_mm'], first['mse_gain_mm2']), ('-1', '-1'))

    def test_nonconfirmation_fixture_solver_verifier_integration(self):
        import verify_structured as verifier
        public, _private = r.old.generate_observation('smoke_only', 'partial', (Q(1, 4), Q(1, 2), Q(3, 4)),
                                                      Q(600), 99125, prefix='structured-bias-smoke-only')
        _model, geometry = verifier._historical.shared_kernels()
        for clipped in (False, True):
            kind = 'biased_clip' if clipped else 'biased_raw'
            image = r.bridge.transform_image(public['observation'], kind)
            for method in ('calibrated', 'joint'):
                arm = r.execute_new(image, public['radius'], method, clipped)
                calibration = verifier.verify_arm_public(arm, image, clipped, geometry, 'fixture')
                branches = {branch: verifier.verify_branch(body, image, clipped, method, branch, geometry,
                                                           calibration, 'fixture/' + branch)
                            for branch, body in arm['branches'].items()}
                checked = verifier.verify_union(arm['outer'], branches, method, clipped, 'fixture')
                self.assertTrue(any(lo <= 600 <= hi for lo, hi in checked['intervals']))


if __name__ == '__main__':
    unittest.main()
