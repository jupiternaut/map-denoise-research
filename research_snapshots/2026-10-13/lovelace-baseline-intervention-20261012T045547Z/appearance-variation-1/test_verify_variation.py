"""Hand fixtures only: no formal generation or private facts are read."""
import ast
from copy import deepcopy
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('independent_variation_verifier_test', Path(__file__).with_name('verify_variation.py'))
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def fixture_branch(image, method, name, calibration, branch):
    dimension = 3 if name == 'strict_shared' else 9
    initial = [[v.encoded(0), v.encoded(1)] for _ in range(dimension)]
    if method == 'joint':
        initial.extend(calibration['intervals'])
    queue = [(None, *v.DOMAIN)]
    trace, retained, evaluated = [], [], 0
    for ident, (parent, low, high) in enumerate(queue):
        node = {'id': ident, 'parent': parent, 'lo': v.encoded(low), 'hi': v.encoded(high)}
        if calibration['excluded']:
            node.update(status='excluded_calibration', certificate=calibration['certificate'])
        else:
            geometry = v.independent_geometry(branch, v.RADIUS, low, high, Q(80))
            if name == 'bounded':
                rows = v.necessary_rows(image, geometry, True, method, calibration['intervals'], v.DELTAS[name])
            else:
                rows = v._appearance.necessary_rows(image, geometry, True, method, calibration['intervals'],
                                                     appearance_scope='global_shared' if name == 'strict_shared' else 'per_view')
            contraction = v.independent_signed_contraction(rows, initial)
            evaluated += 1
            node.update(rounds=contraction['rounds'], updates=contraction['updates'])
            if contraction['excluded']:
                node.update(status='excluded_shared', group_id='global', certificate=contraction['certificate'])
            else:
                node.update(darkness_bounds=contraction['bounds'][:dimension],
                            t_bounds=contraction['bounds'][dimension:] if method == 'joint' else calibration['intervals'])
                if high - low <= v.TOLERANCE:
                    node['status'] = 'retained_tolerance'
                    retained.append(node)
                else:
                    middle = (low + high) / 2
                    node.update(status='split', children=[len(queue), len(queue) + 1])
                    queue.extend(((ident, low, middle), (ident, middle, high)))
        trace.append(node)
    intervals = [(v.exact(node['lo']), v.exact(node['hi'])) for node in retained]
    hull = None if not intervals else (min(a for a, _ in intervals), max(b for _, b in intervals))
    outer = {'domain': list(map(v.encoded, v.DOMAIN)), 'epsilon': v.encoded(v.EPSILON), 'beta': v.encoded(v.BETA),
             'clipped': True, 'tolerance': v.encoded(v.TOLERANCE), 'max_boxes': v.MAX_BOXES, 'evaluated_boxes': evaluated,
             'trace': trace, 'retained_ids': [node['id'] for node in retained], 'hull': None if hull is None else list(map(v.encoded, hull)),
             'shared_tests': evaluated, 'propagation_rounds': v.ROUNDS, 'mode': method, 'rounds': v.ROUNDS,
             'budget': v.MAX_BOXES, 'channel_precheck': 'none', 'initial_bounds': initial,
             'scope': v._prior.GROUP_SCOPE if name == 'strict_shared' else v.GROUP_SCOPE if name == 'bounded' else v._appearance.GROUP_SCOPE}
    if name != 'strict_shared':
        outer.update(appearance_scope='per_view', material_variable_names=list(v.MATERIAL_VARIABLE_NAMES))
    return outer


def fixture_arm(method='joint', name='bounded', conflict=False):
    depth, offsets, colors = Q(643), (Q(1, 17), Q(0), Q(-1, 19)), (Q(1, 5), Q(3, 5), Q(2, 5))
    darkness = v.variation_darkness(colors, Q(1, 10))
    image = v.independent_rgb_per_view('partial', depth, offsets, [1 - d for d in darkness])
    bias = (-v.BETA, Q(0), v.BETA)
    image = tuple(min(Q(1), max(Q(0), value + bias[index // 144])) for index, value in enumerate(image))
    if conflict:
        image = (Q(0),) * 432
    geometry = lambda branch, radius, low, high: v.independent_geometry(branch, radius, low, high, Q(80))
    background = v._prior.public_background_mask(geometry)
    calibration = v._prior.calibration_from_public(image, True, background)
    branches = {branch: fixture_branch(image, method, name, calibration, branch) for branch in v.BRANCHES}
    source = [{'branch': branch, 'id': node['id'], 'lo': node['lo'], 'hi': node['hi']}
              for branch, outer in branches.items() for node in outer['trace'] if node['id'] in outer['retained_ids']]
    intervals = v._arithmetic.merged_intervals((v.exact(node['lo']), v.exact(node['hi'])) for node in source)
    hull = None if not intervals else (intervals[0][0], intervals[-1][1])
    outer = {'domain': list(map(v.encoded, v.DOMAIN)), 'epsilon': v.encoded(v.EPSILON), 'beta': v.encoded(v.BETA), 'clipped': True,
             'tolerance': v.encoded(v.TOLERANCE), 'max_boxes_per_branch': v.MAX_BOXES,
             'evaluated_boxes': sum(branch['evaluated_boxes'] for branch in branches.values()), 'unresolved_budget_leaves': 0,
             'source_leaves': source, 'retained_intervals': [list(map(v.encoded, interval)) for interval in intervals],
             'hull': None if hull is None else list(map(v.encoded, hull)),
             'retained_branches': [branch for branch, value in branches.items() if value['hull'] is not None],
             'mode': method, 'rounds': v.ROUNDS, 'budget': v.MAX_BOXES,
             'scope': 'union_of_branch_depth_outer_leaves_then_convex_hull', 'channel_precheck': 'none'}
    arm = {'branches': branches, 'outer': outer, 'branch_seconds': {branch: 0 for branch in v.BRANCHES},
           'outer_seconds': 0, 'preparation_seconds': 0, 'background_mask': background,
           'calibration_intervals': calibration['intervals'], 'calibration': calibration,
           'appearance_model': name, 'drift_bound': v.encoded(v.DELTAS[name])}
    if name != 'strict_shared':
        arm.update(appearance_scope='per_view', material_variable_names=list(v.MATERIAL_VARIABLE_NAMES))
    public = {'observation_id': 'manual_nonformal', 'unit_id': 'appearance_eta_1_10', 'baseline': '80',
              'radius': '1/5', 'clipped': True, 'observation': list(map(v.encoded, image))}
    return public, arm


class SignedRowsAndArithmeticTests(unittest.TestCase):
    def test_eighteen_exact_link_order_and_bias_zeros(self):
        for method in v.METHODS:
            rows = v.link_rows(method, Q(1, 10))
            self.assertEqual(len(rows), 18)
            index = 0
            for left, right in ((0, 1), (0, 2), (1, 2)):
                for material in range(3):
                    for direction in (1, -1):
                        coefficients = [Q(0)] * (12 if method == 'joint' else 9)
                        coefficients[3 * left + material], coefficients[3 * right + material] = Q(direction), Q(-direction)
                        self.assertEqual(rows[index], ('upper', tuple(coefficients), Q(1, 10)))
                        index += 1
        for bad in (-1, Q(11, 10), True, 0.1):
            with self.assertRaises(v.VerificationError):
                v.link_rows('joint', bad)

    def test_signed_extension_is_exact_old_contractor_on_nonnegative_rows(self):
        for dimension in (3, 6, 9, 12):
            initial = [[v.encoded(0), v.encoded(1)] for _ in range(dimension)]
            for rounds in (0, 1, 3, 8):
                for rows in ((), (('upper', (Q(1),) * dimension, Q(1, 3)),),
                             (('lower', (Q(1),) * dimension, Q(2, 3)),),
                             (('upper', (Q(1),) + (Q(0),) * (dimension - 1), Q(1, 3)),
                              ('lower', (Q(1),) + (Q(0),) * (dimension - 1), Q(2, 3)))):
                    self.assertEqual(v.independent_signed_contraction(rows, initial, rounds),
                                     v._appearance.independent_contraction(rows, initial, rounds))

    def test_negative_coefficient_updates_actual_lower_and_upper_endpoints(self):
        coefficients = (Q(1), Q(-1), Q(0))
        cases = [('upper', [[Q(7, 10), Q(9, 10)], [Q(0), Q(1)], [Q(0), Q(1)]], 1, 'lower', Q(3, 5)),
                 ('lower', [[Q(0), Q(1)], [Q(7, 10), Q(9, 10)], [Q(0), Q(1)]], 0, 'lower', Q(4, 5)),
                 ('upper', [[Q(0), Q(1)], [Q(0), Q(3, 10)], [Q(0), Q(1)]], 0, 'upper', Q(2, 5)),
                 ('lower', [[Q(0), Q(3, 10)], [Q(0), Q(1)], [Q(0), Q(1)]], 1, 'upper', Q(1, 5))]
        for side, bounds, variable, endpoint, value in cases:
            result = v.independent_signed_contraction(((side, coefficients, Q(1, 10)),), bounds, 1)
            self.assertEqual(next(step for step in result['certificate']['steps'] if step['variable'] == variable),
                             {'round': 0, 'row': 0, 'variable': variable, 'side': endpoint, 'value': v.encoded(value)})

    def test_signed_row_contradictions_are_exact_first_conflicts(self):
        for side, bounds, expected in (('upper', [[Q(7, 10), Q(9, 10)], [Q(0), Q(3, 10)], [Q(0), Q(1)]], Q(2, 5)),
                                       ('lower', [[Q(0), Q(3, 10)], [Q(7, 10), Q(9, 10)], [Q(0), Q(1)]], Q(-2, 5))):
            result = v.independent_signed_contraction(((side, (Q(1), Q(-1), Q(0)), Q(1, 10)),), bounds)
            self.assertTrue(result['excluded'])
            self.assertEqual(result['updates'], 0)
            self.assertEqual(result['certificate']['contradiction'],
                             {'kind': 'row', 'round': 0, 'row': 0, 'total': v.encoded(expected), 'bound': v.encoded(Q(1, 10))})

    def test_delta_one_redundant_links_preserve_old_perview_full_receipt(self):
        image = (Q(99, 100),) * 432
        geometry = ((Q(0), Q(1), Q(0), Q(1), Q(0), Q(1)),) * 144
        calibration = [[v.encoded(0), v.encoded(Q(1, 50))]] * 3
        for method in v.METHODS:
            rows = v._appearance.necessary_rows(image, geometry, True, method, calibration)
            initial = [[v.encoded(0), v.encoded(1)]] * 9 + (calibration if method == 'joint' else [])
            self.assertEqual(v.independent_signed_contraction(rows + v.link_rows(method, Q(1)), initial),
                             v._appearance.independent_contraction(rows, initial))


class IndependentForwardTests(unittest.TestCase):
    def source(self):
        colors, depth, offsets = (Q(1, 5), Q(3, 5), Q(2, 5)), Q(643), (Q(1, 17), Q(0), Q(-1, 19))
        bias = (-v.BETA, Q(0), v.BETA)
        noise = tuple(Q((-1, 0, 1)[index % 3], 200) for index in range(432))
        predicted = v._baseline.independent_rgb('partial', depth, offsets, colors, Q(80))
        image = tuple(min(Q(1), max(Q(0), value + noise[index] + bias[index // 144])) for index, value in enumerate(predicted))
        public = {'baseline': '80', 'radius': '1/5', 'clipped': True, 'observation': list(map(v.encoded, image))}
        fact = {'baseline': '80', 'colors': list(map(v.encoded, colors)), 'truth': v.encoded(depth), 'rear_branch': 'partial',
                'actual_offsets': list(map(v.encoded, offsets)), 'image_noise': list(map(v.encoded, noise)),
                'actual_view_bias': list(map(v.encoded, bias))}
        return public, fact

    def test_registered_q_pattern_and_spread(self):
        colors = (Q(1, 5), Q(3, 5), Q(2, 5))
        darkness = v.variation_darkness(colors, Q(1, 10))
        self.assertEqual(darkness, (Q(39, 50), Q(2, 5), Q(16, 25), Q(4, 5), Q(11, 25), Q(14, 25),
                                   Q(41, 50), Q(9, 25), Q(3, 5)))
        self.assertEqual(v.variation_darkness(colors, 0), tuple(1 - color for color in colors) * 3)
        with self.assertRaises(v.VerificationError):
            v.variation_darkness(colors, Q(1, 5))

    def test_forward_zero_identity_and_nonzero_changes_same_noise_pose(self):
        public, fact = self.source()
        zero = v.replay_forward(public, fact, 0)
        changed = v.replay_forward(public, fact, Q(1, 10))
        self.assertEqual(zero['observation'], public['observation'])
        self.assertNotEqual(changed['observation'], zero['observation'])
        self.assertLessEqual(v.exact(changed['material_spread']), Q(1, 10))
        self.assertEqual(len(changed['darkness9']), 9)
        for value in changed['observation']:
            self.assertTrue(0 <= v.exact(value) <= 1)

    def test_point_image_shared_embedding_matches_frozen_geometry(self):
        public, fact = self.source()
        self.assertEqual(v.independent_rgb_per_view(fact['rear_branch'], fact['truth'], fact['actual_offsets'], fact['colors'] * 3),
                         v._baseline.independent_rgb(fact['rear_branch'], fact['truth'], fact['actual_offsets'], fact['colors'], Q(80)))
        for invalid in ([v.encoded(0)] * 3, [0.1] * 9, [True] * 9):
            with self.assertRaises(v.VerificationError):
                v.independent_rgb_per_view('partial', fact['truth'], fact['actual_offsets'], invalid)


class ArmBridgeAndTamperingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public, cls.arm = fixture_arm()
        cls.checked = v.verify_arm(cls.arm, cls.public, 'joint', 'bounded', 'manual_bounded')

    def check_branch(self, arm):
        geometry = lambda branch, radius, low, high: v.independent_geometry(branch, radius, low, high, Q(80))
        return v.verify_bounded_branch(arm['branches']['partial'], self.public['observation'], 'joint', 'partial',
                                       geometry, arm['calibration'], Q(1, 10), 'manual_bounded')

    def test_bounded_hand_fixture_accepted(self):
        self.assertIsNotNone(self.checked['union']['hull'])
        self.assertTrue(self.checked['union']['hull'][0] <= Q(643) <= self.checked['union']['hull'][1])
        self.assertGreater(self.checked['counts']['exact_updates'], 0)

    def test_zero_one_endpoint_schema_bridges_without_new_solver_import(self):
        for name in ('strict_shared', 'independent'):
            public, arm = fixture_arm(name=name, conflict=True)
            checked = v.verify_arm(arm, public, 'joint', name, 'manual_endpoint')
            self.assertIsNone(checked['union']['hull'])
            changed = deepcopy(arm)
            changed['drift_bound'] = v.encoded(Q(1, 10))
            with self.assertRaises(v.VerificationError):
                v.verify_arm(changed, public, 'joint', name, 'bad_endpoint_delta')

    def test_signed_certificate_step_row_and_endpoint_tampering(self):
        for damage in ('row', 'side', 'value'):
            changed = deepcopy(self.arm)
            node = next(node for node in changed['branches']['partial']['trace'] if node['status'] == 'excluded_shared')
            if damage == 'row':
                node['certificate']['contradiction']['row'] += 1
            elif node['certificate']['steps']:
                step = node['certificate']['steps'][0]
                if damage == 'side':
                    step['side'] = 'lower' if step['side'] == 'upper' else 'upper'
                else:
                    step['value'] = v.encoded(v.exact(step['value']) + Q(1, 1000))
            else:
                node['certificate']['steps'].append({'round': 0, 'row': 288, 'variable': 0, 'side': 'lower', 'value': v.encoded(Q(1, 2))})
            with self.subTest(damage=damage), self.assertRaises(v.VerificationError):
                self.check_branch(changed)

    def test_nine_material_bias_scope_and_tree_tampering(self):
        for damage in ('dimension', 'bias', 'scope', 'split'):
            changed = deepcopy(self.arm)
            outer = changed['branches']['partial']
            node = next(node for node in outer['trace'] if node['status'] in ('split', 'retained_tolerance'))
            if damage == 'dimension':
                node['darkness_bounds'] = node['darkness_bounds'][:3]
            elif damage == 'bias':
                node['t_bounds'][0] = [v.encoded(0), v.encoded(0)]
            elif damage == 'scope':
                outer['scope'] = v._appearance.GROUP_SCOPE
            else:
                split = next(node for node in outer['trace'] if node['status'] == 'split')
                split['children'] = list(reversed(split['children']))
            with self.subTest(damage=damage), self.assertRaises(v.VerificationError):
                self.check_branch(changed)

    def test_exact_target_and_no_new_solver_runner_generator_imports(self):
        self.assertIsNone(v.check_target())
        self.assertEqual(v._appearance.ROOT, v.APPEARANCE)
        tree = ast.parse((v.ROOT / 'verify_variation.py').read_text())
        prohibited = {'variation_solver', 'run_variation', 'prepare_variation', 'execute_variation'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name not in prohibited for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, prohibited)


class PopulationActionCoverageAndSourceTests(unittest.TestCase):
    def population(self):
        observations = {f'p{i:03d}b80e{variation}': {'variation_index': variation,
                        'unit_id': 'appearance_eta_0' if variation == 0 else 'appearance_eta_1_10'}
                        for variation in range(2) for i in range(1, 19)}
        tasks = {f'{oid}s{state}': {'task_id': f'{oid}s{state}', 'observation_id': oid, 'incumbent': v.encoded(643)}
                 for oid in observations for state in range(9)}
        rows = [{'task_id': f'{oid}s{state}', 'appearance_model': model, 'method': method,
                 'reused': variation == 0 and model != 'bounded'}
                 for oid, public in observations.items() for variation in [public['variation_index']]
                 for model in v.ARMS for method in v.METHODS for state in range(9)]
        return observations, tasks, rows

    def coverage(self):
        observations, _tasks, _rows = self.population()
        return [{'observation_id': oid, 'appearance_model': model, 'method': method,
                 'variation_index': public['variation_index'], 'material_witness_required': model != 'strict_shared' or public['variation_index'] == 0,
                 'covered': True, 'generating_branch_covered': True, 'incompatible': False,
                 'latent_material_covered': None if model == 'strict_shared' and public['variation_index'] == 1 else True,
                 'latent_bias_covered': True if method == 'joint' else None,
                 'joint_witness_covered': None if model == 'strict_shared' and public['variation_index'] == 1 else True}
                for oid, public in observations.items() for model in v.ARMS for method in v.METHODS]

    def test_complete1944_population_reuse_identity_and_duplicates(self):
        public, tasks, rows = self.population()
        self.assertEqual(len(v.verify_action_population(rows, public, tasks)), 1944)
        for changed in (rows[:-1], rows[:-1] + rows[:1], rows[1:] + rows[:1]):
            with self.assertRaises(v.VerificationError):
                v.verify_action_population(changed, public, tasks)
        changed = deepcopy(rows)
        changed[0]['reused'] = False
        with self.assertRaises(v.VerificationError):
            v.verify_action_population(changed, public, tasks)

    def test_registered_actions_keep_identities_and_mismatch_loss_preserved(self):
        public, tasks, _rows = self.population()
        records = [{'task_id': task['task_id'], 'unit_id': public[task['observation_id']]['unit_id'], 'method': 'keep',
                    'appearance_model': 'keep', 'reused': True, 'output': task['incumbent'], 'gain_lower': v.encoded(0),
                    'status': 'KEEP', 'reason': 'identity', 'branch': 'keep'} for task in tasks.values()]
        v.verify_keep_population(records, public, tasks)
        for changed in (records[:-1], records[:-1] + records[:1]):
            with self.assertRaises(v.VerificationError):
                v.verify_keep_population(changed, public, tasks)
        task = {'task_id': 'manual', 'base_task_id': 'manual_base', 'observation_id': 'manual_image', 'incumbent': v.encoded(645)}
        fact = {'truth': v.encoded(643), 'pair_id': 'manual_pair', 'unit_id': 'appearance_eta_1_10', 'baseline': '80', 'case': 'manual_case'}
        row = v._appearance.make_result(task, fact, {'output': v.encoded(647), 'method': 'joint', 'status': 'MOVE'}, {'hull': (Q(646), Q(648))}, 'strict_shared')
        self.assertEqual(row['actual_gain_mm2'], '-12')
        self.assertTrue(all(cell['harmed'] == 1 for cell in v._appearance.aggregate_results([row])))

    def test_coverage_applicability_preserves_strict_nonzero_failure(self):
        records = self.coverage()
        v.validate_required_coverage(records)
        for row in records:
            if row['appearance_model'] == 'strict_shared' and row['variation_index'] == 1:
                row.update(covered=False, generating_branch_covered=False, incompatible=True,
                           latent_bias_covered=False if row['method'] == 'joint' else None)
        v.validate_required_coverage(records)
        for model, variation in (('bounded', 0), ('bounded', 1), ('independent', 0), ('independent', 1), ('strict_shared', 0)):
            for field in ('covered', 'generating_branch_covered', 'latent_material_covered', 'joint_witness_covered', 'latent_bias_covered'):
                changed = deepcopy(records)
                row = next(row for row in changed if row['appearance_model'] == model and row['variation_index'] == variation and row['method'] == 'joint')
                row[field] = False
                with self.subTest(model=model, variation=variation, field=field), self.assertRaises(v.VerificationError):
                    v.validate_required_coverage(changed)

    def test_material_and_bias_must_share_one_true_depth_leaf(self):
        public = {'observation_id': 'manual', 'base_observation_id': 'manual_base', 'variation_index': 1, 'unit_id': 'appearance_eta_1_10'}
        fact = {'truth': v.encoded(643), 'rear_branch': 'partial', 'view_colors': [v.encoded(Q(1, 2))] * 9,
                'colors': [v.encoded(Q(1, 2))] * 3, 'actual_view_bias': list(map(v.encoded, (-v.BETA, Q(0), v.BETA)))}
        exact_materials = [[v.encoded(Q(1, 2)), v.encoded(Q(1, 2))]] * 9
        exact_bias = [[v.encoded(t), v.encoded(t)] for t in (2 * v.BETA, v.BETA, Q(0))]
        common = {'lo': v.encoded(642), 'hi': v.encoded(644)}
        nodes = [{**common, 'darkness_bounds': exact_materials, 't_bounds': [[v.encoded(0), v.encoded(0)]] * 3},
                 {**common, 'darkness_bounds': [[v.encoded(0), v.encoded(0)]] * 9, 't_bounds': exact_bias}]
        checked = {'union': {'hull': (Q(642), Q(644))},
                   'branches': {name: {'retained': nodes if name == 'partial' else []} for name in v.BRANCHES}}
        coverage = v.make_coverage(public, fact, 'joint', checked, 'bounded')
        self.assertTrue(coverage['latent_material_covered'])
        self.assertTrue(coverage['latent_bias_covered'])
        self.assertFalse(coverage['joint_witness_covered'])
        records = self.coverage()
        row = next(row for row in records if row['appearance_model'] == 'bounded' and row['variation_index'] == 1 and row['method'] == 'joint')
        row['joint_witness_covered'] = coverage['joint_witness_covered']
        with self.assertRaises(v.VerificationError):
            v.validate_required_coverage(records)

    def test_registered_keep_checks_cover_only_applicable_true_initials_and_unchanged_controls(self):
        tasks = {'manual': {'incumbent': v.encoded(643)}}
        facts = {'manual_image': {'variation_index': 1, 'case': 'manual', 'image_unchanged': False}}
        row = {'task_id': 'manual', 'observation_id': 'manual_image', 'method': 'joint', 'condition': 'strict_shared',
               'initial_ae_mm': '0', 'covered': False, 'output': '644', 'status': 'MOVE'}
        v.verify_registered_keeps([row], facts, tasks)
        with self.assertRaises(v.VerificationError):
            v.verify_registered_keeps([{**row, 'covered': True}], facts, tasks)
        facts['manual_image'].update(case='equal_full_negative', image_unchanged=True)
        with self.assertRaises(v.VerificationError):
            v.verify_registered_keeps([row], facts, tasks)

    def test_incomplete_saved_matrix_never_parses_private(self):
        now = time.time()
        budget = {'start_epoch': now, 'deadline_epoch': now + 3600, 'limit_seconds': 3600, 'scope': v.BUDGET_SCOPE}
        def public_load(path):
            if Path(path).name == 'BUDGET.json':
                return budget
            if Path(path).name == 'decisions.json':
                return []
            return {}
        with patch.object(v, 'check_target', return_value=v.ROOT / 'outputs/manual_no_files'), \
             patch.object(v, 'load', side_effect=public_load), patch.object(v, 'verify_source_lock', return_value={}), \
             patch.object(v, 'verify_inputs', return_value={}), patch.object(v, 'verify_generation'), \
             patch.object(v, 'audit_runner_source', return_value={}), patch.object(v, 'audit_executor_source', return_value={}), \
             patch.object(v, 'verify_public_population', return_value=({}, {}, {}, {})), \
             patch.object(v, 'verify_private_population') as private:
            with self.assertRaises(v.VerificationError):
                v.verify(v.ROOT / 'outputs/manual_no_files')
            private.assert_not_called()

    def test_actual_runner_saved_readback_keep_and_private_fences(self):
        source = (v.ROOT / 'run_variation.py').read_text()
        self.assertEqual(v.audit_runner_source(source)['status'], 'PASS')
        evaluate = '        evaluate(output,observations,tasks,outers,decisions,keep)'
        gate = '        check_actions(decisions,tasks)'
        changed = source.replace(evaluate + '\n', '').replace(gate, evaluate + '\n' + gate)
        for changed in (changed, source.replace('        check_keep(keep,tasks)', '        pass'),
                        source.replace("decisions=load(output/'decisions.json')", 'decisions=[]'),
                        source.replace("method,DRIFT[model],clipped=p['clipped']", "method,p['eta'],clipped=p['clipped']"),
                        "_leak=load(DATA/'private/truth.json')\n" + source):
            self.assertNotEqual(changed, source)
            with self.assertRaises(v.VerificationError):
                v.audit_runner_source(changed)

    def test_actual_executor_budget_reuse_and_reset_rejection(self):
        source = (v.ROOT / 'execute_variation.py').read_text()
        self.assertEqual(v.audit_executor_source(source)['status'], 'PASS')
        for changed in (source.replace('remaining=deadline-time.time()', 'remaining=runner.LIMIT'),
                        source.replace('deadline=start+runner.LIMIT', 'deadline=time.time()+runner.LIMIT')):
            with self.assertRaises(v.VerificationError):
                v.audit_executor_source(changed)
        now = time.time()
        budget = {'start_epoch': now, 'deadline_epoch': now + 3600, 'limit_seconds': 3600, 'scope': v.BUDGET_SCOPE}
        self.assertEqual(v.verify_budget(budget), now + 3600)
        with patch.object(v.time, 'time', return_value=budget['deadline_epoch']):
            with self.assertRaises(TimeoutError):
                v.verify_budget(budget)

    def test_verifier_public_replay_precedes_its_only_private_fact_gate(self):
        tree = ast.parse((v.ROOT / 'verify_variation.py').read_text())
        verify = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'verify')
        calls = [(node.func.id, node.lineno) for node in ast.walk(verify) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        private = [line for name, line in calls if name == 'verify_private_population']
        self.assertEqual(len(private), 1)
        for name in ('verify_action_population', 'verify_keep_population', 'verify_arm', 'sealed_arm', 'verify_action'):
            self.assertTrue(all(line < private[0] for called, line in calls if called == name))


if __name__ == '__main__':
    unittest.main()
