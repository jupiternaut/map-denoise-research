"""Independent hand fixtures: no formal seed, generator, solver or private read."""
import ast
from copy import deepcopy
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('independent_appearance_verifier_test', Path(__file__).with_name('verify_appearance.py'))
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def fixture_branch(image, method, calibration, branch):
    """Author a complete fixture with the independent contractor's output."""
    initial = [[v.encoded(0), v.encoded(1)] for _ in range(9)]
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
            rows = v.necessary_rows(image, geometry, True, method, calibration['intervals'])
            contraction = v.independent_contraction(rows, initial)
            evaluated += 1
            node.update(rounds=contraction['rounds'], updates=contraction['updates'])
            if contraction['excluded']:
                node.update(status='excluded_shared', group_id='global', certificate=contraction['certificate'])
            else:
                node.update(darkness_bounds=contraction['bounds'][:9],
                            t_bounds=contraction['bounds'][9:] if method == 'joint' else calibration['intervals'])
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
    return {'domain': list(map(v.encoded, v.DOMAIN)), 'epsilon': v.encoded(v.EPSILON), 'beta': v.encoded(v.BETA),
            'clipped': True, 'tolerance': v.encoded(v.TOLERANCE), 'max_boxes': v.MAX_BOXES, 'evaluated_boxes': evaluated,
            'trace': trace, 'retained_ids': [node['id'] for node in retained],
            'hull': None if hull is None else list(map(v.encoded, hull)), 'shared_tests': evaluated,
            'propagation_rounds': v.ROUNDS, 'mode': method, 'rounds': v.ROUNDS, 'budget': v.MAX_BOXES,
            'channel_precheck': 'none', 'initial_bounds': initial, 'scope': v.GROUP_SCOPE,
            'appearance_scope': 'per_view', 'material_variable_names': list(v.MATERIAL_VARIABLE_NAMES)}


def fixture_arm(method='joint'):
    depth, offsets, colors = Q(643), (Q(1, 17), Q(0), Q(-1, 19)), (Q(1, 5), Q(3, 5), Q(2, 5))
    image = v._baseline.independent_rgb('partial', depth, offsets, colors, Q(80))
    biases = (-v.BETA, Q(0), v.BETA)
    image = tuple(min(Q(1), max(Q(0), value + biases[index // 144])) for index, value in enumerate(image))
    geometry = lambda branch, radius, low, high: v.independent_geometry(branch, radius, low, high, Q(80))
    background = v._prior.public_background_mask(geometry)
    calibration = v._prior.calibration_from_public(image, True, background)
    branches = {branch: fixture_branch(image, method, calibration, branch) for branch in v.BRANCHES}
    source = [{'branch': branch, 'id': node['id'], 'lo': node['lo'], 'hi': node['hi']}
              for branch, outer in branches.items() for node in outer['trace'] if node['id'] in outer['retained_ids']]
    intervals = v._arithmetic.merged_intervals((v.exact(node['lo']), v.exact(node['hi'])) for node in source)
    hull = None if not intervals else (intervals[0][0], intervals[-1][1])
    outer = {'domain': list(map(v.encoded, v.DOMAIN)), 'epsilon': v.encoded(v.EPSILON), 'beta': v.encoded(v.BETA),
             'clipped': True, 'tolerance': v.encoded(v.TOLERANCE), 'max_boxes_per_branch': v.MAX_BOXES,
             'evaluated_boxes': sum(branch['evaluated_boxes'] for branch in branches.values()), 'unresolved_budget_leaves': 0,
             'source_leaves': source, 'retained_intervals': [list(map(v.encoded, interval)) for interval in intervals],
             'hull': None if hull is None else list(map(v.encoded, hull)),
             'retained_branches': [branch for branch, value in branches.items() if value['hull'] is not None],
             'mode': method, 'rounds': v.ROUNDS, 'budget': v.MAX_BOXES,
             'scope': 'union_of_branch_depth_outer_leaves_then_convex_hull', 'channel_precheck': 'none'}
    arm = {'branches': branches, 'outer': outer, 'branch_seconds': {branch: 0 for branch in v.BRANCHES},
           'outer_seconds': 0, 'preparation_seconds': 0, 'background_mask': background,
           'calibration_intervals': calibration['intervals'], 'calibration': calibration,
           'appearance_scope': 'per_view', 'material_variable_names': list(v.MATERIAL_VARIABLE_NAMES)}
    public = {'observation_id': 'fixture_nonformal', 'unit_id': 'baseline_80', 'baseline': '80', 'radius': '1/5',
              'clipped': True, 'observation': list(map(v.encoded, image))}
    private = {'truth': v.encoded(depth), 'colors': list(map(v.encoded, colors)), 'rear_branch': 'partial',
               'actual_view_bias': list(map(v.encoded, biases)), 'seed': 0, 'case': 'manual_nonformal'}
    return public, arm, private


class IndependentRowAndContractorTests(unittest.TestCase):
    def setUp(self):
        self.image = (Q(99, 100),) * 432
        self.geometry = ((Q(1, 10), Q(1, 5), Q(1, 4), Q(1, 3), Q(1, 6), Q(1, 2)),) * 144
        self.calibration = [[v.encoded(Q(view, 1000)), v.encoded(Q(view + 10, 1000))] for view in range(3)]

    def test_global_three_six_rows_match_both_frozen_independent_verifiers(self):
        for method in v.METHODS:
            own = v.necessary_rows(self.image, self.geometry, True, method, self.calibration, appearance_scope='global_shared')
            self.assertEqual(own, v._prior.necessary_rows(self.image, self.geometry, True, method, self.calibration))
            self.assertEqual(own, v._ablation.masked_rows(self.image, self.geometry, True, method, self.calibration, 'none')[0])
            initial = [[v.encoded(0), v.encoded(1)]] * 3 + (self.calibration if method == 'joint' else [])
            self.assertEqual(v.independent_contraction(own, initial), v._prior.independent_contraction(own, initial))

    def test_independent_three_six_contractor_matches_all_old_semantics(self):
        for dimension in (3, 6):
            initial = [[v.encoded(0), v.encoded(1)] for _ in range(dimension)]
            for rounds in (0, 1, 3, 8):
                for target in (Q(0), Q(1, 7), Q(6, 7), Q(2)):
                    rowsets = [(), (('upper', (Q(1),) * dimension, target),),
                               (('lower', (Q(1),) * dimension, target),),
                               (('upper', (Q(1),) + (Q(0),) * (dimension - 1), Q(1, 3)),
                                ('lower', (Q(1),) + (Q(0),) * (dimension - 1), Q(2, 3)))]
                    for rows in rowsets:
                        self.assertEqual(v.independent_contraction(rows, initial, rounds),
                                         v._prior.independent_contraction(rows, initial, rounds))

    def test_perview_row_slots_and_bias_endpoint_subtraction(self):
        for method in v.METHODS:
            rows = v.necessary_rows(self.image, self.geometry, True, method, self.calibration)
            self.assertEqual(len(rows), 288)
            for pixel in (0, 47, 48, 95, 96, 143):
                view = pixel // 48
                for index, (side, coefficients, bound) in enumerate(rows[2 * pixel:2 * pixel + 2]):
                    expected = (Q(0),) * (3 * view) + self.geometry[pixel][index::2] + (Q(0),) * (6 - 3 * view)
                    if method == 'joint':
                        expected += tuple(Q(int(i == view)) for i in range(3))
                        self.assertEqual(bound, Q(3, 100) if side == 'upper' else Q(1, 100))
                    else:
                        self.assertEqual(bound, (Q(3, 100) - Q(view, 1000)) if side == 'upper'
                                         else (Q(1, 100) - Q(view + 10, 1000)))
                    self.assertEqual(coefficients, expected)
                    self.assertEqual(len(coefficients), 12 if method == 'joint' else 9)

    def test_true_shared_material_embedding_preserves_every_row_value(self):
        materials, t = (Q(1, 8), Q(3, 8), Q(7, 8)), (Q(1, 100), Q(1, 200), Q(0))
        for method in v.METHODS:
            shared = v.necessary_rows(self.image, self.geometry, True, method, self.calibration, appearance_scope='global_shared')
            view = v.necessary_rows(self.image, self.geometry, True, method, self.calibration)
            original = materials + (t if method == 'joint' else ())
            embedded = materials * 3 + (t if method == 'joint' else ())
            for old, new in zip(shared, view):
                self.assertEqual((old[0], old[2]), (new[0], new[2]))
                self.assertEqual(sum(a * x for a, x in zip(old[1], original)), sum(a * x for a, x in zip(new[1], embedded)))

    def test_nine_twelve_variable_index_and_crossing_certificates(self):
        for dimension in (9, 12):
            initial = [[v.encoded(0), v.encoded(1)] for _ in range(dimension)]
            coefficients = (Q(0),) * (dimension - 1) + (Q(1),)
            rows = (('upper', coefficients, Q(1, 3)), ('lower', coefficients, Q(2, 3)))
            result = v.independent_contraction(rows, initial)
            self.assertTrue(result['excluded'])
            self.assertEqual(result['updates'], 1)
            self.assertEqual(result['certificate']['steps'][0]['variable'], dimension - 1)
            self.assertEqual(result['certificate']['contradiction'],
                             {'kind': 'row', 'round': 0, 'row': 1, 'total': v.encoded(Q(1, 3)), 'bound': v.encoded(Q(2, 3))})
        with self.assertRaises(v.VerificationError):
            v.independent_contraction([], [[v.encoded(0), v.encoded(1)]] * 13)

    def test_clipped_and_exact_encoded_boundary_rows(self):
        for value, side in ((Q(0), 'lower'), (Q(1), 'upper')):
            rows = v.necessary_rows((value,) * 432, self.geometry, True, 'joint', self.calibration)
            self.assertEqual(len(rows), 144)
            self.assertTrue(all(row[0] == side for row in rows))
        for invalid in (0.99, True, {'numerator': 198, 'denominator': 200}, {'numerator': 1, 'denominator': 0}):
            changed = list(map(v.encoded, self.image))
            changed[0] = invalid
            with self.assertRaises(v.VerificationError):
                v.pixel_bands(changed, True)
        self.assertEqual(v.pixel_bands(list(map(v.encoded, self.image)), True), v.pixel_bands(self.image, True))


class IndependentArmTamperingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public, cls.arm, cls.private = fixture_arm()
        cls.checked = v.verify_appearance_arm(cls.arm, cls.public, 'joint', 'nonformal_fixture')

    def check_branch(self, arm, branch='partial'):
        geometry = lambda name, radius, low, high: v.independent_geometry(name, radius, low, high, Q(80))
        return v.verify_branch(arm['branches'][branch], self.public['observation'], True, 'joint', branch,
                               geometry, arm['calibration'], 'nonformal_fixture')

    def test_nonformal_arm_and_repeated_nine_material_coverage(self):
        self.assertIsNotNone(self.checked['union']['hull'])
        self.assertGreater(self.checked['counts']['exact_updates'], 0)
        self.assertGreater(self.checked['counts']['contradiction_certificates'], 0)
        coverage = v.make_coverage(self.public, self.private, 'joint', self.checked['union'], self.checked['branches'])
        self.assertTrue(coverage['generating_branch_covered'])
        self.assertTrue(coverage['latent_material_covered'])
        self.assertTrue(coverage['latent_bias_covered'])
        altered = deepcopy(self.checked['branches'])
        for node in altered['partial']['retained']:
            node['darkness_bounds'] = node['darkness_bounds'][:3]
        self.assertFalse(v.make_coverage(self.public, self.private, 'joint', self.checked['union'], altered)['latent_material_covered'])

    def test_required_generating_branch_later_materials_and_joint_bias_coverage(self):
        good = v.make_coverage(self.public, self.private, 'joint', self.checked['union'], self.checked['branches'])
        def complete(row):
            return [{**row, 'observation_id': f'manual{i:02d}', 'method': method,
                     'latent_bias_covered': row['latent_bias_covered'] if method == 'joint' else None}
                    for i in range(18) for method in v.METHODS]
        v.validate_required_coverage(complete(good))
        for damage in ('generating_branch', 'later_materials', 'joint_bias'):
            branches = deepcopy(self.checked['branches'])
            if damage == 'generating_branch':
                branches['partial']['retained'] = []
            else:
                for node in branches['partial']['retained']:
                    if damage == 'later_materials':
                        node['darkness_bounds'][8] = [v.encoded(0), v.encoded(0)]
                    else:
                        node['t_bounds'][0] = [v.encoded(0), v.encoded(0)]
            damaged = v.make_coverage(self.public, self.private, 'joint', self.checked['union'], branches)
            with self.subTest(damage=damage), self.assertRaises(v.VerificationError):
                v.validate_required_coverage(complete(damaged))

    def test_scope_names_and_initial_material_dimension_tampering(self):
        for field, value in (('appearance_scope', 'global_shared'), ('scope', v._prior.GROUP_SCOPE),
                             ('material_variable_names', list(reversed(v.MATERIAL_VARIABLE_NAMES))),
                             ('initial_bounds', self.arm['branches']['partial']['initial_bounds'][:6])):
            changed = deepcopy(self.arm)
            changed['branches']['partial'][field] = value
            with self.subTest(field=field), self.assertRaises(v.VerificationError):
                self.check_branch(changed)
        changed = deepcopy(self.arm)
        changed['material_variable_names'] = ['d0', 'd1', 'd2']
        with self.assertRaises(v.VerificationError):
            v.verify_appearance_arm(changed, self.public, 'joint', 'bad_arm_names')

    def test_certificate_row_steps_and_material_bound_tampering(self):
        for modification in ('row', 'steps', 'bounds', 'updates'):
            changed = deepcopy(self.arm)
            outer = changed['branches']['partial']
            if modification in ('row', 'steps'):
                node = next(node for node in outer['trace'] if node['status'] == 'excluded_shared')
                if modification == 'row':
                    node['certificate']['contradiction']['row'] += 1
                else:
                    node['certificate']['steps'].append({'round': 0, 'row': 0, 'variable': 8,
                                                        'side': 'upper', 'value': v.encoded(Q(1, 3))})
            else:
                node = next(node for node in outer['trace'] if node['status'] in ('split', 'retained_tolerance'))
                if modification == 'bounds':
                    node['darkness_bounds'][8] = [v.encoded(0), v.encoded(0)]
                else:
                    node['updates'] += 1
            with self.subTest(modification=modification), self.assertRaises(v.VerificationError):
                self.check_branch(changed)

    def test_tree_depth_split_and_exact_encoding_tampering(self):
        for modification in ('split', 'clip', 'encoding'):
            changed = deepcopy(self.arm)
            outer = changed['branches']['partial']
            if modification == 'split':
                node = next(node for node in outer['trace'] if node['status'] == 'split')
                node['children'] = list(reversed(node['children']))
            elif modification == 'clip':
                outer['clipped'] = False
            else:
                outer['trace'][0]['lo'] = {'numerator': 1080, 'denominator': 2}
            with self.subTest(modification=modification), self.assertRaises(v.VerificationError):
                self.check_branch(changed)

    def test_union_hull_provenance_and_counter_tampering(self):
        for modification in ('hull', 'intervals', 'source', 'count'):
            outer = deepcopy(self.arm['outer'])
            if modification == 'hull':
                outer['hull'][0] = v.encoded(540)
            elif modification == 'intervals':
                outer['retained_intervals'] = []
            elif modification == 'source':
                outer['source_leaves'][0]['branch'] = 'full'
            else:
                outer['evaluated_boxes'] += 1
            with self.subTest(modification=modification), self.assertRaises(v.VerificationError):
                v._prior.verify_union(outer, self.checked['branches'], 'joint', True, 'bad_union')


class ActionsMetricsAndGatesTests(unittest.TestCase):
    def setUp(self):
        self.task = {'task_id': 'fixture', 'incumbent': v.encoded(606)}
        self.public = {'unit_id': 'baseline_80'}
        self.union = {'hull': (Q(600), Q(602))}

    def decision(self, hull=(Q(600), Q(602)), incumbent=Q(606)):
        action = v._arithmetic.hybrid_action(hull, incumbent)
        action = {key: v.encoded(value) if key in ('output', 'gain_lower') and value is not None else value
                  for key, value in action.items()}
        return {'task_id': 'fixture', 'unit_id': 'baseline_80', 'method': 'joint', 'appearance_scope': 'per_view', 'reused': False, **action}

    def test_action_and_every_action_field_tampering(self):
        good = self.decision()
        self.assertEqual(v.verify_action(good, self.task, self.public, self.union), Q(601))
        for field, value in (('status', 'KEEP'), ('output', v.encoded(600)), ('gain_lower', v.encoded(1)),
                             ('reason', 'identity'), ('branch', 'projection'), ('unit_id', 'baseline_70'),
                             ('reused', True), ('appearance_scope', 'global_shared')):
            with self.subTest(field=field), self.assertRaises(v.VerificationError):
                v.verify_action({**good, field: value}, self.task, self.public, self.union)
        for invalid in ({'numerator': 1202, 'denominator': 2}, 601.0, True):
            with self.assertRaises(v.VerificationError):
                v.verify_action({**good, 'output': invalid}, self.task, self.public, self.union)

    def test_keep_and_empty_outer_are_registered_actions(self):
        for hull, incumbent in (((Q(600), Q(602)), Q(601)), (None, Q(606))):
            decision = self.decision(hull, incumbent)
            task = {**self.task, 'incumbent': v.encoded(incumbent)}
            self.assertEqual(v.verify_action(decision, task, self.public, {'hull': hull}), incumbent)

    def test_complete324_population_and_duplicates_scope_origin(self):
        observations = {f'p{i:03d}b80': {} for i in range(1, 19)}
        tasks = {f'{oid}s{state}': {} for oid in observations for state in range(9)}
        rows = [{'task_id': f'{oid}s{state}', 'method': method, 'appearance_scope': 'per_view', 'reused': False}
                for oid in observations for method in v.METHODS for state in range(9)]
        self.assertEqual(len(v.verify_action_population(rows, observations, tasks)), 324)
        for altered in (rows[:-1], rows[:-1] + rows[:1], [*rows[1:], rows[0]]):
            with self.assertRaises(v.VerificationError):
                v.verify_action_population(altered, observations, tasks)
        for field, value in (('appearance_scope', 'global_shared'), ('reused', 0)):
            altered = deepcopy(rows)
            altered[0][field] = value
            with self.assertRaises(v.VerificationError):
                v.verify_action_population(altered, observations, tasks)

    def test_missing_persisted_actions_rejected_before_private_parse(self):
        now = time.time()
        output = v.ROOT / 'outputs/manual_nonformal_no_files'
        budget = {'start_epoch': now, 'deadline_epoch': now + 3600, 'limit_seconds': 3600, 'scope': v.BUDGET_SCOPE}
        def public_only_load(path):
            if Path(path).name == 'BUDGET.json':
                return budget
            if Path(path).name == 'decisions.json':
                return []
            return {}
        with patch.object(v, 'check_target', return_value=output), patch.object(v, 'load', side_effect=public_only_load), \
             patch.object(v, 'verify_source_lock', return_value={}), patch.object(v, 'verify_inputs', return_value={}), \
             patch.object(v, 'audit_runner_source', return_value={}), patch.object(v, 'verify_public_population', return_value=({}, {})), \
             patch.object(v, 'verify_private_population') as private:
            with self.assertRaises(v.VerificationError):
                v.verify(output)
            private.assert_not_called()

    def test_negative_errors_preserved_and_finite_nesting_reported(self):
        task = {'task_id': 'fixture', 'base_task_id': 'pair_fixture', 'observation_id': 'image_fixture', 'incumbent': v.encoded(645)}
        private = {'truth': v.encoded(643), 'pair_id': 'pair_fixture', 'unit_id': 'baseline_80', 'baseline': '80', 'case': 'manual_nonformal'}
        row = v.make_result(task, private, {'output': v.encoded(647), 'method': 'joint', 'status': 'MOVE'}, self.union, 'per_view')
        self.assertEqual(row['actual_gain_mm2'], '-12')
        self.assertTrue(all(cell['harmed'] == 1 for cell in v.aggregate_results([row])))
        self.assertTrue(v.interval_subset([(Q(600), Q(602))], [(Q(599), Q(603))]))
        self.assertFalse(v.interval_subset([(Q(599), Q(603))], [(Q(600), Q(602))]))
        self.assertFalse(v.interval_subset([(Q(599), Q(603))], [(Q(599), Q(600)), (Q(602), Q(603))]))


class SourceIdentityAndBudgetTests(unittest.TestCase):
    def test_absolute3600_window_identity_and_expiry(self):
        now = time.time()
        budget = {'start_epoch': now, 'deadline_epoch': now + 3600, 'limit_seconds': 3600, 'scope': v.BUDGET_SCOPE}
        self.assertEqual(v.verify_budget(budget), now + 3600)
        for changed in ({**budget, 'limit_seconds': 1800}, {**budget, 'deadline_epoch': now + 3601}, {**budget, 'scope': 'reset'}):
            with self.assertRaises(v.VerificationError):
                v.verify_budget(changed)
        with patch.object(v.time, 'time', return_value=budget['deadline_epoch']):
            with self.assertRaises(TimeoutError):
                v.verify_budget(budget)

    def test_actual_runner_source_order_and_mutated_gate_rejection(self):
        source = (v.ROOT / 'run_appearance.py').read_text()
        self.assertEqual(v.audit_runner_source(source)['status'], 'PASS')
        gate = '        check_actions(decisions,tasks)'
        evaluation = '        evaluate(output,observations,tasks,outers,full_outers,decisions)'
        altered = source.replace(evaluation + '\n', '').replace(gate, evaluation + '\n' + gate)
        self.assertNotEqual(altered, source)
        with self.assertRaises(v.VerificationError):
            v.audit_runner_source(altered)
        for altered in (source.replace('        decisions = load(output/\'decisions.json\')', '        decisions = []'),
                        source.replace('    if (len(keys)!=len(expected)', '    if (False'),
                        source.replace("appearance_scope='per_view',clipped=public['clipped']", "appearance_scope='global_shared',clipped=public['clipped']"),
                        "_leak = load(DATA/'private/truth.json')\n" + source):
            self.assertNotEqual(altered, source)
            with self.assertRaises(v.VerificationError):
                v.audit_runner_source(altered)

    def test_verifier_uses_only_frozen_independent_dependency_paths(self):
        source = (v.ROOT / 'verify_appearance.py').read_text()
        tree = ast.parse(source)
        prohibited = {'appearance_solver', 'run_appearance', 'baseline_geometry', 'baseline_solver', 'ablation_solver', 'run_ablation'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name not in prohibited for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, prohibited)
        self.assertEqual(v._baseline.ROOT, v.RECOVERY)
        self.assertEqual(v._ablation.ROOT, v.ABLATION)
        self.assertEqual(v.ROOT, v.EXPECTED_ROOT)
        self.assertEqual(v.check_target(), None)

    def test_verifier_private_gate_follows_complete_arm_action_replay(self):
        tree = ast.parse((v.ROOT / 'verify_appearance.py').read_text())
        verify = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'verify')
        calls = [(node.func.id, node.lineno) for node in ast.walk(verify) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        gate = next(line for name, line in calls if name == 'verify_private_population')
        self.assertEqual(sum(name == 'verify_private_population' for name, _line in calls), 1)
        for name in ('verify_action_population', 'verify_appearance_arm', 'verify_action'):
            self.assertTrue(all(line < gate for called, line in calls if called == name))


if __name__ == '__main__':
    unittest.main()
