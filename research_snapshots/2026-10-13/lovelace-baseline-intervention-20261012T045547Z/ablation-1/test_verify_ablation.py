"""Independent nonformal fixtures; never render the formal seed or read truth."""
import ast
from copy import deepcopy
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('independent_ablation_verifier_test', Path(__file__).with_name('verify_ablation.py'))
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def fixture_branch(image, method, mask, calibration, baseline, branch):
    """Author fixture traces without importing the new inference code."""
    initial = [[v.encoded(0), v.encoded(1)] for _ in range(3)]
    if method == 'joint':
        initial.extend(calibration['intervals'])
    queue = [(None, *v.DOMAIN)]
    trace, retained, evaluated = [], [], 0
    for ident, (parent, low, high) in enumerate(queue):
        node = {'id': ident, 'parent': parent, 'lo': v.encoded(low), 'hi': v.encoded(high)}
        if calibration['excluded']:
            node.update(status='excluded_calibration', certificate=calibration['certificate'])
        else:
            geometry = v.independent_geometry(branch, v.RADIUS, low, high, baseline)
            rows, _ = v.masked_rows(image, geometry, True, method, calibration['intervals'], mask)
            contraction = v.independent_contraction(rows, initial)
            evaluated += 1
            node.update(rounds=contraction['rounds'], updates=contraction['updates'])
            if contraction['excluded']:
                node.update(status='excluded_shared', group_id='global', certificate=contraction['certificate'])
            else:
                node.update(darkness_bounds=contraction['bounds'][:3],
                            t_bounds=contraction['bounds'][3:] if method == 'joint' else calibration['intervals'])
                if high - low <= v.TOLERANCE:
                    node['status'] = 'retained_tolerance'
                    retained.append(node)
                else:
                    midpoint = (low + high) / 2
                    node.update(status='split', children=[len(queue), len(queue) + 1])
                    queue.extend(((ident, low, midpoint), (ident, midpoint, high)))
        trace.append(node)
    intervals = [(v.exact(node['lo']), v.exact(node['hi'])) for node in retained]
    hull = None if not intervals else (min(lo for lo, _ in intervals), max(hi for _, hi in intervals))
    return {'domain': list(map(v.encoded, v.DOMAIN)), 'epsilon': v.encoded(v.EPSILON), 'beta': v.encoded(v.BETA),
            'clipped': True, 'tolerance': v.encoded(v.TOLERANCE), 'max_boxes': v.MAX_BOXES, 'evaluated_boxes': evaluated,
            'trace': trace, 'retained_ids': [node['id'] for node in retained],
            'hull': None if hull is None else list(map(v.encoded, hull)), 'shared_tests': evaluated,
            'propagation_rounds': v.ROUNDS, 'mode': method, 'rounds': v.ROUNDS, 'budget': v.MAX_BOXES,
            'channel_precheck': 'none', 'initial_bounds': initial, 'scope': v._prior.GROUP_SCOPE}


def fixture_arm(method='joint', mask='drop_F'):
    baseline, depth = Q(80), Q(643)
    image = v._baseline.independent_rgb('partial', depth, (Q(1, 17), Q(0), Q(-1, 19)),
                                        (Q(1, 5), Q(3, 5), Q(2, 5)), baseline)
    biases = (-v.BETA, Q(0), v.BETA)
    image = tuple(min(Q(1), max(Q(0), value + biases[channel // 144])) for channel, value in enumerate(image))
    geometry = lambda branch, radius, low, high: v.independent_geometry(branch, radius, low, high, baseline)
    background = v._prior.public_background_mask(geometry)
    calibration = v._prior.calibration_from_public(image, True, background)
    branches = {branch: fixture_branch(image, method, mask, calibration, baseline, branch) for branch in v.BRANCHES}
    source = [{'branch': branch, 'id': node['id'], 'lo': node['lo'], 'hi': node['hi']}
              for branch, outer in branches.items() for node in outer['trace'] if node['id'] in outer['retained_ids']]
    intervals = v._arithmetic.merged_intervals((v.exact(node['lo']), v.exact(node['hi'])) for node in source)
    hull = None if not intervals else (intervals[0][0], intervals[-1][1])
    outer = {'domain': list(map(v.encoded, v.DOMAIN)), 'epsilon': v.encoded(v.EPSILON), 'beta': v.encoded(v.BETA),
             'clipped': True, 'tolerance': v.encoded(v.TOLERANCE), 'max_boxes_per_branch': v.MAX_BOXES,
             'evaluated_boxes': sum(branch['evaluated_boxes'] for branch in branches.values()), 'unresolved_budget_leaves': 0,
             'source_leaves': source, 'retained_intervals': [list(map(v.encoded, pair)) for pair in intervals],
             'hull': None if hull is None else list(map(v.encoded, hull)),
             'retained_branches': [branch for branch, outer in branches.items() if outer['hull'] is not None],
             'mode': method, 'rounds': v.ROUNDS, 'budget': v.MAX_BOXES,
             'scope': 'union_of_branch_depth_outer_leaves_then_convex_hull', 'channel_precheck': 'none'}
    _, metadata = v.masked_rows(image, geometry('none', v.RADIUS, *v.DOMAIN), True, method, calibration['intervals'], mask)
    arm = {'branches': branches, 'outer': outer, 'branch_seconds': {branch: 0 for branch in v.BRANCHES},
           'outer_seconds': 0, 'preparation_seconds': 0, 'background_mask': background,
           'calibration_intervals': calibration['intervals'], 'calibration': calibration, 'inference_mask': metadata}
    public = {'observation_id': 'fixture_nonformal', 'unit_id': 'baseline_80', 'baseline': '80', 'radius': '1/5',
              'clipped': True, 'observation': list(map(v.encoded, image))}
    return public, arm


class PixelRowTests(unittest.TestCase):
    def setUp(self):
        self.image = (Q(99, 100),) * 432
        self.geometry = ((Q(0), Q(1), Q(0), Q(1), Q(0), Q(1)),) * 144
        self.calibration = [[v.encoded(0), v.encoded(Q(1, 50))] for _ in range(3)]

    def test_fixed_public_masks_and_mirrors(self):
        self.assertEqual(v.mask_pixels('drop_F'), [7 + 8 * row for row in range(6)] + [96 + 8 * row for row in range(6)])
        self.assertEqual(v.mask_pixels('matched_drop'), [6 + 8 * row for row in range(6)] + [97 + 8 * row for row in range(6)])
        self.assertEqual(v.mask_pixels('none'), [])
        self.assertTrue(set(v.mask_pixels('drop_F')).isdisjoint(v.mask_pixels('matched_drop')))
        with self.assertRaises(v.VerificationError):
            v.mask_pixels('best_after_truth')

    def test_no_mask_identity_matches_frozen_independent_rows(self):
        for method in v.METHODS:
            own, metadata = v.masked_rows(self.image, self.geometry, True, method, self.calibration, 'none')
            frozen = v._prior.necessary_rows(self.image, self.geometry, True, method, self.calibration)
            self.assertEqual(own, frozen)
            self.assertEqual(metadata, {'name': 'none', 'dropped_pixels': [], 'dropped_rows': [], 'kept_row_count': 288})

    def test_deleted_rows_retain_original_ids_and_pixel_positions(self):
        _, full = v.masked_rows(self.image, self.geometry, True, 'joint', self.calibration, 'none')
        for mask in v.MASKS:
            kept, metadata = v.masked_rows(self.image, self.geometry, True, 'joint', self.calibration, mask)
            self.assertEqual(len(kept), 264)
            self.assertEqual(len(metadata['dropped_rows']), 24)
            self.assertEqual(full['kept_row_count'] - len(kept), 24)
            for index, row in enumerate(metadata['dropped_rows']):
                pixel = metadata['dropped_pixels'][index // 2]
                self.assertEqual(row, {'row_id': pixel * 2 + index % 2, 'pixel': pixel, 'view': pixel // 48,
                                       'row': pixel % 48 // 8, 'column': pixel % 8,
                                       'side': 'upper' if index % 2 == 0 else 'lower'})

    def test_saturated_pixels_only_drop_the_extant_side(self):
        for value, side in ((Q(1), 'upper'), (Q(0), 'lower')):
            kept, metadata = v.masked_rows((value,) * 432, self.geometry, True, 'calibrated', self.calibration, 'drop_F')
            self.assertEqual(len(kept), 132)
            self.assertEqual(len(metadata['dropped_rows']), 12)
            self.assertTrue(all(row['side'] == side and row['row_id'] == row['pixel'] for row in metadata['dropped_rows']))
        mixed = list(self.image)
        mixed[21:24] = [Q(0)] * 3
        _, metadata = v.masked_rows(mixed, self.geometry, True, 'joint', self.calibration, 'drop_F')
        self.assertEqual(metadata['dropped_rows'][0]['side'], 'lower')
        self.assertEqual(metadata['dropped_rows'][0]['row_id'], 14)
        self.assertEqual(metadata['dropped_rows'][1]['row_id'], 29)

    def test_encoded_boundaries_reject_float_bool_and_noncanonical(self):
        for invalid in (0.99, True, {'numerator': 198, 'denominator': 200}, {'numerator': 1, 'denominator': 0}):
            changed = list(map(v.encoded, self.image))
            changed[0] = invalid
            with self.assertRaises(v.VerificationError):
                v.pixel_bands(changed, True)
        self.assertEqual(v.pixel_bands(list(map(v.encoded, self.image)), True), v.pixel_bands(self.image, True))


class IndependentArmTamperingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public, cls.arm = fixture_arm()

    def check(self, arm):
        return v.verify_masked_arm(arm, self.public, 'joint', 'drop_F', 'nonformal_fixture')

    def test_nonformal_arm_is_independently_accepted(self):
        result = self.check(self.arm)
        self.assertIsNotNone(result['union']['hull'])
        self.assertGreater(result['counts']['exact_updates'], 0)
        self.assertGreater(result['counts']['contradiction_certificates'], 0)

    def test_mask_pixel_and_original_row_id_tampering(self):
        changes = [('dropped_pixels', None), ('row_id', 1), ('pixel', 6), ('view', 1), ('row', 5), ('column', 6), ('side', 'lower')]
        for field, value in changes:
            altered = deepcopy(self.arm)
            if field == 'dropped_pixels':
                altered['inference_mask'][field][0] = 6
            else:
                altered['inference_mask']['dropped_rows'][0][field] = value
            with self.subTest(field=field), self.assertRaises(v.VerificationError):
                self.check(altered)
        altered = deepcopy(self.arm)
        altered['inference_mask']['kept_row_count'] += 1
        with self.assertRaises(v.VerificationError):
            self.check(altered)

    def test_row_id_boolean_cannot_equal_integer_identity(self):
        _, metadata = v.masked_rows((Q(99, 100),) * 432, ((Q(0),) * 6,) * 144, True,
                                     'joint', [[v.encoded(0), v.encoded(Q(1, 50))]] * 3, 'drop_F')
        tampered = deepcopy(metadata)
        tampered['dropped_rows'][0]['view'] = False
        with self.assertRaises(v.VerificationError):
            v.verify_mask_metadata(tampered, metadata, 'bool_identity')

    def test_certificate_row_and_propagation_update_tampering(self):
        for modification in ('row', 'steps'):
            altered = deepcopy(self.arm)
            node = next(node for outer in altered['branches'].values() for node in outer['trace'] if node['status'] == 'excluded_shared')
            if modification == 'row':
                node['certificate']['contradiction']['row'] += 1
            else:
                # The first contradiction can legitimately have no updates;
                # clearing that list would not change the fixture at all.
                node['certificate']['steps'].append({'round': 0, 'row': 0, 'variable': 0,
                                                    'side': 'upper', 'value': v.encoded(Q(1, 3))})
            with self.subTest(modification=modification), self.assertRaises(v.VerificationError):
                self.check(altered)

    def test_hull_union_and_provenance_tampering(self):
        for modification in ('hull', 'retained_intervals', 'source_leaves'):
            altered = deepcopy(self.arm)
            if modification == 'hull':
                altered['outer']['hull'][0] = v.encoded(Q(540))
            elif modification == 'retained_intervals':
                altered['outer'][modification] = []
            else:
                altered['outer'][modification][0]['branch'] = 'full'
            with self.subTest(modification=modification), self.assertRaises(v.VerificationError):
                self.check(altered)

    def test_exact_encoded_certificate_boundary_tampering(self):
        altered = deepcopy(self.arm)
        node = altered['branches']['partial']['trace'][0]
        node['lo'] = {'numerator': 1080, 'denominator': 2}
        with self.assertRaises(v.VerificationError):
            self.check(altered)

    def test_clipping_calibration_and_counter_tampering(self):
        for modification in ('clipped', 'calibration_intervals', 'evaluated_boxes'):
            altered = deepcopy(self.arm)
            if modification == 'clipped':
                altered['branches']['none']['clipped'] = False
            elif modification == 'calibration_intervals':
                altered[modification][0][1] = v.encoded(0)
            else:
                altered['outer'][modification] += 1
            with self.subTest(modification=modification), self.assertRaises(v.VerificationError):
                self.check(altered)


class ActionAndPopulationTests(unittest.TestCase):
    def setUp(self):
        self.task = {'task_id': 'fixture', 'incumbent': v.encoded(Q(606))}
        self.public = {'unit_id': 'baseline_80'}
        self.checked = {'hull': (Q(600), Q(602))}

    def decision(self, selector='hybrid', replay=False):
        action = v.selector_action(self.checked['hull'], self.task['incumbent'], selector)
        action = {key: v.encoded(value) if key in ('output', 'gain_lower') and value is not None else value for key, value in action.items()}
        return {'task_id': 'fixture', 'method': 'joint', 'unit_id': 'baseline_80', 'reused': replay,
                'selector' if replay else 'mask_name': selector if replay else 'drop_F', **action}

    def test_actions_and_every_tampered_action_field(self):
        good = self.decision()
        self.assertEqual(v.verify_action(good, self.task, self.public, self.checked, 'drop_F'), Q(601))
        for field, value in (('status', 'KEEP'), ('output', v.encoded(600)), ('gain_lower', v.encoded(1)),
                             ('reason', 'identity'), ('branch', 'projection'), ('unit_id', 'baseline_70'), ('reused', True)):
            changed = {**good, field: value}
            with self.subTest(field=field), self.assertRaises(v.VerificationError):
                v.verify_action(changed, self.task, self.public, self.checked, 'drop_F')

    def test_encoded_action_boundary_rejects_noncanonical_or_float(self):
        for invalid in ({'numerator': 1202, 'denominator': 2}, 601.0, True):
            with self.assertRaises(v.VerificationError):
                v.verify_action({**self.decision(), 'output': invalid}, self.task, self.public, self.checked, 'drop_F')

    def test_selector_replay_schema_and_projection_nested_truth_property(self):
        for selector in v.SELECTORS:
            v.verify_action(self.decision(selector, True), self.task, self.public, self.checked, selector, True)
        for incumbent in (Q(590), Q(601), Q(607)):
            large = v.selector_action((Q(596), Q(608)), incumbent, 'projection')['output']
            small = v.selector_action((Q(600), Q(604)), incumbent, 'projection')['output']
            for truth in (Q(600), Q(602), Q(604)):
                self.assertLessEqual(abs(small - truth), abs(large - truth))
        self.assertEqual(v.selector_action(None, Q(600), 'midpoint')['branch'], 'incompatible')

    def test_missing_population_is_rejected_before_any_private_work(self):
        for selector in (False, True):
            with self.assertRaises(v.VerificationError):
                v.verify_action_population([], {}, {}, selector)

    def test_duplicate_action_population_is_rejected(self):
        images = {f'p{i:03d}b{b}': {} for i in range(1, 19) for b in (70, 80)}
        tasks = {f'{oid}s{s}': {} for oid in images for s in range(9)}
        rows = [{'task_id': f'{oid}s{s}', 'method': method, 'mask_name': mask, 'reused': False}
                for oid in images for mask in v.MASKS for method in v.METHODS for s in range(9)]
        self.assertEqual(len(v.verify_action_population(rows, images, tasks)), 1296)
        with self.assertRaises(v.VerificationError):
            v.verify_action_population(rows[:-1] + rows[:1], images, tasks)

    def test_negative_scientific_errors_are_preserved(self):
        task = {'task_id': 'fixture', 'base_task_id': 'pair_fixture', 'observation_id': 'image_fixture', 'incumbent': v.encoded(645)}
        private = {'truth': v.encoded(643), 'pair_id': 'pair_fixture', 'unit_id': 'baseline_80', 'baseline': '80', 'case': 'fixture'}
        row = v.make_result(task, private, {'output': v.encoded(647), 'method': 'joint', 'status': 'MOVE'}, self.checked, 'drop_F')
        self.assertEqual(row['actual_gain_mm2'], '-12')
        self.assertEqual(row['ae_mm'], '4')
        summaries = v.aggregate_results([row])
        self.assertTrue(all(cell['harmed'] == 1 for cell in summaries))


class FrozenSourceAndBudgetTests(unittest.TestCase):
    def test_new_absolute_3600_window_and_expiry(self):
        now = time.time()
        budget = {'start_epoch': now, 'deadline_epoch': now + 3600, 'limit_seconds': 3600, 'scope': v.BUDGET_SCOPE}
        self.assertEqual(v.verify_budget(budget), now + 3600)
        for changed in ({**budget, 'limit_seconds': 1800}, {**budget, 'deadline_epoch': now + 3601}, {**budget, 'scope': 'reset'}):
            with self.assertRaises(v.VerificationError):
                v.verify_budget(changed)
        with patch.object(v.time, 'time', return_value=budget['deadline_epoch']):
            with self.assertRaises(TimeoutError):
                v.verify_budget(budget)

    def test_source_audit_and_early_evaluation_rejection(self):
        source = (v.ROOT / 'run_ablation.py').read_text()
        self.assertEqual(v.audit_runner_source(source)['status'], 'PASS')
        gate = '        check_actions(decisions,tasks,MASKS,False)'
        evaluate = '        evaluate(output,observations,tasks,outers,full_outers,decisions,selector_decisions)'
        altered = source.replace(evaluate + '\n', '').replace(gate, evaluate + '\n' + gate)
        self.assertNotEqual(altered, source)
        with self.assertRaises(v.VerificationError):
            v.audit_runner_source(altered)
        altered = source.replace("    observations, tasks = load(DATA / 'observations.json'), load(DATA / 'tasks.json')",
                                 "    _leak = load(DATA / 'private/truth.json')\n    observations, tasks = load(DATA / 'observations.json'), load(DATA / 'tasks.json')")
        with self.assertRaises(v.VerificationError):
            v.audit_runner_source(altered)

    def test_verifier_imports_only_frozen_independent_modules(self):
        tree = ast.parse((v.ROOT / 'verify_ablation.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name not in {'ablation_solver', 'run_ablation', 'baseline_geometry'} for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, {'ablation_solver', 'run_ablation', 'baseline_geometry'})
        self.assertEqual(v._baseline.ROOT, v.RECOVERY)
        self.assertEqual(v.ROOT, v.EXPECTED_ROOT)


if __name__ == '__main__':
    unittest.main()
