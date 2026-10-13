"""A10 handwritten fixtures only; no formal inputs or frozen monkeypatches."""
from copy import deepcopy
from fractions import Fraction as Q
import time
import unittest
from unittest.mock import patch

import tightening_solver as s


def fixture(branch='partial', depth=Q(600)):
    a = s.anchor
    offsets = (Q(-3, 25), Q(0), Q(1, 11))
    darkness = (Q(1, 2),) * 9
    ts, ta = (Q(1, 50), Q(1, 100), Q(0)), (Q(0), Q(1, 100), Q(1, 50))
    raw = a.pose._prediction(depth, branch, offsets, darkness, ts)
    public = {'observation_id': 'hand-fixture', 'radius': a.encode(Q(1, 5)), 'baseline': a.encode(80),
              'clipped': True, 'observation': [a.encode(a._clip(value)) for value in raw]}
    oldarm = {'appearance_model': 'bounded', 'drift_bound': a.encode(Q(1, 10)),
              'calibration_intervals': [[a.encode(t), a.encode(t)] for t in ts]}
    k, noises = Q(1, 200), (Q(-1, 200), Q(0), Q(1, 200))
    readings = tuple((-center - offset - k - error) / 1000 for center, offset, error in
                     zip((-Q(80), Q(0), Q(80)), offsets, noises))
    anchors = {'observation_id': 'hand-fixture', 'camera': {
        'u': list(map(a.encode, readings)), 'Xf': a.encode(0), 'Zf': a.encode(1000),
        'survey_bound': a.encode(Q(1, 100)), 'localization_bound': a.encode(Q(1, 100))},
        'appearance': {'coupons': [a.encode(a._clip(1 - d + a.BETA - ta[i // 3])) for i, d in enumerate(darkness)],
                       'references': [a.encode(Q(1, 2) + a.BETA - t) for t in ta],
                       'bias_bound': a.encode(Q(1, 100)), 'transfer_bound': a.encode(Q(1, 200)),
                       'noise_bound': a.encode(Q(1, 200))}}
    coordinates = list(map(a.encode, darkness + ts + ta))
    return public, oldarm, anchors, coordinates, offsets


def source(intervals=((Q(590), Q(610)),), branch='partial'):
    return {'branch': branch, 'outer_intervals': [[s.encode(lo), s.encode(hi)] for lo, hi in intervals]}


def unknown(*_args, **_kwargs):
    return {'status': 'UNKNOWN', 'certificate': None, 'witness': None, 'lp_status': None,
            'seconds': 0.0, 'lp_seconds': 0.0, 'reason': 'hand_unknown'}


def constant_problem(impossible=True):
    return (('upper', (Q(0),) * 15, Q(-1) if impossible else Q(1)),), ((Q(0), Q(1)),) * 15


def pairs(values):
    return tuple(tuple(map(s.exact, pair)) for pair in values)


def check_partition(test, result):
    nodes = result['trace']
    test.assertEqual([n['id'] for n in nodes], list(range(len(nodes))))
    roots = [n for n in nodes if n['parent'] is None]
    test.assertEqual([n['id'] for n in roots], result['roots'])
    test.assertEqual(tuple((s.exact(n['lo']), s.exact(n['hi'])) for n in roots), pairs(result['source_outer_intervals']))
    calls = 0
    for n in nodes:
        test.assertIn(n['root'], result['roots'])
        if n['status'] == 'split':
            lo, hi = s.exact(n['lo']), s.exact(n['hi']);mid = (lo + hi) / 2
            children = [nodes[i] for i in n['children']]
            test.assertEqual([(s.exact(c['lo']), s.exact(c['hi'])) for c in children], [(lo, mid), (mid, hi)])
            test.assertTrue(all(c['parent'] == n['id'] and c['root'] == n['root'] for c in children))
        else:
            test.assertIsNone(n['children'])
        if n['checks']:
            test.assertEqual([c['tile_id'] for c in n['checks']], list(range(len(result['tiles']))))
            test.assertEqual([c['camera_box'] for c in n['checks']], result['tiles'])
        if n['status'] == 'excluded_certificate':
            test.assertEqual(len(n['checks']), len(result['tiles']))
            test.assertTrue(all(c['classification']['status'] == 'CERTIFIED_INFEASIBLE' for c in n['checks']))
        for c in n['checks']:
            calls += int(c['classification'] is not None)
            test.assertEqual(c['reason'] == 'classified', c['classification'] is not None)
    test.assertEqual(calls, result['classification_calls'])
    retained = [n['id'] for n in nodes if n['status'].startswith('retained_')]
    test.assertEqual(retained, result['retained_ids'])
    actual = s.merge_intervals((nodes[i]['lo'], nodes[i]['hi']) for i in retained)
    test.assertEqual(actual, pairs(result['outer_intervals']))
    for lo, hi in actual:
        test.assertTrue(any(a <= lo <= hi <= b for a, b in pairs(result['source_outer_intervals'])))


class TighteningTests(unittest.TestCase):
    def test_fixed_defaults_and_frozen_source_unchanged(self):
        self.assertEqual(s.MODES, ('depth', 'camera_tiles'))
        self.assertEqual((s.MIN_WIDTH, s.MAX_CLASSIFICATIONS, s.BRANCH_SECONDS), (Q(15, 256), 128, 6.0))
        self.assertEqual((s.baseline.MIN_WIDTH, s.baseline.MAX_NODES, s.baseline.BRANCH_SECONDS), (Q(15, 32), 127, 3.0))
        self.assertEqual(s.baseline.__file__, str(s.OLD_ROOT / 'continuous_solver.py'))

    def test_four_closed_tiles_cover_full_camera_box(self):
        box = ((Q(-1, 5), Q(1, 10)), (Q(0), Q(0)), (Q(-1, 10), Q(1, 5)))
        tiles = s.camera_tiles(box)
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[0], ((Q(-1, 5), Q(-1, 20)), (Q(0), Q(0)), (Q(-1, 10), Q(1, 20))))
        for wl in (Q(0), Q(1, 3), Q(1, 2), Q(2, 3), Q(1)):
            for wr in (Q(0), Q(1, 3), Q(1, 2), Q(2, 3), Q(1)):
                point = (box[0][0] + wl * (box[0][1] - box[0][0]), Q(0), box[2][0] + wr * (box[2][1] - box[2][0]))
                self.assertTrue(any(all(lo <= e <= hi for e, (lo, hi) in zip(point, tile)) for tile in tiles))
        degenerate = ((Q(1, 10), Q(1, 10)), (Q(0), Q(0)), (Q(-1, 10), Q(-1, 10)))
        self.assertEqual(s.camera_tiles(degenerate), (degenerate,) * 4)
        self.assertEqual(s.camera_tiles(None), (None,))

    def test_whole_camera_problem_is_exact_A9_both_identity(self):
        public, arm, anchors, coordinates, _ = fixture()
        _, box = s.anchor.camera_intervals(anchors)
        actual = s.build_problem(public, arm, anchors, 590, 610, 'partial', box)
        self.assertEqual(actual, s.baseline.build_interval_problem(public, arm, anchors, 'both', 590, 610, 'partial'))
        self.assertTrue(s.witness_check(*actual, coordinates))

    def test_handwritten_world_covered_by_its_camera_tile(self):
        for branch in s.BRANCHES:
            public, arm, anchors, coordinates, offsets = fixture(branch)
            _, box = s.anchor.camera_intervals(anchors)
            containing = [tile for tile in s.camera_tiles(box) if all(lo <= e <= hi for e, (lo, hi) in zip(offsets, tile))]
            self.assertTrue(containing)
            for tile in containing:
                for lo, hi in ((590, 610), (Q(59999, 100), Q(60001, 100)), (600, 600)):
                    rows, bounds = s.build_problem(public, arm, anchors, lo, hi, branch, tile)
                    self.assertTrue(s.witness_check(rows, bounds, coordinates))
                    self.assertEqual(len(bounds), 15)

    def test_per_tile_cost_and_no_early_feasible_short_circuit(self):
        public, arm, anchors, _, _ = fixture()
        for mode, n in (('depth', 1), ('camera_tiles', 4)):
            with patch.object(s, 'classify', side_effect=unknown):
                result = s.solve_branch(public, arm, anchors, source(), mode, tolerance=Q(20))
            self.assertEqual(result['classification_calls'], n)
            self.assertEqual(result['processed_nodes'], 1)
            self.assertEqual(result['termination'], 'boundaries_closed')
            self.assertEqual(result['schedule'][-1]['action'], 'close_cached')
            check_partition(self, result)

    def test_one_unknown_tile_prevents_exclusion(self):
        public, arm, anchors, _, _ = fixture()
        genuine = s.classify
        count = [0]
        def mixed(*args, **kwargs):
            count[0] += 1
            return unknown() if count[0] == 3 else genuine(*args, **kwargs)
        with patch.object(s, 'build_problem', return_value=constant_problem()), patch.object(s, 'classify', side_effect=mixed):
            result = s.solve_branch(public, arm, anchors, source(), 'camera_tiles', tolerance=Q(20))
        self.assertEqual(result['classification_calls'], 4)
        self.assertEqual(result['pruned_nodes'], 0)
        self.assertEqual(result['outer_intervals'], source()['outer_intervals'])
        check_partition(self, result)

    def test_one_feasible_tile_prevents_exclusion(self):
        public, arm, anchors, _, _ = fixture()
        count = [0]
        def problems(*_args):
            count[0] += 1
            return constant_problem(impossible=count[0] != 2)
        with patch.object(s, 'build_problem', side_effect=problems):
            result = s.solve_branch(public, arm, anchors, source(), 'camera_tiles', tolerance=Q(20))
        self.assertEqual(result['classification_calls'], 4)
        self.assertEqual(result['trace'][0]['checks'][1]['classification']['status'], 'FEASIBLE_WITNESS')
        self.assertEqual(result['status'], 'OUTER_RETAINED')
        self.assertNotIn('FORWARD_WITNESS', repr(result))
        check_partition(self, result)

    def test_all_four_exact_certificates_are_required(self):
        public, arm, anchors, _, _ = fixture()
        with patch.object(s, 'build_problem', return_value=constant_problem()):
            complete = s.solve_branch(public, arm, anchors, source(), 'camera_tiles')
            partial = s.solve_branch(public, arm, anchors, source(), 'camera_tiles', max_classifications=3)
        self.assertEqual(complete['status'], 'OUTER_EMPTY')
        self.assertEqual(complete['classification_calls'], 4)
        self.assertEqual(complete['pruned_nodes'], 1)
        self.assertEqual(partial['status'], 'OUTER_RETAINED')
        self.assertEqual(partial['classification_calls'], 3)
        self.assertEqual(partial['trace'][0]['checks'][3]['reason'], 'not_run_call_budget')
        self.assertIsNone(partial['trace'][0]['checks'][3]['classification'])
        self.assertEqual(partial['outer_intervals'], source()['outer_intervals'])
        check_partition(self, complete);check_partition(self, partial)

    def test_invalid_certificate_or_witness_cannot_be_used(self):
        public, arm, anchors, _, _ = fixture()
        for value in ({'status': 'CERTIFIED_INFEASIBLE', 'certificate': {}, 'witness': None},
                      {'status': 'FEASIBLE_WITNESS', 'certificate': None, 'witness': []}):
            with patch.object(s, 'classify', return_value=value):
                with self.assertRaises(ArithmeticError):
                    s.solve_branch(public, arm, anchors, source(), 'depth')

    def test_expired_deadline_preserves_every_root(self):
        public, arm, anchors, _, _ = fixture()
        old = source(((Q(590), Q(592)), (Q(600), Q(605)), (Q(609), Q(610))))
        with patch.object(s, 'build_problem') as build, patch.object(s, 'classify') as classify:
            result = s.solve_branch(public, arm, anchors, old, 'camera_tiles', deadline=time.monotonic() - 1)
        build.assert_not_called();classify.assert_not_called()
        self.assertEqual(result['classification_calls'], 0)
        self.assertEqual(result['outer_intervals'], old['outer_intervals'])
        self.assertTrue(all(n['status'] == 'retained_deadline' for n in result['trace']))
        check_partition(self, result)

    def test_geometry_deadline_preserves_unrun_tiles(self):
        public, arm, anchors, _, _ = fixture()
        tick = [100.0]
        def build(*_args):
            tick[0] = 107.0
            return constant_problem()
        with patch.object(s, '_now', side_effect=lambda: tick[0]), patch.object(s, 'build_problem', side_effect=build), patch.object(s, 'classify') as classify:
            result = s.solve_branch(public, arm, anchors, source(), 'camera_tiles')
        classify.assert_not_called()
        self.assertEqual(result['termination'], 'deadline')
        self.assertEqual([c['reason'] for c in result['trace'][0]['checks']], ['not_run_deadline'] * 4)
        self.assertEqual(result['outer_intervals'], source()['outer_intervals'])
        check_partition(self, result)

    def test_alternating_boundaries_and_full_budget_frontier(self):
        public, arm, anchors, _, _ = fixture()
        old = source(((Q(590), Q(595)), (Q(605), Q(610))))
        with patch.object(s, 'classify', side_effect=unknown):
            result = s.solve_branch(public, arm, anchors, old, 'depth', max_classifications=4)
        self.assertEqual([x['side'] for x in result['schedule']], ['left', 'right', 'left', 'right'])
        self.assertEqual([x['node_id'] for x in result['schedule']], [0, 1, 2, 5])
        self.assertEqual(result['classification_calls'], 4)
        self.assertEqual(result['outer_intervals'], old['outer_intervals'])
        self.assertEqual(result['termination'], 'call_budget')
        check_partition(self, result)

    def test_tolerance_stops_both_boundaries_leaves_interior_untouched(self):
        public, arm, anchors, _, _ = fixture()
        with patch.object(s, 'classify', side_effect=unknown):
            result = s.solve_branch(public, arm, anchors, source(((Q(590), Q(598)),)), 'depth', tolerance=Q(1))
        self.assertEqual(result['termination'], 'boundaries_closed')
        self.assertEqual(result['closed_sides'], {'left': True, 'right': True})
        self.assertEqual(result['classification_calls'], 7)
        self.assertEqual(sum(n['status'] == 'retained_tolerance' for n in result['trace']), 2)
        interior = [n for n in result['trace'] if n['status'] == 'retained_interior']
        self.assertEqual(len(interior), 4)
        self.assertTrue(all(n['checks'] == [] and n['side'] is None for n in interior))
        check_partition(self, result)

    def test_extreme_certificates_shrink_hull_without_filling_gaps(self):
        public, arm, anchors, _, _ = fixture()
        old = source(((Q(590), Q(591)), (Q(599), Q(601)), (Q(609), Q(610))))
        def build(_p, _a, _an, lo, hi, _b, _box):
            return constant_problem(impossible=hi < 600 or lo > 600)
        with patch.object(s, 'build_problem', side_effect=build):
            result = s.solve_branch(public, arm, anchors, old, 'depth', tolerance=Q(1, 2))
        self.assertGreaterEqual(result['pruned_nodes'], 4)
        self.assertTrue(s.exact(result['hull'][0]) <= 600 <= s.exact(result['hull'][1]))
        self.assertGreater(s.exact(result['hull'][0]), 590)
        self.assertLess(s.exact(result['hull'][1]), 610)
        check_partition(self, result)

    def test_handwritten_true_world_not_pruned_by_actual_classifier(self):
        public, arm, anchors, _, _ = fixture()
        old = source(((Q(5999, 10), Q(6001, 10)),))
        for mode in s.MODES:
            result = s.solve_branch(public, arm, anchors, old, mode, max_classifications=4, tolerance=Q(1, 4))
            self.assertTrue(any(lo <= 600 <= hi for lo, hi in pairs(result['outer_intervals'])))
            check_partition(self, result)

    def test_labels_and_baseline_metadata_do_not_affect_search(self):
        public, arm, anchors, _, _ = fixture()
        changed = [deepcopy(public), deepcopy(arm), deepcopy(anchors), source()]
        for record in changed:
            record.update(case=object(), eta=object(), truth=object(), s0=True, tasks=object(), incumbent=object())
        changed[0]['observation_id'] = 'altered-label'
        changed[2]['observation_id'] = 'different-label'
        changed[3]['trace'] = object()
        with patch.object(s, 'classify', side_effect=unknown):
            first = s.solve_branch(public, arm, anchors, source(), 'depth', max_classifications=2)
            second = s.solve_branch(*changed, 'depth', max_classifications=2)
        for field in ('trace', 'schedule', 'roots', 'retained_ids', 'outer_intervals', 'hull', 'classification_calls'):
            self.assertEqual(first[field], second[field])

    def test_empty_source_and_empty_camera_domain_have_explicit_paths(self):
        public, arm, anchors, _, _ = fixture()
        empty = s.solve_branch(public, arm, anchors, source(()), 'camera_tiles')
        self.assertEqual(empty['classification_calls'], 0)
        self.assertEqual(empty['termination'], 'empty')
        self.assertEqual(empty['trace'], [])
        anchors['camera']['u'][1] = s.encode(Q(-3, 100000))
        for mode in s.MODES:
            result = s.solve_branch(public, arm, anchors, source(), mode)
            self.assertEqual(result['tiles'], [None])
            self.assertEqual(result['classification_calls'], 1)
            self.assertEqual(result['status'], 'OUTER_EMPTY')
            check_partition(self, result)

    def test_validation_and_no_file_read_interface(self):
        public, arm, anchors, _, _ = fixture()
        for kwargs in ({'max_classifications': True}, {'max_classifications': -1}, {'tolerance': 0}, {'branch_seconds': float('nan')}, {'deadline': float('inf')}):
            with self.assertRaises((TypeError, ValueError)):
                s.solve_branch(public, arm, anchors, source(), 'depth', **kwargs)
        _, box = s.anchor.camera_intervals(anchors)
        with self.assertRaises(ValueError):
            s.build_problem(public, arm, anchors, 590, 610, 'partial', s.baseline.DEFAULT_CAMERA_INTERVALS)
        with patch('builtins.open', side_effect=AssertionError('file access forbidden')), patch.object(s.Path, 'read_text', side_effect=AssertionError('file access forbidden')):
            s.build_problem(public, arm, anchors, 590, 610, 'partial', box)
            result = s.solve_branch(public, arm, anchors, source(), 'depth', max_classifications=0)
        self.assertEqual(result['outer_intervals'], source()['outer_intervals'])


if __name__ == '__main__':
    unittest.main()
