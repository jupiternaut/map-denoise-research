"""A11 handwritten public fixtures only: no saved run data or private files."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import accumulation_solver as s


def fixture(branch='partial', depth=Q(600)):
    a = s.kernel.anchor
    offsets, darkness = (Q(-3, 25), Q(0), Q(1, 11)), (Q(1, 2),) * 9
    ts, ta = (Q(1, 50), Q(1, 100), Q(0)), (Q(0), Q(1, 100), Q(1, 50))
    raw = a.pose._prediction(depth, branch, offsets, darkness, ts)
    photo = {'observation_id': 'handwritten', 'radius': a.encode(Q(1, 5)),
             'baseline': a.encode(80), 'clipped': True,
             'observation': [a.encode(a._clip(value)) for value in raw]}
    oldarm = {'appearance_model': 'bounded', 'drift_bound': a.encode(Q(1, 10)),
              'calibration_intervals': [[a.encode(t), a.encode(t)] for t in ts]}
    survey, noise = Q(1, 200), (Q(-1, 200), Q(0), Q(1, 200))
    readings = tuple((-center - offset - survey - error) / 1000 for center, offset, error in
                     zip((-Q(80), Q(0), Q(80)), offsets, noise))
    anchors = {'observation_id': 'handwritten', 'camera': {
        'u': list(map(a.encode, readings)), 'Xf': a.encode(0), 'Zf': a.encode(1000),
        'survey_bound': a.encode(Q(1, 100)), 'localization_bound': a.encode(Q(1, 100))},
        'appearance': {
            'coupons': [a.encode(a._clip(1 - d + a.BETA - ta[i // 3])) for i, d in enumerate(darkness)],
            'references': [a.encode(Q(1, 2) + a.BETA - t) for t in ta],
            'bias_bound': a.encode(Q(1, 100)), 'transfer_bound': a.encode(Q(1, 200)),
            'noise_bound': a.encode(Q(1, 200))}}
    return photo, oldarm, anchors


def source(branch='partial', intervals=((Q(600) - Q(1, 1000), Q(600) + Q(1, 1000)),)):
    return {'branch': branch, 'mode': 'depth',
            'outer_intervals': [[s.encode(lo), s.encode(hi)] for lo, hi in intervals]}


def pairs(values):
    return tuple(tuple(map(s.exact, pair)) for pair in values)


def timeless(value):
    if isinstance(value, dict):
        return {k: timeless(v) for k, v in value.items() if not k.endswith('seconds')}
    if isinstance(value, list):
        return [timeless(v) for v in value]
    return value


class HiddenValue:
    def __deepcopy__(self, _memo):
        raise AssertionError('an unregistered hidden value was inspected')


class AccumulationTests(unittest.TestCase):
    def test_fixed_exports_and_resolved_historical_identity(self):
        logical = Path('/srv/slam-research/grf/map-denoise/runs/'
                       'lovelace-baseline-intervention-20261012T045547Z/continuous-tightening-1/tightening_solver.py')
        self.assertEqual(Path(s.kernel.__file__).resolve(), logical.resolve())
        self.assertTrue(logical.samefile(s.kernel.__file__))
        self.assertEqual(s.ARMS, ('D_then_D', 'D_then_DC'))
        self.assertEqual(s.BRANCHES, ('none', 'partial', 'full'))
        self.assertEqual((s.MIN_WIDTH, s.MAX_CLASSIFICATIONS, s.BRANCH_SECONDS), (Q(15, 1024), 128, 6.0))
        self.assertEqual(s.kernel.MIN_WIDTH, Q(15, 256))
        for name in ('clear_geometry_cache', 'select_depth', 'merge_intervals', 'encode', 'exact'):
            self.assertIs(getattr(s, name), getattr(s.kernel, name))

    def test_forwarded_modes_fixed_budgets_and_unchanged_schema(self):
        photo, oldarm, anchors = fixture()
        frozen_function = s.kernel.solve_branch
        for arm, mode in s.MODES.items():
            sentinel = {'mode': mode, 'opaque-result': object()}
            with patch.object(s, '_kernel_solve', return_value=sentinel) as capture:
                result = s.solve_branch(photo, oldarm, anchors, source(), arm, deadline=123.5)
            self.assertIs(result, sentinel)
            args, kwargs = capture.call_args
            self.assertEqual(args[4], mode)
            self.assertEqual(set(args[3]), {'branch', 'outer_intervals'})
            self.assertEqual(kwargs, {'deadline': 123.5, 'max_classifications': 128,
                                      'tolerance': Q(15, 1024), 'branch_seconds': 6.0})
        self.assertIs(s.kernel.solve_branch, frozen_function)

    def test_normalizer_drops_all_nonmeasurement_labels_and_copies_lists(self):
        photo, oldarm, anchors = fixture()
        old_source = source(intervals=((Q(600), Q(601)), (Q(599), Q(600))))
        for record in (photo, oldarm, anchors, anchors['camera'], anchors['appearance'], old_source):
            record.update(truth=HiddenValue(), case=HiddenValue(), eta=HiddenValue(),
                          unit=HiddenValue(), incumbent=HiddenValue(), trace=HiddenValue(),
                          old_DC_results=HiddenValue())
        photo['observation_id'], anchors['observation_id'] = HiddenValue(), HiddenValue()
        clean_photo, clean_cal, clean_anchor, clean_source = s.normalize_inputs(photo, oldarm, anchors, old_source)
        self.assertEqual(set(clean_photo), {'observation', 'radius', 'baseline', 'clipped', 'observation_id'})
        self.assertEqual(set(clean_cal), {'calibration_intervals', 'appearance_model', 'drift_bound'})
        self.assertEqual(set(clean_anchor), {'camera', 'appearance', 'observation_id'})
        self.assertEqual(set(clean_anchor['camera']), {'u', 'Xf', 'Zf', 'survey_bound', 'localization_bound'})
        self.assertEqual(set(clean_anchor['appearance']), {'coupons', 'references', 'bias_bound', 'transfer_bound', 'noise_bound'})
        self.assertEqual(clean_photo['observation_id'], clean_anchor['observation_id'])
        self.assertEqual(set(clean_source), {'branch', 'outer_intervals'})
        self.assertEqual(pairs(clean_source['outer_intervals']), ((Q(599), Q(601)),))
        self.assertIsNot(clean_photo['observation'], photo['observation'])
        self.assertIsNot(clean_cal['calibration_intervals'][0], oldarm['calibration_intervals'][0])
        self.assertIsNot(clean_anchor['camera']['u'], anchors['camera']['u'])

    def test_wrong_arm_source_mode_and_branch_rejected_without_kernel_call(self):
        photo, oldarm, anchors = fixture()
        with patch.object(s, '_kernel_solve', side_effect=AssertionError('must not run')):
            for arm in ('depth', 'camera_tiles', 'both', None):
                with self.assertRaises(ValueError):
                    s.solve_branch(photo, oldarm, anchors, source(), arm)
            for mode in ('camera_tiles', None):
                bad = dict(source(), mode=mode)
                with self.assertRaises(ValueError):
                    s.solve_branch(photo, oldarm, anchors, bad, 'D_then_D')
            with self.assertRaises(ValueError):
                s.solve_branch(photo, oldarm, anchors, dict(source(), branch='unknown'), 'D_then_D')
            with self.assertRaises(TypeError):
                s.solve_branch(photo, oldarm, anchors, [], 'D_then_D')
            with self.assertRaises(ValueError):
                s.solve_branch(photo, oldarm, anchors, source(intervals=((Q(539), Q(600)),)), 'D_then_D')

    def test_real_world_all_branches_both_arms_and_closed_camera_cover(self):
        for branch in s.BRANCHES:
            photo, oldarm, anchors = fixture(branch)
            baseline = source(branch)
            snapshot = deepcopy((photo, oldarm, anchors, baseline))
            for arm in s.ARMS:
                s.clear_geometry_cache()
                result = s.solve_branch(photo, oldarm, anchors, baseline, arm)
                self.assertEqual(result['mode'], s.MODES[arm])
                self.assertNotIn('arm', result)
                self.assertEqual(result['status'], 'OUTER_RETAINED')
                self.assertEqual(result['termination'], 'boundaries_closed')
                self.assertEqual(result['outer_intervals'], baseline['outer_intervals'])
                self.assertEqual(result['classification_calls'], 1 if arm == 'D_then_D' else 4)
                self.assertEqual(result['max_classifications'], 128)
                self.assertEqual(s.exact(result['min_width']), Q(15, 1024))
                self.assertEqual(result['branch_seconds'], 6.0)
                self.assertTrue(any(lo <= 600 <= hi for lo, hi in pairs(result['outer_intervals'])))
                self.assertEqual(len(result['trace'][0]['checks']), len(result['tiles']))
                if arm == 'D_then_DC':
                    full, tiles = pairs(result['camera_domain']), tuple(pairs(t) for t in result['tiles'])
                    for left in (full[0][0], sum(full[0])/2, full[0][1]):
                        for right in (full[2][0], sum(full[2])/2, full[2][1]):
                            self.assertTrue(any(tile[0][0] <= left <= tile[0][1] and
                                                tile[1] == (Q(0), Q(0)) and
                                                tile[2][0] <= right <= tile[2][1] for tile in tiles))
            self.assertEqual((photo, oldarm, anchors, baseline), snapshot)

    def test_finer_tolerance_actually_splits_and_preserves_hand_world(self):
        photo, oldarm, anchors = fixture()
        baseline = source(intervals=((Q(600)-Q(15,1024), Q(600)+Q(15,1024)),))
        for arm in s.ARMS:
            result = s.solve_branch(photo, oldarm, anchors, baseline, arm)
            self.assertEqual(result['trace'][0]['status'], 'split')
            self.assertEqual(result['trace'][0]['children'], [1, 2])
            self.assertGreaterEqual(result['classification_calls'], 3 if arm == 'D_then_D' else 12)
            self.assertTrue(any(lo <= 600 <= hi for lo, hi in pairs(result['outer_intervals'])))
            self.assertTrue(all(any(sl <= lo <= hi <= sh for sl, sh in pairs(baseline['outer_intervals']))
                                for lo, hi in pairs(result['outer_intervals'])))

    def test_expired_deadline_retains_all_disconnected_frontier(self):
        photo, oldarm, anchors = fixture()
        baseline = source(intervals=((Q(590), Q(591)), (Q(599), Q(601)), (Q(609), Q(610))))
        for arm in s.ARMS:
            result = s.solve_branch(photo, oldarm, anchors, baseline, arm, deadline=time.monotonic()-1)
            self.assertEqual(result['termination'], 'deadline')
            self.assertEqual(result['classification_calls'], 0)
            self.assertEqual(result['outer_intervals'], baseline['outer_intervals'])
            self.assertEqual(result['retained_ids'], [0, 1, 2])
            self.assertTrue(all(n['status'] == 'retained_deadline' and not n['checks'] for n in result['trace']))

    def test_empty_support_and_empty_camera_are_explicit_not_vacuous(self):
        photo, oldarm, anchors = fixture()
        for arm in s.ARMS:
            empty = s.solve_branch(photo, oldarm, anchors, source(intervals=()), arm)
            self.assertEqual((empty['outer_intervals'], empty['trace'], empty['classification_calls']), ([], [], 0))
            self.assertEqual(empty['termination'], 'empty')
            impossible = deepcopy(anchors)
            impossible['camera']['u'][1] = s.encode(Q(-3, 100000))
            result = s.solve_branch(photo, oldarm, impossible, source(), arm)
            self.assertIsNone(result['camera_domain'])
            self.assertEqual(result['tiles'], [None])
            self.assertEqual(result['classification_calls'], 1)
            self.assertEqual(result['outer_intervals'], [])
            self.assertEqual(result['trace'][0]['checks'][0]['classification']['status'], 'CERTIFIED_INFEASIBLE')

    def test_hidden_labels_cannot_change_actual_deadline_frontier(self):
        photo, oldarm, anchors = fixture()
        baseline = source()
        for arm in s.ARMS:
            clean = s.solve_branch(photo, oldarm, anchors, baseline, arm, deadline=0)
            tagged = deepcopy((photo, oldarm, anchors, baseline))
            for record in tagged:
                record.update(truth=HiddenValue(), incumbent=HiddenValue(), old_DC_results=HiddenValue())
            tagged[0]['observation_id'], tagged[2]['observation_id'] = HiddenValue(), HiddenValue()
            dirty = s.solve_branch(*tagged, arm, deadline=0)
            self.assertEqual(timeless(clean), timeless(dirty))

    def test_no_runtime_file_reads_for_method_inputs(self):
        photo, oldarm, anchors = fixture()
        with patch('builtins.open', side_effect=AssertionError('no method file reads')):
            with patch.object(Path, 'read_text', side_effect=AssertionError('no method file reads')):
                result = s.solve_branch(photo, oldarm, anchors, source(), 'D_then_DC', deadline=0)
        self.assertEqual(result['classification_calls'], 0)


if __name__ == '__main__':
    unittest.main()
