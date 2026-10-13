"""Hand-built independent A8 counterexamples; no population/private reads."""
from copy import deepcopy
from fractions import Fraction as Q
import unittest
import verify_anchors as v


def fixture():
    enc = v.encoded
    public = {'observation_id': 'manual_fixture', 'radius': enc(Q(1, 5)), 'baseline': enc(80),
              'clipped': True, 'observation': [enc(1)] * 432}
    frozen = {'appearance_model': 'bounded', 'drift_bound': enc(Q(1, 10)),
              'calibration_intervals': [[enc(0), enc(Q(1, 50))]] * 3}
    offsets, k, camera_noise = (Q(1, 10), Q(0), -Q(1, 10)), Q(1, 200), (Q(1, 200), -Q(1, 200), Q(0))
    anchor = {'observation_id': 'manual_fixture',
              'camera': {'u': [enc((-c - e - k - n) / 1000) for c, e, n in zip(v.CENTERS, offsets, camera_noise)],
                         'Xf': enc(0), 'Zf': enc(1000), 'survey_bound': enc(Q(1, 100)), 'localization_bound': enc(Q(1, 100))},
              'appearance': {'coupons': [enc(Q(99, 100))] * 9, 'references': [enc(Q(49, 100))] * 3,
                             'bias_bound': enc(Q(1, 100)), 'transfer_bound': enc(Q(1, 200)), 'noise_bound': enc(Q(1, 200))}}
    witness = {'offsets': list(map(enc, offsets)), 'darkness': [enc(0)] * 9, 't': [enc(0)] * 3,
               'noise': [enc(-Q(1, 100))] * 432, 'tA': [enc(Q(1, 50))] * 3,
               'camera_k': enc(k), 'camera_noise': list(map(enc, camera_noise)),
               'coupon_transfer': [enc(Q(1, 200))] * 9, 'coupon_noise': [enc(-Q(1, 200))] * 9,
               'reference_noise': [enc(0)] * 3}
    return public, frozen, anchor, witness


class ForwardTests(unittest.TestCase):
    def test_full_forward_independent_anchor_bias_and_single_clip(self):
        public, frozen, anchor, witness = fixture()
        self.assertTrue(v.verify_forward(public, frozen, anchor, Q(9007, 14), 'partial', 'both', witness))
        self.assertNotEqual(witness['t'], witness['tA'])
        # Clipping photo+bias first would produce .99 after this noise; correct
        # single clipping produces1 from1+.01-.01.
        altered = deepcopy(public)
        altered['observation'][0] = v.encoded(Q(99, 100))
        with self.assertRaises(v.VerificationError):
            v.verify_forward(altered, frozen, anchor, Q(9007, 14), 'partial', 'both', witness)
        altered = deepcopy(witness)
        altered['tA'] = altered['t']
        with self.assertRaises(v.VerificationError):
            v.verify_forward(public, frozen, anchor, Q(9007, 14), 'partial', 'both', altered)

    def test_old_world_extension_is_existential_and_keeps_old_coordinates(self):
        public, frozen, anchor, witness = fixture()
        old = {key: witness[key] for key in ('offsets', 'darkness', 't', 'noise')}
        for arm in v.ARMS:
            result = v.extend_saved_witness(public, frozen, anchor, 640, 'none', arm, old)
            self.assertIsNotNone(result)
            self.assertEqual({key: result[key] for key in old}, old)
            self.assertTrue(v.verify_forward(public, frozen, anchor, 640, 'none', arm, result))

    def test_marginal_camera_intervals_do_not_prove_common_survey(self):
        public, frozen, anchor, witness = fixture()
        residuals = (Q(1, 50), Q(0), -Q(1, 50))
        anchor['camera']['u'] = [v.encoded((-c - r) / 1000) for c, r in zip(v.CENTERS, residuals)]
        _, intervals = v.camera_intervals(anchor, 'camera')
        self.assertTrue(all(lo <= 0 <= hi for lo, hi in intervals))
        old = {key: witness[key] for key in ('offsets', 'darkness', 't', 'noise')}
        old['offsets'] = [v.encoded(0)] * 3
        self.assertIsNone(v.extend_saved_witness(public, frozen, anchor, 640, 'none', 'camera', old))
        self.assertIsNotNone(v.extend_saved_witness(public, frozen, anchor, 640, 'none', 'none', old))

    def test_strict_dimensions_inactive_fields_and_exact_scalars(self):
        public, frozen, anchor, witness = fixture()
        for key, bad in (('offsets', [v.encoded(0)]), ('darkness', [False] * 9), ('tA', [0.01] * 3)):
            changed = deepcopy(witness)
            changed[key] = bad
            with self.assertRaises(v.VerificationError):
                v.verify_forward(public, frozen, anchor, 640, 'none', 'both', changed)
        with self.assertRaises(v.VerificationError):
            v.verify_forward(public, frozen, anchor, 640, 'none', 'none', witness)
        changed = deepcopy(anchor)
        changed['camera']['survey_bound'] = True
        with self.assertRaises(v.VerificationError):
            v.validate_anchor(changed)

    def test_disabled_appearance_still_has_three_box_only_bias_variables(self):
        public, frozen, anchor, witness = fixture()
        old = {key: witness[key] for key in ('offsets', 'darkness', 't', 'noise')}
        result = v.extend_saved_witness(public, frozen, anchor, 640, 'none', 'none', old)
        self.assertEqual(len(result['tA']), 3)
        for key in ('coupon_transfer', 'coupon_noise', 'reference_noise'):
            self.assertIsNone(result[key])
        for invalid in (None, [v.encoded(0)] * 2, [v.encoded(Q(3, 100))] * 3):
            with self.assertRaises(v.VerificationError):
                v.verify_forward(public, frozen, anchor, 640, 'none', 'none', {**result, 'tA': invalid})


class OuterTests(unittest.TestCase):
    def test_nonzero_camera_centers_and_disabled_reading_isolation(self):
        public, frozen, anchor, _ = fixture()
        _, active = v.camera_intervals(anchor, 'camera')
        self.assertGreater(active[0][0], 0)
        self.assertLess(active[2][1], 0)
        _, inactive = v.camera_intervals(anchor, 'none')
        self.assertEqual(inactive, ((-Q(1, 5), Q(1, 5)), (Q(0), Q(0)), (-Q(1, 5), Q(1, 5))))
        changed = deepcopy(anchor)
        changed['appearance']['coupons'] = [v.encoded(0)] * 9
        changed['appearance']['references'] = [v.encoded(1)] * 3
        self.assertEqual(v.necessary_problem(public, frozen, anchor, 640, 'partial', 'camera'),
                         v.necessary_problem(public, frozen, changed, 640, 'partial', 'camera'))

    def test_area_enclosure_contains_cells_and_midpoints(self):
        intervals = ((-Q(1, 5), Q(1, 5)), (Q(0), Q(0)), (-Q(1, 5), Q(1, 5)))
        depth = Q(9007, 14)
        for branch in v.BRANCHES:
            bounds = v.area_enclosure(branch, depth, intervals)
            for view, interval in enumerate(intervals):
                cuts = v.area_breakpoints(branch, depth, view, interval)
                for lo, hi in zip(cuts, cuts[1:]):
                    points = []
                    for e in (lo, (lo + hi) / 2, hi):
                        offsets = tuple(e if j == view else Q(0) for j in range(3))
                        points.append(v.point_areas(branch, depth, offsets)[48 * view:48 * (view + 1)])
                    for pixel in range(48):
                        for material in range(3):
                            low, high = bounds[48 * view + pixel][2 * material:2 * material + 2]
                            self.assertTrue(low <= points[1][pixel][material] <= high)
                            self.assertEqual(points[1][pixel][material], (points[0][pixel][material] + points[2][pixel][material]) / 2)

    def test_true_forward_coordinates_satisfy_whole_pose_necessary_rows(self):
        public, frozen, anchor, witness = fixture()
        for arm in v.ARMS:
            rows, bounds = v.necessary_problem(public, frozen, anchor, Q(9007, 14), 'partial', arm)
            self.assertEqual(len(bounds), 15)
            point = witness['darkness'] + witness['t'] + witness['tA']
            v._proof.verify_witness(rows, bounds, point)
            for _, coefficients, _ in rows:
                self.assertEqual(len(coefficients), 15)

    def test_signed_certificate_tamper_and_zero_margin_rejected(self):
        rows = (('upper', (Q(-1), Q(0)), -Q(3, 4)), ('upper', (Q(1), Q(-1)), -Q(3, 10)))
        bounds = ((Q(0), Q(1)), (Q(0), Q(1)))
        certificate = {'sparse_weights': [{'row': 0, 'weight': v.encoded(1)}, {'row': 1, 'weight': v.encoded(1)}],
                       'aggregate_coefficients': [v.encoded(0), v.encoded(-1)], 'weighted_bound': v.encoded(-Q(21, 20)),
                       'box_min': v.encoded(-1), 'margin': v.encoded(Q(1, 20))}
        v.verify_certificate(rows, bounds, certificate)
        for key, value in (('margin', v.encoded(0)), ('box_min', v.encoded(0))):
            with self.assertRaises(v.VerificationError):
                v.verify_certificate(rows, bounds, {**certificate, key: value})
        with self.assertRaises(v.VerificationError):
            v.verify_certificate(rows, (bounds[0], (Q(0), Q(2))), certificate)


class PublicGateTests(unittest.TestCase):
    def test_camera_skips_are_counted_without_fictitious_lp_calls(self):
        public, frozen, anchor, _ = fixture()
        record = {'status': 'UNKNOWN', 'witness': None, 'center_status': 'UNKNOWN', 'center_witness': None,
                  'numeric_evaluations': 3, 'classifier_lp_calls': 0, 'numeric_lp_calls': 1,
                  'skipped_camera_candidates': 2, 'exact_attempts': 0, 'numeric_failures': 0,
                  'powell_starts': 0, 'powell_evaluations': [], 'best_phase_slack': None,
                  'seconds': 0.1, 'termination': 'search_exhausted', 'search_budget_seconds': 3.0,
                  'limits': dict(v.SEARCH_LIMITS)}
        self.assertEqual(v.verify_search(public, frozen, anchor, 640, 'none', 'camera', record), 0.1)
        for key, bad in (('skipped_camera_candidates', -1), ('skipped_camera_candidates', 4),
                         ('skipped_camera_candidates', True), ('numeric_lp_calls', 3)):
            with self.assertRaises(v.VerificationError):
                v.verify_search(public, frozen, anchor, 640, 'none', 'camera', {**record, key: bad})

    def test_saved_public_matrix_requires_every_unique_arm_and_branch(self):
        observations = {f'o{i:03d}': {} for i in range(36)}
        tasks = {oid + f's{s}': {'task_id': oid + f's{s}', 'observation_id': oid, 'incumbent': v.encoded(640)}
                 for oid in observations for s in range(9)}
        records = [{'task_id': tid, 'observation_id': task['observation_id'], 'incumbent': task['incumbent'],
                    'arm': arm, 'a6_status': 'UNKNOWN', 'a7_status': 'UNKNOWN', 'status': 'UNKNOWN',
                    'branches': [{'branch': b, 'status': 'UNKNOWN'} for b in v.BRANCHES]}
                   for tid, task in tasks.items() for arm in v.ARMS]
        self.assertEqual(len(v.verify_record_population(records, observations, tasks)), 1296)
        for changed in (records[:-1], records[:-1] + [records[0]]):
            with self.assertRaises(v.VerificationError):
                v.verify_record_population(changed, observations, tasks)
        changed = deepcopy(records)
        changed[0]['branches'].pop()
        with self.assertRaises(v.VerificationError):
            v.verify_record_population(changed, observations, tasks)
        self.assertEqual(v.aggregate(['INHERITED_INFEASIBLE', 'CERTIFIED_INFEASIBLE', 'INHERITED_INFEASIBLE']), 'CERTIFIED_INFEASIBLE')
        self.assertEqual(v.aggregate(['INHERITED_INFEASIBLE', 'UNKNOWN', 'CERTIFIED_INFEASIBLE']), 'UNKNOWN')

    def test_actual_runner_private_data_only_occurs_in_hash_inventory(self):
        self.assertEqual(v.audit_information_flow()['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
