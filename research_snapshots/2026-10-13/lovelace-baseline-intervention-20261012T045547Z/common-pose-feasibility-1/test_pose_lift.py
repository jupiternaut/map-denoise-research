"""Handwritten common-pose fixtures; no task/result/private file loads."""

from copy import deepcopy
from fractions import Fraction as Q
import json
import unittest
from unittest.mock import patch

import pose_lift as lift


def fixture(depth=Q(601), branch="none", darkness=None, ts=None, offsets=None):
    darkness = tuple(darkness if darkness is not None else (Q(1, 2), Q(1, 4), Q(1, 3)) * 3)
    ts = tuple(ts if ts is not None else (Q(1, 100),) * 3)
    offsets = tuple(offsets if offsets is not None else (Q(3, 20), Q(0), Q(-1, 10)))
    areas = lift.geometry.independent_point_areas(branch, depth, offsets, lift.BASELINE)
    image = []
    for pixel, values in enumerate(areas):
        view = pixel // 48
        raw = 1 - sum((a * d for a, d in zip(values, darkness[3 * view:3 * view + 3])), Q(0))
        raw += lift.BETA - ts[view]
        image.extend((max(Q(0), min(Q(1), raw)),) * 3)
    public = {"observation_id": "handwritten-only", "radius": lift.encode(lift.RADIUS),
              "baseline": lift.encode(lift.BASELINE), "clipped": True,
              "observation": list(map(lift.encode, image))}
    arm = {"appearance_model": "bounded", "drift_bound": lift.encode(lift.DRIFT_BOUND),
           "calibration_intervals": [[lift.encode(0), lift.encode(2 * lift.BETA)] for _ in range(3)]}
    return public, arm, list(map(lift.encode, darkness + ts))


class PoseLiftTests(unittest.TestCase):
    def test_complete_legal_seed_lifts_without_changing_inputs(self):
        for branch in lift.BRANCHES:
            public, arm, witness = fixture(branch=branch)
            old = deepcopy((public, arm, witness))
            result = lift.lift_seed(public, arm, Q(601), branch, witness)
            self.assertEqual(result["status"], "SEED_LIFTED")
            self.assertTrue(lift.verify_offsets(public, arm, Q(601), branch, witness, result["offsets"]))
            self.assertEqual(lift.exact(result["offsets"][1]), 0)
            self.assertEqual((public, arm, witness), old)
            self.assertEqual(len(result["views"]), 3)
            json.dumps(result, allow_nan=False)

    def test_every_partition_cell_is_affine_including_occlusion(self):
        for branch in lift.BRANCHES:
            for depth in (Q(540), Q(576), Q(601), Q(18000, 29), Q(660)):
                for view in range(3):
                    points, cells = lift.camera_cells(branch, depth, view)
                    self.assertEqual(tuple(sorted(set(points))), points)
                    self.assertEqual(cells[0][0], 0 if view == 1 else -lift.RADIUS)
                    self.assertEqual(cells[-1][1], 0 if view == 1 else lift.RADIUS)
                    for cell in cells:
                        affine = lift.cell_affine_areas(branch, depth, view, cell)
                        for weight in (Q(0), Q(1, 4), Q(1, 2), Q(3, 4), Q(1)):
                            offset = cell[0] + weight * (cell[1] - cell[0])
                            actual = lift._view_areas(branch, depth, view, offset)
                            predicted = tuple(tuple(a + b * offset for a, b in pixel) for pixel in affine)
                            self.assertEqual(actual, predicted, (branch, depth, view, cell, offset))

    def test_camera_crossings_and_closed_endpoints_are_exact(self):
        points, cells = lift.camera_cells("none", Q(601), 0)
        self.assertEqual(points, (-Q(1, 5), -Q(1, 10), Q(1, 5)))
        self.assertEqual(cells[0][1], cells[1][0])
        points, _ = lift.camera_cells("partial", Q(18000, 29), 0)
        self.assertIn(Q(0), points)
        self.assertEqual(lift.camera_cells("none", Q(601), 1), ((Q(0),), ((Q(0), Q(0)),)))
        points, cells = lift.camera_cells("none", Q(598), 0)
        self.assertEqual(points, (-Q(1, 5), Q(1, 5)))
        self.assertEqual(len(cells), 1)

    def test_exact_interval_endpoint_zero_slope_and_signed_constraints(self):
        box = (-Q(1, 5), Q(1, 5))
        self.assertEqual(lift._intersect(box, Q(1), Q(0), "lower", Q(1, 5)), (Q(1, 5), Q(1, 5)))
        self.assertEqual(lift._intersect(box, -Q(1), Q(0), "lower", Q(1, 5)), (-Q(1, 5), -Q(1, 5)))
        self.assertIsNone(lift._intersect(box, Q(1), Q(0), "lower", Q(201, 1000)))
        self.assertEqual(lift._intersect(box, Q(0), Q(1), "upper", Q(1)), box)
        self.assertIsNone(lift._intersect(box, Q(0), Q(1), "upper", Q(999, 1000)))
        self.assertEqual(lift._intersect(box, -Q(1), Q(0), "upper", -Q(1, 10)), (Q(1, 10), Q(1, 5)))

    def test_necessary_seed_can_fail_common_pose_without_depth_claim(self):
        depth = Q(601)
        public, arm, witness = fixture(depth, ts=(2 * lift.BETA,) * 3,
                                      offsets=(Q(0), Q(0), Q(0)))
        darkness, ts = tuple(map(lift.exact, witness[:9])), tuple(map(lift.exact, witness[9:]))
        # Same horizontal edge in two rows: one demands e>=3/20, the other
        # demands e<=-1/10. Their separate area intervals still admit the seed.
        for pixel, offset, sign in ((21, Q(3, 20), -1), (29, -Q(3, 20), 1)):
            areas = lift.geometry.independent_point_areas("none", depth, (offset, Q(0), Q(0)), lift.BASELINE)
            raw = 1 - sum((a * d for a, d in zip(areas[pixel], darkness[:3])), Q(0)) + lift.BETA - ts[0]
            y = raw + sign * lift.EPSILON
            self.assertTrue(0 <= y <= 1)
            public["observation"][3 * pixel:3 * pixel + 3] = [lift.encode(y)] * 3
        intervals = lift.geometry.independent_geometry("none", lift.RADIUS, depth, depth, lift.BASELINE)
        for pixel, (values, (high, low)) in enumerate(zip(intervals, lift.pixel_bands(public["observation"], True))):
            view = pixel // 48
            lows = ts[view] + sum((a * d for a, d in zip(values[::2], darkness[3 * view:3 * view + 3])), Q(0))
            highs = ts[view] + sum((a * d for a, d in zip(values[1::2], darkness[3 * view:3 * view + 3])), Q(0))
            self.assertTrue(high is None or lows <= high)
            self.assertTrue(low is None or highs >= low)
        result = lift.lift_seed(public, arm, depth, "none", witness)
        self.assertEqual(result["status"], "SEED_NOT_LIFTABLE")
        self.assertIsNone(result["offsets"])
        self.assertTrue(all(cell["feasible"] is None for cell in result["views"][0]["cells"]))
        self.assertEqual(result["reason"], "fixed_seed_has_no_common_offset_completion")

    def test_saturation_is_one_sided_and_near_boundary_interior_is_not(self):
        public, arm, witness = fixture(darkness=(Q(0),) * 9, ts=(Q(0),) * 3)
        self.assertEqual(lift.lift_seed(public, arm, Q(601), "none", witness)["status"], "SEED_LIFTED")
        # preclip 1.01, y=.999: clipped residual is small but no bounded
        # additive noise can produce the exact interior channel.
        public["observation"][3 * 48] = lift.encode(Q(999, 1000))
        result = lift.lift_seed(public, arm, Q(601), "none", witness)
        self.assertEqual(result["status"], "SEED_NOT_LIFTABLE")
        self.assertTrue(all(cell["feasible"] is None for cell in result["views"][1]["cells"]))
        public, arm, witness = fixture(branch="full", darkness=(Q(1),) * 9, ts=(2 * lift.BETA,) * 3)
        self.assertEqual(lift.lift_seed(public, arm, Q(601), "full", witness)["status"], "SEED_LIFTED")
        pixel = 48 + 2 * 8 + 3
        self.assertEqual(lift.exact(public["observation"][3 * pixel]), 0)
        public["observation"][3 * pixel] = lift.encode(Q(1, 1000))
        self.assertEqual(lift.lift_seed(public, arm, Q(601), "full", witness)["status"], "SEED_NOT_LIFTABLE")

    def test_original_rgb_channels_all_constrain_the_same_offset(self):
        public, arm, witness = fixture()
        public["observation"][3 * 48 + 2] = lift.encode(Q(1, 2))
        result = lift.lift_seed(public, arm, Q(601), "none", witness)
        self.assertEqual(result["status"], "SEED_NOT_LIFTABLE")
        self.assertTrue(all(cell["feasible"] is None for cell in result["views"][1]["cells"]))

    def test_pairwise_material_drift_and_calibration_are_checked(self):
        public, arm, witness = fixture(darkness=(Q(1, 2),) * 3 + (Q(3, 5),) * 3 + (Q(1, 2),) * 3)
        self.assertEqual(lift.lift_seed(public, arm, Q(601), "none", witness)["status"], "SEED_LIFTED")
        bad = deepcopy(witness)
        bad[3] = lift.encode(Q(601, 1000))
        with self.assertRaises(ValueError):
            lift.lift_seed(public, arm, Q(601), "none", bad)
        bad = deepcopy(witness)
        bad[9] = lift.encode(Q(21, 1000))
        with self.assertRaises(ValueError):
            lift.lift_seed(public, arm, Q(601), "none", bad)
        arm["calibration_intervals"][0] = [lift.encode(0), lift.encode(Q(9, 1000))]
        with self.assertRaises(ValueError):
            lift.lift_seed(public, arm, Q(601), "none", witness)

    def test_invalid_seed_public_contract_and_offsets_are_rejected(self):
        public, arm, witness = fixture()
        for bad in (witness[:11], witness + [lift.encode(0)], witness[:1] + [0.5] + witness[2:]):
            with self.assertRaises((TypeError, ValueError)):
                lift.lift_seed(public, arm, Q(601), "none", bad)
        for key, value in (("radius", lift.encode(Q(1, 10))), ("baseline", lift.encode(70)), ("clipped", False)):
            bad = deepcopy(public)
            bad[key] = value
            with self.assertRaises(ValueError):
                lift.lift_seed(bad, arm, Q(601), "none", witness)
        for offsets in (None, [0, 0], [0, Q(1, 100), 0], [Q(1, 4), 0, 0], [0.0, 0, 0]):
            self.assertFalse(lift.verify_offsets(public, arm, Q(601), "none", witness, offsets))
    def test_selected_candidate_must_pass_direct_point_geometry(self):
        public, arm, witness = fixture()
        with patch.object(lift, "_offsets_satisfy", return_value=False):
            with self.assertRaises(ArithmeticError):
                lift.lift_seed(public, arm, Q(601), "none", witness)


if __name__ == "__main__":
    unittest.main()
