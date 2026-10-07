"""Synthetic-only tests. No run inputs or evaluation artifacts are opened."""

import copy
import unittest

from policies import POLICIES, merge_intervals, new_policy, old_policy, project_depth, uniform_set_mean


def obj(cid, depth, score=None, x=0.0):
    return dict(candidate_id=cid, xyz_mm=[x, 0.0, depth], depth_mm=depth, old_score=score)


def row(objects=None, intervals=None, current_id=-1, original_id=-1, tracks=None):
    intervals = [[4.0, 6.0]] if intervals is None else intervals
    return dict(
        objects=[obj(-1, 0.0), obj(0, 5.0)] if objects is None else objects,
        current_id=current_id, original_id=original_id, center=[0.0, 0.0, 0.0],
        ray=[0.0, 0.0, 1.0], historical_photo_id=current_id,
        intervals={evidence: copy.deepcopy(intervals) for evidence in ("star_full", "star", "cycle")},
        tracks={evidence: copy.deepcopy(tracks or []) for evidence in ("star", "cycle")},
    )


def track(tid, depth, support, ncc=(0.8, 0.8)):
    return dict(id=tid, reference_depth_mm=depth, intervals_mm=support, ncc_reference_sources=list(ncc))


class IntervalTests(unittest.TestCase):
    def test_merge_overlap_touching_and_duplicates(self):
        self.assertEqual(merge_intervals([[3, 4], [0, 2], [1, 3], [7, 7], [7, 7]]), [[0, 4], [7, 7]])

    def test_length_uniform_union_mean(self):
        self.assertEqual(uniform_set_mean([[0, 4], [2, 4], [10, 12]]), 5.0)
        self.assertEqual(uniform_set_mean([[0, 4], [100, 100]]), 2.0)

    def test_unique_singleton_mean(self):
        self.assertEqual(uniform_set_mean([[1, 1], [1, 1], [7, 7]]), 4.0)
        self.assertIsNone(uniform_set_mean([]))

    def test_projection_interior_boundary_and_tie(self):
        self.assertEqual(project_depth(3, [[1, 2], [4, 5]]), 2.0)
        self.assertEqual(project_depth(4.5, [[1, 2], [4, 5]]), 4.5)
        self.assertEqual(project_depth(9, [[1, 2], [4, 5]]), 5.0)

    def test_invalid_intervals_rejected(self):
        for intervals in ([[2, 1]], [[0, float("nan")]], [[0, float("inf")]]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                merge_intervals(intervals)


class NewPolicyTests(unittest.TestCase):
    def test_empty_keeps_for_every_policy(self):
        for policy in POLICIES:
            result = new_policy(row(intervals=[]), "star", policy)
            self.assertEqual((result["selected_candidate_id"], result["reason"]), (-1, "EMPTY_UNKNOWN"))

    def test_missing_current_keeps_for_every_policy(self):
        for current_id in (None, 99):
            for policy in POLICIES:
                result = new_policy(row(current_id=current_id), "star", policy)
                self.assertEqual(result["selected_candidate_id"], current_id)
                self.assertEqual(result["reason"], "NO_CURRENT_OUTPUT")

    def test_no_improvement_keeps(self):
        data = row([obj(-1, 5), obj(0, 20)], tracks=[track(0, 5, [[4, 6]])])
        for policy in POLICIES:
            result = new_policy(data, "star", policy)
            self.assertEqual(result["selected_candidate_id"], -1)
            self.assertEqual(result["eligible_ids"], [])

    def test_candidate_tie_is_smallest_id(self):
        data = row([obj(4, 5), obj(-1, 0), obj(2, 5)], tracks=[track(0, 5, [[4, 6]])])
        for policy in POLICIES:
            self.assertEqual(new_policy(data, "star", policy)["selected_candidate_id"], 2)

    def test_original_incumbent_is_available_to_new_policy(self):
        data = row([obj(2, 0), obj(-1, 5), obj(1, 5)], current_id=2, tracks=[track(0, 5, [[4, 6]])])
        for policy in POLICIES:
            self.assertEqual(new_policy(data, "star", policy)["selected_candidate_id"], -1)

    def test_gate_changes_selection(self):
        data = row([obj(-1, 0), obj(0, 4), obj(1, 2)], intervals=[[2, 6]])
        self.assertEqual(new_policy(data, "star_full", "P")["selected_candidate_id"], 0)
        self.assertEqual(new_policy(data, "star_full", "GP")["selected_candidate_id"], 1)
        self.assertEqual(new_policy(data, "star_full", "GP")["eligible_ids"], [1])

    def test_gp_and_r_share_gate_but_rank_differently(self):
        data = row([obj(-1, 0), obj(0, 3), obj(1, 2)], intervals=[[2, 6]])
        gp, robust = [new_policy(data, "star_full", policy) for policy in ("GP", "R")]
        self.assertEqual(gp["eligible_ids"], robust["eligible_ids"])
        self.assertEqual(gp["gate_eligible_ids"], robust["gate_eligible_ids"])
        self.assertEqual(gp["selected_candidate_id"], 0)
        self.assertEqual(robust["selected_candidate_id"], 1)

    def test_singleton_uses_exact_historical_gain(self):
        data = row([obj(-1, 0), obj(0, 2)], intervals=[[1, 1]])
        for policy in ("P", "GP", "R"):
            result = new_policy(data, "star_full", policy)
            self.assertEqual(result["selected_candidate_id"], -1)
            self.assertEqual(next(s for s in result["scores"] if s["candidate_id"] == 0)["gain_low_mm2"], 0.0)

    def test_point_gain_handles_off_ray_objects(self):
        data = row([obj(-1, 0), obj(0, 5, x=100), obj(1, 4)], intervals=[[5, 5]])
        self.assertEqual(new_policy(data, "star_full", "P")["selected_candidate_id"], 1)

    def test_q_highest_mean_track_and_projection(self):
        data = row([obj(-1, 0), obj(0, 2), obj(1, 5)], intervals=[[1, 8]],
                   tracks=[track(1, 2, [[1, 3]], (.9, .6)), track(2, 9, [[4, 5]], (.8, .8))])
        result = new_policy(data, "star", "Q")
        self.assertEqual((result["track_id"], result["target_depth"], result["selected_candidate_id"]), (2, 5.0, 1))

    def test_q_track_tie_and_projection_tie(self):
        data = row(tracks=[track(4, 5, [[4, 6]]), track(2, 3, [[1, 2], [4, 5]])])
        result = new_policy(data, "star", "Q")
        self.assertEqual((result["track_id"], result["target_depth"]), (2, 2.0))

    def test_qg_uses_q_target_but_robust_gate(self):
        data = row([obj(-1, 0), obj(0, 4), obj(1, 2)], intervals=[[2, 6]], tracks=[track(1, 4, [[2, 6]])])
        q, qg = [new_policy(data, "cycle", policy) for policy in ("Q", "QG")]
        self.assertEqual(q["target_depth"], qg["target_depth"])
        self.assertEqual((q["selected_candidate_id"], qg["selected_candidate_id"]), (0, 1))

    def test_no_tracks_keeps(self):
        for policy in ("Q", "QG"):
            result = new_policy(row(), "star_full", policy)
            self.assertEqual((result["selected_candidate_id"], result["reason"]), (-1, "NO_TRACKS"))

    def test_input_not_mutated(self):
        data = row(tracks=[track(0, 5, [[4, 6]])])
        before = copy.deepcopy(data)
        for policy in POLICIES:
            new_policy(data, "star", policy)
        old_policy(data)
        self.assertEqual(data, before)


class OldPolicyTests(unittest.TestCase):
    def test_margin_boundary_inclusive(self):
        data = row([obj(-1, 0, .6), obj(0, 10, .65)])
        self.assertEqual(old_policy(data)["selected_candidate_id"], 0)
        data["objects"][1]["old_score"] = .65 - 2e-12
        self.assertEqual(old_policy(data)["reason"], "KEEP_INCUMBENT_MARGIN")

    def test_ambiguity_score_and_depth_boundaries_inclusive(self):
        data = row([obj(-1, 0, .5), obj(0, 10, .8), obj(1, 15, .75)])
        self.assertEqual(old_policy(data)["reason"], "KEEP_AMBIGUOUS_DEPTH")
        data["objects"][2]["depth_mm"] = 15 - 1e-10
        self.assertEqual(old_policy(data)["selected_candidate_id"], 0)
        data["objects"][2]["depth_mm"] = 15
        data["objects"][2]["old_score"] = .75 - 2e-12
        self.assertEqual(old_policy(data)["selected_candidate_id"], 0)

    def test_only_ambiguity_removed_preserves_margin(self):
        data = row([obj(-1, 0, .77), obj(0, 10, .8), obj(1, 15, .78)])
        self.assertEqual(old_policy(data, remove_ambiguity=True)["reason"], "KEEP_INCUMBENT_MARGIN")
        data["objects"][0]["old_score"] = .5
        self.assertEqual(old_policy(data)["selected_candidate_id"], -1)
        changed = old_policy(data, remove_ambiguity=True)
        self.assertEqual(changed["selected_candidate_id"], 0)
        self.assertEqual(changed["competing_candidate_ids"], [1])

    def test_only_margin_removed_preserves_ambiguity(self):
        data = row([obj(-1, 0, .77), obj(0, 10, .8), obj(1, 15, .78)])
        self.assertEqual(old_policy(data, remove_margin=True)["reason"], "KEEP_AMBIGUOUS_DEPTH")
        data["objects"].pop()
        self.assertEqual(old_policy(data, remove_margin=True)["selected_candidate_id"], 0)

    def test_old_compares_original_not_photo_current(self):
        data = row([obj(-1, 0, .5), obj(0, 10, .8), obj(1, 11, .79)], current_id=1)
        self.assertEqual(old_policy(data)["selected_candidate_id"], 0)

    def test_missing_current_exception_only_for_modified_arms(self):
        data = row([obj(0, 10, .8)], current_id=None, original_id=None)
        self.assertEqual(old_policy(data)["selected_candidate_id"], 0)
        for kwargs in (dict(remove_ambiguity=True), dict(remove_margin=True),
                       dict(remove_ambiguity=True, remove_margin=True)):
            self.assertIsNone(old_policy(data, **kwargs)["selected_candidate_id"])

    def test_nonfinite_old_scores_excluded_and_unscoreable_incumbent(self):
        data = row([obj(-1, 0, None), obj(0, 1, float("nan")), obj(1, 2, float("inf")), obj(2, 3, .7)])
        result = old_policy(data)
        self.assertEqual(result["eligible_ids"], [2])
        self.assertEqual((result["selected_candidate_id"], result["reason"]), (2, "MOVE_UNSCOREABLE_INCUMBENT"))

    def test_old_tie_smallest_id(self):
        data = row([obj(-1, 0, .5), obj(8, 10, .8), obj(2, 11, .8)])
        self.assertEqual(old_policy(data)["selected_candidate_id"], 2)

    def test_original_not_competing_candidate(self):
        data = row([obj(-1, 0, .75), obj(0, 10, .8)])
        result = old_policy(data)
        self.assertEqual((result["selected_candidate_id"], result["competing_candidate_ids"]), (0, []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
