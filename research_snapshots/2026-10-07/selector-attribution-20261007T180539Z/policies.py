"""Pure, GT-free selectors for the frozen attribution experiment.

No data are loaded here. The sole imported historical dependency is the frozen
exact-affine gain kernel. All coordinates, intervals, and scores are supplied
by the caller; every selector returns an existing object ID or its KEEP ID.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
KERNEL_PATH = Path(
    "/srv/slam-research/grf/map-denoise/runs/"
    "track-discrimination-20261007T160716Z/theory/kernel.py"
)
_spec = importlib.util.spec_from_file_location("_frozen_attribution_gain", KERNEL_PATH)
_kernel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kernel)
gain_bounds_world = _kernel.gain_bounds_world

POLICIES = ("P", "GP", "R", "Q", "QG")
EVIDENCE = ("star_full", "star", "cycle")
MOVE_MARGIN = 0.05
COMPETING_DEPTH_MM = 5.0
COMPETING_SCORE_GAP = 0.05
THRESHOLD_TOLERANCE = 1e-12


def _finite(value):
    return value is not None and math.isfinite(float(value))


def merge_intervals(intervals):
    """Return the finite closed interval union, including isolated singletons."""
    ordered = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("an interval must have two endpoints")
        lo, hi = map(float, interval)
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            raise ValueError("interval endpoints must be finite and ordered")
        ordered.append([lo, hi])
    merged = []
    for lo, hi in sorted(ordered):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def uniform_set_mean(intervals):
    """Length-uniform union mean; unique-singleton mean if total length is zero."""
    merged = merge_intervals(intervals)
    if not merged:
        return None
    lengths = [hi - lo for lo, hi in merged]
    total = math.fsum(lengths)
    if total == 0.0:
        return math.fsum(lo for lo, _ in merged) / len(merged)
    return math.fsum(
        length * (lo + (hi - lo) / 2.0)
        for (lo, hi), length in zip(merged, lengths)
    ) / total


def project_depth(depth, intervals):
    """Nearest depth in a closed interval union; lower depth wins equal distance."""
    depth = float(depth)
    if not math.isfinite(depth):
        raise ValueError("target depth must be finite")
    merged = merge_intervals(intervals)
    if not merged:
        return None
    candidates = [min(max(depth, lo), hi) for lo, hi in merged]
    return min(candidates, key=lambda value: (abs(value - depth), value))


def _objects(row):
    objects = list(row["objects"])
    ids = [obj["candidate_id"] for obj in objects]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in ids):
        raise ValueError("candidate IDs must be integers")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    return objects, {obj["candidate_id"]: obj for obj in objects}


def _track_target(tracks):
    ranked = []
    for track in tracks:
        ncc = track["ncc_reference_sources"]
        if len(ncc) != 2 or not all(_finite(value) for value in ncc):
            raise ValueError("a saved track must have two finite reference-source NCCs")
        depth = track["reference_depth_mm"]
        if not _finite(depth):
            raise ValueError("a saved track must have finite reference depth")
        support = merge_intervals(track["intervals_mm"])
        if not support:
            raise ValueError("a saved track must have nonempty support")
        mean_ncc = math.fsum(map(float, ncc)) / 2.0
        ranked.append((mean_ncc, track["id"], track, support))
    if not ranked:
        return None
    mean_ncc, track_id, track, support = min(ranked, key=lambda item: (-item[0], item[1]))
    return dict(
        track_id=track_id,
        track_mean_ncc=mean_ncc,
        track_reference_depth=float(track["reference_depth_mm"]),
        track_intervals=support,
        target_depth=project_depth(track["reference_depth_mm"], support),
    )


def new_policy(row, evidence, policy):
    """Select by point ranking (P/Q), its uniform-gain gate (GP/QG), or R.

    All five strategies start at row.current_id and share the full object pool.
    P/GP use the interval-union mean. Q/QG use the highest-NCC saved track and
    project its triangulated reference depth to its own interval support. R
    orders uniformly improving objects by their worst-case gain. Point gains
    are conservative lower endpoints from the historical singleton kernel.
    """
    if evidence not in EVIDENCE or policy not in POLICIES:
        raise ValueError("unknown evidence or policy")
    objects, lookup = _objects(row)
    current_id = row["current_id"]
    intervals = merge_intervals(row["intervals"][evidence])
    result = dict(
        evidence=evidence,
        policy=policy,
        current_id=current_id,
        selected_candidate_id=current_id,
        reason="NO_CURRENT_OUTPUT",
        target_depth=None,
        track_id=None,
        eligible_ids=[],
        gate_eligible_ids=[],
        scores=[],
        intervals=intervals,
    )
    if current_id not in lookup:
        return result
    if not intervals:
        result["reason"] = "EMPTY_UNKNOWN"
        return result
    if policy in ("P", "GP"):
        result["target_depth"] = uniform_set_mean(intervals)
    elif policy in ("Q", "QG"):
        target = _track_target(row.get("tracks", {}).get(evidence, []))
        if target is None:
            result["reason"] = "NO_TRACKS"
            return result
        result.update(target)

    current = lookup[current_id]["xyz_mm"]
    for obj in sorted(objects, key=lambda item: item["candidate_id"]):
        cid = obj["candidate_id"]
        low, high = gain_bounds_world(
            current, obj["xyz_mm"], row["center"], row["ray"], intervals
        )
        point_low = point_high = None
        if result["target_depth"] is not None:
            target = result["target_depth"]
            point_low, point_high = gain_bounds_world(
                current, obj["xyz_mm"], row["center"], row["ray"], [[target, target]]
            )
        gate_pass = cid != current_id and low > 0.0
        point_pass = cid != current_id and point_low is not None and point_low > 0.0
        ranking_score = low if policy == "R" else point_low
        eligible = gate_pass if policy == "R" else point_pass
        if policy in ("GP", "QG"):
            eligible = eligible and gate_pass
        result["scores"].append(dict(
            candidate_id=cid,
            gain_low_mm2=low,
            gain_high_mm2=high,
            point_gain_low_mm2=point_low,
            point_gain_high_mm2=point_high,
            ranking_score=ranking_score,
            gate_pass=gate_pass,
            positive_target_gain=point_pass,
            eligible=eligible,
        ))
    result["gate_eligible_ids"] = [score["candidate_id"] for score in result["scores"] if score["gate_pass"]]
    eligible = [score for score in result["scores"] if score["eligible"]]
    result["eligible_ids"] = [score["candidate_id"] for score in eligible]
    if not eligible:
        result["reason"] = "NO_UNIFORM_MODEL_GAIN" if policy in ("GP", "QG", "R") else "NO_POSITIVE_POINT_GAIN"
        return result
    winner = min(eligible, key=lambda score: (-score["ranking_score"], score["candidate_id"]))
    result.update(
        selected_candidate_id=winner["candidate_id"],
        reason="POSITIVE_GAIN_ON_ASSUMED_SET" if policy in ("GP", "QG", "R") else "POSITIVE_POINT_GAIN",
        winning_score=winner["ranking_score"],
    )
    return result


def old_policy(row, remove_ambiguity=False, remove_margin=False):
    """Replay the historical photometric policy with optional single gate removal.

    This always compares non--1 proposals against the original -1 incumbent,
    not the current photo-selected object. KEEP returns original_id. Modified
    policies additionally retain a missing current photo output as requested.
    """
    objects, lookup = _objects(row)
    original_id = row["original_id"]
    if original_id not in (-1, None):
        raise ValueError("original_id must be -1 or null")
    if original_id is not None and original_id not in lookup:
        raise ValueError("original incumbent ID is absent from the object pool")
    incumbent = lookup.get(original_id)
    incumbent_score = None if incumbent is None else incumbent.get("old_score")
    incumbent_score = float(incumbent_score) if _finite(incumbent_score) else None
    eligible = [obj for obj in objects if obj["candidate_id"] != -1 and _finite(obj.get("old_score"))]
    eligible.sort(key=lambda obj: (-float(obj["old_score"]), obj["candidate_id"]))
    result = dict(
        policy="OLD",
        remove_ambiguity=bool(remove_ambiguity),
        remove_margin=bool(remove_margin),
        original_id=original_id,
        current_id=row["current_id"],
        selected_candidate_id=original_id,
        reason="KEEP_NO_ADMISSIBLE_CANDIDATE",
        target_depth=None,
        winning_candidate_id=None,
        winning_score=None,
        incumbent_score=incumbent_score,
        score_gap=None,
        competing_candidate_ids=[],
        eligible_ids=[obj["candidate_id"] for obj in eligible],
        admissible_candidate_ids=[obj["candidate_id"] for obj in eligible],
        scores=[dict(candidate_id=obj["candidate_id"], old_score=obj.get("old_score"),
                     admissible=obj["candidate_id"] != -1 and _finite(obj.get("old_score")))
                for obj in sorted(objects, key=lambda item: item["candidate_id"])],
    )
    if (remove_ambiguity or remove_margin) and row["current_id"] is None:
        result.update(selected_candidate_id=None, reason="NO_CURRENT_OUTPUT")
        return result
    if not eligible:
        return result
    winner = eligible[0]
    winning_score = float(winner["old_score"])
    result.update(winning_candidate_id=winner["candidate_id"], winning_score=winning_score)
    if incumbent_score is not None:
        gap = winning_score - incumbent_score
        result["score_gap"] = gap
        if not remove_margin and gap < MOVE_MARGIN - THRESHOLD_TOLERANCE:
            result["reason"] = "KEEP_INCUMBENT_MARGIN"
            return result
    competitors = [
        obj["candidate_id"] for obj in eligible[1:]
        if abs(float(obj["depth_mm"]) - float(winner["depth_mm"])) >= COMPETING_DEPTH_MM
        and winning_score - float(obj["old_score"]) <= COMPETING_SCORE_GAP + THRESHOLD_TOLERANCE
    ]
    result["competing_candidate_ids"] = competitors
    if competitors and not remove_ambiguity:
        result["reason"] = "KEEP_AMBIGUOUS_DEPTH"
        return result
    result.update(selected_candidate_id=winner["candidate_id"],
                  reason="MOVE" if incumbent_score is not None else "MOVE_UNSCOREABLE_INCUMBENT")
    return result
