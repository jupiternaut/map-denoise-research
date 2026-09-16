#!/usr/bin/env python3
"""Capability-upgrade checks. Construction six is smoke, not this score."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from builder.plant import spec_by_id
from host.artifact import (
    ACC_BASE,
    FAMILY_SPEC,
    fit_artifact,
    legal_structures,
    propose_from_residuals,
    search_library,
)
from host.score import HOLDOUT_MENU, predict_artifact, public_history_and_truth
from host.session import make_session
from policies.capability import covering, policy_library, policy_propose


def test_fitter_accepts_unlisted_combination() -> None:
    """A spec absent from FAMILY_SPEC still fits. No family if/elif."""
    named = {tuple(v["acc_terms"]) + v["hidden"] + (v["obs_map"],) for v in FAMILY_SPEC.values()}
    spec = {"acc_terms": ACC_BASE + ("x3", "m"), "hidden": ("m",), "obs_map": "identity"}
    key = tuple(spec["acc_terms"]) + spec["hidden"] + (spec["obs_map"],)
    assert key not in named
    sess = make_session(spec_by_id("C2_memory"))
    covering(sess)
    traces = [{"t": r["t"], "y": r["y"], "u": r["u"]} for r in sess.experiments.values()]
    art = fit_artifact(spec, traces)
    assert art.use_memory
    assert "x3" in art.acc_terms
    assert "m" in art.acc_terms
    assert art.family not in FAMILY_SPEC


def test_legal_pool_is_combinations_not_named_families() -> None:
    pool = legal_structures()
    assert len(pool) > len(FAMILY_SPEC)
    assert any(
        s["hidden"] == ("m",) and s["obs_map"] == "linear_time" and "x3" in s["acc_terms"]
        for s in pool
    )


def test_history_changes_branch_predictions() -> None:
    sess = make_session(spec_by_id("C2_memory"))
    covering(sess)
    rec = sess.fit("propose", list(sess.experiments.keys()))
    cand = sess.candidates[rec["candidate_hash"]]
    pos = [s for s in HOLDOUT_MENU if s["tag"] == "memory_branch_pos"][0]
    neg = [s for s in HOLDOUT_MENU if s["tag"] == "memory_branch_neg"][0]
    hp, yp, visp = public_history_and_truth(sess, pos)
    hn, yn, visn = public_history_and_truth(sess, neg)
    assert hp is not None and hn is not None
    assert visp["initial_state"] == visn["initial_state"]
    truth_gap = float(np.mean(np.abs(yp - yn)))
    assert truth_gap > 0.05, f"truth branches too close: {truth_gap}"
    if cand.use_memory:
        pred_h_p = predict_artifact(cand, sess, pos, hp, visp, True)
        pred_h_n = predict_artifact(cand, sess, neg, hn, visn, True)
        pred_0_p = predict_artifact(cand, sess, pos, hp, visp, False)
        pred_0_n = predict_artifact(cand, sess, neg, hn, visn, False)
        gap_h = float(np.mean(np.abs(pred_h_p - pred_h_n)))
        gap_0 = float(np.mean(np.abs(pred_0_p - pred_0_n)))
        assert gap_0 < 1e-9, f"no-history must stay indistinguishable: {gap_0}"
        assert gap_h > 1e-6, f"history must change branch forecasts: {gap_h}"
    else:
        pred_h_p = predict_artifact(cand, sess, pos, hp, visp, True)
        pred_h_n = predict_artifact(cand, sess, neg, hn, visn, True)
        gap_h = float(np.mean(np.abs(pred_h_p - pred_h_n)))
        assert gap_h < 1e-9


def test_policies_share_primitives() -> None:
    a = make_session(spec_by_id("C1_cubic_damp"))
    b = make_session(spec_by_id("C1_cubic_damp"))
    ra = policy_library(a)
    rb = policy_propose(b)
    assert a.frozen and b.frozen
    la = set(a.candidates[ra["candidate_hash"]].acc_terms)
    lp = set(b.candidates[rb["candidate_hash"]].acc_terms)
    allowed = set(ACC_BASE) | {"x3", "v3", "m"}
    assert la <= allowed and lp <= allowed


if __name__ == "__main__":
    test_fitter_accepts_unlisted_combination()
    test_legal_pool_is_combinations_not_named_families()
    test_history_changes_branch_predictions()
    test_policies_share_primitives()
    print("CAPABILITY_TESTS_OK")
