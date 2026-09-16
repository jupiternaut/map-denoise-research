#!/usr/bin/env python3
"""Isolation, commitment-before-labels, and memory-branch diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from builder.plant import spec_by_id
from host.menu import COVER_PLAN
from host.session import Session, make_session
from host.score import HOLDOUT_MENU, _truth
from policies.four import _run, policy_A, policy_D


def test_honest_invariant_cheat_caught() -> None:
    spec = spec_by_id("C1_cubic_damp")
    sess = make_session(spec, init_mem=0.0)
    _run(sess, COVER_PLAN[0])

    def spec_hash_safe(sp):
        from host.session import spec_hash

        return spec_hash(sp)

    def next_cover(s: Session):
        seen = {rec["spec_hash"] for rec in s.experiments.values()}
        for sp in COVER_PLAN[1:]:
            if spec_hash_safe(sp) not in seen:
                return ("run", spec_hash_safe(sp))
        return None

    def next_cheat(s: Session):
        ys = []
        for sp in COVER_PLAN[1:]:
            y = s.peek_future(spec=sp)
            ys.append((float(np.max(np.abs(y))), spec_hash_safe(sp)))
        ys.sort(reverse=True)
        return ("run", ys[0][1])

    a = sess
    b = sess.clone_unqueried_rewritten()
    assert next_cover(a) == next_cover(b), "honest must be invariant"
    ca, cb = next_cheat(a), next_cheat(b)
    assert ca != cb, f"cheat must change after rewrite: {ca} vs {cb}"


def test_commit_before_labels() -> None:
    sess = make_session(spec_by_id("C0_linear"))
    rec = _run(sess, COVER_PLAN[0])
    fit = sess.fit("linear2", [rec["experiment_id"]])
    spec = COVER_PLAN[1]
    sess.submit_prediction(fit["candidate_hash"], spec)
    rec2 = _run(sess, spec)
    assert sess.commitments[0]["resolved"]
    assert sess.commitments[0]["committed_before_labels"]
    assert rec2["experiment_id"] == sess.commitments[0]["experiment_id"]
    try:
        sess.submit_prediction(fit["candidate_hash"], spec)
        raise SystemExit("commit after labels must fail")
    except RuntimeError:
        pass


def test_memory_branches_differ() -> None:
    spec = spec_by_id("C2_memory")
    sess = make_session(spec, init_mem=0.0)
    pos = [s for s in HOLDOUT_MENU if s.get("tag") == "memory_branch_pos"][0]
    neg = [s for s in HOLDOUT_MENU if s.get("tag") == "memory_branch_neg"][0]
    yp = _truth(sess, pos)
    yn = _truth(sess, neg)
    gap = float(np.mean(np.abs(yp - yn)))
    assert gap > 0.05, f"memory branches too close: {gap}"
    lin = make_session(spec_by_id("C0_linear"))
    gp = float(np.mean(np.abs(_truth(lin, pos) - _truth(lin, neg))))
    assert gp < 1e-9, f"linear plant should not remember: {gp}"


def test_fake_prediction_detected() -> None:
    sess = make_session(spec_by_id("C0_linear"))
    rec = _run(sess, COVER_PLAN[0])
    fit = sess.fit("linear2", [rec["experiment_id"]])
    spec = COVER_PLAN[2]
    sess.submit_prediction(fit["candidate_hash"], spec)
    sess.commitments[-1]["prediction"] = np.zeros_like(
        np.asarray(sess.commitments[-1]["prediction"])
    ).tolist()
    rec2 = _run(sess, spec)
    mse = sess.commitments[0]["realized_mse"]
    assert mse > 0.01, f"fake zero prediction should miss: {mse}"
    assert rec2["experiment_id"]


def test_policies_run() -> None:
    sess = make_session(spec_by_id("C0_linear"))
    policy_A(sess)
    assert sess.frozen and sess.final_hash
    sess2 = make_session(spec_by_id("C2_memory"), init_mem=0.0)
    policy_D(sess2)
    assert sess2.frozen
    assert any(c.get("committed_before_labels") for c in sess2.commitments) or sess2.n_calibrate >= 0


if __name__ == "__main__":
    test_honest_invariant_cheat_caught()
    test_commit_before_labels()
    test_memory_branches_differ()
    test_fake_prediction_detected()
    test_policies_run()
    print("TESTS_OK")
