"""Four policies on the public API.

A: wide library + covering plan (fixed after init, here init is empty so
   the covering menu is the locked open-loop).
B: same library, chooses next experiment from family disagreement.
C: structure search (same primitives as D) + same covering plan.
D: structure search + adaptive experiment after a committed prediction.

A/B vs C/D is a representation change. D-C is the sequential increment
under shared constructors. D-A is a system comparison, not attributed
to scheduling alone.

Trusted algorithm experiment: policies run in-process against Session.
"""
from __future__ import annotations

from typing import Dict, List

from host.candidates import simulate_candidate
from host.menu import COVER_PLAN, PROBE_SPECS
from host.session import Session, spec_hash, start_state, t_grid_of, u_from_spec

import numpy as np

WIDE_FAMILIES = ("linear2", "nl2")
OPEN_FAMILIES = ("linear2", "nl2", "memory3", "obs_drift")


def _ids(session: Session) -> List[str]:
    return list(session.experiments.keys())


def _fit_all(session: Session, families) -> Dict[str, Dict]:
    ids = _ids(session)
    out = {}
    for fam in families:
        out[fam] = session.fit(fam, ids)
    return out


def _best_of(fits: Dict[str, Dict]) -> str:
    return min(fits.values(), key=lambda r: r["train_rmse"])["candidate_hash"]


def _pred_on(session: Session, cand_hash: str, spec: Dict) -> np.ndarray:
    cand = session.candidates[cand_hash]
    t = t_grid_of(spec)
    x0, v0 = start_state(spec)
    return simulate_candidate(cand, t, u_from_spec(spec), x0, v0)


def _disagreement(session: Session, fits: Dict[str, Dict], spec: Dict) -> float:
    preds = [_pred_on(session, r["candidate_hash"], spec) for r in fits.values()]
    stack = np.stack(preds, axis=0)
    return float(np.mean(np.var(stack, axis=0)))


def _remaining_plan(session: Session, plan: List[Dict]) -> List[Dict]:
    seen = {rec["spec_hash"] for rec in session.experiments.values()}
    out = []
    for spec in plan:
        if spec_hash(spec) not in seen:
            out.append(spec)
    return out


def _run(session: Session, spec: Dict) -> Dict:
    return session.run_experiment(
        spec["initial_state"], spec["input_signal"], spec["duration"], spec["sampling"]
    )


def policy_A(session: Session) -> Dict:
    """Wide 2-D library (no new state), covering plan."""
    fits = None
    for spec in COVER_PLAN:
        if session.cost >= session.max_actions:
            break
        _run(session, spec)
        fits = _fit_all(session, WIDE_FAMILIES)
    if not session.experiments:
        return {"policy": "A", "final": None}
    final = _best_of(fits)
    session.submit_final(final)
    return {"policy": "A", "final": final, "fits": fits}


def policy_B(session: Session) -> Dict:
    """Same 2-D library as A; next spec maximises family disagreement."""
    _run(session, COVER_PLAN[0])
    fits = _fit_all(session, WIDE_FAMILIES)
    while session.cost < session.max_actions:
        remaining = _remaining_plan(session, PROBE_SPECS)
        if not remaining:
            break
        scores = [(_disagreement(session, fits, spec), i, spec) for i, spec in enumerate(remaining)]
        scores.sort(key=lambda z: (-z[0], z[1]))
        spec = scores[0][2]
        try:
            session.submit_prediction(_best_of(fits), spec)
        except RuntimeError:
            pass
        _run(session, spec)
        fits = _fit_all(session, WIDE_FAMILIES)
    final = _best_of(fits)
    session.submit_final(final)
    return {"policy": "B", "final": final, "fits": fits}


def policy_C(session: Session) -> Dict:
    """Open constructors (same as D), covering plan frozen."""
    fits = None
    for spec in COVER_PLAN:
        if session.cost >= session.max_actions:
            break
        _run(session, spec)
        fits = _fit_all(session, OPEN_FAMILIES)
    if not session.experiments:
        return {"policy": "C", "final": None}
    final = _best_of(fits)
    session.submit_final(final)
    return {"policy": "C", "final": final, "fits": fits}


def _commit_and_run(session: Session, cand_hash: str, spec: Dict) -> None:
    try:
        session.submit_prediction(cand_hash, spec)
    except RuntimeError:
        pass
    _run(session, spec)


def policy_D(session: Session) -> Dict:
    """Propose structure, commit a prediction, then intervene.

    Diagnostic fork (public, not a hidden answer key):
    - if memory-family train error is much lower, split drive history
      from the same (x,v);
    - if drift-family is better, spend a rest calibration;
    - else take the covering leftover with largest family disagreement.
    """
    _run(session, COVER_PLAN[0])
    fits = _fit_all(session, OPEN_FAMILIES)
    did_cal = False
    did_mem_split = False
    while session.cost < session.max_actions:
        best = min(fits.values(), key=lambda r: r["train_rmse"])
        mem = fits["memory3"]["train_rmse"]
        drift = fits["obs_drift"]["train_rmse"]
        lin = fits["linear2"]["train_rmse"]
        if (not did_cal) and drift < 0.7 * lin and drift <= mem:
            did_cal = True
            session.calibrate("rest")
            fits = _fit_all(session, OPEN_FAMILIES)
            continue
        if (not did_mem_split) and mem < 0.7 * lin:
            did_mem_split = True
            spec = {
                "initial_state": {"x": 0.5, "v": 0.0},
                "input_signal": {"kind": "sine", "amp": 0.5, "freq": 0.35, "phase": 0.0},
                "duration": 3.0,
                "sampling": {"dt": 0.05},
            }
            _commit_and_run(session, best["candidate_hash"], spec)
            if session.cost < session.max_actions:
                spec2 = {
                    "initial_state": {"x": 0.5, "v": 0.0},
                    "input_signal": {"kind": "step", "amp": 0.9, "freq": 0.0, "phase": 0.0},
                    "duration": 3.0,
                    "sampling": {"dt": 0.05},
                }
                _commit_and_run(session, best["candidate_hash"], spec2)
            fits = _fit_all(session, OPEN_FAMILIES)
            continue
        remaining = _remaining_plan(session, COVER_PLAN)
        if not remaining:
            break
        scores = [(_disagreement(session, fits, spec), i, spec) for i, spec in enumerate(remaining)]
        scores.sort(key=lambda z: (-z[0], z[1]))
        _commit_and_run(session, best["candidate_hash"], scores[0][2])
        fits = _fit_all(session, OPEN_FAMILIES)
    final = _best_of(fits)
    session.submit_final(final)
    return {"policy": "D", "final": final, "fits": fits}


POLICIES = {
    "A": policy_A,
    "B": policy_B,
    "C": policy_C,
    "D": policy_D,
}
