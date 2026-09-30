"""Same-capability policies: exhaustive library vs residual proposer.

Both share the public primitive inventory. The library enumerates every
legal combination; the proposer starts from (x,v,u) and composes.
Neither is given true m0. History estimation is applied at scoring,
not as a privileged label.
"""
from __future__ import annotations

from typing import Dict, List

from host.menu import COVER_PLAN
from host.session import Session, spec_hash


def _ids(session: Session) -> List[str]:
    return list(session.experiments.keys())


def _run(session: Session, spec: Dict) -> Dict:
    h = spec_hash(spec)
    for rec in session.experiments.values():
        if rec["spec_hash"] == h:
            return {"experiment_id": rec["experiment_id"], "already": True}
    if session.candidates:
        hsh = next(iter(session.candidates))
        try:
            session.submit_prediction(hsh, spec)
        except RuntimeError:
            pass
    return session.run_experiment(
        spec["initial_state"], spec["input_signal"], spec["duration"], spec["sampling"]
    )


def covering(session: Session) -> None:
    for spec in COVER_PLAN:
        if session.cost >= session.max_actions:
            break
        _run(session, spec)


def policy_library(session: Session) -> Dict:
    covering(session)
    rec = session.fit("all", _ids(session))
    session.submit_final(rec["candidate_hash"])
    rec["policy"] = "library"
    return rec


def policy_propose(session: Session) -> Dict:
    covering(session)
    rec = session.fit("propose", _ids(session))
    session.submit_final(rec["candidate_hash"])
    rec["policy"] = "propose"
    return rec
