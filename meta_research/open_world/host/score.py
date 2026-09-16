"""Frozen holdout scorer with public pre-drives.

Memory-branch items share a visible (x, v) after different public
pre-drives. The plant carries hidden memory from the pre-drive; the
policy never receives true m0. Predictions may estimate hidden IC
from the pre-drive, or roll out from m0=0 (indistinguishable control).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from host.artifact import Artifact, estimate_hidden_ic, simulate_artifact
from host.session import Session, start_state, t_grid_of, u_from_spec

try:
    from policies.nonsym import RidgeRollout, simulate_ridge
except ImportError:  # pragma: no cover
    RidgeRollout = None  # type: ignore
    simulate_ridge = None  # type: ignore


SHARED_XV = {"x": 0.4, "v": 0.0}

PREDRIVES = {
    "none": None,
    "pos": {
        "initial_state": {"x": 0.0, "v": 0.0},
        "input_signal": {"kind": "step", "amp": 1.1, "freq": 0.0, "phase": 0.0},
        "duration": 2.0,
        "sampling": {"dt": 0.05},
    },
    "neg": {
        "initial_state": {"x": 0.0, "v": 0.0},
        "input_signal": {"kind": "step", "amp": -1.1, "freq": 0.0, "phase": 0.0},
        "duration": 2.0,
        "sampling": {"dt": 0.05},
    },
}

INTERVENTION = {
    "input_signal": {"kind": "sine", "amp": 0.5, "freq": 0.35, "phase": 0.0},
    "duration": 4.0,
    "sampling": {"dt": 0.05},
}

HOLDOUT_MENU = [
    {
        "tag": "sine",
        "predrive": "none",
        "initial_state": {"x": 0.6, "v": 0.0},
        "input_signal": {"kind": "sine", "amp": 0.8, "freq": 0.25, "phase": 0.0},
        "duration": 5.0,
        "sampling": {"dt": 0.05},
    },
    {
        "tag": "step",
        "predrive": "none",
        "initial_state": {"x": -0.4, "v": 0.3},
        "input_signal": {"kind": "step", "amp": 0.7, "freq": 0.0, "phase": 0.0},
        "duration": 4.0,
        "sampling": {"dt": 0.05},
    },
    {
        "tag": "chirp",
        "predrive": "none",
        "initial_state": {"x": 0.0, "v": 0.8},
        "input_signal": {"kind": "chirp", "amp": 0.6, "freq": 0.2, "phase": 0.0},
        "duration": 5.0,
        "sampling": {"dt": 0.05},
    },
    {
        "tag": "memory_branch_pos",
        "predrive": "pos",
        "reset_visible": True,
        **INTERVENTION,
    },
    {
        "tag": "memory_branch_neg",
        "predrive": "neg",
        "reset_visible": True,
        **INTERVENTION,
    },
]


def _run_plant(session: Session, spec: Dict, z0: np.ndarray):
    t = t_grid_of(spec)
    u_fun = u_from_spec(spec)
    ts, y, z = session.plant.simulate(z0, t, u_fun)
    u = np.array([u_fun(float(tt)) for tt in ts], dtype=float)
    return ts, y, u, z


def public_history_and_truth(session: Session, spec: Dict):
    """Return (public history or None, truth y, intervention spec with visible x0,v0)."""
    key = spec.get("predrive", "none")
    pd = PREDRIVES.get(key)
    if not pd:
        z0 = session.plant.pack(
            float(spec["initial_state"]["x"]),
            float(spec["initial_state"]["v"]),
            session.init_mem,
        )
        ts, y, u, z = _run_plant(session, spec, z0)
        vis = {
            "initial_state": dict(spec["initial_state"]),
            "input_signal": dict(spec["input_signal"]),
            "duration": spec["duration"],
            "sampling": dict(spec["sampling"]),
        }
        return None, y, vis
    z0 = session.plant.pack(0.0, 0.0, 0.0)
    _ts, hy, hu, hz = _run_plant(session, pd, z0)
    history = {"t": _ts, "y": hy, "u": hu}
    vis = {
        "initial_state": dict(SHARED_XV) if spec.get("reset_visible") else {"x": float(hy[-1, 0]), "v": float(hy[-1, 1])},
        "input_signal": dict(spec["input_signal"]),
        "duration": spec["duration"],
        "sampling": dict(spec["sampling"]),
    }
    z_cont = hz[-1].copy()
    if spec.get("reset_visible"):
        z_cont[0] = float(SHARED_XV["x"])
        z_cont[1] = float(SHARED_XV["v"])
    ts, y, u, _z = _run_plant(session, vis, z_cont)
    return history, y, vis


def predict_artifact(cand, session: Session, spec: Dict, history, vis: Dict, use_history: bool) -> np.ndarray:
    if RidgeRollout is not None and isinstance(cand, RidgeRollout):
        return simulate_ridge(cand, vis)
    m0 = 0.0
    if use_history and isinstance(cand, Artifact) and cand.use_memory:
        m0 = estimate_hidden_ic(cand, history)
    t = t_grid_of(vis)
    x0, v0 = start_state(vis)
    return simulate_artifact(cand, t, u_from_spec(vis), x0, v0, m0=m0)


def scale_of(session: Session, menu: List[Dict] = None) -> float:
    menu = menu or HOLDOUT_MENU
    ys = [public_history_and_truth(session, spec)[1] for spec in menu]
    stacked = np.concatenate(ys, axis=0)
    rms = float(np.sqrt(np.mean(stacked ** 2)))
    return max(rms, 1e-6)


def score_candidate(
    session: Session, cand_hash: str, menu: List[Dict] = None, use_history: bool = True
) -> Dict:
    menu = menu or HOLDOUT_MENU
    cand = session.candidates[cand_hash]
    scale = scale_of(session, menu)
    rows = []
    nmses = []
    branch_preds = {}
    for spec in menu:
        history, y, vis = public_history_and_truth(session, spec)
        pred = predict_artifact(cand, session, spec, history, vis, use_history)
        n = min(len(pred), len(y))
        mse = float(np.mean((pred[:n] - y[:n]) ** 2))
        nmse = mse / (scale ** 2)
        nmses.append(nmse)
        tag = spec.get("tag", spec["input_signal"]["kind"])
        rows.append({"tag": tag, "mse": mse, "nmse": nmse})
        if tag in ("memory_branch_pos", "memory_branch_neg"):
            branch_preds[tag] = pred[:n].copy()
    gap = None
    if "memory_branch_pos" in branch_preds and "memory_branch_neg" in branch_preds:
        gap = float(
            np.mean(np.abs(branch_preds["memory_branch_pos"] - branch_preds["memory_branch_neg"]))
        )
    return {
        "scale": scale,
        "nmse_mean": float(np.mean(nmses)),
        "nmse_by_item": rows,
        "candidate_hash": cand_hash,
        "family": getattr(cand, "family", None),
        "use_memory": bool(getattr(cand, "use_memory", False)),
        "use_drift": bool(getattr(cand, "use_drift", False)),
        "acc_terms": list(getattr(cand, "acc_terms", [])),
        "hidden": list(getattr(cand, "hidden", [])),
        "obs_map": getattr(cand, "obs_map", None),
        "use_history": bool(use_history),
        "branch_pred_gap": gap,
    }


def _truth(session: Session, spec: Dict) -> np.ndarray:
    """Compatibility for isolation tests. Not a participant API."""
    return public_history_and_truth(session, spec)[1]
