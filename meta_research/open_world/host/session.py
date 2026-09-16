"""Participant-facing experiment session.

Trusted-algorithm isolation: hidden plant lives in this process, but
policies receive only API return values. This is NOT an OS sandbox.
Do not claim a closed-book LLM exam.

Predictions must be committed before the host will run that spec.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from builder.plant import HiddenPlant, HiddenSpec, make_u
from host.artifact import (
    Artifact,
    Candidate,
    estimate_hidden_ic,
    fit_artifact,
    fit_family,
    propose_from_residuals,
    search_library,
    simulate_artifact,
    train_rmse,
)

COST_RUN = 1
COST_REPEAT = 1
COST_CALIBRATE = 1
MAX_ACTIONS = 8
DEFAULT_DURATION = 4.0
DEFAULT_DT = 0.05


def spec_hash(spec: Dict) -> str:
    blob = json.dumps(spec, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def candidate_hash(cand: Candidate) -> str:
    blob = json.dumps(cand.to_dict(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def u_from_spec(spec: Dict) -> Callable[[float], float]:
    inp = spec.get("input_signal") or {}
    return make_u(
        str(inp.get("kind", "sine")),
        float(inp.get("amp", 1.0)),
        float(inp.get("freq", 0.4)),
        float(inp.get("phase", 0.0)),
    )


def t_grid_of(spec: Dict) -> np.ndarray:
    duration = float(spec.get("duration", DEFAULT_DURATION))
    sampling = spec.get("sampling") or {}
    dt = float(sampling.get("dt", DEFAULT_DT))
    n = max(3, int(round(duration / dt)) + 1)
    return np.linspace(0.0, duration, n)


def start_state(spec: Dict) -> Tuple[float, float]:
    st = spec.get("initial_state") or {}
    return float(st.get("x", 0.0)), float(st.get("v", 0.0))


@dataclass
class Session:
    plant: HiddenPlant
    max_actions: int = MAX_ACTIONS
    init_mem: float = 0.0
    log: List[Dict] = field(default_factory=list)
    experiments: Dict[str, Dict] = field(default_factory=dict)
    candidates: Dict[str, Candidate] = field(default_factory=dict)
    commitments: List[Dict] = field(default_factory=list)
    cost: int = 0
    n_run: int = 0
    n_repeat: int = 0
    n_calibrate: int = 0
    n_fit: int = 0
    wall_s: float = 0.0
    final_hash: Optional[str] = None
    frozen: bool = False

    def _bill(self, kind: str, amount: int = 1) -> None:
        if self.frozen:
            raise RuntimeError("session frozen after submit_final")
        if self.cost + amount > self.max_actions:
            raise RuntimeError(
                f"budget exhausted: cost={self.cost} max={self.max_actions}"
            )
        self.cost += amount
        if kind == "run":
            self.n_run += 1
        elif kind == "repeat":
            self.n_repeat += 1
        elif kind == "calibrate":
            self.n_calibrate += 1

    def _simulate_visible(self, spec: Dict, mem0: Optional[float] = None) -> Dict:
        t = t_grid_of(spec)
        x0, v0 = start_state(spec)
        u_fun = u_from_spec(spec)
        z0 = self.plant.pack(x0, v0, mem0 if mem0 is not None else self.init_mem)
        ts, y, z = self.plant.simulate(z0, t, u_fun)
        u = np.array([u_fun(float(tt)) for tt in ts], dtype=float)
        return {"t": ts, "y": y, "u": u, "hidden_z": z}

    def _public_trace(self, rec: Dict) -> Dict:
        return {
            "experiment_id": rec["experiment_id"],
            "spec": rec["spec"],
            "t": rec["t"].tolist(),
            "y": rec["y"].tolist(),
            "u": rec["u"].tolist(),
            "cost": rec["cost"],
            "kind": rec["kind"],
        }

    def run_experiment(self, initial_state, input_signal, duration, sampling) -> Dict:
        t0 = time.perf_counter()
        spec = {
            "initial_state": dict(initial_state),
            "input_signal": dict(input_signal),
            "duration": float(duration),
            "sampling": dict(sampling),
        }
        hid = spec_hash(spec)
        pending = [
            c
            for c in self.commitments
            if c["spec_hash"] == hid and not c.get("resolved")
        ]
        self._bill("run", COST_RUN)
        sim = self._simulate_visible(spec)
        eid = f"E{len(self.experiments):03d}"
        rec = {
            "experiment_id": eid,
            "spec": spec,
            "spec_hash": hid,
            "kind": "run",
            "cost": COST_RUN,
            **sim,
        }
        self.experiments[eid] = rec
        for c in pending:
            pred = np.asarray(c["prediction"], dtype=float)
            y = sim["y"]
            n = min(len(pred), len(y))
            c["resolved"] = True
            c["experiment_id"] = eid
            c["realized_mse"] = float(np.mean((pred[:n] - y[:n]) ** 2))
        self.wall_s += time.perf_counter() - t0
        self.log.append({"op": "run_experiment", "id": eid, "spec_hash": hid})
        return self._public_trace(rec)

    def repeat_experiment(self, experiment_id: str) -> Dict:
        t0 = time.perf_counter()
        if experiment_id not in self.experiments:
            raise KeyError(experiment_id)
        src = self.experiments[experiment_id]
        self._bill("repeat", COST_REPEAT)
        sim = self._simulate_visible(src["spec"])
        eid = f"E{len(self.experiments):03d}"
        rec = {
            "experiment_id": eid,
            "spec": src["spec"],
            "spec_hash": src["spec_hash"],
            "kind": "repeat",
            "repeats": experiment_id,
            "cost": COST_REPEAT,
            **sim,
        }
        self.experiments[eid] = rec
        self.wall_s += time.perf_counter() - t0
        self.log.append({"op": "repeat_experiment", "id": eid, "src": experiment_id})
        return self._public_trace(rec)

    def calibrate(self, protocol: str) -> Dict:
        """Physically realizable calibration. Does not return true bias.

        rest — zero input from (0,0)
        hold — zero input from last measured (x,v)
        """
        t0 = time.perf_counter()
        protocol = str(protocol)
        self._bill("calibrate", COST_CALIBRATE)
        if protocol == "rest":
            spec = {
                "initial_state": {"x": 0.0, "v": 0.0},
                "input_signal": {"kind": "zero", "amp": 0.0, "freq": 0.0},
                "duration": 2.0,
                "sampling": {"dt": DEFAULT_DT},
            }
            mem0 = 0.0
        elif protocol == "hold":
            last = None
            for rec in self.experiments.values():
                last = rec
            if last is None:
                x0, v0 = 0.0, 0.0
            else:
                x0 = float(last["y"][-1, 0])
                v0 = float(last["y"][-1, 1])
            spec = {
                "initial_state": {"x": x0, "v": v0},
                "input_signal": {"kind": "zero", "amp": 0.0, "freq": 0.0},
                "duration": 2.0,
                "sampling": {"dt": DEFAULT_DT},
            }
            mem0 = None
        else:
            raise ValueError(protocol)
        sim = self._simulate_visible(spec, mem0=mem0)
        eid = f"E{len(self.experiments):03d}"
        rec = {
            "experiment_id": eid,
            "spec": spec,
            "spec_hash": spec_hash(spec),
            "kind": "calibrate",
            "protocol": protocol,
            "cost": COST_CALIBRATE,
            **sim,
        }
        self.experiments[eid] = rec
        y = sim["y"]
        summary = {
            "mean_y": y.mean(axis=0).tolist(),
            "delta_x": float(y[-1, 0] - y[0, 0]),
            "delta_v": float(y[-1, 1] - y[0, 1]),
        }
        self.wall_s += time.perf_counter() - t0
        self.log.append({"op": "calibrate", "id": eid, "protocol": protocol})
        out = self._public_trace(rec)
        out["summary"] = summary
        return out

    def fit(self, family: str, observed_experiment_ids: List[str], structure: Optional[Dict] = None) -> Dict:
        t0 = time.perf_counter()
        traces = []
        for eid in observed_experiment_ids:
            rec = self.experiments[eid]
            traces.append({"t": rec["t"], "y": rec["y"], "u": rec["u"]})
        if structure is not None:
            cand = fit_artifact(structure, traces)
        elif family == "all":
            cand = search_library(traces)[0]
        elif family == "propose":
            cand = propose_from_residuals(traces)
        else:
            cand = fit_family(family, traces)
        h = candidate_hash(cand)
        cand.hash = h
        self.candidates[h] = cand
        self.n_fit += 1
        err = train_rmse(cand, traces)
        self.wall_s += time.perf_counter() - t0
        self.log.append({"op": "fit", "family": family, "hash": h, "train_rmse": err, "structure": cand.spec_key()})
        return {
            "candidate_hash": h,
            "family": cand.family,
            "theta": cand.theta,
            "use_memory": cand.use_memory,
            "use_drift": cand.use_drift,
            "acc_terms": list(cand.acc_terms),
            "hidden": list(cand.hidden),
            "obs_map": cand.obs_map,
            "train_rmse": err,
            "n_traces": len(traces),
        }

    def submit_prediction(
        self,
        candidate_hash_s: str,
        experiment_spec: Dict,
        history: Optional[Dict] = None,
        use_history: bool = True,
    ) -> Dict:
        if candidate_hash_s not in self.candidates:
            raise KeyError(candidate_hash_s)
        hid = spec_hash(experiment_spec)
        if any(rec["spec_hash"] == hid for rec in self.experiments.values()):
            raise RuntimeError("cannot commit a prediction after that experiment ran")
        cand = self.candidates[candidate_hash_s]
        t = t_grid_of(experiment_spec)
        x0, v0 = start_state(experiment_spec)
        u_fun = u_from_spec(experiment_spec)
        m0 = 0.0
        if use_history and isinstance(cand, Artifact) and cand.use_memory:
            m0 = estimate_hidden_ic(cand, history)
        pred = simulate_artifact(cand, t, u_fun, x0, v0, m0=m0)
        cid = f"P{len(self.commitments):03d}"
        self.commitments.append(
            {
                "id": cid,
                "candidate_hash": candidate_hash_s,
                "spec": experiment_spec,
                "spec_hash": hid,
                "prediction": pred.tolist(),
                "resolved": False,
                "committed_before_labels": True,
                "m0_used": float(m0),
                "use_history": bool(use_history),
            }
        )
        self.log.append({"op": "submit_prediction", "id": cid, "spec_hash": hid, "m0_used": float(m0)})
        return {
            "commitment_id": cid,
            "spec_hash": hid,
            "candidate_hash": candidate_hash_s,
            "n_steps": int(len(t)),
            "m0_used": float(m0),
        }

    def submit_final(self, candidate_hash_s: str) -> Dict:
        if candidate_hash_s not in self.candidates:
            raise KeyError(candidate_hash_s)
        self.final_hash = candidate_hash_s
        self.frozen = True
        self.log.append({"op": "submit_final", "hash": candidate_hash_s})
        return {"final_hash": candidate_hash_s, "cost": self.cost, "frozen": True}

    def peek_future(self, spec: Dict) -> np.ndarray:
        """Cheat: read labels of an unrun experiment."""
        sim = self._simulate_visible(spec)
        if getattr(self, "_rewrite", False):
            y = sim["y"].copy()
            t = sim["t"]
            y[:, 0] = 17.0 * t + 3.0
            y[:, 1] = -11.0 * t
            return y
        return sim["y"]

    def clone_unqueried_rewritten(self) -> "Session":
        other = Session(
            plant=HiddenPlant(self.plant.spec),
            max_actions=self.max_actions,
            init_mem=self.init_mem,
        )
        other.cost = self.cost
        other.n_run = self.n_run
        other.n_repeat = self.n_repeat
        other.n_calibrate = self.n_calibrate
        other.n_fit = self.n_fit
        other.candidates = dict(self.candidates)
        other.commitments = [dict(c) for c in self.commitments]
        other.final_hash = self.final_hash
        other.frozen = self.frozen
        other.experiments = {}
        for eid, rec in self.experiments.items():
            rec2 = dict(rec)
            rec2["y"] = rec["y"].copy()
            rec2["t"] = rec["t"].copy()
            rec2["u"] = rec["u"].copy()
            rec2["hidden_z"] = rec["hidden_z"].copy()
            other.experiments[eid] = rec2
        other._rewrite = True  # type: ignore[attr-defined]
        return other


def make_session(
    spec: HiddenSpec, init_mem: float = 0.0, max_actions: int = MAX_ACTIONS
) -> Session:
    return Session(plant=HiddenPlant(spec), max_actions=max_actions, init_mem=init_mem)
