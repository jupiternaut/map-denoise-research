"""Ridge lag-1 residual predictor. Non-symbolic baseline, not PySINDy.

Rolls out y_{k+1} = W [1, y_k, u_k, y_k**3]. No hidden plant access.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.linear_model import Ridge

from host.menu import COVER_PLAN
from host.session import Session, start_state, t_grid_of, u_from_spec
from policies.four import _run


def _phi(y: np.ndarray, u: float) -> np.ndarray:
    x, v = float(y[0]), float(y[1])
    return np.array([1.0, x, v, float(u), x**3, v**3], dtype=float)


class RidgeRollout:
    family = "ridge_lag1"
    use_memory = False
    use_drift = False

    def __init__(self, W: np.ndarray):
        self.W = W
        self.theta = {"kind": "ridge_lag1"}
        self.hash = "ridge_lag1"

    def to_dict(self):
        return {"family": self.family, "theta": dict(self.theta)}


def fit_ridge(traces: List[Dict]) -> RidgeRollout:
    xs, ys = [], []
    for tr in traces:
        y = np.asarray(tr["y"], dtype=float)
        u = np.asarray(tr["u"], dtype=float)
        for i in range(len(y) - 1):
            xs.append(_phi(y[i], u[i]))
            ys.append(y[i + 1])
    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    model = Ridge(alpha=1e-3, fit_intercept=False)
    model.fit(X, Y)
    return RidgeRollout(model.coef_.copy())


def simulate_ridge(model: RidgeRollout, spec: Dict) -> np.ndarray:
    t = t_grid_of(spec)
    x0, v0 = start_state(spec)
    u_fun = u_from_spec(spec)
    y = np.zeros((len(t), 2), dtype=float)
    y[0] = np.array([x0, v0])
    for i in range(len(t) - 1):
        y[i + 1] = model.W @ _phi(y[i], u_fun(float(t[i])))
    return y


def policy_ridge(session: Session) -> Dict:
    for spec in COVER_PLAN:
        if session.cost >= session.max_actions:
            break
        _run(session, spec)
    traces = [
        {"t": rec["t"], "y": rec["y"], "u": rec["u"]}
        for rec in session.experiments.values()
    ]
    model = fit_ridge(traces)
    session.candidates[model.hash] = model  # type: ignore[assignment]
    session.submit_final(model.hash)
    return {"policy": "ridge", "final": model.hash}
