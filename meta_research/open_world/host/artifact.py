"""Composable executable artifacts. No family if/elif in the fitter.

A candidate is a subset of public primitives, not a named world id
and not a pre-written family string. Adding a new legal combination
does not require editing this fitter.

ASSUMED public primitives:
  acc terms: x, v, u, x**3, v**3, m
  hidden: none | leaky drive memory (dm = -decay*m + u)
  obs: yx = x + b0 + b1*t   (b1 may be 0)

Hidden initial state is estimated from a provided pre-drive (t,y,u).
The true m0 is never an argument.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain, combinations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.linalg import lstsq

from host.sim_core import DT_MAX, rk4 as _rk

ACC_BASE: Tuple[str, ...] = ("x", "v", "u")
ACC_OPTIONAL: Tuple[str, ...] = ("x3", "v3")
TERM_TO_THETA = {"x": "ax", "v": "av", "u": "au", "x3": "ax3", "v3": "av3", "m": "am"}
DECAYS = (0.25, 0.35, 0.5, 0.7)


def _powerset(items: Sequence[str]) -> List[Tuple[str, ...]]:
    s = list(items)
    return [tuple(c) for c in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))]


@dataclass
class Artifact:
    acc_terms: Tuple[str, ...]
    hidden: Tuple[str, ...]
    obs_map: str  # "identity" | "linear_time"
    theta: Dict[str, float] = field(default_factory=dict)
    hash: str = ""

    @property
    def use_memory(self) -> bool:
        return "m" in self.hidden

    @property
    def use_drift(self) -> bool:
        return self.obs_map == "linear_time"

    @property
    def family(self) -> str:
        """Diagnostic label only. Not a fitter branch."""
        parts = ["+".join(self.acc_terms) or "empty"]
        if self.hidden:
            parts.append("hid:" + "+".join(self.hidden))
        parts.append("obs:" + self.obs_map)
        return "|".join(parts)

    def to_dict(self) -> Dict:
        return {
            "acc_terms": list(self.acc_terms),
            "hidden": list(self.hidden),
            "obs_map": self.obs_map,
            "theta": {k: float(v) for k, v in self.theta.items()},
            "hash": self.hash,
            "family": self.family,
            "use_memory": self.use_memory,
            "use_drift": self.use_drift,
        }

    def spec_key(self) -> tuple:
        return (tuple(self.acc_terms), tuple(self.hidden), self.obs_map)


def legal_structures() -> List[Dict]:
    """All combinations of the public primitive inventory."""
    out = []
    for extra in _powerset(ACC_OPTIONAL):
        for hid in ((), ("m",)):
            acc = ACC_BASE + extra + (("m",) if hid else ())
            for obs in ("identity", "linear_time"):
                out.append({"acc_terms": acc, "hidden": hid, "obs_map": obs})
    return out


def default_linear() -> Artifact:
    return Artifact(
        acc_terms=ACC_BASE,
        hidden=(),
        obs_map="identity",
        theta={"ax": -4.0, "av": -0.4, "au": 1.0, "b0": 0.0, "b1": 0.0, "decay": 0.0},
        hash="init_linear",
    )


def finite_acc(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(v)
    if len(t) < 3:
        return acc
    acc[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])
    acc[0] = (v[1] - v[0]) / max(t[1] - t[0], 1e-9)
    acc[-1] = (v[-1] - v[-2]) / max(t[-1] - t[-2], 1e-9)
    return acc


def reconstruct_memory(t: np.ndarray, u: np.ndarray, decay: float, m0: float = 0.0) -> np.ndarray:
    m = np.zeros_like(t, dtype=float)
    m[0] = float(m0)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        m[i] = m[i - 1] + dt * (-decay * m[i - 1] + u[i - 1])
    return m


def _interp_u(t: np.ndarray, u: np.ndarray, tt: float) -> float:
    if tt <= t[0]:
        return float(u[0])
    if tt >= t[-1]:
        return float(u[-1])
    return float(np.interp(tt, t, u))


def _col(name: str, x, v, u, m):
    if name == "x":
        return x
    if name == "v":
        return v
    if name == "u":
        return u
    if name == "x3":
        return x**3
    if name == "v3":
        return v**3
    if name == "m":
        return m
    raise KeyError(name)


def simulate_artifact(
    art: Artifact,
    t_grid: np.ndarray,
    u_of_t: Callable[[float], float],
    x0: float,
    v0: float,
    m0: float = 0.0,
) -> np.ndarray:
    th = art.theta
    decay = float(th.get("decay", 0.6)) if art.use_memory else 0.0
    b0 = float(th.get("b0", 0.0))
    b1 = float(th.get("b1", 0.0)) if art.use_drift else 0.0
    coefs = {name: float(th.get(TERM_TO_THETA[name], 0.0)) for name in art.acc_terms}

    def deriv(_t, z, u):
        x, v, m = np.clip(z, -50.0, 50.0)
        acc = 0.0
        for name, c in coefs.items():
            if name == "x":
                acc += c * x
            elif name == "v":
                acc += c * v
            elif name == "u":
                acc += c * u
            elif name == "x3":
                acc += c * (x**3)
            elif name == "v3":
                acc += c * (v**3)
            elif name == "m":
                acc += c * m
        if not np.isfinite(acc):
            acc = 0.0
        acc = float(np.clip(acc, -200.0, 200.0))
        dm = (-decay * m + u) if art.use_memory else 0.0
        return np.array([v, acc, dm], dtype=float)

    t_grid = np.asarray(t_grid, dtype=float)
    z = np.array([x0, v0, m0], dtype=float)
    y = np.zeros((len(t_grid), 2), dtype=float)
    t0 = float(t_grid[0])
    y[0] = np.array([z[0] + b0, z[1]])
    for i in range(1, len(t_grid)):
        ta = float(t_grid[i - 1])
        tb = float(t_grid[i])
        nsub = max(1, int(np.ceil((tb - ta) / DT_MAX)))
        dt = (tb - ta) / nsub
        t = ta
        for _ in range(nsub):
            u = float(u_of_t(t))
            z = _rk(lambda tt, zz: deriv(tt, zz, u), t, z, dt)
            z = np.clip(z, -50.0, 50.0)
            t += dt
        tt = float(t_grid[i])
        y[i, 0] = z[0] + b0 + b1 * (tt - t0)
        y[i, 1] = z[1]
    if not np.all(np.isfinite(y)):
        y = np.full_like(y, 1e6)
    return y


def fit_artifact(spec: Dict, traces: List[Dict], decay: float = 0.4) -> Artifact:
    acc_terms = tuple(spec["acc_terms"])
    hidden = tuple(spec.get("hidden") or ())
    obs_map = str(spec.get("obs_map") or "identity")
    use_m = "m" in hidden
    xs, vs, us, accs, mems, ts, yxs = [], [], [], [], [], [], []
    for tr in traces:
        t = np.asarray(tr["t"], dtype=float)
        y = np.asarray(tr["y"], dtype=float)
        u = np.asarray(tr["u"], dtype=float)
        x = y[:, 0]
        v = y[:, 1]
        acc = finite_acc(t, v)
        mem = reconstruct_memory(t, u, decay, m0=0.0) if use_m else np.zeros_like(t)
        xs.append(x)
        vs.append(v)
        us.append(u)
        accs.append(acc)
        mems.append(mem)
        ts.append(t)
        yxs.append(x)
    x = np.concatenate(xs)
    v = np.concatenate(vs)
    u = np.concatenate(us)
    acc = np.concatenate(accs)
    m = np.concatenate(mems)
    cols = [_col(name, x, v, u, m) for name in acc_terms]
    a = np.column_stack(cols) if cols else np.ones((len(x), 1))
    coef, *_ = lstsq(a, acc, rcond=None)
    theta = {TERM_TO_THETA[n]: float(c) for n, c in zip(acc_terms, coef)}
    for n, k in TERM_TO_THETA.items():
        theta.setdefault(k, 0.0)
    theta["decay"] = float(decay) if use_m else 0.0
    theta["b0"] = 0.0
    theta["b1"] = 0.0
    if obs_map == "linear_time":
        t_all = np.concatenate(ts)
        yx = np.concatenate(yxs)
        b = np.column_stack([np.ones_like(t_all), t_all - t_all.min()])
        bcoef, *_ = lstsq(b, yx, rcond=None)
        theta["b0"] = float(bcoef[0])
        theta["b1"] = float(bcoef[1])
    return Artifact(acc_terms=acc_terms, hidden=hidden, obs_map=obs_map, theta=theta)


def train_rmse(art: Artifact, traces: List[Dict], m0: float = 0.0) -> float:
    errs = []
    for tr in traces:
        t = np.asarray(tr["t"], dtype=float)
        y = np.asarray(tr["y"], dtype=float)
        u = np.asarray(tr["u"], dtype=float)
        pred = simulate_artifact(
            art, t, lambda tt, u=u, t=t: _interp_u(t, u, tt), float(y[0, 0]), float(y[0, 1]), m0=m0
        )
        mse = float(np.mean((pred - y) ** 2))
        if not np.isfinite(mse):
            mse = 1e12
        errs.append(mse)
    return float(np.sqrt(np.mean(errs))) if errs else float("inf")


def estimate_hidden_ic(art: Artifact, history: Optional[Dict]) -> float:
    """Estimate m at the end of a public pre-drive. Never reads true m0.

    Primary: integrate dm=-decay*m+u from m(0)=0 using the drive.
    Correction: if am is identifiable, blend with acc residual / am.
    """
    if not art.use_memory or history is None:
        return 0.0
    t = np.asarray(history["t"], dtype=float)
    y = np.asarray(history["y"], dtype=float)
    u = np.asarray(history["u"], dtype=float)
    decay = float(art.theta.get("decay", 0.4))
    m_u = reconstruct_memory(t, u, decay, m0=0.0)
    m_end = float(m_u[-1])
    am = float(art.theta.get("am", 0.0))
    if abs(am) > 1e-4 and len(t) >= 4:
        acc = finite_acc(t, y[:, 1])
        x, v = y[:, 0], y[:, 1]
        acc_wo = np.zeros_like(acc)
        for name in art.acc_terms:
            if name == "m":
                continue
            c = float(art.theta.get(TERM_TO_THETA[name], 0.0))
            acc_wo = acc_wo + c * _col(name, x, v, u, m_u)
        m_res = (acc - acc_wo) / am
        m_res = np.clip(m_res, -50.0, 50.0)
        tail = m_res[-min(8, len(m_res)) :]
        if np.all(np.isfinite(tail)):
            m_end = 0.5 * m_end + 0.5 * float(np.median(tail))
    if not np.isfinite(m_end):
        return 0.0
    return float(np.clip(m_end, -50.0, 50.0))


def search_library(traces: List[Dict]) -> List[Artifact]:
    """Fixed wide library: every legal primitive combination."""
    scored: List[Artifact] = []
    for spec in legal_structures():
        decays = DECAYS if spec["hidden"] else (0.4,)
        best = None
        best_err = None
        for d in decays:
            art = fit_artifact(spec, traces, decay=d)
            err = train_rmse(art, traces)
            if best_err is None or err < best_err:
                best, best_err = art, err
        scored.append(best)
    scored.sort(key=lambda a: train_rmse(a, traces))
    return scored


def propose_from_residuals(traces: List[Dict], min_drop: float = 0.10) -> Artifact:
    """Greedy composition over legal_structures. Starts at (x,v,u)+identity;
    adds one primitive combination at a time if train RMSE drops by min_drop.
    Does not switch on family names. New legal specs are already in
    legal_structures(); the fitter is unchanged.
    """
    spec = {"acc_terms": ACC_BASE, "hidden": (), "obs_map": "identity"}
    cur = fit_artifact(spec, traces)
    cur_err = train_rmse(cur, traces)
    pool = legal_structures()
    seen = {tuple(spec["acc_terms"]) + spec["hidden"] + (spec["obs_map"],)}
    improved = True
    while improved:
        improved = False
        best_spec = None
        best_art = None
        best_e = cur_err
        for cand_spec in pool:
            key = tuple(cand_spec["acc_terms"]) + cand_spec["hidden"] + (cand_spec["obs_map"],)
            if key in seen:
                continue
            decays = DECAYS if cand_spec["hidden"] else (0.4,)
            local = None
            local_e = None
            for d in decays:
                art = fit_artifact(cand_spec, traces, decay=d)
                err = train_rmse(art, traces)
                if local_e is None or err < local_e:
                    local, local_e = art, err
            if local_e is not None and local_e < best_e * (1.0 - min_drop):
                best_e = local_e
                best_spec = cand_spec
                best_art = local
        if best_spec is not None:
            seen.add(tuple(best_spec["acc_terms"]) + best_spec["hidden"] + (best_spec["obs_map"],))
            spec = best_spec
            cur = best_art
            cur_err = best_e
            improved = True
        else:
            break
    return cur


# --- compatibility wrappers so old family strings still fit ---
FAMILY_SPEC = {
    "linear2": {"acc_terms": ACC_BASE, "hidden": (), "obs_map": "identity"},
    "nl2": {"acc_terms": ACC_BASE + ("x3", "v3"), "hidden": (), "obs_map": "identity"},
    "memory3": {"acc_terms": ACC_BASE + ("x3", "v3", "m"), "hidden": ("m",), "obs_map": "identity"},
    "obs_drift": {"acc_terms": ACC_BASE, "hidden": (), "obs_map": "linear_time"},
}


def fit_family(family: str, traces: List[Dict], decay: float = 0.4) -> Artifact:
    spec = FAMILY_SPEC[family]
    return fit_artifact(spec, traces, decay=decay)


def search_families(traces: List[Dict]) -> List[Artifact]:
    return search_library(traces)


# aliases used by session / score
simulate_candidate = simulate_artifact
Candidate = Artifact
