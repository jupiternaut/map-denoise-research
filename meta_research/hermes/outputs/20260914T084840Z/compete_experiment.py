#!/usr/bin/env python3
"""Competing explanations: isolate-capable fitter, then query selection.

Experiment A: freeze a history, compare old vs new fitter.
Experiment B: share the new fitter; policies differ only in the next
query/retest. Isolate is internal, reversible, and does not delete records.

Dirty location and breakpoint are sampled; {0,16,32} as the init set is
rejected. 48 is not a policy constant.

ASSUMED: libraries, costs, isolate cap 1, uniform over remaining repairs
for one-step lookahead.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ORIG = Path("/home/grf/.hermes/attachments/outputs")
sys.path.insert(0, str(ORIG))
import mismatch_experiment as m  # noqa: E402

N = m.N
DOMAIN = m.DOMAIN
H0, H1 = m.H0, m.H1
COVER = m.COVER
BUDGETS = (8, 16)
MAX_BUDGET = 16
DEV_BUNDLES = 10
EVAL_BUNDLES = 60
DEV_SEED0 = 1
EVAL_SEED0 = 20_000
CAUSES = ("model", "transient", "persistent")
COST_QUERY = 1
COST_RETEST = 1
POLICIES_B = ("cover", "retest_then_cover", "compete")
FORBIDDEN_INIT = {0, 16, 32}


def mask_obs(table: np.ndarray, believed: Dict[int, int]) -> np.ndarray:
    live = np.ones(len(table), dtype=bool)
    for x, y in believed.items():
        live &= table[:, int(x)] == int(y)
    return live


def trusted_of(believed: Dict[int, int], isol: FrozenSet[int]) -> Dict[int, int]:
    return {int(k): int(v) for k, v in believed.items() if int(k) not in isol}


def nn_fill(believed: Dict[int, int]) -> np.ndarray:
    f = np.empty(N, dtype=np.int32)
    if not believed:
        f[:] = 0
        return f
    xs = np.array(sorted(believed), dtype=np.int32)
    ys = np.array([believed[int(x)] for x in xs], dtype=np.int32)
    for x in DOMAIN:
        f[x] = ys[int(np.argmin(np.abs(xs - x)))]
    return f


def restoring_singles(believed: Dict[int, int], table: np.ndarray) -> List[int]:
    if mask_obs(table, believed).any():
        return []
    hits = []
    for x in believed:
        sub = {k: v for k, v in believed.items() if k != x}
        if mask_obs(table, sub).any():
            hits.append(int(x))
    return sorted(hits)


@dataclass(frozen=True)
class Repair:
    family: str
    isol: FrozenSet[int]
    pred: Tuple[int, ...]
    n_live: int

    def pred_arr(self) -> np.ndarray:
        return np.array(self.pred, dtype=np.int32)


def enumerate_repairs(believed: Dict[int, int], n_score_box: Optional[List[int]] = None) -> List[Repair]:
    """Consistent (family, isolate) pairs. Isolate cap 1; |trusted|>=2.

    Ranking (ASSUMED): fewer isolates, then H0 over H1, then smaller index.
    """
    if not believed:
        return []
    opts: List[FrozenSet[int]] = [frozenset()]
    for x in sorted(believed):
        opts.append(frozenset([int(x)]))
    found: List[Repair] = []
    for isol in opts:
        trusted = trusted_of(believed, isol)
        if len(trusted) < 2:
            continue
        for fam, table in (("H0", H0), ("H1", H1)):
            live = mask_obs(table, trusted)
            if n_score_box is not None:
                n_score_box[0] += int(len(table) * max(1, len(trusted)))
            if not live.any():
                continue
            pred = table[live][0]
            found.append(
                Repair(fam, isol, tuple(int(v) for v in pred.tolist()), int(live.sum()))
            )
    found.sort(
        key=lambda r: (len(r.isol), 0 if r.family == "H0" else 1, min(r.isol) if r.isol else -1)
    )
    return found


def old_deliver(believed: Dict[int, int]) -> Tuple[np.ndarray, Dict]:
    live0 = mask_obs(H0, believed)
    if live0.any():
        return H0[live0][0].copy(), {"family": "H0", "isol": [], "consistent": True, "n_live": int(live0.sum())}
    live1 = mask_obs(H1, believed)
    if live1.any():
        return H1[live1][0].copy(), {"family": "H1", "isol": [], "consistent": True, "n_live": int(live1.sum())}
    return nn_fill(believed), {"family": "nn", "isol": [], "consistent": False, "n_live": 0}


def new_deliver(believed: Dict[int, int], n_score_box: Optional[List[int]] = None) -> Tuple[np.ndarray, Dict]:
    reps = enumerate_repairs(believed, n_score_box)
    if not reps:
        return nn_fill(believed), {"family": "nn", "isol": [], "consistent": False, "n_live": 0, "n_repairs": 0}
    r = reps[0]
    return r.pred_arr().copy(), {
        "family": r.family,
        "isol": sorted(r.isol),
        "consistent": True,
        "n_live": r.n_live,
        "n_repairs": len(reps),
    }


def next_cover(believed: Dict[int, int]) -> Optional[int]:
    for x in COVER:
        if int(x) not in believed:
            return int(x)
    return None


@dataclass
class Task:
    task_id: str
    typ: int
    cause: str
    truth: np.ndarray
    init_obs: Dict[int, int]
    dirty_x: int
    dirty_y: int
    t_break: int
    pair_id: str
    split: str


class Env:
    def __init__(self, task: Task):
        self._task = task
        self._truth = np.array(task.truth, dtype=np.int32)
        self._cause = task.cause
        self._dirty_x = int(task.dirty_x)
        self._dirty_y = int(task.dirty_y)
        self.n_observe = 0
        self.n_retest = 0

    def observe(self, x: int) -> int:
        self.n_observe += 1
        x = int(x)
        if self._cause == "model":
            return int(self._truth[x])
        if x == self._dirty_x:
            return int(self._dirty_y)
        return int(self._truth[x])

    def retest(self, x: int) -> int:
        self.n_retest += 1
        x = int(x)
        if self._cause == "transient":
            return int(self._truth[x])
        if self._cause == "persistent" and x == self._dirty_x:
            return int(self._dirty_y)
        return int(self._truth[x])

    def clone_unqueried_rewritten(self, believed: Dict[int, int]) -> "Env":
        t = self._truth.copy()
        for x in range(N):
            if int(x) not in believed:
                t[int(x)] = int(17 * int(x) + 3)
        env = Env(self._task)
        env._truth = t
        return env


def build_shared_instance(rng: np.random.Generator):
    """Paired causes share init_obs. dirty_x and t are sampled.

    Rejects the previous covering shortcut init {0,16,32}.
    ASSUMED construction.
    """
    t_choices = m.T_RANGE
    for _ in range(30_000):
        f0 = m.H0[int(rng.integers(0, len(m.H0)))].copy()
        a = int(f0[1] - f0[0])
        b = int(f0[0])
        t = int(rng.choice(t_choices))
        a2 = int(rng.choice(m.A_RANGE))
        b2 = int(rng.choice(m.B_RANGE))
        if a2 == a and b2 == b:
            continue
        left = DOMAIN < t
        right = ~left
        if int(left.sum()) < 3 or int(right.sum()) < 3:
            continue
        f1 = np.empty(N, dtype=np.int32)
        f1[left] = a * DOMAIN[left] + b
        f1[right] = a2 * DOMAIN[right] + b2
        if m.in_library(H0, f1) or not m.in_library(H1, f1):
            continue
        right_xs = DOMAIN[right]
        dirty_x = int(rng.choice(right_xs))
        dirty_y = int(f1[dirty_x])
        if dirty_y == int(f0[dirty_x]):
            continue
        left_xs = DOMAIN[left]
        c1, c2 = (int(v) for v in rng.choice(left_xs, size=2, replace=False))
        init_set = {c1, c2, dirty_x}
        if init_set == FORBIDDEN_INIT:
            continue
        init_obs = {c1: int(f0[c1]), c2: int(f0[c2]), dirty_x: int(dirty_y)}
        if int(f1[c1]) != init_obs[c1] or int(f1[c2]) != init_obs[c2]:
            continue
        if int(f1[dirty_x]) != dirty_y:
            continue
        if mask_obs(H0, init_obs).any() or not mask_obs(H1, init_obs).any():
            continue
        return f0, f1, dict(init_obs), dirty_x, dirty_y, t
    raise RuntimeError("failed to sample shared instance")


def make_bundle(split: str, serial: int, seed0: int) -> List[Task]:
    rng = np.random.default_rng(seed0 + serial)
    f0, f1, init_obs, dirty_x, dirty_y, t = build_shared_instance(rng)
    pair_id = f"{split}-b{serial:03d}"
    out = []
    for typ, cause in enumerate(CAUSES):
        truth = f1 if cause == "model" else f0
        out.append(
            Task(
                task_id=f"{pair_id}-{cause}",
                typ=typ,
                cause=cause,
                truth=truth,
                init_obs=dict(init_obs),
                dirty_x=dirty_x,
                dirty_y=dirty_y,
                t_break=t,
                pair_id=pair_id,
                split=split,
            )
        )
    return out


def mae_of(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred.astype(np.int64) - truth.astype(np.int64)).sum()) / float(N)


def snapshot_fitters(believed: Dict[int, int], truth: np.ndarray, cost: int, n_score: int) -> Dict:
    old_pred, old_meta = old_deliver(believed)
    box = [0]
    new_pred, new_meta = new_deliver(believed, box)
    n_score += box[0]
    return {
        "budget": int(cost),
        "n_score": n_score,
        "old": {
            **old_meta,
            "mae": mae_of(old_pred, truth),
            "l1": int(np.abs(old_pred.astype(np.int64) - truth.astype(np.int64)).sum()),
        },
        "new": {
            **new_meta,
            "mae": mae_of(new_pred, truth),
            "l1": int(np.abs(new_pred.astype(np.int64) - truth.astype(np.int64)).sum()),
        },
        "n_believed": len(believed),
        "n_repairs_new": int(new_meta.get("n_repairs", 0)),
    }


def collect_frozen(task: Task, kind: str, max_budget: int = MAX_BUDGET) -> Dict:
    """Fixed history. Fitters are applied to the same believed labels."""
    t0 = time.perf_counter()
    env = Env(task)
    believed: Dict[int, int] = {}
    retested: set = set()
    cost = 0
    n_score = 0
    traj = []
    checkpoints = {}

    def note():
        for b in BUDGETS:
            if cost >= b and str(b) not in checkpoints:
                snap = snapshot_fitters(believed, task.truth, cost, n_score)
                snap["n_retest"] = len(retested)
                checkpoints[str(b)] = snap

    for x in sorted(task.init_obs):
        y = env.observe(int(x))
        believed[int(x)] = int(y)
        cost += COST_QUERY
        traj.append({"a": "query", "x": int(x), "y": int(y), "cost": cost})
    note()

    def do_retests():
        nonlocal cost, n_score
        singles = restoring_singles(believed, H0)
        n_score += len(H0) * max(1, len(believed))
        for x in singles:
            if cost >= max_budget:
                return
            if int(x) in retested:
                continue
            y2 = env.retest(int(x))
            believed[int(x)] = int(y2)
            retested.add(int(x))
            cost += COST_RETEST
            traj.append({"a": "retest", "x": int(x), "y": int(y2), "cost": cost})
            note()

    if kind == "QR":
        do_retests()

    while cost < max_budget:
        x = next_cover(believed)
        if x is None:
            break
        y = env.observe(int(x))
        believed[int(x)] = int(y)
        cost += COST_QUERY
        traj.append({"a": "query", "x": int(x), "y": int(y), "cost": cost})
        note()

    for b in BUDGETS:
        if str(b) not in checkpoints:
            snap = snapshot_fitters(believed, task.truth, cost, n_score)
            snap["n_retest"] = len(retested)
            checkpoints[str(b)] = snap

    return {
        "kind": kind,
        "task": task.task_id,
        "cause": task.cause,
        "pair_id": task.pair_id,
        "dirty_x": task.dirty_x,
        "t_break": task.t_break,
        "init_xs": sorted(task.init_obs),
        "cost": cost,
        "elapsed_s": time.perf_counter() - t0,
        "checkpoints": checkpoints,
        "traj": traj,
        "truth_hash": hashlib.sha256(task.truth.tobytes()).hexdigest()[:16],
    }


def disagreement_xs(repairs: List[Repair], believed: Dict[int, int]) -> List[int]:
    u = [int(x) for x in DOMAIN if int(x) not in believed]
    if len(repairs) < 2 or not u:
        return []
    mats = np.stack([r.pred_arr() for r in repairs], axis=0)
    out = []
    for x in u:
        if int(np.unique(mats[:, x]).size) > 1:
            out.append(int(x))
    return out


def expected_loss(repairs: List[Repair], pick: Repair) -> float:
    if not repairs:
        return 0.0
    p = pick.pred_arr().astype(np.int64)
    acc = 0.0
    for r in repairs:
        acc += float(np.abs(p - r.pred_arr().astype(np.int64)).sum()) / float(N)
    return acc / float(len(repairs))


def simulate_believed(believed: Dict[int, int], x: int, y: int) -> Dict[int, int]:
    out = dict(believed)
    out[int(x)] = int(y)
    return out


def compete_action(
    believed: Dict[int, int],
    retested: set,
    remaining: int,
    costs: Dict[str, int],
    n_score_box: List[int],
) -> Tuple[Optional[str], Optional[int]]:
    """1-step lookahead over remaining repairs. Does not read hidden labels."""
    reps = enumerate_repairs(believed, n_score_box)
    if remaining < min(costs["query"], costs["retest"]):
        return None, None
    if not reps:
        x = next_cover(believed)
        return ("query", x) if x is not None and remaining >= costs["query"] else (None, None)

    pick = reps[0]
    base = expected_loss(reps, pick)
    best = None

    split = disagreement_xs(reps, believed)
    n_score_box[0] += len(reps) * N
    for x in split:
        if remaining < costs["query"]:
            break
        labels = sorted({int(r.pred[x]) for r in reps})
        exp_loss = 0.0
        for y in labels:
            w = sum(1 for r in reps if int(r.pred[x]) == y) / float(len(reps))
            nb = simulate_believed(believed, x, y)
            nr = enumerate_repairs(nb, n_score_box)
            if not nr:
                exp_loss += w * base
                continue
            exp_loss += w * expected_loss(nr, nr[0])
        gain = base - exp_loss
        score = gain / float(costs["query"])
        cand = (score, 0, -int(x), "query", int(x))
        if best is None or cand[:3] > best[:3]:
            best = cand

    for z in sorted(believed):
        if int(z) in retested or remaining < costs["retest"]:
            continue
        labels = sorted({int(r.pred[z]) for r in reps} | {int(believed[z])})
        if len(labels) <= 1:
            continue
        exp_loss = 0.0
        nlab = float(len(labels))
        for y in labels:
            w = sum(1 for r in reps if int(r.pred[z]) == y) / float(len(reps))
            if w == 0:
                w = 1.0 / nlab
            nb = simulate_believed(believed, z, y)
            nr = enumerate_repairs(nb, n_score_box)
            if not nr:
                exp_loss += w * base
                continue
            exp_loss += w * expected_loss(nr, nr[0])
        gain = base - exp_loss
        score = gain / float(costs["retest"])
        cand = (score, -1, -int(z), "retest", int(z))
        if best is None or cand[:3] > best[:3]:
            best = cand

    if best is not None and best[0] > 1e-12:
        return best[3], best[4]
    x = next_cover(believed)
    if x is not None and remaining >= costs["query"]:
        return "query", x
    return None, None


def choose_B(
    name: str,
    believed: Dict[int, int],
    retested: set,
    remaining: int,
    costs: Dict[str, int],
    n_score_box: List[int],
) -> Tuple[Optional[str], Optional[int]]:
    if name == "cover":
        x = next_cover(believed)
        return ("query", x) if x is not None and remaining >= costs["query"] else (None, None)
    if name == "retest_then_cover":
        singles = restoring_singles(believed, H0)
        n_score_box[0] += len(H0) * max(1, len(believed))
        pending = [x for x in singles if int(x) not in retested]
        if pending and remaining >= costs["retest"]:
            return "retest", int(pending[0])
        x = next_cover(believed)
        return ("query", x) if x is not None and remaining >= costs["query"] else (None, None)
    if name == "compete":
        return compete_action(believed, retested, remaining, costs, n_score_box)
    raise ValueError(name)


def run_policy_B(task: Task, name: str, max_budget: int = MAX_BUDGET, costs=None) -> Dict:
    costs = costs or {"query": COST_QUERY, "retest": COST_RETEST}
    t0 = time.perf_counter()
    env = Env(task)
    believed: Dict[int, int] = {}
    retested: set = set()
    cost = 0
    n_query = n_retest = 0
    n_score_box = [0]
    n_invalid_retest = 0
    n_wrong_isolate = 0
    traj = []
    checkpoints = {}

    def note():
        for b in BUDGETS:
            if cost >= b and str(b) not in checkpoints:
                pred, meta = new_deliver(believed, n_score_box)
                isol = set(meta.get("isol") or [])
                wrong = bool(isol) and task.cause == "model"
                checkpoints[str(b)] = {
                    "budget": int(cost),
                    "mae": mae_of(pred, task.truth),
                    "l1": int(np.abs(pred.astype(np.int64) - task.truth.astype(np.int64)).sum()),
                    "family": meta.get("family"),
                    "isol": meta.get("isol"),
                    "consistent": meta.get("consistent"),
                    "n_repairs": meta.get("n_repairs", 0),
                    "n_query": n_query,
                    "n_retest": n_retest,
                    "n_score": n_score_box[0],
                    "n_invalid_retest": n_invalid_retest,
                    "wrong_isolate": wrong,
                }

    for x in sorted(task.init_obs):
        y = env.observe(int(x))
        believed[int(x)] = int(y)
        cost += costs["query"]
        n_query += 1
        traj.append({"a": "query", "x": int(x), "y": int(y), "cost": cost})
    note()

    while cost < max_budget:
        rem = max_budget - cost
        act, x = choose_B(name, believed, retested, rem, costs, n_score_box)
        if act is None or (act == "query" and x is None):
            break
        if act == "query":
            if int(x) in believed:
                break
            y = env.observe(int(x))
            believed[int(x)] = int(y)
            cost += costs["query"]
            n_query += 1
            traj.append({"a": "query", "x": int(x), "y": int(y), "cost": cost})
        elif act == "retest":
            y2 = env.retest(int(x))
            if int(y2) == int(believed[int(x)]):
                n_invalid_retest += 1
            believed[int(x)] = int(y2)
            retested.add(int(x))
            cost += costs["retest"]
            n_retest += 1
            traj.append({"a": "retest", "x": int(x), "y": int(y2), "cost": cost})
        else:
            raise RuntimeError(act)
        _, meta = new_deliver(believed, n_score_box)
        if meta.get("isol") and task.cause == "model":
            n_wrong_isolate += 1
        note()

    for b in BUDGETS:
        if str(b) not in checkpoints:
            pred, meta = new_deliver(believed, n_score_box)
            checkpoints[str(b)] = {
                "budget": int(cost),
                "mae": mae_of(pred, task.truth),
                "l1": int(np.abs(pred.astype(np.int64) - task.truth.astype(np.int64)).sum()),
                "family": meta.get("family"),
                "isol": meta.get("isol"),
                "consistent": meta.get("consistent"),
                "n_repairs": meta.get("n_repairs", 0),
                "n_query": n_query,
                "n_retest": n_retest,
                "n_score": n_score_box[0],
                "n_invalid_retest": n_invalid_retest,
                "wrong_isolate": bool(meta.get("isol")) and task.cause == "model",
            }

    _, meta = new_deliver(believed, n_score_box)
    return {
        "policy": name,
        "task": task.task_id,
        "cause": task.cause,
        "typ": task.typ,
        "pair_id": task.pair_id,
        "dirty_x": task.dirty_x,
        "t_break": task.t_break,
        "init_xs": sorted(task.init_obs),
        "cost": cost,
        "elapsed_s": time.perf_counter() - t0,
        "n_score": n_score_box[0],
        "wrong_isolate_steps": n_wrong_isolate,
        "checkpoints": checkpoints,
        "final": checkpoints[str(MAX_BUDGET)],
        "traj": traj,
        "truth_hash": hashlib.sha256(task.truth.tobytes()).hexdigest()[:16],
        "meta_final": meta,
    }


def cheat_next(env: Env, believed, remaining, costs):
    u = [int(x) for x in range(N) if int(x) not in believed]
    if u and remaining >= costs["query"]:
        hidden = np.array([int(env._truth[x]) for x in u], dtype=np.int64)
        score = hidden * 31 + np.array(u, dtype=np.int64)
        return "query", int(u[int(np.argmax(score))])
    return None, None


def dual_env_isolation() -> Dict:
    costs = {"query": COST_QUERY, "retest": COST_RETEST}
    trials = []
    for policy in list(POLICIES_B) + ["cheat"]:
        for serial in range(3):
            bundle = make_bundle("iso", serial, seed0=70_000)
            for task in bundle:
                env_a = Env(task)
                believed: Dict[int, int] = {}
                retested: set = set()
                cost = 0
                box = [0]
                for x in sorted(task.init_obs):
                    believed[int(x)] = env_a.observe(int(x))
                    cost += costs["query"]
                steps = 0
                while cost < 6 and steps < 2:
                    rem = 8 - cost
                    if policy == "cheat":
                        act, x = cheat_next(env_a, believed, rem, costs)
                    else:
                        act, x = choose_B(policy, believed, retested, rem, costs, box)
                    if act is None:
                        break
                    if act == "query" and x is not None and int(x) not in believed:
                        believed[int(x)] = env_a.observe(int(x))
                        cost += costs["query"]
                    elif act == "retest":
                        believed[int(x)] = env_a.retest(int(x))
                        retested.add(int(x))
                        cost += costs["retest"]
                    steps += 1
                env_b = env_a.clone_unqueried_rewritten(believed)
                rem = 16 - cost
                box2 = [0]
                if policy == "cheat":
                    a = cheat_next(env_a, dict(believed), rem, costs)
                    b = cheat_next(env_b, dict(believed), rem, costs)
                else:
                    a = choose_B(policy, dict(believed), set(retested), rem, costs, box2)
                    b = choose_B(policy, dict(believed), set(retested), rem, costs, box2)
                trials.append(
                    {
                        "policy": policy,
                        "cause": task.cause,
                        "next_a": a,
                        "next_b": b,
                        "ok": a == b,
                        "envs_distinct": not np.array_equal(env_a._truth, env_b._truth),
                    }
                )
    by: Dict[str, list] = {}
    for t in trials:
        by.setdefault(t["policy"], []).append(t)
    summary = {
        p: {"n": len(ts), "n_fail": sum(1 for t in ts if not t["ok"]), "all_ok": all(t["ok"] for t in ts)}
        for p, ts in by.items()
    }
    return {
        "honest_must_pass": all(summary[p]["all_ok"] for p in POLICIES_B),
        "cheat_must_fail": not summary["cheat"]["all_ok"],
        "by_policy": summary,
    }


def mean_ci(xs: Sequence[float]) -> Dict:
    arr = np.array(list(xs), dtype=np.float64)
    n = arr.size
    m = float(arr.mean()) if n else float("nan")
    if n < 2:
        return {"n": int(n), "mean": m, "sd": 0.0, "se": 0.0}
    sd = float(arr.std(ddof=1))
    return {"n": int(n), "mean": m, "sd": sd, "se": sd / float(np.sqrt(n))}


def paired_delta(a: Sequence[float], b: Sequence[float]) -> Dict:
    da = np.array(list(a), dtype=np.float64) - np.array(list(b), dtype=np.float64)
    stats = mean_ci(da)
    stats["n_pos"] = int((da > 0).sum())
    stats["n_neg"] = int((da < 0).sum())
    stats["n_tie"] = int((da == 0).sum())
    return stats


def summarize_A(rows: List[Dict]) -> Dict:
    by = {}
    for cause in CAUSES:
        sub = [r for r in rows if r["cause"] == cause]
        hashes = {r["truth_hash"] for r in sub}
        block = {"n": len(sub), "n_distinct_truths": len(hashes), "kinds": {}}
        for kind in ("Q", "QR"):
            ksub = [r for r in sub if r["kind"] == kind]
            kb = {}
            for b in BUDGETS:
                old = [r["checkpoints"][str(b)]["old"]["mae"] for r in ksub]
                new = [r["checkpoints"][str(b)]["new"]["mae"] for r in ksub]
                isol = [float(bool(r["checkpoints"][str(b)]["new"]["isol"])) for r in ksub]
                fam = [r["checkpoints"][str(b)]["new"]["family"] for r in ksub]
                kb[str(b)] = {
                    "old_mae": mean_ci(old),
                    "new_mae": mean_ci(new),
                    "paired_old_minus_new": paired_delta(old, new),
                    "new_isolate_rate": mean_ci(isol),
                    "new_H0_rate": float(np.mean([f == "H0" for f in fam])),
                    "new_H1_rate": float(np.mean([f == "H1" for f in fam])),
                    "new_nn_rate": float(np.mean([f == "nn" for f in fam])),
                    "old_nn_rate": float(
                        np.mean([r["checkpoints"][str(b)]["old"]["family"] == "nn" for r in ksub])
                    ),
                }
            block["kinds"][kind] = kb
        by[cause] = block
    inits = [tuple(r["init_xs"]) for r in rows]
    dirtys = [r["dirty_x"] for r in rows]
    ts = [r["t_break"] for r in rows]
    return {
        "by_cause": by,
        "n_rows": len(rows),
        "frac_init_shortcut": float(np.mean([set(xs) == FORBIDDEN_INIT for xs in inits])),
        "n_unique_dirty_x": len(set(dirtys)),
        "n_unique_t": len(set(ts)),
        "rows_compact": [
            {
                "task": r["task"],
                "cause": r["cause"],
                "kind": r["kind"],
                "dirty_x": r["dirty_x"],
                "t": r["t_break"],
                "init_xs": r["init_xs"],
                **{
                    f"{fit}_mae{b}": r["checkpoints"][str(b)][fit]["mae"]
                    for b in BUDGETS
                    for fit in ("old", "new")
                },
                **{f"new_isol{b}": r["checkpoints"][str(b)]["new"]["isol"] for b in BUDGETS},
                **{f"new_fam{b}": r["checkpoints"][str(b)]["new"]["family"] for b in BUDGETS},
            }
            for r in rows
        ],
    }


def summarize_B(rows: List[Dict]) -> Dict:
    by = {}
    for cause in CAUSES:
        sub = [r for r in rows if r["cause"] == cause]
        hashes = {r["recs"][POLICIES_B[0]]["truth_hash"] for r in sub}
        block = {"n": len(sub), "n_distinct_truths": len(hashes), "budgets": {}}
        for b in BUDGETS:
            bblock = {}
            mae = {p: [r["recs"][p]["checkpoints"][str(b)]["mae"] for r in sub] for p in POLICIES_B}
            isol = {
                p: [float(bool(r["recs"][p]["checkpoints"][str(b)]["isol"])) for r in sub]
                for p in POLICIES_B
            }
            wrong = {
                p: [float(r["recs"][p]["checkpoints"][str(b)]["wrong_isolate"]) for r in sub]
                for p in POLICIES_B
            }
            nret = {p: [r["recs"][p]["checkpoints"][str(b)]["n_retest"] for r in sub] for p in POLICIES_B}
            nq = {p: [r["recs"][p]["checkpoints"][str(b)]["n_query"] for r in sub] for p in POLICIES_B}
            inv = {p: [r["recs"][p]["checkpoints"][str(b)]["n_invalid_retest"] for r in sub] for p in POLICIES_B}
            nsc = {p: [r["recs"][p]["checkpoints"][str(b)]["n_score"] for r in sub] for p in POLICIES_B}
            el = {p: [r["recs"][p]["elapsed_s"] for r in sub] for p in POLICIES_B}
            for p in POLICIES_B:
                bblock[p] = {
                    "mae": mean_ci(mae[p]),
                    "isolate_rate": mean_ci(isol[p]),
                    "wrong_isolate_rate": mean_ci(wrong[p]),
                    "n_retest": mean_ci(nret[p]),
                    "n_query": mean_ci(nq[p]),
                    "invalid_retests": mean_ci(inv[p]),
                    "n_score": mean_ci(nsc[p]),
                    "elapsed_s": mean_ci(el[p]),
                }
            bblock["paired_cover_minus_compete"] = paired_delta(mae["cover"], mae["compete"])
            bblock["paired_retest_minus_compete"] = paired_delta(mae["retest_then_cover"], mae["compete"])
            bblock["paired_cover_minus_retest"] = paired_delta(mae["cover"], mae["retest_then_cover"])
            block["budgets"][str(b)] = bblock
        by[cause] = block
    return {
        "by_cause": by,
        "n_tasks": len(rows),
        "rows_compact": [
            {
                "task": r["task"],
                "cause": r["cause"],
                "pair_id": r["pair_id"],
                "dirty_x": r["dirty_x"],
                "t": r["t_break"],
                "init_xs": r["init_xs"],
                **{f"{p}_mae8": r["recs"][p]["checkpoints"]["8"]["mae"] for p in POLICIES_B},
                **{f"{p}_mae16": r["recs"][p]["checkpoints"]["16"]["mae"] for p in POLICIES_B},
                **{f"{p}_isol16": r["recs"][p]["checkpoints"]["16"]["isol"] for p in POLICIES_B},
                **{f"{p}_fam16": r["recs"][p]["checkpoints"]["16"]["family"] for p in POLICIES_B},
                **{f"{p}_nret16": r["recs"][p]["checkpoints"]["16"]["n_retest"] for p in POLICIES_B},
            }
            for r in rows
        ],
    }


def evaluate_A(split: str, n_bundles: int, seed0: int):
    rows = []
    for serial in range(n_bundles):
        for task in make_bundle(split, serial, seed0):
            for kind in ("Q", "QR"):
                rows.append(collect_frozen(task, kind))
    return summarize_A(rows), rows


def evaluate_B(split: str, n_bundles: int, seed0: int):
    tasks = []
    for serial in range(n_bundles):
        tasks.extend(make_bundle(split, serial, seed0))
    rows = []
    trajs = []
    for i, task in enumerate(tasks):
        recs = {p: run_policy_B(task, p) for p in POLICIES_B}
        rows.append(
            {
                "task": task.task_id,
                "cause": task.cause,
                "typ": task.typ,
                "pair_id": task.pair_id,
                "dirty_x": task.dirty_x,
                "t_break": task.t_break,
                "init_xs": sorted(task.init_obs),
                "recs": recs,
            }
        )
        trajs.append(
            {
                "task": task.task_id,
                "cause": task.cause,
                "pair_id": task.pair_id,
                "traj": {p: recs[p]["traj"] for p in POLICIES_B},
            }
        )
        if split == "eval" and (i + 1) % 45 == 0:
            print(f"B {split} {i+1}/{len(tasks)}", flush=True)
    return summarize_B(rows), rows, trajs


def library_checks() -> List[Dict]:
    out = []
    b = make_bundle("chk", 0, 0)
    inits = [tuple(sorted(t.init_obs)) for t in b]
    out.append({"name": "paired_init", "ok": len(set(inits)) == 1, "detail": inits})
    out.append(
        {
            "name": "not_shortcut_init",
            "ok": set(b[0].init_obs) != FORBIDDEN_INIT,
            "detail": sorted(b[0].init_obs),
        }
    )
    dirtys = []
    ts = []
    for serial in range(12):
        bun = make_bundle("div", serial, 9)
        dirtys.append(bun[0].dirty_x)
        ts.append(bun[0].t_break)
        out.append(
            {
                "name": f"not_shortcut_{serial}",
                "ok": set(bun[0].init_obs) != FORBIDDEN_INIT,
                "detail": sorted(bun[0].init_obs),
            }
        )
    out.append({"name": "dirty_varies", "ok": len(set(dirtys)) > 1, "detail": dirtys})
    out.append({"name": "t_varies", "ok": len(set(ts)) > 1, "detail": ts})
    for t in b:
        env = Env(t)
        got = {int(x): env.observe(int(x)) for x in t.init_obs}
        out.append({"name": f"init_obs_{t.cause}", "ok": got == t.init_obs, "detail": got})
        out.append(
            {
                "name": f"h0_conflict_{t.cause}",
                "ok": not mask_obs(H0, t.init_obs).any(),
                "detail": int(mask_obs(H0, t.init_obs).sum()),
            }
        )
    return out


def code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main():
    t0 = time.perf_counter()
    checks = library_checks()
    iso = dual_env_isolation()
    if not all(c["ok"] for c in checks):
        raise SystemExit(f"sanity failed: {checks}")
    if not (iso["honest_must_pass"] and iso["cheat_must_fail"]):
        raise SystemExit(f"isolation failed: {iso}")

    print("A dev...", flush=True)
    a_dev, _ = evaluate_A("dev", DEV_BUNDLES, DEV_SEED0)
    print("B dev...", flush=True)
    b_dev, _, _ = evaluate_B("dev", DEV_BUNDLES, DEV_SEED0)

    locked = {
        "N": N,
        "BUDGETS": list(BUDGETS),
        "CAUSES": list(CAUSES),
        "costs_ASSUMED": {"query": COST_QUERY, "retest": COST_RETEST, "isolate": 0},
        "isolate": "reversible trusted-set mask, cap 1, records kept",
        "old_fitter": "H0 all else H1 all else nearest neighbour",
        "new_fitter": "enumerate H0/H1 x isolate-cap-1; rank fewer isol then H0 then index",
        "policies_B": list(POLICIES_B),
        "compete": "1-step lookahead: expected MAE among remaining repairs, uniform ASSUMED",
        "forbidden_init": sorted(FORBIDDEN_INIT),
        "code_sha256": code_hash(),
        "note": "locked after 30-task development; eval seeds 20000+",
        "parameter_provenance": {
            "libraries": "ASSUMED inherited",
            "generator": "ASSUMED",
            "lookahead_prior": "ASSUMED uniform",
            "isolation": "MEASURED",
            "metrics": "MEASURED",
        },
    }
    (HERE / "PROTOCOL.json").write_text(json.dumps(locked, indent=2) + "\n")

    print("A eval...", flush=True)
    a_ev, _ = evaluate_A("eval", EVAL_BUNDLES, EVAL_SEED0)
    print("B eval...", flush=True)
    b_ev, _, b_traj = evaluate_B("eval", EVAL_BUNDLES, EVAL_SEED0)
    elapsed = time.perf_counter() - t0
    result = {
        "protocol": locked,
        "sanity": {"all_ok": True, "checks": checks},
        "isolation": iso,
        "A_dev": a_dev,
        "B_dev": b_dev,
        "A_eval": a_ev,
        "B_eval": b_ev,
        "timing_s": elapsed,
    }
    (HERE / "EVAL_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "A_ROWS.json").write_text(json.dumps(a_ev["rows_compact"], indent=2) + "\n")
    (HERE / "B_ROWS.json").write_text(json.dumps(b_ev["rows_compact"], indent=2) + "\n")
    (HERE / "B_TRAJ.json").write_text(json.dumps(b_traj, indent=2) + "\n")
    print("elapsed_s", round(elapsed, 3))
    print("iso", iso)
    print("init_shortcut_frac", a_ev["frac_init_shortcut"], "dirty_n", a_ev["n_unique_dirty_x"], "t_n", a_ev["n_unique_t"])
    for cause in CAUSES:
        print("==== A", cause)
        for kind in ("Q", "QR"):
            for b in ("8", "16"):
                blk = a_ev["by_cause"][cause]["kinds"][kind][b]
                print(
                    f"  {kind}@{b} old={blk['old_mae']['mean']:.3f}±{blk['old_mae']['se']:.3f} "
                    f"new={blk['new_mae']['mean']:.3f}±{blk['new_mae']['se']:.3f} "
                    f"isol={blk['new_isolate_rate']['mean']:.2f} H0={blk['new_H0_rate']:.2f} "
                    f"nn_old={blk['old_nn_rate']:.2f}"
                )
        print("==== B", cause)
        for b in ("8", "16"):
            blk = b_ev["by_cause"][cause]["budgets"][b]
            for p in POLICIES_B:
                g = blk[p]["mae"]
                print(
                    f"  {p:18s}@{b} MAE={g['mean']:.3f}±{g['se']:.3f} "
                    f"isol={blk[p]['isolate_rate']['mean']:.2f} "
                    f"wrong={blk[p]['wrong_isolate_rate']['mean']:.2f} "
                    f"ret={blk[p]['n_retest']['mean']:.2f} "
                    f"t={blk[p]['elapsed_s']['mean']:.4f}"
                )
            print("  paired cover-compete", {k: blk["paired_cover_minus_compete"][k] for k in ("mean", "se", "n_pos", "n_neg", "n_tie")})
            print("  paired retest-compete", {k: blk["paired_retest_minus_compete"][k] for k in ("mean", "se", "n_pos", "n_neg", "n_tie")})


if __name__ == "__main__":
    main()
