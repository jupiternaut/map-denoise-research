#!/usr/bin/env python3
"""Conflict: retest vs expand under a shared budget.

Question: with the same information, repair tools, and total budget, does
selecting retest vs expand from conflict evidence beat a fixed process on
final global MAE?

Apparatus only. Candidate ranges, dirt, and action costs are ASSUMED.
Policies never read cause labels, true parameters, or unqueried answers.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ORIG = Path("/home/grf/.hermes/attachments/outputs")
sys.path.insert(0, str(ORIG))
import mismatch_experiment as m  # noqa: E402

N = m.N
DOMAIN = m.DOMAIN
H0, H1 = m.H0, m.H1
COVER = m.COVER
N_INIT = 3
INIT_X = [int(x) for x in COVER[:N_INIT]]
BUDGETS = (8, 16)
MAX_BUDGET = 16
DEV_BUNDLES = 10
EVAL_BUNDLES = 60
DEV_SEED0 = 1
EVAL_SEED0 = 10_000
CAUSES = ("model", "transient", "persistent")
COST_QUERY = 1
COST_RETEST = 1
COST_EXPAND = 1
POLICIES = ("expand_first", "retest_first", "diagnose")


def mask_obs(table: np.ndarray, believed: Dict[int, int]) -> np.ndarray:
    live = np.ones(len(table), dtype=bool)
    for x, y in believed.items():
        live &= table[:, int(x)] == int(y)
    return live


def restoring_singles(believed: Dict[int, int], table: np.ndarray) -> List[int]:
    if mask_obs(table, believed).any():
        return []
    hits = []
    for x in believed:
        sub = {k: v for k, v in believed.items() if k != x}
        if mask_obs(table, sub).any():
            hits.append(int(x))
    return hits


def restoring_pairs(believed: Dict[int, int], table: np.ndarray) -> List[Tuple[int, int]]:
    if mask_obs(table, believed).any():
        return []
    xs = list(believed)
    out = []
    for i, x in enumerate(xs):
        for y in xs[i + 1 :]:
            sub = {k: v for k, v in believed.items() if k != x and k != y}
            if mask_obs(table, sub).any():
                out.append((int(x), int(y)))
    return out


def impact_order(believed: Dict[int, int], table: np.ndarray) -> Tuple[List[int], int]:
    """Rank observations by how much their removal restores consistency.

    Diagnostic clue only — not treated as proof that the observation is wrong.
    """
    n_score = len(table) * max(1, len(believed))
    singles = restoring_singles(believed, table)
    if singles:
        return sorted(singles), n_score
    pairs = restoring_pairs(believed, table)
    n_score += len(table) * max(1, len(believed) * (len(believed) - 1) // 2)
    score: Dict[int, int] = {int(x): 0 for x in believed}
    for a, y in pairs:
        score[a] += 1
        score[y] += 1
    ranked = sorted(score, key=lambda x: (-score[x], x))
    return ranked, n_score


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
    pair_id: str
    split: str


class Env:
    """Hidden-cause environment. Policies may call observe/retest only."""

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
    """Paired causes share init_obs.

    dirty_x=32 sits on the far side of t∈{24,32}, so {0,16} still lie on f0
    and H1 can explain the corrupted 32 with a different right-hand piece.
    A raw integer offset at x=16 cannot: 16 is between the other two init
    points, so no one-break H1 member matches the endpoints and misses the
    middle with slopes in {-1,0,1}. ASSUMED construction, not a claim about
    natural error rates.
    """
    t_choices = np.array([24, 32], dtype=np.int32)
    for _ in range(20_000):
        f0 = m.H0[int(rng.integers(0, len(m.H0)))].copy()
        a = int(f0[1] - f0[0])
        b = int(f0[0])
        t = int(rng.choice(t_choices))
        a2 = int(rng.choice(m.A_RANGE))
        b2 = int(rng.choice(m.B_RANGE))
        if a2 == a and b2 == b:
            continue
        dirty_x = 32
        dirty_y = int(a2 * dirty_x + b2)
        if dirty_y == int(f0[dirty_x]):
            continue
        f1 = np.empty(N, dtype=np.int32)
        left = DOMAIN < t
        f1[left] = a * DOMAIN[left] + b
        f1[~left] = a2 * DOMAIN[~left] + b2
        if m.in_library(H0, f1) or not m.in_library(H1, f1):
            continue
        init_obs = {0: int(f0[0]), 16: int(f0[16]), 32: int(dirty_y)}
        if int(f1[0]) != init_obs[0] or int(f1[16]) != init_obs[16] or int(f1[32]) != dirty_y:
            continue
        if mask_obs(H0, init_obs).any() or not mask_obs(H1, init_obs).any():
            continue
        return f0, f1, dict(init_obs), dirty_x, dirty_y
    raise RuntimeError("failed to sample shared instance")


def make_bundle(split: str, serial: int, seed0: int) -> List[Task]:
    rng = np.random.default_rng(seed0 + serial)
    f0, f1, init_obs, dirty_x, dirty_y = build_shared_instance(rng)
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
                pair_id=pair_id,
                split=split,
            )
        )
    return out


def choose_action(
    name: str,
    believed: Dict[int, int],
    retested: set,
    expanded: bool,
    remaining: int,
    costs: Dict[str, int],
) -> Tuple[Optional[str], Optional[int], int]:
    table = H1 if expanded else H0
    inconsistent = not bool(mask_obs(table, believed).any())
    n_score = 0
    cq, cr, ce = costs["query"], costs["retest"], costs["expand"]

    if name == "expand_first":
        if inconsistent and not expanded and remaining >= ce:
            return "expand", None, n_score
        if remaining >= cq:
            return "query", next_cover(believed), n_score
        return None, None, n_score

    if name == "retest_first":
        if inconsistent:
            for x in believed:
                if int(x) not in retested and remaining >= cr:
                    return "retest", int(x), n_score
            if not expanded and remaining >= ce:
                return "expand", None, n_score
        if remaining >= cq:
            return "query", next_cover(believed), n_score
        return None, None, n_score

    if name == "diagnose":
        if inconsistent:
            ranked, n_score = impact_order(believed, table)
            pending = [x for x in ranked if x not in retested]
            if pending and remaining >= cr:
                return "retest", int(pending[0]), n_score
            if not expanded and remaining >= ce:
                return "expand", None, n_score
            leftover = [int(x) for x in believed if int(x) not in retested]
            if leftover and remaining >= cr:
                return "retest", leftover[0], n_score
        if remaining >= cq:
            return "query", next_cover(believed), n_score
        return None, None, n_score

    raise ValueError(name)


def deliver(believed: Dict[int, int], expanded: bool) -> np.ndarray:
    table = H1 if expanded else H0
    live = mask_obs(table, believed)
    if live.any():
        return table[live][0].copy()
    f = np.empty(N, dtype=np.int32)
    xs = np.array(sorted(believed), dtype=np.int32)
    ys = np.array([believed[int(x)] for x in xs], dtype=np.int32)
    for x in DOMAIN:
        f[x] = ys[int(np.argmin(np.abs(xs - x)))]
    return f


def snapshot(believed, expanded, truth, cost, n_query, n_retest, n_expand, n_score, n_invalid_retest):
    table = H1 if expanded else H0
    live = mask_obs(table, believed)
    pred = deliver(believed, expanded)
    l1 = int(np.abs(pred.astype(np.int64) - truth.astype(np.int64)).sum())
    return {
        "budget": int(cost),
        "n_query": n_query,
        "n_retest": n_retest,
        "n_expand": n_expand,
        "n_score": n_score,
        "n_invalid_retest": n_invalid_retest,
        "expanded": bool(expanded),
        "consistent": bool(live.any()),
        "n_live": int(live.sum()),
        "global_l1": l1,
        "mae": float(l1) / float(N),
    }


def run_policy(task: Task, name: str, max_budget: int = MAX_BUDGET, costs=None) -> Dict:
    costs = costs or {"query": COST_QUERY, "retest": COST_RETEST, "expand": COST_EXPAND}
    t0 = time.perf_counter()
    env = Env(task)
    believed: Dict[int, int] = {}
    retested = set()
    expanded = False
    cost = 0
    n_query = n_retest = n_expand = n_score = n_invalid_retest = 0
    traj = []
    checkpoints = {}

    def note():
        for b in BUDGETS:
            if cost >= b and str(b) not in checkpoints:
                checkpoints[str(b)] = snapshot(
                    believed, expanded, task.truth, cost, n_query, n_retest,
                    n_expand, n_score, n_invalid_retest,
                )

    for x in INIT_X:
        y = env.observe(int(x))
        believed[int(x)] = int(y)
        cost += costs["query"]
        n_query += 1
        traj.append({"a": "query", "x": int(x), "y": int(y), "cost": cost})
    note()

    while cost < max_budget:
        rem = max_budget - cost
        act, x, scored = choose_action(name, believed, retested, expanded, rem, costs)
        n_score += int(scored)
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
            changed = int(y2) != int(believed[int(x)])
            if not changed:
                n_invalid_retest += 1
            believed[int(x)] = int(y2)
            retested.add(int(x))
            cost += costs["retest"]
            n_retest += 1
            traj.append({"a": "retest", "x": int(x), "y": int(y2), "changed": bool(changed), "cost": cost})
        elif act == "expand":
            expanded = True
            cost += costs["expand"]
            n_expand += 1
            traj.append({"a": "expand", "x": None, "cost": cost})
        else:
            raise RuntimeError(act)
        note()

    for b in BUDGETS:
        if str(b) not in checkpoints:
            checkpoints[str(b)] = snapshot(
                believed, expanded, task.truth, cost, n_query, n_retest,
                n_expand, n_score, n_invalid_retest,
            )

    final = snapshot(
        believed, expanded, task.truth, cost, n_query, n_retest,
        n_expand, n_score, n_invalid_retest,
    )
    unnecessary_expand = bool(expanded and task.cause == "transient")
    unresolved = not bool(final["consistent"])
    return {
        "policy": name,
        "task": task.task_id,
        "cause": task.cause,
        "typ": task.typ,
        "pair_id": task.pair_id,
        "cost": cost,
        "elapsed_s": time.perf_counter() - t0,
        "unnecessary_expand": unnecessary_expand,
        "unresolved": unresolved,
        "checkpoints": checkpoints,
        "final": final,
        "traj": traj,
        "believed": {str(k): int(v) for k, v in believed.items()},
        "truth_hash": hashlib.sha256(task.truth.tobytes()).hexdigest()[:16],
    }


def honest_next(name, believed, retested, expanded, remaining, costs):
    act, x, _ = choose_action(name, believed, retested, expanded, remaining, costs)
    return act, x


def cheat_next(env: Env, believed, retested, expanded, remaining, costs):
    """Peeks hidden labels. Isolation must catch this."""
    u = [int(x) for x in range(N) if int(x) not in believed]
    if u:
        hidden = np.array([int(env._truth[x]) for x in u], dtype=np.int64)
        score = hidden * 31 + np.array(u, dtype=np.int64)
        peek_x = int(u[int(np.argmax(score))])
    else:
        peek_x = None
    clean_obs = {int(x): int(env._truth[int(x)]) for x in believed}
    h0_ok = bool(mask_obs(H0, clean_obs).any())
    if not mask_obs(H0 if not expanded else H1, believed).any():
        if h0_ok and remaining >= costs["retest"]:
            dirtyish = [
                int(x)
                for x in believed
                if int(believed[int(x)]) != int(env._truth[int(x)]) and int(x) not in retested
            ]
            if dirtyish:
                return "retest", dirtyish[0]
            pending = [int(x) for x in believed if int(x) not in retested]
            if pending:
                return "retest", pending[0]
        if not expanded and remaining >= costs["expand"]:
            return "expand", None
    if peek_x is not None and remaining >= costs["query"]:
        return "query", peek_x
    return None, None


def dual_env_isolation(costs=None) -> Dict:
    costs = costs or {"query": COST_QUERY, "retest": COST_RETEST, "expand": COST_EXPAND}
    trials = []
    for policy in list(POLICIES) + ["cheat"]:
        for serial in range(4):
            bundle = make_bundle("iso", serial, seed0=50_000)
            for task in bundle:
                env_a = Env(task)
                believed = {}
                retested = set()
                expanded = False
                cost = 0
                for x in INIT_X:
                    believed[int(x)] = env_a.observe(int(x))
                    cost += costs["query"]
                steps = 0
                while cost < 6 and steps < 3:
                    rem = 8 - cost
                    if policy == "cheat":
                        act, x = cheat_next(env_a, believed, retested, expanded, rem, costs)
                    else:
                        act, x = honest_next(policy, believed, retested, expanded, rem, costs)
                    if act is None:
                        break
                    if act == "query" and x is not None and int(x) not in believed:
                        believed[int(x)] = env_a.observe(int(x))
                        cost += costs["query"]
                    elif act == "retest":
                        y2 = env_a.retest(int(x))
                        believed[int(x)] = int(y2)
                        retested.add(int(x))
                        cost += costs["retest"]
                    elif act == "expand":
                        expanded = True
                        cost += costs["expand"]
                    steps += 1
                env_b = env_a.clone_unqueried_rewritten(believed)
                rem = 16 - cost
                if policy == "cheat":
                    a = cheat_next(env_a, dict(believed), set(retested), expanded, rem, costs)
                    b = cheat_next(env_b, believed, set(retested), expanded, rem, costs)
                else:
                    a = honest_next(policy, dict(believed), set(retested), expanded, rem, costs)
                    b = honest_next(policy, dict(believed), set(retested), expanded, rem, costs)
                trials.append(
                    {
                        "policy": policy,
                        "cause": task.cause,
                        "next_a": a,
                        "next_b": b,
                        "ok": a == b,
                        "envs_distinct": env_a is not env_b
                        and not np.array_equal(env_a._truth, env_b._truth),
                    }
                )
    by: Dict[str, list] = {}
    for t in trials:
        by.setdefault(t["policy"], []).append(t)
    summary = {
        p: {
            "n": len(ts),
            "n_fail": sum(1 for t in ts if not t["ok"]),
            "all_ok": all(t["ok"] for t in ts),
        }
        for p, ts in by.items()
    }
    return {
        "honest_must_pass": all(summary[p]["all_ok"] for p in POLICIES),
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


def summarize(split: str, rows: List[Dict], costs: Dict) -> Dict:
    by_cause = {}
    for cause in CAUSES:
        sub = [r for r in rows if r["cause"] == cause]
        hashes = {r["recs"][POLICIES[0]]["truth_hash"] for r in sub}
        block = {"n_tasks": len(sub), "n_distinct_truths": len(hashes), "budgets": {}}
        for b in BUDGETS:
            bblock = {}
            mae = {
                p: [r["recs"][p]["checkpoints"][str(b)]["mae"] for r in sub]
                for p in POLICIES
            }
            l1 = {
                p: [r["recs"][p]["checkpoints"][str(b)]["global_l1"] for r in sub]
                for p in POLICIES
            }
            unexp = {
                p: [float(r["recs"][p]["unnecessary_expand"]) for r in sub]
                for p in POLICIES
            }
            inv = {
                p: [r["recs"][p]["checkpoints"][str(b)]["n_invalid_retest"] for r in sub]
                for p in POLICIES
            }
            unres = {
                p: [float(not r["recs"][p]["checkpoints"][str(b)]["consistent"]) for r in sub]
                for p in POLICIES
            }
            nexp = {
                p: [r["recs"][p]["checkpoints"][str(b)]["n_expand"] for r in sub]
                for p in POLICIES
            }
            nret = {
                p: [r["recs"][p]["checkpoints"][str(b)]["n_retest"] for r in sub]
                for p in POLICIES
            }
            for p in POLICIES:
                bblock[p] = {
                    "mae": mean_ci(mae[p]),
                    "global_l1": mean_ci(l1[p]),
                    "unnecessary_expand_rate": mean_ci(unexp[p]),
                    "invalid_retests": mean_ci(inv[p]),
                    "unresolved_rate": mean_ci(unres[p]),
                    "n_expand": mean_ci(nexp[p]),
                    "n_retest": mean_ci(nret[p]),
                    "elapsed_s": mean_ci([r["recs"][p]["elapsed_s"] for r in sub]),
                }
            bblock["paired_mae_retest_minus_diagnose"] = paired_delta(mae["retest_first"], mae["diagnose"])
            bblock["paired_mae_expand_minus_diagnose"] = paired_delta(mae["expand_first"], mae["diagnose"])
            bblock["paired_mae_retest_minus_expand"] = paired_delta(mae["expand_first"], mae["retest_first"])
            block["budgets"][str(b)] = bblock
        by_cause[cause] = block
    return {
        "split": split,
        "n_tasks": len(rows),
        "costs_ASSUMED": costs,
        "by_cause": by_cause,
        "rows_compact": [
            {
                "task": r["task"],
                "cause": r["cause"],
                "pair_id": r["pair_id"],
                **{f"{p}_mae16": r["recs"][p]["checkpoints"]["16"]["mae"] for p in POLICIES},
                **{f"{p}_mae8": r["recs"][p]["checkpoints"]["8"]["mae"] for p in POLICIES},
                **{f"{p}_expand": r["recs"][p]["final"]["n_expand"] for p in POLICIES},
                **{f"{p}_retest": r["recs"][p]["final"]["n_retest"] for p in POLICIES},
                **{f"{p}_unres": r["recs"][p]["unresolved"] for p in POLICIES},
            }
            for r in rows
        ],
    }


def evaluate(split: str, n_bundles: int, seed0: int, costs=None):
    costs = costs or {"query": COST_QUERY, "retest": COST_RETEST, "expand": COST_EXPAND}
    tasks = []
    for serial in range(n_bundles):
        tasks.extend(make_bundle(split, serial, seed0))
    rows = []
    trajs = []
    for i, task in enumerate(tasks):
        recs = {p: run_policy(task, p, MAX_BUDGET, costs) for p in POLICIES}
        rows.append(
            {
                "task": task.task_id,
                "cause": task.cause,
                "typ": task.typ,
                "pair_id": task.pair_id,
                "init_obs": {str(k): int(v) for k, v in task.init_obs.items()},
                "recs": recs,
            }
        )
        trajs.append(
            {
                "task": task.task_id,
                "cause": task.cause,
                "pair_id": task.pair_id,
                "traj": {p: recs[p]["traj"] for p in POLICIES},
            }
        )
        if split == "eval" and (i + 1) % 45 == 0:
            print(f"{split} {i+1}/{len(tasks)}", flush=True)
    return summarize(split, rows, costs), rows, trajs


def paired_history_check(n: int = 10) -> Dict:
    ok = 0
    n_tot = 0
    for serial in range(n):
        bundle = make_bundle("paircheck", serial, seed0=3)
        h0 = bundle[0].init_obs
        for t in bundle:
            n_tot += 1
            if t.init_obs == h0:
                ok += 1
    return {"n": n_tot, "n_ok": ok, "all_ok": ok == n_tot}


def library_checks() -> List[Dict]:
    out = []
    b = make_bundle("chk", 0, 0)
    for t in b:
        env = Env(t)
        got = {int(x): env.observe(int(x)) for x in INIT_X}
        out.append(
            {
                "name": f"init_matches_declared_{t.cause}",
                "ok": got == t.init_obs,
                "detail": {str(k): int(v) for k, v in got.items()},
            }
        )
        out.append(
            {
                "name": f"h0_conflict_at_init_{t.cause}",
                "ok": not mask_obs(H0, t.init_obs).any(),
                "detail": int(mask_obs(H0, t.init_obs).sum()),
            }
        )
    ph = paired_history_check(10)
    out.append({"name": "init_x", "ok": INIT_X == [0, 32, 16], "detail": INIT_X})
    out.append({"name": "paired_histories", "ok": ph["all_ok"], "detail": ph})
    return out


def code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main():
    t0 = time.perf_counter()
    costs = {"query": COST_QUERY, "retest": COST_RETEST, "expand": COST_EXPAND}
    checks = library_checks()
    iso = dual_env_isolation(costs)
    if not all(c.get("ok", False) for c in checks):
        raise SystemExit(f"sanity failed: {checks}")
    if not (iso["honest_must_pass"] and iso["cheat_must_fail"]):
        raise SystemExit(f"isolation failed: {iso}")

    print("dev...", flush=True)
    dev_sum, _dev_rows, _ = evaluate("dev", DEV_BUNDLES, DEV_SEED0, costs)
    locked = {
        "N": N,
        "N_INIT": N_INIT,
        "INIT_X": INIT_X,
        "BUDGETS": list(BUDGETS),
        "CAUSES": list(CAUSES),
        "costs_ASSUMED": costs,
        "H0": int(len(H0)),
        "H1": int(len(H1)),
        "policies": list(POLICIES),
        "diagnose": "retest highest-impact restoring observation, else expand, else covering query",
        "delivery": "first live row of active family (H0 until expand, else H1); else nearest observed",
        "code_sha256": code_hash(),
        "note": "rules locked after 30-task development; eval uses new seeds",
        "parameter_provenance": {
            "library_ranges": "ASSUMED inherited from mismatch_experiment",
            "INIT_X": "ASSUMED covering prefix of length 3",
            "dirty_offset": "ASSUMED",
            "action_costs": "ASSUMED",
            "isolation": "MEASURED",
            "metrics": "MEASURED",
        },
    }
    (HERE / "PROTOCOL.json").write_text(json.dumps(locked, indent=2) + "\n")
    (HERE / "DEV_ROWS.json").write_text(json.dumps(dev_sum["rows_compact"], indent=2) + "\n")

    print("eval...", flush=True)
    ev_sum, _ev_rows, ev_traj = evaluate("eval", EVAL_BUNDLES, EVAL_SEED0, costs)
    elapsed = time.perf_counter() - t0
    result = {
        "protocol": locked,
        "sanity": {"all_ok": True, "checks": checks},
        "isolation": iso,
        "dev": dev_sum,
        "eval": ev_sum,
        "timing_s": elapsed,
        "n_eval_tasks": ev_sum["n_tasks"],
        "n_distinct_truths_by_cause": {
            c: ev_sum["by_cause"][c]["n_distinct_truths"] for c in CAUSES
        },
    }
    (HERE / "EVAL_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "EVAL_ROWS.json").write_text(json.dumps(ev_sum["rows_compact"], indent=2) + "\n")
    (HERE / "EVAL_TRAJ.json").write_text(json.dumps(ev_traj, indent=2) + "\n")
    print("wrote PROTOCOL/EVAL_SUMMARY")
    print("isolation", iso)
    print("elapsed_s", round(elapsed, 3))
    for cause in CAUSES:
        print(f"--- {cause} ---")
        for b in ("8", "16"):
            block = ev_sum["by_cause"][cause]["budgets"][b]
            print(f"  budget {b}")
            for p in POLICIES:
                g = block[p]["mae"]
                print(
                    f"    {p:14s} MAE={g['mean']:.4f}±{g['se']:.3f}  "
                    f"exp={block[p]['n_expand']['mean']:.2f}  "
                    f"ret={block[p]['n_retest']['mean']:.2f}  "
                    f"unres={block[p]['unresolved_rate']['mean']:.3f}"
                )
            print(
                "    paired retest-diag",
                {k: block["paired_mae_retest_minus_diagnose"][k] for k in ("mean", "se", "n_pos", "n_neg", "n_tie")},
            )


if __name__ == "__main__":
    main()
