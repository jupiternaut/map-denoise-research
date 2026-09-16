#!/usr/bin/env python3
"""Retest vs expand under a shared budget.

Three hidden causes, same initial visible history within a pair.
Policies share libraries, fitter, retest API, and delivery.
They differ only in whether they retest or expand.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
N = 64
DOMAIN = np.arange(N, dtype=np.int32)
A_RANGE = np.array([-1, 0, 1], dtype=np.int32)
B_RANGE = np.arange(-8, 9, dtype=np.int32)
T_RANGE = np.array([16, 24, 32, 40, 48], dtype=np.int32)
INIT_TRUE = (0, 32)
CONFLICT_X = 48
INIT_X = INIT_TRUE + (CONFLICT_X,)
COST_OBS = 1
COST_RETEST = 1
COST_EXPAND = 2
BUDGETS = (8, 16)
DEV_PAIRS = 10
EVAL_PAIRS = 60
DEV_SEED0 = 101
EVAL_SEED0 = 20_000


def build_libraries() -> Tuple[np.ndarray, np.ndarray]:
    h0 = np.stack([a * DOMAIN + b for a in A_RANGE for b in B_RANGE], axis=0).astype(np.int32)
    pieces = []
    for t in T_RANGE:
        left = DOMAIN < t
        right = ~left
        xl, xr = DOMAIN[left], DOMAIN[right]
        for a1 in A_RANGE:
            for b1 in B_RANGE:
                left_vals = a1 * xl + b1
                for a2 in A_RANGE:
                    for b2 in B_RANGE:
                        if a1 == a2 and b1 == b2:
                            continue
                        f = np.empty(N, dtype=np.int32)
                        f[left] = left_vals
                        f[right] = a2 * xr + b2
                        pieces.append(f)
    h1 = np.concatenate([h0, np.stack(pieces, axis=0)], axis=0)
    return h0, h1


H0, H1 = build_libraries()


def mask_consistent(table: np.ndarray, obs: Dict[int, int]) -> np.ndarray:
    live = np.ones(len(table), dtype=bool)
    for x, y in obs.items():
        live &= table[:, int(x)] == int(y)
    return live


def first_fit(table: np.ndarray, obs: Dict[int, int]) -> Optional[np.ndarray]:
    live = mask_consistent(table, obs)
    if not live.any():
        return None
    return table[live][0].copy()


def global_l1(pred: np.ndarray, truth: np.ndarray) -> int:
    return int(np.abs(pred.astype(np.int64) - truth.astype(np.int64)).sum())


def nearest_deliver(obs: Dict[int, int]) -> np.ndarray:
    xs = np.array(sorted(obs), dtype=np.int32)
    ys = np.array([obs[int(x)] for x in xs], dtype=np.int32)
    out = np.empty(N, dtype=np.int32)
    for x in DOMAIN:
        out[x] = ys[int(np.argmin(np.abs(xs - x)))]
    return out


def covering_next(queried: Sequence[int]) -> Optional[int]:
    have = set(int(x) for x in queried)
    step = N
    while step >= 1:
        for x in range(0, N, step):
            if x not in have:
                return x
        step //= 2
    return None


@dataclass
class Task:
    task_id: str
    pair_id: str
    cause: str
    truth: np.ndarray
    init_obs: Dict[int, int]
    conflict_x: int
    polluted_y: int
    split: str

    def label(self, x: int, retest: bool, seen_once: Dict[int, int]) -> int:
        x = int(x)
        if self.cause == "persistent" and x == self.conflict_x:
            return int(self.polluted_y)
        if self.cause == "transient" and x == self.conflict_x and (not retest) and x not in seen_once:
            return int(self.polluted_y)
        return int(self.truth[x])


def find_piecewise_matching(init_obs: Dict[int, int]) -> Optional[np.ndarray]:
    pieces = H1[len(H0) :]
    live = mask_consistent(pieces, init_obs)
    if not live.any():
        return None
    rows = pieces[live]
    for row in rows:
        if not np.any(np.all(H0 == row[None, :], axis=1)):
            return row.copy()
    return rows[0].copy()


def h0_row(a: int, b: int) -> np.ndarray:
    return (a * DOMAIN + b).astype(np.int32)


def make_pair(split: str, serial: int, seed0: int) -> List[Task]:
    rng = np.random.default_rng(seed0 + serial)
    for _ in range(8000):
        a = int(rng.choice(A_RANGE))
        b = int(rng.choice(B_RANGE))
        f0 = h0_row(a, b)
        xc = int(CONFLICT_X)
        delta = int(rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))
        polluted = int(f0[xc] + delta)
        init_obs = {int(x): int(f0[x]) for x in INIT_TRUE}
        init_obs[xc] = polluted
        if mask_consistent(H0, init_obs).any():
            continue
        f1 = find_piecewise_matching(init_obs)
        if f1 is None:
            continue
        if np.array_equal(f1, f0):
            continue
        pair_id = f"{split}-pair-{serial:03d}"
        return [
            Task(f"{pair_id}-model", pair_id, "model", f1, dict(init_obs), xc, polluted, split),
            Task(f"{pair_id}-transient", pair_id, "transient", f0, dict(init_obs), xc, polluted, split),
            Task(f"{pair_id}-persistent", pair_id, "persistent", f0, dict(init_obs), xc, polluted, split),
        ]
    raise RuntimeError(f"could not build pair {split} {serial}")


@dataclass
class Agent:
    obs: Dict[int, int]
    expanded: bool = False
    cost: int = 0
    n_obs: int = 0
    n_retest: int = 0
    n_expand: int = 0
    n_score: int = 0
    traj: List[Dict] = field(default_factory=list)
    confirmed: set = field(default_factory=set)
    retested: set = field(default_factory=set)
    first_seen: Dict[int, int] = field(default_factory=dict)

    def library(self) -> np.ndarray:
        return H1 if self.expanded else H0

    def live(self) -> np.ndarray:
        self.n_score += 1
        return mask_consistent(self.library(), self.obs)


def blame_points(obs: Dict[int, int], table: np.ndarray) -> List[Tuple[int, int]]:
    if mask_consistent(table, obs).any():
        return []
    ranked = []
    for x in obs:
        reduced = {k: v for k, v in obs.items() if k != x}
        n = int(mask_consistent(table, reduced).sum())
        if n > 0:
            ranked.append((x, n))
    ranked.sort(key=lambda t: (-t[1], t[0]))
    return ranked


def deliver(agent: Agent) -> np.ndarray:
    fit = first_fit(agent.library(), agent.obs)
    if fit is not None:
        return fit
    return nearest_deliver(agent.obs)


class Env:
    def __init__(self, task: Task, budget: int):
        self.task = task
        self.budget = budget
        self.agent = Agent(obs=dict(task.init_obs), cost=COST_OBS * len(task.init_obs), n_obs=len(task.init_obs))
        self.agent.first_seen = dict(task.init_obs)
        self.agent.traj.append({"action": "init", "obs": dict(task.init_obs), "cost": self.agent.cost})

    def remaining(self) -> int:
        return self.budget - self.agent.cost

    def observe(self, x: int) -> Optional[int]:
        x = int(x)
        if x in self.agent.obs or self.remaining() < COST_OBS:
            return None
        y = self.task.label(x, retest=False, seen_once=self.agent.first_seen)
        self.agent.obs[x] = y
        self.agent.first_seen[x] = y
        self.agent.cost += COST_OBS
        self.agent.n_obs += 1
        self.agent.traj.append({"action": "observe", "x": x, "y": int(y)})
        return y

    def retest(self, x: int) -> Optional[int]:
        x = int(x)
        if x not in self.agent.obs or self.remaining() < COST_RETEST:
            return None
        y = self.task.label(x, retest=True, seen_once=self.agent.first_seen)
        old = self.agent.obs[x]
        self.agent.obs[x] = y
        self.agent.retested.add(x)
        self.agent.cost += COST_RETEST
        self.agent.n_retest += 1
        changed = int(y) != int(old)
        if not changed:
            self.agent.confirmed.add(x)
        self.agent.traj.append({"action": "retest", "x": x, "old": int(old), "y": int(y), "changed": changed})
        return y

    def expand(self) -> bool:
        if self.agent.expanded or self.remaining() < COST_EXPAND:
            return False
        self.agent.expanded = True
        self.agent.cost += COST_EXPAND
        self.agent.n_expand += 1
        self.agent.traj.append({"action": "expand"})
        return True


def policy_direct_expand(env: Env) -> None:
    while env.remaining() > 0:
        if (not env.agent.live().any()) and (not env.agent.expanded):
            if env.expand():
                continue
            break
        x = covering_next(env.agent.obs)
        if x is None or env.observe(x) is None:
            break


def policy_retest_first(env: Env) -> None:
    while env.remaining() > 0:
        if (not env.agent.live().any()) and (not env.agent.expanded):
            pending = [x for x in sorted(env.agent.obs) if x not in env.agent.retested]
            if pending:
                if env.retest(pending[0]) is None:
                    break
                continue
            if env.expand():
                continue
            x = covering_next(env.agent.obs)
            if x is None or env.observe(x) is None:
                break
            continue
        x = covering_next(env.agent.obs)
        if x is None or env.observe(x) is None:
            break


def policy_conflict(env: Env) -> None:
    while env.remaining() > 0:
        if env.agent.live().any() or env.agent.expanded:
            x = covering_next(env.agent.obs)
            if x is None or env.observe(x) is None:
                break
            continue
        blamed = [x for x, _n in blame_points(env.agent.obs, H0) if x not in env.agent.retested]
        if blamed:
            if env.retest(blamed[0]) is None:
                break
            continue
        if env.expand():
            continue
        x = covering_next(env.agent.obs)
        if x is None or env.observe(x) is None:
            break


POLICIES: Dict[str, Callable[[Env], None]] = {
    "direct_expand": policy_direct_expand,
    "retest_first": policy_retest_first,
    "conflict": policy_conflict,
}


def run_policy(task: Task, name: str, budget: int) -> Dict:
    t0 = time.perf_counter()
    env = Env(task, budget)
    POLICIES[name](env)
    pred = deliver(env.agent)
    l1 = global_l1(pred, task.truth)
    return {
        "task": task.task_id,
        "pair": task.pair_id,
        "cause": task.cause,
        "policy": name,
        "budget": budget,
        "l1": l1,
        "cost": env.agent.cost,
        "n_obs": env.agent.n_obs,
        "n_retest": env.agent.n_retest,
        "n_expand": env.agent.n_expand,
        "n_score": env.agent.n_score,
        "expanded": env.agent.expanded,
        "unnecessary_expand": int(env.agent.expanded and task.cause == "transient"),
        "wasted_retest": int(env.agent.n_retest > 0 and task.cause == "model"),
        "unresolved": int(l1 > 0),
        "elapsed_s": time.perf_counter() - t0,
        "traj": env.agent.traj,
        "truth_hash": hashlib.sha256(task.truth.tobytes()).hexdigest()[:16],
    }


def next_action(env: Env, name: str) -> Optional[Tuple]:
    """Inspect the action a policy would take now, without executing it."""
    live = env.agent.live().any()
    if name == "direct_expand":
        if (not live) and (not env.agent.expanded):
            if env.remaining() >= COST_EXPAND:
                return ("expand",)
            return None
        x = covering_next(env.agent.obs)
        return ("observe", x) if x is not None else None
    if name == "retest_first":
        if (not live) and (not env.agent.expanded):
            pending = [x for x in sorted(env.agent.obs) if x not in env.agent.retested]
            if pending and env.remaining() >= COST_RETEST:
                return ("retest", pending[0])
            if env.remaining() >= COST_EXPAND:
                return ("expand",)
        x = covering_next(env.agent.obs)
        return ("observe", x) if x is not None else None
    if name == "conflict":
        if live or env.agent.expanded:
            x = covering_next(env.agent.obs)
            return ("observe", x) if x is not None else None
        blamed = [x for x, _n in blame_points(env.agent.obs, H0) if x not in env.agent.retested]
        if blamed and env.remaining() >= COST_RETEST:
            return ("retest", blamed[0])
        if env.remaining() >= COST_EXPAND:
            return ("expand",)
        x = covering_next(env.agent.obs)
        return ("observe", x) if x is not None else None
    raise KeyError(name)


def cheating_next(task: Task, queried: Dict[int, int]) -> Tuple:
    """Illegal: reads hidden unqueried labels. Used only to test the detector."""
    u = [i for i in range(N) if i not in queried]
    return ("observe", max(u, key=lambda x: (int(task.truth[x]), x)))


def isolation_tests() -> Dict:
    pairs = [make_pair("iso", i, 50_000) for i in range(6)]
    honest = []
    cheat = []
    pair_same_init = []
    for pair in pairs:
        inits = [tuple(sorted(t.init_obs.items())) for t in pair]
        pair_same_init.append(len(set(inits)) == 1)
        for name in POLICIES:
            acts = [next_action(Env(t, 16), name) for t in pair]
            honest.append(len(set(acts)) == 1)
        task = pair[0]
        env_a = Env(task, 16)
        queried = dict(env_a.agent.obs)
        mutated = task.truth.copy()
        u = [i for i in range(N) if i not in queried]
        # Force the cheater's current pick to lose, and give another unqueried
        # point a unique sentinel. Honest policies never read these labels.
        xa = cheating_next(task, queried)[1]
        other = next(x for x in u if x != xa)
        mutated[xa] = -10**9
        mutated[other] = 10**9
        t2 = Task(
            task.task_id + "-mut",
            task.pair_id,
            task.cause,
            mutated,
            dict(task.init_obs),
            task.conflict_x,
            task.polluted_y,
            "iso",
        )
        env_b = Env(t2, 16)
        for name in POLICIES:
            honest.append(next_action(env_a, name) == next_action(env_b, name))
        cheat.append(cheating_next(task, queried) != cheating_next(t2, queried))
        blamed = blame_points(env_a.agent.obs, H0)
        honest.append(bool(blamed) and blamed[0][0] == task.conflict_x)
    return {
        "honest_all_ok": all(honest),
        "n_honest": len(honest),
        "cheat_caught_frac": float(np.mean(cheat)),
        "cheat_all_caught": all(cheat),
        "n_pairs": len(pairs),
        "paired_init_identical": all(pair_same_init),
        "n_cheat": len(cheat),
    }


def mean_se(xs: Sequence[float]) -> Dict:
    arr = np.array(list(xs), dtype=np.float64)
    n = arr.size
    m = float(arr.mean()) if n else float("nan")
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"n": int(n), "mean": m, "se": se}


def paired(a: Sequence[float], b: Sequence[float]) -> Dict:
    d = np.array(list(a), dtype=np.float64) - np.array(list(b), dtype=np.float64)
    s = mean_se(d)
    s.update(n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()), n_tie=int((d == 0).sum()))
    return s


def summarize(rows: List[Dict], split: str) -> Dict:
    by: Dict = {}
    for budget in BUDGETS:
        by[str(budget)] = {}
        sub_b = [r for r in rows if r["budget"] == budget]
        for cause in ("model", "transient", "persistent"):
            sub = [r for r in sub_b if r["cause"] == cause]
            block = {}
            for pol in POLICIES:
                pr = [r for r in sub if r["policy"] == pol]
                block[pol] = {
                    "l1": mean_se([r["l1"] for r in pr]),
                    "unresolved": mean_se([r["unresolved"] for r in pr]),
                    "n_expand": mean_se([r["n_expand"] for r in pr]),
                    "n_retest": mean_se([r["n_retest"] for r in pr]),
                    "unnecessary_expand": mean_se([r["unnecessary_expand"] for r in pr]),
                    "wasted_retest": mean_se([r["wasted_retest"] for r in pr]),
                    "elapsed_s": mean_se([r["elapsed_s"] for r in pr]),
                    "cost": mean_se([r["cost"] for r in pr]),
                }
            ids = sorted({r["task"] for r in sub})

            def col(pol: str) -> List[float]:
                m = {r["task"]: r["l1"] for r in sub if r["policy"] == pol}
                return [m[i] for i in ids]

            block["paired_l1_retest_minus_direct"] = paired(col("retest_first"), col("direct_expand"))
            block["paired_l1_conflict_minus_direct"] = paired(col("conflict"), col("direct_expand"))
            block["paired_l1_conflict_minus_retest"] = paired(col("conflict"), col("retest_first"))
            by[str(budget)][cause] = block
        overall = {}
        for pol in POLICIES:
            pr = [r for r in sub_b if r["policy"] == pol]
            overall[pol] = {"l1": mean_se([r["l1"] for r in pr])}
        by[str(budget)]["all"] = overall
    return {
        "split": split,
        "n_rows": len(rows),
        "n_tasks": len({r["task"] for r in rows}),
        "n_pairs": len({r["pair"] for r in rows}),
        "n_distinct_truths": len({r["truth_hash"] for r in rows}),
        "by_budget": by,
    }


def evaluate(split: str, n_pairs: int, seed0: int) -> Tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    for serial in range(n_pairs):
        for task in make_pair(split, serial, seed0):
            for budget in BUDGETS:
                for name in POLICIES:
                    rows.append(run_policy(task, name, budget))
        if split == "eval" and (serial + 1) % 15 == 0:
            print(f"{split} pairs {serial+1}/{n_pairs}", flush=True)
    return rows, summarize(rows, split)


def main() -> None:
    t0 = time.perf_counter()
    iso = isolation_tests()
    if not iso["honest_all_ok"]:
        raise SystemExit(f"isolation failed: {iso}")
    if iso["cheat_caught_frac"] < 0.5:
        raise SystemExit(f"cheat detector too weak: {iso}")

    dev_rows, dev_sum = evaluate("dev", DEV_PAIRS, DEV_SEED0)
    protocol = {
        "ASSUMED": {
            "INIT_X": list(INIT_X),
            "COST_OBS": COST_OBS,
            "COST_RETEST": COST_RETEST,
            "COST_EXPAND": COST_EXPAND,
            "BUDGETS": list(BUDGETS),
            "n_h0": int(len(H0)),
            "n_h1": int(len(H1)),
            "causes": ["model", "transient", "persistent"],
            "pair_construction": "same init_obs; model truth in H1; transient/persistent truth in H0; xc=CONFLICT_X=48 polluted",
        },
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "isolation": iso,
        "locked_after": "dev",
    }
    (HERE / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
    eval_rows, eval_sum = evaluate("eval", EVAL_PAIRS, EVAL_SEED0)
    result = {
        "isolation": iso,
        "protocol": protocol,
        "dev": dev_sum,
        "eval": eval_sum,
        "timing_s": time.perf_counter() - t0,
        "parameter_provenance": {"costs_and_libraries": "ASSUMED", "metrics": "MEASURED"},
        "executed": ["isolation", "dev_30_tasks", "eval_180_tasks"],
        "not_executed_yet": ["cost_ablation", "generator_ablation", "holdout_confirmation"],
    }
    (HERE / "EVAL_SUMMARY.json").write_text(json.dumps(result, indent=2))
    (HERE / "EVAL_ROWS.json").write_text(json.dumps([{k: r[k] for k in r if k != "traj"} for r in eval_rows]))
    (HERE / "EVAL_TRAJ.json").write_text(
        json.dumps([{"task": r["task"], "policy": r["policy"], "budget": r["budget"], "traj": r["traj"]} for r in eval_rows])
    )
    (HERE / "DEV_ROWS.json").write_text(json.dumps([{k: r[k] for k in r if k != "traj"} for r in dev_rows]))
    print("iso", iso)
    print("dev", dev_sum["n_tasks"], "truths", dev_sum["n_distinct_truths"])
    print("eval", eval_sum["n_tasks"], "truths", eval_sum["n_distinct_truths"], "pairs", eval_sum["n_pairs"])
    for b in ("8", "16"):
        print(f"== budget {b} ==")
        for cause in ("model", "transient", "persistent"):
            print(" ", cause)
            block = eval_sum["by_budget"][b][cause]
            for pol in POLICIES:
                l1 = block[pol]["l1"]
                print(
                    f"    {pol:16s} L1={l1['mean']:.2f}±{l1['se']:.2f} "
                    f"unres={block[pol]['unresolved']['mean']:.2f} "
                    f"exp={block[pol]['n_expand']['mean']:.2f} ret={block[pol]['n_retest']['mean']:.2f}"
                )
            print("    C-D", {k: block["paired_l1_conflict_minus_direct"][k] for k in ("mean", "se", "n_pos", "n_neg", "n_tie")})
            print("    C-R", {k: block["paired_l1_conflict_minus_retest"][k] for k in ("mean", "se", "n_pos", "n_neg", "n_tie")})
            print("    R-D", {k: block["paired_l1_retest_minus_direct"][k] for k in ("mean", "se", "n_pos", "n_neg", "n_tie")})
    print("elapsed", result["timing_s"])


if __name__ == "__main__":
    main()
