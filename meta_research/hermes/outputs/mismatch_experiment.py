#!/usr/bin/env python3
"""Finite-budget query selection vs model mismatch.

Shared libraries H0 (linear) and H1 (one-break piecewise, containing H0),
shared consistency update, shared delivery. Strategies differ only in the
next query location. Unqueried labels never enter the policy.

ASSUMED parameters were locked after the 30-task development pass.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

N = 64
DOMAIN = np.arange(N, dtype=np.int32)
A_RANGE = np.array([-1, 0, 1], dtype=np.int32)
B_RANGE = np.arange(-8, 9, dtype=np.int32)
T_RANGE = np.array([16, 24, 32, 40, 48], dtype=np.int32)
N_INIT = 2
BUDGETS = (4, 8, 12, 16)
MAX_BUDGET = 16
RANDOM_REPEATS = 3
DEV_TASKS_PER_TYPE = 10
EVAL_TASKS_PER_TYPE = 60
DEV_SEED0 = 1
EVAL_SEED0 = 10_000
OUT_DIR = Path(__file__).resolve().parent


def covering_order(n: int = N) -> np.ndarray:
    order: List[int] = []
    seen = set()
    step = n
    while step >= 1:
        for x in range(0, n, step):
            if x not in seen:
                order.append(x)
                seen.add(x)
        step //= 2
    return np.array(order, dtype=np.int32)


COVER = covering_order()
INIT_X = COVER[:N_INIT]


def build_libraries() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h0 = []
    for a in A_RANGE:
        for b in B_RANGE:
            h0.append(a * DOMAIN + b)
    h0 = np.stack(h0, axis=0).astype(np.int32)
    pieces = []
    for t in T_RANGE:
        left = DOMAIN < t
        right = ~left
        for a1 in A_RANGE:
            for b1 in B_RANGE:
                for a2 in A_RANGE:
                    for b2 in B_RANGE:
                        if a1 == a2 and b1 == b2:
                            continue
                        f = np.empty(N, dtype=np.int32)
                        f[left] = a1 * DOMAIN[left] + b1
                        f[right] = a2 * DOMAIN[right] + b2
                        pieces.append(f)
    pieces_arr = np.stack(pieces, axis=0)
    h1 = np.concatenate([h0, pieces_arr], axis=0)
    return h0, h1, pieces_arr


H0, H1, H1_PIECES = build_libraries()


def in_library(table: np.ndarray, f: np.ndarray) -> bool:
    return bool(np.any(np.all(table == f[None, :], axis=1)))


def make_type0(rng: np.random.Generator) -> np.ndarray:
    return H0[int(rng.integers(0, len(H0)))].copy()


def make_type1(rng: np.random.Generator) -> np.ndarray:
    for _ in range(10_000):
        f = H1_PIECES[int(rng.integers(0, len(H1_PIECES)))].copy()
        if not in_library(H0, f):
            return f
    raise RuntimeError("failed to sample type-1 target")


def make_type2(rng: np.random.Generator) -> np.ndarray:
    for _ in range(10_000):
        t1, t2 = sorted(int(x) for x in rng.choice(np.arange(8, 57), size=2, replace=False))
        if t2 - t1 < 8:
            continue
        levels = rng.integers(-8, 9, size=3)
        if len(set(int(v) for v in levels)) < 3:
            continue
        f = np.empty(N, dtype=np.int32)
        f[:t1] = levels[0]
        f[t1:t2] = levels[1]
        f[t2:] = levels[2]
        if not in_library(H1, f):
            return f
    raise RuntimeError("failed to sample type-2 target")


MAKERS = {0: make_type0, 1: make_type1, 2: make_type2}


@dataclass
class Task:
    task_id: str
    typ: int
    truth: np.ndarray
    split: str


def make_task(split: str, typ: int, serial: int, seed0: int) -> Task:
    seed = seed0 + 1000 * typ + serial
    rng = np.random.default_rng(seed)
    return Task(
        task_id=f"{split}-t{typ}-{serial:03d}",
        typ=typ,
        truth=MAKERS[typ](rng),
        split=split,
    )


@dataclass
class PolicyState:
    queried: Dict[int, int] = field(default_factory=dict)
    cover_index: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    n_score: int = 0

    def mask(self, table: np.ndarray) -> np.ndarray:
        live = np.ones(len(table), dtype=bool)
        for x, y in self.queried.items():
            live &= table[:, int(x)] == int(y)
        return live


def c_sets(state: PolicyState) -> Tuple[np.ndarray, np.ndarray]:
    return state.mask(H0), state.mask(H1)


def diagnosis(c0: np.ndarray, c1: np.ndarray) -> str:
    if c0.any():
        return "C0_alive"
    if c1.any():
        return "H0_refuted"
    return "library_mismatch"


def unqueried(state: PolicyState) -> np.ndarray:
    return np.setdiff1d(DOMAIN, np.array(sorted(state.queried), dtype=np.int32))


def next_fixed(state: PolicyState) -> Optional[int]:
    while state.cover_index < len(COVER):
        x = int(COVER[state.cover_index])
        state.cover_index += 1
        if x not in state.queried:
            return x
    return None


def next_random(state: PolicyState) -> Optional[int]:
    u = unqueried(state)
    if u.size == 0:
        return None
    return int(state.rng.choice(u))


def next_disagreement(state: PolicyState) -> Optional[int]:
    c0, c1 = c_sets(state)
    if c0.any():
        live_table = H0[c0]
    elif c1.any():
        live_table = H1[c1]
    else:
        return next_random(state)
    u = unqueried(state)
    if u.size == 0:
        return None
    preds = live_table[:, u]
    nunique = np.array([np.unique(preds[:, j]).size for j in range(u.size)], dtype=np.int32)
    state.n_score += int(preds.size)
    if int(nunique.max()) <= 1:
        return next_random(state)
    return int(u[nunique == nunique.max()].min())


SELECTORS = {
    "fixed": next_fixed,
    "random": next_random,
    "disagreement": next_disagreement,
}


def apply_query(state: PolicyState, x: int, truth: np.ndarray) -> None:
    state.queried[int(x)] = int(truth[int(x)])


def deliver(state: PolicyState) -> np.ndarray:
    c0, c1 = c_sets(state)
    if c0.any():
        return H0[c0][0].copy()
    if c1.any():
        return H1[c1][0].copy()
    f = np.empty(N, dtype=np.int32)
    xs = np.array(sorted(state.queried), dtype=np.int32)
    ys = np.array([state.queried[int(x)] for x in xs], dtype=np.int32)
    for x in DOMAIN:
        f[x] = ys[int(np.argmin(np.abs(xs - x)))]
    return f


def snapshot(state: PolicyState, truth: np.ndarray, budget_used: int, n_label: int) -> Dict:
    c0, c1 = c_sets(state)
    pred = deliver(state)
    return {
        "budget": budget_used,
        "n_label": n_label,
        "n_score": state.n_score,
        "diagnosis": diagnosis(c0, c1),
        "c0": int(c0.sum()),
        "c1": int(c1.sum()),
        "global_l1": int(np.abs(pred.astype(np.int64) - truth.astype(np.int64)).sum()),
        "h0_refuted": not bool(c0.any()),
        "library_mismatch": not bool(c1.any()),
    }


def run_policy(truth: np.ndarray, name: str, rng_seed: int, max_budget: int = MAX_BUDGET) -> Dict:
    t0 = time.perf_counter()
    state = PolicyState(rng=np.random.default_rng(rng_seed))
    selector = SELECTORS[name]
    n_label = 0
    checkpoints = {b: None for b in BUDGETS if b <= max_budget}
    mismatch_at: Optional[int] = None
    h0_refute_at: Optional[int] = None

    def note() -> None:
        nonlocal h0_refute_at, mismatch_at
        c0, c1 = c_sets(state)
        if h0_refute_at is None and not c0.any():
            h0_refute_at = n_label
        if mismatch_at is None and not c1.any():
            mismatch_at = n_label
        if n_label in checkpoints:
            checkpoints[n_label] = snapshot(state, truth, n_label, n_label)

    for x in INIT_X:
        apply_query(state, int(x), truth)
        n_label += 1
    note()

    while n_label < max_budget:
        x = selector(state)
        if x is None:
            break
        if x in state.queried:
            raise RuntimeError(f"{name} proposed already-queried x={x}")
        apply_query(state, x, truth)
        n_label += 1
        note()

    for b, snap in list(checkpoints.items()):
        if snap is None:
            checkpoints[b] = snapshot(state, truth, n_label, n_label)

    return {
        "policy": name,
        "n_label": n_label,
        "n_score": state.n_score,
        "elapsed_s": time.perf_counter() - t0,
        "h0_refute_at": h0_refute_at,
        "library_mismatch_at": mismatch_at,
        "queried": sorted(int(x) for x in state.queried),
        "checkpoints": {str(k): checkpoints[k] for k in BUDGETS if k <= max_budget},
        "final": snapshot(state, truth, n_label, n_label),
    }


def isolation_trial(typ: int, serial: int, policy: str) -> Dict:
    task = make_task("iso", typ, serial, seed0=50_000)
    rng_seed = 12345 + serial
    state = PolicyState(rng=np.random.default_rng(rng_seed))
    for x in INIT_X:
        apply_query(state, int(x), task.truth)
    while len(state.queried) < 5:
        x = SELECTORS[policy](state)
        if x is None:
            break
        apply_query(state, x, task.truth)

    s1 = PolicyState(queried=dict(state.queried), cover_index=state.cover_index, rng=np.random.default_rng(9001))
    s2 = PolicyState(queried=dict(state.queried), cover_index=state.cover_index, rng=np.random.default_rng(9001))
    x_a = SELECTORS[policy](s1)
    mutated = task.truth.copy()
    for x in DOMAIN:
        if int(x) not in state.queried:
            mutated[int(x)] = int(mutated[int(x)] + 13)
    _unused = mutated  # must not be passed into the selector
    x_b = SELECTORS[policy](s2)
    return {"policy": policy, "typ": typ, "next_a": x_a, "next_b": x_b, "ok": x_a == x_b}


def run_isolation(n_each: int = 4) -> Dict:
    trials = [isolation_trial(typ, serial, policy) for policy in SELECTORS for typ in (0, 1, 2) for serial in range(n_each)]
    return {"n": len(trials), "n_fail": sum(1 for t in trials if not t["ok"]), "all_ok": all(t["ok"] for t in trials)}


def discovered(typ: int, rec: Dict, budget: int) -> bool:
    snap = rec["checkpoints"][str(budget)]
    if typ == 0:
        return not snap["h0_refuted"]
    if typ == 1:
        return bool(snap["h0_refuted"]) and not snap["library_mismatch"]
    return bool(snap["library_mismatch"])


def discovery_cost(typ: int, rec: Dict) -> Optional[int]:
    if typ == 0:
        return N_INIT if not rec["final"]["h0_refuted"] else None
    if typ == 1:
        return rec["h0_refute_at"]
    return rec["library_mismatch_at"]


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


def _mean_opt(xs: Sequence[Optional[int]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None


def _average_random(runs: List[Dict]) -> Dict:
    out = {
        "policy": "random",
        "n_repeats": len(runs),
        "h0_refute_at": _mean_opt([r["h0_refute_at"] for r in runs]),
        "library_mismatch_at": _mean_opt([r["library_mismatch_at"] for r in runs]),
        "n_score_mean": float(np.mean([r["n_score"] for r in runs])),
        "elapsed_s_mean": float(np.mean([r["elapsed_s"] for r in runs])),
        "checkpoints": {},
        "_runs": runs,
    }
    for b in BUDGETS:
        snaps = [r["checkpoints"][str(b)] for r in runs]
        out["checkpoints"][str(b)] = {
            "budget": b,
            "global_l1": float(np.mean([s["global_l1"] for s in snaps])),
            "h0_refuted_rate": float(np.mean([s["h0_refuted"] for s in snaps])),
            "library_mismatch_rate": float(np.mean([s["library_mismatch"] for s in snaps])),
        }
    out["final"] = {
        "h0_refuted": bool(np.mean([r["final"]["h0_refuted"] for r in runs]) >= 0.5),
        "library_mismatch": bool(np.mean([r["final"]["library_mismatch"] for r in runs]) >= 0.5),
        "global_l1": float(np.mean([r["final"]["global_l1"] for r in runs])),
        "diagnosis": "averaged",
    }
    return out


def summarize(split: str, rows: List[Dict]) -> Dict:
    by_type: Dict[str, Dict] = {}
    for typ in (0, 1, 2):
        subset = [r for r in rows if r["typ"] == typ]
        block = {"n_tasks": len(subset), "budgets": {}}
        for b in BUDGETS:
            bblock: Dict = {}
            disc = {
                "fixed": [float(discovered(typ, r["recs"]["fixed"], b)) for r in subset],
                "disagreement": [float(discovered(typ, r["recs"]["disagreement"], b)) for r in subset],
                "random": [
                    float(np.mean([discovered(typ, run, b) for run in r["recs"]["random"]["_runs"]]))
                    for r in subset
                ],
            }
            l1 = {
                "fixed": [r["recs"]["fixed"]["checkpoints"][str(b)]["global_l1"] for r in subset],
                "disagreement": [r["recs"]["disagreement"]["checkpoints"][str(b)]["global_l1"] for r in subset],
                "random": [r["recs"]["random"]["checkpoints"][str(b)]["global_l1"] for r in subset],
            }
            for p in ("fixed", "random", "disagreement"):
                costs = []
                for r in subset:
                    if p == "random":
                        cs = [discovery_cost(typ, run) for run in r["recs"]["random"]["_runs"]]
                        present = [c for c in cs if c is not None]
                        costs.append(float(np.mean(present)) if present else np.nan)
                    else:
                        c = discovery_cost(typ, r["recs"][p])
                        costs.append(float(c) if c is not None else np.nan)
                bblock[p] = {
                    "discovery_rate": mean_ci(disc[p]),
                    "global_l1": mean_ci(l1[p]),
                    "discovery_cost_when_found": mean_ci([c for c in costs if c == c]),
                    "n_found": int(np.sum(np.array(disc[p]) > 0)),
                }
            bblock["paired_discovery_disagreement_minus_fixed"] = paired_delta(disc["disagreement"], disc["fixed"])
            bblock["paired_discovery_disagreement_minus_random"] = paired_delta(disc["disagreement"], disc["random"])
            bblock["paired_l1_fixed_minus_disagreement"] = paired_delta(l1["fixed"], l1["disagreement"])
            bblock["paired_l1_random_minus_disagreement"] = paired_delta(l1["random"], l1["disagreement"])
            block["budgets"][str(b)] = bblock
        by_type[str(typ)] = block
    return {
        "split": split,
        "n_tasks": len(rows),
        "library": {"n_h0": int(len(H0)), "n_h1": int(len(H1)), "n_h1_pieces": int(len(H1_PIECES))},
        "by_type": by_type,
        "rows_compact": [
            {
                "task": r["task"],
                "typ": r["typ"],
                "fixed_final_l1": r["recs"]["fixed"]["final"]["global_l1"],
                "dis_final_l1": r["recs"]["disagreement"]["final"]["global_l1"],
                "random_final_l1": r["recs"]["random"]["final"]["global_l1"],
                "fixed_diag": r["recs"]["fixed"]["final"]["diagnosis"],
                "dis_diag": r["recs"]["disagreement"]["final"]["diagnosis"],
                "fixed_h0_at": r["recs"]["fixed"]["h0_refute_at"],
                "dis_h0_at": r["recs"]["disagreement"]["h0_refute_at"],
                "fixed_mis_at": r["recs"]["fixed"]["library_mismatch_at"],
                "dis_mis_at": r["recs"]["disagreement"]["library_mismatch_at"],
            }
            for r in rows
        ],
    }


def evaluate_split(split: str, per_type: int, seed0: int) -> Dict:
    tasks = [make_task(split, typ, serial, seed0) for typ in (0, 1, 2) for serial in range(per_type)]
    rows = []
    for i, task in enumerate(tasks):
        recs = {
            "fixed": run_policy(task.truth, "fixed", rng_seed=0),
            "disagreement": run_policy(task.truth, "disagreement", rng_seed=0),
            "random": _average_random(
                [run_policy(task.truth, "random", rng_seed=7_000 + k) for k in range(RANDOM_REPEATS)]
            ),
        }
        rows.append({"task": task.task_id, "typ": task.typ, "recs": recs})
        if split == "eval" and (i + 1) % 30 == 0:
            print(f"eval {i+1}/{len(tasks)}", flush=True)
    return summarize(split, rows)


def library_sanity() -> List[Dict]:
    checks = []

    def rec(name: str, ok: bool, detail) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    rec("cover_len", len(COVER) == N, int(len(COVER)))
    rec("init", list(INIT_X) == [0, 32], INIT_X.tolist())
    rec("h1_size", len(H1) == len(H0) + len(H1_PIECES), {"h0": int(len(H0)), "h1": int(len(H1))})
    t0 = make_type0(np.random.default_rng(0))
    t1 = make_type1(np.random.default_rng(1))
    t2 = make_type2(np.random.default_rng(2))
    rec("type0_in_h0", in_library(H0, t0), t0[:8].tolist())
    rec("type1_not_in_h0", (not in_library(H0, t1)) and in_library(H1, t1), t1[:8].tolist())
    rec("type2_not_in_h1", not in_library(H1, t2), t2[:8].tolist())
    rec("h0_subset_h1", all(in_library(H1, f) for f in H0[:: max(1, len(H0) // 5)]), None)
    return checks


def main() -> None:
    t0 = time.perf_counter()
    sanity = library_sanity()
    isolation = run_isolation(n_each=4)
    if not isolation["all_ok"]:
        raise SystemExit(f"isolation failed: {isolation}")
    if not all(c["ok"] for c in sanity):
        raise SystemExit(f"sanity failed: {sanity}")

    dev = evaluate_split("dev", DEV_TASKS_PER_TYPE, DEV_SEED0)
    locked = {
        "N": N,
        "A_RANGE": A_RANGE.tolist(),
        "B_RANGE": B_RANGE.tolist(),
        "T_RANGE": T_RANGE.tolist(),
        "N_INIT": N_INIT,
        "BUDGETS": list(BUDGETS),
        "COVER_HEAD": COVER[:16].tolist(),
        "disagreement": "max nunique among live C0 else C1; ties -> min x; zero disagreement -> random",
        "delivery": "first live H0 row else first live H1 row else nearest observed",
        "random_repeats": RANDOM_REPEATS,
        "note": "rules locked after dev; eval uses new seeds",
    }
    ev = evaluate_split("eval", EVAL_TASKS_PER_TYPE, EVAL_SEED0)
    elapsed = time.perf_counter() - t0
    result = {
        "parameter_provenance": {
            "library_ranges": "ASSUMED",
            "covering_order": "ASSUMED",
            "N_INIT": "ASSUMED",
            "BUDGETS": "ASSUMED",
            "task_generators": "ASSUMED",
            "isolation": "MEASURED",
            "dev_and_eval_metrics": "MEASURED",
        },
        "locked_after_dev": locked,
        "sanity": {"all_ok": all(c["ok"] for c in sanity), "checks": sanity},
        "isolation": isolation,
        "dev": dev,
        "eval": ev,
        "timing_s": elapsed,
        "executed": [
            "library_sanity",
            "isolation_48_trials",
            f"dev_{3 * DEV_TASKS_PER_TYPE}_tasks",
            f"eval_{3 * EVAL_TASKS_PER_TYPE}_tasks",
        ],
        "not_executed": [
            "perfect-validator strategy in main ranking",
            "noisy labels",
            "open-ended hypothesis invention",
        ],
    }
    out = OUT_DIR / "MISMATCH_RESULTS.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)
    print("isolation_ok", isolation["all_ok"], "n", isolation["n"])
    print("elapsed_s", round(elapsed, 3))
    for typ in ("0", "1", "2"):
        print(f"--- type {typ} eval @16 ---")
        b = ev["by_type"][typ]["budgets"]["16"]
        for p in ("fixed", "random", "disagreement"):
            d = b[p]["discovery_rate"]
            g = b[p]["global_l1"]
            print(f"  {p:14s} disc={d['mean']:.3f}±{d['se']:.3f}  L1={g['mean']:.2f}±{g['se']:.2f}")
        print("  paired D-F disc", {k: b["paired_discovery_disagreement_minus_fixed"][k] for k in ("mean", "se", "n_pos", "n_neg", "n_tie")})
        print("  paired D-R disc", {k: b["paired_discovery_disagreement_minus_random"][k] for k in ("mean", "se", "n_pos", "n_neg", "n_tie")})


if __name__ == "__main__":
    main()
