#!/usr/bin/env python3
"""Phase 1 only: seed alignment on the frozen mismatch experiment, and a
real dual-environment isolation test that can catch a cheating policy.

Does not modify the original mismatch files.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORIG = Path("/home/grf/.hermes/attachments/outputs")
sys.path.insert(0, str(ORIG))
import mismatch_experiment as m  # noqa: E402


class Env:
    """Actual environment object. Policies receive this, not a raw array."""

    def __init__(self, truth: np.ndarray):
        self._truth = np.array(truth, dtype=np.int32)

    def label(self, x: int) -> int:
        return int(self._truth[int(x)])

    def clone_with_unqueried_rewritten(self, queried) -> "Env":
        """Rewrite unqueried labels so a peeking policy can change its argmax.

        A constant shift leaves argmax(hidden) unchanged; this replacement
        assigns each unqueried x a unique value 17*x+3.
        """
        t = self._truth.copy()
        for x in range(m.N):
            if int(x) not in queried:
                t[int(x)] = int(17 * int(x) + 3)
        return Env(t)


def next_disagreement_instrumented(state: m.PolicyState):
    c0, c1 = m.c_sets(state)
    if c0.any():
        live_table = m.H0[c0]
        stage = "C0"
    elif c1.any():
        live_table = m.H1[c1]
        stage = "C1"
    else:
        return m.next_random(state), "random_fallback", "library_empty"
    u = m.unqueried(state)
    if u.size == 0:
        return None, "none", stage
    preds = live_table[:, u]
    nunique = np.array(
        [np.unique(preds[:, j]).size for j in range(u.size)], dtype=np.int32
    )
    state.n_score += int(preds.size)
    if int(nunique.max()) <= 1:
        return m.next_random(state), "random_fallback", stage
    return int(u[nunique == nunique.max()].min()), "disagreement", stage


def run_aligned(truth, rng_seed: int, max_budget: int = m.MAX_BUDGET):
    st_r = m.PolicyState(rng=np.random.default_rng(rng_seed))
    st_d = m.PolicyState(rng=np.random.default_rng(rng_seed))
    for x in m.INIT_X:
        m.apply_query(st_r, int(x), truth)
        m.apply_query(st_d, int(x), truth)

    n_label = len(m.INIT_X)
    h0_refute_at = None
    same_until_refute = True
    n_dis_true = 0
    n_rand_fb = 0
    queries_before_refute_r = []
    queries_before_refute_d = []

    def c0_empty(st):
        return not bool(m.c_sets(st)[0].any())

    if c0_empty(st_r):
        h0_refute_at = n_label

    while n_label < max_budget:
        if h0_refute_at is None and c0_empty(st_r):
            h0_refute_at = n_label
        xr = m.next_random(st_r)
        xd, kind, stage = next_disagreement_instrumented(st_d)
        if xr is None or xd is None:
            break
        if kind == "disagreement":
            n_dis_true += 1
        else:
            n_rand_fb += 1
        if h0_refute_at is None:
            queries_before_refute_r.append(int(xr))
            queries_before_refute_d.append(int(xd))
            if int(xr) != int(xd):
                same_until_refute = False
        m.apply_query(st_r, int(xr), truth)
        m.apply_query(st_d, int(xd), truth)
        n_label += 1

    if h0_refute_at is None and c0_empty(st_r):
        h0_refute_at = n_label
    return {
        "same_queries_until_h0_refute": bool(same_until_refute)
        if h0_refute_at is not None
        else True,
        "h0_refute_at": h0_refute_at,
        "n_true_disagreement": n_dis_true,
        "n_random_fallback": n_rand_fb,
        "queries_before_refute_random": queries_before_refute_r,
        "queries_before_refute_dis": queries_before_refute_d,
        "h0_never_refuted": h0_refute_at is None,
    }


def honest_next(queried, cover_index: int, rng_seed: int, policy: str):
    state = m.PolicyState(
        queried=dict(queried),
        cover_index=cover_index,
        rng=np.random.default_rng(rng_seed),
    )
    return m.SELECTORS[policy](state)


def cheat_next(env: Env, queried):
    """Deliberately peeks at hidden labels among unqueried x.

    Mixes hidden values with coordinates so a rewrite of unqueried
    answers changes the argmax. A constant shift of all hidden labels
    would not (that was the first detector bug).
    """
    u = [int(x) for x in range(m.N) if int(x) not in queried]
    hidden = np.array([env.label(x) for x in u], dtype=np.int64)
    score = hidden * 31 + np.array(u, dtype=np.int64)
    return int(u[int(np.argmax(score))])


def dual_env_isolation():
    trials = []
    for policy in ("fixed", "random", "disagreement", "cheat"):
        for typ in (0, 1, 2):
            for serial in range(4):
                task = m.make_task("iso", typ, serial, seed0=50_000)
                rng_seed = 12345 + serial
                env_a = Env(task.truth)
                queried = {}
                cover_index = 0
                for x in m.INIT_X:
                    queried[int(x)] = env_a.label(int(x))
                    while cover_index < len(m.COVER) and int(m.COVER[cover_index]) in queried:
                        cover_index += 1
                while len(queried) < 5:
                    if policy == "cheat":
                        x = cheat_next(env_a, queried)
                    else:
                        x = honest_next(queried, cover_index, rng_seed, policy)
                    if x is None:
                        break
                    queried[int(x)] = env_a.label(int(x))
                    while cover_index < len(m.COVER) and int(m.COVER[cover_index]) in queried:
                        cover_index += 1

                env_b = env_a.clone_with_unqueried_rewritten(queried)
                if policy == "cheat":
                    xa = cheat_next(env_a, queried)
                    xb = cheat_next(env_b, queried)
                else:
                    xa = honest_next(queried, cover_index, 9001, policy)
                    xb = honest_next(queried, cover_index, 9001, policy)
                trials.append(
                    {
                        "policy": policy,
                        "typ": typ,
                        "serial": serial,
                        "next_a": xa,
                        "next_b": xb,
                        "ok": xa == xb,
                        "envs_are_distinct": env_a is not env_b
                        and not np.array_equal(env_a._truth, env_b._truth),
                    }
                )
    by = {}
    for t in trials:
        by.setdefault(t["policy"], []).append(t)
    summary = {}
    for p, ts in by.items():
        summary[p] = {
            "n": len(ts),
            "n_fail": sum(1 for t in ts if not t["ok"]),
            "all_ok": all(t["ok"] for t in ts),
            "envs_distinct": all(t["envs_are_distinct"] for t in ts),
        }
    return {
        "honest_must_pass": all(
            summary[p]["all_ok"] for p in ("fixed", "random", "disagreement")
        ),
        "cheat_must_fail": not summary["cheat"]["all_ok"],
        "by_policy": summary,
        "trials": trials,
    }


def alignment_pass():
    rows = []
    for typ in (0, 1, 2):
        for serial in range(20):
            task = m.make_task("eval", typ, serial, seed0=m.EVAL_SEED0)
            rec = run_aligned(task.truth, rng_seed=7_000)
            rec.update({"task": task.task_id, "typ": typ})
            rows.append(rec)
    by = {}
    for typ in (0, 1, 2):
        sub = [r for r in rows if r["typ"] == typ]
        by[str(typ)] = {
            "n": len(sub),
            "frac_same_until_h0_refute": float(
                np.mean([r["same_queries_until_h0_refute"] for r in sub])
            ),
            "mean_true_disagreement": float(
                np.mean([r["n_true_disagreement"] for r in sub])
            ),
            "mean_random_fallback": float(np.mean([r["n_random_fallback"] for r in sub])),
            "n_h0_never_refuted": int(sum(r["h0_never_refuted"] for r in sub)),
        }
    return {"by_type": by, "n_rows": len(rows), "rows": rows}


def main():
    t0 = time.perf_counter()
    iso = dual_env_isolation()
    aln = alignment_pass()
    out = {
        "isolation": {
            "honest_must_pass": iso["honest_must_pass"],
            "cheat_must_fail": iso["cheat_must_fail"],
            "by_policy": iso["by_policy"],
            "note": (
                "Two Env objects; mutated copy is a second environment. "
                "Honest selectors never receive hidden labels. "
                "Cheat peeks via env.label on unqueried x."
            ),
        },
        "alignment": aln["by_type"],
        "elapsed_s": time.perf_counter() - t0,
        "pass": bool(iso["honest_must_pass"] and iso["cheat_must_fail"]),
    }
    (HERE / "PHASE1.json").write_text(json.dumps(out, indent=2) + "\n")
    (HERE / "PHASE1_ALIGNMENT_ROWS.json").write_text(json.dumps(aln["rows"], indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("isolation", "alignment", "elapsed_s", "pass")}, indent=2))
    if not out["pass"]:
        raise SystemExit("phase1 isolation contract failed")


if __name__ == "__main__":
    main()
