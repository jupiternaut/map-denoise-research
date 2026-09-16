#!/usr/bin/env python3
"""Phase 1: seed alignment + real isolation (with a cheating detector).

Does not overwrite the original mismatch experiment. Imports it read-only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path("/home/grf/.hermes/attachments/outputs")
ORIG = ROOT / "mismatch_experiment.py"
HERE = Path(__file__).resolve().parent


def load_orig():
    spec = importlib.util.spec_from_file_location("mismatch_orig", ORIG)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mismatch_orig"] = mod
    spec.loader.exec_module(mod)
    return mod


m = load_orig()


def instrumented_disagreement(state, log: Dict) -> Optional[int]:
    c0, c1 = m.c_sets(state)
    if c0.any():
        live_table = m.H0[c0]
    elif c1.any():
        live_table = m.H1[c1]
    else:
        log["fallback_library_empty"] = log.get("fallback_library_empty", 0) + 1
        x = m.next_random(state)
        log["queries"].append({"kind": "random_fallback", "x": x, "reason": "library_empty"})
        return x
    u = m.unqueried(state)
    if u.size == 0:
        return None
    preds = live_table[:, u]
    nunique = np.array([np.unique(preds[:, j]).size for j in range(u.size)], dtype=np.int32)
    state.n_score += int(preds.size)
    if int(nunique.max()) <= 1:
        log["fallback_zero_disagreement"] = log.get("fallback_zero_disagreement", 0) + 1
        x = m.next_random(state)
        log["queries"].append({"kind": "random_fallback", "x": x, "reason": "zero_disagreement"})
        return x
    x = int(u[nunique == nunique.max()].min())
    log["true_disagreement"] = log.get("true_disagreement", 0) + 1
    log["queries"].append({"kind": "true_disagreement", "x": x, "nunique": int(nunique.max())})
    return x


def run_until_h0_refute(truth, name: str, rng_seed: int, max_n: int = 16) -> Dict:
    state = m.PolicyState(rng=np.random.default_rng(rng_seed))
    log = {"queries": [], "true_disagreement": 0, "fallback_zero_disagreement": 0, "fallback_library_empty": 0}
    n = 0
    seq = []
    for x in m.INIT_X:
        m.apply_query(state, int(x), truth)
        seq.append(("init", int(x)))
        n += 1
    c0, _ = m.c_sets(state)
    if not c0.any():
        return {"seq": seq, "n_until_refute": n, "log": log, "refuted": True}

    while n < max_n:
        if name == "random":
            x = m.next_random(state)
            kind = "random"
        elif name == "disagreement":
            x = instrumented_disagreement(state, log)
            kind = log["queries"][-1]["kind"] if log["queries"] else "unknown"
        else:
            raise ValueError(name)
        if x is None:
            break
        m.apply_query(state, int(x), truth)
        seq.append((kind, int(x)))
        n += 1
        c0, _ = m.c_sets(state)
        if not c0.any():
            return {"seq": seq, "n_until_refute": n, "log": log, "refuted": True}
    return {"seq": seq, "n_until_refute": None, "log": log, "refuted": False}


def seed_alignment(n_type1: int = 60, seed0: int = 10_000) -> Dict:
    """Same per-task seed for random and disagreement (repeat 0)."""
    rows = []
    for serial in range(n_type1):
        task = m.make_task("eval", 1, serial, seed0)
        seed = 7_000 + 0
        r = run_until_h0_refute(task.truth, "random", rng_seed=seed)
        d = run_until_h0_refute(task.truth, "disagreement", rng_seed=seed)
        first_true_d = None
        same_until = 0
        for i, (rd, dd) in enumerate(zip(r["seq"], d["seq"])):
            if dd[0] == "true_disagreement" and first_true_d is None:
                first_true_d = i
                break
            if rd[1] == dd[1]:
                same_until += 1
            else:
                break
        rows.append(
            {
                "task": task.task_id,
                "random_seq": r["seq"],
                "dis_seq": d["seq"],
                "same_prefix": same_until,
                "first_true_disagreement_index": first_true_d,
                "random_n_refute": r["n_until_refute"],
                "dis_n_refute": d["n_until_refute"],
                "true_disagreement_count": d["log"]["true_disagreement"],
                "fallback_zero": d["log"]["fallback_zero_disagreement"],
                "fallback_empty": d["log"]["fallback_library_empty"],
                "identical_until_refute": r["seq"] == d["seq"] and r["refuted"] and d["refuted"],
            }
        )
    n_ident = sum(1 for r in rows if r["identical_until_refute"])
    n_any_true = sum(1 for r in rows if r["true_disagreement_count"] > 0)
    n_fallback_only = sum(
        1
        for r in rows
        if r["true_disagreement_count"] == 0 and (r["fallback_zero"] + r["fallback_empty"]) > 0
    )
    return {
        "n_tasks": n_type1,
        "n_identical_until_h0_refute": n_ident,
        "n_with_true_disagreement_query": n_any_true,
        "n_fallback_only_before_refute": n_fallback_only,
        "mean_true_disagreement": float(np.mean([r["true_disagreement_count"] for r in rows])),
        "mean_fallback_zero": float(np.mean([r["fallback_zero"] for r in rows])),
        "rows": rows,
    }


class HiddenWorld:
    """Environment that policies query; unqueried values stay private."""

    def __init__(self, truth: np.ndarray):
        self._truth = np.array(truth, dtype=np.int32)

    def label(self, x: int) -> int:
        return int(self._truth[int(x)])

    def mutate_unqueried(self, queried: Dict[int, int], delta: int = 13) -> "HiddenWorld":
        t = self._truth.copy()
        for i in range(len(t)):
            if int(i) not in queried:
                t[i] = int(1000 + i)
        return HiddenWorld(t)


def honest_next(policy: str, queried: Dict[int, int], rng_seed: int, cover_index: int) -> Optional[int]:
    state = m.PolicyState(queried=dict(queried), cover_index=cover_index, rng=np.random.default_rng(rng_seed))
    return m.SELECTORS[policy](state)


def cheating_next(queried: Dict[int, int], world: HiddenWorld) -> Optional[int]:
    """Illegal: uses unqueried labels vs current delivery residual."""
    state = m.PolicyState(queried=dict(queried))
    pred = m.deliver(state)
    u = [int(x) for x in m.DOMAIN if int(x) not in queried]
    if not u:
        return None
    scores = [abs(world.label(x) - int(pred[int(x)])) for x in u]
    best = max(scores)
    return int(min(x for x, s in zip(u, scores) if s == best))


def isolation_suite() -> Dict:
    trials = []
    for policy in ("fixed", "random", "disagreement"):
        for typ in (0, 1, 2):
            for serial in range(4):
                task = m.make_task("iso", typ, serial, seed0=50_000)
                world_a = HiddenWorld(task.truth)
                rng_seed = 12345 + serial
                state = m.PolicyState(rng=np.random.default_rng(rng_seed))
                for x in m.INIT_X:
                    state.queried[int(x)] = world_a.label(int(x))
                while len(state.queried) < 5:
                    x = m.SELECTORS[policy](state)
                    if x is None:
                        break
                    state.queried[int(x)] = world_a.label(int(x))
                cover_index = state.cover_index
                queried = dict(state.queried)
                world_b = world_a.mutate_unqueried(queried)

                honest_a = honest_next(policy, queried, 9001, cover_index)
                honest_b = honest_next(policy, queried, 9001, cover_index)
                cheat_a = cheating_next(queried, world_a)
                cheat_b = cheating_next(queried, world_b)
                trials.append(
                    {
                        "policy": policy,
                        "typ": typ,
                        "serial": serial,
                        "honest_a": honest_a,
                        "honest_b": honest_b,
                        "honest_ok": honest_a == honest_b,
                        "cheat_a": cheat_a,
                        "cheat_b": cheat_b,
                        "cheat_caught": cheat_a != cheat_b,
                    }
                )
    return {
        "n": len(trials),
        "honest_n_fail": sum(1 for t in trials if not t["honest_ok"]),
        "honest_all_ok": all(t["honest_ok"] for t in trials),
        "cheat_n_caught": sum(1 for t in trials if t["cheat_caught"]),
        "cheat_all_caught": all(t["cheat_caught"] for t in trials),
        "trials": trials,
    }


def main() -> None:
    t0 = time.perf_counter()
    orig_hash = hashlib.sha256(ORIG.read_bytes()).hexdigest()
    align = seed_alignment()
    iso = isolation_suite()
    out = {
        "original_hash": orig_hash,
        "seed_alignment": {k: v for k, v in align.items() if k != "rows"},
        "seed_alignment_examples": align["rows"][:5],
        "isolation": {k: v for k, v in iso.items() if k != "trials"},
        "elapsed_s": time.perf_counter() - t0,
    }
    (HERE / "PHASE1.json").write_text(json.dumps(out, indent=2))
    (HERE / "PHASE1_ALIGNMENT_ROWS.json").write_text(json.dumps(align["rows"], indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
