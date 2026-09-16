#!/usr/bin/env python3
"""Horizon-correct DP with history-consistent belief and empty-catalog fallback.

Does not modify 20260914T111013Z. Frozen fitter/generator imported from there.
No soft weights. Confirmation uses new seeds, not the inspected holdout.

Acceptance:
1. Filter uses orig (first look) and retest outcome, not cur-only.
2. Empty catalog -> support_failed, covering fallback; not treated as zero risk.
3. Separate policies for horizons 4 and 6 (no slice of H=6 at step 4).
4. Catalog expected MAE of DP_h <= expected MAE of cover and of retest_then_cover
   at the same horizon.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path("/home/grf/.hermes/attachments/outputs/20260914T111013Z")
sys.path.insert(0, str(OLD))
import exact_experiment as e  # noqa: E402

HORIZONS = (4, 6)
CONFIRM_SEED0 = 20_000
CONFIRM_BUNDLES = 6
POLICIES = ("cover", "retest_then_cover", "optimal")


def tape_consistent(w: e.World, orig: Dict[int, int], cur: Dict[int, int], retested) -> bool:
    """First look remains evidence even if cur was overwritten by retest."""
    for x, y0 in orig.items():
        if int(w.observe(int(x))) != int(y0):
            return False
        if int(x) in retested:
            if int(w.retest(int(x))) != int(cur[int(x)]):
                return False
        elif int(cur[int(x)]) != int(y0):
            return False
    return True


def filter_ids(catalog: Sequence[e.World], env: e.TapeEnv) -> FrozenSet[int]:
    return frozenset(
        w.wid for w in catalog if tape_consistent(w, env.orig, env.cur, env.retested)
    )


def run_cover_until(env: e.TapeEnv, horizon: int) -> None:
    while env.cost < horizon:
        x = e.next_cover(env.orig)
        if x is None:
            return
        env.query(int(x))


def run_retest_then_cover(env: e.TapeEnv, horizon: int) -> None:
    while env.cost < horizon:
        singles = e.restoring_singles(env.believed(), e.H0)
        pending = [x for x in singles if int(x) not in env.retested]
        if pending:
            env.retest(int(pending[0]))
            continue
        x = e.next_cover(env.orig)
        if x is None:
            return
        env.query(int(x))


def run_optimal(env: e.TapeEnv, dp: e.ExactDP, horizon: int) -> Dict:
    """History-consistent filter. Empty catalog -> covering fallback."""
    support_failed_at = None
    fallback_steps = 0
    while env.cost < horizon:
        rem = horizon - env.cost
        ids = filter_ids(dp.catalog, env)
        if not ids:
            if support_failed_at is None:
                support_failed_at = env.cost
            x = e.next_cover(env.orig)
            if x is None:
                break
            env.query(int(x))
            fallback_steps += 1
            continue
        _, act = dp.value(ids, frozenset(env.orig), frozenset(env.retested), rem)
        if act is None:
            if support_failed_at is None:
                support_failed_at = env.cost
            x = e.next_cover(env.orig)
            if x is None:
                break
            env.query(int(x))
            fallback_steps += 1
            continue
        a, x = act
        if a == "query":
            env.query(int(x))
        elif a == "retest":
            env.retest(int(x))
        else:
            break
    return {
        "support_failed_at": support_failed_at,
        "fallback_steps": fallback_steps,
        "n_ids_final": len(filter_ids(dp.catalog, env)),
    }


def snapshot(env: e.TapeEnv) -> Dict:
    pred, meta = e.new_deliver(env.believed())
    isol = meta.get("isol") or []
    return {
        "cost": env.cost,
        "mae": e.mae_of(pred, env.world.truth),
        "family": meta.get("family"),
        "isol": isol,
        "wrong_isolate": bool(isol) and env.world.cause == "model",
        "n_query": env.n_query,
        "n_retest": env.n_retest,
        "orig": {str(k): int(v) for k, v in env.orig.items()},
        "cur": {str(k): int(v) for k, v in env.cur.items()},
    }


def run_policy(world: e.World, name: str, horizon: int, dp: Optional[e.ExactDP]) -> Dict:
    t0 = time.perf_counter()
    env = e.TapeEnv(world)
    e.apply_init(env)
    extra = {"support_failed_at": None, "fallback_steps": 0, "n_ids_final": None}
    if name == "cover":
        run_cover_until(env, horizon)
    elif name == "retest_then_cover":
        run_retest_then_cover(env, horizon)
    elif name == "optimal":
        extra = run_optimal(env, dp, horizon)
    else:
        raise ValueError(name)
    snap = snapshot(env)
    orig_ok = all(env.orig[x] == world.observe(x) for x in env.orig)
    cur_changed = env.orig != env.cur
    return {
        "policy": name,
        "horizon": horizon,
        "wid": world.wid,
        "cause": world.cause,
        "bundle": world.bundle,
        "dirty_x": world.dirty_x,
        "init_xs": list(world.init_xs),
        "elapsed_s": time.perf_counter() - t0,
        "snap": snap,
        "orig_untouched": orig_ok,
        "cur_differs_from_orig": cur_changed,
        "truth_hash": hashlib.sha256(world.truth.tobytes()).hexdigest()[:16],
        **extra,
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


def summarize(records: List[Dict], horizon: int) -> Dict:
    rows = [r for r in records if r["horizon"] == horizon]
    by = {}
    for cause in e.CAUSES:
        sub = [r for r in rows if r["cause"] == cause]
        grouped = defaultdict(dict)
        for r in sub:
            grouped[r["wid"]][r["policy"]] = r
        wids = sorted(grouped)
        mae = {p: [grouped[w][p]["snap"]["mae"] for w in wids] for p in POLICIES}
        isol = {
            p: [float(bool(grouped[w][p]["snap"]["isol"])) for w in wids] for p in POLICIES
        }
        wrong = {
            p: [float(grouped[w][p]["snap"]["wrong_isolate"]) for w in wids] for p in POLICIES
        }
        empty = [float(grouped[w]["optimal"]["support_failed_at"] is not None) for w in wids]
        fb = [grouped[w]["optimal"]["fallback_steps"] for w in wids]
        block = {"n": len(wids), "policies": {}}
        for p in POLICIES:
            block["policies"][p] = {
                "mae": mean_ci(mae[p]),
                "isolate_rate": mean_ci(isol[p]),
                "wrong_isolate_rate": mean_ci(wrong[p]),
            }
        block["paired_cover_minus_optimal"] = paired_delta(mae["cover"], mae["optimal"])
        block["paired_retest_minus_optimal"] = paired_delta(
            mae["retest_then_cover"], mae["optimal"]
        )
        block["optimal_support_failed_rate"] = mean_ci(empty)
        block["optimal_fallback_steps"] = mean_ci(fb)
        by[cause] = block
    all_mae = {p: [r["snap"]["mae"] for r in rows if r["policy"] == p] for p in POLICIES}
    return {
        "horizon": horizon,
        "n": len(rows) // len(POLICIES),
        "by_cause": by,
        "expected_mae_uniform": {p: mean_ci(all_mae[p]) for p in POLICIES},
        "acceptance_dp_leq_cover": float(np.mean(all_mae["optimal"])) <= float(np.mean(all_mae["cover"])) + 1e-12,
        "acceptance_dp_leq_retest": float(np.mean(all_mae["optimal"]))
        <= float(np.mean(all_mae["retest_then_cover"])) + 1e-12,
    }


def evaluate_split(worlds: List[e.World], dp: e.ExactDP) -> Tuple[Dict, List[Dict]]:
    recs = []
    for h in HORIZONS:
        for w in worlds:
            for p in POLICIES:
                recs.append(run_policy(w, p, h, dp))
    out = {str(h): summarize(recs, h) for h in HORIZONS}
    compact = []
    for w in worlds:
        row = {
            "wid": w.wid,
            "cause": w.cause,
            "bundle": w.bundle,
            "dirty_x": w.dirty_x,
            "init_xs": list(w.init_xs),
        }
        for h in HORIZONS:
            for p in POLICIES:
                hit = next(r for r in recs if r["wid"] == w.wid and r["horizon"] == h and r["policy"] == p)
                row[f"{p}_h{h}_mae"] = hit["snap"]["mae"]
                row[f"{p}_h{h}_isol"] = hit["snap"]["isol"]
            hit = next(r for r in recs if r["wid"] == w.wid and r["horizon"] == h and r["policy"] == "optimal")
            row[f"opt_h{h}_empty_at"] = hit["support_failed_at"]
            row[f"opt_h{h}_fallback"] = hit["fallback_steps"]
        compact.append(row)
    return out, compact


def history_unit(catalog: List[e.World]) -> Dict:
    found = None
    for w in catalog:
        env = e.TapeEnv(w)
        e.apply_init(env)
        if w.dirty_x not in env.orig:
            continue
        y0 = env.orig[w.dirty_x]
        y2 = env.retest(w.dirty_x)
        if y2 == y0:
            continue
        cur_only = frozenset(
            u.wid
            for u in catalog
            if all(
                (u.retest(x) if x in env.retested else u.observe(x)) == env.cur[x]
                for x in env.orig
            )
        )
        tape = filter_ids(catalog, env)
        found = {
            "wid": w.wid,
            "cause": w.cause,
            "orig": {int(k): int(v) for k, v in env.orig.items()},
            "cur": {int(k): int(v) for k, v in env.cur.items()},
            "y0": int(y0),
            "y2": int(y2),
            "n_cur_only": len(cur_only),
            "n_tape": len(tape),
            "orig_unchanged": env.orig[w.dirty_x] == y0,
            "cur_changed": env.cur[w.dirty_x] == y2 and y2 != y0,
        }
        break
    ok = (
        found is not None
        and found["cur_changed"]
        and found["orig_unchanged"]
        and found["n_tape"] < found["n_cur_only"]
    )
    return {"ok": bool(ok), "example": found}


def isolation_check(catalog: List[e.World], dp: e.ExactDP) -> Dict:
    trials = []
    for policy in ("cover", "retest_then_cover", "optimal", "cheat"):
        for w in catalog[:9]:
            env_a = e.TapeEnv(w)
            e.apply_init(env_a)
            if policy == "cover":
                x = e.next_cover(env_a.orig)
                if x is not None:
                    env_a.query(int(x))
            elif policy == "retest_then_cover":
                run_retest_then_cover(env_a, min(4, 6))
            elif policy == "optimal":
                run_optimal(env_a, dp, 4)
            else:
                u = [int(x) for x in range(e.N) if int(x) not in env_a.orig]
                hidden = np.array([int(env_a.world.truth[x]) for x in u], dtype=np.int64)
                env_a.query(int(u[int(np.argmax(hidden * 31 + np.array(u)))]))
            env_b = env_a.clone_unqueried_rewritten()

            def next_act(env, pol):
                if pol == "cheat":
                    u = [int(x) for x in range(e.N) if int(x) not in env.orig]
                    if not u:
                        return None
                    hidden = np.array([int(env.world.truth[x]) for x in u], dtype=np.int64)
                    return "query", int(u[int(np.argmax(hidden * 31 + np.array(u)))])
                if pol == "cover":
                    return "query", e.next_cover(env.orig)
                if pol == "retest_then_cover":
                    singles = e.restoring_singles(env.believed(), e.H0)
                    pending = [x for x in singles if int(x) not in env.retested]
                    if pending:
                        return "retest", int(pending[0])
                    return "query", e.next_cover(env.orig)
                ids = filter_ids(dp.catalog, env)
                if not ids:
                    return "query", e.next_cover(env.orig)
                _, act = dp.value(ids, frozenset(env.orig), frozenset(env.retested), 2)
                return act

            na = next_act(env_a, policy)
            nb = next_act(env_b, policy)
            trials.append({"policy": policy, "ok": na == nb, "a": na, "b": nb})
    by = defaultdict(list)
    for t in trials:
        by[t["policy"]].append(t)
    summary = {
        p: {"n": len(ts), "n_fail": sum(1 for t in ts if not t["ok"]), "all_ok": all(t["ok"] for t in ts)}
        for p, ts in by.items()
    }
    return {
        "honest_must_pass": all(summary[p]["all_ok"] for p in ("cover", "retest_then_cover", "optimal")),
        "cheat_must_fail": not summary["cheat"]["all_ok"],
        "by_policy": summary,
    }


def code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main():
    t0 = time.perf_counter()
    catalog = e.build_catalog(e.CATALOG_BUNDLES, e.CATALOG_SEED0, "catalog", 0)
    confirm = e.build_catalog(CONFIRM_BUNDLES, CONFIRM_SEED0, "confirm", 2000)
    hist = history_unit(catalog)
    if not hist["ok"]:
        raise SystemExit(f"history unit failed: {hist}")

    dp = e.ExactDP(catalog)
    env0 = e.TapeEnv(catalog[0])
    e.apply_init(env0)
    ids0 = filter_ids(catalog, env0)
    for h in HORIZONS:
        rem = h - env0.cost
        dp.value(ids0, frozenset(env0.orig), frozenset(), rem)

    iso = isolation_check(catalog, dp)
    if not (iso["honest_must_pass"] and iso["cheat_must_fail"]):
        raise SystemExit(f"isolation failed: {iso}")

    print("catalog eval...", flush=True)
    cat_sum, cat_rows = evaluate_split(catalog, dp)
    print("confirm eval...", flush=True)
    conf_sum, conf_rows = evaluate_split(confirm, dp)

    locked = {
        "horizons": list(HORIZONS),
        "fitter": "imported frozen isolate-cap-1 from 20260914T111013Z",
        "belief": "orig first-look + retest outcome; cur is fitter input only",
        "empty_catalog": "support_failed + covering fallback; not zero loss",
        "confirm_seed0": CONFIRM_SEED0,
        "code_sha256": code_hash(),
        "parameter_provenance": {
            "generator": "ASSUMED inherited",
            "catalog_prior": "ASSUMED uniform 18 worlds",
            "metrics": "MEASURED",
        },
    }
    (HERE / "PROTOCOL.json").write_text(json.dumps(locked, indent=2) + "\n")
    result = {
        "protocol": locked,
        "history_unit": hist,
        "isolation": iso,
        "dp": {"n_states": len(dp._memo), "n_calls": dp.n_calls},
        "catalog": cat_sum,
        "confirm_NEW_SEEDS": conf_sum,
        "timing_s": time.perf_counter() - t0,
    }
    (HERE / "EVAL_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "CATALOG_ROWS.json").write_text(json.dumps(cat_rows, indent=2) + "\n")
    (HERE / "CONFIRM_ROWS.json").write_text(json.dumps(conf_rows, indent=2) + "\n")

    print("history_unit", hist["ok"], hist["example"])
    print("iso", iso)
    print("elapsed", round(result["timing_s"], 3), "dp_states", len(dp._memo))
    for split, sm in (("catalog", cat_sum), ("confirm", conf_sum)):
        print("====", split)
        for h in HORIZONS:
            block = sm[str(h)]
            print(
                f"  H={h} E[mae] cover={block['expected_mae_uniform']['cover']['mean']:.4f} "
                f"retest={block['expected_mae_uniform']['retest_then_cover']['mean']:.4f} "
                f"opt={block['expected_mae_uniform']['optimal']['mean']:.4f} "
                f"acc_leq_cover={block['acceptance_dp_leq_cover']} "
                f"acc_leq_retest={block['acceptance_dp_leq_retest']}"
            )
            for cause in e.CAUSES:
                b = block["by_cause"][cause]
                print(
                    f"    {cause:11s}",
                    {p: round(b["policies"][p]["mae"]["mean"], 4) for p in POLICIES},
                    "empty",
                    b["optimal_support_failed_rate"]["mean"],
                    "fb",
                    b["optimal_fallback_steps"]["mean"],
                )


if __name__ == "__main__":
    main()
