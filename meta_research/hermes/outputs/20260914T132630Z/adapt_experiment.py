#!/usr/bin/env python3
"""Enumerated-catalog open-loop vs adaptive DP.

Frozen: 20260914T124618Z (not modified). Fitter imported from
20260914T111013Z. No soft weights. No second generator.

Question: given a support-complete catalog from the public generator,
does adapting the next experiment to new outcomes beat the best
open-loop plan that may use the catalog and the init tape but cannot
change after that?

Policies share fitter, actions, horizons, and delivery:
  cover              — fixed covering order
  retest_then_cover  — restoring-single retest, then cover (partially adaptive)
  openloop           — optimal sequence chosen at init; not revised
  adaptive           — ExactDP, replans after each outcome

ASSUMED: uniform prior over enumerated worlds; action cost 1.
MEASURED: catalog size, MAE, support, wall time.
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
HZ = Path("/home/grf/.hermes/attachments/outputs/20260914T124618Z")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HZ))
sys.path.insert(0, str(OLD))
import exact_experiment as e  # noqa: E402
import horizon_experiment as h  # noqa: E402
import enum_probe as ep  # noqa: E402

HORIZONS = (4, 6)
POLICIES = ("cover", "retest_then_cover", "openloop", "adaptive")
CONFIRM_SEED0 = 40_000
CONFIRM_BUNDLES = 30
CONFIRM_WID0 = 100_000

Action = Tuple[str, int]


class FastDP(e.ExactDP):
    def __init__(self, catalog: List[e.World]):
        super().__init__(catalog)
        self.by_id = {w.wid: w for w in catalog}

    def remaining(self, ids: FrozenSet[int]) -> List[e.World]:
        return [self.by_id[i] for i in ids]


class CatalogIndex:
    def __init__(self, worlds: Sequence[e.World]):
        self.worlds = list(worlds)
        self.by_id = {w.wid: w for w in worlds}
        obs: Dict[Tuple[int, int], set] = {}
        ret: Dict[Tuple[int, int], set] = {}
        n = e.N
        wids = [w.wid for w in worlds]
        max_id = max(wids) if wids else 0
        self.obs_arr = np.zeros((max_id + 1, n), dtype=np.int32)
        self.ret_arr = np.zeros((max_id + 1, n), dtype=np.int32)
        self.truth_arr = np.zeros((max_id + 1, n), dtype=np.int32)
        for w in worlds:
            for x in range(n):
                yo = int(w.observe(x))
                yr = int(w.retest(x))
                self.obs_arr[w.wid, x] = yo
                self.ret_arr[w.wid, x] = yr
                obs.setdefault((x, yo), set()).add(w.wid)
                ret.setdefault((x, yr), set()).add(w.wid)
            self.truth_arr[w.wid] = w.truth
        self.obs_post = {k: frozenset(v) for k, v in obs.items()}
        self.ret_post = {k: frozenset(v) for k, v in ret.items()}
        self.all_ids = frozenset(wids)

    def filter_env(self, env: e.TapeEnv) -> FrozenSet[int]:
        ids = self.all_ids
        for x, y0 in env.orig.items():
            ids = ids & self.obs_post.get((int(x), int(y0)), frozenset())
            if not ids:
                return ids
        for x in env.retested:
            ids = ids & self.ret_post.get((int(x), int(env.cur[int(x)])), frozenset())
            if not ids:
                return ids
        return ids


def worlds_from_instances(inst, split: str = "enum", wid0: int = 0) -> List[e.World]:
    worlds: List[e.World] = []
    wid = wid0
    for serial, d in enumerate(inst):
        init_obs = d["init_obs"]
        for k, cause in enumerate(e.CAUSES):
            truth = d["f1"] if cause == "model" else d["f0"]
            worlds.append(
                e.World(
                    wid=wid + k,
                    bundle=serial,
                    cause=cause,
                    truth=truth.copy(),
                    init_xs=tuple(sorted(init_obs)),
                    orig_init={int(x): int(y) for x, y in init_obs.items()},
                    dirty_x=d["dirty_x"],
                    dirty_y=d["dirty_y"],
                    t_break=d["t"],
                    split=split,
                )
            )
        wid += 3
    return worlds


_DELIVER: Dict[Tuple[Tuple[int, int], ...], np.ndarray] = {}


def deliver_pred(items: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    pred = _DELIVER.get(items)
    if pred is None:
        pred, _ = e.new_deliver({int(k): int(v) for k, v in items})
        _DELIVER[items] = pred
    return pred


def legal_actions(queried: FrozenSet[int], retested: FrozenSet[int]) -> List[Action]:
    acts: List[Action] = []
    for x in range(e.N):
        if x not in queried:
            acts.append(("query", int(x)))
    for x in sorted(queried):
        if x not in retested:
            acts.append(("retest", int(x)))
    return acts


def step_sets(queried: FrozenSet[int], retested: FrozenSet[int], act: Action):
    a, x = act
    if a == "query":
        return queried | {int(x)}, retested
    return queried, retested | {int(x)}


def terminal_mae(
    rows: np.ndarray,
    index: CatalogIndex,
    queried: FrozenSet[int],
    retested: FrozenSet[int],
) -> float:
    n = int(rows.size)
    if n == 0:
        return 0.0
    xs = sorted(int(x) for x in queried)
    buckets: Dict[Tuple[int, ...], List[int]] = {}
    obs = index.obs_arr
    ret = index.ret_arr
    rset = retested
    for wid in rows:
        key = tuple(
            int(ret[wid, x] if x in rset else obs[wid, x]) for x in xs
        )
        buckets.setdefault(key, []).append(int(wid))
    acc = 0.0
    truths = index.truth_arr
    n_dom = float(e.N)
    for key, wids in buckets.items():
        items = tuple((xs[j], key[j]) for j in range(len(xs)))
        pred = deliver_pred(items).astype(np.int64)
        t = truths[np.array(wids, dtype=np.int32)].astype(np.int64)
        acc += float(np.abs(t - pred[None, :]).sum()) / n_dom
    return acc / float(n)


def plan_openloop(
    rows: np.ndarray,
    index: CatalogIndex,
    queried: FrozenSet[int],
    retested: FrozenSet[int],
    rem: int,
) -> Tuple[List[Action], float, int]:
    """Optimal open-loop sequence. Policy depends on (Q,R,rem) only."""
    memo: Dict[tuple, Tuple[float, Optional[Action]]] = {}

    def value(q: FrozenSet[int], r: FrozenSet[int], k: int):
        key = (q, r, k)
        hit = memo.get(key)
        if hit is not None:
            return hit
        if k <= 0:
            v = terminal_mae(rows, index, q, r)
            memo[key] = (v, None)
            return memo[key]
        acts = legal_actions(q, r)
        if not acts:
            v = terminal_mae(rows, index, q, r)
            memo[key] = (v, None)
            return memo[key]
        best_v: Optional[float] = None
        best_a: Optional[Action] = None
        for act in acts:
            nq, nr = step_sets(q, r, act)
            v_next, _ = value(nq, nr, k - 1)
            if (
                best_v is None
                or v_next < best_v - 1e-15
                or (
                    abs(v_next - best_v) <= 1e-15
                    and best_a is not None
                    and act < best_a
                )
            ):
                best_v = v_next
                best_a = act
        memo[key] = (float(best_v), best_a)
        return memo[key]

    v0, _ = value(queried, retested, rem)
    seq: List[Action] = []
    q, r, k = queried, retested, rem
    while k > 0:
        _, act = memo[(q, r, k)]
        if act is None:
            break
        seq.append(act)
        q, r = step_sets(q, r, act)
        k -= 1
    return seq, float(v0), len(memo)


def run_cover(env: e.TapeEnv, horizon: int) -> Dict:
    h.run_cover_until(env, horizon)
    return {
        "support_failed_at": None,
        "fallback_steps": 0,
        "n_ids_init": None,
        "n_ids_final": None,
        "actions": [],
        "planned": [],
        "n_adapt_vs_plan": None,
    }


def run_retest(env: e.TapeEnv, horizon: int) -> Dict:
    h.run_retest_then_cover(env, horizon)
    return {
        "support_failed_at": None,
        "fallback_steps": 0,
        "n_ids_init": None,
        "n_ids_final": None,
        "actions": [],
        "planned": [],
        "n_adapt_vs_plan": None,
    }


def run_openloop(
    env: e.TapeEnv,
    index: CatalogIndex,
    horizon: int,
    seq_cache: Dict,
) -> Dict:
    ids = index.filter_env(env)
    rem = horizon - env.cost
    extra = {
        "support_failed_at": None,
        "fallback_steps": 0,
        "n_ids_init": len(ids),
        "n_ids_final": None,
        "actions": [],
        "planned": [],
        "n_adapt_vs_plan": None,
        "openloop_planned_mae": None,
        "openloop_n_states": None,
    }
    if not ids:
        extra["support_failed_at"] = env.cost
        while env.cost < horizon:
            x = e.next_cover(env.orig)
            if x is None:
                break
            env.query(int(x))
            extra["fallback_steps"] += 1
            extra["actions"].append(("fallback_query", int(x)))
        extra["n_ids_final"] = 0
        return extra
    key = (ids, frozenset(env.orig), frozenset(env.retested), rem)
    if key not in seq_cache:
        rows = np.array(sorted(ids), dtype=np.int32)
        seq, v, n_st = plan_openloop(
            rows, index, frozenset(int(x) for x in env.orig), frozenset(env.retested), rem
        )
        seq_cache[key] = (seq, v, n_st)
    seq, v, n_st = seq_cache[key]
    extra["planned"] = [list(a) for a in seq]
    extra["openloop_planned_mae"] = v
    extra["openloop_n_states"] = n_st
    for act, x in seq:
        if env.cost >= horizon:
            break
        extra["actions"].append((act, int(x)))
        if act == "query":
            env.query(int(x))
        else:
            env.retest(int(x))
    extra["n_ids_final"] = len(index.filter_env(env))
    return extra


def run_adaptive(
    env: e.TapeEnv,
    dp: FastDP,
    index: CatalogIndex,
    horizon: int,
) -> Dict:
    extra = {
        "support_failed_at": None,
        "fallback_steps": 0,
        "n_ids_init": None,
        "n_ids_final": None,
        "actions": [],
        "planned": [],
        "n_adapt_vs_plan": None,
        "min_ids": None,
    }
    ids = index.filter_env(env)
    extra["n_ids_init"] = len(ids)
    extra["min_ids"] = len(ids)
    while env.cost < horizon:
        rem = horizon - env.cost
        ids = index.filter_env(env)
        extra["min_ids"] = min(extra["min_ids"], len(ids))
        if not ids:
            if extra["support_failed_at"] is None:
                extra["support_failed_at"] = env.cost
            x = e.next_cover(env.orig)
            if x is None:
                break
            env.query(int(x))
            extra["fallback_steps"] += 1
            extra["actions"].append(("fallback_query", int(x)))
            continue
        _, act = dp.value(ids, frozenset(env.orig), frozenset(env.retested), rem)
        if act is None:
            if extra["support_failed_at"] is None:
                extra["support_failed_at"] = env.cost
            x = e.next_cover(env.orig)
            if x is None:
                break
            env.query(int(x))
            extra["fallback_steps"] += 1
            extra["actions"].append(("fallback_query", int(x)))
            continue
        a, x = act
        extra["actions"].append((a, int(x)))
        if a == "query":
            env.query(int(x))
        elif a == "retest":
            env.retest(int(x))
        else:
            break
    extra["n_ids_final"] = len(index.filter_env(env))
    return extra


def snapshot(env: e.TapeEnv) -> Dict:
    return h.snapshot(env)


def run_policy(
    world: e.World,
    name: str,
    horizon: int,
    dp: Optional[FastDP],
    index: CatalogIndex,
    seq_cache: Dict,
) -> Dict:
    t0 = time.perf_counter()
    env = e.TapeEnv(world)
    e.apply_init(env)
    if name == "cover":
        extra = run_cover(env, horizon)
    elif name == "retest_then_cover":
        extra = run_retest(env, horizon)
    elif name == "openloop":
        extra = run_openloop(env, index, horizon, seq_cache)
    elif name == "adaptive":
        extra = run_adaptive(env, dp, index, horizon)
    else:
        raise ValueError(name)
    snap = snapshot(env)
    orig_ok = all(env.orig[x] == world.observe(x) for x in env.orig)
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
        "cur_differs_from_orig": env.orig != env.cur,
        "truth_hash": hashlib.sha256(world.truth.tobytes()).hexdigest()[:16],
        **extra,
    }


def mean_ci(xs: Sequence[float]) -> Dict:
    return h.mean_ci(xs)


def paired_delta(a: Sequence[float], b: Sequence[float]) -> Dict:
    return h.paired_delta(a, b)


def summarize(records: List[Dict], horizon: int) -> Dict:
    rows = [r for r in records if r["horizon"] == horizon]
    by = {}
    for cause in e.CAUSES:
        sub = [r for r in rows if r["cause"] == cause]
        grouped: Dict[int, Dict[str, Dict]] = defaultdict(dict)
        for r in sub:
            grouped[r["wid"]][r["policy"]] = r
        wids = sorted(grouped)
        mae = {p: [grouped[w][p]["snap"]["mae"] for w in wids] for p in POLICIES}
        isol = {
            p: [float(bool(grouped[w][p]["snap"]["isol"])) for w in wids] for p in POLICIES
        }
        wrong = {
            p: [float(grouped[w][p]["snap"]["wrong_isolate"]) for w in wids]
            for p in POLICIES
        }
        empty = [
            float(grouped[w]["adaptive"]["support_failed_at"] is not None) for w in wids
        ]
        empty_ol = [
            float(grouped[w]["openloop"]["support_failed_at"] is not None) for w in wids
        ]
        fb = [grouped[w]["adaptive"]["fallback_steps"] for w in wids]
        seq_diff = []
        for w in wids:
            a = [tuple(x) if not isinstance(x, tuple) else x for x in grouped[w]["adaptive"]["actions"]]
            b = [tuple(x) if not isinstance(x, tuple) else x for x in grouped[w]["openloop"]["actions"]]
            seq_diff.append(float(a != b))
        block: Dict = {"n": len(wids), "policies": {}}
        for p in POLICIES:
            block["policies"][p] = {
                "mae": mean_ci(mae[p]),
                "isolate_rate": mean_ci(isol[p]),
                "wrong_isolate_rate": mean_ci(wrong[p]),
            }
        block["paired_openloop_minus_adaptive"] = paired_delta(mae["openloop"], mae["adaptive"])
        block["paired_cover_minus_adaptive"] = paired_delta(mae["cover"], mae["adaptive"])
        block["paired_retest_minus_adaptive"] = paired_delta(
            mae["retest_then_cover"], mae["adaptive"]
        )
        block["paired_cover_minus_openloop"] = paired_delta(mae["cover"], mae["openloop"])
        block["paired_retest_minus_openloop"] = paired_delta(
            mae["retest_then_cover"], mae["openloop"]
        )
        block["adaptive_support_failed_rate"] = mean_ci(empty)
        block["openloop_support_failed_rate"] = mean_ci(empty_ol)
        block["adaptive_fallback_steps"] = mean_ci(fb)
        block["frac_adapt_seq_differs_openloop"] = mean_ci(seq_diff)
        by[cause] = block
    all_mae = {p: [r["snap"]["mae"] for r in rows if r["policy"] == p] for p in POLICIES}
    e_ad = float(np.mean(all_mae["adaptive"])) if all_mae["adaptive"] else float("nan")
    e_ol = float(np.mean(all_mae["openloop"])) if all_mae["openloop"] else float("nan")
    e_cv = float(np.mean(all_mae["cover"])) if all_mae["cover"] else float("nan")
    e_rt = float(np.mean(all_mae["retest_then_cover"])) if all_mae["retest_then_cover"] else float("nan")
    return {
        "horizon": horizon,
        "n": len(rows) // len(POLICIES),
        "by_cause": by,
        "expected_mae_uniform": {p: mean_ci(all_mae[p]) for p in POLICIES},
        "delta_adapt_openloop_minus_adaptive": e_ol - e_ad,
        "acceptance_adaptive_leq_openloop": e_ad <= e_ol + 1e-12,
        "acceptance_adaptive_leq_cover": e_ad <= e_cv + 1e-12,
        "acceptance_adaptive_leq_retest": e_ad <= e_rt + 1e-12,
    }


def compact_rows(worlds: List[e.World], recs: List[Dict]) -> List[Dict]:
    out = []
    for w in worlds:
        row = {
            "wid": w.wid,
            "cause": w.cause,
            "bundle": w.bundle,
            "dirty_x": w.dirty_x,
            "init_xs": list(w.init_xs),
            "truth_hash": hashlib.sha256(w.truth.tobytes()).hexdigest()[:16],
        }
        for hz in HORIZONS:
            for p in POLICIES:
                hit = next(
                    r
                    for r in recs
                    if r["wid"] == w.wid and r["horizon"] == hz and r["policy"] == p
                )
                row[f"{p}_h{hz}_mae"] = hit["snap"]["mae"]
                row[f"{p}_h{hz}_isol"] = hit["snap"]["isol"]
            ad = next(
                r
                for r in recs
                if r["wid"] == w.wid and r["horizon"] == hz and r["policy"] == "adaptive"
            )
            ol = next(
                r
                for r in recs
                if r["wid"] == w.wid and r["horizon"] == hz and r["policy"] == "openloop"
            )
            row[f"ad_h{hz}_empty_at"] = ad["support_failed_at"]
            row[f"ad_h{hz}_fallback"] = ad["fallback_steps"]
            row[f"ad_h{hz}_n_ids_init"] = ad["n_ids_init"]
            row[f"ol_h{hz}_n_ids_init"] = ol["n_ids_init"]
            row[f"ol_h{hz}_planned"] = ol["planned"]
            row[f"ad_h{hz}_actions"] = [list(a) for a in ad["actions"]]
            row[f"seq_diff_h{hz}"] = [tuple(x) for x in ad["actions"]] != [
                tuple(x) if not isinstance(x, list) else tuple(x) for x in ol["actions"]
            ]
        out.append(row)
    return out


def evaluate_split(
    worlds: List[e.World],
    dp: FastDP,
    index: CatalogIndex,
    seq_cache: Dict,
) -> Tuple[Dict, List[Dict]]:
    recs: List[Dict] = []
    n = len(worlds)
    t0 = time.perf_counter()
    for i, w in enumerate(worlds):
        for hz in HORIZONS:
            for p in POLICIES:
                recs.append(run_policy(w, p, hz, dp, index, seq_cache))
        if (i + 1) % 60 == 0 or i + 1 == n:
            print(
                f"  eval {i+1}/{n} elapsed {time.perf_counter()-t0:.1f}s "
                f"ol_cache {len(seq_cache)} dp {len(dp._memo)} deliver {len(_DELIVER)}",
                flush=True,
            )
    out = {str(hz): summarize(recs, hz) for hz in HORIZONS}
    return out, compact_rows(worlds, recs)


def support_complete(index: CatalogIndex, catalog: List[e.World]) -> Dict:
    missing = 0
    empty = 0
    sizes = []
    later_empty = 0
    for w in catalog:
        env = e.TapeEnv(w)
        e.apply_init(env)
        ids = index.filter_env(env)
        sizes.append(len(ids))
        if w.wid not in ids:
            missing += 1
        if not ids:
            empty += 1
        env2 = e.TapeEnv(w)
        e.apply_init(env2)
        h.run_cover_until(env2, 6)
        if not index.filter_env(env2):
            later_empty += 1
    return {
        "n": len(catalog),
        "missing_true_world_after_init": missing,
        "empty_after_init": empty,
        "empty_after_cover_h6": later_empty,
        "support_min": int(min(sizes) if sizes else 0),
        "support_max": int(max(sizes) if sizes else 0),
        "support_mean": float(np.mean(sizes) if sizes else 0),
    }


def _cheat_pick(env: e.TapeEnv, u: Sequence[int]) -> int:
    """Peek unqueried labels. Rewrite 17x+3 must change the pick.

    Original y is small, so y-17x is maximized at the smallest x.
    After rewrite y=17x+3, the first key is constant and the
    tie-break picks the largest x.
    """

    def key(x):
        y = int(env.world.truth[int(x)])
        return (y - 17 * int(x), int(x))

    return int(max(u, key=key))


def isolation_check(catalog: List[e.World], dp: FastDP, index: CatalogIndex) -> Dict:
    seq_cache: Dict = {}
    trials = []
    policies = ("cover", "retest_then_cover", "openloop", "adaptive", "cheat")
    for policy in policies:
        for w in catalog[:9]:
            env_a = e.TapeEnv(w)
            e.apply_init(env_a)
            if policy == "cover":
                x = e.next_cover(env_a.orig)
                if x is not None:
                    env_a.query(int(x))
            elif policy == "retest_then_cover":
                h.run_retest_then_cover(env_a, 4)
            elif policy == "openloop":
                run_openloop(env_a, index, 4, seq_cache)
            elif policy == "adaptive":
                run_adaptive(env_a, dp, index, 4)
            else:
                u = [int(x) for x in range(e.N) if int(x) not in env_a.orig]
                env_a.query(int(_cheat_pick(env_a, u)))
            env_b = env_a.clone_unqueried_rewritten()

            def next_act(env, pol):
                if pol == "cheat":
                    u = [int(x) for x in range(e.N) if int(x) not in env.orig]
                    if not u:
                        return None
                    return "query", int(_cheat_pick(env, u))
                if pol == "cover":
                    x = e.next_cover(env.orig)
                    return ("query", x) if x is not None else None
                if pol == "retest_then_cover":
                    singles = e.restoring_singles(env.believed(), e.H0)
                    pending = [x for x in singles if int(x) not in env.retested]
                    if pending:
                        return "retest", int(pending[0])
                    x = e.next_cover(env.orig)
                    return ("query", x) if x is not None else None
                if pol == "openloop":
                    ids = index.filter_env(env)
                    rem = max(1, 4 - env.cost)
                    if not ids:
                        x = e.next_cover(env.orig)
                        return ("query", x) if x is not None else None
                    rows = np.array(sorted(ids), dtype=np.int32)
                    seq, _, _ = plan_openloop(
                        rows,
                        index,
                        frozenset(int(x) for x in env.orig),
                        frozenset(env.retested),
                        rem,
                    )
                    return seq[0] if seq else None
                ids = index.filter_env(env)
                if not ids:
                    x = e.next_cover(env.orig)
                    return ("query", x) if x is not None else None
                _, act = dp.value(ids, frozenset(env.orig), frozenset(env.retested), 2)
                return act

            na = next_act(env_a, policy)
            nb = next_act(env_b, policy)
            trials.append({"policy": policy, "ok": na == nb, "a": na, "b": nb})
    by = defaultdict(list)
    for t in trials:
        by[t["policy"]].append(t)
    summary = {
        p: {
            "n": len(ts),
            "n_fail": sum(1 for t in ts if not t["ok"]),
            "all_ok": all(t["ok"] for t in ts),
        }
        for p, ts in by.items()
    }
    honest = ("cover", "retest_then_cover", "openloop", "adaptive")
    return {
        "honest_must_pass": all(summary[p]["all_ok"] for p in honest),
        "cheat_must_fail": not summary["cheat"]["all_ok"],
        "by_policy": summary,
    }


def unit_adapt_leq_openloop(catalog: List[e.World], dp: FastDP, index: CatalogIndex) -> Dict:
    best = None
    for w in catalog:
        env = e.TapeEnv(w)
        e.apply_init(env)
        ids = index.filter_env(env)
        if best is None or len(ids) > len(best[0]):
            best = (ids, env)
    ids, env = best
    rem = 4
    v_ad, act_ad = dp.value(ids, frozenset(env.orig), frozenset(env.retested), rem)
    rows = np.array(sorted(ids), dtype=np.int32)
    seq, v_ol, n_st = plan_openloop(
        rows, index, frozenset(int(x) for x in env.orig), frozenset(env.retested), rem
    )
    return {
        "n_ids": len(ids),
        "v_adaptive": float(v_ad),
        "v_openloop": float(v_ol),
        "adaptive_first": list(act_ad) if act_ad else None,
        "openloop_seq": [list(a) for a in seq],
        "openloop_states": n_st,
        "ok": float(v_ad) <= float(v_ol) + 1e-12,
    }


def sampled_in_enum(inst, n: int = 20) -> Dict:
    hits = 0
    for serial in range(n):
        bun = e.make_bundle("probe", serial, CONFIRM_SEED0, 0)
        f0 = bun[1].truth
        f1 = bun[0].truth
        hit = False
        for d in inst:
            if (
                tuple(d["f0"].tolist()) == tuple(f0.tolist())
                and tuple(d["f1"].tolist()) == tuple(f1.tolist())
                and d["dirty_x"] == bun[0].dirty_x
                and d["init_obs"] == bun[0].orig_init
            ):
                hit = True
                break
        hits += int(hit)
    return {"n": n, "hits": hits, "all_in": hits == n}


def distinct_targets(worlds: Sequence[e.World]) -> int:
    keys = set()
    for w in worlds:
        keys.add((w.cause, w.dirty_x, hashlib.sha256(w.truth.tobytes()).hexdigest(), w.t_break))
    return len(keys)


def code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def print_split(name: str, sm: Dict) -> None:
    print("====", name)
    for hz in HORIZONS:
        block = sm[str(hz)]
        em = block["expected_mae_uniform"]
        print(
            f"  H={hz} E[mae] cover={em['cover']['mean']:.4f} "
            f"retest={em['retest_then_cover']['mean']:.4f} "
            f"openloop={em['openloop']['mean']:.4f} "
            f"adaptive={em['adaptive']['mean']:.4f} "
            f"d_adapt={block['delta_adapt_openloop_minus_adaptive']:.4f} "
            f"ad<=ol={block['acceptance_adaptive_leq_openloop']} "
            f"ad<=cv={block['acceptance_adaptive_leq_cover']} "
            f"ad<=rt={block['acceptance_adaptive_leq_retest']}"
        )
        for cause in e.CAUSES:
            b = block["by_cause"][cause]
            print(
                f"    {cause:11s}",
                {p: round(b["policies"][p]["mae"]["mean"], 4) for p in POLICIES},
                "d_adapt",
                round(b["paired_openloop_minus_adaptive"]["mean"], 4),
                "seq_diff",
                round(b["frac_adapt_seq_differs_openloop"]["mean"], 3),
                "empty_ad",
                b["adaptive_support_failed_rate"]["mean"],
            )


def main() -> None:
    t0 = time.perf_counter()
    print("enumerating public generator...", flush=True)
    inst = ep.enumerate_instances()
    catalog = worlds_from_instances(inst, split="enum", wid0=0)
    print("catalog worlds", len(catalog), "instances", len(inst), flush=True)
    index = CatalogIndex(catalog)
    dp = FastDP(catalog)

    cov = support_complete(index, catalog)
    print("support_complete", cov, flush=True)
    if cov["missing_true_world_after_init"] or cov["empty_after_init"]:
        raise SystemExit(f"catalog support incomplete: {cov}")

    hist = h.history_unit(catalog)
    print("history_unit", hist["ok"], flush=True)
    if not hist["ok"]:
        raise SystemExit(f"history unit failed: {hist}")

    gen_hit = sampled_in_enum(inst, 20)
    print("sampled_in_enum", gen_hit, flush=True)
    if not gen_hit["all_in"]:
        raise SystemExit(f"generator not covered by enum: {gen_hit}")

    unit = unit_adapt_leq_openloop(catalog, dp, index)
    print("unit_adapt_leq_openloop", unit, flush=True)
    if not unit["ok"]:
        raise SystemExit(f"adaptive Bayes risk > openloop: {unit}")

    iso = isolation_check(catalog, dp, index)
    print("iso", iso, flush=True)
    if not (iso["honest_must_pass"] and iso["cheat_must_fail"]):
        raise SystemExit(f"isolation failed: {iso}")

    seq_cache: Dict = {}
    print("catalog eval...", flush=True)
    cat_sum, cat_rows = evaluate_split(catalog, dp, index, seq_cache)

    locked = {
        "horizons": list(HORIZONS),
        "fitter": "imported frozen isolate-cap-1 from 20260914T111013Z",
        "catalog": "exhaustive enumeration of public generator; confirm sampled, not used to pick worlds",
        "n_instances": len(inst),
        "n_worlds": len(catalog),
        "prior": "ASSUMED uniform over enumerated worlds",
        "openloop": "sequence chosen at init from catalog posterior; not revised",
        "adaptive": "ExactDP on same catalog; replans after each outcome",
        "empty_catalog": "support_failed + covering fallback",
        "belief": "orig first-look + retest outcome",
        "confirm_seed0": CONFIRM_SEED0,
        "confirm_bundles": CONFIRM_BUNDLES,
        "code_sha256": code_hash(),
        "parameter_provenance": {
            "generator": "ASSUMED inherited public rules",
            "catalog_prior": "ASSUMED uniform",
            "metrics": "MEASURED",
        },
    }
    (HERE / "PROTOCOL.json").write_text(json.dumps(locked, indent=2) + "\n")
    if hashlib.sha256((HERE / "adapt_experiment.py").read_bytes()).hexdigest() != locked["code_sha256"]:
        raise SystemExit("protocol hash mismatch")

    print("confirm eval...", flush=True)
    confirm = e.build_catalog(CONFIRM_BUNDLES, CONFIRM_SEED0, "confirm", CONFIRM_WID0)
    conf_sum, conf_rows = evaluate_split(confirm, dp, index, seq_cache)

    result = {
        "protocol": locked,
        "support_complete": cov,
        "history_unit": hist,
        "generator_cover_sample": gen_hit,
        "unit_adapt_leq_openloop": unit,
        "isolation": iso,
        "dp": {"n_states": len(dp._memo), "n_calls": dp.n_calls},
        "openloop_cache": len(seq_cache),
        "deliver_cache": len(_DELIVER),
        "catalog_distinct_targets": distinct_targets(catalog),
        "confirm_distinct_targets": distinct_targets(confirm),
        "catalog": cat_sum,
        "confirm_NEW_SEEDS": conf_sum,
        "timing_s": time.perf_counter() - t0,
    }
    (HERE / "EVAL_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "CATALOG_ROWS.json").write_text(json.dumps(cat_rows, indent=2) + "\n")
    (HERE / "CONFIRM_ROWS.json").write_text(json.dumps(conf_rows, indent=2) + "\n")

    print("elapsed", round(result["timing_s"], 3), "dp_states", len(dp._memo))
    print_split("catalog", cat_sum)
    print_split("confirm", conf_sum)


if __name__ == "__main__":
    main()
