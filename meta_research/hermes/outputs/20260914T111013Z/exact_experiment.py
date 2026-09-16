#!/usr/bin/env python3
"""Exact decision reference after a frozen isolate-capable fitter.

Cleanup vs 20260914T084840Z (that directory is not modified):
- orig log vs current believed (retest edits cur only)
- repairs keep competing live rows, not only table[live][0]
- lookahead (supplement) uses normalized outcome weights and an
  ASSUMED retest model; it is not the primary comparator
- TLA claims are in TLA_CLAIMS.md; this file is the numeric experiment

Question: after repair can isolate one observation, is there still
scheduling value worth paying for? Frozen fitter. No soft weights.

ASSUMED: mini library, catalog prior, action costs 1, isolate unbilled.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent

N = 8
DOMAIN = np.arange(N, dtype=np.int32)
A_RANGE = np.array([0, 1], dtype=np.int32)
B_RANGE = np.array([-1, 0, 1], dtype=np.int32)
T_RANGE = np.array([3, 5], dtype=np.int32)
N_INIT = 2
BUDGETS = (3, 4, 5, 6)
MAX_BUDGET = 6
COST_QUERY = 1
COST_RETEST = 1
CAUSES = ("model", "transient", "persistent")
POLICIES = ("cover", "retest_then_cover", "optimal")
CATALOG_BUNDLES = 6
HOLDOUT_BUNDLES = 6
CATALOG_SEED0 = 1
HOLDOUT_SEED0 = 10_000
FORBIDDEN_INIT = {0, 4}
P_TRANSIENT_RETEST = 0.5  # ASSUMED, used only by lookahead supplement


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


def build_libraries() -> Tuple[np.ndarray, np.ndarray]:
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
    h1 = np.concatenate([h0, np.stack(pieces, axis=0)], axis=0)
    return h0, h1


H0, H1 = build_libraries()


def in_library(table: np.ndarray, f: np.ndarray) -> bool:
    return bool(np.any(np.all(table == f[None, :], axis=1)))


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


def competing_preds(table: np.ndarray, live: np.ndarray, cap: int = 8) -> List[np.ndarray]:
    """Keep distinct live rows, not only the first index.

    cap is an ASSUMED bound so lookahead stays finite.
    """
    rows = table[live]
    uniq = []
    seen = set()
    for row in rows:
        key = tuple(int(v) for v in row.tolist())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row.copy())
        if len(uniq) >= cap:
            break
    return uniq


def enumerate_repairs(believed: Dict[int, int]) -> List[Repair]:
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
            if not live.any():
                continue
            for pred in competing_preds(table, live):
                found.append(
                    Repair(fam, isol, tuple(int(v) for v in pred.tolist()), int(live.sum()))
                )
    found.sort(
        key=lambda r: (
            len(r.isol),
            0 if r.family == "H0" else 1,
            min(r.isol) if r.isol else -1,
            r.pred,
        )
    )
    return found


def new_deliver(believed: Dict[int, int]) -> Tuple[np.ndarray, Dict]:
    reps = enumerate_repairs(believed)
    if not reps:
        return nn_fill(believed), {
            "family": "nn",
            "isol": [],
            "consistent": False,
            "n_repairs": 0,
        }
    r = reps[0]
    return r.pred_arr().copy(), {
        "family": r.family,
        "isol": sorted(r.isol),
        "consistent": True,
        "n_repairs": len(reps),
        "n_distinct_preds": len({r.pred for r in reps}),
    }


def next_cover(queried: Sequence[int]) -> Optional[int]:
    q = set(int(x) for x in queried)
    for x in COVER:
        if int(x) not in q:
            return int(x)
    return None


def mae_of(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred.astype(np.int64) - truth.astype(np.int64)).sum()) / float(N)


@dataclass
class World:
    wid: int
    bundle: int
    cause: str
    truth: np.ndarray
    init_xs: Tuple[int, ...]
    orig_init: Dict[int, int]
    dirty_x: int
    dirty_y: int
    t_break: int
    split: str

    def observe(self, x: int) -> int:
        x = int(x)
        if self.cause == "model":
            return int(self.truth[x])
        if x == self.dirty_x:
            return int(self.dirty_y)
        return int(self.truth[x])

    def retest(self, x: int) -> int:
        x = int(x)
        if self.cause == "transient":
            return int(self.truth[x])
        if self.cause == "persistent" and x == self.dirty_x:
            return int(self.dirty_y)
        return int(self.truth[x])


def build_shared_instance(rng: np.random.Generator):
    for _ in range(30_000):
        f0 = H0[int(rng.integers(0, len(H0)))].copy()
        a = int(f0[1] - f0[0]) if N > 1 else int(f0[0])
        b = int(f0[0])
        t = int(rng.choice(T_RANGE))
        a2 = int(rng.choice(A_RANGE))
        b2 = int(rng.choice(B_RANGE))
        if a2 == a and b2 == b:
            continue
        left = DOMAIN < t
        right = ~left
        if int(left.sum()) < 2 or int(right.sum()) < 2:
            continue
        f1 = np.empty(N, dtype=np.int32)
        f1[left] = a * DOMAIN[left] + b
        f1[right] = a2 * DOMAIN[right] + b2
        if in_library(H0, f1) or not in_library(H1, f1):
            continue
        dirty_x = int(rng.choice(DOMAIN[right]))
        dirty_y = int(f1[dirty_x])
        if dirty_y == int(f0[dirty_x]):
            continue
        left_xs = DOMAIN[left]
        c1 = int(rng.choice(left_xs))
        init_set = {c1, dirty_x}
        if init_set == FORBIDDEN_INIT or len(init_set) < 2:
            continue
        init_obs = {c1: int(f0[c1]), dirty_x: int(dirty_y)}
        if int(f1[c1]) != init_obs[c1]:
            continue
        if mask_obs(H0, init_obs).any() or not mask_obs(H1, init_obs).any():
            continue
        return f0, f1, dict(init_obs), dirty_x, dirty_y, t
    raise RuntimeError("failed to sample shared instance")


def make_bundle(split: str, serial: int, seed0: int, wid0: int) -> List[World]:
    rng = np.random.default_rng(seed0 + serial)
    f0, f1, init_obs, dirty_x, dirty_y, t = build_shared_instance(rng)
    out = []
    for k, cause in enumerate(CAUSES):
        truth = f1 if cause == "model" else f0
        out.append(
            World(
                wid=wid0 + k,
                bundle=serial,
                cause=cause,
                truth=truth.copy(),
                init_xs=tuple(sorted(init_obs)),
                orig_init={int(x): int(y) for x, y in init_obs.items()},
                dirty_x=dirty_x,
                dirty_y=dirty_y,
                t_break=t,
                split=split,
            )
        )
    return out


def build_catalog(n_bundles: int, seed0: int, split: str, wid0: int = 0) -> List[World]:
    worlds = []
    wid = wid0
    for serial in range(n_bundles):
        bun = make_bundle(split, serial, seed0, wid)
        worlds.extend(bun)
        wid += len(CAUSES)
    return worlds


class TapeEnv:
    """orig is first-write only. retest updates cur. Isolate is fitter-only."""

    def __init__(self, world: World):
        self.world = world
        self.orig: Dict[int, int] = {}
        self.cur: Dict[int, int] = {}
        self.retested: set = set()
        self.cost = 0
        self.n_query = 0
        self.n_retest = 0

    def believed(self) -> Dict[int, int]:
        return {int(k): int(self.cur[k]) for k in self.cur}

    def query(self, x: int) -> int:
        x = int(x)
        if x in self.orig:
            raise RuntimeError("query of recorded x")
        y = self.world.observe(x)
        self.orig[x] = int(y)
        self.cur[x] = int(y)
        self.cost += COST_QUERY
        self.n_query += 1
        return int(y)

    def retest(self, x: int) -> int:
        x = int(x)
        if x not in self.orig:
            raise RuntimeError("retest of unrecorded x")
        y2 = self.world.retest(x)
        self.cur[x] = int(y2)
        self.retested.add(x)
        self.cost += COST_RETEST
        self.n_retest += 1
        return int(y2)

    def clone_unqueried_rewritten(self) -> "TapeEnv":
        t = self.world.truth.copy()
        for x in range(N):
            if int(x) not in self.orig:
                t[int(x)] = int(17 * int(x) + 3)
        w = World(
            wid=self.world.wid,
            bundle=self.world.bundle,
            cause=self.world.cause,
            truth=t,
            init_xs=self.world.init_xs,
            orig_init=dict(self.world.orig_init),
            dirty_x=self.world.dirty_x,
            dirty_y=self.world.dirty_y,
            t_break=self.world.t_break,
            split=self.world.split,
        )
        env = TapeEnv(w)
        env.orig = dict(self.orig)
        env.cur = dict(self.cur)
        env.retested = set(self.retested)
        env.cost = self.cost
        env.n_query = self.n_query
        env.n_retest = self.n_retest
        return env


def apply_init(env: TapeEnv) -> None:
    for x in env.world.init_xs:
        env.query(int(x))


def expected_mae(belief: Sequence[World], believed: Dict[int, int]) -> float:
    if not belief:
        return 0.0
    pred, _ = new_deliver(believed)
    return float(np.mean([mae_of(pred, w.truth) for w in belief]))


def believed_from_worlds(
    remaining: Sequence[World], queried: FrozenSet[int], retested: FrozenSet[int]
) -> Dict[int, int]:
    if not remaining:
        return {}
    w0 = remaining[0]
    out = {}
    for x in queried:
        out[int(x)] = int(w0.retest(x) if int(x) in retested else w0.observe(x))
    for w in remaining[1:]:
        for x in queried:
            y = int(w.retest(x) if int(x) in retested else w.observe(x))
            if y != out[int(x)]:
                raise RuntimeError("remaining worlds disagree on believed")
    return out


class ExactDP:
    def __init__(self, catalog: List[World]):
        self.catalog = catalog
        self.n_calls = 0
        self._memo: Dict[tuple, Tuple[float, Optional[Tuple[str, Optional[int]]]]] = {}

    def key(
        self,
        ids: FrozenSet[int],
        queried: FrozenSet[int],
        retested: FrozenSet[int],
        rem: int,
    ):
        return (ids, queried, retested, rem)

    def remaining(self, ids: FrozenSet[int]) -> List[World]:
        return [w for w in self.catalog if w.wid in ids]

    def value(
        self,
        ids: FrozenSet[int],
        queried: FrozenSet[int],
        retested: FrozenSet[int],
        rem: int,
    ) -> Tuple[float, Optional[Tuple[str, Optional[int]]]]:
        k = self.key(ids, queried, retested, rem)
        if k in self._memo:
            return self._memo[k]
        self.n_calls += 1
        rem_w = self.remaining(ids)
        believed = believed_from_worlds(rem_w, queried, retested)
        if rem <= 0 or not rem_w:
            v = expected_mae(rem_w, believed)
            self._memo[k] = (v, None)
            return self._memo[k]

        actions: List[Tuple[str, Optional[int]]] = []
        for x in range(N):
            if x not in queried:
                actions.append(("query", int(x)))
        for x in sorted(queried):
            if x not in retested:
                actions.append(("retest", int(x)))
        if not actions:
            v = expected_mae(rem_w, believed)
            self._memo[k] = (v, None)
            return self._memo[k]

        best_v = None
        best_a = None
        for act, x in actions:
            groups: Dict[int, List[World]] = {}
            for w in rem_w:
                y = int(w.observe(x) if act == "query" else w.retest(x))
                groups.setdefault(y, []).append(w)
            exp = 0.0
            n = float(len(rem_w))
            for y, ws in groups.items():
                p = len(ws) / n
                nq = queried | ({int(x)} if act == "query" else queried)
                nr = retested | ({int(x)} if act == "retest" else retested)
                nids = frozenset(w.wid for w in ws)
                v_next, _ = self.value(nids, nq, nr, rem - 1)
                exp += p * v_next
            if best_v is None or exp < best_v - 1e-15 or (
                abs(exp - best_v) <= 1e-15 and best_a is not None and (act, x) < best_a
            ):
                best_v = exp
                best_a = (act, x)
        self._memo[k] = (float(best_v), best_a)
        return self._memo[k]


def lookahead_action(
    believed: Dict[int, int],
    orig: Dict[int, int],
    retested: set,
    remaining: int,
) -> Tuple[Optional[str], Optional[int]]:
    """Corrected 1-step. Supplement only. ASSUMED retest mixture."""
    reps = enumerate_repairs(believed)
    if remaining < 1:
        return None, None
    if not reps:
        x = next_cover(orig)
        return ("query", x) if x is not None else (None, None)

    pick = reps[0]
    base = 0.0
    for r in reps:
        base += float(np.abs(pick.pred_arr().astype(np.int64) - r.pred_arr().astype(np.int64)).sum()) / float(N)
    base /= float(len(reps))
    best = None

    u = [int(x) for x in DOMAIN if int(x) not in orig]
    mats = np.stack([r.pred_arr() for r in reps], axis=0) if reps else None
    for x in u:
        labels = sorted({int(r.pred[x]) for r in reps})
        if len(labels) <= 1:
            continue
        counts = {y: sum(1 for r in reps if int(r.pred[x]) == y) for y in labels}
        ztot = float(sum(counts.values()))
        exp_loss = 0.0
        for y, c in counts.items():
            p = c / ztot
            nb = dict(believed)
            nb[int(x)] = int(y)
            nr = enumerate_repairs(nb)
            if not nr:
                exp_loss += p * base
            else:
                pr = nr[0].pred_arr().astype(np.int64)
                el = 0.0
                for r in nr:
                    el += float(np.abs(pr - r.pred_arr().astype(np.int64)).sum()) / float(N)
                exp_loss += p * (el / float(len(nr)))
        gain = (base - exp_loss) / float(COST_QUERY)
        cand = (gain, 0, -int(x), "query", int(x))
        if best is None or cand[:3] > best[:3]:
            best = cand

    for z in sorted(orig):
        if int(z) in retested:
            continue
        y_now = int(believed[int(z)])
        y_alts = sorted({int(r.pred[z]) for r in reps if int(r.pred[z]) != y_now})
        if not y_alts:
            continue
        p_change = P_TRANSIENT_RETEST
        p_same = 1.0 - p_change
        exp_loss = 0.0
        nb_same = dict(believed)
        nr_same = enumerate_repairs(nb_same)
        if not nr_same:
            loss_same = base
        else:
            pr = nr_same[0].pred_arr().astype(np.int64)
            loss_same = 0.0
            for r in nr_same:
                loss_same += float(np.abs(pr - r.pred_arr().astype(np.int64)).sum()) / float(N)
            loss_same /= float(len(nr_same))
        exp_loss += p_same * loss_same
        p_each = p_change / float(len(y_alts))
        for y in y_alts:
            nb = dict(believed)
            nb[int(z)] = int(y)
            nr = enumerate_repairs(nb)
            if not nr:
                exp_loss += p_each * base
                continue
            pr = nr[0].pred_arr().astype(np.int64)
            el = 0.0
            for r in nr:
                el += float(np.abs(pr - r.pred_arr().astype(np.int64)).sum()) / float(N)
            exp_loss += p_each * (el / float(len(nr)))
        gain = (base - exp_loss) / float(COST_RETEST)
        cand = (gain, -1, -int(z), "retest", int(z))
        if best is None or cand[:3] > best[:3]:
            best = cand

    if best is not None and best[0] > 1e-12:
        return best[3], best[4]
    x = next_cover(orig)
    return ("query", x) if x is not None else (None, None)


def choose_fixed(name: str, env: TapeEnv) -> Tuple[Optional[str], Optional[int]]:
    believed = env.believed()
    if name == "cover":
        x = next_cover(env.orig)
        return ("query", x) if x is not None else (None, None)
    if name == "retest_then_cover":
        singles = restoring_singles(believed, H0)
        pending = [x for x in singles if int(x) not in env.retested]
        if pending:
            return "retest", int(pending[0])
        x = next_cover(env.orig)
        return ("query", x) if x is not None else (None, None)
    if name == "lookahead":
        return lookahead_action(believed, env.orig, env.retested, MAX_BUDGET - env.cost)
    raise ValueError(name)


def cheat_next(env: TapeEnv) -> Tuple[Optional[str], Optional[int]]:
    u = [int(x) for x in range(N) if int(x) not in env.orig]
    if not u:
        return None, None
    hidden = np.array([int(env.world.truth[x]) for x in u], dtype=np.int64)
    score = hidden * 31 + np.array(u, dtype=np.int64)
    return "query", int(u[int(np.argmax(score))])


def run_policy(world: World, name: str, dp: Optional[ExactDP], catalog_ids: FrozenSet[int]) -> Dict:
    t0 = time.perf_counter()
    env = TapeEnv(world)
    traj = []
    apply_init(env)
    checkpoints = {}

    def note():
        pred, meta = new_deliver(env.believed())
        isol = meta.get("isol") or []
        snap = {
            "budget": env.cost,
            "mae": mae_of(pred, world.truth),
            "family": meta.get("family"),
            "isol": isol,
            "wrong_isolate": bool(isol) and world.cause == "model",
            "n_query": env.n_query,
            "n_retest": env.n_retest,
            "n_orig": len(env.orig),
            "n_cur": len(env.cur),
        }
        for b in BUDGETS:
            if env.cost >= b and str(b) not in checkpoints:
                checkpoints[str(b)] = snap

    note()
    ids = catalog_ids
    queried = frozenset(env.orig)
    retested = frozenset(env.retested)
    if name == "optimal" and dp is not None:
        ids = frozenset(
            w.wid
            for w in dp.catalog
            if all(
                (w.observe(x) if x not in retested else w.retest(x)) == env.cur[x]
                for x in env.orig
            )
            and all(int(x) in env.orig for x in env.orig)
            and set(env.orig) <= set(range(N))
        )
        ids = frozenset(
            w.wid
            for w in dp.catalog
            if all(int(x) in env.orig for x in env.orig)
            and all(
                (w.retest(x) if x in env.retested else w.observe(x)) == env.cur[x]
                for x in env.orig
            )
        )

    while env.cost < MAX_BUDGET:
        rem = MAX_BUDGET - env.cost
        if name == "optimal":
            if dp is None:
                raise RuntimeError("optimal requires dp")
            ids = frozenset(
                w.wid
                for w in dp.catalog
                if all(
                    (w.retest(x) if x in env.retested else w.observe(x)) == env.cur[x]
                    for x in env.orig
                )
            )
            _, act = dp.value(ids, frozenset(env.orig), frozenset(env.retested), rem)
            if act is None:
                break
            a, x = act
        else:
            a, x = choose_fixed(name, env)
        if a is None or x is None:
            break
        if a == "query":
            y = env.query(int(x))
            traj.append({"a": "query", "x": int(x), "y": int(y), "cost": env.cost})
        elif a == "retest":
            y2 = env.retest(int(x))
            traj.append({"a": "retest", "x": int(x), "y": int(y2), "cost": env.cost})
        else:
            raise RuntimeError(a)
        note()

    for b in BUDGETS:
        if str(b) not in checkpoints:
            pred, meta = new_deliver(env.believed())
            checkpoints[str(b)] = {
                "budget": env.cost,
                "mae": mae_of(pred, world.truth),
                "family": meta.get("family"),
                "isol": meta.get("isol") or [],
                "wrong_isolate": bool(meta.get("isol")) and world.cause == "model",
                "n_query": env.n_query,
                "n_retest": env.n_retest,
                "n_orig": len(env.orig),
                "n_cur": len(env.cur),
            }
    pred, meta = new_deliver(env.believed())
    orig_ok = all(env.orig[x] == world.observe(x) for x in env.orig)
    return {
        "policy": name,
        "wid": world.wid,
        "cause": world.cause,
        "bundle": world.bundle,
        "dirty_x": world.dirty_x,
        "t_break": world.t_break,
        "init_xs": list(world.init_xs),
        "elapsed_s": time.perf_counter() - t0,
        "checkpoints": checkpoints,
        "final": checkpoints[str(MAX_BUDGET)],
        "traj": traj,
        "orig": {str(k): int(v) for k, v in env.orig.items()},
        "cur": {str(k): int(v) for k, v in env.cur.items()},
        "orig_untouched_by_retest": orig_ok,
        "truth_hash": hashlib.sha256(world.truth.tobytes()).hexdigest()[:16],
    }


def dual_env_isolation(catalog: List[World]) -> Dict:
    trials = []
    for policy in ("cover", "retest_then_cover", "lookahead", "cheat"):
        for world in catalog[:9]:
            env_a = TapeEnv(world)
            apply_init(env_a)
            steps = 0
            while env_a.cost < 4 and steps < 2:
                if policy == "cheat":
                    a, x = cheat_next(env_a)
                else:
                    a, x = choose_fixed(policy, env_a)
                if a is None:
                    break
                if a == "query" and x is not None and int(x) not in env_a.orig:
                    env_a.query(int(x))
                elif a == "retest" and x is not None:
                    env_a.retest(int(x))
                steps += 1
            env_b = env_a.clone_unqueried_rewritten()
            if policy == "cheat":
                na = cheat_next(env_a)
                nb = cheat_next(env_b)
            else:
                na = choose_fixed(policy, env_a)
                nb = choose_fixed(policy, env_b)
            trials.append(
                {
                    "policy": policy,
                    "cause": world.cause,
                    "next_a": na,
                    "next_b": nb,
                    "ok": na == nb,
                    "orig_equal": env_a.orig == env_b.orig,
                    "envs_distinct": not np.array_equal(env_a.world.truth, env_b.world.truth),
                }
            )
    by: Dict[str, list] = {}
    for t in trials:
        by.setdefault(t["policy"], []).append(t)
    summary = {
        p: {"n": len(ts), "n_fail": sum(1 for t in ts if not t["ok"]), "all_ok": all(t["ok"] for t in ts)}
        for p, ts in by.items()
    }
    honest = ("cover", "retest_then_cover", "lookahead")
    return {
        "honest_must_pass": all(summary[p]["all_ok"] for p in honest),
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


def summarize(rows: List[Dict], policies: Sequence[str]) -> Dict:
    by = {}
    for cause in CAUSES:
        sub = [r for r in rows if r["cause"] == cause]
        hashes = {r["recs"][policies[0]]["truth_hash"] for r in sub}
        block = {"n": len(sub), "n_distinct_truths": len(hashes), "budgets": {}}
        for b in BUDGETS:
            bblock = {}
            mae = {p: [r["recs"][p]["checkpoints"][str(b)]["mae"] for r in sub] for p in policies}
            isol = {
                p: [float(bool(r["recs"][p]["checkpoints"][str(b)]["isol"])) for r in sub]
                for p in policies
            }
            wrong = {
                p: [float(r["recs"][p]["checkpoints"][str(b)]["wrong_isolate"]) for r in sub]
                for p in policies
            }
            nret = {p: [r["recs"][p]["checkpoints"][str(b)]["n_retest"] for r in sub] for p in policies}
            el = {p: [r["recs"][p]["elapsed_s"] for r in sub] for p in policies}
            for p in policies:
                bblock[p] = {
                    "mae": mean_ci(mae[p]),
                    "isolate_rate": mean_ci(isol[p]),
                    "wrong_isolate_rate": mean_ci(wrong[p]),
                    "n_retest": mean_ci(nret[p]),
                    "elapsed_s": mean_ci(el[p]),
                }
            if "cover" in policies and "optimal" in policies:
                bblock["paired_cover_minus_optimal"] = paired_delta(mae["cover"], mae["optimal"])
            if "retest_then_cover" in policies and "optimal" in policies:
                bblock["paired_retest_minus_optimal"] = paired_delta(
                    mae["retest_then_cover"], mae["optimal"]
                )
            if "cover" in policies and "retest_then_cover" in policies:
                bblock["paired_cover_minus_retest"] = paired_delta(
                    mae["cover"], mae["retest_then_cover"]
                )
            block["budgets"][str(b)] = bblock
        by[cause] = block
    return {
        "by_cause": by,
        "n_tasks": len(rows),
        "rows_compact": [
            {
                "wid": r["wid"],
                "cause": r["cause"],
                "bundle": r["bundle"],
                "dirty_x": r["dirty_x"],
                "t": r["t_break"],
                "init_xs": r["init_xs"],
                **{f"{p}_mae{b}": r["recs"][p]["checkpoints"][str(b)]["mae"] for p in policies for b in BUDGETS},
                **{f"{p}_isol6": r["recs"][p]["checkpoints"]["6"]["isol"] for p in policies},
                **{f"{p}_nret6": r["recs"][p]["checkpoints"]["6"]["n_retest"] for p in policies},
            }
            for r in rows
        ],
    }


def evaluate(worlds: List[World], dp: ExactDP, policies: Sequence[str]) -> Tuple[Dict, List[Dict], List[Dict]]:
    catalog_ids = frozenset(w.wid for w in dp.catalog)
    rows = []
    trajs = []
    for i, world in enumerate(worlds):
        recs = {p: run_policy(world, p, dp if p == "optimal" else None, catalog_ids) for p in policies}
        rows.append(
            {
                "wid": world.wid,
                "cause": world.cause,
                "bundle": world.bundle,
                "dirty_x": world.dirty_x,
                "t_break": world.t_break,
                "init_xs": list(world.init_xs),
                "recs": recs,
            }
        )
        trajs.append({"wid": world.wid, "cause": world.cause, "traj": {p: recs[p]["traj"] for p in policies}})
        if (i + 1) % 9 == 0:
            print(f"eval {i+1}/{len(worlds)}", flush=True)
    return summarize(rows, policies), rows, trajs


def dual_filter_check(catalog: List[World]) -> Dict:
    n_ok = 0
    n = 0
    for w in catalog:
        env = TapeEnv(w)
        apply_init(env)
        n += 1
        if env.orig == dict(w.orig_init) or set(env.orig) == set(w.orig_init):
            if all(env.orig[x] == w.orig_init[x] for x in w.orig_init):
                n_ok += 1
    return {"n": n, "n_ok": n_ok, "all_ok": n_ok == n}


def library_checks(catalog: List[World]) -> List[Dict]:
    out = []
    out.append({"name": "h0_size", "ok": len(H0) == 6, "detail": int(len(H0))})
    out.append({"name": "h1_gt_h0", "ok": len(H1) > len(H0), "detail": int(len(H1))})
    inits = [tuple(w.init_xs) for w in catalog]
    out.append(
        {
            "name": "no_forbidden_init",
            "ok": all(set(xs) != FORBIDDEN_INIT for xs in inits),
            "detail": list({xs for xs in inits}),
        }
    )
    dirtys = [w.dirty_x for w in catalog]
    ts = [w.t_break for w in catalog]
    out.append({"name": "dirty_varies", "ok": len(set(dirtys)) > 1, "detail": dirtys})
    out.append({"name": "t_varies", "ok": len(set(ts)) > 1, "detail": ts})
    ph = dual_filter_check(catalog)
    out.append({"name": "init_matches", "ok": ph["all_ok"], "detail": ph})
    triples = {}
    for w in catalog:
        triples.setdefault(w.bundle, []).append(w)
    pair_ok = True
    for bun, ws in triples.items():
        inits = [w.orig_init for w in ws]
        if not all(d == inits[0] for d in inits):
            pair_ok = False
    out.append({"name": "paired_init", "ok": pair_ok, "detail": len(triples)})
    env = TapeEnv(catalog[0])
    apply_init(env)
    x = next_cover(env.orig)
    if x is not None:
        env.query(int(x))
    before = dict(env.orig)
    if env.orig:
        z = next(iter(env.orig))
        env.retest(int(z))
    out.append(
        {
            "name": "orig_stable_after_retest",
            "ok": env.orig == before,
            "detail": {"orig": env.orig, "cur": env.cur},
        }
    )
    return out


def code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main():
    t0 = time.perf_counter()
    catalog = build_catalog(CATALOG_BUNDLES, CATALOG_SEED0, "catalog", 0)
    holdout = build_catalog(HOLDOUT_BUNDLES, HOLDOUT_SEED0, "holdout", 1000)
    checks = library_checks(catalog)
    iso = dual_env_isolation(catalog)
    if not all(c["ok"] for c in checks):
        raise SystemExit(f"sanity failed: {checks}")
    if not (iso["honest_must_pass"] and iso["cheat_must_fail"]):
        raise SystemExit(f"isolation failed: {iso}")

    print("solve DP on catalog...", flush=True)
    dp = ExactDP(catalog)
    tdp = time.perf_counter()
    for w in catalog:
        env = TapeEnv(w)
        apply_init(env)
        ids = frozenset(
            u.wid
            for u in catalog
            if all((u.observe(x) == env.cur[x]) for x in env.orig)
        )
        dp.value(ids, frozenset(env.orig), frozenset(), MAX_BUDGET - env.cost)
    dp_s = time.perf_counter() - tdp
    print("dp_states", len(dp._memo), "calls", dp.n_calls, "s", round(dp_s, 3), flush=True)

    locked = {
        "N": N,
        "H0": int(len(H0)),
        "H1": int(len(H1)),
        "BUDGETS": list(BUDGETS),
        "MAX_BUDGET": MAX_BUDGET,
        "CAUSES": list(CAUSES),
        "policies_primary": list(POLICIES),
        "costs_ASSUMED": {"query": COST_QUERY, "retest": COST_RETEST, "isolate": 0},
        "prior_ASSUMED": "uniform over locked catalog worlds",
        "fitter": "isolate-cap-1, competing live rows kept, rank fewer isol then H0",
        "orig_vs_cur": "orig first-write; retest edits cur only",
        "lookahead_ASSUMED": {"P_TRANSIENT_RETEST": P_TRANSIENT_RETEST},
        "forbidden_init": sorted(FORBIDDEN_INIT),
        "catalog_bundles": CATALOG_BUNDLES,
        "holdout_bundles": HOLDOUT_BUNDLES,
        "code_sha256": code_hash(),
        "parameter_provenance": {
            "libraries": "ASSUMED mini",
            "catalog": "ASSUMED generator",
            "dp_prior": "ASSUMED uniform catalog",
            "isolation": "MEASURED",
            "metrics": "MEASURED",
        },
    }
    (HERE / "PROTOCOL.json").write_text(json.dumps(locked, indent=2) + "\n")

    print("catalog eval...", flush=True)
    cat_sum, _, cat_traj = evaluate(catalog, dp, POLICIES)
    print("holdout eval...", flush=True)
    hol_sum, _, hol_traj = evaluate(holdout, dp, POLICIES)

    print("lookahead supplement on catalog...", flush=True)
    extra_policies = ("cover", "retest_then_cover", "lookahead")
    extra_rows = []
    for world in catalog:
        recs = {p: run_policy(world, p, None, frozenset()) for p in extra_policies}
        extra_rows.append(
            {
                "wid": world.wid,
                "cause": world.cause,
                "bundle": world.bundle,
                "dirty_x": world.dirty_x,
                "t_break": world.t_break,
                "init_xs": list(world.init_xs),
                "recs": recs,
            }
        )
    extra_sum = summarize(extra_rows, extra_policies)

    elapsed = time.perf_counter() - t0
    result = {
        "protocol": locked,
        "sanity": {"all_ok": True, "checks": checks},
        "isolation": iso,
        "dp": {"n_states": len(dp._memo), "n_calls": dp.n_calls, "solve_s": dp_s},
        "catalog": cat_sum,
        "holdout": hol_sum,
        "lookahead_supplement_NOT_PRIMARY": extra_sum,
        "timing_s": elapsed,
        "n_unique_dirty_catalog": len({w.dirty_x for w in catalog}),
        "n_unique_t_catalog": len({w.t_break for w in catalog}),
    }
    (HERE / "EVAL_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "CATALOG_ROWS.json").write_text(json.dumps(cat_sum["rows_compact"], indent=2) + "\n")
    (HERE / "HOLDOUT_ROWS.json").write_text(json.dumps(hol_sum["rows_compact"], indent=2) + "\n")
    (HERE / "CATALOG_TRAJ.json").write_text(json.dumps(cat_traj, indent=2) + "\n")
    print("elapsed_s", round(elapsed, 3))
    print("iso", iso)
    for split_name, sm in (("catalog", cat_sum), ("holdout", hol_sum)):
        print("====", split_name)
        for cause in CAUSES:
            print(" ", cause, "n", sm["by_cause"][cause]["n"], "truths", sm["by_cause"][cause]["n_distinct_truths"])
            for b in BUDGETS:
                blk = sm["by_cause"][cause]["budgets"][str(b)]
                print(f"  @{b}")
                for p in POLICIES:
                    g = blk[p]["mae"]
                    print(
                        f"    {p:18s} MAE={g['mean']:.4f}±{g['se']:.3f} "
                        f"isol={blk[p]['isolate_rate']['mean']:.2f} "
                        f"wrong={blk[p]['wrong_isolate_rate']['mean']:.2f} "
                        f"ret={blk[p]['n_retest']['mean']:.2f}"
                    )
                if "paired_cover_minus_optimal" in blk:
                    print(
                        "    paired cover-opt",
                        {k: blk["paired_cover_minus_optimal"][k] for k in ("mean", "se", "n_pos", "n_neg", "n_tie")},
                    )


if __name__ == "__main__":
    main()
