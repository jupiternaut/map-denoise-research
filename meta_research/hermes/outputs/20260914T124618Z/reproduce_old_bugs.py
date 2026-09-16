#!/usr/bin/env python3
"""Reproduce the two comparison bugs on the frozen 20260914T111013Z code.

Does not modify that directory.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OLD = Path("/home/grf/.hermes/attachments/outputs/20260914T111013Z")
sys.path.insert(0, str(OLD))
import exact_experiment as e


def catalog_worlds():
    return e.build_catalog(e.CATALOG_BUNDLES, e.CATALOG_SEED0, "catalog", 0)


def holdout_worlds():
    return e.build_catalog(e.HOLDOUT_BUNDLES, e.HOLDOUT_SEED0, "holdout", 1000)


def run_horizon(world, dp, horizon, fallback_cover: bool):
    env = e.TapeEnv(world)
    e.apply_init(env)
    empty_at = None
    while env.cost < horizon:
        rem = horizon - env.cost
        ids = frozenset(
            w.wid
            for w in dp.catalog
            if all(
                (w.retest(x) if x in env.retested else w.observe(x)) == env.cur[x]
                for x in env.orig
            )
        )
        if not ids and empty_at is None:
            empty_at = env.cost
        if not ids:
            if not fallback_cover:
                break
            a, x = "query", e.next_cover(env.orig)
            if x is None:
                break
        else:
            _, act = dp.value(ids, frozenset(env.orig), frozenset(env.retested), rem)
            if act is None:
                if fallback_cover:
                    a, x = "query", e.next_cover(env.orig)
                    if x is None:
                        break
                else:
                    break
            else:
                a, x = act
        if a == "query":
            env.query(int(x))
        elif a == "retest":
            env.retest(int(x))
        else:
            break
    pred, _ = e.new_deliver(env.believed())
    return e.mae_of(pred, world.truth), env.cost, empty_at


def history_mismatch_example(catalog):
    """Find a world where retest changes a label and cur-only vs tape differ."""
    for w in catalog:
        env = e.TapeEnv(w)
        e.apply_init(env)
        dirty = w.dirty_x
        if dirty not in env.orig:
            continue
        y0 = env.orig[dirty]
        y2 = env.retest(dirty)
        if y2 == y0:
            continue
        tape_ids = []
        cur_ids = []
        for u in catalog:
            ok_cur = all(
                (u.retest(x) if x in env.retested else u.observe(x)) == env.cur[x]
                for x in env.orig
            )
            ok_tape = True
            for x in env.orig:
                if x == dirty:
                    if int(u.observe(x)) != int(y0):
                        ok_tape = False
                    if int(u.retest(x)) != int(y2):
                        ok_tape = False
                else:
                    y = u.retest(x) if x in env.retested else u.observe(x)
                    if y != env.cur[x]:
                        ok_tape = False
            if ok_cur:
                cur_ids.append(u.wid)
            if ok_tape:
                tape_ids.append(u.wid)
        return {
            "wid": w.wid,
            "cause": w.cause,
            "orig": {int(k): int(v) for k, v in env.orig.items()},
            "cur": {int(k): int(v) for k, v in env.cur.items()},
            "dirty_x": int(dirty),
            "y0": int(y0),
            "y2": int(y2),
            "n_cur": len(cur_ids),
            "n_tape": len(tape_ids),
            "cur_ids": cur_ids,
            "tape_ids": tape_ids,
        }
    return None


def main():
    catalog = catalog_worlds()
    holdout = holdout_worlds()
    dp = e.ExactDP(catalog)
    env0 = e.TapeEnv(catalog[0])
    e.apply_init(env0)
    ids0 = frozenset(
        u.wid for u in catalog if all(u.observe(x) == env0.cur[x] for x in env0.orig)
    )
    dp.value(ids0, frozenset(env0.orig), frozenset(), 4)
    dp.value(ids0, frozenset(env0.orig), frozenset(), 6)

    by = defaultdict(lambda: {"cover4": [], "slice6at4": [], "dp4": []})
    for w in catalog:
        env = e.TapeEnv(w)
        e.apply_init(env)
        while env.cost < 4:
            x = e.next_cover(env.orig)
            if x is None:
                break
            env.query(x)
        pred, _ = e.new_deliver(env.believed())
        cover4 = e.mae_of(pred, w.truth)

        env = e.TapeEnv(w)
        e.apply_init(env)
        while env.cost < 4:
            rem = 6 - env.cost
            ids = frozenset(
                u.wid
                for u in dp.catalog
                if all(
                    (u.retest(x) if x in env.retested else u.observe(x)) == env.cur[x]
                    for x in env.orig
                )
            )
            _, act = dp.value(ids, frozenset(env.orig), frozenset(env.retested), rem)
            if act is None:
                break
            a, x = act
            if a == "query":
                env.query(int(x))
            else:
                env.retest(int(x))
        pred, _ = e.new_deliver(env.believed())
        slice6at4 = e.mae_of(pred, w.truth)
        dp4, _, _ = run_horizon(w, dp, 4, fallback_cover=False)
        by[w.cause]["cover4"].append(cover4)
        by[w.cause]["slice6at4"].append(slice6at4)
        by[w.cause]["dp4"].append(dp4)

    print("=== catalog horizon-4 vs slice of horizon-6 ===")
    n_zero = 0
    n = 0
    for cause in e.CAUSES:
        d = by[cause]
        print(
            cause,
            "cover4",
            float(np.mean(d["cover4"])),
            "slice6@4",
            float(np.mean(d["slice6at4"])),
            "dp4",
            float(np.mean(d["dp4"])),
            "n_dp4_zero",
            sum(x == 0 for x in d["dp4"]),
            "/",
            len(d["dp4"]),
        )
        n += len(d["dp4"])
        n_zero += sum(x == 0 for x in d["dp4"])
    print("dp4 zeros", n_zero, "/", n)

    print("=== holdout empty-catalog stop vs cover fallback ===")
    empty_at_init = 0
    empty_later = 0
    empty_never = 0
    mae_stop = defaultdict(list)
    mae_fb = defaultdict(list)
    for w in holdout:
        m_stop, cost_stop, empty_at = run_horizon(w, dp, 6, fallback_cover=False)
        m_fb, _, _ = run_horizon(w, dp, 6, fallback_cover=True)
        mae_stop[w.cause].append(m_stop)
        mae_fb[w.cause].append(m_fb)
        if empty_at is None:
            empty_never += 1
        elif empty_at == 2:
            empty_at_init += 1
        else:
            empty_later += 1
            print("empty_later", w.wid, w.cause, "at", empty_at)
    print("empty_at_init", empty_at_init, "empty_later", empty_later, "never", empty_never)
    for cause in e.CAUSES:
        print(
            cause,
            "stop",
            float(np.mean(mae_stop[cause])),
            "fallback",
            float(np.mean(mae_fb[cause])),
        )

    ex = history_mismatch_example(catalog)
    print("=== history mismatch example ===")
    print(ex)


if __name__ == "__main__":
    main()
