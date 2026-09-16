#!/usr/bin/env python3
"""Time ExactDP on the largest post-init support set."""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path("/home/grf/.hermes/attachments/outputs/20260914T132630Z")
sys.path.insert(0, str(HERE))
import enum_probe as ep
import support_probe as sp

OLD = Path("/home/grf/.hermes/attachments/outputs/20260914T111013Z")
sys.path.insert(0, str(OLD))
import exact_experiment as e

HZ = Path("/home/grf/.hermes/attachments/outputs/20260914T124618Z")
sys.path.insert(0, str(HZ))
import horizon_experiment as h


def main():
    inst = ep.enumerate_instances()
    worlds = sp.worlds_from_instances(inst)
    best = None
    for w in worlds:
        env = e.TapeEnv(w)
        e.apply_init(env)
        ids = h.filter_ids(worlds, env)
        if best is None or len(ids) > len(best[0]):
            best = (ids, w, env)
    ids, w, env = best
    print("largest_support", len(ids), "init", env.orig, "cause", w.cause)
    dp = e.ExactDP(worlds)
    t0 = time.perf_counter()
    v, act = dp.value(ids, frozenset(env.orig), frozenset(env.retested), 4)
    print("H=6 rem4", v, act, "states", len(dp._memo), "s", round(time.perf_counter() - t0, 3))
    t0 = time.perf_counter()
    v2, act2 = dp.value(ids, frozenset(env.orig), frozenset(env.retested), 2)
    print("H=4 rem2", v2, act2, "states", len(dp._memo), "s", round(time.perf_counter() - t0, 3))


if __name__ == "__main__":
    main()
