#!/usr/bin/env python3
"""How large is catalog support after a typical init?"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path("/home/grf/.hermes/attachments/outputs/20260914T132630Z")
sys.path.insert(0, str(HERE))
import enum_probe as ep

OLD = Path("/home/grf/.hermes/attachments/outputs/20260914T111013Z")
sys.path.insert(0, str(OLD))
import exact_experiment as e

HZ = Path("/home/grf/.hermes/attachments/outputs/20260914T124618Z")
sys.path.insert(0, str(HZ))
import horizon_experiment as h


def worlds_from_instances(inst, split="enum", wid0=0):
    worlds = []
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


def main():
    inst = ep.enumerate_instances()
    worlds = worlds_from_instances(inst)
    print("n_worlds", len(worlds))
    sizes = []
    empty = 0
    seen_init = set()
    for w in worlds:
        key = tuple(sorted(w.orig_init.items()))
        if key in seen_init:
            continue
        seen_init.add(key)
        env = e.TapeEnv(w)
        e.apply_init(env)
        ids = h.filter_ids(worlds, env)
        sizes.append(len(ids))
        if not ids:
            empty += 1
    print("n_unique_init_checked", len(sizes), "empty", empty)
    print("support min/median/mean/max", min(sizes), int(np.median(sizes)), float(np.mean(sizes)), max(sizes))
    print("hist", Counter(sizes).most_common(12))

    confirm_empty_init = 0
    confirm_sizes = []
    for serial in range(30):
        bun = e.make_bundle("confirm_probe", serial, 30_000, 5000 + 3 * serial)
        env = e.TapeEnv(bun[0])
        e.apply_init(env)
        ids = h.filter_ids(worlds, env)
        confirm_sizes.append(len(ids))
        if not ids:
            confirm_empty_init += 1
    print("confirm_sample_n", 30, "empty_after_init", confirm_empty_init, "sizes", confirm_sizes)


if __name__ == "__main__":
    main()
