#!/usr/bin/env python3
"""Enumerate feasible worlds under the public generator (no sampling)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OLD = Path("/home/grf/.hermes/attachments/outputs/20260914T111013Z")
sys.path.insert(0, str(OLD))
import exact_experiment as e


def enumerate_instances():
    inst = []
    for ia in range(len(e.H0)):
        f0 = e.H0[ia].copy()
        a = int(f0[1] - f0[0])
        b = int(f0[0])
        for t in e.T_RANGE:
            left = e.DOMAIN < int(t)
            right = ~left
            if int(left.sum()) < 2 or int(right.sum()) < 2:
                continue
            for a2 in e.A_RANGE:
                for b2 in e.B_RANGE:
                    if int(a2) == a and int(b2) == b:
                        continue
                    f1 = np.empty(e.N, dtype=np.int32)
                    f1[left] = a * e.DOMAIN[left] + b
                    f1[right] = int(a2) * e.DOMAIN[right] + int(b2)
                    if e.in_library(e.H0, f1) or not e.in_library(e.H1, f1):
                        continue
                    for dirty_x in e.DOMAIN[right]:
                        dirty_x = int(dirty_x)
                        dirty_y = int(f1[dirty_x])
                        if dirty_y == int(f0[dirty_x]):
                            continue
                        for c1 in e.DOMAIN[left]:
                            c1 = int(c1)
                            init_set = {c1, dirty_x}
                            if init_set == e.FORBIDDEN_INIT or len(init_set) < 2:
                                continue
                            init_obs = {c1: int(f0[c1]), dirty_x: int(dirty_y)}
                            if int(f1[c1]) != init_obs[c1]:
                                continue
                            if e.mask_obs(e.H0, init_obs).any() or not e.mask_obs(e.H1, init_obs).any():
                                continue
                            inst.append(
                                {
                                    "f0": f0.copy(),
                                    "f1": f1.copy(),
                                    "init_obs": dict(init_obs),
                                    "dirty_x": dirty_x,
                                    "dirty_y": dirty_y,
                                    "t": int(t),
                                    "c1": c1,
                                }
                            )
    return inst


def main():
    inst = enumerate_instances()
    print("n_instances", len(inst))
    print("n_worlds", 3 * len(inst))
    inits = [tuple(sorted(d["init_obs"].items())) for d in inst]
    print("n_unique_init", len(set(inits)))
    print("n_unique_dirty", len({d["dirty_x"] for d in inst}))
    print("n_unique_t", len({d["t"] for d in inst}))
    print("n_unique_f0", len({tuple(d["f0"].tolist()) for d in inst}))
    print("n_unique_f1", len({tuple(d["f1"].tolist()) for d in inst}))
    rng_hits = 0
    rng_tot = 0
    for serial in range(20):
        bun = e.make_bundle("probe", serial, 20_000, 0)
        rng_tot += 1
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
        rng_hits += int(hit)
    print("sampled_in_enum", rng_hits, "/", rng_tot)


if __name__ == "__main__":
    main()
