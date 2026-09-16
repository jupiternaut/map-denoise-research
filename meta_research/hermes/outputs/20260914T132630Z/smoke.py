#!/usr/bin/env python3
"""Smoke: history, isolation, openloop frozen, adaptive <= openloop."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path("/home/grf/.hermes/attachments/outputs/20260914T132630Z")
HZ = Path("/home/grf/.hermes/attachments/outputs/20260914T124618Z")
OLD = Path("/home/grf/.hermes/attachments/outputs/20260914T111013Z")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HZ))
sys.path.insert(0, str(OLD))
import adapt_experiment as a  # noqa: E402
import enum_probe as ep  # noqa: E402
import exact_experiment as e  # noqa: E402
import horizon_experiment as h  # noqa: E402


def main() -> None:
    inst = ep.enumerate_instances()
    catalog = a.worlds_from_instances(inst[:40], split="enum", wid0=0)
    index = a.CatalogIndex(catalog)
    dp = a.FastDP(catalog)

    hist = h.history_unit(catalog)
    print("history_unit", hist["ok"], hist["example"])
    if not hist["ok"]:
        raise SystemExit("history unit failed")

    env = e.TapeEnv(catalog[0])
    e.apply_init(env)
    ids = index.filter_env(env)
    tape = h.filter_ids(catalog, env)
    if ids != tape:
        raise SystemExit(f"index filter != tape filter {len(ids)} {len(tape)}")

    unit = a.unit_adapt_leq_openloop(catalog, dp, index)
    print("unit_adapt_leq_openloop", unit)
    if not unit["ok"]:
        raise SystemExit("adaptive > openloop")

    env2 = e.TapeEnv(catalog[0])
    e.apply_init(env2)
    ids2 = index.filter_env(env2)
    rows = np.array(sorted(ids2), dtype=np.int32)
    rem = 4
    seq, _, _ = a.plan_openloop(
        rows, index, frozenset(int(x) for x in env2.orig), frozenset(env2.retested), rem
    )
    if not seq:
        raise SystemExit("empty openloop plan")
    act0 = seq[0]
    if act0[0] == "query":
        env2.query(int(act0[1]))
    else:
        env2.retest(int(act0[1]))
    seq_rest, _, _ = a.plan_openloop(
        rows, index, frozenset(int(x) for x in env2.orig), frozenset(env2.retested), rem - 1
    )
    if seq_rest != seq[1:]:
        raise SystemExit(f"openloop not frozen: {seq} vs rest {seq_rest}")
    print("openloop_frozen", seq)

    iso = a.isolation_check(catalog, dp, index)
    print("iso", iso)
    if not (iso["honest_must_pass"] and iso["cheat_must_fail"]):
        raise SystemExit("isolation failed")

    print("SMOKE_OK")


if __name__ == "__main__":
    main()
