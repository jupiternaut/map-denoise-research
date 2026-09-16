#!/usr/bin/env python3
"""Independent audit: value-only posterior vs generator-init locations.

Does not modify 20260914T132630Z. Does not re-run H=6. Does not
upgrade the scheduler. Frozen imports only.

The public generator chooses init locations from the hidden world
(left point c1, dirty_x on the far side of the breakpoint). The locked
experiment filters remaining worlds by observed *values* at those
locations, not by whether a candidate would have generated the same
location pair.

This script re-solves H=4 open-loop under both observation models for
every unique init history, and executes both plans on the enumerated
catalog. Adaptive is re-solved only for the example init named in the
review, not for a new ranking.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
PREV = Path("/home/grf/.hermes/attachments/outputs/20260914T132630Z")
OLD = Path("/home/grf/.hermes/attachments/outputs/20260914T111013Z")
HZ = Path("/home/grf/.hermes/attachments/outputs/20260914T124618Z")
sys.path.insert(0, str(PREV))
sys.path.insert(0, str(HZ))
sys.path.insert(0, str(OLD))
import adapt_experiment as a  # noqa: E402
import enum_probe as ep  # noqa: E402
import exact_experiment as e  # noqa: E402

EXAMPLE_INIT = {2: 1, 5: -1}


def init_key(obs: Dict[int, int]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((int(x), int(y)) for x, y in obs.items()))


def loc_key(obs: Dict[int, int]) -> Tuple[int, ...]:
    return tuple(sorted(int(x) for x in obs))


def value_match(w: e.World, obs: Dict[int, int]) -> bool:
    return all(int(w.observe(int(x))) == int(y) for x, y in obs.items())


def gen_match(w: e.World, obs: Dict[int, int]) -> bool:
    return tuple(w.init_xs) == loc_key(obs) and value_match(w, obs)


def ids_of(worlds: Sequence[e.World], obs: Dict[int, int], mode: str) -> FrozenSet[int]:
    pred = value_match if mode == "value" else gen_match
    return frozenset(w.wid for w in worlds if pred(w, obs))


def execute_seq(world: e.World, seq: List[a.Action], horizon: int = 4) -> float:
    env = e.TapeEnv(world)
    e.apply_init(env)
    for act, x in seq:
        if env.cost >= horizon:
            break
        if act == "query":
            if int(x) in env.orig:
                break
            env.query(int(x))
        else:
            if int(x) not in env.orig or int(x) in env.retested:
                break
            env.retest(int(x))
    pred, _ = e.new_deliver(env.believed())
    return e.mae_of(pred, world.truth)


def adaptive_second_branch(
    dp: a.FastDP,
    ids0: FrozenSet[int],
    queried0: FrozenSet[int],
    rem: int,
) -> Dict:
    v0, act0 = dp.value(ids0, queried0, frozenset(), rem)
    if act0 is None:
        return {"v": float(v0), "first": None, "branches": []}
    a0, x0 = act0
    rem_w = [w for w in dp.catalog if w.wid in ids0]
    groups: Dict[int, List[e.World]] = {}
    for w in rem_w:
        y = int(w.observe(x0) if a0 == "query" else w.retest(x0))
        groups.setdefault(y, []).append(w)
    nq = queried0 | ({int(x0)} if a0 == "query" else queried0)
    nr = frozenset() if a0 == "query" else frozenset([int(x0)])
    branches = []
    for y, ws in sorted(groups.items()):
        nids = frozenset(u.wid for u in ws)
        v1, act1 = dp.value(nids, nq, nr, rem - 1)
        branches.append(
            {
                "y": int(y),
                "n": len(ws),
                "second": list(act1) if act1 else None,
                "v": float(v1),
            }
        )
    return {
        "v": float(v0),
        "first": list(act0),
        "branches": branches,
    }


def main() -> None:
    t0 = time.perf_counter()
    frozen_hash = hashlib.sha256((PREV / "adapt_experiment.py").read_bytes()).hexdigest()
    proto = json.loads((PREV / "PROTOCOL.json").read_text())
    if frozen_hash != proto["code_sha256"]:
        raise SystemExit("frozen checkpoint hash mismatch; refusing to audit")

    inst = ep.enumerate_instances()
    catalog = a.worlds_from_instances(inst, split="enum", wid0=0)
    index = a.CatalogIndex(catalog)
    by_id = {w.wid: w for w in catalog}

    unique: Dict[Tuple[Tuple[int, int], ...], Dict[int, int]] = {}
    worlds_by_init: Dict[Tuple[Tuple[int, int], ...], List[e.World]] = defaultdict(list)
    for w in catalog:
        k = init_key(w.orig_init)
        unique[k] = dict(w.orig_init)
        worlds_by_init[k].append(w)
    if len(unique) != 438:
        print("WARN unique_init", len(unique), "expected 438")

    rows = []
    n_plan_change = 0
    seq_cache: Dict = {}
    example_block = None

    for k, obs in unique.items():
        ids_v = ids_of(catalog, obs, "value")
        ids_g = ids_of(catalog, obs, "gen")
        q = frozenset(int(x) for x in obs)
        rem = 2  # H=4, init cost 2
        rows_v = np.array(sorted(ids_v), dtype=np.int32)
        rows_g = np.array(sorted(ids_g), dtype=np.int32)
        key_v = ("value", ids_v, q, rem)
        key_g = ("gen", ids_g, q, rem)
        if key_v not in seq_cache:
            seq_cache[key_v] = a.plan_openloop(rows_v, index, q, frozenset(), rem)
        if key_g not in seq_cache:
            seq_cache[key_g] = a.plan_openloop(rows_g, index, q, frozenset(), rem)
        seq_v, v_v, n_v = seq_cache[key_v]
        seq_g, v_g, n_g = seq_cache[key_g]
        changed = seq_v != seq_g
        n_plan_change += int(changed)
        rec = {
            "init": [[int(x), int(y)] for x, y in k],
            "n_value": len(ids_v),
            "n_gen": len(ids_g),
            "n_catalog_worlds_with_this_init": len(worlds_by_init[k]),
            "seq_value": [list(s) for s in seq_v],
            "seq_gen": [list(s) for s in seq_g],
            "planned_mae_value": float(v_v),
            "planned_mae_gen": float(v_g),
            "plan_changed": bool(changed),
            "true_world_in_value": all(w.wid in ids_v for w in worlds_by_init[k]),
            "true_world_in_gen": all(w.wid in ids_g for w in worlds_by_init[k]),
        }
        rows.append(rec)
        if dict(obs) == EXAMPLE_INIT or k == init_key(EXAMPLE_INIT):
            example_block = rec

    # Execute both H=4 open-loop plans on every catalog world.
    mae_value = []
    mae_gen = []
    residual_value = []
    residual_gen = []
    for w in catalog:
        k = init_key(w.orig_init)
        rec = next(r for r in rows if tuple(tuple(p) for p in r["init"]) == k)
        seq_v = [tuple(s) for s in rec["seq_value"]]
        seq_g = [tuple(s) for s in rec["seq_gen"]]
        mv = execute_seq(w, seq_v, 4)
        mg = execute_seq(w, seq_g, 4)
        mae_value.append(mv)
        mae_gen.append(mg)
        if mv > 0:
            residual_value.append(
                {
                    "wid": w.wid,
                    "cause": w.cause,
                    "mae": mv,
                    "seq": [list(s) for s in seq_v],
                }
            )
        if mg > 0:
            residual_gen.append(
                {
                    "wid": w.wid,
                    "cause": w.cause,
                    "mae": mg,
                    "seq": [list(s) for s in seq_g],
                }
            )

    catalog_rows = json.loads((PREV / "CATALOG_ROWS.json").read_text())
    locked_ol = [r["openloop_h4_mae"] for r in catalog_rows]
    locked_ad = [r["adaptive_h4_mae"] for r in catalog_rows]
    locked_ol_h6 = [r["openloop_h6_mae"] for r in catalog_rows]
    locked_ad_h6 = [r["adaptive_h6_mae"] for r in catalog_rows]

    # Example adaptive branches on the *value-only* posterior used in the lock.
    example_adapt = None
    if example_block is not None:
        obs = {int(x): int(y) for x, y in example_block["init"]}
        ids_v = ids_of(catalog, obs, "value")
        dp = a.FastDP(catalog)
        example_adapt = adaptive_second_branch(
            dp, ids_v, frozenset(int(x) for x in obs), 2
        )
        example_adapt_gen = adaptive_second_branch(
            dp, ids_of(catalog, obs, "gen"), frozenset(int(x) for x in obs), 2
        )
    else:
        example_adapt_gen = None

    sizes_v = [r["n_value"] for r in rows]
    sizes_g = [r["n_gen"] for r in rows]
    changed_rows = [r for r in rows if r["plan_changed"]]

    out = {
        "frozen_code_sha256": frozen_hash,
        "n_unique_init": len(unique),
        "n_catalog": len(catalog),
        "example_init": EXAMPLE_INIT,
        "example": example_block,
        "example_adaptive_value_posterior": example_adapt,
        "example_adaptive_gen_posterior": example_adapt_gen,
        "n_init_with_plan_change": n_plan_change,
        "n_catalog_worlds_whose_init_plan_changed": int(
            sum(r["n_catalog_worlds_with_this_init"] for r in changed_rows)
        ),
        "support": {
            "value_min_median_mean_max": [
                int(min(sizes_v)),
                int(np.median(sizes_v)),
                float(np.mean(sizes_v)),
                int(max(sizes_v)),
            ],
            "gen_min_median_mean_max": [
                int(min(sizes_g)),
                int(np.median(sizes_g)),
                float(np.mean(sizes_g)),
                int(max(sizes_g)),
            ],
            "true_world_always_in_value": all(r["true_world_in_value"] for r in rows),
            "true_world_always_in_gen": all(r["true_world_in_gen"] for r in rows),
        },
        "h4_openloop_mae": {
            "locked_checkpoint": float(np.mean(locked_ol)),
            "reexecuted_value_filter": float(np.mean(mae_value)),
            "reexecuted_gen_filter": float(np.mean(mae_gen)),
            "n_residual_value": len(residual_value),
            "n_residual_gen": len(residual_gen),
            "residual_value_by_cause": dict(Counter(r["cause"] for r in residual_value)),
            "residual_gen_by_cause": dict(Counter(r["cause"] for r in residual_gen)),
            "residual_value_mae_hist": dict(
                Counter(round(r["mae"], 6) for r in residual_value)
            ),
            "residual_gen_mae_hist": dict(
                Counter(round(r["mae"], 6) for r in residual_gen)
            ),
        },
        "locked_h4_adaptive_mae": float(np.mean(locked_ad)),
        "locked_h6_openloop_mae": float(np.mean(locked_ol_h6)),
        "locked_h6_adaptive_mae": float(np.mean(locked_ad_h6)),
        "locked_h6_any_positive": bool(
            any(x > 0 for x in locked_ol_h6) or any(x > 0 for x in locked_ad_h6)
        ),
        "h4_delta_adapt_locked": float(np.mean(locked_ol) - np.mean(locked_ad)),
        "timing_s": time.perf_counter() - t0,
    }
    (HERE / "INIT_DESIGN_AUDIT.json").write_text(json.dumps(out, indent=2) + "\n")
    (HERE / "INIT_DESIGN_ROWS.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in out if k != "example_adaptive_gen_posterior"}, indent=2))
    print("example_adaptive_gen", example_adapt_gen)
    print("elapsed", round(out["timing_s"], 3))


if __name__ == "__main__":
    main()
