#!/usr/bin/env python3
"""Supplements. Do not mix with locked EVAL_SUMMARY conclusions."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conflict_experiment as c

CAUSES = c.CAUSES
POLICIES = c.POLICIES


def worst_best(rows):
    out = {}
    for cause in CAUSES:
        sub = [r for r in rows if r["cause"] == cause]
        deltas = []
        for r in sub:
            d = r["diagnose_mae16"] - r["expand_first_mae16"]
            deltas.append((d, r))
        deltas.sort(key=lambda z: z[0])
        out[cause] = {
            "best_for_diagnose_vs_expand": [
                {
                    "task": r["task"],
                    "delta_mae16": d,
                    "diagnose": r["diagnose_mae16"],
                    "expand": r["expand_first_mae16"],
                    "retest": r["retest_first_mae16"],
                }
                for d, r in deltas[:3]
            ],
            "worst_for_diagnose_vs_expand": [
                {
                    "task": r["task"],
                    "delta_mae16": d,
                    "diagnose": r["diagnose_mae16"],
                    "expand": r["expand_first_mae16"],
                    "retest": r["retest_first_mae16"],
                }
                for d, r in deltas[-3:]
            ],
        }
    return out


def run_costs(tag, n_bundles, seed0, costs):
    summary, _, _ = c.evaluate(tag, n_bundles, seed0, costs)
    compact = {}
    for cause in CAUSES:
        b16 = summary["by_cause"][cause]["budgets"]["16"]
        compact[cause] = {
            p: {
                "mae": b16[p]["mae"],
                "n_expand": b16[p]["n_expand"]["mean"],
                "n_retest": b16[p]["n_retest"]["mean"],
                "unresolved": b16[p]["unresolved_rate"]["mean"],
            }
            for p in POLICIES
        }
        compact[cause]["paired_retest_minus_diagnose"] = {
            k: b16["paired_mae_retest_minus_diagnose"][k]
            for k in ("mean", "se", "n_pos", "n_neg", "n_tie")
        }
        compact[cause]["paired_expand_minus_diagnose"] = {
            k: b16["paired_mae_expand_minus_diagnose"][k]
            for k in ("mean", "se", "n_pos", "n_neg", "n_tie")
        }
    return compact


def alt_bundle(split: str, serial: int, seed0: int):
    """Different generator: dirty_x=48, t=40. Same three hidden causes."""
    rng = np.random.default_rng(seed0 + serial)
    for _ in range(20_000):
        f0 = c.m.H0[int(rng.integers(0, len(c.m.H0)))].copy()
        a = int(f0[1] - f0[0])
        b = int(f0[0])
        t = 40
        a2 = int(rng.choice(c.m.A_RANGE))
        b2 = int(rng.choice(c.m.B_RANGE))
        if a2 == a and b2 == b:
            continue
        dirty_x = 48
        dirty_y = int(a2 * dirty_x + b2)
        if dirty_y == int(f0[dirty_x]):
            continue
        f1 = np.empty(c.N, dtype=np.int32)
        left = c.DOMAIN < t
        f1[left] = a * c.DOMAIN[left] + b
        f1[~left] = a2 * c.DOMAIN[~left] + b2
        if c.m.in_library(c.H0, f1) or not c.m.in_library(c.H1, f1):
            continue
        init_obs = {0: int(f0[0]), 32: int(f0[32]), 48: int(dirty_y)}
        if int(f1[0]) != init_obs[0] or int(f1[32]) != init_obs[32] or int(f1[48]) != dirty_y:
            continue
        if c.mask_obs(c.H0, init_obs).any() or not c.mask_obs(c.H1, init_obs).any():
            continue
        pair_id = f"{split}-alt{serial:03d}"
        out = []
        for typ, cause in enumerate(c.CAUSES):
            truth = f1 if cause == "model" else f0
            out.append(
                c.Task(
                    task_id=f"{pair_id}-{cause}",
                    typ=typ,
                    cause=cause,
                    truth=truth,
                    init_obs=dict(init_obs),
                    dirty_x=dirty_x,
                    dirty_y=dirty_y,
                    pair_id=pair_id,
                    split=split,
                )
            )
        return out
    raise RuntimeError("alt generator failed")


def evaluate_alt(n_bundles: int, seed0: int):
    tasks = []
    for serial in range(n_bundles):
        tasks.extend(alt_bundle("alt", serial, seed0))
    rows = []
    for task in tasks:
        recs = {p: c.run_policy(task, p) for p in POLICIES}
        rows.append(
            {
                "task": task.task_id,
                "cause": task.cause,
                "typ": task.typ,
                "pair_id": task.pair_id,
                "init_obs": {str(k): int(v) for k, v in task.init_obs.items()},
                "recs": recs,
            }
        )
    return c.summarize("alt", rows, {"query": 1, "retest": 1, "expand": 1})


def main():
    t0 = time.perf_counter()
    rows = json.loads((HERE / "EVAL_ROWS.json").read_text())
    cases = worst_best(rows)
    print("1 worst/best done")

    print("2 costs retest=2 expand=1")
    cost_r2 = run_costs("supp_r2", 20, 30_000, {"query": 1, "retest": 2, "expand": 1})
    print("2 costs retest=1 expand=2")
    cost_e2 = run_costs("supp_e2", 20, 30_000, {"query": 1, "retest": 1, "expand": 2})

    print("4 extra 20 bundles new seeds")
    extra, extra_rows, _ = c.evaluate("extra", 20, 40_000)
    extra_compact = {
        cause: {p: extra["by_cause"][cause]["budgets"]["16"][p]["mae"] for p in POLICIES}
        for cause in CAUSES
    }

    print("5 alt generator 20 bundles")
    alt = evaluate_alt(20, 50_000)
    alt_compact = {
        cause: {p: alt["by_cause"][cause]["budgets"]["16"][p]["mae"] for p in POLICIES}
        for cause in CAUSES
    }

    out = {
        "tag": "SUPPLEMENT_NOT_LOCKED_EVAL",
        "note": "Not mixed into EVAL_SUMMARY. Persistence contrast is the locked eval itself (transient vs persistent).",
        "worst_best": cases,
        "cost_retest2_expand1_n20": cost_r2,
        "cost_retest1_expand2_n20": cost_e2,
        "extra20_mae16": extra_compact,
        "alt_generator_dirty48_t40_mae16": alt_compact,
        "elapsed_s": time.perf_counter() - t0,
    }
    (HERE / "SUPPLEMENT.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote SUPPLEMENT.json", round(out["elapsed_s"], 3))


if __name__ == "__main__":
    main()
