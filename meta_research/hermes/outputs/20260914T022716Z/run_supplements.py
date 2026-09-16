#!/usr/bin/env python3
"""Post-repair supplements. New seeds only. Does not rewrite EVAL_*. """

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE / "conflict_experiment.py"


def load():
    spec = importlib.util.spec_from_file_location("ce", CODE)
    ce = importlib.util.module_from_spec(spec)
    sys.modules["ce"] = ce
    spec.loader.exec_module(ce)
    return ce


def slim(rows):
    return [{k: r[k] for k in r if k != "traj"} for r in rows]


def eval_n(ce, split, n_pairs, seed0, budgets=None):
    budgets = budgets or ce.BUDGETS
    rows = []
    for serial in range(n_pairs):
        for task in ce.make_pair(split, serial, seed0):
            for budget in budgets:
                for name in ce.POLICIES:
                    rows.append(ce.run_policy(task, name, budget))
    return rows, ce.summarize(rows, split)


def run_costs(ce, obs, ret, exp, n_pairs, seed0, tag):
    old = (ce.COST_OBS, ce.COST_RETEST, ce.COST_EXPAND)
    ce.COST_OBS, ce.COST_RETEST, ce.COST_EXPAND = obs, ret, exp
    rows, sm = eval_n(ce, tag, n_pairs, seed0)
    ce.COST_OBS, ce.COST_RETEST, ce.COST_EXPAND = old
    sm["costs"] = {"obs": obs, "retest": ret, "expand": exp}
    return rows, sm


def make_pair_alt_conflict(ce, split, serial, seed0, xc: int):
    rng = np.random.default_rng(seed0 + serial)
    for _ in range(8000):
        a = int(rng.choice(ce.A_RANGE))
        b = int(rng.choice(ce.B_RANGE))
        f0 = ce.h0_row(a, b)
        delta = int(rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))
        polluted = int(f0[xc] + delta)
        init_obs = {int(x): int(f0[x]) for x in ce.INIT_TRUE}
        init_obs[int(xc)] = polluted
        if ce.mask_consistent(ce.H0, init_obs).any():
            continue
        f1 = ce.find_piecewise_matching(init_obs)
        if f1 is None or np.array_equal(f1, f0):
            continue
        pair_id = f"{split}-pair-{serial:03d}"
        return [
            ce.Task(f"{pair_id}-model", pair_id, "model", f1, dict(init_obs), int(xc), polluted, split),
            ce.Task(f"{pair_id}-transient", pair_id, "transient", f0, dict(init_obs), int(xc), polluted, split),
            ce.Task(f"{pair_id}-persistent", pair_id, "persistent", f0, dict(init_obs), int(xc), polluted, split),
        ]
    raise RuntimeError(f"alt pair failed {split} {serial} xc={xc}")


def main():
    ce = load()
    t0 = time.perf_counter()
    iso = ce.isolation_tests()
    if not iso["honest_all_ok"] or not iso["cheat_all_caught"]:
        raise SystemExit(iso)

    out = {
        "code_sha256": hashlib.sha256(CODE.read_bytes()).hexdigest(),
        "isolation": iso,
        "note": "post-repair deliver(); new seeds; EVAL_* not reused as confirmation",
    }

    confirm_rows = json.loads((HERE / "CONFIRM_ROWS.json").read_text())
    extremes = []
    for budget in (8, 16):
        for cause in ("model", "transient", "persistent"):
            pairs = {}
            for r in confirm_rows:
                if r["budget"] == budget and r["cause"] == cause:
                    pairs.setdefault(r["pair"], {})[r["policy"]] = r
            deltas = []
            for pid, m in pairs.items():
                d = m["conflict"]["l1"] - m["retest_first"]["l1"]
                deltas.append((d, pid, {k: m[k]["l1"] for k in m}))
            deltas.sort()
            extremes.append(
                {
                    "budget": budget,
                    "cause": cause,
                    "conflict_minus_retest_best": deltas[:3],
                    "conflict_minus_retest_worst": deltas[-3:],
                }
            )
    out["extremes_from_confirm"] = extremes

    t1 = time.perf_counter()
    rows, sm = run_costs(ce, 1, 2, 2, 20, 50_000, "cost_retest2")
    (HERE / "SUPP_COST_RETEST2_ROWS.json").write_text(json.dumps(slim(rows)))
    out["cost_retest2"] = {
        "timing_s": time.perf_counter() - t1,
        "n_tasks": sm["n_tasks"],
        "n_distinct_truths": sm["n_distinct_truths"],
        "by_budget": sm["by_budget"],
        "costs": sm["costs"],
    }

    t1 = time.perf_counter()
    rows, sm = run_costs(ce, 1, 1, 1, 20, 50_000, "cost_expand1")
    (HERE / "SUPP_COST_EXPAND1_ROWS.json").write_text(json.dumps(slim(rows)))
    out["cost_expand1"] = {
        "timing_s": time.perf_counter() - t1,
        "n_tasks": sm["n_tasks"],
        "n_distinct_truths": sm["n_distinct_truths"],
        "by_budget": sm["by_budget"],
        "costs": sm["costs"],
    }

    t1 = time.perf_counter()
    rows, sm = eval_n(ce, "extra20", 20, 60_000)
    (HERE / "SUPP_EXTRA20_ROWS.json").write_text(json.dumps(slim(rows)))
    out["extra20"] = {
        "timing_s": time.perf_counter() - t1,
        "n_tasks": sm["n_tasks"],
        "n_distinct_truths": sm["n_distinct_truths"],
        "by_budget": sm["by_budget"],
    }

    t1 = time.perf_counter()
    rows = []
    for serial in range(20):
        for task in make_pair_alt_conflict(ce, "gen40", serial, 70_000, 40):
            for budget in ce.BUDGETS:
                for name in ce.POLICIES:
                    rows.append(ce.run_policy(task, name, budget))
    sm = ce.summarize(rows, "gen40")
    (HERE / "SUPP_GEN40_ROWS.json").write_text(json.dumps(slim(rows)))
    out["gen40"] = {
        "timing_s": time.perf_counter() - t1,
        "n_tasks": sm["n_tasks"],
        "n_distinct_truths": sm["n_distinct_truths"],
        "by_budget": sm["by_budget"],
        "note": "ASSUMED generator change: CONFLICT_X=40, INIT_TRUE unchanged",
    }

    out["elapsed_s"] = time.perf_counter() - t0
    (HERE / "SUPPLEMENT_POSTREPAIR.json").write_text(json.dumps(out, indent=2))
    print("iso", iso)
    for tag in ("cost_retest2", "cost_expand1", "extra20", "gen40"):
        print("==", tag, "==")
        blocksrc = out[tag]["by_budget"]["8"]
        for cause in ("model", "transient", "persistent"):
            print(" ", cause)
            for pol in ce.POLICIES:
                l1 = blocksrc[cause][pol]["l1"]
                print(
                    f"    {pol:16s} L1={l1['mean']:.2f}±{l1['se']:.2f} "
                    f"exp={blocksrc[cause][pol]['n_expand']['mean']:.2f} "
                    f"ret={blocksrc[cause][pol]['n_retest']['mean']:.2f}"
                )
            print(
                "    C-R",
                {k: blocksrc[cause]["paired_l1_conflict_minus_retest"][k] for k in ("mean", "se", "n_pos", "n_neg", "n_tie")},
            )
    print("elapsed", out["elapsed_s"])


if __name__ == "__main__":
    main()
