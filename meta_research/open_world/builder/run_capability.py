#!/usr/bin/env python3
"""Capability trial: same primitives, three search modes, history vs none.

Construction six is smoke. These six T-systems are the score of this round.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from builder.capability_tasks import tasks
from host.artifact import FAMILY_SPEC
from host.score import score_candidate
from host.session import make_session
from policies.capability import covering, policy_library, policy_propose


OUT = ROOT / "outputs" / "capability_v1"


def policy_named(session) -> Dict:
    covering(session)
    ids = list(session.experiments.keys())
    best = None
    best_err = None
    for fam in FAMILY_SPEC:
        rec = session.fit(fam, ids)
        if best_err is None or rec["train_rmse"] < best_err:
            best, best_err = rec, rec["train_rmse"]
    session.submit_final(best["candidate_hash"])
    best["policy"] = "named4"
    return best


POLICIES = {
    "named4": policy_named,
    "library": policy_library,
    "propose": policy_propose,
}


def run_one(task: Dict, name: str, fn: Callable) -> Dict:
    sess = make_session(task["spec"], init_mem=0.0)
    t0 = time.perf_counter()
    info = fn(sess)
    elapsed = time.perf_counter() - t0
    cand = sess.candidates[sess.final_hash]
    scored_h = score_candidate(sess, sess.final_hash, use_history=True)
    scored_0 = score_candidate(sess, sess.final_hash, use_history=False)
    named_keys = {
        tuple(v["acc_terms"]) + v["hidden"] + (v["obs_map"],) for v in FAMILY_SPEC.values()
    }
    key = tuple(cand.acc_terms) + cand.hidden + (cand.obs_map,)
    return {
        "system_id": task["system_id"],
        "mismatch": task["mismatch"],
        "policy": name,
        "family": cand.family,
        "acc_terms": list(cand.acc_terms),
        "hidden": list(cand.hidden),
        "obs_map": cand.obs_map,
        "in_named4": key in named_keys,
        "use_memory": bool(cand.use_memory),
        "nmse_history": scored_h["nmse_mean"],
        "nmse_no_history": scored_0["nmse_mean"],
        "branch_pred_gap_history": scored_h["branch_pred_gap"],
        "branch_pred_gap_no_history": scored_0["branch_pred_gap"],
        "nmse_by_item_history": scored_h["nmse_by_item"],
        "nmse_by_item_no_history": scored_0["nmse_by_item"],
        "cost": sess.cost,
        "n_fit": sess.n_fit,
        "wall_s": elapsed,
        "train_rmse": info.get("train_rmse"),
    }


def main() -> None:
    t0 = time.perf_counter()
    rows = []
    for task in tasks():
        for name, fn in POLICIES.items():
            print(f"run {task['system_id']} {name}", flush=True)
            rows.append(run_one(task, name, fn))
    by: Dict = {}
    for r in rows:
        by.setdefault(r["system_id"], {})[r["policy"]] = r
    summary = []
    for task in tasks():
        block = {"system_id": task["system_id"], "mismatch": task["mismatch"], "policies": {}}
        for name in POLICIES:
            r = by[task["system_id"]][name]
            block["policies"][name] = {
                "nmse_history": r["nmse_history"],
                "nmse_no_history": r["nmse_no_history"],
                "family": r["family"],
                "in_named4": r["in_named4"],
                "use_memory": r["use_memory"],
                "branch_gap_h": r["branch_pred_gap_history"],
                "branch_gap_0": r["branch_pred_gap_no_history"],
            }
        h = {n: block["policies"][n]["nmse_history"] for n in POLICIES}
        strongest = min(h, key=h.get)
        block["strongest"] = strongest
        block["library_minus_named4"] = h["library"] - h["named4"]
        block["propose_minus_library"] = h["propose"] - h["library"]
        block["history_gain_library"] = (
            by[task["system_id"]]["library"]["nmse_no_history"]
            - by[task["system_id"]]["library"]["nmse_history"]
        )
        summary.append(block)
    locked = {
        "kind": "capability_v1",
        "construction_six": "smoke_only_not_this_score",
        "trusted_algorithm_experiment": True,
        "os_sandbox": False,
        "same_primitives": True,
        "true_m0_never_an_argument": True,
        "holdout_visible_reset": True,
        "code_sha256_artifact": hashlib.sha256((ROOT / "host" / "artifact.py").read_bytes()).hexdigest(),
        "timing_s": time.perf_counter() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ROWS.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "SUMMARY.json").write_text(json.dumps({"protocol": locked, "systems": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("elapsed", round(locked["timing_s"], 3))


if __name__ == "__main__":
    main()
