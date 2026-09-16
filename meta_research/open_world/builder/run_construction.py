#!/usr/bin/env python3
"""Run construction tasks through A/B/C/D + ridge. Trusted algorithm experiment."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from builder.tasks import tasks
from host.score import score_candidate
from host.session import make_session
from policies.four import POLICIES
from policies.nonsym import policy_ridge


OUT = ROOT / "outputs" / "construction"


def run_one(task: Dict, name: str, policy_fn: Callable) -> Dict:
    sess = make_session(task["spec"], init_mem=task["init_mem"])
    t0 = time.perf_counter()
    info = policy_fn(sess)
    elapsed = time.perf_counter() - t0
    scored = score_candidate(sess, sess.final_hash)
    n_commit = len(sess.commitments)
    n_resolved = sum(1 for c in sess.commitments if c.get("resolved"))
    realized = [c.get("realized_mse") for c in sess.commitments if c.get("resolved")]
    cand = sess.candidates[sess.final_hash]
    return {
        "system_id": task["system_id"],
        "mismatch": task["mismatch"],
        "policy": name,
        "final_hash": sess.final_hash,
        "family": cand.family,
        "use_memory": bool(getattr(cand, "use_memory", False)),
        "use_drift": bool(getattr(cand, "use_drift", False)),
        "nmse_mean": scored["nmse_mean"],
        "scale": scored["scale"],
        "nmse_by_item": scored["nmse_by_item"],
        "cost": sess.cost,
        "n_run": sess.n_run,
        "n_repeat": sess.n_repeat,
        "n_calibrate": sess.n_calibrate,
        "n_fit": sess.n_fit,
        "n_commit": n_commit,
        "n_commit_resolved": n_resolved,
        "mean_realized_commit_mse": float(sum(realized) / len(realized)) if realized else None,
        "wall_s": elapsed,
        "init_mem": task["init_mem"],
        "train_family_rmse": {
            k: info.get("fits", {}).get(k, {}).get("train_rmse")
            for k in ("linear2", "nl2", "memory3", "obs_drift")
        }
        if isinstance(info, dict)
        else {},
    }


def main() -> None:
    t0 = time.perf_counter()
    ts = tasks()
    rows = []
    for task in ts:
        for name, fn in list(POLICIES.items()) + [("ridge", policy_ridge)]:
            print(f"run {task['system_id']} {name}", flush=True)
            rows.append(run_one(task, name, fn))
    by: Dict = {}
    for r in rows:
        by.setdefault(r["system_id"], {})[r["policy"]] = r
    summary = []
    for task in ts:
        block = {"system_id": task["system_id"], "mismatch": task["mismatch"], "policies": {}}
        for name in ("A", "B", "C", "D", "ridge"):
            r = by[task["system_id"]][name]
            block["policies"][name] = {
                "nmse": r["nmse_mean"],
                "family": r["family"],
                "cost": r["cost"],
                "n_commit": r["n_commit"],
            }
        nmses = {n: block["policies"][n]["nmse"] for n in ("A", "B", "C", "D", "ridge")}
        strongest = min(nmses, key=nmses.get)
        block["strongest"] = strongest
        block["D_minus_C"] = nmses["D"] - nmses["C"]
        block["D_minus_A"] = nmses["D"] - nmses["A"]
        block["C_minus_A"] = nmses["C"] - nmses["A"]
        block["B_minus_A"] = nmses["B"] - nmses["A"]
        summary.append(block)
    locked = {
        "kind": "construction_v0",
        "trusted_algorithm_experiment": True,
        "os_sandbox": False,
        "pysindy": False,
        "pysr": False,
        "llm_aces_official": "not runnable here (needs pysr + LLM API)",
        "code_sha256_session": hashlib.sha256((ROOT / "host" / "session.py").read_bytes()).hexdigest(),
        "holdout_menu_locked": True,
        "max_actions": 8,
        "timing_s": time.perf_counter() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ROWS.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "SUMMARY.json").write_text(
        json.dumps({"protocol": locked, "systems": summary}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print("elapsed", round(locked["timing_s"], 3))


if __name__ == "__main__":
    main()
