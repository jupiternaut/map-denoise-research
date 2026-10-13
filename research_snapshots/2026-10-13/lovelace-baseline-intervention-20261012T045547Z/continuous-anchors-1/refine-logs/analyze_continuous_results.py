"""Post-evaluation descriptive analysis; never invokes a formal solver."""
import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from fractions import Fraction as F
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "continuous-v1"


def load(path):
    return json.loads(path.read_text())


def metrics(rows):
    n = len(rows)
    a = [F(r["initial_ae_mm"]) for r in rows]
    b = [F(r["final_ae_mm"]) for r in rows]
    mae0 = sum(a, F()) / n
    mae1 = sum(b, F()) / n
    mse0 = sum((x*x for x in a), F()) / n
    mse1 = sum((x*x for x in b), F()) / n
    return {
        "n": n, "initial_MAE_mm": str(mae0), "final_MAE_mm": str(mae1),
        "initial_MSE_mm2": str(mse0), "final_MSE_mm2": str(mse1),
        "MAE_reduction_percent": float(100 * (mae0-mae1) / mae0) if mae0 else None,
        "MSE_reduction_percent": float(100 * (mse0-mse1) / mse0) if mse0 else None,
        "improved": sum(y < x for x, y in zip(a, b)),
        "equal": sum(y == x for x, y in zip(a, b)),
        "worsened": sum(y > x for x, y in zip(a, b)),
        "moved": sum(r["moved"] for r in rows),
    }


def widths(rows):
    values = [F(r["hull_width_mm"]) for r in rows]
    return {"n": len(rows), "mean_mm": str(sum(values, F())/len(values)),
            "median_mm": str(statistics.median(values)), "max_mm": str(max(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", required=True)
    args = parser.parse_args()
    assert load(OUT / "STAGE_STATUS.json")["status"] == "COMPLETE"
    assert load(OUT / "verification" / "VERIFY.json")["status"] == "PASS"
    raw = load(OUT / "evaluation" / "ROWS.json")
    rows = [r for r in raw if r["policy"] == "HULL"]
    stats = load(OUT / "evaluation" / "OUTER_STATS.json")
    arms = ["none", "camera", "appearance", "both"]
    summaries = load(OUT / "evaluation" / "SUMMARY.json")
    grouped = {}
    by_case = {}
    outer = {}
    observation_gains = {}
    paired_gains = {}
    improved = {}
    for arm in arms:
        selected = [r for r in rows if r["arm"] == arm]
        grouped[arm] = {}
        for group in ["all", "shifted", "correct", "informative", "negative", "primary"]:
            g = [r for r in selected if group in r["groups"]]
            m = metrics(g)
            stored = next(s for s in summaries if s["arm"] == arm and s["policy"] == "HULL"
                          and s["unit"] == "ALL" and s["group"] == group)
            for key in ["n", "initial_MAE_mm", "final_MAE_mm", "initial_MSE_mm2",
                        "final_MSE_mm2", "improved", "equal", "worsened", "moved"]:
                assert m[key] == stored[key], (arm, group, key)
            grouped[arm][group] = m
        by_case[arm] = {case: metrics([r for r in selected if r["case"] == case and "shifted" in r["groups"]])
                       for case in sorted({r["case"] for r in selected})}
        kind = {r["observation_id"]: ("negative" if "negative" in r["groups"] else "informative")
                for r in selected if "shifted" in r["groups"]}
        ss = [s for s in stats if s["arm"] == arm]
        outer[arm] = {"all": widths(ss)}
        for group in ["negative", "informative"]:
            outer[arm][group] = widths([s for s in ss if kind[s["observation_id"]] == group])
        outer[arm].update({k: sum(s[k] for s in ss) for k in
                           ["classified_nodes", "pruned_nodes", "retained_deadline",
                            "retained_node_budget", "unknown_classifications", "seconds"]})
        outer[arm]["components_total"] = sum(s["components"] for s in ss)
        obs = defaultdict(F)
        pairs = defaultdict(F)
        for r in selected:
            gain = F(r["ae_gain_mm"])
            obs[r["observation_id"]] += gain
            # variation_index is eta (0/1), not the acquisition realization.
            pairs[r["observation_id"].rsplit("e", 1)[0]] += gain
        assert len(obs) == 36 and len(pairs) == 18
        count = lambda d: {"improved": sum(v > 0 for v in d.values()),
                           "equal": sum(v == 0 for v in d.values()),
                           "worsened": sum(v < 0 for v in d.values())}
        observation_gains[arm] = count(obs)
        paired_gains[arm] = count(pairs)
        improved[arm] = {r["task_id"] for r in selected if F(r["ae_gain_mm"]) > 0}
    singles = improved["camera"] | improved["appearance"]
    hull_map = {(r["task_id"], r["arm"]): r["selected_mm"] for r in rows}
    union_map = {(r["task_id"], r["arm"]): r["selected_mm"] for r in raw if r["policy"] == "UNION"}
    assert hull_map == union_map
    assert all(s["components"] == 1 for s in stats)
    data = {
        "stage": "A9", "kind": "descriptive post-evaluation, no new formal run",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "statistical_units": {"observations": 36, "paired_realizations": 18,
                              "incumbents": 324, "shifted_incumbents": 288, "primary_conditions": 40},
        "metric_groups": grouped, "case_shifted": by_case, "outer_sets": outer,
        "per_observation_gain_counts": observation_gains, "paired_gain_counts": paired_gains,
        "joint_task_set": {"single_arm_union": len(singles), "joint_improved": len(improved["both"]),
                           "joint_only": len(improved["both"]-singles),
                           "single_improvement_lost": len(singles-improved["both"]),
                           "joint_only_task_ids": sorted(improved["both"]-singles)},
        "HULL_UNION_identical_pairs": len(hull_map),
        "remaining_partition": {"informative_unchanged": grouped["both"]["informative"]["equal"],
                                "negative_unchanged": grouped["both"]["negative"]["equal"],
                                "cause_not_identified": True},
        "source_files": ["evaluation/ROWS.json", "evaluation/SUMMARY.json", "evaluation/OUTER_STATS.json"],
    }
    encoded = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    versioned = ROOT / f"ANALYSIS_{args.stamp}.json"
    if versioned.exists():
        raise FileExistsError(versioned)
    versioned.write_text(encoded)
    (ROOT / "ANALYSIS.json").write_text(encoded)
    print(json.dumps({"status": "PASS", "shifted": grouped["both"]["shifted"],
                      "outer_widths": outer["both"], "joint_task_set": data["joint_task_set"],
                      "versioned": str(versioned)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
