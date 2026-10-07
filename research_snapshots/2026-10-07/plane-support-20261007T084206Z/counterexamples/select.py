"""Truth-blind candidate selection. Reads only protocol and observations."""

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def propagated_support(neighbor):
    b = neighbor["normal"][2]
    if b == 0:
        return {"depth": None, "radius": None, "finite": False}
    center = neighbor["offset"] / b
    margin = abs(b) - neighbor["eps_normal"]
    if margin <= 0:
        return {"depth": center, "radius": None, "finite": False}
    radius = (neighbor["eps_offset"] + abs(center) * neighbor["eps_normal"]) / margin
    return {"depth": center, "radius": radius, "finite": True}


def candidate_support(scene, protocol):
    supports = [propagated_support(n) for n in scene["neighbors"]]
    centers = [s["depth"] for s in supports if s["depth"] is not None and math.isfinite(s["depth"])]
    median = statistics.median(centers) if centers else None
    rows = []
    for candidate in scene["candidates"]:
        depth = candidate["depth"]
        interval = sum(not s["finite"] or abs(depth - s["depth"]) <= s["radius"] + 1e-12 for s in supports)
        fixed = sum(s["depth"] is not None and abs(depth - s["depth"]) <= protocol["fixed_radius_m"] + 1e-12 for s in supports)
        rows.append({**candidate, "interval_votes": interval, "fixed_radius_votes": fixed,
                     "median_distance": None if median is None else abs(depth - median)})
    return supports, rows


def select_scene(scene, protocol):
    supports, rows = candidate_support(scene, protocol)
    tie = lambda c: (c["photo_loss"], c["id"])
    threshold = len(scene["neighbors"]) - scene["declared_max_bad_neighbors"]
    qualified = [c for c in rows if c["interval_votes"] >= threshold]
    choices = {
        "photo": min(rows, key=tie),
        "median": min(rows, key=lambda c: (round(c["median_distance"], 12) if c["median_distance"] is not None else float("inf"), *tie(c))),
        "fixed_radius": min(rows, key=lambda c: (-c["fixed_radius_votes"], *tie(c))),
        "interval_vote": min(rows, key=lambda c: (-c["interval_votes"], *tie(c))),
        "unique_threshold": qualified[0] if len(qualified) == 1 else next(c for c in rows if c["id"] == scene["incumbent"]),
    }
    return {
        "observation_id": scene["id"], "mechanism": scene["mechanism"], "parameters": scene["parameters"],
        "neighbor_supports": supports, "candidate_scores": rows, "threshold": threshold,
        "qualified_candidates": [c["id"] for c in qualified],
        "decisions": [{"method": method, "selected_id": c["id"], "selected_depth": c["depth"],
                       "abstained": method == "unique_threshold" and len(qualified) != 1}
                      for method, c in choices.items()],
    }


def main():
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    observed = json.loads((ROOT / "observations.json").read_text())
    output = {"seed": observed["seed"], "scenes": [select_scene(s, protocol) for s in observed["scenes"]]}
    (ROOT / "decisions.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    files = ("PROTOCOL.json", "observations.json", "select.py", "decisions.json")
    seal = {"sealed_files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
            "truth_files_read": [], "decision_rows": sum(len(s["decisions"]) for s in output["scenes"])}
    (ROOT / "decision_seal.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sealed_decision_rows": seal["decision_rows"], "truth_files_read": []}))


if __name__ == "__main__":
    main()
