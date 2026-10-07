"""Deterministic synthetic environment. Truth is emitted to a separate file."""

import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def plane_at_depth(normal, depth):
    return {"normal": normal, "offset": normal[2] * depth}


def regular_neighbors(rng, depths, protocol):
    observed, true = [], []
    for i, depth in enumerate(depths):
        theta = rng.uniform(-0.25, 0.25)
        delta = rng.uniform(-0.0005, 0.0005)
        normal = [math.sin(theta), 0.0, math.cos(theta)]
        measured = [math.sin(theta + delta), 0.0, math.cos(theta + delta)]
        true_plane = plane_at_depth(normal, depth)
        observed.append({
            "id": f"n{i}", "normal": measured,
            "offset": true_plane["offset"] + rng.uniform(-0.001, 0.001),
            "eps_normal": protocol["regular_normal_error_budget"],
            "eps_offset": protocol["regular_offset_error_budget_m"],
            "source_group": f"source_{i}",
        })
        true.append({"id": f"n{i}", **true_plane, "layer_depth": depth})
    return observed, true


def photo_alias_costs(depths):
    """Repeat period 2px: disparities 102px and 100px alias exactly.

    Each image has a bounded amplitude .03 structured perturbation. The
    source perturbation happens to align with the target at the false alias.
    No random search or post-evaluation selection of this construction.
    """
    focal_baseline = 204.0
    true_disparity, nuisance_alignment = 102.0, 100.0
    period, amplitude = 2.0, 0.03
    samples = [(i - 32) / 8 for i in range(65)]

    def target(x):
        return math.cos(2 * math.pi * x / period) + amplitude * math.cos(2 * math.pi * x / 17)

    def source(u):
        return math.cos(2 * math.pi * (u + true_disparity) / period) + amplitude * math.cos(2 * math.pi * (u + nuisance_alignment) / 17)

    return [sum((target(x) - source(x - focal_baseline / z)) ** 2 for x in samples) / len(samples) for z in depths]


def main():
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    rng = random.Random(protocol["seed"])
    depths, count = protocol["candidate_depths"], protocol["neighbor_count"]
    candidates = [{"id": f"c{i}", "depth": z} for i, z in enumerate(depths)]
    alias_costs = photo_alias_costs(depths)
    scenes, worlds = [], []

    def add(scene_id, mechanism, neighbors, actual, target=2.0, params=None, costs=None):
        scenes.append({
            "id": scene_id, "mechanism": mechanism,
            "candidates": [{**c, "photo_loss": loss} for c, loss in zip(candidates, alias_costs if costs is None else costs)],
            "neighbors": neighbors, "incumbent": protocol["incumbent_candidate_id"],
            "declared_max_bad_neighbors": protocol["declared_max_bad_neighbors"],
            "parameters": params or {},
        })
        worlds.append({"world_id": scene_id, "observation_id": scene_id, "target_depth": target, "true_neighbors": actual})

    for replicate in range(protocol["bounded_replicates"]):
        observed, actual = regular_neighbors(rng, [2.0] * count, protocol)
        add(f"bounded_{replicate:03d}", "same_layer_bounded", observed, actual, params={"replicate": replicate})

    for contaminants in range(count + 1):
        observed, actual = regular_neighbors(rng, [2.0] * (count - contaminants) + [2.04] * contaminants, protocol)
        add(f"contamination_{contaminants}", "cross_layer", observed, actual, params={"environment_variant": contaminants})

    observed = [{"id": f"n{i}", "normal": [0.0, 0.0, 1.0], "offset": 2.04,
                 "eps_normal": 0.0005, "eps_offset": 0.001, "source_group": "shared_pose_bias"} for i in range(count)]
    actual = [{"id": f"n{i}", **plane_at_depth([0.0, 0.0, 1.0], 2.0), "layer_depth": 2.0} for i in range(count)]
    add("common_shift_shared", "common_shift", observed, actual)
    worlds[-1]["world_id"] = "common_shift_truth_2_00"
    worlds.append({"world_id": "common_shift_truth_2_04", "observation_id": "common_shift_shared", "target_depth": 2.04,
                   "true_neighbors": [{"id": f"n{i}", **plane_at_depth([0.0, 0.0, 1.0], 2.04), "layer_depth": 2.04} for i in range(count)]})

    for cosine in protocol["near_tangent_cosines"]:
        normal = [math.sqrt(1 - cosine * cosine), 0.0, cosine]
        observed = [{"id": f"n{i}", "normal": normal, "offset": 2 * cosine + protocol["near_tangent_offset_error_budget_m"],
                     "eps_normal": protocol["near_tangent_normal_error_budget"], "eps_offset": protocol["near_tangent_offset_error_budget_m"],
                     "source_group": f"source_{i}"} for i in range(count)]
        actual = [{"id": f"n{i}", **plane_at_depth(normal, 2.0), "layer_depth": 2.0} for i in range(count)]
        add(f"tangent_{cosine:g}", "near_tangent", observed, actual, params={"observed_ray_cosine": cosine})

    observed, actual = regular_neighbors(rng, [2.0] * count, protocol)
    add("periodic_texture_strong_geometry", "periodic_texture", observed, actual, params={"geometry": "well_conditioned"})
    observed, actual = regular_neighbors(rng, [2.0] * count, protocol)
    for row in observed:
        row["eps_offset"] = 0.09
    add("periodic_texture_wide_geometry", "periodic_texture", observed, actual, params={"geometry": "broad_valid_budget"})

    dump(ROOT / "observations.json", {"seed": protocol["seed"], "scenes": scenes})
    dump(ROOT / "evaluation_truth.json", {"worlds": worlds})
    dump(ROOT / "generation_manifest.json", {
        "seed": protocol["seed"], "observations_count": len(scenes), "world_count": len(worlds),
        "files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ("PROTOCOL.json", "generate.py", "observations.json", "evaluation_truth.json")},
        "photo_model": {"period_pixels": 2, "focal_length_times_baseline": 204, "pure_texture_true_disparity": 102,
                        "structured_nuisance_alignment_disparity": 100, "perturbation_amplitude_each_image": 0.03,
                        "patch_samples": 65, "loss": "mean squared pixel residual", "costs_in_candidate_order": alias_costs},
        "identifiability_pair": {"world_ids": ["common_shift_truth_2_00", "common_shift_truth_2_04"], "single_shared_observation_id": "common_shift_shared"},
    })
    print(json.dumps({"generated_scenes": len(scenes), "worlds": len(worlds), "seed": protocol["seed"]}))


if __name__ == "__main__":
    main()
