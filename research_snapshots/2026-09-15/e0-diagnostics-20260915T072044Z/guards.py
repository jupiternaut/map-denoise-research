"""E0-D C: observed-gap guard construction and deliberately ambiguous fixtures.

Units are millimetres. Frozen fixture choices supplement PROTOCOL.md: 20 x 20
XY grids, layer heights +/-3, independent uniform +/-0.1 observation noise,
rigid translation +0.3, and tilted plane z=1.8*x-0.7*y. These choices are made
before the first run; no fixture or guard parameter is selected from outcomes.
Only evaluation functions receive truth. The two-world fixture reuses identical
observations/proposals, changing only evaluation truth.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import ROOT, load_old, plain, write_json

H_VALUES = (0.4, 0.8, 1.6)
SEEDS = (101, 102, 103)
GRID_SIDE = 20
LAYER_HALF_GAP = 3.0
NOISE_HALF_WIDTH = 0.1
TRANSLATION = 0.3
PLANE_SLOPE = (1.8, -0.7)
XY_RADIUS_FACTOR = 1.5
MIN_INITIAL_GAP = 2.0
CONTRACTION_FACTOR = 0.5
SCENARIOS = (
    "wrong_layer_merge",
    "valid_ghost_merge",
    "within_layer_noise_repair",
    "rigid_translation",
    "smooth_tilted_plane",
    "same_xy_collision_control",
)


def gap_contraction_guard(theta, x, y, z0, h):
    """Reject moved endpoints of contracting observed XY-neighbor pairs once.

All pair decisions refer to the *original* observation and prescribed proposal.
There is no recomputation after vetoes, and no truth, labels or source-sheet
membership enters this function. Arbitrary input arrays are not modified.
"""
    theta, x, y, z0 = [np.asarray(a, dtype=float) for a in (theta, x, y, z0)]
    if any(a.ndim != 1 for a in (theta, x, y, z0)):
        raise ValueError("theta, x, y, z0 must be one-dimensional")
    if any(a.shape != theta.shape for a in (x, y, z0)):
        raise ValueError("theta, x, y, z0 must have identical shapes")
    if not all(np.isfinite(a).all() for a in (theta, x, y, z0)):
        raise ValueError("geometry and motion must be finite")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    xy = np.column_stack((x, y))
    pairs = cKDTree(xy).query_pairs(XY_RADIUS_FACTOR * h, output_type="ndarray")
    # Sorting makes raw output order reproducible independently of tree order.
    if len(pairs):
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    initial_gap = np.abs(z0[pairs[:, 0]] - z0[pairs[:, 1]])
    candidate_z = z0 + theta
    candidate_gap = np.abs(candidate_z[pairs[:, 0]] - candidate_z[pairs[:, 1]])
    eligible_pair = initial_gap >= MIN_INITIAL_GAP
    flagged_pair = eligible_pair & (candidate_gap < CONTRACTION_FACTOR * initial_gap)
    flagged = np.zeros(len(theta), dtype=bool)
    flagged[pairs[flagged_pair].ravel()] = True
    flagged &= theta != 0
    guarded = theta.copy()
    guarded[flagged] = 0.0
    return guarded, flagged, {
        "pairs": pairs,
        "initial_gap": initial_gap,
        "candidate_gap": candidate_gap,
        "eligible_pair": eligible_pair,
        "flagged_pair": flagged_pair,
    }


def make_fixtures(h, seed):
    """Generate independent scenes, except for the explicitly paired worlds."""
    h_index = H_VALUES.index(h)
    streams = np.random.SeedSequence([seed, h_index, 314159]).spawn(5)
    rng_pair, rng_noise, rng_rigid, rng_slope, rng_collision = [
        np.random.default_rng(s) for s in streams
    ]
    ix, iy = np.meshgrid(np.arange(GRID_SIDE), np.arange(GRID_SIDE), indexing="ij")
    x, y = h * ix.ravel(), h * iy.ravel()
    sheet = (ix.ravel() + iy.ravel()) % 2
    layer_z = np.where(sheet == 0, -LAYER_HALF_GAP, LAYER_HALF_GAP)

    def fixture(scenario, z0, target, proposal, **extra):
        return {
            "scenario": scenario, "h": h, "seed": seed,
            "x": x.copy(), "y": y.copy(), "z0": z0.copy(),
            "truth_z": target.copy(), "theta": (proposal - z0).copy(),
            "source_sheet": sheet.copy(), **extra,
        }

    paired_z0 = layer_z + rng_pair.uniform(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, len(x))
    paired_proposal = np.zeros(len(x))
    pair_id = f"paired_h{h:.1f}_seed{seed}"
    wrong = fixture(
        "wrong_layer_merge", paired_z0, layer_z, paired_proposal,
        pair_id=pair_id, evidence_kind="independent_development_fixture",
        truth_description="Two real source layers; merging erases their separation.",
    )
    ghost = fixture(
        "valid_ghost_merge", paired_z0, paired_proposal, paired_proposal,
        pair_id=pair_id, evidence_kind="reused_observation_alternative_truth",
        truth_description="One real plane; the two observed heights are ghosts.",
    )
    noise_z0 = layer_z + rng_noise.uniform(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, len(x))
    noise = fixture(
        "within_layer_noise_repair", noise_z0, layer_z, layer_z,
        evidence_kind="independent_development_fixture",
        truth_description="Two source layers with small within-layer observation noise.",
    )
    rigid_z0 = layer_z + rng_rigid.uniform(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, len(x))
    rigid_target = rigid_z0 + TRANSLATION
    rigid = fixture(
        "rigid_translation", rigid_z0, rigid_target, rigid_target,
        evidence_kind="independent_development_fixture",
        truth_description="The entire observed cloud is offset from its correct rigid location.",
    )
    plane_z = PLANE_SLOPE[0] * x + PLANE_SLOPE[1] * y
    slope_z0 = plane_z + rng_slope.uniform(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, len(x))
    slope = fixture(
        "smooth_tilted_plane", slope_z0, plane_z, plane_z,
        evidence_kind="independent_development_fixture",
        truth_description="Smooth sloped surface repaired to its noise-free target.",
    )
    slope["source_sheet"] = np.zeros(len(x), dtype=int)
    origin = rng_collision.uniform(-1.0, 1.0, 3)
    collision_z = origin[2] + np.array([-LAYER_HALF_GAP, LAYER_HALF_GAP])
    collision = {
        "scenario": "same_xy_collision_control", "h": h, "seed": seed,
        "x": np.full(2, origin[0]), "y": np.full(2, origin[1]),
        "z0": collision_z, "truth_z": collision_z.copy(),
        "theta": np.array([0.0, -2 * LAYER_HALF_GAP]),
        "source_sheet": np.array([0, 1]),
        "evidence_kind": "positive_control_random_translation",
        "truth_description": "One moved point collides with its stationary same-XY neighbor.",
    }
    # Truth alone changes in the paired worlds. Enforce this before any guard call.
    for name in ("x", "y", "z0", "theta"):
        assert np.array_equal(wrong[name], ghost[name])
    return [wrong, ghost, noise, rigid, slope, collision]


def evaluate(fixture, theta, flagged, proposal_theta):
    """Truth-based descriptive evaluation, never called by either selector."""
    z0, target = fixture["z0"], fixture["truth_z"]
    error_before = np.abs(z0 - target)
    error_proposal = np.abs(z0 + proposal_theta - target)
    error_after = np.abs(z0 + theta - target)
    moved = proposal_theta != 0
    accepted = theta != 0
    beneficial = moved & (error_proposal < error_before - 1e-12)
    harmful = moved & (error_proposal > error_before + 1e-12)
    potential_repair = np.maximum(error_before - error_proposal, 0.0)
    realized_repair = np.maximum(error_before - error_after, 0.0)
    retained_on_beneficial = float(realized_repair[beneficial].sum())
    total_potential = float(potential_repair.sum())
    sheet_errors = {}
    for label in np.unique(fixture["source_sheet"]):
        subset = fixture["source_sheet"] == label
        sheet_errors[str(label)] = {
            "n": int(subset.sum()),
            "mae_before_mm": float(error_before[subset].mean()),
            "mae_after_mm": float(error_after[subset].mean()),
        }
    return {
        "n": len(z0), "proposed_movement_count": int(moved.sum()),
        "alarm_count": int(flagged.sum()),
        "alarm_fraction_of_proposed": float(flagged.sum() / moved.sum()) if moved.any() else None,
        "accepted_movement_count": int(accepted.sum()),
        "source_sheet_mae_before_mm": float(error_before.mean()),
        "source_sheet_mae_after_mm": float(error_after.mean()),
        "source_sheet_rmse_before_mm": float(np.sqrt(np.mean(error_before ** 2))),
        "source_sheet_rmse_after_mm": float(np.sqrt(np.mean(error_after ** 2))),
        "source_sheet_p95_error_after_mm": float(np.quantile(error_after, 0.95)),
        "proposal_beneficial_count": int(beneficial.sum()),
        "proposal_harmful_count": int(harmful.sum()),
        "beneficial_proposals_retained_count": int((beneficial & accepted).sum()),
        "beneficial_proposals_blocked_count": int((beneficial & ~accepted).sum()),
        "harmful_proposals_retained_count": int((harmful & accepted).sum()),
        "harmful_proposals_blocked_count": int((harmful & ~accepted).sum()),
        "potential_repair_sum_mm": total_potential,
        "repair_retained_sum_mm": retained_on_beneficial,
        "repair_blocked_sum_mm": total_potential - retained_on_beneficial,
        "repair_retained_fraction": retained_on_beneficial / total_potential if total_potential else None,
        "source_sheet_breakdown": sheet_errors,
    }


def run_case(fixture, old, output):
    started = time.perf_counter()
    theta, x, y, z0 = [fixture[name] for name in ("theta", "x", "y", "z0")]
    old_start = time.perf_counter()
    old_theta, old_flags, old_nn = old.apply_guard(theta, x, y, z0)
    old_seconds = time.perf_counter() - old_start
    new_start = time.perf_counter()
    new_theta, new_flags, diagnostics = gap_contraction_guard(theta, x, y, z0, fixture["h"])
    new_seconds = time.perf_counter() - new_start
    zero_flags = np.zeros(len(theta), dtype=bool)
    arms = {
        "identity": (np.zeros_like(theta), zero_flags, 0.0),
        "proposal": (theta, zero_flags, 0.0),
        "old_nn_guard": (old_theta, old_flags, old_seconds),
        "initial_gap_guard": (new_theta, new_flags, new_seconds),
    }
    case_id = f"{fixture['scenario']}_h{fixture['h']:.1f}_seed{fixture['seed']}"
    metadata = {k: v for k, v in fixture.items() if not isinstance(v, np.ndarray)}
    metadata.update({
        "case_id": case_id,
        "coordinate_error_interpretation": "Fixed XY support: absolute z error to each point's assigned true source surface; not nearest-surface reassignment.",
        "neighbor_pair_count": len(diagnostics["pairs"]),
        "eligible_pair_count": int(diagnostics["eligible_pair"].sum()),
        "flagged_pair_count": int(diagnostics["flagged_pair"].sum()),
        "old_nn_min_moved_mm": float(old_nn.min()) if len(old_nn) else None,
        "old_nn_threshold_shortfall_max_mm": float(np.maximum(old.D_GUARD - old_nn, 0.0).max()) if len(old_nn) else None,
        "guard_scope": "Single pass with original proposal; no iterative veto feedback.",
    })
    metrics = {
        arm: {**evaluate(fixture, arm_theta, flags, theta), "guard_seconds": seconds}
        for arm, (arm_theta, flags, seconds) in arms.items()
    }
    arrays = {name: fixture[name] for name in ("x", "y", "z0", "theta", "truth_z", "source_sheet")}
    arrays.update({
        "initial_xyz": np.column_stack((x, y, z0)),
        "proposal_xyz": np.column_stack((x, y, z0 + theta)),
        "truth_xyz": np.column_stack((x, y, fixture["truth_z"])),
        "old_nn_moved_mm": old_nn, **diagnostics,
    })
    for arm, (arm_theta, flags, _) in arms.items():
        arrays[f"{arm}_theta"] = arm_theta
        arrays[f"{arm}_flags"] = flags
        arrays[f"{arm}_xyz"] = np.column_stack((x, y, z0 + arm_theta))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{case_id}.npz", **arrays)
    metadata["case_total_seconds"] = time.perf_counter() - started
    write_json(output / f"{case_id}.json", {**metadata, "arms": metrics})
    rows = []
    for arm, values in metrics.items():
        rows.append({
            **metadata, "arm": arm,
            **{k: v for k, v in values.items() if not isinstance(v, dict)},
        })
    return {**metadata, "arms": metrics}, rows, (old_flags, new_flags, old_theta, new_theta)


def summarize(cases, pair_checks, elapsed):
    groups = []
    for scenario in SCENARIOS:
        for h in H_VALUES:
            selected = [case for case in cases if case["scenario"] == scenario and case["h"] == h]
            for arm in ("proposal", "old_nn_guard", "initial_gap_guard"):
                total_n = sum(case["arms"][arm]["n"] for case in selected)
                groups.append({
                    "scenario": scenario, "h": h, "arm": arm, "case_count": len(selected),
                    "alarm_count": sum(case["arms"][arm]["alarm_count"] for case in selected),
                    "accepted_movement_count": sum(case["arms"][arm]["accepted_movement_count"] for case in selected),
                    "mae_before_mm": sum(case["arms"][arm]["source_sheet_mae_before_mm"] * case["arms"][arm]["n"] for case in selected) / total_n,
                    "mae_after_mm": sum(case["arms"][arm]["source_sheet_mae_after_mm"] * case["arms"][arm]["n"] for case in selected) / total_n,
                    "beneficial_proposals_blocked_count": sum(case["arms"][arm]["beneficial_proposals_blocked_count"] for case in selected),
                    "harmful_proposals_blocked_count": sum(case["arms"][arm]["harmful_proposals_blocked_count"] for case in selected),
                })
    grid_cases = [c for c in cases if c["scenario"] != "same_xy_collision_control" and c["h"] >= 0.8]
    boundary_cases = [c for c in cases if c["scenario"] != "same_xy_collision_control" and c["h"] == 0.4 and c["arms"]["old_nn_guard"]["alarm_count"]]
    return {
        "case_count": len(cases), "metric_row_count": 4 * len(cases),
        "unique_regular_observation_fixtures": 4 * len(H_VALUES) * len(SEEDS),
        "reused_alternative_truth_cases": len(pair_checks),
        "same_xy_positive_controls": len(H_VALUES) * len(SEEDS),
        "elapsed_seconds": elapsed,
        "configuration": {
            "h_mm": H_VALUES, "seeds": SEEDS, "grid_side": GRID_SIDE,
            "layer_half_gap_mm": LAYER_HALF_GAP, "noise_half_width_mm": NOISE_HALF_WIDTH,
            "rigid_translation_mm": TRANSLATION, "plane_slope": PLANE_SLOPE,
            "xy_radius_factor": XY_RADIUS_FACTOR, "min_initial_gap_mm": MIN_INITIAL_GAP,
            "contraction_factor_strict": CONTRACTION_FACTOR, "old_nn_threshold_strict_mm": 0.4,
        },
        "old_grid_bound": {
            "argument": "For distinct grid XY with mathematical spacing h, Euclidean 3D NN distance is >= XY distance >= h for every z-only proposal. Thus h=.8 > .4 makes NN<.4 unreachable.",
            "numerical_control_h_at_least_point8_alarm_count": sum(c["arms"]["old_nn_guard"]["alarm_count"] for c in grid_cases),
            "point4_boundary_cases_with_alarms": [{
                "case_id": c["case_id"], "alarm_count": c["arms"]["old_nn_guard"]["alarm_count"],
                "nn_min_mm": c["old_nn_min_moved_mm"],
                "threshold_shortfall_max_mm": c["old_nn_threshold_shortfall_max_mm"],
            } for c in boundary_cases],
            "floating_point_note": "At h=.4 the strict threshold equals the mathematical spacing; finite-precision lattice coordinates may produce roundoff-scale threshold crossings. Recorded unchanged, not treated as structural detection.",
        },
        "paired_world_checks": pair_checks,
        "groups": groups,
        "limitations": [
            "Prescribed-proposal point-cloud fixtures; not an autonomous reconstruction algorithm or real-scene validation.",
            "Observed gap contraction cannot distinguish real layer preservation from correct removal of a ghost layer.",
            "One-pass incumbent-relative heuristic provides no global safety guarantee; unchecked new neighbors, XY motion, small gaps and insufficient contraction are outside this construction.",
            "Pointwise source-sheet errors use evaluation-only correspondence; truth never enters the selectors.",
            "Seeded synthetic fixtures are development evidence; the paired alternative worlds are not independent scenes.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "guards")
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = (ROOT / "results" / "guards").resolve()
    if output != allowed and allowed not in output.parents:
        parser.error("output must be within this study's results/guards directory")
    if output.exists() and any(output.iterdir()):
        parser.error("output directory already contains artifacts; use a fresh child to preserve outcomes")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    old = load_old()
    cases, all_rows, pair_checks = [], [], []
    for h in H_VALUES:
        for seed in SEEDS:
            paired_decisions = []
            for fixture in make_fixtures(h, seed):
                case, rows, decisions = run_case(fixture, old, output)
                cases.append(case)
                all_rows.extend(rows)
                if fixture["scenario"] in ("wrong_layer_merge", "valid_ghost_merge"):
                    paired_decisions.append(decisions)
            equality = [bool(np.array_equal(a, b)) for a, b in zip(*paired_decisions)]
            pair_checks.append({
                "h": h, "seed": seed,
                "old_flags_identical": equality[0], "initial_gap_flags_identical": equality[1],
                "old_guarded_proposals_identical": equality[2],
                "initial_gap_guarded_proposals_identical": equality[3],
                "observations_and_proposals_identical": True,
            })
            assert all(equality), "A selector distinguished observationally identical worlds"
    columns = list(dict.fromkeys(k for row in all_rows for k in row))
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(output / "metrics.json", cases)
    summary = summarize(cases, pair_checks, time.perf_counter() - started)
    write_json(output / "summary.json", summary)
    print(json.dumps(plain({
        "output": str(output), "case_count": summary["case_count"],
        "metric_row_count": summary["metric_row_count"],
        "paired_world_checks_passed": all(all(v for k, v in check.items() if k not in ("h", "seed")) for check in pair_checks),
        "elapsed_seconds": summary["elapsed_seconds"],
        "old_h_at_least_point8_alarm_count": summary["old_grid_bound"]["numerical_control_h_at_least_point8_alarm_count"],
        "point4_boundary_case_count": len(summary["old_grid_bound"]["point4_boundary_cases_with_alarms"]),
    }), indent=2))


if __name__ == "__main__":
    main()
