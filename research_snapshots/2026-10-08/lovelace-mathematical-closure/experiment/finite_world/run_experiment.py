"""Frozen, reproducible stage-1 finite-world experiment; no third-party packages."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from certificates import Hypothesis, certify, encode, minimum_residual, linf
from renderer import Camera, ImageGrid, Material, Surface, World, render, evaluation_target_z
from mesh_patch import lovelace_world, render_triangle

ROOT = Path(__file__).resolve().parent
DEPTHS = tuple(Q(z) for z in range(540, 661, 15))
TRUTH_DEPTHS = (Q(570), Q(600), Q(630))
SEEDS = tuple(range(1000, 1012))
EPSILON = Q(1, 100)
CAMERAS = (Camera(-70, "left"), Camera(0, "reference"), Camera(70, "right"))
GRID = ImageGrid(8, 6, Q(-1, 5), Q(1, 5), Q(-2, 25), Q(2, 25))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hashes():
    files = sorted(ROOT.glob("*.py")) + [ROOT / "EXPERIMENT_PLAN.md"] + sorted((ROOT / "assets").glob("*.json"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def make_world(family, depth):
    name = f"{family}_{depth}"
    checker = Material("checker", (Q(1, 10), Q(1, 5), Q(3, 10)),
                       (Q(4, 5), Q(9, 10), Q(7, 10)), 10)
    foreground = Surface(depth, -38, 38, -28, 28, checker, "target")
    if family == "checker_single":
        return World(name, (foreground,), (1, 1, 1))
    if family == "two_opaque_layers":
        foreground = Surface(depth, -20, 20, -22, 22, checker, "target_front")
        rear = Surface(800, -130, 130, -110, 110,
                       Material("solid", (Q(3, 5), Q(9, 20), Q(7, 20))), "rear")
        return World(name, (foreground, rear), (1, 1, 1))
    if family == "same_color":
        color = (Q(1, 2), Q(1, 2), Q(1, 2))
        return World(name, (Surface(depth, -38, 38, -28, 28,
                                    Material("solid", color), "invisible_target"),), color)
    if family == "lovelace_m4_facet":
        return lovelace_world(depth, name=name)
    raise ValueError(family)


def predict(family, world):
    return render_triangle(world, CAMERAS, GRID) if family == "lovelace_m4_facet" else render(world, CAMERAS, GRID)


def compile_library():
    """Build ALL forecasts before choosing any confirmation world or noise."""
    library = {"schema": 1, "scope": "declared_finite_family_only",
               "epsilon": encode(EPSILON), "norm": "linfinity",
               "cameras": [c.to_dict() for c in CAMERAS], "grid": GRID.to_dict(),
               "noise": "unclipped additive bounded rational per-channel noise",
               "target": "first hit of fixed reference +Z unit ray; millimeters",
               "families": []}
    compiled = {}
    for family in ("checker_single", "two_opaque_layers", "same_color", "lovelace_m4_facet"):
        model_worlds, hypotheses = [], []
        for depth in DEPTHS:
            world = make_world(family, depth)
            target = world.target_z if family == "lovelace_m4_facet" else evaluation_target_z(world)
            prediction = predict(family, world)
            hypothesis = Hypothesis(world.name, target, prediction)
            hypotheses.append(hypothesis)
            model_worlds.append({"world_id": hypothesis.world_id, "depth": encode(target),
                                 "prediction": [encode(x) for x in prediction], "model": world.to_dict()})
        pairs = []
        for i, left in enumerate(hypotheses):
            for right in hypotheses[i+1:]:
                distance = linf(left.prediction, right.prediction)
                pairs.append({"left": left.world_id, "right": right.world_id,
                              "observation_distance": encode(distance),
                              "distance_per_mm": encode(distance / abs(left.depth-right.depth))})
        kappa = min(q(p["distance_per_mm"]) for p in pairs)
        minimum_separation = min(q(p["observation_distance"]) for p in pairs)
        library["families"].append({"family_id": family,
                                    "kind": "mesh_single_facet" if family == "lovelace_m4_facet" else "opaque_rectangles",
                                    "actions": [encode(x) for x in DEPTHS], "worlds": model_worlds,
                                    "pairwise_separation": pairs, "finite_kappa_per_mm": encode(kappa),
                                    "minimum_separation": encode(minimum_separation),
                                    "uniform_2epsilon_separation": minimum_separation > 2*EPSILON,
                                    "finite_target_diameter_bound_mm": None if kappa == 0 else encode(2*EPSILON/kappa)})
        compiled[family] = tuple(hypotheses)
    return library, compiled


def decision_json(decision):
    return {"status": decision.status, "output": encode(decision.output),
            "feasible_ids": list(decision.feasible_ids),
            "certified_gain": None if decision.certified_gain is None else encode(decision.certified_gain),
            "reason": decision.reason}


def noisy_observation(h, family, seed):
    rng = random.Random(f"{family}:{h.depth}:{seed}")
    choices = (-EPSILON, -EPSILON / 2, Q(0), EPSILON / 2, EPSILON)
    return tuple(x + rng.choice(choices) for x in h.prediction)


def summary_rows(trials, compiled):
    groups = defaultdict(list)
    for trial in trials:
        family = trial["family_id"]
        truth = next(h.depth for h in compiled[family] if h.world_id == trial["truth_world_id"])
        a = q(trial["incumbent"])
        outputs = {"certificate": q(trial["decision"]["output"]),
                   **{k: q(v) for k, v in trial["baselines"].items()}}
        for method, b in outputs.items():
            groups[(family, trial["state"], method)].append((abs(a-truth), abs(b-truth), b != a))
    rows = []
    for (family, state, method), values in sorted(groups.items()):
        n = len(values)
        rows.append({"family": family, "state": state, "method": method, "count": n,
                     "improved": sum(new < old for old, new, _ in values),
                     "worse": sum(new > old for old, new, _ in values),
                     "same": sum(new == old for old, new, _ in values),
                     "moved": sum(move for _, _, move in values),
                     "mae_mm": encode(sum(new for _, new, _ in values)/n),
                     "mse_mm2": encode(sum(new*new for _, new, _ in values)/n)})
    return rows


def q(value):
    return Q(value["numerator"], value["denominator"])


def controls(compiled):
    cases = []
    # Empty set: every physical forecast is in [0,1], observation=2 violates model/budget.
    family = "checker_single"
    hs = compiled[family]
    y = (Q(2),) * len(hs[0].prediction)
    cases.append({"name": "empty_feasible_set", "contract": "violated_observation_budget",
                  "family_id": family, "observation": [encode(x) for x in y], "epsilon": encode(EPSILON),
                  "decision": decision_json(certify(y, hs, EPSILON, 600, DEPTHS)),
                  "expected": "INCOMPATIBLE and unchanged 600"})
    # Truth absent from finite family although its image is identical to all forecasts.
    family = "same_color"
    actual = make_world(family, Q(533))
    y = predict(family, actual)
    d = certify(y, compiled[family], EPSILON, 533, DEPTHS)
    cases.append({"name": "out_of_family_true_depth", "contract": "truth_not_in_declared_family",
                  "family_id": family, "observation": [encode(x) for x in y], "epsilon": encode(EPSILON),
                  "actual_truth_depth": encode(Q(533)), "incumbent": encode(Q(533)),
                  "decision": decision_json(d),
                  "actual_gain": encode(-(d.output-Q(533))**2),
                  "expected": "finite-family certificate can damage excluded true world"})
    # Wrong observation can imitate another admissible world exactly.
    family = "checker_single"
    truth = next(h for h in compiled[family] if h.depth == 540)
    wrong = next(h for h in compiled[family] if h.depth == 660)
    d = certify(wrong.prediction, compiled[family], 0, truth.depth, DEPTHS)
    cases.append({"name": "underestimated_error_budget", "contract": "actual_error_exceeds_epsilon",
                  "family_id": family, "observation": [encode(x) for x in wrong.prediction], "epsilon": encode(Q(0)),
                  "actual_truth_depth": encode(truth.depth), "incumbent": encode(truth.depth),
                  "decision": decision_json(d), "actual_gain": encode(-(d.output-truth.depth)**2),
                  "expected": "excluded truth can be damaged despite positive finite certificate"})
    return cases


def write_visuals(out, library):
    # SVG is a scientific display of generated observations, never a proof computation.
    body = []
    cell = 11
    family_count = len(library["families"])
    for fi, family in enumerate(library["families"]):
        y0 = 38 + fi * 112
        body.append(f'<text x="14" y="{y0}" font-size="14">{family["family_id"]}: depth 600 mm</text>')
        world = next(w for w in family["worlds"] if q(w["depth"]) == 600)
        values = [q(v) for v in world["prediction"]]
        for ci, camera in enumerate(CAMERAS):
            x0 = 16 + ci * 145
            body.append(f'<text x="{x0}" y="{y0+17}" font-size="11">{camera.name} cx={camera.cx} mm</text>')
            for row in range(GRID.height):
                for col in range(GRID.width):
                    offset = ((ci * GRID.height + row) * GRID.width + col) * 3
                    rgb = tuple(round(float(values[offset+k])*255) for k in range(3))
                    body.append(f'<rect x="{x0+col*cell}" y="{y0+23+row*cell}" width="{cell}" height="{cell}" fill="rgb{rgb}"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="460" height="{family_count*112+65}" viewBox="0 0 460 {family_count*112+65}">'
           '<rect width="100%" height="100%" fill="#f4f4ef"/><g font-family="sans-serif" fill="#222">'
           '<text x="14" y="19" font-size="12">Exact area-integrated forecasts (display rounded to RGB8)</text>'
           + ''.join(body) + '</g></svg>')
    (out / "worlds.svg").write_text(svg, encoding="utf-8")


def run(out):
    if out.exists():
        raise FileExistsError("preserve existing evidence; select a fresh --output directory")
    started = time.perf_counter()
    # Lock sources and protocol before forecasts or confirmation are generated.
    source_lock = {"schema": 1, "utc": datetime.now(timezone.utc).isoformat(),
                   "python": sys.version, "source_hashes": hashes(),
                   "confirmation_seeds": list(SEEDS), "truth_depths_mm": [encode(x) for x in TRUTH_DEPTHS],
                   "epsilon": encode(EPSILON), "margin": encode(Q(0))}
    out.mkdir(parents=True)
    write_json(out / "PROTOCOL_LOCK.json", source_lock)
    for relative in source_lock["source_hashes"]:
        snapshot = out / "source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / relative).read_bytes())
    library, compiled = compile_library()
    write_json(out / "library.json", library)
    forecast_hash = hashlib.sha256((out / "library.json").read_bytes()).hexdigest()
    write_json(out / "FORECAST_LOCK.json", {"sha256": forecast_hash,
               "note": "all candidate forecasts compiled before actual world/noise selection"})
    trials = []
    with (out / "trials.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for family, hypotheses in compiled.items():
            for depth in TRUTH_DEPTHS:
                truth = next(h for h in hypotheses if h.depth == depth)
                for seed in SEEDS:
                    observation = noisy_observation(truth, family, seed)
                    for state, offset in (("correct", 0), ("minus60", -60), ("plus60", 60)):
                        incumbent = depth + offset
                        decision = certify(observation, hypotheses, EPSILON, incumbent, DEPTHS)
                        baseline = minimum_residual(observation, hypotheses, incumbent, DEPTHS)
                        trial = {"trial_id": f"{family}:{depth}:{seed}:{state}", "family_id": family,
                                 "truth_world_id": truth.world_id, "seed": seed, "state": state,
                                 "incumbent": encode(incumbent), "observation": [encode(x) for x in observation],
                                 "epsilon": encode(EPSILON), "decision": decision_json(decision),
                                 "baselines": {"keep": encode(incumbent), "minimum_residual": encode(baseline)}}
                        stream.write(json.dumps(trial, separators=(",", ":")) + "\n")
                        trials.append(trial)
    write_json(out / "controls.json", controls(compiled))
    summary = {"scope": "finite_model_confirmation_only", "families": len(compiled),
               "distinct_geometry_worlds": len(compiled)*len(TRUTH_DEPTHS),
               "scene_noise_configurations": len(compiled)*len(TRUTH_DEPTHS)*len(SEEDS),
               "incumbent_decisions": len(trials), "rows": summary_rows(trials, compiled),
               "elapsed_seconds": f"{time.perf_counter()-started:.3f}"}
    write_json(out / "summary.json", summary)
    write_visuals(out, library)
    current_hashes = hashes()
    if current_hashes != source_lock["source_hashes"]:
        raise RuntimeError("source changed during confirmation; evidence is not frozen")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "confirmation-v1")
    args = parser.parse_args()
    run(args.output.resolve())
