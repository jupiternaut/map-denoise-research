"""Post-confirmation exact adversarial witness, not a new confirmation score."""
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "finite_world"))
from certificates import Hypothesis, certify, encode, linf, minimum_residual


def q(d):
    return Q(d["numerator"], d["denominator"])


def main():
    path = ROOT / "finite_world" / "outputs" / "confirmation-v1" / "library.json"
    library = json.loads(path.read_text(encoding="utf-8"))
    epsilon = q(library["epsilon"])
    family = next(f for f in library["families"] if f["family_id"] == "two_opaque_layers")
    hypotheses = tuple(Hypothesis(w["world_id"], q(w["depth"]), tuple(map(q, w["prediction"])))
                       for w in family["worlds"])
    pair = min(family["pairwise_separation"], key=lambda p: q(p["observation_distance"]))
    left = next(h for h in hypotheses if h.world_id == pair["left"])
    right = next(h for h in hypotheses if h.world_id == pair["right"])
    delta = linf(left.prediction, right.prediction)
    if not (0 < delta < 2*epsilon):
        raise ValueError("this library has no selected overlapping-noise-ball witness")
    fraction = Q(1) if delta <= epsilon else (Q(1, 2) + epsilon/delta)/2
    observation = tuple((1-fraction)*a + fraction*b for a,b in zip(left.prediction,right.prediction))
    actions = tuple(map(q, family["actions"]))
    safe = certify(observation, hypotheses, epsilon, left.depth, actions)
    residual = minimum_residual(observation, hypotheses, left.depth, actions)
    if linf(observation,left.prediction) > epsilon or linf(observation,right.prediction) > epsilon:
        raise AssertionError("constructed witness must satisfy both closed error budgets")
    if safe.output != left.depth or residual == left.depth:
        raise AssertionError("expected correct point KEEP and residual baseline damage")
    report = {"status": "verified_exact_construction",
              "scope": "post-confirmation analytical counterexample; not a new statistical confirmation",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "library_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "family": family["family_id"], "actual_world": left.world_id,
              "competitor": right.world_id, "actual_truth_depth_mm": encode(left.depth),
              "incumbent_mm": encode(left.depth), "epsilon": encode(epsilon),
              "pair_distance": encode(delta), "interpolation_fraction": encode(fraction),
              "true_observation_error": encode(linf(observation,left.prediction)),
              "competitor_observation_error": encode(linf(observation,right.prediction)),
              "observation": [encode(x) for x in observation],
              "certificate_output_mm": encode(safe.output), "certificate_status": safe.status,
              "feasible_worlds": list(safe.feasible_ids),
              "minimum_residual_output_mm": encode(residual),
              "minimum_residual_error_mm": encode(abs(residual-left.depth)),
              "minimum_residual_actual_gain_mm2": encode(-(residual-left.depth)**2)}
    out = path.parent.parent / "adversarial-witness.json"
    out.write_text(json.dumps(report, indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("observation","source_sha256","library_sha256")}))


if __name__ == "__main__":
    main()
