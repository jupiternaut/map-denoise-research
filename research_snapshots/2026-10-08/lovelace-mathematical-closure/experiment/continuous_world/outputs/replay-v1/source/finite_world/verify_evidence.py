"""Independent exact replay of finite-world confirmation evidence.

This checker does not import certificates.py or invoke its decision functions.
It recomputes consistency, noise coverage, action legality, worst gains and the
minimum-residual baseline from the serialized public hypothesis library. It
does not re-render library geometry or certify a physical imaging model.
Contract-violation controls are deliberately excluded from these safety counts.
"""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys


Q = Fraction
CONFIRMATION_DEPTHS = (Q(570), Q(600), Q(630))
CONFIRMATION_SEEDS = tuple(range(1000, 1012))
STATE_OFFSETS = {"correct": Q(0), "minus60": Q(-60), "plus60": Q(60)}
EXPECTED_LIBRARY_DEPTHS = {Q(540 + 15 * i) for i in range(9)}


def rational(value, field):
    """Decode the experiment's exact rational JSON, without accepting floats."""
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError(f"{field}: expected numerator/denominator object")
    numerator, denominator = value["numerator"], value["denominator"]
    if (isinstance(numerator, bool) or isinstance(denominator, bool)
            or not isinstance(numerator, int) or not isinstance(denominator, int)
            or denominator <= 0):
        raise ValueError(f"{field}: integer numerator and positive integer denominator required")
    return Q(numerator, denominator)


def vector(value, field):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field}: nonempty rational vector required")
    return tuple(rational(item, f"{field}[{i}]") for i, item in enumerate(value))


def exact_json(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {str(key): exact_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact_json(item) for item in value]
    return value


def linf(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("Observation dimensions differ or are empty")
    return max(abs(a - b) for a, b in zip(left, right))


class Audit:
    def __init__(self):
        self.failures = []
        self.check_counts = Counter()

    def expect(self, condition, check, context, detail=""):
        self.check_counts[check] += 1
        if not condition:
            self.failures.append({"check": check, "context": context, "detail": detail})

    def malformed(self, context, exc):
        self.failures.append({"check": "schema", "context": context,
                              "detail": f"{type(exc).__name__}: {exc}"})


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def parse_library(raw, audit):
    epsilon = rational(raw["epsilon"], "library.epsilon")
    audit.expect(epsilon == Q(1, 100), "frozen_epsilon", "library", "Expected epsilon=1/100")
    audit.expect(epsilon >= 0, "nonnegative_epsilon", "library")
    families = {}
    for raw_family in raw["families"]:
        family_id = raw_family["family_id"]
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("Family id must be a nonempty string")
        if family_id in families:
            raise ValueError(f"Duplicate family id {family_id}")
        worlds = []
        for raw_world in raw_family["worlds"]:
            world_id = raw_world["world_id"]
            if not isinstance(world_id, str) or not world_id:
                raise ValueError("World id must be a nonempty string")
            prediction = vector(raw_world["prediction"], f"{family_id}.{world_id}.prediction")
            audit.expect(all(0 <= value <= 1 for value in prediction),
                         "prediction_rgb_range", f"{family_id}/{world_id}")
            if not isinstance(raw_world["model"], dict):
                raise ValueError("Serialized world model must be an object")
            worlds.append({"world_id": world_id,
                           "depth": rational(raw_world["depth"], f"{world_id}.depth"),
                           "prediction": prediction})
        actions = vector(raw_family["actions"], f"{family_id}.actions")
        audit.expect(len(worlds) == 9, "nine_worlds", family_id)
        audit.expect(len({world["world_id"] for world in worlds}) == len(worlds),
                     "unique_world_ids", family_id)
        audit.expect({world["depth"] for world in worlds} == EXPECTED_LIBRARY_DEPTHS,
                     "frozen_depth_grid", family_id)
        audit.expect(len(set(actions)) == len(actions), "unique_actions", family_id)
        audit.expect(set(actions) == EXPECTED_LIBRARY_DEPTHS, "frozen_action_grid", family_id)
        audit.expect(len({len(world["prediction"]) for world in worlds}) == 1,
                     "fixed_observation_dimension", family_id)
        audit.expect(all(len(world["prediction"]) == 432 for world in worlds),
                     "three_view_8x6_rgb", family_id)
        families[family_id] = {"kind": raw_family["kind"], "worlds": worlds,
                               "by_id": {world["world_id"]: world for world in worlds},
                               "actions": actions}
    audit.expect(bool(families), "nonempty_library", "library")
    return epsilon, families


def residual_baseline(observation, worlds, actions, incumbent):
    """Independent target-class minimum residual, KEEP wins a tied class."""
    allowed = set(actions) | {incumbent}
    target_scores = {}
    for world in worlds:
        target = world["depth"]
        if target in allowed:
            score = linf(observation, world["prediction"])
            if target not in target_scores or score < target_scores[target]:
                target_scores[target] = score
    if not target_scores:
        return incumbent
    best_score = min(target_scores.values())
    targets = [target for target, score in target_scores.items() if score == best_score]
    if incumbent in targets:
        return incumbent
    return min(targets, key=lambda target: (abs(target - incumbent), target))


def verify_trial(raw, epsilon, families, audit, scene_observations, seen_keys, seen_ids,
                 totals, family_counts, state_counts):
    trial_id = raw["trial_id"]
    if not isinstance(trial_id, str) or not trial_id:
        raise ValueError("Trial id must be a nonempty string")
    audit.expect(trial_id not in seen_ids, "unique_trial_id", trial_id)
    seen_ids.add(trial_id)
    family_id = raw["family_id"]
    family = families[family_id]
    truth_id = raw["truth_world_id"]
    truth = family["by_id"][truth_id]
    seed, state = raw["seed"], raw["state"]
    audit.expect(isinstance(seed, int) and not isinstance(seed, bool) and seed in CONFIRMATION_SEEDS,
                 "confirmation_seed", trial_id)
    if state not in STATE_OFFSETS:
        raise ValueError(f"Unsupported confirmation state {state!r}")
    audit.expect(truth["depth"] in CONFIRMATION_DEPTHS, "confirmation_truth_depth", trial_id)
    scene_key = (family_id, truth_id, seed)
    trial_key = scene_key + (state,)
    audit.expect(trial_key not in seen_keys, "one_trial_per_scene_state", trial_id)
    seen_keys.add(trial_key)
    incumbent = rational(raw["incumbent"], f"{trial_id}.incumbent")
    audit.expect(incumbent == truth["depth"] + STATE_OFFSETS[state],
                 "frozen_initial_offset", trial_id)
    observation = vector(raw["observation"], f"{trial_id}.observation")
    trial_epsilon = rational(raw["epsilon"], f"{trial_id}.epsilon")
    audit.expect(trial_epsilon == epsilon, "shared_epsilon", trial_id)
    noise = linf(observation, truth["prediction"])
    audit.expect(noise <= trial_epsilon, "in_contract_noise_bound", trial_id,
                 f"Noise={noise}, epsilon={trial_epsilon}")
    if scene_key in scene_observations:
        audit.expect(scene_observations[scene_key] == observation,
                     "three_states_same_observation", trial_id)
    else:
        scene_observations[scene_key] = observation
    residuals = {world["world_id"]: linf(observation, world["prediction"])
                 for world in family["worlds"]}
    feasible = [world for world in family["worlds"]
                if residuals[world["world_id"]] <= trial_epsilon]
    feasible_ids = {world["world_id"] for world in feasible}
    audit.expect(truth_id in feasible_ids, "actual_world_coverage", trial_id)
    decision = raw["decision"]
    status = decision["status"]
    output = rational(decision["output"], f"{trial_id}.decision.output")
    claimed_ids = decision["feasible_ids"]
    if not isinstance(claimed_ids, list) or not all(isinstance(item, str) for item in claimed_ids):
        raise ValueError("Decision feasible_ids must be a string list")
    audit.expect(len(set(claimed_ids)) == len(claimed_ids) and set(claimed_ids) == feasible_ids,
                 "independent_feasible_set", trial_id)
    audit.expect(status in ("MOVE", "KEEP", "INCOMPATIBLE"), "valid_status", trial_id)
    audit.expect(isinstance(decision["reason"], str) and bool(decision["reason"]),
                 "decision_reason", trial_id)
    audit.expect(output in set(family["actions"]) | {incumbent}, "legal_output", trial_id)
    certificate_raw = decision.get("certified_gain")
    certificate = (None if certificate_raw is None else
                   rational(certificate_raw, f"{trial_id}.decision.certified_gain"))
    actual_gain = (incumbent - truth["depth"]) ** 2 - (output - truth["depth"]) ** 2
    if status == "MOVE":
        audit.expect(bool(feasible), "move_nonempty_feasibility", trial_id)
        audit.expect(output != incumbent, "nontrivial_move", trial_id)
        audit.expect(certificate is not None and certificate > 0,
                     "strict_positive_certificate", trial_id)
        if feasible:
            worst_gain = min((incumbent - world["depth"]) ** 2
                             - (output - world["depth"]) ** 2 for world in feasible)
            audit.expect(worst_gain > 0, "all_feasible_targets_strict_gain", trial_id,
                         f"Worst gain={worst_gain}")
            if certificate is not None:
                audit.expect(certificate <= worst_gain, "certificate_below_worst_gain", trial_id)
                audit.expect(certificate <= actual_gain, "certificate_below_actual_gain", trial_id)
        audit.expect(actual_gain > 0, "move_improves_actual_target", trial_id)
    elif status in ("KEEP", "INCOMPATIBLE"):
        audit.expect(output == incumbent, "abstention_keeps_incumbent", trial_id)
        if status == "INCOMPATIBLE":
            audit.expect(not feasible, "incompatible_exactly_empty", trial_id)
        else:
            audit.expect(bool(feasible), "keep_has_feasible_world", trial_id)
    if not feasible:
        audit.expect(status == "INCOMPATIBLE", "empty_set_abstains_incompatible", trial_id)
    if state == "correct":
        audit.expect(output == truth["depth"] and status == "KEEP",
                     "correct_truth_strictly_preserved", trial_id)
    baseline_keep = rational(raw["baselines"]["keep"], f"{trial_id}.baseline.keep")
    baseline_minimum = rational(raw["baselines"]["minimum_residual"],
                               f"{trial_id}.baseline.minimum_residual")
    audit.expect(baseline_keep == incumbent, "keep_baseline", trial_id)
    expected_minimum = residual_baseline(observation, family["worlds"], family["actions"], incumbent)
    audit.expect(baseline_minimum == expected_minimum, "minimum_residual_baseline", trial_id,
                 f"Expected output={expected_minimum}")
    minimum_gain = ((incumbent - truth["depth"]) ** 2
                    - (baseline_minimum - truth["depth"]) ** 2)
    for counter in (totals, family_counts[family_id], state_counts[state]):
        counter["trials"] += 1
        counter["moves"] += int(status == "MOVE")
        counter["keeps"] += int(status == "KEEP")
        counter["incompatible"] += int(status == "INCOMPATIBLE")
        counter["truth_covered"] += int(truth_id in feasible_ids)
        counter["correct_trials"] += int(state == "correct")
        counter["correct_preserved"] += int(state == "correct" and output == incumbent)
        counter["offset_trials"] += int(state != "correct")
        counter["offset_improved"] += int(state != "correct" and actual_gain > 0)
        counter["offset_exactly_recovered"] += int(state != "correct" and output == truth["depth"])
        counter["damaged"] += int(actual_gain < 0)
        counter["minimum_residual_improved"] += int(minimum_gain > 0)
        counter["minimum_residual_damaged"] += int(minimum_gain < 0)
        counter["minimum_residual_correct_damaged"] += int(state == "correct" and minimum_gain < 0)


def verify_directory(directory):
    audit = Audit()
    required = ("library.json", "trials.jsonl", "controls.json", "summary.json", "PROTOCOL_LOCK.json")
    hashes = {}
    for name in required:
        path = directory / name
        audit.expect(path.is_file(), "required_file", name)
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    totals, family_counts, state_counts = Counter(), defaultdict(Counter), defaultdict(Counter)
    families, seen_ids, seen_keys, scenes = {}, set(), set(), {}
    controls_present, summary_present, protocol_present = False, False, False
    if all((directory / name).is_file() for name in required):
        try:
            epsilon, families = parse_library(load_json(directory / "library.json"), audit)
            controls = load_json(directory / "controls.json")
            controls_present = isinstance(controls, (dict, list))
            audit.expect(controls_present, "controls_json_container", "controls.json")
            summary = load_json(directory / "summary.json")
            summary_present = isinstance(summary, dict)
            audit.expect(summary_present, "summary_json_object", "summary.json")
            protocol = load_json(directory / "PROTOCOL_LOCK.json")
            protocol_present = isinstance(protocol, dict)
            audit.expect(protocol_present, "protocol_json_object", "PROTOCOL_LOCK.json")
            with (directory / "trials.jsonl").open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        audit.expect(False, "nonempty_jsonl_line", f"line {line_number}")
                        continue
                    try:
                        raw = json.loads(line)
                        verify_trial(raw, epsilon, families, audit, scenes, seen_keys, seen_ids,
                                     totals, family_counts, state_counts)
                    except (KeyError, TypeError, ValueError, IndexError) as exc:
                        audit.malformed(f"trials.jsonl line {line_number}", exc)
            expected = {(family_id, world["world_id"], seed, state)
                        for family_id, family in families.items()
                        for world in family["worlds"] if world["depth"] in CONFIRMATION_DEPTHS
                        for seed in CONFIRMATION_SEEDS for state in STATE_OFFSETS}
            audit.expect(seen_keys == expected, "complete_confirmation_cartesian_product", "trials",
                         f"Missing={len(expected-seen_keys)}, unexpected={len(seen_keys-expected)}")
            audit.expect(totals["trials"] == len(expected), "complete_parsed_trial_count", "trials")
            audit.expect(len(scenes) == len(expected) // 3, "scene_count_not_state_count", "trials")
            audit.expect(totals["offset_improved"] > 0, "nontrivial_offset_repair_exists", "trials")
        except (KeyError, TypeError, ValueError, IndexError, OSError) as exc:
            audit.malformed("evidence files", exc)
    return {"verification_version": 1, "ok": not audit.failures,
            "scope": "Independent exact finite-library confirmation replay; controls excluded",
            "directory": str(directory.resolve()), "input_sha256": hashes,
            "checked": {"families": len(families),
                        "worlds": sum(len(family["worlds"]) for family in families.values()),
                        "scenes": len(scenes), "trials": totals["trials"],
                        "checks": dict(sorted(audit.check_counts.items()))},
            "computed": {"totals": dict(totals),
                         "by_family": {key: dict(value) for key, value in sorted(family_counts.items())},
                         "by_state": {key: dict(value) for key, value in sorted(state_counts.items())}},
            "controls": {"json_present": controls_present, "included_in_contract_counts": False},
            "summary_json_present": summary_present, "protocol_json_present": protocol_present,
            "not_checked": ["Geometry-to-library rendering replay",
                            "Source hashes against protocol lock",
                            "Reported summary metrics (use independently computed counts)",
                            "Deliberate contract-violation control safety",
                            "Continuous-world or real-image coverage"],
            "failure_count": len(audit.failures), "failures": audit.failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "confirmation-v1")
    args = parser.parse_args()
    report = verify_directory(args.directory)
    args.directory.mkdir(parents=True, exist_ok=True)
    target = args.directory / "VERIFICATION.json"
    target.write_text(json.dumps(exact_json(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "trials": report["checked"]["trials"],
                      "scenes": report["checked"]["scenes"],
                      "failure_count": report["failure_count"], "report": str(target.resolve())},
                     ensure_ascii=False))
    if report["failures"]:
        print(json.dumps(report["failures"][:10], indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
