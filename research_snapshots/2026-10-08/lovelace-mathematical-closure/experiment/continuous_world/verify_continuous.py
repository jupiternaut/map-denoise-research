"""Independent replay of continuous outer-set partition and action evidence.

The checker never calls solve_outer or decide. It reuses the separately reviewed
Model enclosure primitive; this is NOT a second independent geometric renderer.
Its proof obligations are complete root partition, justified exclusions,
retention of unfinished work, and the scalar endpoint gain formula.
"""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

from .interval_model import Model


def rational(raw, context):
    if not isinstance(raw, dict) or set(raw) != {"numerator", "denominator"}:
        raise ValueError(f"{context}: expected exact rational JSON")
    n, d = raw["numerator"], raw["denominator"]
    if isinstance(n, bool) or isinstance(d, bool) or not isinstance(n, int) or not isinstance(d, int) or d <= 0:
        raise ValueError(f"{context}: integer numerator/positive denominator required")
    return Q(n, d)


def pair(raw, context):
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{context}: exact interval pair required")
    lo, hi = (rational(value, context) for value in raw)
    if lo > hi:
        raise ValueError(f"{context}: inverted interval")
    return lo, hi


def encode(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def vector(raw, context):
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context}: nonempty exact observation required")
    return tuple(rational(value, context) for value in raw)


class Audit:
    def __init__(self):
        self.failures, self.checks = [], Counter()

    def expect(self, condition, check, context, detail=""):
        self.checks[check] += 1
        if not condition:
            self.failures.append({"check": check, "context": context, "detail": detail})

    def error(self, context, exc):
        self.failures.append({"check": "schema", "context": context,
                              "detail": f"{type(exc).__name__}: {exc}"})


def verify_outer(observation, model, outer, audit, context):
    """Reconstruct the entire partition and verify every claimed exclusion."""
    domain = pair(outer["domain"], f"{context}.domain")
    epsilon = rational(outer["epsilon"], f"{context}.epsilon")
    tolerance = rational(outer["tolerance"], f"{context}.tolerance")
    max_boxes = outer["max_boxes"]
    audit.expect(epsilon >= 0 and tolerance > 0, "valid_error_and_tolerance", context)
    audit.expect(isinstance(max_boxes, int) and not isinstance(max_boxes, bool) and max_boxes >= 0,
                 "valid_work_budget", context)
    nodes = {}
    for raw in outer["trace"]:
        node_id = raw["id"]
        if not isinstance(node_id, int) or isinstance(node_id, bool) or node_id < 0:
            raise ValueError("Node id must be a nonnegative integer")
        if node_id in nodes:
            raise ValueError("Duplicate trace node id")
        lo, hi = rational(raw["lo"], context), rational(raw["hi"], context)
        audit.expect(domain[0] <= lo <= hi <= domain[1], "node_inside_root_domain", f"{context}/{node_id}")
        nodes[node_id] = {"raw": raw, "lo": lo, "hi": hi}
    if 0 not in nodes:
        raise ValueError("Partition root id 0 is missing")
    audit.expect(nodes[0]["raw"]["parent"] is None, "root_has_no_parent", context)
    audit.expect((nodes[0]["lo"], nodes[0]["hi"]) == domain, "root_exact_domain", context)
    reached, stack, incoming = set(), [0], Counter()
    while stack:
        node_id = stack.pop()
        if node_id in reached:
            audit.expect(False, "tree_without_shared_nodes_or_cycles", f"{context}/{node_id}")
            continue
        reached.add(node_id)
        node = nodes[node_id]
        raw, lo, hi = node["raw"], node["lo"], node["hi"]
        status = raw["status"]
        audit.expect(status in ("split", "excluded", "retained_tolerance", "unresolved_budget"),
                     "valid_node_status", f"{context}/{node_id}")
        if status == "split":
            children = raw["children"]
            if not isinstance(children, list) or len(children) != 2 or len(set(children)) != 2:
                raise ValueError("Split must list two distinct children")
            if any(child not in nodes for child in children):
                raise ValueError("Split child missing from archive")
            left, right = (nodes[child] for child in children)
            middle = (lo + hi) / 2
            audit.expect((left["lo"], left["hi"]) == (lo, middle)
                         and (right["lo"], right["hi"]) == (middle, hi),
                         "children_exactly_partition_parent", f"{context}/{node_id}")
            audit.expect(hi - lo > tolerance, "split_above_tolerance", f"{context}/{node_id}")
            for child in children:
                incoming[child] += 1
                audit.expect(nodes[child]["raw"]["parent"] == node_id,
                             "child_parent_agrees", f"{context}/{child}")
                stack.append(child)
        else:
            audit.expect(not raw.get("children"), "leaf_has_no_children", f"{context}/{node_id}")
    audit.expect(reached == set(nodes), "no_orphan_or_omitted_partition_nodes", context)
    audit.expect(all(incoming[node_id] == 1 for node_id in nodes if node_id != 0),
                 "one_parent_per_nonroot_node", context)

    retained, evaluated, exclusion_count, unfinished = [], 0, 0, 0
    for node_id, node in nodes.items():
        raw, lo, hi = node["raw"], node["lo"], node["hi"]
        status = raw["status"]
        if status == "unresolved_budget":
            retained.append(node_id)
            unfinished += 1
            continue
        evaluated += 1
        bounds = tuple(model.enclosure(lo, hi))
        audit.expect(len(bounds) == len(observation), "model_bound_dimension", f"{context}/{node_id}")
        if len(bounds) != len(observation):
            raise ValueError("Enclosure dimension differs from observation")
        audit.expect(all(isinstance(a, Q) and isinstance(b, Q) and a <= b for a, b in bounds),
                     "exact_ordered_model_bounds", f"{context}/{node_id}")
        separated = [index for index, (a, b) in enumerate(bounds)
                     if a > observation[index] + epsilon or b < observation[index] - epsilon]
        if status == "excluded":
            exclusion_count += 1
            channel = raw["witness_channel"]
            if not isinstance(channel, int) or isinstance(channel, bool) or not 0 <= channel < len(bounds):
                raise ValueError("Invalid exclusion witness channel")
            archived = pair(raw["prediction_interval"], f"{context}/{node_id}.witness")
            audit.expect(archived == bounds[channel], "archived_exclusion_bound_recomputed", f"{context}/{node_id}")
            audit.expect(channel in separated, "strict_observation_interval_separation", f"{context}/{node_id}")
        elif status in ("split", "retained_tolerance"):
            audit.expect(not separated, "nonexcluded_box_necessary_consistency", f"{context}/{node_id}")
            if status == "retained_tolerance":
                audit.expect(hi - lo <= tolerance, "retained_box_within_tolerance", f"{context}/{node_id}")
                retained.append(node_id)
    audit.expect(outer["evaluated_boxes"] == evaluated <= max_boxes,
                 "evaluated_box_count_and_budget", context)
    if unfinished:
        audit.expect(evaluated == max_boxes, "unfinished_only_after_budget_exhaustion", context)
    claimed = outer["retained_ids"]
    audit.expect(isinstance(claimed, list) and len(claimed) == len(set(claimed))
                 and set(claimed) == set(retained), "all_unfinished_and_retained_leaves_preserved", context)
    expected_hull = None if not retained else (min(nodes[node]["lo"] for node in retained),
                                               max(nodes[node]["hi"] for node in retained))
    archived_hull = None if outer["hull"] is None else pair(outer["hull"], f"{context}.hull")
    audit.expect(archived_hull == expected_hull, "hull_equals_all_retained_leaves", context)
    return {"hull": expected_hull, "retained": [(nodes[node]["lo"], nodes[node]["hi"]) for node in retained],
            "epsilon": epsilon, "nodes": len(nodes), "excluded": exclusion_count,
            "unfinished": unfinished, "evaluated": evaluated}


def verify_decision(checked_outer, incumbent, decision, audit, context,
                    actual_truth=None, proposal=None):
    """Replay midpoint policy and its endpoint proof without importing decide."""
    output = rational(decision["output"], f"{context}.output")
    gain = None if decision["gain_lower"] is None else rational(decision["gain_lower"], f"{context}.gain")
    interval = checked_outer["hull"]
    status = decision["status"]
    audit.expect(status in ("MOVE", "KEEP", "INCOMPATIBLE"), "valid_decision_status", context)
    if interval is None:
        audit.expect(status == "INCOMPATIBLE" and output == incumbent and gain is None,
                     "empty_outer_set_abstains", context)
    else:
        lo, hi = interval
        candidate = (lo + hi) / 2 if proposal is None else proposal
        minimum = min((incumbent - lo) ** 2 - (candidate - lo) ** 2,
                      (incumbent - hi) ** 2 - (candidate - hi) ** 2)
        if candidate != incumbent and minimum > 0:
            audit.expect(status == "MOVE" and output == candidate,
                         "midpoint_policy_positive_gain_move", context)
            audit.expect(gain == minimum and gain > 0,
                         "exact_strict_endpoint_gain_certificate", context)
        else:
            audit.expect(status == "KEEP" and output == incumbent and gain == 0,
                         "no_positive_midpoint_gain_keeps", context)
        if status == "MOVE":
            true_worst = min((incumbent - lo) ** 2 - (output - lo) ** 2,
                             (incumbent - hi) ** 2 - (output - hi) ** 2)
            audit.expect(true_worst > 0 and gain is not None and 0 < gain <= true_worst,
                         "move_safe_over_entire_continuous_hull", context)
    actual_gain = None
    if actual_truth is not None:
        covered = any(lo <= actual_truth <= hi for lo, hi in checked_outer["retained"])
        audit.expect(covered, "evaluation_truth_depth_in_retained_outer_set", context)
        actual_gain = (incumbent - actual_truth) ** 2 - (output - actual_truth) ** 2
        audit.expect(actual_gain >= 0, "covered_actual_target_not_damaged", context)
        if status == "MOVE" and gain is not None:
            audit.expect(actual_gain >= gain > 0, "actual_gain_covers_certificate", context)
        if incumbent == actual_truth:
            audit.expect(status == "KEEP" and output == incumbent,
                         "correct_point_strictly_preserved", context)
    return {"status": status, "output": output, "gain": gain, "actual_gain": actual_gain}


def linf(left, right):
    if not left or len(left) != len(right):
        raise ValueError("Nonempty matching observation dimensions required")
    return max(abs(a - b) for a, b in zip(left, right))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_archive(archive):
    """Check archived evidence; evaluator truth is used only for coverage/results."""
    audit = Audit()
    counts = Counter()
    groups = defaultdict(Counter)
    control_results = []
    required = ("PROTOCOL.json", "SOURCE_LOCK.json", "BASELINE_FORECASTS.json",
                "FORECAST_LOCK.json", "scenes.jsonl", "controls.json", "summary.json")
    for name in required:
        audit.expect((archive / name).is_file(), "required_archive_file", name)
    if audit.failures:
        return make_report(audit, counts, groups, control_results)
    try:
        protocol = read_json(archive / "PROTOCOL.json")
        domain = pair(protocol["depth_domain_mm"], "protocol.depth")
        epsilon = rational(protocol["epsilon"], "protocol.epsilon")
        tolerance = rational(protocol["tolerance_mm"], "protocol.tolerance")
        max_boxes = protocol["max_boxes"]
        kinds = tuple(protocol["kinds"])
        radii = tuple(rational(value, "protocol.radius") for value in protocol["camera_radii_mm"])
        truths = tuple(rational(value, "protocol.truth") for value in protocol["truth_depths_mm"])
        seeds = tuple(protocol["seeds"])
        audit.expect(protocol["schema"] == 1 and protocol["scope"] ==
                     "continuous_depth_and_bounded_lateral_camera_positions_only",
                     "fixed_protocol_scope", "protocol")
        audit.expect(domain == (Q(540), Q(660)) and epsilon == Q(1, 100)
                     and tolerance == Q(15, 32) and max_boxes == 511,
                     "fixed_protocol_domain_and_budgets", "protocol")
        audit.expect(kinds == ("lovelace_triangle", "rectangle")
                     and radii == (Q(0), Q(1, 5), Q(1))
                     and truths == (Q(3921, 7), Q(4205, 7), Q(4477, 7))
                     and seeds == tuple(range(2000, 2004)),
                     "fixed_confirmation_parameter_product", "protocol")
        if len(set(kinds)) != len(kinds) or len(set(radii)) != len(radii) or len(set(truths)) != len(truths) or len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate protocol parameter")
        models = {(kind, radius): Model(kind, radius) for kind in kinds for radius in radii}
        audit.expect(all(model.depth_range == domain for model in models.values()),
                     "model_depth_contract_agrees", "protocol")
        audit.expect(all(domain[0] <= truth <= domain[1] and truth.denominator > 1
                         and truth.denominator & (truth.denominator - 1) != 0 for truth in truths),
                     "truths_are_interior_nongrid_nondyadic_depths", "protocol")
        raw_forecasts = read_json(archive / "BASELINE_FORECASTS.json")
        lock = read_json(archive / "FORECAST_LOCK.json")
        audit.expect(hashlib.sha256((archive / "BASELINE_FORECASTS.json").read_bytes()).hexdigest() == lock["sha256"],
                     "forecast_receipt_hash_matches", "forecasts")
        audit.expect(set(raw_forecasts) == set(kinds), "forecast_kinds_exact", "forecasts")
        forecasts = {}
        for kind in kinds:
            rows = [(rational(row["depth"], "forecast.depth"), vector(row["prediction"], "forecast.prediction"))
                    for row in raw_forecasts[kind]]
            audit.expect(tuple(z for z, _ in rows) == tuple(Q(z) for z in range(540, 661)),
                         "forecast_integer_grid_exact", kind)
            nominal = Model(kind, Q(0))
            for z, prediction in rows:
                audit.expect(prediction == nominal.point_prediction(z),
                             "forecast_nominal_prediction_recomputed", f"{kind}/{z}")
            forecasts[kind] = rows
        expected = {(kind, radius, truth, seed) for kind in kinds for radius in radii
                    for truth in truths for seed in seeds}
        seen, scene_ids = set(), set()
        with (archive / "scenes.jsonl").open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                context = f"scenes:{line_number}"
                try:
                    scene = json.loads(line)
                    kind = scene["kind"]
                    radius = rational(scene["camera_radius_mm"], context)
                    truth = rational(scene["truth_depth_mm"], context)
                    seed = scene["seed"]
                    key = (kind, radius, truth, seed)
                    audit.expect(key in expected and key not in seen, "scene_parameter_product_unique", context)
                    seen.add(key)
                    audit.expect(scene["scene_id"] == f"{kind}:{radius}:{truth}:{seed}"
                                 and scene["scene_id"] not in scene_ids, "scene_id_exact_and_unique", context)
                    scene_ids.add(scene["scene_id"])
                    model = models[(kind, radius)]
                    offsets = vector(scene["actual_offsets_mm"], context + ".offsets")
                    audit.expect(len(offsets) == 3 and offsets[1] == 0
                                 and abs(offsets[0]) <= radius and abs(offsets[2]) <= radius,
                                 "actual_pose_inside_continuous_nuisance_contract", context)
                    audit.expect(domain[0] <= truth <= domain[1] and truth.denominator > 1
                                 and truth.denominator & (truth.denominator - 1) != 0,
                                 "actual_truth_depth_nongrid_inside_domain", context)
                    observation = vector(scene["observation"], context + ".observation")
                    audit.expect(len(observation) == model.observation_size == 432,
                                 "fixed_three_view_image_dimension", context)
                    audit.expect(rational(scene["epsilon"], context) == epsilon,
                                 "scene_noise_budget_agrees", context)
                    true_prediction = model.point_prediction(truth, offsets)
                    noise = linf(observation, true_prediction)
                    audit.expect(noise <= epsilon, "actual_observation_noise_covered", context)
                    outer = scene["outer"]
                    audit.expect(pair(outer["domain"], context) == domain
                                 and rational(outer["epsilon"], context) == epsilon
                                 and rational(outer["tolerance"], context) == tolerance
                                 and outer["max_boxes"] == max_boxes,
                                 "scene_solver_budgets_match_protocol", context)
                    checked = verify_outer(observation, model, outer, audit, context)
                    audit.expect(any(lo <= truth <= hi for lo, hi in checked["retained"]),
                                 "scene_truth_covered_by_retained_partition", context)
                    counts["scene_configurations"] += 1
                    counts["trace_nodes"] += checked["nodes"]
                    counts["evaluated_boxes"] += checked["evaluated"]
                    counts["excluded_boxes"] += checked["excluded"]
                    counts["unresolved_budget_leaves"] += checked["unfinished"]
                    expected_states = {"correct": Q(0), "minus60": Q(-60), "plus60": Q(60)}
                    states = [row["state"] for row in scene["decisions"]]
                    audit.expect(len(states) == 3 and set(states) == set(expected_states),
                                 "all_three_initial_states_share_one_observation", context)
                    # Per-scene residual forecasts are shared by the three states.
                    grid_scores = [(linf(prediction, observation), z) for z, prediction in forecasts[kind]]
                    for row in scene["decisions"]:
                        state = row["state"]
                        incumbent = rational(row["incumbent"], context)
                        decision_context = f"{context}/{state}"
                        audit.expect(incumbent == truth + expected_states[state],
                                     "incumbent_state_exact", decision_context)
                        checked_decision = verify_decision(checked, incumbent, row["certificate"], audit,
                                                           decision_context, actual_truth=truth)
                        output = checked_decision["output"]
                        audit.expect(output == incumbent or domain[0] <= output <= domain[1],
                                     "move_output_inside_allowed_depth_domain", decision_context)
                        keep = rational(row["baselines"]["keep"], decision_context)
                        grid = rational(row["baselines"]["nominal_grid"], decision_context)
                        scored = grid_scores + [(linf(model.point_prediction(incumbent), observation), incumbent)]
                        best = min(loss for loss, _ in scored)
                        tied = {z for loss, z in scored if loss == best}
                        expected_grid = incumbent if incumbent in tied else min(tied, key=lambda z: (abs(z - incumbent), z))
                        audit.expect(keep == incumbent, "keep_baseline_unchanged", decision_context)
                        audit.expect(grid == expected_grid, "nominal_residual_baseline_tie_policy", decision_context)
                        counts["decisions"] += 1
                        counts["certificate_" + checked_decision["status"]] += 1
                        if state == "correct":
                            counts["correct_states"] += 1
                            counts["correct_KEEP"] += int(checked_decision["status"] == "KEEP" and output == truth)
                        else:
                            counts["offset_states"] += 1
                            counts["offset_improved"] += int(abs(output - truth) < abs(incumbent - truth))
                        counts["certificate_damage"] += int(abs(output - truth) > abs(incumbent - truth))
                        for method, result in (("certificate", output), ("keep", keep), ("nominal_grid", grid)):
                            group = groups[(kind, str(radius), state, method)]
                            group["count"] += 1
                            group["moved"] += int(result != incumbent)
                            group["improved"] += int(abs(result - truth) < abs(incumbent - truth))
                            group["worse"] += int(abs(result - truth) > abs(incumbent - truth))
                            group["same"] += int(abs(result - truth) == abs(incumbent - truth))
                except Exception as exc:
                    audit.error(context, exc)
        audit.expect(seen == expected and len(scene_ids) == len(expected) == 72,
                     "all_72_parameter_noise_configurations_present", "scenes")
        audit.expect(counts["decisions"] == 216 and counts["correct_states"] == 72
                     and counts["offset_states"] == 144,
                     "fixed_confirmation_decision_counts", "scenes")
        control_results = verify_controls(read_json(archive / "controls.json"), audit, domain, epsilon)
    except Exception as exc:
        audit.error("archive", exc)
    return make_report(audit, counts, groups, control_results)


def verify_controls(controls, audit, domain, epsilon):
    """Keep deliberate contract violations outside confirmation result counts."""
    if not isinstance(controls, list):
        raise ValueError("Controls must be a separate list")
    by_name = {row["name"]: row for row in controls}
    names = {"wrong_world_photo_exceeds_noise_budget", "zero_search_budget_retains_root"}
    audit.expect(set(by_name) == names and len(controls) == 2, "separate_control_names", "controls")
    results = []
    wrong = by_name["wrong_world_photo_exceeds_noise_budget"]
    model = Model(wrong["kind"], rational(wrong["camera_radius_mm"], "control.radius"))
    truth = rational(wrong["truth_depth_mm"], "control.truth")
    offsets = vector(wrong["actual_offsets_mm"], "control.offsets")
    observation = vector(wrong["observation"], "control.observation")
    actual_prediction = model.point_prediction(truth, offsets)
    noise = linf(observation, actual_prediction)
    audit.expect(wrong["contract"] == "violated_photometric_budget"
                 and rational(wrong["epsilon"], "control.epsilon") == epsilon
                 and noise == rational(wrong["actual_noise_linf"], "control.noise")
                 and noise > epsilon, "control_noise_budget_intentionally_violated", "wrong_world")
    checked = verify_outer(observation, model, wrong["outer"], audit, "wrong_world")
    incumbent = rational(wrong["incumbent"], "control.incumbent")
    decision = verify_decision(checked, incumbent, wrong["decision"], audit, "wrong_world")
    covered = any(lo <= truth <= hi for lo, hi in checked["retained"])
    actual_gain = (incumbent - truth) ** 2 - (decision["output"] - truth) ** 2
    audit.expect(actual_gain == rational(wrong["actual_gain_mm2"], "control.gain"),
                 "violated_control_actual_gain_recomputed", "wrong_world")
    audit.expect(incumbent == truth and not covered and actual_gain < 0 and decision["status"] == "MOVE",
                 "invalid_budget_control_exposes_missing_coverage", "wrong_world")
    results.append({"name": wrong["name"], "in_confirmation_counts": False,
                    "noise_linf": encode(noise), "truth_covered": covered,
                    "status": decision["status"], "actual_gain_mm2": encode(actual_gain)})
    zero = by_name["zero_search_budget_retains_root"]
    audit.expect(zero["contract"] == "in_contract_termination_control"
                 and zero["outer"]["max_boxes"] == 0
                 and pair(zero["outer"]["domain"], "zero.domain") == domain,
                 "zero_work_control_budget_and_domain", "zero_work")
    # No observations are archived in this control. With zero evaluations,
    # retention is observation independent: structural replay uses any vector.
    zero_outer = verify_outer(tuple(Q(0) for _ in range(model.observation_size)), model,
                              zero["outer"], audit, "zero_work")
    audit.expect(zero_outer["hull"] == domain and zero_outer["retained"] == [domain]
                 and zero_outer["evaluated"] == 0 and zero_outer["unfinished"] == 1,
                 "zero_work_retains_complete_root_for_any_observation", "zero_work")
    zero_decision = verify_decision(zero_outer, truth, zero["decision"], audit, "zero_work", actual_truth=truth)
    results.append({"name": zero["name"], "in_confirmation_counts": False,
                    "status": zero_decision["status"], "root_retained": True,
                    "observation_check": "not archived; zero-work partition check is observation independent"})
    return results


def make_report(audit, counts, groups, controls):
    return {"schema": 1, "passed": not audit.failures,
            "scope": "archived continuous outer partition and scalar certificate evidence",
            "geometry_kernel_reused": "Model.enclosure and Model.point_prediction; separately mathematically reviewed, not a second independent geometry implementation",
            "does_not_call": ["solve_outer", "decide", "run_experiment"],
            "checks": dict(sorted(audit.checks.items())), "failures": audit.failures,
            "counts": dict(sorted(counts.items())),
            "groups": [{"kind": key[0], "camera_radius_mm": key[1], "state": key[2], "method": key[3], **dict(value)}
                       for key, value in sorted(groups.items())],
            "controls_separate": controls,
            "not_checked": ["source snapshot hashes and summary aggregation (separate independent audit)",
                            "independent reimplementation of geometric enclosure mathematics",
                            "forecast chronological ordering beyond frozen file receipt (source audit)",
                            "physical camera/lighting/model error coverage",
                            "full character mesh reconstruction or optimal action search"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "confirmation-v1")
    archive = parser.parse_args().archive.resolve()
    report = verify_archive(archive)
    if archive.is_dir():
        output = archive / "VERIFICATION.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"passed": report["passed"], "failure_count": len(report["failures"]),
                          "counts": report["counts"], "report": str(output)}, ensure_ascii=False))
    else:
        print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
