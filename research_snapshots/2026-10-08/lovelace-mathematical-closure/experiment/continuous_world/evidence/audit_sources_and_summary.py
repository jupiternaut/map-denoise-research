"""Independent readback of archived Stage-2 receipts, statistics and controls.

This is an evidence auditor, not a renderer/search/decision experiment. It imports
no project modules. Its one point-photo calculation checks the existing violated
noise-budget control by separate rational half-plane clipping. The reviewed
continuous box-enclosure proof and separate full-trace checker remain necessary.

Run from any directory: python -B path/to/audit_sources_and_summary.py
Only continuous_world/outputs/confirmation-v1/AUDIT.json is written.
"""

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction as Q
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "continuous_world/outputs/confirmation-v1"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q(value):
    if isinstance(value, dict) and set(value) == {"numerator", "denominator"}:
        return Q(value["numerator"], value["denominator"])
    return Q(value)


def encode(value):
    if isinstance(value, Q):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def round4(value):
    with localcontext() as context:
        context.prec = 80
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
        return str(decimal.quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN))


def row_key(row):
    return tuple(row[k] for k in ("kind", "camera_radius_mm", "state", "method"))


def clip_axis(polygon, axis, edge, positive):
    """Separate exact clipping kernel; no project geometry implementation used."""
    if not polygon:
        return []
    result = []
    previous = polygon[-1]
    sign = 1 if positive else -1
    previous_distance = (previous[axis] - edge) * sign
    for current in polygon:
        distance = (current[axis] - edge) * sign
        if (previous_distance >= 0) != (distance >= 0):
            t = previous_distance / (previous_distance - distance)
            result.append(tuple(previous[j] + t * (current[j] - previous[j])
                                for j in range(2)))
        if distance >= 0:
            result.append(current)
        previous, previous_distance = current, distance
    return result


def polygon_area(polygon):
    if len(polygon) < 3:
        return Q(0)
    return abs(sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
                   - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
                   for i in range(len(polygon)))) / 2


def control_triangle_image(vertices, depth, offsets):
    values = []
    du, dv = Q(1, 20), Q(2, 75)
    pixel_area = du * dv
    for cx, offset in zip((-70, 0, 70), offsets):
        triangle = [((x - cx - offset) / (depth + dz), y / (depth + dz))
                    for x, y, dz in vertices]
        for row in range(6):
            bottom = Q(-2, 25) + row * dv
            for column in range(8):
                left = Q(-1, 5) + column * du
                polygon = triangle
                for axis, edge, positive in ((0, left, True), (0, left + du, False),
                                             (1, bottom, True), (1, bottom + dv, False)):
                    polygon = clip_axis(polygon, axis, edge, positive)
                shade = 1 - polygon_area(polygon) / pixel_area
                values.extend([shade] * 3)
    return values


def main():
    checks = []

    def check(name, condition, details=None):
        item = {"check": name, "passed": bool(condition)}
        if details is not None:
            item["details"] = encode(details)
        checks.append(item)

    source = read("SOURCE_LOCK.json")
    forecast_lock = read("FORECAST_LOCK.json")
    checker_lock = read("CHECKER_LOCK.json")
    protocol, summary = read("PROTOCOL.json"), read("summary.json")
    controls, verification = read("controls.json"), read("VERIFICATION.json")
    forecasts = read("BASELINE_FORECASTS.json")
    scenes = [json.loads(line) for line in
              (OUT / "scenes.jsonl").read_text(encoding="utf-8-sig").splitlines()
              if line.strip()]
    source_records = []
    for relative, expected in source["hashes"].items():
        current = digest(ROOT / relative)
        archived = digest(OUT / "source" / relative)
        record = {"path": relative, "locked": expected, "current": current,
                  "source_snapshot": archived}
        source_records.append(record)
        check("source_hash:" + relative, current == archived == expected, record)
    check("forecast_lock", digest(OUT / "BASELINE_FORECASTS.json") == forecast_lock["sha256"])
    check("checker_lock", digest(ROOT / checker_lock["checker"]) == checker_lock["sha256"])
    check("checker_receipt_scope", checker_lock["scope"] ==
          "independent_trace_checker_locked_before_execution")
    check("verification_passed", verification["passed"] is True and verification["failures"] == [])
    check("checker_does_not_call_solver_or_aggregator", set(verification["does_not_call"]) ==
          {"solve_outer", "decide", "run_experiment"})
    check("protocol_scope", protocol["scope"] == summary["scope"] ==
          "continuous_depth_and_bounded_lateral_camera_positions_only")
    check("protocol_contract", list(map(q, protocol["depth_domain_mm"])) == [Q(540), Q(660)]
          and q(protocol["epsilon"]) == Q(1, 100)
          and q(protocol["tolerance_mm"]) == Q(15, 32) and protocol["max_boxes"] == 511)
    check("protocol_baseline_scope", "integer grid 540..660 plus incumbent; KEEP ties"
          in protocol["baseline"])
    check("protocol_evaluator_only", "never enter solve_outer or decide" in protocol["evaluator_only"])
    expected = set(itertools.product(protocol["kinds"], map(q, protocol["camera_radii_mm"]),
                                    map(q, protocol["truth_depths_mm"]), protocol["seeds"]))
    actual = [(s["kind"], q(s["camera_radius_mm"]), q(s["truth_depth_mm"]), s["seed"])
              for s in scenes]
    check("scene_configuration_cartesian_product", len(actual) == len(set(actual)) == 72
          and set(actual) == expected)
    check("confirmation_count_excludes_controls", len(scenes) == summary["scene_configurations"] == 72
          and sum(len(s["decisions"]) for s in scenes) == summary["decisions"] == 216)

    groups, totals = defaultdict(list), defaultdict(Counter)
    boxes, unresolved = 0, 0
    offset_errors = []
    for scene in scenes:
        kind, radius = scene["kind"], q(scene["camera_radius_mm"])
        truth, epsilon = q(scene["truth_depth_mm"]), q(scene["epsilon"])
        outer = scene["outer"]
        low, high = map(q, outer["hull"])
        width = high - low
        boxes += outer["evaluated_boxes"]
        unresolved += sum(n["status"] == "unresolved_budget" for n in outer["trace"])
        offsets = list(map(q, scene["actual_offsets_mm"]))
        contract = (epsilon == q(protocol["epsilon"]) and len(scene["observation"]) == 432
                    and offsets[1] == 0 and abs(offsets[0]) <= radius and abs(offsets[2]) <= radius)
        state_counts = Counter(d["state"] for d in scene["decisions"])
        incumbents = all(q(d["incumbent"]) == truth + {"correct": 0, "minus60": -60, "plus60": 60}[d["state"]]
                         for d in scene["decisions"])
        check("scene_contract:" + scene["scene_id"], contract and incumbents
              and state_counts == Counter(correct=1, minus60=1, plus60=1))
        check("scene_outer_hull_contains_true_target:" + scene["scene_id"], low <= truth <= high)
        for decision in scene["decisions"]:
            incumbent, state = q(decision["incumbent"]), decision["state"]
            certificate = decision["certificate"]
            outputs = {"certificate": q(certificate["output"]),
                       **{method: q(value) for method, value in decision["baselines"].items()}}
            suffix = scene["scene_id"] + ":" + state
            check("keep_baseline:" + suffix, outputs["keep"] == incumbent)
            output, gain = outputs["certificate"], q(certificate["gain_lower"])
            actual_gain = (incumbent - truth) ** 2 - (output - truth) ** 2
            if certificate["status"] == "MOVE":
                endpoint_gain = min((incumbent - low) ** 2 - (output - low) ** 2,
                                    (incumbent - high) ** 2 - (output - high) ** 2)
                valid = output == (low + high) / 2 and gain == endpoint_gain > 0 and actual_gain >= gain
            elif certificate["status"] == "KEEP":
                valid = output == incumbent and gain == 0
            else:
                valid = False
            check("scalar_certificate:" + suffix, valid)
            if state != "correct":
                offset_errors.append(abs(output - truth))
            for method, output in outputs.items():
                new_error, old_error = abs(output - truth), abs(incumbent - truth)
                groups[(kind, str(radius), state, method)].append(
                    (new_error, old_error, output != incumbent, width))
                t = totals[(state, method)]
                t["count"] += 1
                t["moved"] += output != incumbent
                t["improved"] += new_error < old_error
                t["worse"] += new_error > old_error
                t["same"] += new_error == old_error

    rows = []
    for (kind, radius, state, method), values in sorted(groups.items()):
        n = len(values)
        rows.append({"kind": kind, "camera_radius_mm": radius, "state": state, "method": method,
                     "count": n, "improved": sum(new < old for new, old, moved, w in values),
                     "worse": sum(new > old for new, old, moved, w in values),
                     "same": sum(new == old for new, old, moved, w in values),
                     "moved": sum(moved for new, old, moved, w in values),
                     "mae_mm": sum((new for new, old, moved, w in values), Q(0)) / n,
                     "mse_mm2": sum((new * new for new, old, moved, w in values), Q(0)) / n,
                     "mean_hull_width_mm": sum((w for new, old, moved, w in values), Q(0)) / n})
    stored_rows = {row_key(row): row for row in summary["rows"]}
    check("summary_group_keys", len(summary["rows"]) == len(stored_rows) == 54
          and set(stored_rows) == set(map(row_key, rows)))
    for row in rows:
        check("summary_row:" + ":".join(row_key(row)), stored_rows.get(row_key(row)) == encode(row))
    check("summary_evaluated_boxes", boxes == summary["evaluated_boxes_total"] == 1866)
    check("summary_unresolved_leaves", unresolved == summary["unresolved_budget_leaves"] == 0)
    check("correct_certificate_all_keep", totals[("correct", "certificate")] ==
          Counter(count=72, moved=0, improved=0, worse=0, same=72))
    check("offset_certificate_all_improve", all(totals[(s, "certificate")] ==
          Counter(count=72, moved=72, improved=72, worse=0, same=0) for s in ("minus60", "plus60")))
    check("nominal_grid_correct_damage", totals[("correct", "nominal_grid")]["worse"] == 37
          and totals[("correct", "nominal_grid")]["count"] == 72)
    counts = {"certificate_KEEP": 72, "certificate_MOVE": 144, "certificate_damage": 0,
              "correct_KEEP": 72, "correct_states": 72, "decisions": 216, "evaluated_boxes": boxes,
              "offset_improved": 144, "offset_states": 144, "scene_configurations": 72,
              "unresolved_budget_leaves": unresolved,
              "trace_nodes": sum(len(s["outer"]["trace"]) for s in scenes),
              "excluded_boxes": sum(n["status"] == "excluded" for s in scenes for n in s["outer"]["trace"])}
    check("independent_counts_match_checker", all(verification["counts"].get(k) == v for k, v in counts.items()), counts)
    checker_groups = {row_key(row): row for row in verification["groups"]}
    check("independent_group_counts_match_checker", len(checker_groups) == 54 and all(
        checker_groups.get(row_key(row)) == {k: v for k, v in row.items()
        if k not in ("mae_mm", "mse_mm2", "mean_hull_width_mm")} for row in rows))

    asset = json.loads((ROOT / "finite_world/assets/lovelace_patch.json").read_text(encoding="utf-8-sig"))
    vertices = tuple(tuple(map(Q, vertex)) for vertex in asset["abstraction"]["scaled_centered_seed_vertices"])
    check("asset_gauge_and_positive_z", all(sum(v[j] for v in vertices) == 0 for j in range(3))
          and all(Q(540) + v[2] > 0 for v in vertices))
    check("protocol_source_provenance", all(m["source_sha256"] == asset["source"]["sha256"]
          for m in protocol["models"] if m["kind"] == "lovelace_triangle"))
    check("forecast_structure", set(forecasts) == {"lovelace_triangle", "rectangle"} and all(
        len(fs) == 121 and [q(e["depth"]) for e in fs] == list(map(Q, range(540, 661)))
        and all(len(e["prediction"]) == 432 for e in fs) for fs in forecasts.values()))
    check("control_names_separate", len(controls) == 2 and {c["name"] for c in controls} ==
          {"wrong_world_photo_exceeds_noise_budget", "zero_search_budget_retains_root"})
    control_notes = []
    for control in controls:
        if control["name"] == "wrong_world_photo_exceeds_noise_budget":
            truth, incumbent, epsilon = map(q, (control["truth_depth_mm"], control["incumbent"], control["epsilon"]))
            y = list(map(q, control["observation"]))
            actual_image = control_triangle_image(vertices, truth, list(map(q, control["actual_offsets_mm"])))
            residual = max(abs(a - b) for a, b in zip(y, actual_image))
            check("control_independent_exact_noise_linf", len(actual_image) == len(y) == 432
                  and residual == q(control["actual_noise_linf"]), residual)
            check("control_noise_violates_contract", residual > epsilon and
                  control["contract"] == "violated_photometric_budget", {"noise_linf": residual, "epsilon": epsilon})
            foreign = next(e for e in forecasts["lovelace_triangle"] if q(e["depth"]) == 645)
            check("control_observation_equals_foreign_depth_photo", y == list(map(q, foreign["prediction"])))
            output = q(control["decision"]["output"])
            gain = (incumbent - truth) ** 2 - (output - truth) ** 2
            low, high = map(q, control["outer"]["hull"])
            endpoint_gain = min((incumbent - low) ** 2 - (output - low) ** 2,
                                (incumbent - high) ** 2 - (output - high) ** 2)
            check("control_wrong_model_certificate_vs_actual_gain", control["decision"]["status"] == "MOVE"
                  and output == 645 and endpoint_gain == q(control["decision"]["gain_lower"]) > 0
                  and gain == q(control["actual_gain_mm2"]) == Q(-352836, 49) and not low <= truth <= high,
                  {"truth_mm": truth, "incumbent_mm": incumbent, "output_mm": output,
                   "hull_mm": [low, high], "certificate_gain_mm2": endpoint_gain, "actual_gain_mm2": gain})
            control_notes.append({"name": control["name"], "in_confirmation_counts": False,
                                  "noise_linf": residual, "epsilon": epsilon, "truth_covered": False,
                                  "actual_gain_mm2": gain, "meaning": "A positive gain on an incorrect outer set cannot protect a world excluded by a violated noise budget."})
        else:
            outer = control["outer"]
            nodes = outer["trace"]
            node = nodes[0] if nodes else {}
            output = q(control["decision"]["output"])
            check("control_zero_budget_retains_complete_root", outer["max_boxes"] == outer["evaluated_boxes"] == 0
                  and len(nodes) == 1 and node.get("id") == 0 and node.get("parent") is None
                  and node.get("status") == "unresolved_budget"
                  and [q(node["lo"]), q(node["hi"])] == [Q(540), Q(660)]
                  and outer["retained_ids"] == [0] and list(map(q, outer["hull"])) == [Q(540), Q(660)])
            check("control_zero_budget_keeps_correct_point", control["decision"]["status"] == "KEEP"
                  and output == q(protocol["truth_depths_mm"][0]) and q(control["decision"]["gain_lower"]) == 0
                  and Q(540) <= output <= Q(660))
            control_notes.append({"name": control["name"], "in_confirmation_counts": False, "root_retained": True,
                                  "status": control["decision"]["status"], "meaning": "Zero work retains the whole root and returns KEEP; its archived partition check is observation independent because no observation is stored."})

    plan = " ".join((ROOT / "continuous_world/EXPERIMENT_PLAN.md").read_text(encoding="utf-8-sig").split())
    theory = " ".join((ROOT / "continuous_world/THEORY.md").read_text(encoding="utf-8-sig").split())
    check("scope_finite_confirmation_not_continuous_sampling_proof", "continuous contract covers MORE" in plan
          and "not the logical basis of the enclosure theorem" in theory)
    check("scope_outer_not_joint_feasibility", "**not** thereby proved jointly feasible" in theory)
    check("scope_no_full9_or_full_character_claim", "reproduction of published map-denoise full9" in plan
          and "does not describe the whole LOVELACE character" in theory)
    check("scope_unresolved_must_be_retained", "unresolved boxes must be retained" in plan
          and "preserve unresolved boxes" in theory)
    check("scope_no_continuous_kappa_from_finite", "do not imply a continuous kappa or coverage theorem" in theory)

    report_path = ROOT / "continuous_world/REPORT.zh.md"
    prose = report_path.read_text(encoding="utf-8-sig")
    table = [tuple(c.strip() for c in line.strip().strip("|").split("|")) for line in prose.splitlines()
             if line.startswith("| LOVELACE 面片 |") or line.startswith("| 矩形对照 |")]
    row_map = {row_key(row): row for row in rows}
    report_rows = []
    check("report_table_six_rows", len(table) == 6)
    for label, radius_string, certificate_mae, nominal_mae, width_string, damage in table:
        kind = {"LOVELACE 面片": "lovelace_triangle", "矩形对照": "rectangle"}[label]
        radius = str(Q(radius_string))
        def offset_mae(method):
            selected = [row_map[(kind, radius, state, method)] for state in ("minus60", "plus60")]
            return sum((r["mae_mm"] * r["count"] for r in selected), Q(0)) / sum(r["count"] for r in selected)
        width = row_map[(kind, radius, "correct", "certificate")]["mean_hull_width_mm"]
        correct_grid = row_map[(kind, radius, "correct", "nominal_grid")]
        actual_row = (round4(offset_mae("certificate")), round4(offset_mae("nominal_grid")),
                      round4(width), f'{correct_grid["worse"]}/{correct_grid["count"]}')
        check("report_table:" + kind + ":" + radius, actual_row ==
              (certificate_mae, nominal_mae, width_string, damage), {"independent_rounded_values": actual_row})
        report_rows.append({"kind": kind, "camera_radius_mm": radius, "rounded_values": actual_row})
    check("report_offset_max_error", round4(max(offset_errors)) == "3.3192"
          and "3.3192 mm" in prose, {"exact_max_mm": max(offset_errors)})
    example = next(s for s in scenes if s["kind"] == "lovelace_triangle" and q(s["camera_radius_mm"]) == 1
                   and s["seed"] == 2000 and q(s["truth_depth_mm"]) == Q(3921, 7))
    check("report_numeric_example", list(map(q, example["outer"]["hull"])) == [Q(8955, 16), Q(4485, 8)]
          and all(q(d["certificate"]["output"]) == Q(17925, 32) for d in example["decisions"] if d["state"] != "correct"))
    check("report_claims_bounded", all(text in prose for text in (
        "并非精确恢复真值", "不是 72 个独立物理对象", "不是现有 map-denoise full9 的复现",
        "它复用了经过数学审阅的几何内核", "并非无条件正确", "尚无 Lean 编译证明")))

    replay = OUT.parent / "replay-v1"
    reproducibility = read("REPRODUCIBILITY.json")
    check("reproduction_receipt_passed", reproducibility["passed"] is True)
    for name, receipt in reproducibility["mathematical_data"].items():
        check("reproduction_bytes:" + name, receipt["byte_equal"] is True and digest(OUT / name) ==
              receipt["sha256"] == digest(replay / name) == receipt["replay_sha256"])
    replay_source = json.loads((replay / "SOURCE_LOCK.json").read_text(encoding="utf-8-sig"))
    check("reproduction_source_hash_maps", source["hashes"] == replay_source["hashes"]
          and reproducibility["algorithm_source_hash_maps_equal"] is True)
    replay_summary = json.loads((replay / "summary.json").read_text(encoding="utf-8-sig"))
    check("reproduction_summary_without_time", {k: v for k, v in summary.items() if k != "elapsed_seconds"} ==
          {k: v for k, v in replay_summary.items() if k != "elapsed_seconds"}
          and reproducibility["summary_equal_excluding_elapsed_seconds"] is True)
    replay_protocol = json.loads((replay / "PROTOCOL.json").read_text(encoding="utf-8-sig"))
    def remove_absolute_asset(value):
        if isinstance(value, dict):
            return {k: remove_absolute_asset(v) for k, v in value.items() if k != "source_asset"}
        if isinstance(value, list):
            return [remove_absolute_asset(v) for v in value]
        return value
    check("reproduction_protocol_without_absolute_asset_path", remove_absolute_asset(protocol) ==
          remove_absolute_asset(replay_protocol) and reproducibility["protocol_equal_excluding_absolute_asset_path"] is True)
    tests = read("TESTS.json")
    log = (OUT / "tests.log").read_text(encoding="utf-8-sig")
    check("tests_receipt_and_log", tests["passed"] is True and tests["exit_code"] == 0
          and tests["expected_tests"] == 33 and digest(OUT / "tests.log") == tests["log_sha256"]
          and "Ran 33 tests" in log and log.rstrip().endswith("OK"))
    check("tests_source_hashes", all(digest(ROOT / relative) == value == source["hashes"][relative]
          for relative, value in tests["test_hashes"].items()))

    failures = [item for item in checks if not item["passed"]]
    names = ["SOURCE_LOCK.json", "PROTOCOL.json", "FORECAST_LOCK.json", "BASELINE_FORECASTS.json",
             "scenes.jsonl", "summary.json", "controls.json", "CHECKER_LOCK.json", "VERIFICATION.json",
             "REPRODUCIBILITY.json", "TESTS.json", "tests.log", "STAGE1_PRESERVATION.json"]
    result = {"schema": 1, "passed": not failures, "audit_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Independent frozen-source/forecast/checker receipts, exact archived-scene reaggregation, control readback, report numbers, and frozen replay readback; no new confirmation run.",
              "audit_source": "continuous_world/evidence/audit_sources_and_summary.py",
              "audit_source_sha256": digest(Path(__file__)),
              "does_not_import_or_call": ["continuous_world.run_experiment", "run_experiment.aggregate",
                  "continuous_world.interval_solver", "solve_outer", "decide", "continuous_world.interval_model",
                  "Model", "finite_world.renderer", "finite_world.mesh_patch"],
              "independent_control_geometry": "Separate exact rational projection, axis half-plane clipping and shoelace area of three archived scaled-centred vertices; only verifies the archived violated-noise-budget control.",
              "checks": checks, "failures": failures, "source_records": source_records,
              "input_sha256": {name: digest(OUT / name) for name in names},
              "report_sha256": digest(report_path), "recomputed_counts": counts, "recomputed_groups": rows,
              "state_method_totals": [{"state": state, "method": method, **dict(t)}
                                      for (state, method), t in sorted(totals.items())],
              "controls": control_notes, "report_table": report_rows,
              "mathematical_boundaries": [
                  "Safety is conditional on the declared world class, positive Z throughout every cell, correct enclosures, and the hard l-infinity observation budget. The violation control demonstrates this dependence.",
                  "All real depths and independent lateral offsets in the declared intervals are covered by mathematical enclosures. Confirmation samples only three truth depths, five-level generated offsets/noise, and four seeds per condition.",
                  "Retained cells are an outer approximation, not jointly feasible-world witnesses. Unresolved cells must remain; a hull width is not a statistical confidence interval.",
                  "The certificate concerns one scalar reference-ray action. It does not certify a moved surface, same-layer assignment, global topology or full-character reconstruction.",
                  "LOVELACE supplies one isolated M4 seed facet. Its abstract-world millimetres come from a uniform rescaling of source coordinates, not physical calibration; the rectangle is a separate analytic toy.",
                  "Only independent side-camera lateral x errors are covered. Reference gauge, intrinsics, scale, material, illumination, geometry and background are fixed; rotations and arbitrary occlusion are excluded.",
                  "72 parameter/noise configurations produce 216 initial-state decisions sharing 72 observations. They are not 72 independent physical objects or 216 independent observations.",
                  "The nominal baseline is a 1 mm grid with nominal cameras plus incumbent and KEEP ties. Its 37/72 correct-point damage is protocol-specific; no full9 reproduction, continuous global optimum or universal method ranking follows.",
                  "Stage-1 finite separation and kappa do not establish a continuous stability constant. No global continuous kappa is claimed.",
                  "File hashes establish present identity with archived receipts, not externally attested historical chronology or protection against coordinated replacement of every receipt.",
                  "The separate trace checker reuses the reviewed geometry kernel; it is not an independently implemented continuous enclosure. This audit independently reaggregates statistics and verifies one control photo, not every box enclosure.",
                  "The replay uses the same frozen source and same conditions; it verifies reproducibility, not generalization to new scenes. Recorded wall times are not independently measured by this audit."],
              "not_checked": ["new experiment replication", "physical error calibration", "Lean formal proof",
                  "full-character or map-denoise full9 behavior", "independent implementation of every continuous enclosure",
                  "external timestamp attestation", "elapsed-time measurement", "reverification of stage-1 ZIP preservation"]}
    (OUT / "AUDIT.json").write_text(json.dumps(encode(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": len(checks), "failures": failures,
                      "audit": str(OUT / "AUDIT.json"), "counts": counts}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
