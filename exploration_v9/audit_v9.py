"""Independent read-only audit of sealed V9 arrays and reported scores.

Imports only a previous independent metric helper, not V9/V8/V7 estimators.
This checks implementation consistency and stored split provenance. It cannot
establish physical-model validity or whole-pipeline statistical independence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "exploration_v6"))
from audit_metrics import synthetic_scores


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def jsonread(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run.resolve()
    assert socket.gethostname() == "liekkas"
    assert not (run / "AUDIT.json").exists(), "audit is append-only"
    started = time.perf_counter()
    summary = jsonread(run / "SUMMARY.json")
    records = jsonread(run / "SEALED_BEFORE_GT.json")
    sources = jsonread(run / "SOURCES.json")
    protected = jsonread(run / "PROTECTED.json")
    search_records = jsonread(run / "SEARCH_RECORDS.json")
    with (run / "RESULTS.csv").open(newline="") as source:
        csv_rows = list(csv.DictReader(source))
    rows = {row["output"]: row for row in csv_rows}
    assert len(records) == len(csv_rows) == len(rows) == summary["outputs"] == 864
    checked_hashes = {}

    def verify_hash(path, expected):
        path = str(Path(path).resolve())
        if path in checked_hashes:
            assert checked_hashes[path] == expected
            return
        assert digest(path) == expected, path
        checked_hashes[path] = expected

    for path, expected in {**sources, **protected}.items():
        verify_hash(path, expected)
    for path, expected in sources.items():
        snapshot = run / "source" / Path(path).relative_to(PROJECT)
        verify_hash(snapshot, expected)
    maximum = dict(stored_coefficient_projection_mm=0., independent_wls_prediction_mm=0.,
                   metric_mm=0., metric_fraction=0., metric_degrees=0.,
                   frozen_displacement_additivity_mm=0., constant_slope_coefficient_change=0.,
                   slope_group_weighted_mean_mm=0., normal_only_edit_mm=0.)
    cache = {}

    def cached(path):
        path = str(path)
        if path not in cache:
            cache[path] = read(path)
        return cache[path]

    indexed = {}
    fit_parameters, fit_ranks, candidate_parameters = [], [], []
    reconstruction_count = independent_fit_count = 0
    metric_count = 0
    for record in records:
        output_path = Path(record["output"])
        for field in ("output", "state", "candidate", "input"):
            verify_hash(record[field], record[field + "_sha256"])
        assert jsonread(output_path.with_suffix(".json")) == record
        row = rows[str(output_path)]
        assert row["output_sha256"] == record["output_sha256"]
        output = cached(output_path)
        state, candidate, inputs = (cached(record[key]) for key in ("state", "candidate", "input"))
        world, xyz = inputs["xyz_world"], output["xyz_world"]
        order, active = state["order"], state["active"]
        original = order[active]
        n = len(world)
        assert xyz.shape == world.shape == (n, 3) and np.isfinite(xyz).all()
        np.testing.assert_array_equal(np.sort(order), np.arange(n))
        np.testing.assert_array_equal(state["world"], world)
        np.testing.assert_array_equal(output["scan_id"], inputs["scan_id"])
        np.testing.assert_array_equal(output["source_point_index"], inputs["source_point_index"])
        support = np.zeros(n, bool); support[order] = state["support"]
        active_mask = np.zeros(n, bool); active_mask[original] = True
        assert not np.any(support & ~active_mask)
        np.testing.assert_array_equal(output["support_mask"], support)
        np.testing.assert_array_equal(candidate["support_mask"], support)
        np.testing.assert_array_equal(output["group_ids"], candidate["group_ids"])
        np.testing.assert_array_equal(xyz[~support], world[~support])
        group = output["group_ids"][original]
        assert np.all(group >= 0)
        x = state["design"][active]
        if record["action"] == "candidate":
            beta = output["output_candidate_coefficients"]
            np.testing.assert_array_equal(beta, candidate["coefficients"])
            prediction = np.sum(x * beta[group], axis=1)
        else:
            np.testing.assert_array_equal(output["active_original_indices"], original)
            ids, beta = output["coefficient_ids"], output["coefficients"]
            np.testing.assert_array_equal(np.unique(group), ids)
            np.testing.assert_array_equal(output["compensated_coefficients"], beta)
            np.testing.assert_array_equal(output["compensated_coefficient_ids"], ids)
            indices = np.searchsorted(ids, group)
            prediction = np.sum(x * beta[indices], axis=1)
            # A single block design is independent of the estimator's group loop.
            block = np.hstack([x * (group == value)[:, None] for value in ids])
            weight = np.sqrt(state["weights"][active])
            y = state["corrected"][active] - output["subtracted_noise_mm"][original]
            coefficient, _, rank, _ = np.linalg.lstsq(block * weight[:, None], y * weight, rcond=1e-12)
            error = float(np.max(abs(block @ coefficient - prediction)))
            maximum["independent_wls_prediction_mm"] = max(maximum["independent_wls_prediction_mm"], error)
            assert error < 1e-8, (record["method"], error)
            assert int(row["final_fit_parameter_count"]) == block.shape[1]
            assert int(row["final_fit_rank"]) == int(rank)
            fit_parameters.append(block.shape[1]); fit_ranks.append(int(rank))
            independent_fit_count += 1
            if record["action"] == "slope":
                for value in ids:
                    take = group == value; w = state["weights"][active][take]
                    mean = float(np.dot(w, output["subtracted_noise_mm"][original][take]) / w.sum())
                    maximum["slope_group_weighted_mean_mm"] = max(maximum["slope_group_weighted_mean_mm"], abs(mean))
                    assert abs(mean) < 1e-10
        height = state["local"][:, 2].copy(); height[active] = prediction
        supported = np.flatnonzero(state["support"])
        expected = world.copy()
        expected[order[supported]] += ((height[supported]-state["local"][supported, 2])/1000.)[:, None]*state["normal"]
        error = float(np.max(abs(expected-xyz))*1000.)
        maximum["stored_coefficient_projection_mm"] = max(maximum["stored_coefficient_projection_mm"], error)
        assert error < 1e-8
        edit = xyz-world; axis = state["normal"]/np.linalg.norm(state["normal"])
        error = float(np.max(np.linalg.norm(edit-(edit@axis)[:, None]*axis, axis=1))*1000.)
        maximum["normal_only_edit_mm"] = max(maximum["normal_only_edit_mm"], error)
        assert error < 1e-8
        reconstruction_count += 1
        candidate_parameters.append(int(candidate["coefficients"].size))
        assert int(row["candidate_parameter_count"]) == candidate["coefficients"].size
        assert int(row["group_count"]) == len(np.unique(group))
        assert abs(float(row["supported_fraction"])-support.mean()) < 1e-14

        # Evaluation GT enters only this audit of already sealed output arrays.
        evaluation_path = Path(record["input"]).parent / "evaluation" / (record["case"] + ".eval.npz")
        ev = dict(cached(evaluation_path))
        ev.update(json.loads(ev.pop("json").tobytes().decode()))
        reference, labels = ev["gt_clean_xyz_world"], ev["gt_layer"]
        scores = synthetic_scores(xyz, reference, ev["surface_rectangles_mm"], labels, ev.get("true_gap_mm"))
        z = xyz[:, 2]*1000.; means = {int(label): float(z[labels == label].mean()) for label in np.unique(labels)}
        deviation = z-np.asarray([means[int(label)] for label in labels])
        scores["within_source_layer_rms_mm"] = float(np.sqrt(np.mean(deviation**2)))
        scores["input_edit_rms_mm"] = float(1000*np.sqrt(np.mean(np.sum(edit**2, axis=1))))
        if set(means) == {0, 1}:
            gap = means[1]-means[0]
            scores["source_group_gap_mm"] = gap
            scores["source_group_gap_error_mm"] = abs(gap-float(ev["true_gap_mm"]))
        # Brute-force distances instead of the production evaluator's KD-tree.
        distances = np.concatenate([np.sqrt(cdist(reference[start:start+128], xyz, "sqeuclidean").min(axis=1))*1000.
                                    for start in range(0, len(reference), 128)])
        scores["reference_sample_coverage_1mm"] = float(np.mean(distances <= 1. + 1e-9))
        tilt_sum = tilt_count = 0.
        for label in np.unique(labels):
            points = xyz[labels == label]*1000.
            if len(points) < 6: continue
            design = np.column_stack((points[:, :2], np.ones(len(points))))
            if np.linalg.matrix_rank(design) != 3: continue
            fit = np.linalg.lstsq(design, points[:, 2], rcond=None)[0]
            tilt_sum += len(points)*float(np.degrees(np.arctan(np.linalg.norm(fit[:2]))))
            tilt_count += len(points)
        if tilt_count:
            scores["source_surface_tilt_mean_deg"] = tilt_sum/tilt_count
        for key, value in scores.items():
            if row.get(key) in (None, ""): continue
            delta = abs(value-float(row[key]))
            which = "metric_fraction" if "coverage" in key else "metric_degrees" if "deg" in key else "metric_mm"
            maximum[which] = max(maximum[which], delta)
            assert delta < 1e-8, (record["method"], key, delta)
            metric_count += 1
        indexed[(record["case"], record["budget"], record["stage"], record["action"])] = output

    additive_cases = constant_cases = 0
    for case in sorted({record["case"] for record in records}):
        for budget in (1, 3, 6):
            for stage in ("frozen", "alltrain", "crossfit"):
                none = indexed[(case, budget, stage, "none")]
                constant = indexed[(case, budget, stage, "constant")]
                np.testing.assert_array_equal(none["coefficient_ids"], constant["coefficient_ids"])
                change = float(np.max(abs(none["coefficients"][:, 1:]-constant["coefficients"][:, 1:])))
                maximum["constant_slope_coefficient_change"] = max(maximum["constant_slope_coefficient_change"], change)
                assert change == 0.
                constant_cases += 1
            all_actions = {action: indexed[(case, budget, "frozen", action)]["xyz_world"]
                           for action in ("none", "constant", "slope", "full")}
            residual = (all_actions["full"]-all_actions["none"])-(all_actions["constant"]-all_actions["none"])-(all_actions["slope"]-all_actions["none"])
            error = float(np.max(abs(residual))*1000.)
            maximum["frozen_displacement_additivity_mm"] = max(maximum["frozen_displacement_additivity_mm"], error)
            assert error < 1e-8
            additive_cases += 1

    split_audits, overlap, fallback_groups = [], 0, 0
    for record in search_records:
        verify_hash(record["path"], record["sha256"])
        candidate = cached(record["path"])
        state_record = next(item for item in records if item["candidate"] == record["path"])
        state = cached(state_record["state"])
        original = state["order"][state["active"]]
        active_mask = np.zeros(len(state["world"]), bool); active_mask[original] = True
        train, query = candidate["candidate_training_mask"], candidate["candidate_query_mask"]
        assert train.shape == query.shape == (record["folds"], len(active_mask))
        np.testing.assert_array_equal(query.sum(axis=0), active_mask.astype(int))
        assert not np.any(train[:, ~active_mask])
        actual_overlap = int(np.sum(train & query))
        if record["folds"] == 2:
            assert actual_overlap == 0
            np.testing.assert_array_equal(train[0], query[1])
            np.testing.assert_array_equal(train[1], query[0])
            overlap += actual_overlap
        else:
            np.testing.assert_array_equal(train, query)
        gcount = record["dictionary_groups_per_fold"]
        assert candidate["coefficients"].shape == (gcount*record["folds"], 3)
        for fold in range(record["folds"]):
            selection = query[fold]
            assert np.all(candidate["fold_id"][selection] == fold)
            chosen = candidate["group_ids"][selection]
            assert np.all((chosen >= fold*gcount) & (chosen < (fold+1)*gcount))
            assert not candidate["candidate_mask"][selection, :fold*gcount].any()
            assert not candidate["candidate_mask"][selection, (fold+1)*gcount:].any()
            indices = np.flatnonzero(selection)
            assert candidate["candidate_mask"][indices, chosen].all()
            diagnostic = record["fold_diagnostics"][fold]
            assert diagnostic["train_count"] == int(train[fold].sum())
            assert diagnostic["query_count"] == int(query[fold].sum())
            assert diagnostic["train_query_overlap"] == int(np.sum(train[fold] & query[fold]))
            assert len(diagnostic["initial_seed_fallback_groups"]) == sum(rank != 3 for rank in diagnostic["initial_fit_rank"])
        fallbacks = sum(len(value["initial_seed_fallback_groups"]) for value in record["fold_diagnostics"])
        assert fallbacks == record["fallback_count"]
        fallback_groups += fallbacks
        split_audits.append(dict(case=record["case"], stage=record["stage"], budget=record["budget"],
            folds=record["folds"], fallback_groups=fallbacks,
            train_query_overlap=actual_overlap, parameters=int(candidate["coefficients"].size)))

    # Recheck all previously inspected files at audit completion.
    for path, expected in checked_hashes.items():
        assert digest(path) == expected, path
    report = dict(outputs=len(records),conditions=len({r["case"] for r in records}),new_independent_data=False,
        snapshot_source_files=len(sources), protected_files=len(protected), checked_file_hashes=len(checked_hashes),
        coordinate_and_id_order_checks=len(records), coefficient_output_reconstructions=reconstruction_count,
        independent_block_wls_fits=independent_fit_count, metric_value_recomputations=metric_count,
        metric_output_recomputations=len(records), constant_slope_checks=constant_cases,
        frozen_additivity_checks=additive_cases, candidate_split_checks=len(split_audits),
        crossfit_train_query_overlap=overlap, full_data_seed_fallback_groups=fallback_groups,
        candidate_parameter_range=[min(candidate_parameters), max(candidate_parameters)],
        final_fit_parameter_range=[min(fit_parameters), max(fit_parameters)],
        final_fit_rank_range=[min(fit_ranks), max(fit_ranks)], maximum_errors=maximum,
        hashes_unchanged=True, scope="numeric implementation, stored provenance, and evaluator checks only; not physical-model validity",
        independence_scope="split is conditional on frozen full-data upstream; this audit does not certify whole-pipeline independence",
        sealed_manifest_scope="all 864 output hashes match stored pre-evaluation seal; timing/absence-of-GT access is not inferred from hashes alone",
        split_audits=split_audits,seconds=time.perf_counter()-started,
        audit_source=str(Path(__file__).resolve()),audit_sha256=digest(__file__))
    with (run / "AUDIT.json").open("x") as target:
        json.dump(report, target, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != "split_audits"}, indent=2))


if __name__ == "__main__":
    main()
