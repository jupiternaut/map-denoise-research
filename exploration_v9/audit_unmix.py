"""Independent SVD and geometry audit of sealed moment-unmixing outputs.

Does not import or call any estimator. Reconstructs the saved design and uses
an explicit truncated SVD rather than the estimator's least-squares routine.
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
    parser = argparse.ArgumentParser(); parser.add_argument("run", type=Path)
    run = parser.parse_args().run.resolve()
    assert socket.gethostname() == "liekkas"
    assert not (run / "AUDIT.json").exists(), "audit output is append-only"
    started = time.perf_counter()
    records = jsonread(run / "SEALED_BEFORE_GT.json")
    sources, protected = jsonread(run / "SOURCES.json"), jsonread(run / "PROTECTED.json")
    with (run / "RESULTS.csv").open(newline="") as source:
        all_rows = list(csv.DictReader(source))
    rows = {row["output"]: row for row in all_rows}
    assert len(records) == len(all_rows) == len(rows) == jsonread(run / "SUMMARY.json")["outputs"] == 144
    hashes, cached_files = {}, {}

    def verify(path, expected):
        path = str(Path(path).resolve())
        if path in hashes:
            assert hashes[path] == expected
            return
        assert digest(path) == expected, path
        hashes[path] = expected

    def cached(path):
        path = str(path)
        if path not in cached_files: cached_files[path] = read(path)
        return cached_files[path]

    for path, expected in {**sources, **protected}.items(): verify(path, expected)
    for path, expected in sources.items(): verify(run / "source" / Path(path).relative_to(PROJECT), expected)
    maximum = dict(delta_coefficient_absolute=0., coefficient_absolute=0.,
        singular_value_absolute=0., condition_relative=0., projection_mm=0.,
        sse_absolute_mm2=0., sse_increase_mm2=0., metric_mm=0., metric_fraction=0.,
        metric_degrees=0., probability_row_sum=0., normal_only_edit_mm=0.)
    rank_rows, conditions, low_rows, parameter_counts, metric_count = [], [], [], [], 0
    sse_reductions, mode_records = [], []
    for record in records:
        path = Path(record["output"])
        verify(path, record["output_sha256"]); verify(record["candidate"], record["candidate_sha256"])
        assert str(Path(record["input"]).resolve()) in protected
        assert str(Path(record["state"]).resolve()) in protected
        assert jsonread(path.with_suffix(".json")) == record
        out, state, inputs, candidate = (cached(record[key]) for key in ("output", "state", "input", "candidate"))
        info, row = record["info"], rows[str(path)]
        assert row["output_sha256"] == record["output_sha256"]
        world, xyz = inputs["xyz_world"], out["xyz_world"]
        order, active = state["order"], state["active"]
        original = order[active]; n = len(world)
        np.testing.assert_array_equal(np.sort(order), np.arange(n))
        np.testing.assert_array_equal(state["world"], world)
        np.testing.assert_array_equal(out["scan_id"], inputs["scan_id"])
        np.testing.assert_array_equal(out["source_point_index"], inputs["source_point_index"])
        np.testing.assert_array_equal(out["active_original_indices"], original)
        np.testing.assert_array_equal(out["group_ids"], candidate["group_ids"])
        np.testing.assert_array_equal(out["seed_coefficients"], candidate["coefficients"])
        support = np.zeros(n, bool); support[order] = state["support"]
        np.testing.assert_array_equal(out["support_mask"], support)
        np.testing.assert_array_equal(out["support_mask"], candidate["support_mask"])
        assert xyz.shape == world.shape == (n, 3) and np.isfinite(xyz).all()
        np.testing.assert_array_equal(xyz[~support], world[~support])
        x, y, weights = state["design"][active], state["corrected"][active], state["weights"][active]
        mass = out["conditional_event_probability"]
        assert mass.shape == (len(active),)
        assert np.isfinite(mass).all() and np.all((mass >= 0.) & (mass <= 1.+1e-12))
        low = mass <= 1e-14
        low_world = np.zeros(n, bool); low_world[original] = low
        np.testing.assert_array_equal(out["low_probability_mask"], low_world)
        np.testing.assert_array_equal(xyz[low_world], world[low_world])
        fit = ~low & (weights > 0.)
        fit_world = np.zeros(n, bool); fit_world[original] = fit
        np.testing.assert_array_equal(out["fit_row_mask"], fit_world)
        assert info["fit_rows"] == int(fit.sum())
        assert info["low_probability_rows"] == int(low.sum())
        assert info["zero_weight_rows"] == int(np.sum(weights == 0.))
        mixture = out["conditional_component_probability"][original]
        assert np.isfinite(mixture).all() and np.all(mixture >= 0.)
        assert np.all(mixture[~candidate["candidate_mask"][original]] == 0.)
        if (~low).any():
            error = float(np.max(abs(mixture[~low].sum(axis=1)-1.)))
            maximum["probability_row_sum"] = max(maximum["probability_row_sum"], error)
            assert error < 1e-12
        np.testing.assert_array_equal(mixture[low], np.zeros_like(mixture[low]))
        seed = candidate["coefficients"]
        if record["mode"] == "offset":
            design = mixture
        elif record["mode"] == "affine":
            design = np.column_stack([mixture[:, group]*x[:, column]
                for group in range(len(seed)) for column in range(x.shape[1])])
        else:
            raise AssertionError("unknown mode")
        prior_prediction = np.sum(mixture*(x@seed.T), axis=1)
        response = y-out["subtracted_noise_mm"][original]-prior_prediction
        a, rhs = design[fit]*np.sqrt(weights[fit])[:, None], response[fit]*np.sqrt(weights[fit])
        if fit.any():
            u, singular, vt = np.linalg.svd(a, full_matrices=False)
            keep = singular > (1e-8*singular[0])
            delta = vt[keep].T @ ((u[:, keep].T@rhs)/singular[keep])
        else:
            singular = np.empty(0); keep = np.empty(0, bool); delta = np.zeros(design.shape[1])
        rank = int(keep.sum())
        assert rank == info["fit_rank"] and design.shape[1]-rank == info["numerical_nullity"]
        assert design.shape[1] == info["fit_parameter_count"]
        change = np.zeros_like(seed)
        if record["mode"] == "offset": change[:, 0] = delta
        else: change[:] = delta.reshape(seed.shape)
        expected_beta = seed+change
        error = float(np.max(abs(change-out["coefficient_delta"])))
        maximum["delta_coefficient_absolute"] = max(maximum["delta_coefficient_absolute"], error)
        np.testing.assert_allclose(change, out["coefficient_delta"], atol=1e-8, rtol=1e-9)
        error = float(np.max(abs(expected_beta-out["coefficients"])))
        maximum["coefficient_absolute"] = max(maximum["coefficient_absolute"], error)
        np.testing.assert_allclose(expected_beta, out["coefficients"], atol=1e-8, rtol=1e-9)
        if len(singular):
            error = float(np.max(abs(singular-out["singular_values"])))
            maximum["singular_value_absolute"] = max(maximum["singular_value_absolute"], error)
        np.testing.assert_allclose(singular, out["singular_values"], atol=1e-10, rtol=1e-9)
        condition = float(singular[0]/singular[keep][-1]) if rank else None
        if condition is not None:
            error = abs(condition-info["effective_condition"])/max(1., condition)
            maximum["condition_relative"] = max(maximum["condition_relative"], error)
            assert error < 1e-8
            conditions.append(condition)
        else: assert info["effective_condition"] is None
        if record["mode"] == "offset":
            np.testing.assert_array_equal(out["coefficients"][:, 1:], seed[:, 1:])
        before, after = float(rhs@rhs), float(np.sum((rhs-a@delta)**2))
        assert after <= before+1e-8*max(1., before)
        for value, key in ((before, "weighted_moment_sse_before_mm2"), (after, "weighted_moment_sse_after_mm2")):
            error = abs(value-info[key]); maximum["sse_absolute_mm2"] = max(maximum["sse_absolute_mm2"], error)
            assert error < 1e-8*max(1., value)
            assert abs(float(row[key])-info[key]) < 1e-12
        maximum["sse_increase_mm2"] = max(maximum["sse_increase_mm2"], after-before)
        sse_reductions.append(before-after)
        groups = out["group_ids"][original]
        predicted = np.sum(x*expected_beta[groups], axis=1)
        movable = ~low & support[original]
        expected = world.copy()
        expected[original[movable]] += ((predicted[movable]-state["local"][active, 2][movable])/1000.)[:, None]*state["normal"]
        error = float(np.max(abs(expected-xyz))*1000.)
        maximum["projection_mm"] = max(maximum["projection_mm"], error)
        assert error < 1e-8
        edit = xyz-world; axis = state["normal"]/np.linalg.norm(state["normal"])
        error = float(np.max(np.linalg.norm(edit-(edit@axis)[:, None]*axis, axis=1))*1000.)
        maximum["normal_only_edit_mm"] = max(maximum["normal_only_edit_mm"], error)
        assert error < 1e-8

        # GT is used only for scoring these existing sealed arrays.
        evaluation_path = Path(record["input"]).parent/"evaluation"/(record["case"]+".eval.npz")
        ev = dict(cached(evaluation_path)); ev.update(json.loads(ev.pop("json").tobytes().decode()))
        reference, labels = ev["gt_clean_xyz_world"], ev["gt_layer"]
        scores = synthetic_scores(xyz, reference, ev["surface_rectangles_mm"], labels, ev.get("true_gap_mm"))
        z = xyz[:, 2]*1000.; means = {int(label): float(z[labels == label].mean()) for label in np.unique(labels)}
        deviation = z-np.asarray([means[int(label)] for label in labels])
        scores["within_source_layer_rms_mm"] = float(np.sqrt(np.mean(deviation**2)))
        scores["input_edit_rms_mm"] = float(1000*np.sqrt(np.mean(np.sum(edit**2, axis=1))))
        if set(means) == {0, 1}:
            gap = means[1]-means[0]
            scores["source_group_gap_mm"] = gap; scores["source_group_gap_error_mm"] = abs(gap-float(ev["true_gap_mm"]))
        distances = np.concatenate([np.sqrt(cdist(reference[start:start+128], xyz, "sqeuclidean").min(axis=1))*1000.
            for start in range(0, len(reference), 128)])
        scores["reference_sample_coverage_1mm"] = float(np.mean(distances <= 1.+1e-9))
        tilt_sum = tilt_count = 0.
        for label in np.unique(labels):
            points = xyz[labels == label]*1000.; tangent = np.column_stack((points[:, :2], np.ones(len(points))))
            if len(points) < 6 or np.linalg.matrix_rank(tangent) != 3: continue
            coefficient = np.linalg.lstsq(tangent, points[:, 2], rcond=None)[0]
            tilt_sum += len(points)*float(np.degrees(np.arctan(np.linalg.norm(coefficient[:2])))); tilt_count += len(points)
        if tilt_count: scores["source_surface_tilt_mean_deg"] = tilt_sum/tilt_count
        for key, value in scores.items():
            if row.get(key) in (None, ""): continue
            error = abs(value-float(row[key]))
            target = "metric_fraction" if "coverage" in key else "metric_degrees" if "deg" in key else "metric_mm"
            maximum[target] = max(maximum[target], error)
            assert error < 1e-8, (record["method"], key, error)
            metric_count += 1
        rank_rows.append(rank); low_rows.append(int(low.sum())); parameter_counts.append(design.shape[1])
        mode_records.append(dict(case=record["case"],budget=record["budget"],mode=record["mode"],
            fit_rank=rank,parameters=design.shape[1],condition=condition,low_probability_rows=int(low.sum()),
            sse_before_mm2=before,sse_after_mm2=after))
    for path, expected in hashes.items(): assert digest(path) == expected, path
    report = dict(outputs=len(records), conditions=len({r["case"] for r in records}),
        explicit_truncated_svd_reconstructions=len(records), metric_output_recomputations=len(records),
        metric_value_recomputations=metric_count, source_snapshot_files=len(sources),protected_files=len(protected),
        checked_hashes=len(hashes),hashes_unchanged=True,maximum_errors=maximum,
        svd_relative_cutoff=1e-8,rank_range=[min(rank_rows),max(rank_rows)],
        parameter_range=[min(parameter_counts),max(parameter_counts)],
        effective_condition_range=[min(conditions),max(conditions)] if conditions else None,
        low_probability_rows_total=sum(low_rows),low_probability_rows_max=max(low_rows),
        minimum_moment_sse_reduction_mm2=min(sse_reductions),all_moment_sse_nonincreasing=True,
        scope="numeric consistency of a frozen working-model update, not proof of physical correctness or unbiased moments",
        saved_probability_scope="validity, accessibility, and normalization checked; probability model is not calibrated by this audit",
        seal_scope="output hashes match the saved pre-evaluation manifest; no claim that hashes alone prove absence of earlier GT access",
        per_output=mode_records,seconds=time.perf_counter()-started,
        audit_source=str(Path(__file__).resolve()),audit_sha256=digest(__file__))
    with (run/"AUDIT.json").open("x") as target: json.dump(report,target,indent=2,allow_nan=False)
    print(json.dumps({key:value for key,value in report.items() if key != "per_output"},indent=2))


if __name__ == "__main__": main()
