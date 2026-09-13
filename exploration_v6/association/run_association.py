"""Frozen, bounded evaluation-only 2x2; no generation or candidate selection."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(PROJECT), str(PROJECT/"exploration_v3")]
import association_counterfactual as model
from evaluate_v2 import synthetic_geometry
from metrics import structure_metrics

RUNS = Path("/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1")
INPUTS = RUNS/"repair-v2-oyuie4pl"/"synthetics"/"identifiable"
PRIMARY = ("dual_g4_s912101_b4", "dual_g8_s912101_b4", "ghost_s912101_b4")
CASES = tuple(f"{family}_s{seed}_b{bias}" for seed in (912101, 912113, 912127)
              for family in ("ghost", "dual_g2", "dual_g4", "dual_g8") for bias in (0, 4))
METRICS = ("surface_accuracy_mean_mm", "matched_point_rms_mm", "fitted_gap_at_same_xy_error_mm")
REPLAY_TOL_M = 1e-12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluation_path(case):
    return INPUTS/"evaluation"/(case+".eval.npz")


def read_evaluation(case):
    # Called only after all four fits have returned.
    with np.load(evaluation_path(case), allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files if key != "json"}
        if "json" in data:
            result.update(json.loads(data["json"].tobytes().decode()))
    return result


def source_paths():
    own = [HERE/name for name in ("PROTOCOL.md", "association_counterfactual.py", "run_association.py",
                                 "verify_association.py", "test_association.py")]
    return own+[model.V5_PATH, model.V5._V4_PATH, model.V5._V4._V3_PATH,
                PROJECT/"evaluate_v2.py", PROJECT/"exploration_v3"/"metrics.py"]


def protected_paths():
    paths = set(source_paths())
    for directory in (PROJECT/"exploration_v3", PROJECT/"exploration_v4", PROJECT/"exploration_v5"):
        paths.update(p for p in directory.rglob("*") if p.is_file())
    for case in CASES:
        paths.update((INPUTS/(case+".npz"), INPUTS/(case+".json"), evaluation_path(case),
                      evaluation_path(case).with_suffix(".json")))
        for method, run_name in (("pool_compatible", "exploration-v4-dev-e0c_n4hk"),
                                 ("shared_group_slope", "exploration-v5-development-vdbkdouk")):
            path = RUNS/run_name/"outputs"/(case+"__"+method+".npz")
            if not path.exists():
                raise FileNotFoundError(path)
            paths.update((path, path.with_suffix(".json")))
    return sorted(paths)


def contrasts(rows):
    result = []
    for case in CASES:
        lookup = {row["arm"]: row for row in rows if row["case"] == case}
        record = dict(case=case, primary=case in PRIMARY, gap_mm=lookup[model.ARMS[0]]["gap_mm"],
                      bias_rms_mm=lookup[model.ARMS[0]]["bias_rms_mm"])
        for metric in METRICS:
            if metric not in lookup[model.ARMS[0]]:
                continue
            original = lookup["original_shared"][metric]-lookup["original_independent"][metric]
            oracle = lookup["oracle_shared"][metric]-lookup["oracle_independent"][metric]
            record.update({metric+"__original_sharing_loss": original,
                           metric+"__oracle_sharing_loss": oracle,
                           metric+"__difference_in_differences": oracle-original})
        result.append(record)
    return result


def aggregates(rows, comparisons):
    def summarize(subset):
        metrics = {}
        for arm in model.ARMS:
            armrows = [r for r in subset if r["arm"] == arm]
            metrics[arm] = {key: float(np.mean([r[key] for r in armrows if key in r]))
                            for key in METRICS if any(key in r for r in armrows)}
        cases = {r["case"] for r in subset}
        contrast = [r for r in comparisons if r["case"] in cases]
        changes = {}
        for key in METRICS:
            field = key+"__difference_in_differences"
            values = np.array([r[field] for r in contrast if field in r])
            if not len(values):
                continue
            changes[key] = dict(case_count=len(values), mean=float(values.mean()),
                                negative=int(np.sum(values < -1e-8)), tie=int(np.sum(abs(values) <= 1e-8)),
                                positive=int(np.sum(values > 1e-8)))
        return dict(input_count=len(cases), mean_metrics=metrics, difference_in_differences=changes)
    groups = {"primary": summarize([r for r in rows if r["primary"]]), "all_24_public": summarize(rows),
              "dual_18_public": summarize([r for r in rows if r["gap_mm"] > 0])}
    for gap in (0, 2, 4, 8):
        groups[f"gap_{gap}"] = summarize([r for r in rows if r["gap_mm"] == gap])
    for bias in (0, 4):
        groups[f"bias_{bias}"] = summarize([r for r in rows if r["bias_rms_mm"] == bias])
    return groups


def run():
    if socket.gethostname().split(".")[0] != "liekkas":
        raise RuntimeError("exact target must be liekkas")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(key+" must be 1")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("PYTHONDONTWRITEBYTECODE must be 1")
    assert len(CASES) == 24 and set(PRIMARY).issubset(CASES)
    started = time.perf_counter()
    before = {str(p): digest(p) for p in protected_paths()}
    dest = Path(tempfile.mkdtemp(prefix="association-v6-", dir=RUNS))
    print(dest, flush=True)
    save_json(dest/"PROTECTED_BEFORE.json", before)
    sources = {str(p): digest(p) for p in source_paths()}
    save_json(dest/"SOURCE_MANIFEST.json", sources)
    for source in source_paths():
        target = dest/"source"/source.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(HERE/"PROTOCOL.md", dest/"PROTOCOL.md")
    for directory in ("inputs", "states", "outputs"):
        (dest/directory).mkdir()
    tested = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_association.py"],
                            cwd=HERE, capture_output=True, text=True)
    with (dest/"TESTS.log").open("x") as stream:
        stream.write(tested.stdout+tested.stderr)
    if tested.returncode != 0:
        raise AssertionError("contract tests failed; see TESTS.log")
    rows, replays, input_manifest = [], [], []
    for case in CASES:
        path = INPUTS/(case+".npz")
        meta = json.loads((INPUTS/(case+".json")).read_text())
        with np.load(path, allow_pickle=False) as data:
            world, scans = data["xyz_world"].copy(), data["scan_id"].copy()
            point_ids = data["source_point_index"].copy()
        shutil.copyfile(path, dest/"inputs"/(case+".npz"))
        input_manifest.append(dict(case=case, input=str(path), input_sha256=digest(path),
                                   evaluation=str(evaluation_path(case)), evaluation_sha256=digest(evaluation_path(case)),
                                   metadata=str(INPUTS/(case+".json")), primary=case in PRIMARY))
        state = model.freeze(world, scans, 1.)
        # Original association fits finish without loading any evaluator file.
        results = {"original_"+sharing: model.fit_original(state, sharing) for sharing in model.SHARING}
        # EVALUATION-ONLY boundary: only this field may reach oracle fitting.
        with np.load(evaluation_path(case), allow_pickle=False) as data:
            labels = data["gt_layer"].copy()
        results.update({"oracle_"+sharing: model.fit_oracle_split(state, labels, sharing) for sharing in model.SHARING})
        assert state["common_sha256"] == model.common_fingerprints(state)
        # No full evaluator data exists above this point.
        ev = read_evaluation(case)
        common_arrays = {key: value for key, value in state.items() if isinstance(value, np.ndarray) and key != "baseline"}
        with (dest/"states"/(case+".npz")).open("xb") as stream:
            np.savez_compressed(stream, **common_arrays, common_scale_mm=state["common_scale"])
        save_json(dest/"states"/(case+".json"), dict(fingerprints=state["common_sha256"],
                  state_npz_sha256=digest(dest/"states"/(case+".npz")), v4_info=state["v4_info"],
                  sigma_mm=1., input=str(path), total_fit_weight=float(state["weights"][state["active"]].sum())))
        support = np.empty(len(world), dtype=bool)
        support[state["order"]] = state["support"]
        active = state["order"][state["active"]]
        for arm in model.ARMS:
            output, info, point_groups = results[arm]
            row = dict(case=case, arm=arm, method=arm, primary=case in PRIMARY, stage="public_development_diagnostic",
                       seed=int(meta["seed"]), gap_mm=float(meta["gap_mm"]), bias_rms_mm=float(meta["bias_rms_mm"]),
                       n_points=len(world), sigma_mm=1., group_count=info["group_count"],
                       fit_parameter_count=info["fit_parameter_count"], fit_rank=info["fit_rank"],
                       fit_condition_number=info["fit_condition_number"], supported_fraction=info["supported_fraction"],
                       total_fit_weight=info["total_fit_weight"], active_point_count=info["active_point_count"],
                       frozen_common_sha256=info["frozen_common_sha256"], fit_seconds=info["fit_seconds"])
            row.update(synthetic_geometry(output, {}, ev))
            row.update(structure_metrics(output, ev))
            row["input_edit_rms_mm"] = float(1000*np.sqrt(np.mean(np.sum((output-world)**2, axis=1))))
            target = dest/"outputs"/(case+"__"+arm+".npz")
            with target.open("xb") as stream:
                np.savez_compressed(stream, xyz_world=output, scan_id=scans, source_point_index=point_ids,
                                    support_mask=support, active_original_indices=active, group_assignment=point_groups)
            row.update(output=str(target), output_sha256=digest(target), ok=True)
            # Composition is evaluator-only metadata, not fed back to any arm.
            composition = []
            for group in range(info["group_count"]):
                take = point_groups == group
                composition.append(dict(group=group, true_layer_counts={str(int(k)): int(np.sum(labels[take] == k)) for k in np.unique(labels)},
                                        supported_output_count=int(np.sum(take & support))))
            save_json(target.with_suffix(".json"), dict(info=info, row=row, evaluation_only_group_composition=composition))
            rows.append(row)
        for arm, old_method, old_run in (("original_independent", "pool_compatible", "exploration-v4-dev-e0c_n4hk"),
                                         ("original_shared", "shared_group_slope", "exploration-v5-development-vdbkdouk")):
            old_path = RUNS/old_run/"outputs"/(case+"__"+old_method+".npz")
            with np.load(old_path, allow_pickle=False) as data:
                old_output = data["xyz_world"]
            difference = float(np.max(abs(results[arm][0]-old_output)))
            if difference > REPLAY_TOL_M:
                raise AssertionError(f"replay exceeded protocol tolerance: {case} {arm} {difference}")
            replays.append(dict(case=case, arm=arm, reference=str(old_path), reference_sha256=digest(old_path),
                                max_abs_world_coordinate_difference_m=difference,
                                max_abs_world_coordinate_difference_mm=1000*difference,
                                bitwise_equal=bool(np.array_equal(results[arm][0], old_output))))
        print(case, "4/4", flush=True)
    comparisons = contrasts(rows)
    write_csv(dest/"RESULTS.csv", rows)
    write_csv(dest/"CONTRASTS.csv", comparisons)
    save_json(dest/"AGGREGATES.json", aggregates(rows, comparisons))
    save_json(dest/"INPUT_MANIFEST.json", input_manifest)
    save_json(dest/"REPRODUCTION.json", replays)
    after = {path: digest(path) for path in before}
    save_json(dest/"PROTECTED_AFTER.json", after)
    if before != after:
        raise AssertionError("protected source or input changed")
    summary = dict(run_dir=str(dest), host=socket.gethostname(), inputs=24, outputs=len(rows), primary_inputs=3,
                   all_outputs_saved=len(rows)==96, new_inputs_generated=False, new_confirmation=False,
                   protected_files=len(before), protected_unchanged=True, elapsed_seconds=time.perf_counter()-started,
                   freeze_seconds=sum(json.loads((dest/"outputs"/(case+"__original_independent.json")).read_text())["info"]["freeze_seconds"] for case in CASES),
                   conditional_fit_seconds=sum(row["fit_seconds"] for row in rows),
                   peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   cost_scope="CPU run plus one unit-test subprocess; parent peak RSS excludes test child; one shared upstream freeze per input, not per-arm production cost")
    save_json(dest/"SUMMARY.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
