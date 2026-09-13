"""Pilot runner. Protocol must exist before this writes results."""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from adapt import adapter_payload, build_adapter, from_operator_mm, to_operator_mm
from evaluate import _method_k, point_errors, synthetic_geometry
from hashutil import dump_json, sha256_file
from paths import OLD_CHECKPOINT, PATCHES, PILOT, SYNTHETICS, require_liekkas
from perturb import make_perturbation
from schema import pose_table, read_evaluation, read_patch
from transforms import as_matrix44

METHODS = ("identity", "xyz_mixture", "fast", "open3d_icp_then_xyz")
REAL_SIGMAS = (2.0, 5.0)
PERTURB_SEEDS = (912401, 912409, 912419)
TRANSLATION_RMS_M = 0.005
ROTATION_RMS_RAD = np.deg2rad(0.05)


def import_operators():
    sys.path.insert(0, str(OLD_CHECKPOINT))
    import operators  # noqa: E402

    return operators


def peak_rss_kib():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def list_json_cases(root: Path):
    return sorted(p for p in Path(root).rglob("*.json") if p.name != "SUMMARY.json" and "evaluation" not in p.parts and p.name != "EXTRACT_REPORT.json")


def call_method(operators, method, xyz_mm, frame, sigma_mm):
    t0 = time.perf_counter()
    try:
        output, info = operators.estimate(method, xyz_mm, frame, sigma_mm)
        return {
            "ok": True,
            "output_mm": np.asarray(output, dtype=np.float64),
            "info": info,
            "seconds": time.perf_counter() - t0,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "output_mm": None,
            "info": {"method": method},
            "seconds": time.perf_counter() - t0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def run_one(operators, points, meta, sigma_mm, adapter=None, eval_data=None, tag=""):
    if PILOT.exists():
        raise FileExistsError("legacy fixed output directory exists; use repair_pilot_v2.py")
    if adapter is None:
        adapter = build_adapter(points["xyz_world"])
    xyz_mm = to_operator_mm(points["xyz_world"], adapter)
    frame = np.asarray(points["scan_id"], dtype=np.int64)
    rows = []
    for method in METHODS:
        result = call_method(operators, method, xyz_mm, frame, sigma_mm)
        row = {
            "case": tag or meta.get("patch_id") or meta.get("family"),
            "method": method,
            "ok": result["ok"],
            "seconds": result["seconds"],
            "sigma_mm": sigma_mm,
            "n_points": int(len(xyz_mm)),
            "n_stations": int(len(np.unique(frame))),
            "error": result["error"],
            "independent_real_quality": "本轮尚未测得",
        }
        if result["ok"]:
            world = from_operator_mm(result["output_mm"], adapter)
            row["output_finite"] = bool(np.isfinite(world).all())
            row["output_n"] = int(len(world))
            if eval_data is not None and meta.get("dataset") == "synthetic_exact":
                row.update(synthetic_geometry(result["output_mm"], result["info"], eval_data))
            row["k_info"] = _method_k(result["info"] or {})
            out_dir = PILOT / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"{row['case']}__sig{sigma_mm}__{method}".replace("/", "_")
            np.savez_compressed(out_dir / f"{name}.npz", xyz_world=world, xyz_mm=result["output_mm"])
            dump_json(out_dir / f"{name}.json", {
                "info": {k: v for k, v in result["info"].items() if k != "backend_info"},
                "adapter": adapter_payload(adapter),
                "seconds": result["seconds"],
            })
            row["output_world"] = world
        rows.append(row)
    return rows, adapter


def run_synthetics(operators):
    rows = []
    for path in list_json_cases(SYNTHETICS):
        points, meta = read_patch(path)
        ev = read_evaluation(path)
        case_rows, _ = run_one(operators, points, meta, sigma_mm=1.0, eval_data=ev, tag=path.stem)
        for row in case_rows:
            row["split"] = "synthetic_exact"
            row["family"] = meta.get("family")
            row["seed"] = meta.get("seed")
            row["gap_mm"] = meta.get("gap_mm")
            row["bias_rms_mm"] = meta.get("bias_rms_mm")
            row["noise_protocol"] = "supplied-noise sigma=1 mm"
            row.pop("output_world", None)
        rows.extend(case_rows)
    return rows


def run_real(operators):
    rows = []
    patch_files = [p for p in list_json_cases(PATCHES) if p.suffix == ".json"]
    for path in patch_files:
        points, meta = read_patch(path)
        adapter = build_adapter(points["xyz_world"])
        poses = pose_table(meta)
        ref_sid = min(int(s) for s in np.unique(points["scan_id"]))
        unperturbed = {}
        for sigma in REAL_SIGMAS:
            case_rows, adapter = run_one(
                operators, points, meta, sigma_mm=sigma, adapter=adapter, tag=f"{path.stem}_unpert"
            )
            for row in case_rows:
                row["split"] = "real_unperturbed"
                row["patch_id"] = path.stem
                row["scene"] = meta.get("scene")
                row["noise_protocol"] = f"pre-fixed wiring/sensitivity sigma={sigma} mm; not sensor truth"
                if row.get("ok"):
                    unperturbed[(row["method"], sigma)] = row.pop("output_world")
                else:
                    row.pop("output_world", None)
            rows.extend(case_rows)
        variants = [("trans", TRANSLATION_RMS_M, 0.0), ("trans_rot", TRANSLATION_RMS_M, ROTATION_RMS_RAD)]
        for kind, t_rms, r_rms in variants:
            for seed in PERTURB_SEEDS:
                pert = make_perturbation(
                    points["xyz_world"],
                    points["scan_id"],
                    poses,
                    seed,
                    t_rms,
                    r_rms,
                    ref_sid,
                )
                pert_points = dict(points)
                pert_points["xyz_world"] = pert["xyz_world_perturbed"]
                pert_meta = dict(meta)
                pert_meta["T_world_from_scan_input"] = pert["current_T_world_from_scan"]
                # Inverse extra transforms stay in evaluation only.
                ev_dir = PILOT / "evaluation"
                ev_dir.mkdir(parents=True, exist_ok=True)
                dump_json(
                    ev_dir / f"{path.stem}_{kind}_s{seed}.json",
                    {k: v for k, v in pert.items() if k != "xyz_world_perturbed"},
                )
                for sigma in REAL_SIGMAS:
                    case_rows, _ = run_one(
                        operators,
                        pert_points,
                        pert_meta,
                        sigma_mm=sigma,
                        adapter=adapter,
                        tag=f"{path.stem}_{kind}_s{seed}",
                    )
                    for row in case_rows:
                        row["split"] = "real_perturbed"
                        row["patch_id"] = path.stem
                        row["scene"] = meta.get("scene")
                        row["perturbation"] = kind
                        row["perturb_seed"] = seed
                        row["actual_point_delta_rms_mm"] = pert["actual_point_delta_rms_mm"]
                        row["noise_protocol"] = f"pre-fixed wiring/sensitivity sigma={sigma} mm; not sensor truth"
                        if row.get("ok") and (row["method"], sigma) in unperturbed:
                            rec = point_errors(row["output_world"], points["xyz_world"])
                            stab = point_errors(row["output_world"], unperturbed[(row["method"], sigma)])
                            row["recovery_vs_unperturbed_input_rms_mm"] = rec["point_rms_mm"]
                            row["stability_vs_own_unperturbed_output_rms_mm"] = stab["point_rms_mm"]
                            row["recovery_note"] = (
                                "reference is the unperturbed measured cloud, which still has measurement error"
                            )
                            row["stability_note"] = "stability vs own unperturbed output; not real accuracy"
                        row.pop("output_world", None)
                    rows.extend(case_rows)
    return rows


def write_csv(path: Path, rows: list[dict]):
    skip = {"output_world", "traceback"}
    keys = []
    for row in rows:
        for key in row:
            if key not in skip and key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def main():
    # V1 used fixed filenames and reference-derived adapters. Keep for audit.
    raise SystemExit("Use repair_pilot_v2.py; legacy runner disabled to preserve V1 results")
    require_liekkas()
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-synth", action="store_true")
    args = parser.parse_args()
    protocol = Path(__file__).resolve().parent / "PILOT_PROTOCOL.md"
    if not protocol.exists():
        raise SystemExit("PILOT_PROTOCOL.md must be written before the pilot runs")
    PILOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    operators = import_operators()
    hashes_before = {
        "operators.py": sha256_file(OLD_CHECKPOINT / "operators.py"),
        "experiment.py": operators.source_hashes(),
    }
    t0 = time.perf_counter()
    warmup = operators.warmup(METHODS)
    rows = []
    if not args.skip_synth:
        rows.extend(run_synthetics(operators))
    if not args.skip_real:
        rows.extend(run_real(operators))
    elapsed = time.perf_counter() - t0
    hashes_after = {
        "operators.py": sha256_file(OLD_CHECKPOINT / "operators.py"),
        "experiment.py": operators.source_hashes(),
    }
    write_csv(Path(__file__).resolve().parent / "PILOT_RESULTS.csv", rows)
    write_csv(PILOT / "PILOT_RESULTS.csv", rows)
    summary = {
        "host": "liekkas",
        "started_like": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "n_fail": sum(1 for r in rows if not r.get("ok")),
        "elapsed_s_including_warmup_io": elapsed,
        "peak_rss_kib": peak_rss_kib(),
        "warmup": warmup,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "old_checkpoint_unchanged": hashes_before == hashes_after,
        "methods": METHODS,
    }
    dump_json(PILOT / "SUMMARY.json", summary)
    print(json.dumps({k: summary[k] for k in ("n_rows", "n_ok", "n_fail", "elapsed_s_including_warmup_io", "peak_rss_kib")}, indent=2))


if __name__ == "__main__":
    main()
