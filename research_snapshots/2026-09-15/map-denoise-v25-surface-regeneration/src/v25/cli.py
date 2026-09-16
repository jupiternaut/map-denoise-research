"""V25 CLI. Record actual commands in COMMANDS.md after they run."""

from __future__ import annotations

import argparse
import csv
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .cameras import Scene, load_scene, project_matrix, require_host
from .evidence import EvidenceBundle, build_evidence, bundle_public_dict, save_bundle
from .io_util import append_jsonl, read_json, sha256_file, voxel_downsample, write_json, write_ply
from .observations import define_rois, load_roi_specs, roi_to_dict
from .paths import EVAL_ONLY, WORKSPACE
from .regenerate import ARMS, run_arm

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

RUN_ID = "2026-09-14T165314Z"
ARMS_TO_RUN = ("identity", "fusion_wta", "restricted_single", "restricted_point_move", "v25_atlas")
EXTRA_ARMS = ("v25_wta_atlas", "v25_depth_cc", "v25_gated_move")


def run_dir() -> Path:
    path = WORKSPACE / "runs" / RUN_ID
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config() -> dict:
    return read_json(WORKSPACE / "configs" / "default.json")


def cmd_prepare(_args) -> int:
    require_host()
    dest = run_dir()
    config = load_config()
    spec = load_roi_specs(WORKSPACE / "configs" / "roi_image_boxes.json")
    all_rois = []
    for scene_id in (24, 37):
        t0 = time.perf_counter()
        scene = load_scene(scene_id, load_mesh=True)
        write_json(dest / f"scan{scene_id}_provenance.json", scene.provenance)
        rois = define_rois(scene, spec)
        payload = [roi_to_dict(roi) for roi in rois]
        write_json(dest / f"scan{scene_id}_rois.json", payload)
        _write_roi_previews(scene, rois, dest / "previews")
        all_rois.extend(payload)
        append_jsonl(
            dest / "ledger.jsonl",
            {
                "id": f"prepare_scan{scene_id}",
                "seconds": time.perf_counter() - t0,
                "reprojection_median_px": scene.provenance["sparse_reprojection_median_px"],
                "reprojection_p95_px": scene.provenance["sparse_reprojection_p95_px"],
                "n_rois": len(rois),
            },
        )
        print(f"prepare scan{scene_id}: median={scene.provenance['sparse_reprojection_median_px']:.3f} p95={scene.provenance['sparse_reprojection_p95_px']:.3f}", flush=True)
    write_json(dest / "rois.json", all_rois)
    write_json(dest / "config_used.json", config)
    write_json(dest / "prepare_status.json", {"status": "SUCCEEDED", "n_rois": len(all_rois), "host": socket.gethostname()})
    return 0


def _write_roi_previews(scene: Scene, rois, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    ref = scene.view_by_name(rois[0].ref_view)
    image = Image.fromarray(scene.load_image(ref.name))
    draw = ImageDraw.Draw(image)
    for roi in rois:
        color = (255, 220, 0) if roi.kind == "structure" else (0, 220, 120)
        draw.rectangle(roi.box_xyxy, outline=color, width=3)
        draw.text((roi.box_xyxy[0] + 4, roi.box_xyxy[1] + 4), roi.roi_id, fill=color)
        uv, z = project_matrix(np.asarray([roi.centroid]), ref.P_phys)
        if np.isfinite(uv).all() and z[0] > 0:
            x, y = map(int, uv[0])
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=color)
    image.save(dest / f"scan{scene.scene_id}_roi_boxes.png")


def _load_rois() -> list[dict]:
    path = run_dir() / "rois.json"
    if not path.is_file():
        raise FileNotFoundError("run prepare first")
    return read_json(path)


def _roi_row(roi_id: str) -> dict:
    for row in _load_rois():
        if row["roi_id"] == roi_id:
            return row
    raise KeyError(roi_id)


def _roi_object(row: dict):
    from .observations import Roi

    return Roi(
        roi_id=row["roi_id"],
        scene_id=row["scene_id"],
        kind=row["kind"],
        reason=row["reason"],
        aabb_min=np.asarray(row["aabb_min_mm"], dtype=np.float64),
        aabb_max=np.asarray(row["aabb_max_mm"], dtype=np.float64),
        ref_view=row["ref_view"],
        box_xyxy=row["box_xyxy"],
        n_mesh_points=row["n_mesh_points"],
        centroid=np.asarray(row["centroid_mm"], dtype=np.float64),
    )


def cmd_evidence(args) -> int:
    row = _roi_row(args.roi)
    scene = load_scene(row["scene_id"], load_mesh=True)
    roi = _roi_object(row)
    config = load_config()
    t0 = time.perf_counter()
    bundle = build_evidence(scene, roi, config)
    dest = run_dir() / row["roi_id"]
    dest.mkdir(parents=True, exist_ok=True)
    save_bundle(dest / "evidence.npz", bundle)
    write_json(dest / "evidence.json", bundle_public_dict(bundle))
    write_json(
        dest / "evidence_status.json",
        {"status": "SUCCEEDED", "seconds": time.perf_counter() - t0, "hashes": bundle.hashes, "views": bundle.view_names},
    )
    print(f"evidence {args.roi}: views={bundle.view_names} seconds={time.perf_counter()-t0:.1f}", flush=True)
    return 0


def _load_bundle(roi_id: str) -> EvidenceBundle:
    path = run_dir() / roi_id / "evidence.npz"
    data = np.load(path, allow_pickle=True)
    config = load_config()
    config["scale_factor"] = float(config.get("scale_factor", 0.0))
    # scale_factor is scene-specific; recover from depths + default file, then provenance.
    scene_id = int(data["scene_id"])
    provenance = read_json(run_dir() / f"scan{scene_id}_provenance.json")
    config["scale_factor"] = float(provenance["scale_factor"])
    return EvidenceBundle(
        scene_id=scene_id,
        roi_id=str(data["roi_id"]),
        view_names=[str(x) for x in data["view_names"].tolist()],
        ref_name=str(data["ref_name"]),
        depths_norm=data["depths_norm"].astype(np.float64),
        zncc=data["zncc"],
        n_valid_src=data["n_valid_src"],
        crop_meta=[],
        aabb_min=data["aabb_min"].astype(np.float64),
        aabb_max=data["aabb_max"].astype(np.float64),
        config=config,
        hashes={},
    )


def cmd_regenerate(args) -> int:
    row = _roi_row(args.roi)
    scene = load_scene(row["scene_id"], load_mesh=True)
    bundle = _load_bundle(args.roi)
    dest = run_dir() / args.roi
    if args.arm:
        names = [args.arm]
    elif getattr(args, "extra", False):
        names = list(EXTRA_ARMS)
    else:
        names = list(ARMS_TO_RUN)
    for name in names:
        t0 = time.perf_counter()
        points, meta = run_arm(name, scene, bundle)
        ply = dest / f"{name}.ply"
        write_ply(ply, points)
        meta.update(
            {
                "status": "SUCCEEDED" if len(points) else "EMPTY",
                "seconds": time.perf_counter() - t0,
                "ply": str(ply),
                "sha256": sha256_file(ply),
                "n_points": int(len(points)),
            }
        )
        write_json(dest / f"{name}.json", meta)
        print(f"regenerate {args.roi} {name}: n={len(points)} seconds={meta['seconds']:.2f}", flush=True)
    return 0


def cmd_run_roi(args) -> int:
    cmd_evidence(args)
    return cmd_regenerate(args)


_LASER_CACHE = {}


def _cached_laser_and_mask(scene_id: int):
    from evaluation.mask import load_obs_mask
    from evaluation.metrics import read_points

    if scene_id not in _LASER_CACHE:
        spec = EVAL_ONLY[scene_id]
        _LASER_CACHE[scene_id] = (read_points(spec["laser"]), load_obs_mask(spec["mask"]))
    return _LASER_CACHE[scene_id]


def cmd_evaluate(args) -> int:
    # Imported only here so constructors stay GT-free even if this file is imported.
    from evaluation.metrics import read_points, score_arrays

    row = _roi_row(args.roi)
    scene_id = row["scene_id"]
    dest = run_dir() / args.roi
    voxel = float(load_config()["voxel_mm"])
    laser, obs = _cached_laser_and_mask(scene_id)
    rows = []
    arms = list(ARMS_TO_RUN) + [a for a in EXTRA_ARMS if (dest / f"{a}.ply").is_file()]
    for name in dict.fromkeys(arms):
        ply = dest / f"{name}.ply"
        if not ply.is_file():
            rows.append({"arm": name, "status": "NOT_RUN"})
            continue
        t0 = time.perf_counter()
        metrics = score_arrays(
            read_points(ply),
            laser,
            np.asarray(row["aabb_min_mm"], dtype=np.float64),
            np.asarray(row["aabb_max_mm"], dtype=np.float64),
            obs,
            voxel,
        )
        metrics.update({"arm": name, "roi_id": args.roi, "seconds": time.perf_counter() - t0, "ply_sha256": sha256_file(ply)})
        rows.append(metrics)
        print(f"eval {args.roi} {name}: E_sym={metrics.get('E_sym_mm')} n={metrics.get('n_output_eval')}", flush=True)
    write_json(dest / "metrics.json", rows)
    return 0


def cmd_seal(_args) -> int:
    dest = run_dir()
    sealed = []
    for row in _load_rois():
        folder = dest / row["roi_id"]
        item = {"roi_id": row["roi_id"], "kind": row["kind"], "arms": {}}
        for name in list(ARMS_TO_RUN) + list(EXTRA_ARMS):
            ply = folder / f"{name}.ply"
            if ply.is_file():
                item["arms"][name] = {"ply": str(ply), "sha256": sha256_file(ply), "n_bytes": ply.stat().st_size}
        ev = folder / "evidence.npz"
        if ev.is_file():
            item["evidence_sha256"] = sha256_file(ev)
        sealed.append(item)
    write_json(dest / "SEALED.json", {"sealed_at_utc": datetime.now(timezone.utc).isoformat(), "items": sealed})
    return 0


def cmd_report(_args) -> int:
    dest = run_dir()
    figures = WORKSPACE / "figures"
    figures.mkdir(exist_ok=True)
    rows = []
    for roi in _load_rois():
        metrics_path = dest / roi["roi_id"] / "metrics.json"
        if not metrics_path.is_file():
            continue
        for item in read_json(metrics_path):
            rows.append({**item, "kind": roi["kind"], "scene_id": roi["scene_id"]})
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (dest / "RESULTS.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        write_json(dest / "RESULTS.json", rows)
        _bar_figure(rows, figures / "esym_by_roi.png")
    write_json(dest / "report_status.json", {"n_rows": len(rows), "csv": str(dest / "RESULTS.csv")})
    print(f"report rows={len(rows)}", flush=True)
    return 0


def _bar_figure(rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rois = list(dict.fromkeys(r["roi_id"] for r in rows))
    preferred = list(ARMS_TO_RUN) + list(EXTRA_ARMS)
    arms = [a for a in preferred if any(r.get("arm") == a for r in rows)]
    x = np.arange(len(rois))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, arm in enumerate(arms):
        values = []
        for roi in rois:
            hit = [r for r in rows if r["roi_id"] == roi and r.get("arm") == arm]
            values.append(hit[0]["E_sym_mm"] if hit and hit[0].get("E_sym_mm") is not None else np.nan)
        ax.bar(x + i * width, values, width, label=arm)
    ax.set_xticks(x + width * (len(arms) - 1) / 2)
    ax.set_xticklabels(rois, rotation=25, ha="right")
    ax.set_ylabel("E_sym mm")
    ax.set_title("Development ROI symmetric distance (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def cmd_core(args) -> int:
    """Rebuild evidence on a uniformly shrunken input-mesh core of each ROI."""
    from .observations import shrink_roi

    row = _roi_row(args.roi.replace("_core", ""))
    scene = load_scene(row["scene_id"], load_mesh=True)
    parent = _roi_object(row)
    roi = shrink_roi(scene, parent, keep_frac=0.4)
    dest = run_dir()
    cores_path = dest / "rois_core.json"
    cores = read_json(cores_path) if cores_path.is_file() else []
    cores = [c for c in cores if c["roi_id"] != roi.roi_id]
    cores.append(roi_to_dict(roi))
    write_json(cores_path, cores)
    # Temporarily expose the core ROI through the main list for later evaluate/seal.
    all_rois = _load_rois()
    if not any(r["roi_id"] == roi.roi_id for r in all_rois):
        all_rois.append(roi_to_dict(roi))
        write_json(dest / "rois.json", all_rois)
    config = load_config()
    t0 = time.perf_counter()
    bundle = build_evidence(scene, roi, config)
    folder = dest / roi.roi_id
    folder.mkdir(parents=True, exist_ok=True)
    save_bundle(folder / "evidence.npz", bundle)
    write_json(folder / "evidence.json", bundle_public_dict(bundle))
    print(f"core-evidence {roi.roi_id}: n_mesh={roi.n_mesh_points} seconds={time.perf_counter()-t0:.1f}", flush=True)
    ns = argparse.Namespace(roi=roi.roi_id, arm=None, extra=False)
    cmd_regenerate(ns)
    ns.extra = True
    ns.arm = None
    cmd_regenerate(ns)
    cmd_evaluate(argparse.Namespace(roi=roi.roi_id))
    return 0


def cmd_run_matrix(args) -> int:
    if not (run_dir() / "rois.json").is_file():
        cmd_prepare(args)
    rois = [row["roi_id"] for row in _load_rois()]
    if args.roi:
        rois = [args.roi]
    for roi_id in rois:
        ns = argparse.Namespace(roi=roi_id, arm=None)
        cmd_run_roi(ns)
        cmd_evaluate(ns)
    cmd_seal(args)
    cmd_report(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="v25")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    p = sub.add_parser("evidence")
    p.add_argument("--roi", required=True)
    p.set_defaults(func=cmd_evidence)
    p = sub.add_parser("regenerate")
    p.add_argument("--roi", required=True)
    p.add_argument("--arm")
    p.add_argument("--extra", action="store_true")
    p.set_defaults(func=cmd_regenerate)
    p = sub.add_parser("run-roi")
    p.add_argument("--roi", required=True)
    p.add_argument("--arm")
    p.set_defaults(func=cmd_run_roi)
    p = sub.add_parser("evaluate")
    p.add_argument("--roi", required=True)
    p.set_defaults(func=cmd_evaluate)
    sub.add_parser("seal").set_defaults(func=cmd_seal)
    sub.add_parser("report").set_defaults(func=cmd_report)
    p = sub.add_parser("core")
    p.add_argument("--roi", required=True)
    p.set_defaults(func=cmd_core)
    p = sub.add_parser("run-matrix")
    p.add_argument("--roi")
    p.set_defaults(func=cmd_run_matrix)
    return parser


def main(argv=None) -> int:
    require_host()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
