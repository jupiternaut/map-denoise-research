"""Exact synthetic controls. Lengths here are millimetres, then stored as metres."""
from __future__ import annotations

from pathlib import Path
import tempfile
import json

import numpy as np

from hashutil import dump_json
from paths import SYNTHETICS, require_liekkas
from schema import empty_points, fill_rays, write_patch
from transforms import as_matrix44

SEEDS = (912101, 912113, 912127)
HOLD_OUT_SEEDS = (912201, 912211, 912223)
N_STATIONS = 8
POINTS_PER_STATION = 96
SIGMA_MM = 1.0
GAPS_MM = (2.0, 4.0, 8.0)
BIAS_RMS_MM = (0.0, 4.0)
CAMERA_Z_MM = 220.0


def _pose_from_origin(origin_mm) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(origin_mm, dtype=np.float64) / 1000.0
    return matrix


def _station_origin(frame: int) -> np.ndarray:
    return np.array([-35.0 + 70.0 * frame / max(N_STATIONS - 1, 1), -80.0, CAMERA_Z_MM])


def _finite_hits(origin, targets, plates):
    """Nearest positive hit on finite opaque axis-aligned plates z=const."""
    hits = []
    for z, x0, x1, y0, y1, label in plates:
        direction = targets - origin
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - origin[2]) / direction[:, 2]
        hit = origin + t[:, None] * direction
        ok = (
            np.isfinite(t)
            & (t > 1e-9)
            & (hit[:, 0] >= x0)
            & (hit[:, 0] <= x1)
            & (hit[:, 1] >= y0)
            & (hit[:, 1] <= y1)
        )
        hits.append((t, hit, ok, label))
    chosen_t = np.full(len(targets), np.inf)
    chosen_xyz = np.full((len(targets), 3), np.nan)
    chosen_label = np.full(len(targets), -1, dtype=np.int64)
    for t, hit, ok, label in hits:
        better = ok & (t < chosen_t)
        chosen_t[better] = t[better]
        chosen_xyz[better] = hit[better]
        chosen_label[better] = label
    return chosen_xyz, chosen_label, np.isfinite(chosen_t)


def _sample_hits(rng, origin, plates, count):
    # Oversample target directions, keep first-return hits.
    want = max(count * 40, 4000)
    targets = np.column_stack(
        (
            rng.uniform(-70.0, 70.0, want),
            rng.uniform(-45.0, 45.0, want),
            np.zeros(want),
        )
    )
    xyz, label, ok = _finite_hits(origin, targets, plates)
    usable = np.flatnonzero(ok)
    if len(usable) < count:
        return xyz[usable], label[usable]
    pick = rng.choice(usable, size=count, replace=False)
    return xyz[pick], label[pick]


def frame_bias(seed, amplitude, n=N_STATIONS) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 303]))
    raw = rng.normal(size=n)
    raw -= raw.mean()
    if amplitude == 0:
        return np.zeros(n)
    return raw * (amplitude / np.sqrt(np.mean(raw ** 2)))


def _pack(xyz_mm, labels, frames, origins_mm, bias_mm, eps_mm, meta_extra, evaluation_extra):
    n = len(xyz_mm)
    points = empty_points(n)
    points["xyz_world"] = np.asarray(xyz_mm, dtype=np.float64) / 1000.0
    # Poses translate scanner-local coordinates into the common world frame.
    points["xyz_local"] = points["xyz_world"] - np.asarray(origins_mm) / 1000.0
    points["scan_id"] = np.asarray(frames, dtype=np.int64)
    points["source_point_index"] = np.arange(n, dtype=np.int64)
    points["scanner_origin_world"] = np.asarray(origins_mm, dtype=np.float64) / 1000.0
    points["xyz_local_valid"] = np.ones(n, dtype=np.bool_)
    fill_rays(points)
    poses = {}
    for sid in np.unique(frames):
        poses[str(int(sid))] = _pose_from_origin(_station_origin(int(sid))).tolist()
    meta = {
        "dataset": "synthetic_exact",
        "units": "xyz_world metres; generation millimetres",
        "T_world_from_scan_input": poses,
        "pose_role": "scanner-local to world; scalar normal noise/bias belongs to measured coordinates",
        "sigma_mm_generation": SIGMA_MM,
        "supplied_noise_experiment": True,
        **meta_extra,
    }
    evaluation = {
        "gt_layer": np.asarray(labels, dtype=np.int64),
        "gt_clean_xyz_world": (np.asarray(xyz_mm) - np.c_[np.zeros((n, 2)), eps_mm + bias_mm[frames]]) / 1000.0,
        "gt_bias_mm": np.asarray(bias_mm, dtype=np.float64),
        "gt_eps_mm": np.asarray(eps_mm, dtype=np.float64),
        "identifiable": bool(meta_extra.get("identifiable", True)),
        **evaluation_extra,
    }
    return points, meta, evaluation


def make_ghost(seed: int, bias_rms_mm: float):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    plates = [(0.0, -60.0, 60.0, -50.0, 50.0, 0)]
    xyz, labels, frames, origins = [], [], [], []
    for frame in range(N_STATIONS):
        origin = _station_origin(frame)
        hit, lab = _sample_hits(rng, origin, plates, POINTS_PER_STATION)
        xyz.append(hit)
        labels.append(lab)
        frames.append(np.full(len(hit), frame, dtype=np.int64))
        origins.append(np.repeat(origin[None], len(hit), axis=0))
    xyz = np.concatenate(xyz)
    labels = np.concatenate(labels)
    frames = np.concatenate(frames)
    origins = np.concatenate(origins)
    xyz[:, 2] = 0.0
    eps = np.random.default_rng(np.random.SeedSequence([seed, 202])).normal(0.0, SIGMA_MM, len(xyz))
    bias = frame_bias(seed, bias_rms_mm)
    xyz = xyz.copy()
    xyz[:, 2] += eps + bias[frames]
    points, meta, evaluation = _pack(
        xyz,
        labels,
        frames,
        origins,
        bias,
        eps,
        {
            "family": "ghost_monolayer",
            "gap_mm": 0.0,
            "bias_rms_mm": bias_rms_mm,
            "seed": seed,
            "identifiable": True,
            "description": "one finite plane plus per-frame common bias; ghosts are not a second surface",
        },
        {"n_true_layers": 1, "true_gap_mm": 0.0,
         "surface_rectangles_mm": [list(p[:5]) for p in plates]},
    )
    return points, meta, evaluation


def make_dual(seed: int, gap_mm: float, bias_rms_mm: float):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    plates = [
        (0.0, -60.0, 15.0, -50.0, 50.0, 0),
        (gap_mm, -15.0, 60.0, -50.0, 50.0, 1),
    ]
    xyz, labels, frames, origins = [], [], [], []
    for frame in range(N_STATIONS):
        origin = _station_origin(frame)
        hit, lab = _sample_hits(rng, origin, plates, POINTS_PER_STATION)
        xyz.append(hit)
        labels.append(lab)
        frames.append(np.full(len(hit), frame, dtype=np.int64))
        origins.append(np.repeat(origin[None], len(hit), axis=0))
    xyz = np.concatenate(xyz)
    labels = np.concatenate(labels)
    frames = np.concatenate(frames)
    origins = np.concatenate(origins)
    xyz[:, 2] = gap_mm * labels
    eps = np.random.default_rng(np.random.SeedSequence([seed, 202])).normal(0.0, SIGMA_MM, len(xyz))
    bias = frame_bias(seed, bias_rms_mm)
    xyz = xyz.copy()
    xyz[:, 2] += eps + bias[frames]
    support = {
        int(layer): {
            "n": int(np.sum(labels == layer)),
            "stations": sorted(int(s) for s in np.unique(frames[labels == layer])),
        }
        for layer in (0, 1)
    }
    points, meta, evaluation = _pack(
        xyz,
        labels,
        frames,
        origins,
        bias,
        eps,
        {
            "family": "true_bilayer",
            "gap_mm": gap_mm,
            "bias_rms_mm": bias_rms_mm,
            "seed": seed,
            "identifiable": True,
            "layer_support": support,
            "description": "two finite opaque plates; front layer occludes; plus per-frame bias",
        },
        {"n_true_layers": 2, "true_gap_mm": gap_mm, "layer_support": support,
         "surface_rectangles_mm": [list(p[:5]) for p in plates]},
    )
    return points, meta, evaluation


def make_ambiguous(seed: int, gap_mm: float = 8.0):
    """Two physical stories produce the same points+frames.

    A: two real surfaces at 0 and gap, no extra frame bias, even/odd frames
       each see only one surface.
    B: one surface at gap/2, even/odd frames carry opposite common bias ±gap/2.
    The stored observations are identical; both interpretations are recorded.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    xyz, labels_a, frames, origins = [], [], [], []
    for frame in range(N_STATIONS):
        origin = _station_origin(frame)
        if frame % 2 == 0:
            plates = [(0.0, -60.0, 60.0, -50.0, 50.0, 0)]
        else:
            plates = [(gap_mm, -60.0, 60.0, -50.0, 50.0, 1)]
        hit, lab = _sample_hits(rng, origin, plates, POINTS_PER_STATION)
        hit = hit.copy()
        hit[:, 2] = 0.0 if frame % 2 == 0 else gap_mm
        xyz.append(hit)
        labels_a.append(lab)
        frames.append(np.full(len(hit), frame, dtype=np.int64))
        origins.append(np.repeat(origin[None], len(hit), axis=0))
    xyz = np.concatenate(xyz)
    labels_a = np.concatenate(labels_a)
    frames = np.concatenate(frames)
    origins = np.concatenate(origins)
    eps = np.random.default_rng(np.random.SeedSequence([seed, 202])).normal(0.0, SIGMA_MM, len(xyz))
    xyz = xyz.copy()
    xyz[:, 2] += eps
    bias_a = np.zeros(N_STATIONS)
    bias_b = np.array([(-gap_mm / 2.0) if f % 2 == 0 else (gap_mm / 2.0) for f in range(N_STATIONS)])
    labels_b = np.zeros(len(xyz), dtype=np.int64)
    points, meta, evaluation = _pack(
        xyz,
        labels_a,
        frames,
        origins,
        bias_a,
        eps,
        {
            "family": "ambiguous_two_explanations",
            "visibility_scope": "scalar XYZ/frame ambiguity only; per-frame scenes are not a joint static visibility model",
            "gap_mm": gap_mm,
            "bias_rms_mm": 0.0,
            "seed": seed,
            "identifiable": False,
            "description": (
                "even stations only see z=0, odd only see z=gap; "
                "one-plane-plus-opposite-bias and two-planes-no-bias are observationally equivalent"
            ),
        },
        {
            "n_true_layers": None,
            "true_gap_mm": None,
            "explanation_a": {
                "name": "two_planes_no_bias",
                "layers_mm": [0.0, gap_mm],
                "bias_mm": bias_a.tolist(),
            },
            "explanation_b": {
                "name": "one_plane_opposite_frame_bias",
                "layers_mm": [gap_mm / 2.0],
                "bias_mm": bias_b.tolist(),
            },
            "labels_explanation_a": labels_a,
            "labels_explanation_b": labels_b,
        },
    )
    return points, meta, evaluation


def generate_all(destination: Path | None = None) -> dict:
    require_liekkas()
    if destination is None:
        SYNTHETICS.parent.mkdir(parents=True, exist_ok=True)
        destination = tempfile.mkdtemp(prefix="synthetics-v2-", dir=SYNTHETICS.parent)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in SEEDS:
        for bias in BIAS_RMS_MM:
            pid = f"ghost_s{seed}_b{int(bias)}"
            points, meta, ev = make_ghost(seed, bias)
            written.append(write_patch(destination / "identifiable", pid, points, meta, ev))
        for gap in GAPS_MM:
            for bias in BIAS_RMS_MM:
                pid = f"dual_g{int(gap)}_s{seed}_b{int(bias)}"
                points, meta, ev = make_dual(seed, gap, bias)
                written.append(write_patch(destination / "identifiable", pid, points, meta, ev))
        pid = f"ambig_g8_s{seed}"
        points, meta, ev = make_ambiguous(seed, 8.0)
        written.append(write_patch(destination / "ambiguous", pid, points, meta, ev))
    summary = {
        "destination": str(destination),
        "seeds_used": list(SEEDS),
        "hold_out_seeds_not_used": list(HOLD_OUT_SEEDS),
        "n_cases": len(written),
        "target_stations": N_STATIONS,
        "target_points": N_STATIONS * POINTS_PER_STATION,
        "sigma_mm": SIGMA_MM,
        "cases": written,
        "note": "hold-out seeds are reserved; this round does not retune on them",
    }
    dump_json(destination / "SUMMARY.json", summary)
    return summary


if __name__ == "__main__":
    out = generate_all()
    print(json.dumps({"n_cases": out["n_cases"], "dest": out["destination"]}, indent=2))
