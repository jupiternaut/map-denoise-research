"""Independent 7x7 synthetic producer for the locked decision-interface study.

No inference module or historical producer is imported. Importing is read-only;
the main runner must pass its B2 gate before requesting confirmation generation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


MECHANISMS = ("flat_contrast", "flat_equal", "textured_boundary", "textured_single")
STAGE_SEEDS = {"calibration": (31001, 31002, 31003),
               "confirmation": (41001, 41002, 41003)}
ACTION_GRID = np.arange(450., 901., 15.)
IMAGE_SHAPE = (128, 128)
INTEGRATION_SAMPLES = 7


def cameras():
    """Reference then negative/positive source; R maps world to camera."""
    intrinsics = np.array([[160., 0., 64.], [0., 160., 64.], [0., 0., 1.]])
    result = [{"K": intrinsics.copy(), "R": np.eye(3), "C": np.zeros(3)}]
    for sign in (-1., 1.):
        yaw, roll = sign*.07, sign*.11
        cy, sy, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll)
        # Expanded Ry(yaw) @ Rz(roll), independent of the old helper.
        rotation = np.array([[cy*cr, -cy*sr, sy], [sr, cr, 0.],
                             [-sy*cr, sy*sr, cy]])
        result.append({"K": intrinsics.copy(), "R": rotation,
                       "C": np.array([sign*60., 0., 0.])})
    return result


def _texture(rng, amplitude_bounds):
    wavelength = rng.uniform(4., 12., 4)
    direction = rng.uniform(-np.pi, np.pi, 4)
    frequency = np.column_stack((np.cos(direction), np.sin(direction))) / wavelength[:, None]
    return {"amplitude": float(rng.uniform(*amplitude_bounds)),
            "frequencies": frequency.tolist(),
            "phases": rng.uniform(-np.pi, np.pi, 4).tolist()}


def make_object(stage, batch_index, mechanism, within_batch_index):
    """Deterministic producer-only specification for one registered object."""
    if stage not in STAGE_SEEDS:
        raise ValueError("stage must be calibration or confirmation")
    if mechanism not in MECHANISMS or batch_index not in range(3) or within_batch_index not in range(4):
        raise ValueError("invalid registered object address")
    seed = STAGE_SEEDS[stage][batch_index]
    rng = np.random.default_rng(np.random.SeedSequence([seed, MECHANISMS.index(mechanism), within_batch_index]))
    on_grid = within_batch_index < 2
    if on_grid:
        depth = float(rng.choice(ACTION_GRID[(ACTION_GRID >= 520.) & (ACTION_GRID <= 690.)]))
    else:
        depth = float(rng.uniform(520., 690.))
        while np.min(np.abs(ACTION_GRID-depth)) <= 1e-6:
            depth = float(rng.uniform(520., 690.))
    spec = {"stage": stage, "batch_seed": seed, "batch_index": int(batch_index),
            "within_batch_index": int(within_batch_index), "mechanism": mechanism,
            "on_action_grid": bool(on_grid), "true_depth": depth,
            "center": None, "half_width": None, "half_height": None, "angle": None,
            "background_depth": None, "foreground_level": None, "background_level": None,
            "foreground_texture": None, "background_texture": None, "equal_level": None}
    if mechanism == "flat_equal":
        color_bin = batch_index*4+within_batch_index
        spec["equal_level"] = float(20.+(color_bin+rng.uniform())*(235.-20.)/12.)
    elif mechanism == "textured_single":
        spec["foreground_level"] = float(rng.uniform(70., 190.))
        spec["foreground_texture"] = _texture(rng, (15., 30.))
    else:
        spec.update({"center": (64.+rng.uniform(-.5, .5, 2)).tolist(),
                     "half_width": float(rng.uniform(1.2, 3.2)), "half_height": 8., "angle": .17,
                     "background_depth": depth+float(rng.uniform(150., 300.))})
        back = float(rng.uniform(25., 65.))
        spec.update({"background_level": back, "foreground_level": back+float(rng.uniform(85., 140.))})
        if mechanism == "textured_boundary":
            spec["foreground_texture"] = _texture(rng, (8., 18.))
            spec["background_texture"] = _texture(rng, (5., 12.))
    return spec


def object_specs(stage):
    """Return the48 producer specifications in a fixed opaque-ID order; no I/O."""
    if stage not in STAGE_SEEDS:
        raise ValueError("stage must be calibration or confirmation")
    records = [make_object(stage, batch, mechanism, within)
               for batch in range(3) for mechanism in MECHANISMS for within in range(4)]
    rng = np.random.default_rng(np.random.SeedSequence([*STAGE_SEEDS[stage], 991]))
    return [records[int(index)] for index in rng.permutation(len(records))]


def plane_homography(depth, reference_camera, target_camera):
    """Map reference homogeneous pixels on a reference optical-Z plane."""
    depth = float(depth)
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("plane depth must be finite and positive")
    kr, rr, cr = (np.asarray(reference_camera[key], float) for key in ("K", "R", "C"))
    kt, rt, ct = (np.asarray(target_camera[key], float) for key in ("K", "R", "C"))
    translation = rt @ (cr-ct)
    plane_transform = rt @ rr.T + np.outer(translation, [0., 0., 1.])/depth
    return kt @ plane_transform @ np.linalg.inv(kr)


def _map_to_reference(inverse_homography, sensor_x, sensor_y):
    matrix = inverse_homography
    denominator = matrix[2, 0]*sensor_x+matrix[2, 1]*sensor_y+matrix[2, 2]
    x = (matrix[0, 0]*sensor_x+matrix[0, 1]*sensor_y+matrix[0, 2])/denominator
    y = (matrix[1, 0]*sensor_x+matrix[1, 1]*sensor_y+matrix[1, 2])/denominator
    return x, y


def _radiance(spec, layer, x, y):
    result = np.full(np.broadcast_shapes(np.shape(x), np.shape(y)), float(spec[f"{layer}_level"]))
    texture = spec.get(f"{layer}_texture")
    if texture is None:
        return result
    waves = np.zeros_like(result)
    for frequency, phase in zip(texture["frequencies"], texture["phases"]):
        waves += np.cos(2*np.pi*(frequency[0]*(x-64.)+frequency[1]*(y-64.))+phase)
    return result+float(texture["amplitude"])*waves/len(texture["phases"])


def _rectangle_membership(spec, x, y):
    c, s = np.cos(spec["angle"]), np.sin(spec["angle"])
    dx, dy = x-spec["center"][0], y-spec["center"][1]
    return ((np.abs(c*dx+s*dy) <= spec["half_width"]) &
            (np.abs(-s*dx+c*dy) <= spec["half_height"]))


def render_object(spec, camera_records=None, *, samples_per_axis=INTEGRATION_SAMPLES, row_chunk=16):
    """Independent midpoint integration. Dataset production always uses7×7."""
    mechanism = spec["mechanism"]
    if mechanism not in MECHANISMS:
        raise ValueError("unknown mechanism")
    records = cameras() if camera_records is None else camera_records
    samples_per_axis, row_chunk = int(samples_per_axis), int(row_chunk)
    if samples_per_axis < 1 or row_chunk < 1:
        raise ValueError("sample count and row chunk must be positive")
    height, width = IMAGE_SHAPE
    if mechanism == "flat_equal":
        # Exact absence of geometry information; even numerical integration
        # summation cannot create tiny per-view differences for this control.
        return np.full((len(records), height, width), float(spec["equal_level"]), dtype=np.float64)
    image = np.empty((len(records), height, width), dtype=np.float64)
    offsets = (np.arange(samples_per_axis)+.5)/samples_per_axis-.5
    fine_x = (np.arange(width)[:, None]+offsets[None]).reshape(-1)[None, :]
    for view, record in enumerate(records):
        front_inverse = np.linalg.inv(plane_homography(spec["true_depth"], records[0], record))
        back_inverse = None if mechanism == "textured_single" else np.linalg.inv(
            plane_homography(spec["background_depth"], records[0], record))
        for y0 in range(0, height, row_chunk):
            y1 = min(y0+row_chunk, height)
            fine_y = (np.arange(y0, y1)[:, None]+offsets[None]).reshape(-1)[:, None]
            front_x, front_y = _map_to_reference(front_inverse, fine_x, fine_y)
            radiance = _radiance(spec, "foreground", front_x, front_y)
            if mechanism != "textured_single":
                inside = _rectangle_membership(spec, front_x, front_y)
                back_x, back_y = _map_to_reference(back_inverse, fine_x, fine_y)
                background = _radiance(spec, "background", back_x, back_y)
                radiance = np.where(inside, radiance, background)
            sensor_samples = radiance.reshape(y1-y0, samples_per_axis, width, samples_per_axis)
            image[view, y0:y1] = sensor_samples.mean(axis=(1, 3))
    if not np.all(np.isfinite(image)) or np.min(image) < 0 or np.max(image) > 255:
        raise ValueError("fixture radiance is outside its registered finite gray range")
    return image


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json_exclusive(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False, default=_json_value)
        stream.write("\n")


def create_dataset(root, stage):
    """Explicitly create one locked stage; caller enforces the confirmation gate.

    Returns a summary dictionary. Existing observed/truth directories are never
    replaced, and completed inputs.json is written only after all images exist.
    """
    if stage not in STAGE_SEEDS:
        raise ValueError("stage must be calibration or confirmation")
    root = Path(root).resolve()
    observed_dir, truth_dir = root/"observed", root/"truth"
    if observed_dir.exists() or truth_dir.exists():
        raise FileExistsError("refusing to overwrite existing observed or truth stage")
    observed_dir.mkdir(parents=True, exist_ok=False)
    truth_dir.mkdir(parents=False, exist_ok=False)
    supplied_cameras = cameras()
    observed, truth = [], []
    for index, spec in enumerate(object_specs(stage)):
        identifier = f"o{index:03d}"
        relative_image = f"observed/{identifier}.npz"
        image_path = root/relative_image
        with image_path.open("xb") as stream:
            np.savez_compressed(stream, images=render_object(spec, supplied_cameras, samples_per_axis=7))
        with image_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        observed.append({"id": identifier, "cameras": supplied_cameras,
                         "image_file": relative_image, "sha256": digest})
        truth.append({"id": identifier, "true_depth": spec["true_depth"],
                      "mechanism": spec["mechanism"], "producer_parameters": spec})
    _write_json_exclusive(truth_dir/"metadata.json", truth)
    _write_json_exclusive(observed_dir/"inputs.json", observed)
    return {"stage": stage, "object_count": len(observed), "root": str(root),
            "observed_manifest": str(observed_dir/"inputs.json"),
            "truth_manifest": str(truth_dir/"metadata.json"), "samples_per_axis": 7}
