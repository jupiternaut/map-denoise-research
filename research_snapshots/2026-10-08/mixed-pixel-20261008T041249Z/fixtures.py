"""Independent E1 observations and separate privileged reference auxiliaries.

Importing does not generate data. Only create_e1 writes the full development
set. This module is a producer, never an ordinary-estimator dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MECHANISMS = ("flat_contrast", "flat_equal", "textured_boundary", "textured_single")
HALF_WIDTHS = (1.65, 2.65)
SEEDS = (1103, 2207, 3301)
OFFSETS = np.array([(x, y) for y in (-1 / 3, 0., 1 / 3)
                    for x in (-1 / 3, 0., 1 / 3)], dtype=np.float64)
IMAGE_SHAPE = (128, 128)


def cameras():
    """Exact supplied camera convention: world to camera Ry @ Rz."""
    K = np.array([[160., 0., 64.], [0., 160., 64.], [0., 0., 1.]])
    result = [dict(K=K.copy(), R=np.eye(3), C=np.zeros(3))]
    for sign in (-1, 1):
        a, b = sign * .07, sign * .11
        Ry = np.array([[np.cos(a), 0., np.sin(a)], [0., 1., 0.],
                       [-np.sin(a), 0., np.cos(a)]])
        Rz = np.array([[np.cos(b), -np.sin(b), 0.],
                       [np.sin(b), np.cos(b), 0.], [0., 0., 1.]])
        result.append(dict(K=K.copy(), R=Ry @ Rz,
                           C=np.array([sign * 60., 0., 0.])))
    return result


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    default=jsonable, allow_nan=False) + "\n")


def make_world(mechanism, half_width, seed, background_pair=0):
    """Return producer-only scene parameters; never passed to an estimator."""
    if mechanism not in MECHANISMS:
        raise ValueError("Unknown mechanism")
    if half_width not in HALF_WIDTHS or seed not in SEEDS:
        raise ValueError("E1 factors must match the locked design")
    if background_pair not in (0, 1):
        raise ValueError("background_pair must be 0 or 1")
    if background_pair and mechanism not in ("flat_contrast", "textured_boundary"):
        raise ValueError("Background interventions only exist in two contrast classes")
    # One seed controls target and both potential backgrounds before choosing
    # the intervention. Selecting the background never advances target RNG.
    rng = np.random.default_rng(seed)
    center = np.array([64., 64.]) + rng.uniform(-.18, .18, 2)
    foreground_level = float(rng.uniform(175., 195.))
    equal_level = float(rng.uniform(105., 145.))
    phases = rng.uniform(-np.pi, np.pi, (3, 4))
    return dict(mechanism=mechanism, half_width=float(half_width), seed=int(seed),
                background_pair=int(background_pair), center=center,
                half_height=8., angle=.17, true_depth=600., background_depth=900.,
                foreground_level=foreground_level, equal_level=equal_level,
                phases=phases)


def reference_polygon(world):
    if world["mechanism"] == "textured_single":
        return np.empty((0, 2), dtype=np.float64)
    w, h, a = world["half_width"], world["half_height"], world["angle"]
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
    return corners @ rotation.T + world["center"]


def foreground_membership(world, uv):
    if world["mechanism"] == "textured_single":
        return np.ones(uv.shape[:-1], dtype=bool)
    d = uv - world["center"]
    c, s = np.cos(world["angle"]), np.sin(world["angle"])
    # Inverse rotation evaluated independently of the predictor polygon test.
    x = c * d[..., 0] + s * d[..., 1]
    y = -s * d[..., 0] + c * d[..., 1]
    return (np.abs(x) <= world["half_width"]) & (np.abs(y) <= world["half_height"])


def radiance(world, uv, layer):
    """Continuous, bounded grayscale radiance in reference pixel coordinates."""
    if layer not in ("foreground", "background"):
        raise ValueError("Unknown layer")
    mechanism = world["mechanism"]
    if mechanism == "flat_equal":
        return np.full(uv.shape[:-1], world["equal_level"], dtype=np.float64)
    if layer == "foreground":
        level, phase_index, amplitude = world["foreground_level"], 0, 25.
    else:
        phase_index = 1 + world["background_pair"]
        level = (45., 95.)[world["background_pair"]]
        amplitude = 18.
    if mechanism == "flat_contrast":
        return np.full(uv.shape[:-1], level, dtype=np.float64)
    x, y = uv[..., 0] - 64., uv[..., 1] - 64.
    phase = world["phases"][phase_index]
    # No imported historical renderer or scoring implementation. The oracle
    # gets array samples of this field, never these coefficients or phases.
    value = (.45 * np.cos(.31 * x + .17 * y + phase[0])
             + .30 * np.sin(.12 * x - .37 * y + phase[1])
             + .15 * np.cos(.51 * x - .23 * y + phase[2])
             + .10 * np.sin(.29 * x + .43 * y + phase[3]))
    return np.asarray(level + amplitude * value, dtype=np.float64)


def _world_intersection(camera, sensor_uv, plane_z):
    homogeneous = np.concatenate((sensor_uv, np.ones(sensor_uv.shape[:-1] + (1,))), axis=-1)
    directions = (homogeneous @ np.linalg.inv(camera["K"]).T) @ camera["R"]
    distance = (plane_z - camera["C"][2]) / directions[..., 2]
    return camera["C"] + distance[..., None] * directions


def _reference_uv(points):
    # Reference camera is the supplied identity camera. This direct producer
    # mapping deliberately does not reuse predictor homography/warp code.
    return 160. * points[..., :2] / points[..., 2, None] + 64.


def render_world(world, camera_records=None, return_diagnostics=False):
    """Render independent 3x3 equal-area midpoint samples of opaque planes.

    Optional source ownership is producer diagnostics only and is never
    exported into observed/ or oracle/ and never used for candidate scoring.
    """
    camera_records = cameras() if camera_records is None else camera_records
    yy, xx = np.indices(IMAGE_SHAPE, dtype=np.float64)
    sensor_centers = np.stack((xx, yy), axis=-1)
    sensor_rays = sensor_centers[None, ...] + OFFSETS[:, None, None, :]
    images, coverage = [], []
    for camera in camera_records:
        foreground_uv = _reference_uv(_world_intersection(camera, sensor_rays, world["true_depth"]))
        background_uv = _reference_uv(_world_intersection(camera, sensor_rays, world["background_depth"]))
        owned = foreground_membership(world, foreground_uv)
        # This also makes F=B exactly constant at every ray and pixel, despite
        # a hidden geometric silhouette: no texture, sky, noise or clipping.
        values = np.where(owned, radiance(world, foreground_uv, "foreground"),
                          radiance(world, background_uv, "background"))
        images.append(values.mean(axis=0))
        if return_diagnostics:
            coverage.append(owned.mean(axis=0))
    images = np.stack(images)
    if return_diagnostics:
        return images, dict(source_alpha=np.stack(coverage))
    return images


def oracle_aux(world):
    """Same predictor fields as estimation; intentionally no target depth."""
    yy, xx = np.indices(IMAGE_SHAPE, dtype=np.float64)
    uv = np.stack((xx, yy), axis=-1)
    single = world["mechanism"] == "textured_single"
    return dict(mode=np.asarray("single" if single else "two"),
                foreground=radiance(world, uv, "foreground"),
                background=radiance(world, uv, "background"),
                mask=foreground_membership(world, uv).astype(np.float64),
                polygon=reference_polygon(world),
                background_depth=np.asarray(world["background_depth"]),
                valid=np.asarray(True), auxiliary_valid=np.asarray(True),
                sigma_valid=np.asarray(False))


def world_specs():
    result = []
    for mechanism in MECHANISMS:
        for half_width in HALF_WIDTHS:
            for seed in SEEDS:
                variants = (0, 1) if mechanism in ("flat_contrast", "textured_boundary") else (0,)
                for pair in variants:
                    result.append(make_world(mechanism, half_width, seed, pair))
    # Observation ids and order carry no human-readable class/seed/phase label.
    order = np.random.default_rng(710208).permutation(len(result))
    return [result[i] for i in order]


def create_e1(output_dir):
    """Write new, unscored E1 observations. Refuse any existing destination.

    Main invokes this only after the source/protocol lock and E0 gate.
    """
    root = Path(output_dir).resolve()
    targets = [root / name for name in ("observed", "truth", "oracle")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite an existing fixture dataset")
    for path in targets:
        path.mkdir(parents=True)
    supplied_cameras = cameras()
    observed, truth, oracles = [], [], []
    for index, world in enumerate(world_specs()):
        identifier = f"w{index:03d}"
        image_file = root / "observed" / f"{identifier}.npz"
        oracle_file = root / "oracle" / f"{identifier}.npz"
        np.savez_compressed(image_file, images=render_world(world, supplied_cameras))
        np.savez_compressed(oracle_file, **oracle_aux(world))
        observed.append(dict(id=identifier, cameras=supplied_cameras,
                             image_file=str(image_file), sha256=sha256(image_file)))
        group = f"{world['mechanism']}_w{world['half_width']:.2f}_s{world['seed']}"
        truth.append(dict(id=identifier, group=group, group_id=group, mechanism=world["mechanism"],
                          background_pair=world["background_pair"], seed=world["seed"],
                          half_width=world["half_width"], true_depth=world["true_depth"],
                          background_depth=world["background_depth"], producer_parameters=world))
        oracles.append(dict(id=identifier, aux_file=str(oracle_file),
                            sha256=sha256(oracle_file), privileged=True,
                            target_depth_in_aux=False))
    write_json(root / "observed" / "inputs.json", observed)
    write_json(root / "truth" / "metadata.json", truth)
    write_json(root / "oracle" / "manifest.json", oracles)
    summary = dict(worlds=len(observed), groups=len({row["group"] for row in truth}),
                   background_pairs=sum(row["background_pair"] for row in truth),
                   observed_manifest_sha256=sha256(root / "observed" / "inputs.json"),
                   truth_manifest_sha256=sha256(root / "truth" / "metadata.json"),
                   oracle_manifest_sha256=sha256(root / "oracle" / "manifest.json"))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    arguments = parser.parse_args()
    print(json.dumps(create_e1(arguments.output_dir), indent=2))
