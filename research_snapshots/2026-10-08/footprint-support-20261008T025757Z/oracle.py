"""Truth-dependent ownership diagnostics for the existing physical renderer.

Importing this module never renders, scores, evaluates, or writes artifacts.
The historical renderer is loaded lazily and used read-only.  These fields are
oracle diagnostics, not observations available to the ordinary estimator.
"""

from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import map_coordinates


HISTORICAL_RENDERER = Path(
    "/srv/slam-research/grf/map-denoise/runs/"
    "surface-owned-support-20261008T022918Z/mechanism/run_mechanism.py"
)
AREA_OFFSETS = np.array(
    [(ox, oy) for oy in (-1 / 3, 0., 1 / 3)
     for ox in (-1 / 3, 0., 1 / 3)], dtype=float
)
AREA_OFFSETS.setflags(write=False)


@lru_cache(maxsize=1)
def _renderer():
    """Load only definitions; suppress bytecode writes to the historical run."""
    spec = importlib.util.spec_from_file_location(
        "_footprint_readonly_renderer", HISTORICAL_RENDERER
    )
    module = importlib.util.module_from_spec(spec)
    old_flag = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_flag
    return module


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize ownership metadata: {type(value)!r}")


def _ownership_key(meta):
    # Ownership does not depend on textures or phases.  Constant textures avoid
    # needless appearance evaluation inside the unmodified renderer. All plane
    # geometry, IDs, ordering, actual camera matrices, and target ID are retained.
    scene = meta["scene"]
    planes = [dict(plane, texture="flat") for plane in scene["planes"]]
    geometry = dict(target_id=scene["target_id"], planes=planes, phases=[])
    actual_cameras = meta["actual_cameras"]
    if len(actual_cameras) != 3:
        raise ValueError("The frozen experiment requires exactly three cameras")
    return json.dumps(
        dict(scene=geometry, actual_cameras=actual_cameras), sort_keys=True,
        separators=(",", ":"), allow_nan=False, default=_json_default
    )


def _appearance_key(meta):
    return json.dumps(
        dict(scene=meta["scene"], actual_cameras=meta["actual_cameras"]),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=_json_default
    )


@lru_cache(maxsize=12)
def _cached_fields(geometry_key, height, width):
    data = json.loads(geometry_key)
    scene = data["scene"]
    renderer = _renderer()
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx, yy), axis=-1).astype(float)
    # One vectorized intersection call per camera for all nine rays. The center
    # ray is the fifth sample, so it requires no separate intersection pass.
    rays_uv = pixels[None, ...] + AREA_OFFSETS[:, None, None, :]
    center, area = [], []
    for camera in data["actual_cameras"]:
        camera = renderer.deserialize_cam(camera)
        owner = renderer.intersect(scene, camera, rays_uv)["owner"]
        membership = owner == scene["target_id"]
        center.append(membership[4])
        area.append(membership.mean(axis=0))
    center = np.stack(center)
    area = np.stack(area)
    # Callers may safely reuse these cached arrays but must not mutate them.
    center.setflags(write=False)
    area.setflags(write=False)
    return center, area


@lru_cache(maxsize=12)
def _cached_components(appearance_key, height, width):
    """Actual target/background intensity contributions, for stronger controls."""
    data = json.loads(appearance_key)
    renderer = _renderer()
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx, yy), axis=-1).astype(float)
    rays_uv = pixels[None, ...] + AREA_OFFSETS[:, None, None, :]
    numerator, background = [], []
    for camera in data["actual_cameras"]:
        hit = renderer.intersect(
            data["scene"], renderer.deserialize_cam(camera), rays_uv
        )
        owned = hit["owner"] == data["scene"]["target_id"]
        numerator.append(np.where(owned, hit["value"], 0.).mean(axis=0))
        background.append(np.where(owned, 0., hit["value"]).mean(axis=0))
    numerator, background = np.stack(numerator), np.stack(background)
    numerator.setflags(write=False)
    background.setflags(write=False)
    return numerator, background


def ownership_fields(meta, images_shape=(128, 128)):
    """Return center labels, area fractions, and target intensity contributions.

    All arrays have shape ``(3, H, W)`` in actual rendered camera order.
    ``center`` is boolean target membership at integer image coordinates;
    ``area`` averages membership at offsets {-1/3, 0, 1/3}², exactly as the
    historical renderer integrates image intensities.  This is a discrete
    nine-ray fraction, not an analytic integral of continuous pixel area.

    ``numerator`` averages ``intensity * target_membership`` over those same
    rays. It carries target appearance truth beyond known fractions and is
    exclusively for an explicitly stronger component-unmixing oracle control.

    Only ``scene`` and ``actual_cameras`` are accessed from metadata. Camera
    projection, including any supplied-camera bias, belongs to the caller.
    Center/area cache keys ignore appearance/seed; numerator keys include both.
    """
    if len(images_shape) != 2:
        raise ValueError("images_shape must be (height, width)")
    height, width = images_shape
    if (int(height) != height or int(width) != width
            or height <= 0 or width <= 0):
        raise ValueError("Image dimensions must be positive integers")
    center, area = _cached_fields(_ownership_key(meta), int(height), int(width))
    numerator, _ = _cached_components(
        _appearance_key(meta), int(height), int(width)
    )
    return dict(center=center, area=area, numerator=numerator)


def sample_fraction(fields, uv):
    """Bilinearly sample integrated target membership at (..., x/y) locations.

    ``fields`` may be the ownership_fields dictionary, its ``area`` array
    ``(3,H,W)``, or one camera's ``area[camera]`` array ``(H,W)``.  The same
    coordinate batch is applied to each supplied camera, yielding respectively
    ``(3, ...)`` or ``(...)``. For different coordinates per camera, pass each
    camera's 2-D field separately. A single ``uv=(x,y)`` is also supported.

    Uses scipy map_coordinates(order=1, mode='constant', cval=NaN), matching
    the image sampler's finite domain. Nonfinite or out-of-image coordinates
    return NaN; they are invalid observations, not zero target membership.
    Interior values equal the weighted ownership of all four neighboring
    pixels' nine area rays (36 ray contributions, including zero weights).
    Passing a 2-D/3-D numerator array uses the identical interpolation kernel
    for the component oracle; it does not normalize or clip intensities.
    """
    area = fields["area"] if isinstance(fields, dict) else fields
    area = np.asarray(area, dtype=float)
    if area.ndim not in (2, 3):
        raise ValueError("Expected area field shape (H,W) or (cameras,H,W)")
    uv = np.asarray(uv, dtype=float)
    if uv.ndim < 1 or uv.shape[-1] != 2:
        raise ValueError("uv must have shape (..., 2) in x/y order")
    batch_shape = uv.shape[:-1]
    flat = uv.reshape(-1, 2)
    finite = np.isfinite(flat).all(axis=1)
    # Avoid relying on scipy's behavior for infinite coordinate conversion.
    coords = np.where(finite[:, None], flat, -1.).T[[1, 0]]

    def sample_one(field):
        sampled = map_coordinates(
            field, coords, order=1, mode="constant", cval=np.nan,
            prefilter=False
        )
        sampled[~finite] = np.nan
        return sampled.reshape(batch_shape)

    if area.ndim == 2:
        return sample_one(area)
    return np.stack([sample_one(field) for field in area])
