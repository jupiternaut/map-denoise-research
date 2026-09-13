"""Legal-input adapters for six frozen local scalar/ICP estimators.

Coordinates and sigma are in millimetres. Column 2 is the supplied common
normal coordinate. Estimation never receives a dataset path or evaluation data.
Call warmup() before collecting method timings; source_hashes() is also untimed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time

import numpy as np


METHODS = (
    "identity",
    "xyz_mixture",
    "frame_center_then_xyz",
    "joint_forced",
    "fast",
    "open3d_icp_then_xyz",
)

LEGACY_SOURCE = Path(
    "/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/"
    "tracks/t1_t2/experiment.py"
)
FAST_SOURCE = Path(
    "/home/grf/Documents/Codex/2026-09-11/map-denoise-t2-boundary-v1/"
    "baseline/scalar_reference.py"
)

_ORIGINS = {
    "identity": "unchanged input control",
    "xyz_mixture": "own fixed-noise scalar mixture, one/two layers, BIC selection",
    "frame_center_then_xyz": "own frame-normal-mean removal followed by shared scalar mixture",
    "joint_forced": "own latent layer/common-frame-bias EM, forced output without ambiguity guard",
    "fast": "own scalar hard profile reference; not an official JRMPC/BALM implementation",
    "open3d_icp_then_xyz": (
        "official Open3D point-to-point ICP followed by own shared scalar mixture; "
        "not JRMPC, BALM, or joint bundle adjustment"
    ),
}


def _load_source(path, name):
    """Import the exact frozen file without executing its command-line runner."""
    if name in sys.modules:
        loaded = sys.modules[name]
        if Path(loaded.__file__).resolve() != path.resolve():
            raise RuntimeError(f"module identity mismatch for {name}")
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Keep immutable source directories free of adapter-created bytecode files.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    finally:
        sys.dont_write_bytecode = previous
    return module


def _legacy():
    return _load_source(LEGACY_SOURCE, "_correlation_v1_frozen_t1_t2")


def _fast():
    return _load_source(FAST_SOURCE, "_correlation_v1_frozen_scalar_reference")


def source_hashes():
    """Return absolute-source-path -> SHA256 for adapter and frozen dependencies."""
    paths = (Path(__file__).resolve(), LEGACY_SOURCE, FAST_SOURCE)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def warmup(methods=METHODS, run_estimates=True):
    """Preimport dependencies and run a tiny generated observation per method.

    No data files are loaded. Record the returned costs outside the experiment
    timings. A small sample does not warm every possible numerical branch.
    """
    selected = tuple(methods)
    unknown = set(selected) - set(METHODS)
    if unknown:
        raise KeyError(f"unknown methods: {sorted(unknown)}")
    total_started = time.perf_counter()
    started = time.perf_counter()
    _legacy()
    if "fast" in selected:
        _fast()
    numerical_import_s = time.perf_counter() - started
    open3d_import_s = 0.0
    open3d_version = None
    if "open3d_icp_then_xyz" in selected:
        started = time.perf_counter()
        import open3d

        open3d_import_s = time.perf_counter() - started
        open3d_version = open3d.__version__
    import scipy

    method_warmup_s = {}
    if run_estimates:
        rng = np.random.default_rng(911071)
        frame = np.repeat(np.arange(8), 12)
        xyz = np.column_stack((
            rng.uniform(-30, 30, (len(frame), 2)),
            rng.normal(0, 1, len(frame)) + 4 * (np.arange(len(frame)) % 2),
        ))
        for method in selected:
            started = time.perf_counter()
            estimate(method, xyz, frame, 1.0)
            method_warmup_s[method] = time.perf_counter() - started
    return {
        "scope": "dependency imports and optional generated 8x12-point calls; outside experiment timing",
        "numerical_import_s": numerical_import_s,
        "open3d_import_s": open3d_import_s,
        "method_warmup_s": method_warmup_s,
        "total_warmup_s": time.perf_counter() - total_started,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "open3d_version": open3d_version,
        "python_executable": sys.executable,
        "threads": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def _prepare_input(xyz_mm, frame, sigma_mm):
    xyz = np.asarray(xyz_mm)
    frame_array = np.asarray(frame)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("xyz_mm must be a nonempty N by 3 array")
    if xyz.dtype.kind not in "fiu" or not np.isfinite(xyz).all():
        raise ValueError("xyz_mm must contain finite real coordinates")
    if frame_array.shape != (len(xyz),) or frame_array.dtype.kind not in "iu":
        raise ValueError("frame must contain one integer frame ID per point")
    sigma_array = np.asarray(sigma_mm)
    if sigma_array.shape != () or sigma_array.dtype.kind not in "fiu":
        raise ValueError("sigma_mm must be a positive finite scalar")
    sigma = float(sigma_array)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma_mm must be a positive finite scalar")
    # Frozen estimators index frames 0..nf-1. Sorting unique IDs preserves frame
    # membership; the ICP reference is therefore the lowest supplied frame ID.
    frame_ids, dense_frame = np.unique(frame_array, return_inverse=True)
    prior_bound = max(8.0, 8.0 * sigma)
    if not np.isfinite(prior_bound):
        raise ValueError("sigma_mm is too large for the fixed prior-bound rule")
    inp = _legacy().Input(
        xyz_mm=np.array(xyz, dtype=np.float64, order="C", copy=True),
        frame=np.array(dense_frame, dtype=np.int64, copy=True),
        roi=np.zeros(len(xyz), dtype=np.int64),
        sigma_mm=sigma,
        bias_bound_mm=prior_bound,
    )
    return inp, frame_ids


def estimate(method, xyz_mm, frame, sigma_mm):
    """Return (N-by-3 output in mm, metadata) using only the legal inputs.

    Input arrays are copied. All points retain their identity and order. Frame
    IDs may be sparse; no points or frames are removed. The fixed bias bound is
    a prior computed from sigma, never an intervention/truth-dependent bound.
    """
    if method not in METHODS:
        raise KeyError(method)
    inp, frame_ids = _prepare_input(xyz_mm, frame, sigma_mm)
    backend_method = "scalar_profile_hard" if method == "fast" else method
    backend = _fast() if method == "fast" else _legacy()
    output, bias, original_info = backend.estimate(inp, backend_method)
    output = np.asarray(output, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if output.shape != inp.xyz_mm.shape or not np.isfinite(output).all():
        raise FloatingPointError(f"{method}: invalid output shape or nonfinite output")
    if bias.shape != (len(frame_ids),) or not np.isfinite(bias).all():
        raise FloatingPointError(f"{method}: invalid frame bias output")
    info = dict(original_info)
    info.update(
        method=method,
        backend_method=backend_method,
        backend_info=dict(original_info),
        method_origin=_ORIGINS[method],
        source_path=str(FAST_SOURCE if method == "fast" else LEGACY_SOURCE),
        frame_ids=frame_ids.tolist(),
        bias_mm=bias.tolist(),
        sigma_mm=inp.sigma_mm,
        bias_bound_mm=inp.bias_bound_mm,
        bias_bound_role="fixed prior max(8 mm, 8*sigma_mm); not actual perturbation truth",
        bias_bound_used=bool(method == "joint_forced"),
        roi_policy="all zero; no known clean anchor ROI",
        normal_coordinate="xyz_mm[:, 2]; caller supplies common normal coordinate",
        point_count=len(output),
    )
    if method == "open3d_icp_then_xyz":
        info["icp_reference_frame_id"] = frame_ids[0].item()
    return output.copy(), info
