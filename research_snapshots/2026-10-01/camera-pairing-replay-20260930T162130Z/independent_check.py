"""Independent read-only checks for the camera-pairing replay.

This checker never imports the reconstruction, adapter, or scoring implementation.
It is an existing same-family checker, not a fresh semantic review. Preflight does
not open GT coordinates. Numerical evaluation must pass the prediction-seal gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

for _thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np

ROOT = Path("/srv/slam-research/grf/map-denoise/runs/camera-pairing-replay-20260930T162130Z")
OLD = Path("/srv/slam-research/grf/map-denoise/runs/upstream-photo-holdout-20260930T113213Z")
ROIS = {"scan24_window_left": 24, "scan24_gable_center": 24,
        "scan37_scissor_cross": 37, "scan37_clamp_jaw": 37}
REVIEW = {"reviewer_status": "existing", "review_independence": "same-family",
          "acceptance_status": "provisional"}
ARMS = ("KEEP", "restore", "construct_witness", "heldout_witness")
REFERENCES = {
    24: ("/srv/slam-research/grf/map-denoise/datasets/published-outputs-v2-reference/stl024_total.ply",
         "963f2893d40d72957acdeb0affcd8aaa888408b180f5a62b62af747554ba4665"),
    37: ("/srv/slam-research/grf/map-denoise/datasets/reconstruction-v22-scan37/stl037_total.ply",
         "dcd290f8d6bee24b51fa6df7d0fa3cf017e2c46d9edea0897dc935af2cc56c55"),
}
_GT_ALLOWED = False


class CheckError(RuntimeError):
    pass


def require(value, reason):
    if not value:
        raise CheckError(reason)


def identity():
    require(socket.gethostname().split(".")[0] == "liekkas", "wrong host")
    require(Path(__file__).resolve() == ROOT / "independent_check.py", "wrong checker path")
    require(ROOT.resolve() == ROOT and OLD.resolve() == OLD, "unexpected target symlink")


def install_gt_read_guard():
    """Fail before any PLY/GT file access until the full prediction seal passes."""
    def guard(event, args):
        if event != "open" or isinstance(args[0], int) or _GT_ALLOWED:
            return
        path = Path(args[0]).resolve()
        parts = {part.lower() for part in path.parts}
        if path.suffix.lower() in {".ply", ".mat"} or parts & {"ground_truth", "gt", "references"}:
            raise CheckError("GT/geometry read forbidden before PREDICTIONS_SEALED.json verification")
    sys.addaudithook(guard)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def arrays(path):
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key].copy() for key in handle.files}


def verify_manifest(base, record):
    files = record.get("files")
    require(isinstance(files, dict) and bool(files), "missing file/hash manifest")
    verified = {}
    for name, expected in files.items():
        require(not Path(name).is_absolute(), f"absolute manifest member: {name}")
        path = (base / name).resolve()
        require(path.is_relative_to(base.resolve()), f"manifest path escapes target: {name}")
        require(path.is_file(), f"missing manifest file: {name}")
        require(isinstance(expected, str) and len(expected) == 64, f"invalid digest: {name}")
        actual = digest(path)
        require(actual == expected, f"sealed file changed: {name}")
        verified[name] = actual
    return verified


def prediction_seal_gate(*, authorize_gt=False):
    """Verify the actual new seal before permitting any GT coordinate read."""
    global _GT_ALLOWED
    path = ROOT / "PREDICTIONS_SEALED.json"
    require(path.is_file(), "PREDICTIONS_SEALED.json absent; GT access remains forbidden")
    record = load_json(path)
    require(record.get("status", "complete") == "complete", "prediction seal is incomplete")
    require(record.get("expected_case_count") == 24, "prediction seal must declare all 24 cases")
    verified = verify_manifest(ROOT, record)
    lock_path = ROOT / "EVALUATION_PROTOCOL_LOCK.json"
    lock = load_json(lock_path)
    require(lock.get("before_new_GT_evaluation") is True, "evaluation protocol lock not declared pre-GT")
    protocol = verify_manifest(ROOT, lock)
    require(protocol.get("PREDICTIONS_SEALED.json") == digest(path), "evaluation protocol lock refers to another prediction seal")
    required = {"CAMERA_MAPPING.json", "PLAN_CORRECTED.json", "INITIALIZER_PROTOCOL.md",
                "corrected_rebuild.py", "replay.py", "run_all.py", "evaluate_replay.py",
                "AGENTS.md", "refine-logs/EXPERIMENT_PLAN.md"}
    combined = set(verified) | set(protocol)
    require(required <= combined, "prediction/protocol seals omit code/protocol: " + str(sorted(required - combined)))
    if authorize_gt:
        _GT_ALLOWED = True
    return {"sha256": digest(path), "verified_files": len(verified),
            "protocol_lock_sha256": digest(lock_path), "verified_protocol_files": len(protocol)}


def uv_keys(uv):
    value = np.asarray(uv, dtype="<f8")
    require(value.ndim == 2 and value.shape[1] == 2 and np.isfinite(value).all(), "invalid UV array")
    keys = [row.tobytes() for row in value]
    require(len(keys) == len(set(keys)), "duplicate pixel coordinates")
    return keys


def hash_order(uv):
    keys = uv_keys(uv)
    return np.asarray(sorted(range(len(keys)), key=lambda i: hashlib.sha256(keys[i]).digest()), dtype=np.int64)


def priority_context_indices(grid_uv, qualified, fixed_query_uv, budget=384):
    """Same deterministic selector for U0 and U1; final rows retain hash order.

    Reserving passing fixed queries changes the unmodified hash-top384 policy in
    U1 when query ranks exceed the budget. Record this explicitly, even when the
    symmetric selector exactly reproduces the already-frozen U0 context.
    """
    keys = uv_keys(grid_uv)
    query_keys = set(uv_keys(fixed_query_uv))
    qualified = np.asarray(qualified)
    require(qualified.shape == (len(keys),) and qualified.dtype == bool, "qualified must be a boolean grid mask")
    require(query_keys <= set(keys), "fixed query missing from registered pixel grid")
    ordered = [int(i) for i in hash_order(grid_uv) if qualified[i]]
    reserved = [i for i in ordered if keys[i] in query_keys]
    require(len(reserved) <= budget, "fixed passing queries exceed context budget")
    selected = set(reserved)
    selected.update(i for i in [j for j in ordered if j not in selected][:budget - len(selected)])
    result = np.asarray([i for i in ordered if i in selected], dtype=np.int64)
    plain = set(ordered[:budget])
    forced = selected - plain
    displaced = plain - selected
    return result, {
        "qualified_count": len(ordered), "context_count": len(result),
        "passing_fixed_queries": len(reserved),
        "forced_query_count_vs_plain_hash": len(forced),
        "displaced_nonquery_count_vs_plain_hash": len(displaced),
        "forced_grid_indices": sorted(forced), "displaced_grid_indices": sorted(displaced),
        "final_context_order": "original pixel SHA256 order",
    }


def check_context_selection(grid_uv, qualified, fixed_query_uv, context_indices, query_context_indices,
                            policy="query_priority_then_hash", budget=384):
    keys = uv_keys(grid_uv)
    query_keys = uv_keys(fixed_query_uv)
    qualified = np.asarray(qualified)
    require(qualified.dtype == bool and qualified.shape == (len(keys),), "invalid qualified mask")
    got = np.asarray(context_indices)
    require(got.ndim == 1 and got.dtype.kind in "iu" and len(set(got.tolist())) == len(got), "invalid context indices")
    require(((got >= 0) & (got < len(keys))).all(), "context indices outside grid")
    expected, details = priority_context_indices(grid_uv, qualified, fixed_query_uv, budget)
    if policy == "plain_hash":
        expected = np.asarray([i for i in hash_order(grid_uv) if qualified[i]][:budget], dtype=np.int64)
    else:
        require(policy == "query_priority_then_hash", "unknown context policy")
    require(np.array_equal(got, expected), "context selection/order differs from frozen policy")
    query_rows = np.asarray(query_context_indices)
    require(query_rows.shape == (len(query_keys),) and query_rows.dtype.kind in "iu", "invalid fixed-query context rows")
    lookup = {keys[int(i)]: row for row, i in enumerate(got)}
    expected_rows = np.asarray([lookup.get(key, -1) for key in query_keys])
    require(np.array_equal(query_rows, expected_rows), "query identities or failure sentinels differ")
    qualified_lookup = {key: bool(qualified[i]) for i, key in enumerate(keys)}
    reasons = ["valid" if row >= 0 else "budget_excluded" if qualified_lookup[key] else "threshold_failed"
               for key, row in zip(query_keys, expected_rows)]
    details.update(policy=policy, fixed_query_count=len(query_keys),
                   valid_fixed_queries=int((expected_rows >= 0).sum()),
                   threshold_failed=reasons.count("threshold_failed"),
                   budget_excluded=reasons.count("budget_excluded"),
                   failure_reason=reasons)
    return details


def old_context_preflight():
    """Check fixed-query roster and a symmetric selector's U0 identity, without GT."""
    seal = load_json(OLD / "SEALED.json")
    require(seal.get("status") == "complete", "old run not complete")
    verified = verify_manifest(OLD, seal)
    plan = load_json(OLD / "PLAN.json")
    report = {"old_seal_sha256": digest(OLD / "SEALED.json"), "old_sealed_files_verified": len(verified),
              "rois": {}, "base_query_count": 0, "gt_opened": False}
    seen_rois = {r["id"] for s in plan["scenes"].values() for r in s["rois"]}
    require(seen_rois == set(ROIS), "old ROI roster differs")
    for name, sid in ROIS.items():
        folder = OLD / "construction-dense" / f"scan{sid}" / name
        inp = arrays(folder / "input.npz")
        rebuild = load_json(folder / "REBUILD.json")
        uv, ids, points = inp["reference_pixel_xy"], inp["query_ids"], inp["points_mm"]
        uv_keys(uv)
        require(points.shape == (len(uv), 3) and np.isfinite(points).all(), "invalid old context")
        require(64 <= len(uv) <= 384 and len(ids) == 128, "unexpected context/query budget")
        require(np.array_equal(ids, np.linspace(0, len(uv) - 1, 128, dtype=np.int64)), "old query rule mismatch")
        require(np.array_equal(hash_order(uv), np.arange(len(uv))), "old context not in pixel hash order")
        query_uv = uv[ids]
        order, _ = priority_context_indices(uv, np.ones(len(uv), dtype=bool), query_uv)
        require(np.array_equal(order, np.arange(len(uv))), "symmetric priority rule changes U0 context")
        for condition in ("native", "minus3", "plus3"):
            stored = arrays(OLD / "inference" / f"scan{sid}" / f"{name}__{condition}" / "INPUT.npz")
            require(np.array_equal(stored["query_ids"], ids), "old condition query IDs differ")
            require(np.array_equal(stored["reference_pixel_xy"], query_uv), "old condition query UV differs")
        report["rois"][name] = {
            "scene": sid, "context_count": len(uv), "fixed_query_count": len(ids),
            "registered_grid_count": rebuild["grid_rows"], "old_qualified_count": rebuild["passing_rows"],
            "priority_rule_reproduces_saved_U0_context_and_order": True,
            "proof_boundary": "old full qualified UV list is not saved; original hash-top384 code plus query subset establishes unseen passing pixels cannot displace old context",
            "input_sha256": digest(folder / "input.npz"),
            "pixel_ids": [name + ":" + hashlib.sha256(k).hexdigest() for k in uv_keys(query_uv)],
            "fixed_query_uv": query_uv.tolist(),
        }
        report["base_query_count"] += len(ids)
    require(report["base_query_count"] == 512, "fixed base-query denominator must be 512")
    return report


def identical(first, second, label, *, nan_equal=False):
    require(first.shape == second.shape and first.dtype == second.dtype, f"array type/shape changed: {label}")
    require(np.array_equal(first, second, equal_nan=nan_equal), f"array changed: {label}")


def initializer_checks(name, sid):
    folder = ROOT / "initialization" / f"scan{sid}" / name
    value = arrays(folder / "input.npz")
    info = load_json(folder / "REBUILD.json")
    old = arrays(OLD / "construction-dense" / f"scan{sid}" / name / "input.npz")
    fixed = old["reference_pixel_xy"][old["query_ids"]]
    identical(value["fixed_query_pixel_xy"], fixed, name + " fixed query UV")
    selected = value["selected_context_grid_indices"]
    detail = check_context_selection(value["grid_uv"], value["qualified_mask"], fixed,
                                     selected, value["query_context_indices"])
    require(np.array_equal(value["reference_pixel_xy"], value["grid_uv"][selected]), "context UV/grid mismatch")
    qualified_hash = np.asarray([i for i in hash_order(value["grid_uv"]) if value["qualified_mask"][i]])
    require(np.array_equal(value["hash_order"], qualified_hash), "recorded qualified hash order mismatch")
    fixed_grid = value["fixed_query_grid_indices"]
    require(fixed_grid.shape == (128,) and np.array_equal(value["grid_uv"][fixed_grid], fixed), "fixed grid IDs mismatch")
    success = value["qualified_mask"][fixed_grid]
    identical(value["construction_success"], success, "construction success mask")
    require(np.array_equal(value["query_old_ids"], np.flatnonzero(success)), "valid query denominator changed")
    require(np.array_equal(value["query_ids"], value["query_context_indices"][success]), "query context IDs mismatch")
    require(np.array_equal(value["query_pixel_xy"], fixed[success]), "valid query pixels changed")
    require(np.array_equal(value["failure_reason"], np.where(success, "", "quality_threshold")), "failure reason mismatch")
    require(value["points_mm"].shape == (len(selected), 3) and np.isfinite(value["points_mm"]).all(), "initializer points invalid")
    require(value["construction_zncc"].shape == (len(selected),) and
            (value["construction_zncc"] >= .6).all(), "initializer changed ZNCC threshold")
    score = np.asarray(info["fixed_query_score"])
    variance = np.asarray(info["fixed_query_variance"])
    require(score.shape == variance.shape == (128,), "missing full fixed-query threshold evidence")
    require(np.array_equal(success, (score >= .6) & (variance > 1e-5)), "fixed-query quality rule differs")
    for field, expected in (("fixed_query_rows", 128), ("successful_query_rows", int(success.sum())),
                            ("failed_query_rows", int((~success).sum())), ("context_rows", len(selected)),
                            ("forced_queries_outside_pure_hash384", detail["forced_query_count_vs_plain_hash"])):
        require(info[field] == expected, f"initializer report count mismatch: {field}")
    require(info["mapping_sha256"] == digest(ROOT / "CAMERA_MAPPING.json"), "initializer mapping changed")
    require(info["plan_sha256"] == digest(ROOT / "PLAN_CORRECTED.json"), "initializer plan changed")
    require(info["input_source_sha256"] == digest(OLD / "construction-dense" / f"scan{sid}" / name / "input.npz"), "old initializer source changed")
    detail.update(roi=name, scene=sid, input_sha256=digest(folder / "input.npz"))
    return value, detail


def witness_checks(folder, pool, inp):
    out, routes, witness = (arrays(folder / name) for name in ("points_outputs.npz", "routes.npz", "witness.npz"))
    geometry = pool["geometry_mm"]
    n = len(geometry)
    original = pool["restore_routes"]
    require(np.array_equal(routes["original_routes"], original), "original route changed")
    require(np.array_equal(routes["query_ids"], inp["query_ids"]), "witness query IDs changed")
    require(np.array_equal(routes["query_old_ids"], inp["query_old_ids"]), "witness original pixel IDs changed")
    require((routes["KEEP"] == 0).all() and np.array_equal(routes["restore"], original), "KEEP/restore route changed")
    counts = {}
    for arm in ARMS:
        route = routes[arm]
        require(route.shape == (n,) and route.dtype.kind in "iu" and np.isin(route, [0, 1, 2]).all(), "invalid output route")
        require(out[arm].shape == (n, 3) and np.isfinite(out[arm]).all(), "invalid output coordinates")
        require(np.array_equal(out[arm], geometry[np.arange(n), route]), "output created new coordinates")
        counts[arm] = int(np.count_nonzero(np.linalg.norm(out[arm] - out["KEEP"], axis=1) > 1e-9))
    for prefix, arm in (("construct", "construct_witness"), ("heldout", "heldout_witness")):
        scores, pixels = witness[prefix + "_scores"], witness[prefix + "_shared_pixels"]
        require(scores.shape == (n, 4, 2) and not np.isinf(scores).any(), "invalid witness scores")
        require(pixels.shape == (n, 4) and pixels.dtype.kind in "iu" and np.isin(pixels, range(50)).all(), "invalid shared pixels")
        valid = np.isfinite(scores).all(axis=2)
        require((pixels[valid] >= 40).all(), "witness violates 80% common-pixel threshold")
        margins = np.where(valid, scores[:, :, 1] - scores[:, :, 0], 0.)
        count = valid.sum(1)
        positive = ((margins > 0) & valid).sum(1)
        mean = margins.sum(1) / np.maximum(count, 1)
        accepted = (count >= 3) & (positive >= 3) & (mean > 0)
        require(np.array_equal(routes[prefix + "_accepted"], accepted), "witness acceptance rule changed")
        require(np.array_equal(routes[prefix + "_valid_views"], count), "valid-view count changed")
        require(np.array_equal(witness[prefix + "_positive_views"], positive), "positive-view count changed")
        require(np.array_equal(witness[prefix + "_mean_margin"], mean), "mean margin changed")
        require(np.array_equal(routes[arm], np.where(accepted, original, 0)), "witness is not veto-only")
    return out, witness, counts


def event_checks(folder, spec):
    meta, events = load_json(folder / "CASE.json"), load_json(folder / "READ_EVENTS.json")
    candidate_path = folder / "candidates" / "SEALED.json"
    require(meta["candidate_sha256"] == digest(candidate_path), "candidate seal identity changed")
    require(meta.get("gt_accessed") is False and meta.get("labels_accessed") is False, "prohibited-access declaration")
    candidate_sealed = False
    groups = {key: set(spec[key]) for key in ("Q", "C", "H")}
    for index, event in enumerate(events):
        require(event["sequence"] == index, "read event sequence changed")
        if event["phase"] == "candidate_seal":
            require(event["sha256"] == digest(candidate_path), "event seal identity changed")
            candidate_sealed = True
            continue
        path, group = Path(event["path"]), event["group"]
        require(digest(path) == event["sha256"], f"source changed: {path}")
        if group in {"Q", "C", "H", "reference"}:
            require(path.parent.resolve() == Path(spec["images_dir"]).resolve(), "photo came from wrong dataset")
            require(path.name == event["view_id"], "photo/view identity differs")
            require(path.name == spec["reference"] if group == "reference" else path.name in groups[group], "photo group mismatch")
        if group == "H":
            require(candidate_sealed and event["phase"] == "witness_heldout", "H pixels before candidate seal")
    require(candidate_sealed, "missing candidate seal event")
    sources = load_json(candidate_path)["sources"]
    hpaths = {str((Path(spec["images_dir"]) / name).resolve()) for name in spec["H"]}
    require(not set(sources) & hpaths, "H photograph in candidate source seal")
    return {"events": len(events), "H_after_candidate_seal": True,
            "boundary": "recorded bindings and inspected Python path; not arbitrary native-I/O attestation"}


def prediction_checks(*, partial=False):
    plan = load_json(ROOT / "PLAN_CORRECTED.json")
    old_plan = load_json(OLD / "PLAN.json")
    require(len(plan["cases"]) == 12 and {c["case_id"] for c in plan["cases"]} ==
            {c["case_id"] for c in old_plan["cases"]}, "case roster changed")
    report = {"gt_opened": False, "partial": partial, "cases": {}, "initializers": {}, "missing": []}
    initialized = {}
    for name, sid in ROIS.items():
        if not (ROOT / "initialization" / f"scan{sid}" / name / "input.npz").exists():
            require(partial, f"missing U1 initialization: {name}")
            report["missing"].append("initialization/" + name)
            continue
        initialized[name], report["initializers"][name] = initializer_checks(name, sid)
    for upstream in ("U0", "U1"):
        for case in plan["cases"]:
            cid, sid, name = case["case_id"], case["scene"], case["roi"]
            folder = ROOT / "predictions" / upstream / cid
            label = upstream + "/" + cid
            if not (folder / "SEALED.json").exists():
                require(partial, f"case not sealed: {label}")
                report["missing"].append(label)
                continue
            verify_manifest(folder, load_json(folder / "SEALED.json"))
            if (folder / "INCOMPLETE.json").exists():
                require(upstream == "U1", "U0 cannot lose original queries")
                incomplete = load_json(folder / "INCOMPLETE.json")
                source = initialized[name]
                require(len(source["query_ids"]) < 64 or len(source["points_mm"]) < 64, "unexpected incomplete case")
                require(incomplete["total_rows"] == 128 and incomplete["valid_rows"] == len(source["query_ids"]), "incomplete denominator differs")
                report["cases"][label] = {"status": "INCOMPLETE", "valid_rows": len(source["query_ids"]), "total_rows": 128}
                continue
            pool, inp = arrays(folder / "candidates/CANDIDATES.npz"), arrays(folder / "INPUT.npz")
            n = len(pool["geometry_mm"])
            require(64 <= n <= 128, "invalid completed query count")
            require(pool["geometry_mm"].shape == (n, 3, 3) and pool["x"].shape == (n, 2, 225) and pool["scores"].shape == (n, 2), "candidate shape differs")
            require(all(np.isfinite(pool[key]).all() for key in ("geometry_mm", "x", "scores")), "nonfinite candidates/features")
            require(pool["restore_routes"].shape == (n,) and np.isin(pool["restore_routes"], [0, 1, 2]).all(), "invalid frozen model routes")
            for key in ("query_ids", "query_old_ids"):
                identical(pool[key], inp[key], key)
            require(inp["normals"].shape == inp["points_mm"].shape and np.isfinite(inp["normals"]).all(), "invalid frozen normals")
            old_folder = OLD / "inference" / f"scan{sid}" / cid
            if upstream == "U0":
                source = arrays(OLD / "construction-dense" / f"scan{sid}" / name / "input.npz")
                saved = arrays(old_folder / "candidates/CANDIDATES.npz")
                for key in ("geometry_mm", "x", "scores", "restore_routes", "query_ids"):
                    identical(pool[key], saved[key], "U0 " + key)
                require(np.array_equal(inp["query_old_ids"], np.arange(128)), "U0 lost/reordered original queries")
            else:
                source = initialized[name]
                identical(inp["query_old_ids"], source["query_old_ids"], "U1 query_old_ids")
            identical(inp["query_ids"], source["query_ids"], "input query_ids")
            identical(inp["reference_pixel_xy"], source["reference_pixel_xy"], "input context pixels")
            ref = plan["scenes"][str(sid)]["reference"]
            center = np.asarray(plan["scenes"][str(sid)]["cameras"][ref]["center"])
            native = source["points_mm"]
            rays = native - center
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            bias = {"native": 0., "minus3": -3., "plus3": 3.}[case["condition"]]
            require(np.allclose(inp["points_mm"], native + bias * rays, rtol=0, atol=1e-10), "condition or mm-ray perturbation changed")
            require(np.allclose(pool["geometry_mm"][:, 0], inp["points_mm"][inp["query_ids"]], rtol=0, atol=1e-10), "candidate KEEP/input mismatch")
            pair = {}
            for w in ("W0", "W1"):
                out, witness, counts = witness_checks(folder / w, pool, inp)
                pair[w] = out
                if upstream == "U0" and w == "W0":
                    old_out, old_w = arrays(old_folder / "points_outputs.npz"), arrays(old_folder / "witness.npz")
                    for arm in ARMS:
                        identical(out[arm], old_out[arm], "U0W0 " + arm)
                    for key in old_w:
                        identical(witness[key], old_w[key], "U0W0 " + key, nan_equal=True)
            for arm in ("KEEP", "restore"):
                identical(pair["W0"][arm], pair["W1"][arm], "W-invariant " + arm)
            report["cases"][label] = {"status": "PASS", "valid_rows": n, "total_rows": 128,
                                      "context_rows": len(inp["points_mm"]), "events": event_checks(folder, plan["scenes"][str(sid)])}
    seal = ROOT / "PREDICTIONS_SEALED.json"
    if seal.exists():
        require(not report["missing"], "global seal exists with missing predictions")
        report["global_seal"] = prediction_seal_gate()
    else:
        report["global_seal"] = {"present": False, "GT_authorized": False}
    return report


def numeric_checks():
    """Use Open3D and sklearn KDTree; no author evaluation function is imported."""
    import open3d as o3d
    from sklearn.neighbors import KDTree

    checks = prediction_checks(partial=False)
    require(checks["global_seal"].get("verified_files", 0) > 0, "numeric check requires verified complete seal")
    seal = prediction_seal_gate(authorize_gt=True)
    primary_path = ROOT / "evaluation/RESULTS.json"
    primary = load_json(primary_path)
    require(primary["prediction_seal_sha256"] == seal["sha256"], "primary results refer to a different prediction seal")
    plan = load_json(ROOT / "PLAN_CORRECTED.json")
    references, trees, reference_report = {}, {}, {}
    for sid, (path, expected) in REFERENCES.items():
        require(primary["references"][str(sid)]["path"] == path, "primary reference identity changed")
        require(primary["references"][str(sid)]["sha256"] == expected, "primary reference hash changed")
        require(digest(path) == expected, "dataset reference changed")
        xyz = np.asarray(o3d.io.read_point_cloud(path).points).copy()
        require(xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) and np.isfinite(xyz).all(), "invalid PLY reference")
        references[sid], trees[sid] = xyz, KDTree(xyz, leaf_size=40, metric="euclidean")
        reference_report[str(sid)] = {"path": path, "sha256": expected, "points": len(xyz)}
    bundles = {}
    for u in ("U0", "U1"):
        for case in plan["cases"]:
            folder = ROOT / "predictions" / u / case["case_id"]
            if (folder / "INCOMPLETE.json").exists():
                continue
            pool = arrays(folder / "candidates/CANDIDATES.npz")
            b = {"pool": pool, "outputs": {}, "routes": {}, "distances": {}}
            for w in ("W0", "W1"):
                b["outputs"][w] = arrays(folder / w / "points_outputs.npz")
                b["routes"][w] = arrays(folder / w / "routes.npz")
                b["distances"][w] = {arm: trees[case["scene"]].query(points, k=1)[0][:, 0]
                                      for arm, points in b["outputs"][w].items()}
            b["pool_distances"] = trees[case["scene"]].query(pool["geometry_mm"].reshape(-1, 3), k=1)[0].reshape(-1, 3)
            bundles[u, case["case_id"]] = b
    def footprint(sid, points):
        hits = trees[sid].query_radius(points, r=1.)
        nonempty = [row for row in hits if len(row)]
        ids = np.unique(np.concatenate(nonempty)) if nonempty else np.empty(0, dtype=np.int64)
        return references[sid][ids]
    footprints, valid_base = {}, 0
    for case in plan["cases"]:
        if case["condition"] != "native":
            continue
        cid, sid, roi = case["case_id"], case["scene"], case["roi"]
        b0, b1 = bundles["U0", cid], bundles.get(("U1", cid))
        common = b1["pool"]["query_old_ids"] if b1 else np.empty(0, dtype=np.int64)
        valid_base += len(common)
        for u, b in (("U0", b0), ("U1", b1)):
            if b is not None:
                footprints[u, roi, "all_valid"] = footprint(sid, b["pool"]["geometry_mm"][:, 0])
        common_points = np.concatenate((b0["pool"]["geometry_mm"][common, 0], b1["pool"]["geometry_mm"][:, 0])) if b1 else np.empty((0, 3))
        fp = footprint(sid, common_points) if len(common_points) else np.empty((0, 3))
        for u in ("U0", "U1"):
            footprints[u, roi, "paired_common"] = fp
    differences, rows, brute_errors = [], [], []
    def compare(actual, expected, label):
        if isinstance(actual, dict):
            require(set(actual) == set(expected), "numeric field roster differs: " + label)
            for key in actual:
                compare(actual[key], expected[key], label + "/" + key)
        elif actual is None or isinstance(actual, (str, bool, list)):
            require(actual == expected, "numeric identity/null mismatch: " + label)
        else:
            require(expected is not None and np.isfinite(actual) and np.isfinite(expected), "nonfinite numeric field: " + label)
            delta = abs(float(actual) - float(expected))
            differences.append(delta)
            require(delta <= 1e-8, f"numeric mismatch {label}: {actual} versus {expected}")
    expected_rows = {(r["case_id"], r["upstream"], r["witness"], r["scope"]): r for r in primary["records"]}
    require(len(expected_rows) == len(primary["records"]), "duplicate primary result rows")
    for case in plan["cases"]:
        cid, sid, roi = case["case_id"], case["scene"], case["roi"]
        b1 = bundles.get(("U1", cid))
        common = b1["pool"]["query_old_ids"] if b1 else np.empty(0, dtype=np.int64)
        for u in ("U0", "U1"):
            b = bundles.get((u, cid))
            if b is None:
                continue
            for scope in ("all_valid", "paired_common"):
                if scope == "paired_common" and not len(common):
                    continue
                ix = common if scope == "paired_common" and u == "U0" else np.arange(len(b["pool"]["geometry_mm"]))
                oracle = np.min(b["pool_distances"][ix], axis=1) ** 2
                fp = footprints[u, roi, scope]
                for w in ("W0", "W1"):
                    outputs = {arm: value[ix] for arm, value in b["outputs"][w].items()}
                    distance = {arm: value[ix] for arm, value in b["distances"][w].items()}
                    keep_mse, oracle_mse = float(np.mean(distance["KEEP"] ** 2)), float(np.mean(oracle))
                    potential = float(np.mean(distance["KEEP"] ** 2 - oracle))
                    metrics, veto = {}, {}
                    for arm in ARMS:
                        move_mm = np.linalg.norm(outputs[arm] - outputs["KEEP"], axis=1)
                        moved = move_mm > 1e-9
                        change = distance["KEEP"] - distance[arm]
                        good, bad = moved & (change > .1), moved & (change < -.1)
                        mse = float(np.mean(distance[arm] ** 2))
                        coverage = float(np.mean(KDTree(outputs[arm]).query(fp, k=1)[0][:, 0] <= 1.)) if len(fp) else None
                        prefix = "construct" if arm == "construct_witness" else "heldout" if arm == "heldout_witness" else None
                        valid = b["routes"][w][prefix + "_valid_views"][ix] if prefix else None
                        metrics[arm] = {
                            "nearest_mse_mm2": mse,
                            "improvement_percent": 100. * (keep_mse - mse) / keep_mse if keep_mse > 0 else None,
                            "precision_1mm": float(np.mean(distance[arm] <= 1.)), "fixed_keep_footprint_coverage_1mm": coverage,
                            "moved_count": int(moved.sum()), "moved_fraction": float(moved.mean()),
                            "mean_movement_mm": float(move_mm[moved].mean()) if moved.any() else 0.,
                            "beneficial_count": int(good.sum()), "harmful_count": int(bad.sum()),
                            "neutral_moved_count": int((moved & ~(good | bad)).sum()),
                            "beneficial_error_reduction_sum_mm": float(change[good].sum()),
                            "beneficial_error_reduction_mean_mm": float(change[good].mean()) if good.any() else 0.,
                            "harmful_error_increase_sum_mm": float(-change[bad].sum()),
                            "harmful_error_increase_mean_mm": float(-change[bad].mean()) if bad.any() else 0.,
                            "beneficial_movement_sum_mm": float(move_mm[good].sum()), "harmful_movement_sum_mm": float(move_mm[bad].sum()),
                            "witness_valid_3of4_fraction": float(np.mean(valid >= 3)) if valid is not None else None,
                            "selection_regret_mm2": mse - oracle_mse,
                            "extracted_potential_fraction": float(np.mean(distance["KEEP"] ** 2 - distance[arm] ** 2)) / potential if potential > 0 else None,
                        }
                        if prefix:
                            original = b["routes"][w]["restore"][ix]
                            delta = distance["KEEP"] - distance["restore"]
                            good_original, bad_original = (original != 0) & (delta > .1), (original != 0) & (delta < -.1)
                            denied = (original != 0) & (b["routes"][w][arm][ix] == 0)
                            veto[arm] = {"bad_original": int(bad_original.sum()), "good_original": int(good_original.sum()),
                                         "bad_blocked": int((denied & bad_original).sum()), "good_lost": int((denied & good_original).sum()),
                                         "bad_blocked_fraction": float((denied & bad_original).sum() / bad_original.sum()) if bad_original.any() else None,
                                         "good_lost_fraction": float((denied & good_original).sum() / good_original.sum()) if good_original.any() else None}
                    row = {"case_id": cid, "scene": sid, "roi": roi, "condition": case["condition"],
                           "upstream": u, "witness": w, "scope": scope, "n": len(ix), "requested": 128,
                           "query_old_ids": b["pool"]["query_old_ids"][ix].tolist(), "footprint_n": len(fp),
                           "oracle_mse_mm2": oracle_mse, "potential_mm2": potential, "arms": metrics, "veto": veto}
                    key = cid, u, w, scope
                    require(key in expected_rows, "missing primary record: " + str(key))
                    compare(row, expected_rows[key], str(key))
                    rows.append(row)
            if u == "U1" and case["condition"] == "native":
                for qi in np.linspace(0, len(b["pool"]["geometry_mm"]) - 1, 5, dtype=int):
                    point = b["pool"]["geometry_mm"][qi, 0]
                    brute = float(np.sqrt(np.min(np.sum((references[sid] - point) ** 2, axis=1))))
                    delta = abs(brute - float(b["pool_distances"][qi, 0]))
                    require(delta < 1e-10, "independent tree/brute-force mismatch")
                    brute_errors.append(delta)
    require(len(rows) == len(expected_rows), "primary has unexpected/unverified records")
    require(primary["requested_base_queries"] == 512 and primary["valid_u1_base_queries"] == valid_base, "primary query denominator differs")
    summaries, acceptance = {}, {}
    for scope in ("all_valid", "paired_common"):
        summaries[scope], acceptance[scope] = {}, {}
        for u in ("U0", "U1"):
            for w in ("W0", "W1"):
                cell = u + w
                summaries[scope][cell], acceptance[scope][cell] = {}, {}
                for condition in ("native", "minus3", "plus3"):
                    subset = [r for r in rows if (r["scope"], r["upstream"], r["witness"], r["condition"]) == (scope, u, w, condition)]
                    if not subset:
                        continue
                    summary = {"roi_count": len(subset), "n": sum(r["n"] for r in subset),
                               "oracle_mse_mm2": float(np.mean([r["oracle_mse_mm2"] for r in subset])), "arms": {}}
                    for arm in ARMS:
                        summary["arms"][arm] = {field: float(np.mean([r["arms"][arm][field] for r in subset]))
                                                if all(r["arms"][arm][field] is not None for r in subset) else None
                                                for field in subset[0]["arms"][arm]}
                    summaries[scope][cell][condition] = summary
                if len(summaries[scope][cell]) < 3:
                    continue
                for arm in ("construct_witness", "heldout_witness"):
                    n = summaries[scope][cell]["native"]["arms"]
                    noninferior, conditions = n[arm]["nearest_mse_mm2"] <= n["KEEP"]["nearest_mse_mm2"], {}
                    for condition in ("minus3", "plus3"):
                        m = summaries[scope][cell][condition]["arms"]
                        gain, restore = m[arm]["improvement_percent"], m["restore"]["improvement_percent"]
                        retained = gain / restore if restore > 0 else None
                        conditions[condition] = {"restore_gain_percent": restore, "gain_percent": gain, "retained": retained,
                                                 "status": "PREMISE_ABSENT" if retained is None else "PASS" if retained >= .9 else "FAIL"}
                    acceptance[scope][cell][arm] = {"native_noninferior": bool(noninferior), "perturbed": conditions,
                                                  "joint_pass": bool(noninferior and all(v["status"] == "PASS" for v in conditions.values()))}
    compare(summaries, primary["summary"], "summary")
    compare(acceptance, primary["acceptance"], "acceptance")
    return {"gt_opened": True, "prediction_seal": seal, "primary_result_sha256": digest(primary_path),
            "independent_backends": {"PLY": "Open3D", "nearest": "sklearn.neighbors.KDTree", "brute_subset": "numpy Euclidean against all reference points"},
            "references": reference_report, "verified_records": len(rows), "verified_arm_MSEs": len(rows) * 4,
            "numeric_fields_compared": len(differences), "max_absolute_difference": max(differences, default=0.),
            "brute_query_count": len(brute_errors), "brute_max_absolute_difference_mm": max(brute_errors, default=0.),
            "requested_base_queries": 512, "valid_U1_base_queries": valid_base,
            "summary": summaries, "acceptance": acceptance,
            "interpretation": "numerical reproducibility of the frozen replay; existing same-family provisional check, not fresh semantic or external-calibration certification"}


def write_report(relative, report):
    target = (ROOT / relative).resolve()
    require(target.is_relative_to(ROOT) and target != ROOT, "report output escapes new run")
    require(target.suffix == ".json", "report must be a JSON file")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="read-only old seal and fixed-query checks; never read GT")
    parser.add_argument("--predictions", action="store_true", help="verify candidate identity, initializer support, witness rule, and seals; no GT")
    parser.add_argument("--numeric", action="store_true", help="after complete seal only: independently re-read reference PLY and reproduce every primary metric")
    parser.add_argument("--partial", action="store_true", help="report pending cases for a smoke check without claiming full completion")
    parser.add_argument("--output", help="new exclusive JSON report path relative to the new run")
    args = parser.parse_args()
    identity()
    install_gt_read_guard()
    require(sum((args.preflight, args.predictions, args.numeric)) == 1, "select exactly one check mode")
    require(not args.numeric or not args.partial, "numeric check cannot use partial predictions")
    result = old_context_preflight()
    if args.predictions:
        result = {"old_identity": result, **prediction_checks(partial=args.partial)}
    elif args.numeric:
        result = {"old_identity": result, **numeric_checks()}
    report = {**REVIEW, "host": socket.gethostname(), "root": str(ROOT),
              "check": "post_seal_numeric" if args.numeric else "pre_gt_predictions" if args.predictions else "pre_gt_query_context",
              "result": result, "checker_sha256": digest(__file__)}
    if args.output:
        write_report(args.output, report)
    if args.numeric:
        compact = {**REVIEW, "check": report["check"], **{key: result[key] for key in
                   ("gt_opened", "verified_records", "verified_arm_MSEs", "numeric_fields_compared", "max_absolute_difference",
                    "brute_query_count", "brute_max_absolute_difference_mm", "requested_base_queries", "valid_U1_base_queries")}}
    elif args.predictions:
        compact = {**REVIEW, "check": report["check"], "gt_opened": False,
                   "cases": result["cases"], "missing": result["missing"], "global_seal": result["global_seal"]}
    else:
        compact = {**REVIEW, "check": report["check"], "gt_opened": False,
                   "old_sealed_files_verified": result["old_sealed_files_verified"],
                   "base_query_count": result["base_query_count"],
                   "rois": {name: {k: v for k, v in row.items() if k not in {"pixel_ids", "fixed_query_uv"}}
                            for name, row in result["rois"].items()}}
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
