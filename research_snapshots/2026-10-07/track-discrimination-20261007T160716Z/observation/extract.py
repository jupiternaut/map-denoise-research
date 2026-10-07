#!/usr/bin/env python3
"""Candidate-blind finite, raw-image three-view correspondence extraction."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
READ_LOG = []
DENIED_LOG = []


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(name, obj):
    with open(HERE / name, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def install_read_guard(photos):
    """Python audit-hook allowlist; intentionally not an OS security sandbox."""
    allowed = {str(p.resolve()) for p in [ROOT / "REQUESTS.json", ROOT / "PROTOCOL.json",
               HERE / "CONFIG.json", HERE / "LOCK.json", Path(__file__), *photos]}
    outputs = {str(HERE / n) for n in ("TRACKS.json", "SCANS.json", "SELFTEST.json", "SEAL.json")}

    def guard(event, args):
        if event == "open":
            raw, mode, flags = args
            if isinstance(raw, int):
                return
            path = os.path.realpath(os.fsdecode(raw))
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
            if (writing and path not in outputs) or (not writing and path not in allowed | outputs):
                DENIED_LOG.append({"event": event, "path": path, "writing": writing})
                raise PermissionError("candidate-blind file allowlist denied: " + path)
            READ_LOG.append({"path": path, "writing": writing})
        elif event in ("socket.connect", "subprocess.Popen", "os.system", "os.listdir", "os.scandir"):
            DENIED_LOG.append({"event": event})
            raise PermissionError("candidate-blind execution disallows " + event)
    sys.addaudithook(guard)


def camera(data):
    K = np.asarray(data["K_half"], float)
    R = np.asarray(data["R"], float)
    C = np.asarray(data["center"], float)
    return {"K": K, "R": R, "C": C, "P": K @ np.column_stack((R, -R @ C))}


def ray(cam, xy):
    return cam["R"].T @ np.linalg.solve(cam["K"], [*xy, 1.0])


def project(cam, points):
    points = np.atleast_2d(points)
    xyz = (points - cam["C"]) @ cam["R"].T
    uvw = xyz @ cam["K"].T
    uv = np.divide(uvw[:, :2], uvw[:, 2:], out=np.full((len(points), 2), np.nan),
                   where=np.abs(uvw[:, 2:]) > 1e-12)
    return uv, xyz[:, 2]


def patches(img, centers, radius):
    centers = np.atleast_2d(centers)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dx, dy = xx.ravel(), yy.ravel()
    valid = np.isfinite(centers).all(axis=1)
    valid &= (centers[:, 0] >= radius) & (centers[:, 0] <= img.shape[1] - 1 - radius)
    valid &= (centers[:, 1] >= radius) & (centers[:, 1] <= img.shape[0] - 1 - radius)
    p = np.zeros((len(centers), len(dx)), float)
    if valid.any():
        x = centers[valid, 0, None] + dx
        y = centers[valid, 1, None] + dy
        p[valid] = map_coordinates(img, [y, x], order=1, mode="constant", cval=0.0)
    return p, valid


def ncc_curve(anchor_image, anchor_xy, target_image, target_xy, positive, obs):
    a, av = patches(anchor_image, anchor_xy, obs["patch_radius_px"])
    b, bv = patches(target_image, target_xy, obs["patch_radius_px"])
    a = a[0] - np.mean(a[0])
    b -= b.mean(axis=1, keepdims=True)
    std_a = float(np.sqrt(np.mean(a * a)))
    std_b = np.sqrt(np.mean(b * b, axis=1))
    valid = bv & positive & bool(av[0]) & (std_a >= obs["min_patch_std_255"])
    valid &= std_b >= obs["min_patch_std_255"]
    scores = np.full(len(target_xy), np.nan)
    scores[valid] = (b[valid] @ a) / (len(a) * std_a * std_b[valid])
    return scores, {"anchor_std_255": std_a, "valid_samples": int(valid.sum()),
                    "invalid_samples": int((~valid).sum())}


def runs(mask):
    loc = np.flatnonzero(mask)
    if not len(loc):
        return []
    cuts = np.flatnonzero(np.diff(loc) > 1) + 1
    return [(int(x[0]), int(x[-1])) for x in np.split(loc, cuts)]


def merge(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if hi < lo:
            continue
        if out and lo <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], float(hi))
        else:
            out.append([float(lo), float(hi)])
    return out


def intersection(a, b):
    return merge([[max(x[0], y[0]), min(x[1], y[1])] for x in a for y in b
                  if max(x[0], y[0]) <= min(x[1], y[1])])


def subtract(a, b):
    out = []
    for lo, hi in merge(a):
        pieces = [[lo, hi]]
        for blo, bhi in merge(b):
            updated = []
            for plo, phi in pieces:
                if bhi <= plo or blo >= phi:
                    updated.append([plo, phi])
                else:
                    if plo < blo:
                        updated.append([plo, min(phi, blo)])
                    if bhi < phi:
                        updated.append([max(plo, bhi), phi])
            pieces = updated
        out.extend(pieces)
    return out


def mask_intervals(mask, grid, padding):
    half = (grid[1] - grid[0]) / 2.0 if len(grid) > 1 else 0.0
    return [[max(float(grid[0]), float(grid[i]) - half - padding),
             min(float(grid[-1]), float(grid[j]) + half + padding)] for i, j in runs(mask)]


def local_peaks(scores, start, end):
    """All local maxima, with each exactly flat plateau represented at its midpoint."""
    out = []
    i = start
    while i <= end:
        j = i
        while j < end and scores[j + 1] == scores[i]:
            j += 1
        left = scores[i - 1] if i > start else -np.inf
        right = scores[j + 1] if j < end else -np.inf
        if scores[i] >= left and scores[i] >= right:
            out.append((i + j) // 2)
        i = j + 1
    return out


def summarize_scan(grid, uv, scores, threshold, obs, prefix, diag):
    mask = np.isfinite(scores) & (scores >= threshold)
    components, nodes = [], []
    all_intervals = mask_intervals(mask, grid, obs["interval_padding_mm"])
    for ci, ((i, j), interval) in enumerate(zip(runs(mask), all_intervals)):
        peaks = local_peaks(scores, i, j)
        components.append({"id": ci, "sample_index_range": [i, j],
                           "raw_depth_range_mm": [float(grid[i]), float(grid[j])],
                           "padded_interval_mm": interval, "peak_sample_indices": peaks})
        for pi in peaks:
            nodes.append({"id": f"{prefix}/c{ci}/k{pi}", "component": ci, "sample_index": pi,
                          "pixel_xy": uv[pi].tolist(), "ncc": float(scores[pi]),
                          "scan_reference_depth_mm": float(grid[pi]), "component_interval_mm": interval})
    return {"components": components, "nodes": nodes, "intervals": merge(all_intervals),
            "diagnostics": {**diag, "threshold": threshold, "accepted_samples": int(mask.sum()),
                            "component_count": len(components), "peak_count": len(nodes),
                            "all_threshold_components_retained": True, "top_k_limit": None}}, mask


def scan(anchor_image, anchor_xy, target_image, target_cam, points, grid, positive, obs, threshold, prefix):
    uv, target_z = project(target_cam, points)
    scores, diag = ncc_curve(anchor_image, anchor_xy, target_image, uv, positive & (target_z > 0), obs)
    result, mask = summarize_scan(grid, uv, scores, threshold, obs, prefix, diag)
    raw = {"id": prefix, "reference_depth_grid_mm": grid.tolist(),
           "target_pixels_xy": [[float(x), float(y)] if np.isfinite([x, y]).all() else None for x, y in uv],
           "ncc": [float(v) if np.isfinite(v) else None for v in scores],
           "accepted_sample_indices": np.flatnonzero(mask).tolist()}
    return result, raw, uv


def triangulate(cams, pixels, ref):
    A = np.stack([row for cam, (u, v) in zip(cams, pixels)
                  for row in (u * cam["P"][2] - cam["P"][0], v * cam["P"][2] - cam["P"][1])])
    point, _, rank, _ = np.linalg.lstsq(A[:, :3], -A[:, 3], rcond=None)
    projections = [project(cam, point) for cam in cams]
    errors = [float(np.linalg.norm(p[0][0] - xy)) for p, xy in zip(projections, pixels)]
    depth = float((ref["R"] @ (point - ref["C"]))[2])
    return {"world_point_mm": point.tolist(), "reference_depth_mm": depth,
            "reprojection_errors_px": errors, "max_reprojection_error_px": max(errors),
            "all_positive_depth": bool(all(p[1][0] > 0 for p in projections)), "rank": int(rank)}


def geometry_ok(t, domain, obs):
    return (t["rank"] == 3 and t["all_positive_depth"] and domain[0] <= t["reference_depth_mm"] <= domain[1]
            and t["max_reprojection_error_px"] <= obs["reprojection_tolerance_px"])


def node_ray_support(node, ref_ray_projections, grid, obs):
    delta = np.linalg.norm(ref_ray_projections - node["pixel_xy"], axis=1)
    pix_intervals = mask_intervals(delta <= obs["reprojection_tolerance_px"], grid,
                                  obs["interval_padding_mm"])
    return intersection([node["component_interval_mm"]], pix_intervals)


def extract_row(request, scene, cams, images, obs):
    ref_name = scene["reference"]
    s1_name, s2_name = scene["sources"]
    ref, s1, s2 = [cams[n] for n in [ref_name, s1_name, s2_name]]
    xy = request["pixel_xy"]
    lo, hi = scene["depth_range_mm"]
    grid = np.arange(lo, hi + obs["depth_step_mm"] * 0.01, obs["depth_step_mm"])
    if grid[-1] < hi:
        grid = np.append(grid, hi)
    points = ref["C"] + grid[:, None] * ray(ref, xy)
    scans, raws, uv = [], [], []
    for name, cam in [(s1_name, s1), (s2_name, s2)]:
        summ, raw, proj = scan(images[ref_name], xy, images[name], cam, points, grid,
                               np.ones(len(grid), bool), obs, obs["reference_ncc_min"], "ref-to-" + name)
        for node in summ["nodes"]:
            node["ray_support_intervals_mm"] = node_ray_support(node, proj, grid, obs)
        scans.append(summ)
        raws.append(raw)
        uv.append(proj)
    full_star = intersection(scans[0]["intervals"], scans[1]["intervals"])
    star_tracks, star_rejections = [], {"no_common_interval": 0, "geometry": 0}
    for n1 in scans[0]["nodes"]:
        for n2 in scans[1]["nodes"]:
            support = intersection(n1["ray_support_intervals_mm"], n2["ray_support_intervals_mm"])
            if not support:
                star_rejections["no_common_interval"] += 1
                continue
            tri = triangulate([ref, s1, s2], [xy, n1["pixel_xy"], n2["pixel_xy"]], ref)
            if not geometry_ok(tri, [lo, hi], obs):
                star_rejections["geometry"] += 1
                continue
            star_tracks.append({"id": len(star_tracks), "source1_node": n1["id"], "source2_node": n2["id"],
                                "pixels_xy": [xy, n1["pixel_xy"], n2["pixel_xy"]],
                                "ncc_reference_sources": [n1["ncc"], n2["ncc"]],
                                "intervals_mm": support, "triangulation": tri,
                                "provenance": "independently measured reference-source local maxima"})
    # Fixed measured source1 pixels are anchors: the source-source curve is not
    # obtained by moving either endpoint with any tested repair candidate.
    direct, cycle_tracks = [], []
    cycle_rejections = {"pixel_incompatible": 0, "no_common_interval": 0, "geometry": 0}
    for n1 in scans[0]["nodes"]:
        d1 = ray(s1, n1["pixel_xy"])
        a = float(ref["R"][2] @ (s1["C"] - ref["C"]))
        b = float(ref["R"][2] @ d1)
        if abs(b) < 1e-12:
            direct.append({"anchor_source1_node": n1["id"], "status": "reference-depth parameterization degenerate"})
            continue
        t = (grid - a) / b
        direct_points = s1["C"] + t[:, None] * d1
        summ, raw, _ = scan(images[s1_name], n1["pixel_xy"], images[s2_name], s2, direct_points, grid,
                             t > 0, obs, obs["source_ncc_min"], "direct-" + n1["id"])
        summ["anchor_source1_node"] = n1["id"]
        summ["anchor_pixel_xy"] = n1["pixel_xy"]
        summ["provenance"] = "source1 measured patch directly matched over full public reference-depth domain"
        direct.append(summ)
        raws.append(raw)
        for st in (s for s in star_tracks if s["source1_node"] == n1["id"]):
            for nd in summ["nodes"]:
                dist = float(np.linalg.norm(np.asarray(st["pixels_xy"][2]) - nd["pixel_xy"]))
                if dist > obs["match_pixel_tolerance"]:
                    cycle_rejections["pixel_incompatible"] += 1
                    continue
                # An added direct measurement narrows the same reference-ray
                # support. Its component uses reference-camera depth coordinates.
                support = intersection(st["intervals_mm"], node_ray_support(nd, uv[1], grid, obs))
                if not support:
                    cycle_rejections["no_common_interval"] += 1
                    continue
                tri = triangulate([ref, s1, s2, s2], [xy, n1["pixel_xy"], st["pixels_xy"][2], nd["pixel_xy"]], ref)
                if not geometry_ok(tri, [lo, hi], obs):
                    cycle_rejections["geometry"] += 1
                    continue
                cycle_tracks.append({"id": len(cycle_tracks), "parent_star_track": st["id"],
                                     "source1_node": n1["id"], "source2_reference_node": st["source2_node"],
                                     "source2_direct_node": nd["id"],
                                     "pixels_xy": [xy, n1["pixel_xy"], st["pixels_xy"][2], nd["pixel_xy"]],
                                     "ncc_reference_sources": st["ncc_reference_sources"], "ncc_source_source": nd["ncc"],
                                     "source2_pixel_disagreement": dist, "intervals_mm": support,
                                     "triangulation": tri, "provenance": "star plus direct source1-source2 image measurement"})
    star_intervals = merge([i for st in star_tracks for i in st["intervals_mm"]])
    cycle_intervals = merge([i for ct in cycle_tracks for i in ct["intervals_mm"]])
    assert not subtract(cycle_intervals, star_intervals), "cycle must be nested in star"
    return {**request, "source_pair": [s1_name, s2_name], "reference": ref_name,
            "depth_domain_mm": [lo, hi], "star_intervals": star_intervals, "cycle_intervals": cycle_intervals,
            "star_full_intervals": full_star, "independent_union_intervals": merge(scans[0]["intervals"] + scans[1]["intervals"]),
            "reference_source_scans": scans, "direct_source_source_scans": direct,
            "star_tracks": star_tracks, "cycle_tracks": cycle_tracks,
            "diagnostics": {"star_status": "nonempty" if star_intervals else "empty_unknown",
                            "cycle_status": "nonempty" if cycle_intervals else "empty_unknown",
                            "star_track_count": len(star_tracks), "cycle_track_count": len(cycle_tracks),
                            "star_rejections": star_rejections, "cycle_rejections": cycle_rejections,
                            "full_star_intervals_not_represented_by_accepted_nodes": subtract(full_star, star_intervals),
                            "finite_search": True, "confidence_set": False}}, raws


def selftests(obs):
    checks = []
    K = [[500., 0., 64.], [0., 500., 64.], [0., 0., 1.]]
    cams = {n: camera({"K_half": K, "R": np.eye(3).tolist(), "center": [x, 0., 0.]})
            for n, x in [("ref", 0.), ("s1", 20.), ("s2", -20.)]}
    X = np.array([0., 0., 1000.])
    pix = [project(cams[n], X)[0][0] for n in ["ref", "s1", "s2"]]
    tri = triangulate(list(cams.values()), pix, cams["ref"])
    assert np.linalg.norm(np.asarray(tri["world_point_mm"]) - X) < 1e-7
    checks.append({"name": "synthetic_calibrated_three_camera_triangulation", "passed": True, "triangulation": tri})
    ncc = np.array([0., .7, .9, .8, .7, 0., .8, .9, .9, 0.])
    summ, _ = summarize_scan(np.arange(10.), np.zeros((10, 2)), ncc, .6, obs, "test", {})
    assert len(summ["components"]) == 2 and len(summ["nodes"]) == 2
    assert summ["nodes"][1]["sample_index"] == 7
    checks.append({"name": "all_ambiguous_components_and_plateau_representatives_retained", "passed": True})
    rng = np.random.default_rng(62341)
    refimg = rng.uniform(0., 255., (128, 128))
    ims = {"ref": refimg, "s1": np.roll(refimg, -10, axis=1), "s2": np.roll(refimg, 10, axis=1)}
    scene = {"reference": "ref", "sources": ["s1", "s2"], "depth_range_mm": [900., 1100.]}
    row, _ = extract_row({"scene": -1, "roi": "synthetic_plane", "query": 0, "pixel_xy": [64., 64.]}, scene, cams, ims, obs)
    assert any(lo <= 1000. <= hi for lo, hi in row["star_intervals"])
    assert any(lo <= 1000. <= hi for lo, hi in row["cycle_intervals"])
    checks.append({"name": "raw_image_plane_direct_cycle_contains_known_depth", "passed": True,
                   "star_intervals": row["star_intervals"], "cycle_intervals": row["cycle_intervals"]})
    denied = []
    for name in ["candidates.json", "source_depth.npy", "evaluation.json", "EVIDENCE.json"]:
        try:
            with open(ROOT / name, "rb"):
                pass
        except PermissionError:
            denied.append(name)
    assert len(denied) == 4
    checks.append({"name": "forbidden_reads_rejected_before_file_access", "passed": True, "denied": denied})
    bad = dict({"scene": 118, "roi": "x", "query": 0, "pixel_xy": [1, 2]}, candidate_depth=1)
    try:
        validate_request_rows([bad])
    except ValueError:
        checks.append({"name": "candidate_fields_rejected_by_request_schema", "passed": True})
    else:
        raise AssertionError("candidate field accepted")
    return {"checks": checks, "all_passed": True, "scope": "synthetic geometry and candidate-blind Python interface; not OS sandbox"}


def validate_request_rows(rows):
    for row in rows:
        if set(row) != {"scene", "roi", "query", "pixel_xy"}:
            raise ValueError("request row fields violate candidate-blind interface")


def main():
    start = time.monotonic()
    config = json.loads((HERE / "CONFIG.json").read_text())
    lock = json.loads((HERE / "LOCK.json").read_text())
    for rel, expected in lock["sha256"].items():
        assert digest(ROOT / rel) == expected, "locked input/source changed: " + rel
    request = json.loads((ROOT / "REQUESTS.json").read_text())
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    obs = protocol["observation"]
    validate_request_rows(request["rows"])
    expected_cam_keys = {"K_half", "R", "center", "image_path", "width", "height", "image_sha256"}
    for scene in request["scenes"].values():
        assert len(scene["sources"]) == 2
        for cam in scene["cameras"].values():
            assert set(cam) == expected_cam_keys, "candidate-blind camera schema mismatch"
    photos = [Path(c["image_path"]) for s in request["scenes"].values() for c in s["cameras"].values()]
    Image.init()  # Complete plugin imports before enforcing extraction IO allowlist.
    install_read_guard(photos)
    tests = selftests(obs)
    write_json("SELFTEST.json", tests)
    if "--selftest-only" in sys.argv:
        print(json.dumps(tests), flush=True)
        return
    images, cams, photo_hashes = {}, {}, {}
    for sn, scene in request["scenes"].items():
        images[sn], cams[sn] = {}, {}
        for name, data in scene["cameras"].items():
            path = data["image_path"]
            photo_hashes[path] = digest(path)
            assert photo_hashes[path] == data["image_sha256"], "original image hash mismatch"
            with Image.open(path) as im:
                images[sn][name] = np.asarray(im.convert("L").resize((data["width"], data["height"]),
                                                    resample=Image.Resampling.BILINEAR), dtype=float)
            cams[sn][name] = camera(data)
    rows, raw_rows = [], []
    for req in request["rows"]:
        sn = str(req["scene"])
        row, raw = extract_row(req, request["scenes"][sn], cams[sn], images[sn], obs)
        rows.append(row)
        raw_rows.append({"scene": req["scene"], "roi": req["roi"], "query": req["query"], "scans": raw})
        print(json.dumps({"scene": req["scene"], "query": req["query"], "star": row["star_intervals"],
                          "cycle": row["cycle_intervals"], "elapsed_s": round(time.monotonic() - start, 3)}), flush=True)
    result = {"schema_version": 1, "config": config, "lock_sha256": digest(HERE / "LOCK.json"),
              "candidate_access": False, "source_depth_access": False, "evaluation_access": False,
              "rows": rows, "wallclock_seconds_before_serialization": time.monotonic() - start}
    write_json("TRACKS.json", result)
    write_json("SCANS.json", {"schema_version": 1, "rows": raw_rows})
    seal = {"created_at": datetime.now(timezone.utc).isoformat(), "wallclock_seconds": time.monotonic() - start,
            "sha256": {name: digest(HERE / name) for name in ["extract.py", "CONFIG.json", "LOCK.json", "SELFTEST.json", "TRACKS.json", "SCANS.json"]},
            "request_sha256": digest(ROOT / "REQUESTS.json"), "protocol_sha256": digest(ROOT / "PROTOCOL.json"),
            "input_photo_sha256": photo_hashes, "read_write_audit": READ_LOG, "expected_negative_tests_denied": DENIED_LOG,
            "isolation": "Python audit-hook open allowlist and strict schema; not OS sandbox or proof against hostile native code",
            "evaluation_seen": False, "candidate_coordinates_seen": False}
    write_json("SEAL.json", seal)
    print(json.dumps({"sealed": True, "rows": len(rows), "wallclock_seconds": seal["wallclock_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
