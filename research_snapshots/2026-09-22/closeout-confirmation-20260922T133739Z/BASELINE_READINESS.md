# Official COLMAP baseline readiness

2026-09-22 13:44:28 UTC, exact host `liekkas`, exact workspace
`/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z`.

**Outcome: preflight completed; official reconstruction remains `NOT_RUN`.** No
COLMAP executable/build/package was found in the stated search scope. No scene
reconstruction, installation, data download, GPU allocation, process termination,
or environment modification was performed. This is not a failed quality result.
The map-denoise-research skill informed the separation of software readiness,
external reproduction, information fairness, and research evidence.

## Measured readiness

| Check | Observed result |
|---|---|
| Read-only executable preflight | `python3 -B baseline_probe.py`, exit 0 |
| COLMAP on PATH and standard bin locations | Not found |
| Research environment directories | 13 inspected by bounded path/name checks; no COLMAP/pycolmap package names found |
| `/opt`, named research tools/models and caches | No matching COLMAP/OpenMVS names found within recorded depth limits |
| Unreadable locations | `/opt/containerd`, `/opt/bytedance/feishu/vulcan`; not bypassed |
| Cached APT/PIP/UV package filenames | None found; opaque PIP cache objects were not decoded |
| APT metadata, no network update | `colmap 3.12.6-4`, not installed; package 3,566,916 bytes, not an installation/dependency total |
| GPU | RTX 3060 Ti, 8,192 MiB total, 6,701 MiB occupied, 4% utilization at snapshot; six compute PID records |
| Existing V28 image manifests | All 8 native ROI five-view lists and original-image hashes pass; corresponding ±3 mm cases use identical lists/hashes |
| Calibration assets | Existing `cameras.bin`, `images.bin`, `points3D.bin` found and hashed for scans 24/37; new five-view calibration adapter not yet implemented |
| Independent confirmation result | None |

The JSON records scan roots/depth/entry limits, permission denials, exact device
UUID, process IDs and command outputs. Absence is restricted to this search scope,
not a claim about every file on the machine. GPU occupancy is a point-in-time
observation; low utilization is not permission to use another task's allocation.

Run the audit again from this exact directory:

```bash
python3 -B baseline_probe.py
```

It only emits JSON to stdout, bounds directory traversal, checks host/cwd, hashes
existing development inputs, and calls read-only software/resource probes. Add
`--skip-development-inputs` for a software/resource-only check. The saved
`BASELINE_READINESS.json` is the completed audit snapshot, not a live status file.

## Proposed pin and backend

The proposed official upstream version is **COLMAP 4.2.0**, commit
`be5e29168d4aff238409d60424812df66aac919f`. `git ls-remote` resolved that exact tag
on this host. It is **not installed**. Record build flags, compiler/CUDA/library
versions, executable SHA-256 and command help before any run. An eventual isolated
build is a separate pending step; the APT candidate above is a different version
and must not be silently substituted. [Official 4.2.0 release](https://github.com/colmap/colmap/releases/tag/4.2.0),
[pinned commit](https://github.com/colmap/colmap/commit/be5e29168d4aff238409d60424812df66aac919f).

For this NVIDIA host, official PatchMatch requires a CUDA-enabled build and an
available GPU allocation. Version 4.2.0 also supports AMD HIP; saying that every
current COLMAP build is CUDA-only would be inaccurate. Its pinned source excludes
the dense estimator when neither CUDA nor HIP is compiled. CPU feature extraction,
matching and fusion do not create a CPU replacement for this dense estimator.
[Pinned MVS backend check](https://github.com/colmap/colmap/blob/be5e29168d4aff238409d60424812df66aac919f/src/colmap/exe/mvs.cc#L240),
[4.2.0 backend release notes](https://github.com/colmap/colmap/releases/tag/4.2.0).

## Comparison scope and input adapter checklist

The official native point cloud is a **system reference**. V28 receives a GeoSVR
geometry prior plus five photographs and calibration. Native COLMAP would receive
the same photographs/calibration but estimate geometry without that GeoSVR prior.
Equal photo count alone does not mean equal information or equal output support.
Report this difference alongside native completeness, surface error and cost.

One could separately predefine a fixed-input-point adapter that lifts official
depths onto the V28 initial point rows. Such a lift would be custom code, **not an
official COLMAP algorithm**. It is not implemented in this closeout preparation,
and cannot supply a result or stand in for official reproduction. If later used,
freeze its visibility, no-hit, fallback and depth-sampling rules before references
are opened; report its modified/support fractions and official native output too.

Required five-view adapter steps, still pending:

1. Read the frozen ordered view manifest (first reference, four sources), original
   image hashes and calibration. Preserve the same list for native/±3 mm arms.
   The eight old ROI lists in the JSON are exposed development evidence only.
2. Export only those five images and corresponding known cameras to a **new**
   baseline workspace. Use V28's half-size bilinear resize and pixel-center
   transform `u' = sx*(u+0.5)-0.5`, likewise for `v`; for the old DTU images this
   is 777×581. Do not feed 256×256 V26 crops as if they were V28 inputs. Validate
   ray/projection round trips and original-image dimensions before reconstruction.
3. Use known camera poses/calibration without a new pose/scale fit. If sparse
   tracks are needed, triangulate from these five photos with the known cameras
   fixed; do not retain tracks/geometry derived from all 49 photos and call it a
   five-photo reconstruction. This sparse-track preparation still needs an
   executable adapter and pinned CLI option verification.
4. Derive native-COLMAP depth limits from the five-photo sparse geometry/calibration
   under a fixed input-only rule, **not from the GeoSVR mesh or evaluator GT**.
   Write all five explicit reference/source pairs in `patch-match.cfg`; never
   permit automatic extra-view selection. The planned config text is in the JSON.
   Lock fusion inputs and all option defaults before outcomes.
5. Preserve the original COLMAP coordinate frame during dense estimation. Apply
   the frozen input calibration `scale_mat` once when mapping results to physical
   millimetres. Verify camera/world conventions independently of GT. Use the same
   predefined ROI for evaluation and retain raw output plus empty/failure rows.
6. For future scenes, validate lens models, rectification and metric transforms
   first. The old DTU checklist is not proof that an Oxford fish-eye/LiDAR adapter
   is correct. If adaptation changes code, that scene is development.

Known poses can initialize dense COLMAP without rerunning full SfM. An empty
track model requires explicit neighbors/depth bounds and has a special fusion
setting in the official FAQ. We have not selected that weaker fallback or changed
`min_num_pixels` after seeing outputs. The intended track route above remains
pending. [Official known-pose workflow](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses),
[pinned explicit source-list format](https://github.com/colmap/colmap/blob/be5e29168d4aff238409d60424812df66aac919f/src/colmap/mvs/patch_match.h#L91).

## Stage commands and stopping conditions

`planned_stages` in the JSON contains the exact argument-array templates for
`image_undistorter`, `patch_match_stereo` and `stereo_fusion`. Template tokens for
the installed binary, locked case, image dimensions and depth bounds are
intentionally unresolved; these are **not an executable reconstruction pipeline
yet**. The preflight itself is executable and has run successfully.

| Stage | Status | Remaining prerequisite |
|---|---|---|
| Five-view images/camera/track adapter | `NOT_IMPLEMENTED` | Fixed calibrated export and projection checks |
| Official undistortion | `NOT_RUN` | Pinned binary and prepared five-view inputs |
| Explicit source/fusion config lock | `NOT_RUN` | Verified prepared workspace and final dimensions |
| Official PatchMatch depth/normal estimation | `NOT_RUN` | CUDA build and resource allocation plus locked depth interval |
| Official fusion | `NOT_RUN` | Complete geometric depth maps and fixed fusion policy |
| Coordinate/evaluation adapter | `NOT_IMPLEMENTED` | Physical-unit/support checks and geometry sealing |

Pinned PatchMatch defaults remain fixed except recorded resource settings and
matched image dimensions; proposed limits are one PatchMatch worker and 2 GB
cache, with four fusion CPU threads and 2 GB cache. No extra parameter search is
authorized by this plan. Measure full-pipeline time, process-tree RAM and GPU
memory with their actual scope when it eventually runs; do not compare a GPU
kernel time against V28's complete CPU time. Retain `NOT_RUN`, software error,
empty output and completed result as distinct states. [Official stage sequence](https://colmap.github.io/cli.html),
[pinned PatchMatch options](https://github.com/colmap/colmap/blob/be5e29168d4aff238409d60424812df66aac919f/src/colmap/mvs/patch_match_options.h).
