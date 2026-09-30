# Pre-reference integrity review

Target verified: host `liekkas`, absolute directory `/srv/slam-research/grf/map-denoise/runs/colmap-transfer-20260930T180000Z`. This is a fresh same-model-family, provisional semantic/source review, with separately identified deterministic checks. It is not cross-family review, a full Anti-Autoresearch audit, or a finding of CLEAN. The available integrity-forensics skill was inspected and found to specify a paper-specific pinned deterministic launcher; that workflow was not run or substituted with this review.

## Disposition

No material pre-evaluation defect was established in the inspected effective paths. Continue only under the registered narrow interpretation: two same-family DTU objects, two manually photo-selected ROIs per object, 512 fixed requested pixels, and conditioned on the supplied cameras. Final reference provenance, numerical scores, and conclusion support remain unreviewed here. No reference PLY was opened for this report.

One latent provenance weakness must be closed by the final independent audit: `fetch_reference.py:18-22` trusts an existing target PLY without independently checking that its bytes/CRC match the official member. The official ZIP directory is checked at lines 14-16, but the size assertion and ZIP CRC-checked stream are only on the fresh-download branch. The output manifest's SHA at line 22 then records whatever existing bytes were present. This is not evidence that the current data are wrong: the references directory was absent at the first inventory and fetching was in progress later. Final review should verify actual reference size and CRC against the pre-registered metadata, plus its seal binding. Do not silently change the frozen fetcher.

## A. Provenance and scene scope

- `SCENE_SELECTION.json:5-15` and `PLAN.json` contain precisely scans 118/122, reference 0022, and sources 0016/0035/0017/0007. Each of four ROIs contains 128 unique points. No extra scene or substituted result was found.
- The historical `/srv/slam-research/grf/map-denoise/datasets/external-confirmation-20260928T213020Z/SCENES.json:10-18,51-59` independently records 118 and 122 after the six selected scenes, with `RESERVE_PRIORITY_NOT_REACHED` status. `acquire_inputs.py:89-103` verifies the historical manifest/metadata chain, full archive hash, and reserve statuses before extraction.
- `acquire_inputs.py:24-27,150-174` extracts only ten images, two NPZ camera files, and four sparse camera/pose metadata binaries. It does not extract points3D. `INPUT_MANIFEST.json:106-111` records 44,041,722 extracted bytes and the original archive identity.
- The exposure search is bounded, not proof of universal novelty. `acquire_inputs.py:46-55` restricts roots, filename globs, and textual patterns; files over 16 MiB would be skipped at lines 59-61. `SCENE_SELECTION.json:40-57` discloses this scope and reports no skipped files. “Previously unrun” must retain the documented-local-history limitation.

## B. GT independence and leakage

- `prepare_transfer.py:62-98` derives the operational camera from supplied `world_mat`, raw PINHOLE intrinsics, and the matching pose metadata, with no GT file read. The inherited `camera_mapping.py:73-99` decodes 2D observations and track IDs, not point coordinates; only pose/camera fields are used by the transfer preparation.
- `cpu_baseline.py:65-101` computes the range from the five photos/SIFT matches, falling back to the camera-rig focus if fewer than ten triangulations survive. The inherited `rebuild.py:132-161` performs the registered reprojection/angle/mutual-ratio tests. `dense_rebuild.py:42-65` implements 5×5 plane sweep, per-source ZNCC, at least two finite source scores, and the top-two mean. Its import-level code does not open geometry.
- `mvs_transfer.py:81-105` writes separate photo and geo workspaces, explicitly empty `points3D.txt`, and five registered cameras. The geo arm computes five unfiltered photo maps, then the reference geometric map; it is not merely a renamed photometric output.
- `fetch_reference.py:6` requires the seal and verifies all sealed files before accessing the official ZIP. `evaluate_transfer.py:44-60` re-verifies the run/seal, loads independent reference vertices for nearest-neighbor scoring, and does not fit alignment or choose output points from GT.
- No GT-to-inference path was found. This is code and artifact evidence, not OS-level proof of every historical file access. Supplied calibration/normalization upstream provenance is not independently established by matching two representations of the same supplied rig.

## C. Pixel/camera conventions and effective implementation

- `prepare_transfer.py:81-89` converts raw corner-coordinate intrinsics to half-resolution integer-array coordinates; `test_transfer_preparation.py:23-31` checks self-consistent projection/backprojection. All 10 supplied pose comparisons and projection-example assertions passed in the saved checks; maximum center disagreement is approximately 0.000116 mm.
- A separate narrow source check found no supported additional half-pixel bug: official COLMAP 4.2.1 [MVS model loading](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/model.cc#L54) imports the calibration unchanged, and [ComputePointAtDepth](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/patch_match_cuda.cu#L198) uses integer col/row with `x=z*(col-cx)/fx`, `y=z*(row-cy)/fy`. The [+0.5 source texture lookup](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/patch_match_cuda.cu#L486) reaches CUDA texel centers. Thus `mvs_transfer.py:87-100,129-133` exports and backprojects the matching rays. This does not independently certify the physical camera calibration.
- All 14 job specifications have `max_image_size=-1`; no extra resize is specified. The installed target environment `/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421` imports `pycolmap` 4.2.1 with CUDA, distributed as `pycolmap-cuda12` 4.2.1. Observed `_core` SHA256: `affd3457a5ab80302e392a5fe23ee0d9d4b426ea68331cddd1138fb7584457ce`. This is a current identity observation, not pre-run binary attestation; RUN_LOCK pins experiment sources, not every library binary.

## D. Denominators and intended scores

- The primary fallback score uses the complete CPU-valid support: `evaluate_transfer.py:60` copies CPU distances and replaces only `valid & v`; its fallback mask remains `valid`. Missing MVS values are not scored as zero. The deterministic count is 484/512 CPU-valid points: 116, 119, 126, 123 by ROI.
- Standalone photo/geo `cpu_valid_common` rows instead use their respective intersections (`evaluate_transfer.py:74`); these are not the same primary population and must stay separate. `paired_cpu_mse_mm2` at line 78 supplies the matched CPU comparison for these rows.
- `summarize` at lines 38-42 averages the four ROI MSEs equally. With exactly two ROIs in each of two scenes this also gives equal scene weight. It is not a pooled 484-point MSE. Scene summaries at line 85 permit the registered both-scenes-improve criterion to be evaluated.
- Lines 61-67 count damage to originally ≤1 mm CPU points, rescue of originally >5 mm points, and new support separately. Improved/worsened/unchanged are intended to partition CPU-valid support. These arithmetic identities remain to be checked after evaluation.
- Nearest-neighbor distance to the reference point cloud is a one-way accuracy metric. This pipeline contains no official DTU observation-mask/full-scene completeness protocol; `refine-logs/EXPERIMENT_PLAN_20260930_180000.md:17` and evaluator line 91 correctly limit the claim.

## E. Deterministic pre-GT checks

Ran `python3 -B review/verify_pre_gt.py` successfully. The independent verifier imports no experiment module and never opens the references directory. It verifies 30 run files, 4 imported source dependencies, 21 plan sources, 60 MVS-preparation files, and all 106 prediction-seal entries; regenerates all fixed pixel lists; matches saved grayscale pixels; checks all 14 job-output hashes and depth counts; and reconstructs CPU/MVS coordinates from the stored depth samples independently.

| ROI | Requested | CPU valid | Photo valid / CPU overlap | Geo valid / CPU overlap | Geo fallback retained | Geo new support |
|---|---:|---:|---:|---:|---:|---:|
| scan118_upper_fold | 128 | 116 | 124 / 113 | 109 / 102 | 14 | 7 |
| scan118_base_ridge | 128 | 119 | 128 / 119 | 121 / 113 | 6 | 8 |
| scan122_feather | 128 | 126 | 128 / 126 | 123 / 122 | 4 | 1 |
| scan122_book_edge | 128 | 123 | 127 / 122 | 113 / 110 | 13 | 3 |

Real depth maps, normal maps, distinct output hashes, nonzero recorded execution times, and full job logs exist. `MVS_RUN.log:922-925` records final geometric output and sealing; `PREDICTIONS_SEALED.json:110-111` records the seal at `2026-09-30T18:09:07Z`. These observations rule out absent/phantom artifacts in the inspected run; they do not attest how an adversarial actor might have generated arbitrary bytes.

## F. Outstanding final audit

After reference acquisition/evaluation completes: independently verify reference CRC/size/provenance and seal timing; recompute point distances using an independent reader; recompute ROI/scene/equal-ROI summaries; verify fallback point identity and tail partitions; check that both scenes improve before claiming migration support; retain the originally-correct-point damage count even if aggregate MSE improves. No score or success conclusion is approved by this pre-GT report.
