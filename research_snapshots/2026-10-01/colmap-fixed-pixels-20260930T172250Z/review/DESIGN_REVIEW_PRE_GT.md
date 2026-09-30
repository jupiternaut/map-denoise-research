# Fixed-pixel COLMAP design review (pre-GT)

Target: `liekkas:/srv/slam-research/grf/map-denoise/runs/colmap-fixed-pixels-20260930T172250Z`.

Reviewer role: existing reviewer, not fresh; `review_independence=same-family`; `acceptance_status=provisional`.

This review reads plans, camera metadata, fixed-query identity, old construction code and upstream COLMAP 4.2.1 source. It does not read laser GT, old error/result tables, or produce performance scores. No author code was modified and no MVS experiment was run by this reviewer. Review snapshot: initial `EXPERIMENT_PLAN.md` and `AGENTS.md`; implementation and addendum were not yet present.

## Verdict: WARN — contract clarification required before inference

The fixed-query, photo-set, no-GT and missing-value design is appropriate for a development comparison. The initial plan's blanket array-to-COLMAP `+0.5` assertion is insufficient for the *dense* 4.2.1 implementation. The sparse-camera convention and the dense kernel's actual ray equation must be distinguished in a pre-GT addendum. No correctness or performance acceptance is implied by this design review.

## Findings

### 1. Same grayscale images: PASS design; runtime identity pending

- New `refine-logs/EXPERIMENT_PLAN.md:14` requires exact old half-gray bytes, at 777 by 581, for reference 0022 and Q 0016/0035/0017/0007.
- Old `/srv/slam-research/grf/map-denoise/runs/upstream-photo-holdout-20260930T113213Z/rebuild.py:147` implements RGB conversion, RGB BILINEAR resize, then float dot product `[.299,.587,.114]`, clipping, and uint8 truncation. Converting to PIL `L` before resizing or using another gray conversion is not an exact substitute.
- Require decoded PNG array equality for all ten images, not only matching dimensions. Ensure COLMAP does not downsize/undistort/resample these already prepared images. Record input and decoded-byte hashes, shape and camera association.

### 2. Dense versus sparse pixel coordinates: FAIL if the initial plan is implemented literally

Local premise: new `refine-logs/EXPERIMENT_PLAN.md:15` specifies half-resolution COLMAP principal point `(388.5,290.5)`. Old `CAMERA_MAPPING.json:70` defines integer array centers; `:90` defines half-scale mapping. Its scan24 reference `K_raw_colmap` is at `:217`, and `K_full_array` at `:234`. Thus the corrected half-array principal point is `(388,290)`.

The [official camera-conversion documentation](https://colmap.github.io/faq.html#using-calibration-from-opencv-kalibr-or-other-tools) supports `+0.5` for standard COLMAP camera coordinates. However, direct review of the pinned dense implementation gives a different operational requirement:

- [camera.cc, lines 63–71](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/scene/camera.cc#L63): `CalibrationMatrix()` returns the stored principal point unchanged.
- [mvs/model.cc, lines 66–79](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/model.cc#L66) and [mvs/image.cc, lines 45–56](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/image.cc#L45): pass/copy that matrix without a half-pixel correction.
- [patch_match_cuda.cu, lines 1712–1738](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/patch_match_cuda.cu#L1712): extract the unchanged K and inverse K. Its [lines 198–205](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/patch_match_cuda.cu#L198) use integer column/row for the optical-Z ray. [Lines 503–528](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/patch_match_cuda.cu#L503) add 0.5 only after homography projection, for CUDA texture addressing.
- [fusion.cc, lines 452–455](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/fusion.cc#L452) likewise unprojects integer column/row multiplied by depth.

Inference from this complete source path: a direct dense workspace using byte-identical half images needs the half-array K to preserve the old physical rays. Feeding corner K to this dense path shifts the implied rays by half a pixel. Adding 0.5 only at output unprojection does not repair the rays used during optimization.

Minimal fix: freeze a clearly named 4.2.1 dense-array adapter (K_array and integer D[v,u]) before inference, while recording standard corner K separately for sparse-camera projection checks. Do not modify the official library or shift/resample the images. The adapter should be verified against the installed version and tested by independent world-to-image/ray equations, not merely a round trip through the same adapter. A pure projection self-consistency test still does not establish correct external calibration.

### 3. Same reference plus four Q photos: PASS design

New `AGENTS.md:5` and `:8`, and `refine-logs/EXPERIMENT_PLAN.md:14`, fix the five photos and exclude C/H. Old `corrected_rebuild.py:25` and `:29` show the same set used by CPU construction. Explicit patch-match source lists should be sealed; do not use automatic neighbor selection or substitute other photos. Official [patch_match.h, lines 109–125](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/patch_match.h#L109) defines the two-line reference/source format. Empty points3D is appropriate here; automatic source/range selection cannot be inferred from an empty sparse model.

### 4. Q-only depth range: WARN — specify actual interval and auxiliary differences

Old `corrected_rebuild.py:31` through `:47` uses reference-to-Q SIFT, valid triangulation, 5th/95th percentiles plus/minus 20 mm, then `arange(floor(low),ceil(high)+.1,1)`.

- `initialization/scan24/scan24_window_left/REBUILD.json:10`: nominal `[559.6343720497265,785.0107480782123]` mm; actual CPU grid endpoints `[559,786]` mm.
- `initialization/scan37/scan37_scissor_cross/REBUILD.json:10`: nominal `[581.0229425288945,864.4923335033219]` mm; actual CPU grid endpoints `[581,865]` mm.

Use the actual old scan endpoints for the primary ref-only photometric arm, or explicitly qualify a nominal-interval comparison. COLMAP continuous hypotheses and CPU discretization remain intended algorithm differences, not identical search procedures.

Geometric consistency needs source depth/normal maps ([patch_match_options.h, lines 111–116](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/patch_match_options.h#L111)). Source cameras have different optical-Z coordinates: repeating the reference interval blindly is not equivalent support. If a range expansion is required, derive it solely from frozen source-only geometry/camera frusta and declare the exact deterministic rule and values before outputs. Keep geometric results auxiliary; they do not isolate the geometric term if support/initialization differs. Use a separate workspace so its photo prepass cannot replace primary filtered photometric outputs.

### 5. Fixed queries and filtered missing outputs: PASS design; execution pending

New `AGENTS.md:9` and `refine-logs/EXPERIMENT_PLAN.md:13`, `:27`, `:29` retain all 512 original query identities. The 15 old CPU failures are not silently removed. COLMAP's filter assigns zero to rejected depths ([patch_match_cuda.cu, lines 1267–1271](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/patch_match_cuda.cu#L1267)).

Record native zero/nonfinite missing at the same pixel; no nearest-valid substitution, depth interpolation, GT fill or new query selection. Unproject positive finite optical Z at the exact integer array index under the frozen dense contract. Coverage has denominator 512. Conditional native error and common-valid paired error are descriptive secondary quantities, not replacements for this denominator. Compare deployment-style COLMAP-else-CPU fallback on the original 497 CPU-valid identities, with four-ROI equal weights fixed in advance; newly covered 15 queries are separate. Report valid-and-correct count/512, original-correct harm and missing counts per ROI. Oracle candidate-pool expansion is only a diagnostic upper bound.

### 6. Repeats and caching: WARN — no independent CUDA seeds are exposed here

New `refine-logs/EXPERIMENT_PLAN.md:37` permits honest repeats if the CUDA seed is not configurable. [gpu_mat_prng.cu, lines 36–46](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/gpu_mat_prng.cu#L36) initializes from pixel-thread identity; PatchMatchOptions has no random-seed field. Python seeding therefore is not evidence of independent CUDA seeds. Use the term repeated executions and report byte identity if outputs repeat. Use separate workspaces: [patch_match.cc, lines 410–413](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/mvs/patch_match.cc#L410) skips already existing output files. Seal all executed repeats and report each or an equal-weight predeclared aggregate; never choose the best run using GT.

### 7. No difficulty-based tuning and claim scope: PASS design, provisional only

New `AGENTS.md:10` and `refine-logs/EXPERIMENT_PLAN.md:27` reserve the historical 75 all-candidates-over-5-mm cases for post-seal diagnosis. Do not read that membership to select pixels, views, ranges, filter settings, repeat counts or stopping rules. The full 512-query manifest is the construction population. New `refine-logs/EXPERIMENT_PLAN.md:9` correctly labels both scenes as exposed development scenes. Results cannot establish generalization, official DTU benchmark ranking, external calibration correctness, or a deployable GT-oracle selection method.

## Minimum pre-inference lock checklist

1. Resolve dense-array versus sparse-corner convention in ADDENDUM; test actual dense ray equation against old P, separately from sparse projection.
2. Freeze exact ten image hashes/decoded gray bytes, five-view associations, units, R/C, complete 512 identity manifest and actual primary intervals.
3. Freeze primary filtered-photo options, auxiliary source-depth range rule/options, missing/fallback policy, all repeat workspaces and aggregation.
4. Save installed wheel/version provenance and actual option dump. Run pre-GT camera/image/ray checks only; do not consult GT scores.
5. Seal code/config/input identities and every intended prediction before opening GT. Any later scoring uses a separately frozen evaluation protocol.

This report is a pre-inference design review, not a fresh post-experiment A–F integrity acceptance or a numerical-result audit.
