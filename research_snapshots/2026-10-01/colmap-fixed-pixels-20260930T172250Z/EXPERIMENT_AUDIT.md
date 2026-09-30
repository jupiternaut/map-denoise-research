# Fresh integrity review

Verdict: **WARN**. No failed numeric, fixed-query, pixel-coordinate, or execution check was found. The original CSV reporting issue is corrected in a separately versioned derivative; an evaluation-dependency provenance limitation remains. The geometric fallback satisfies the stated development-scene acceptance condition; this is not confirmation on unseen scenes.

- `review_independence=same-family`
- `acceptance_status=provisional`
- Host verified directly: `liekkas`.
- Target: `/srv/slam-research/grf/map-denoise/runs/colmap-fixed-pixels-20260930T172250Z` (realpath verified).
- Reviewer operated read-only on inputs, code, predictions, and evaluation outputs. Only this review and its JSON companion were written. No MVS rerun or evaluator-output rewrite was performed.

## Findings

1. **RESOLVED IN DERIVATIVE — auxiliary CSV fields did not follow the row's common-point scope.** `evaluate_mvs.py:68` defines `mask=v&valid` for `cpu_valid_common`, and the error/count fields correctly use that mask. However, `evaluate_mvs.py:70` computes `coverage` and `correct_per_requested` from `v` instead. Eight original rows are affected: photo, geo, and geo_unfiltered for both scan24 ROIs, and photo plus geo_unfiltered for scan37_clamp_jaw. For example, `evaluation/ROI_METRICS.csv:11` has `n=111` and `correct_1mm=99` but coverage `119/128` and correct_per_requested `106/128`. Headline common-point MSE and common_n are correct (`evaluate_mvs.py:91`) and independently reproduce. Subsequently inspected `normalize_scopes.py:8` correctly selects the scoped point group, `:11` preserves the original values under explicit method-wide names, and `:13` adds scope-specific fields. All 96 rows in `evaluation/ROI_METRICS_SCOPED.csv` preserve every original error/count value and have correct scoped fields; `evaluation/SCOPE_CORRECTION.json:2` hashes both versions correctly. Main evaluator, predictions, and results hashes remain unchanged. `REPORT.md:87` discloses the correction. Use the scoped derivative for reporting; the original table retains its historical ambiguity.

2. **WARN — the new lock does not bind all evaluation dependencies.** `evaluate_mvs.py:11` imports the prior `evaluate.py`; `evaluate_mvs.py:27` obtains expected GT hashes dynamically from historical metadata. The prior `evaluate.py`, its `audit_sources.py` import, and both reference metadata JSON files are absent from the union of this RUN_LOCK, this prediction seal, and the prior camera-replay prediction seal. Thus the hash checks at `evaluate_mvs.py:21` do not independently guarantee those dependencies are the versions used before evaluation. Current contents were inspected and raw references independently rehashed; results reproduce exactly, so this is a reproducibility gap, not evidence of changed GT or fabricated numbers. Current dependency hashes are recorded below and in the JSON companion.

3. **PASS — real reference provenance and physical units.** `../upstream-photo-holdout-20260930T113213Z/evaluate.py:18` identifies the two DTU point files, and `:38` validates member/path identity and the official `Points.zip` source URL. The scan24 historical manifest records member, hash, and absolute path at `/srv/slam-research/grf/map-denoise/datasets/published-outputs-v2-reference/MANIFEST.json:3`; scan37 does so at `/srv/slam-research/grf/map-denoise/datasets/reconstruction-v22-scan37/DOWNLOAD_RETRY_1789282524663424354.json:26`. Independently hashed bytes match those records and `evaluation/RESULTS.json:354`. Independently parsed vertex counts are 5,169,152 and 5,166,240. No GT coordinate normalization is applied: the prior reader returns raw xyz (`evaluate.py:208`), and the current evaluator computes nearest-reference Euclidean distances directly (`evaluate_mvs.py:30`, `:39`, `:47`). Camera centers and depth are physical mm, with no second scale_mat transform in this run (`run_colmap.py:73`, `:83`, `:147`). Historical metadata was checked locally; the archive was not downloaded again during this review.

4. **PASS — actual execution and artifact consistency.** All 87 RUN_LOCK file hashes, all 118 prediction-seal hashes, and all 302 old prediction-seal hashes match. All 15 job records have the expected index, exact locked options, and matching depth/normal hashes. Job durations sum to 68.85280033998424 s. The installed distribution is `pycolmap-cuda12==4.2.1`; code calls its `patch_match_stereo` directly (`run_colmap.py:128`). `RUN.log:1` begins job 0, `RUN.log:990` completes job 14, and `RUN.log:991` records the 118-file seal. The independent workspace reads find exactly five registered images and zero sparse 3D points per workspace. `job_records/00.json:3` and `:5` illustrate actual duration and binary-output binding. File timestamps support lock → jobs → prediction seal → evaluation ordering; timestamps and source inspection are evidence of sequence, not an OS-level proof that no other process ever read GT.

5. **PASS — all original queries, exact missing-data handling.** Each of the four prediction arrays retains 128 unique integer pixels. They exactly match QUERY_MANIFEST order and the earlier upstream initialization's reference_pixel_xy[query_ids], for 512 total queries. Extraction directly indexes `D[v,u]` at the fixed pixel (`run_colmap.py:143`, `:147`) and leaves invalid xyz as NaN (`:148`); no neighbor search or replacement exists. Fallback uses the original CPU validity mask and replaces only the intersection (`evaluate_mvs.py:49`), giving the fixed 497-point comparison. New support among the other 15 queries is separately counted (`:60`). No outlier clipping is performed.

6. **PASS — pixel convention matches the official dense kernel.** The pre-run addendum explicitly supersedes the initial +0.5 generic-SfM description (`ADDENDUM_PRE_RUN.md:1`, `:5`; initial plan `refine-logs/EXPERIMENT_PLAN.md:15`). The actual MVS camera text has cx=388, cy=290 (`workspaces/scan24/photo/sparse/cameras.txt:1`). Official COLMAP 4.2.1 source independently confirms that Model passes camera K through, Image copies it, ComputePointAtDepth uses integer col/row rays, texture fetches add 0.5, and fusion backprojects [col*d,row*d,d,1]. Sources: [model.cc](https://raw.githubusercontent.com/colmap/colmap/4.2.1/src/colmap/mvs/model.cc), [image.cc](https://raw.githubusercontent.com/colmap/colmap/4.2.1/src/colmap/mvs/image.cc), [patch_match_cuda.cu](https://raw.githubusercontent.com/colmap/colmap/4.2.1/src/colmap/mvs/patch_match_cuda.cu), [fusion.cc](https://raw.githubusercontent.com/colmap/colmap/4.2.1/src/colmap/mvs/fusion.cc). Independently decoding raw depth and backprojecting via K/R/C reproduces every valid stored xyz within 1e-8 mm. Every generated PNG is byte-identical in pixels to the frozen half_gray formula.

7. **PASS — no phantom numbers.** Independent binary PLY and COLMAP depth parsers, independent K/R/C backprojection, and independent aggregation reproduce all 6,144 per-method/per-query distances, all 96 ROI error/count rows, all 12 tail rows, and all 12 summary records (tolerance 1e-8 for distances and 1e-10 relative for numeric summaries). The only auxiliary field discrepancy is finding 1. `EVALUATION.log` parses to precisely `RESULTS.json.summary`. MSE is the mean squared nearest-reference distance within each valid ROI, then an equal mean over four ROIs (`evaluate_mvs.py:16`, `:88`), not a pooled or clipped score.

8. **PASS — stated scope and algorithm separation.** Each workspace contains only reference 0022 and Q=0016,0035,0017,0007. Every config lists exactly the other four images. Geometric runs build five unfiltered photo maps before the filtered reference geometric map (`run_colmap.py:100`); independent workspaces prevent mixing these with filtered photo maps. Source-image depth intervals are computed from reference-frustum vertices (`run_colmap.py:58`), matching the addendum. Core algorithm parameters equal the installed 4.2.1 defaults; only resource controls, ranges, and named arm flags differ. The reported result explicitly says development replay and not the official full-scene DTU benchmark (`evaluation/RESULTS.json:372`). The scan24 repeat has identical full reference depth/normal file hashes, consistent with a repeatability check rather than independent random-seed evidence.

9. **PASS — final report, checkpoint, correction, and plotted claims.** `REPORT.md:9`, `:31`, `:44`, `:49`, `:56`, `:58`, `:66`, `:69` and `CHECKPOINT.md:7` match independent recomputation. The reported reductions are 76.072427% for geometric fallback and 95.513542% for the explicitly labeled oracle; photo worsens MSE by 74.076122%. Equal-ROI MAEs are CPU 4.232051703, photo fallback 3.125903960, geometric fallback 1.648408523 mm. Counts are 263 improved, 173 worsened, 61 unchanged/fallback, with 61 entering the ≤1 mm set. Every POINT_METRICS row agrees with the independently recomputed POINT_DISTANCES entries. `plot_results.py:21` plots original 497-query fallback MSE and `:28` uses the same paired points; all distances lie between 0.0354353 and 114.5796 mm, within the scatter axes. Both PNGs were visually inspected, showing fixed ROI identity, common scales, retained missing markers, and stated counts. The spatial map uses a bounded 0.05–50 mm log color scale (`plot_results.py:43`); this visual saturation is not numerical outlier removal. Four figure files were hashed. Environment/package/GPU statements in `COMMANDS.md:21` match direct inspection. No independent rerun of plotting or the documented full MVS workflow was performed. Unrelated literature/process citations were not substantively audited.

## Acceptance interpretation

| Equal-ROI MSE, mm² | scan24 | scan37 | Four ROIs |
|---|---:|---:|---:|
| CPU | 113.0303984356 | 187.2118110725 | 150.1211047541 |
| Photo + CPU fallback | 96.8595438454 | 425.7904500399 | 261.3249969427 |
| Geo + CPU fallback | 7.7227272745 | 64.1179464366 | 35.9203368555 |

The geometric fallback passes the preregistered two-scene improvement condition (`refine-logs/EXPERIMENT_PLAN.md:31`) and rescues 51/75 old all-candidate >5 mm queries to ≤5 mm, including 34 to ≤1 mm. It moves 23 originally ≤1 mm CPU queries above 1 mm. Photo rescues 57/75 but moves 47 old good queries above 1 mm and fails scan37's fallback-MSE improvement condition. These counts reproduce from `evaluation/TAIL_METRICS.csv:2` through `:13`; the geometric raw-output coverage is 450/512, and 14 of the 15 previously CPU-missing queries become supported (13 correct within 1 mm). The 35.9203 fallback MSE is conditional on the fixed original 497 CPU-valid queries. Oracle metrics remain GT-assisted candidate-capacity diagnostics (`evaluate_mvs.py:51`, `:63`; plan `:25`), not deployable results.

## Audit snapshot

| Artifact | SHA-256 |
|---|---|
| run_colmap.py | 1b16a6763e32048434daed9e5e106cdda45771f7579e9a5042eeb49b25913065 |
| evaluate_mvs.py | c34b85e27fb4e94aedb2f913c1ab2438d968a6eb356e2aae15ae7c514e14f16f |
| RUN_LOCK.json | cb9157b904c15da2c47e30e6ecb63cf36e2bf13b5d39056c2e380e348370477c |
| PREDICTIONS_SEALED.json | a5936d719f85ef7c435140db9398e2f4d3977f9c1b80b826fe3af45cec301bf0 |
| evaluation/RESULTS.json | b41b77a7b513a875ad8263a22dd3be893c3cdd3f777bd3017224050e82a8667a |
| normalize_scopes.py | ceac2dcc5fae583802d5162b5c91887a02b1c9677f12ac152866cf73268de5c2 |
| evaluation/ROI_METRICS.csv | beb53aabff3c4ed62ce398ff879913b73bc6edd9eeeddcdf4892b382c2e5e6b9 |
| evaluation/ROI_METRICS_SCOPED.csv | 98721f8796dcdb1a68a7342e9485b29b984d7c6f7806b3822e9cb8c631937342 |
| evaluation/SCOPE_CORRECTION.json | bb40841ab75d1b863d945400df120b5f9b4c7c0ce28bed2d3caed74dcd55b463 |
| evaluation/POINT_METRICS.csv | 03c8de008fb3ee84ff7ea765415b74bada08013d35d4f51bd93056cd34d5058b |
| REPORT.md | fa4315e8690e887da47c70f280dc46ed69149929764af90708cc648b1cdda507 |
| CHECKPOINT.md | b63f499b637568ed974aed8123dac231254239def9d2bf2090bafc9571297e0e |
| COMMANDS.md | ed39a1961883970890b8ee3c8d7e8e6c29d2a546e23f6e887689f4c066aed420 |
| plot_results.py | 84dda5f3e38a6dc5ad054c24a838db869805177c798507628d193ca77fa9c958 |
| figures/mvs_comparison.png | 5faba70c8ce0ac001984dbfd1335a6a6627b4828dce54ff5e337e76ab8ee9567 |
| figures/mvs_comparison.pdf | cba752214b8ed02118730f534b0d9936ecae691d8618225230566621d36a277f |
| figures/fixed_pixel_errors.png | cf1d2c7be66eb11632a3c1e9878461a856a2bb3d3d670ab08e1f785d8ab73b70 |
| figures/fixed_pixel_errors.pdf | 215b04c1e9d8513c1d0fbcd2c0a9f373255ae6f638e7e9921856d65031c7f97d |
| prior evaluate.py | 46003825691f594c50f389cdd13f4f495dbfe9bb30fa5c8c6097c5ac55c44a44 |
| prior audit_sources.py | 877459dcaad2ba1b5affcc250020817f2785471b860de2159c02552e56092cad |
| scan24 GT MANIFEST.json | 0c587e7116ac2d4992f5640fbee19f8cb343b2607442d36dae733d8fed55ffd9 |
| scan37 GT DOWNLOAD_RETRY_1789282524663424354.json | a6d3471e1978e6e86906537766b9ecd228b47f5a080eda9cf4c14458de7bad50 |

Fresh same-family review provides independent recomputation and source checking within this model family. It does not confer cross-family or human acceptance.
