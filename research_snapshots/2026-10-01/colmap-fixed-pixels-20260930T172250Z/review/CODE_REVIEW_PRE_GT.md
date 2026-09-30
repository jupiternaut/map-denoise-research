# Frozen-code review before reviewer GT access

Target: `liekkas:/srv/slam-research/grf/map-denoise/runs/colmap-fixed-pixels-20260930T172250Z`.

Reviewer: **existing**, not fresh; `review_independence=same-family`; `acceptance_status=provisional`.

Review stage: author inference/evaluation had completed during this review, but the reviewer had not opened GT coordinates or consulted numerical results when completing the code and input checks below. The separate independent numeric check is subsequent. This is not evidence that a fresh reviewer approved execution before it started.

## Verdict: PASS for the declared construction contract; qualified reporting WARN

No code path was found that uses laser GT or historical difficult-point membership to construct, filter, select, or rerun predictions. The dense K and actual-depth-endpoint issues in `DESIGN_REVIEW_PRE_GT.md` are resolved by the frozen addendum and implementation. Claims remain conditional on the supplied operational calibration and exposed two-scene development sample.

### Dense coordinate and image contract: PASS

- `ADDENDUM_PRE_RUN.md:1` explicitly supersedes the initial generic corner-coordinate description; `:5` fixes dense-array K with principal point `(388,290)`.
- `run_colmap.py:73` scales the original K and subtracts 0.5; `:74` through `:86` checks the reconstructed P and the actual integer-pixel dense ray equation.
- `run_colmap.py:30` preserves the original RGB BILINEAR resize followed by the exact matmul luminance operation and uint8 truncation. `:77` through `:79` saves and checks lossless PNGs.
- Independent read-only validation in `review/independent_numeric_check.py --identity` checked all **25 image copies / 10 distinct scene-view pairs**, all five serialized camera sets, and explicit source lists. Maximum serialized P difference from old corrected P: **1.1641532182693481e-10**. Maximum independent ray difference: **8.455458555545192e-13 mm**. These numerical identities do not validate the external physical calibration.
- The checker preserves the original `@` operation intentionally: an initial independent `np.dot` rewrite changed uint8 rounding ties, although the author's actual images exactly matched the original `@` recipe. This was a checker substitution error, not an input discrepancy.

### View set, ranges, workspaces and repeats: PASS

- `run_colmap.py:56` fixes reference plus four Q; `:92` writes each explicit source list. No C/H inputs enter these five-view workspaces.
- `:57` uses actual old reference grid endpoints `[559,786]` and `[581,865]` mm. `:58` through `:64` computes each Q range from the eight reference-frustum vertices without sparse points or GT.
- `:65` through `:69` creates separate photo, geo, and scan24 photo-repeat workspaces; `:95` writes empty points3D and `:96` checks zero points.
- `:100` through `:107` fixes five unfiltered photo maps followed by one filtered reference geometric map; the primary photo map is separately filtered. `RUN_LOCK.json` contains 15 matching jobs, default patch settings, `max_image_size=-1`, and the declared resource limits.
- `ADDENDUM_PRE_RUN.md:8` appropriately changes the unsupported independent-seed claim into a single independent-workspace repetition of scan24 photometric. This does not establish multi-seed robustness.
- Geometric consistency is a same-photo/camera/query mature-method comparison, not a solver-only or geometric-term-only ablation: the source-depth pass and support ranges differ (`ADDENDUM_PRE_RUN.md:7`, `:10`).

### Immutable queries and missing values: PASS

- `run_colmap.py:109` uses the old initializer's complete 128 fixed queries per ROI, not only its successful points. `:143` through `:149` samples exactly D[v,u], retains nonpositive/nonfinite depth as invalid, and outputs NaN XYZ instead of relocating a pixel.
- Independent identity checks verified all **512 manifest IDs/pixels/hashes**, exact success-mask correspondence and the original **497 CPU-valid** identities.
- `evaluate_mvs.py:38` reconstructs CPU points by original query ID. `:49` through `:50` substitutes COLMAP only where both old CPU and that method are valid; missing COLMAP retains CPU. The remaining 15 positions are not inserted into this fallback denominator.
- `:68` computes native-valid and CPU-valid-common metrics separately; `:88` through `:92` computes four-ROI equal-weight means. No outlier trimming, fitted coordinate transformation or prediction-statistic normalization is present.

### GT boundary and historical 75-point stratum: PASS

- `run_colmap.py` imports no evaluator and contains no GT or difficult-stratum input. Its initialization NPZ access supplies fixed pixel coordinates; no historical candidate error is loaded.
- `evaluate_mvs.py:21` through `:26` verifies new predictions, the run lock and replay prediction hashes before `:27` through `:30` obtains the fixed official references and opens PLY coordinates.
- Imported original `evaluate.py` and `audit_sources.py` have no import-time GT read. The reference paths and unit convention reside at original `evaluate.py:18` through `:45`; file coordinates are only opened inside its explicit reader.
- `evaluate_mvs.py:40` through `:43` computes old candidate distances only after GT is opened; `:53` defines `hard = valid & (oracle > 5)`. `:54` through `:60` reports this stratum, with no subsequent call to construction or change to predictions. Its population is therefore diagnostic, not a construction selector.
- `:51` through `:52` and `:61` through `:65` are GT-oracle candidate-pool bounds, explicitly separate from fixed fallback. They are not deployable routing results.

### Seal and dependency verification: PASS at the reviewed snapshot

The independent identity check verified **87 RUN_LOCK entries, 118 new prediction-seal files, 302 replay prediction-seal files, 162 original seal files and 52 original external sources**. No hash mismatch occurred.

The imported prior evaluator dependencies are not directly listed in the new RUN_LOCK, but their bytes match the original source seal, now independently checked and additionally recorded here:

- Original `evaluate.py`: `46003825691f594c50f389cdd13f4f495dbfe9bb30fa5c8c6097c5ac55c44a44`.
- Original `audit_sources.py`: `877459dcaad2ba1b5affcc250020817f2785471b860de2159c02552e56092cad`.

The numerical checker records these hashes with the reviewed new seals and its own source hash. The source check is evidence of current consistency, not proof against all unlogged historical I/O.

## Reporting cautions (no method change requested)

1. `evaluate_mvs.py:70` writes full-method `coverage` and `correct_per_requested` even on `cpu_valid_common` rows. Those fields are method-wide, while `n` and error use the row's selected common mask. Do not describe that coverage field as common-subset coverage; common coverage is `n/128`. The paired common MSE summary itself uses matching masks.
2. `all_valid` native MSE is conditional on each method's validity. It cannot by itself prove improvement across 512 queries. Lead with coverage plus the fixed-497 fallback and report new support separately.
3. `ADDENDUM_PRE_RUN.md:7` calls geometric a main arm whereas the earlier design review proposed it as auxiliary. Both were frozen before inference; any result language must disclose the additional depth-map/support mechanism and avoid isolated-causal attribution. Do not select which arm is primary after reading GT.
4. The code reports raw rows sufficient for scene-specific acceptance but does not implement the plan's two-scene pass/fail criterion automatically. Any final acceptance statement needs explicit evidence for each scene, not only an aggregate.
5. This remains a same-family provisional check by an existing participant, not a fresh independent post-experiment review or official full-scene DTU evaluation.
