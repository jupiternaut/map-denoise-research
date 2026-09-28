# Continuous-step independent audit

Status: PASS for the checked implementation and numerical contracts. Host `liekkas`.
All prior files remained read-only. This review did not tune a method or change a result.

## Concrete findings

- The segment oracle is genuinely continuous for the **finite reference cloud**. It analytically projects each retrieved reference point onto a closed segment; the midpoint bounding ball includes every potentially improving reference. There is no lambda grid in this oracle.
- Independent exhaustive projection checked 692 synthetic segments against 257 references (177,844 pairs), including 28 zero-length segments, coordinate scales 0.001/1/1000 and a 1e8 translation. Maximum lambda difference was 2.22e-16. Squared-distance absolute discrepancy was at most 4.66e-10 at scale 1000; ordinary scale discrepancy was at most 6.66e-16. All scale-relative comparisons passed.
- A real artifact check used 32 fixed-support rows and both A/B segments in `scan55_roi0__minus3`, exhaustively testing all 97,914 reference points: 6,266,496 projections. Stored oracle squared distances and lambda both agreed within 2.22e-16.
- Six sealed observation cases, 117,330 active rows, reproduced the common-source mask and grid argmin directly from saved photo scores. All 15,139 accepted quadratic refinements had strictly lower freshly measured photo cost on the same sources. This checks their decision rule, not that photo improvement implies geometry improvement.

## Evaluation contract

The evaluator reuses the historical ROI crop, official observation mask and 0.8 mm reference voxelization. The point-row support comes from native input and is checked against the saved support mask; revised coordinates do not redefine support. All three observation inference seals precede evaluator reference loading. The observation module reads current input/candidates, frozen routes, cameras/images and current-input normals, not reference laser or hidden unperturbed geometry.

All 600 CSV rows are unique. Independently recomputed 600 summary values (12 metrics × 10 arms × 5 conditions), with equal ROI means within each scene and equal scene weights. Maximum discrepancy from SUMMARY was 5.33e-15. Every case satisfied union-segment oracle ≤ discrete oracle and selected-branch oracle ≤ prior recovery. A selected-branch oracle is **not** required to beat unrestricted discrete A/B: the selector may have retained the wrong branch or KEEP.

An audit-discovered secondary metric issue was fixed before evaluation: constant 0.25/0.50/0.75 controls originally omitted their step arrays, reporting zero interior-step fraction. They now report real interior steps and exclude stationary points. Primary MSE was unaffected. The new `move_RMS_mm` uses fixed-support rows; the previous visibility experiment used all input rows. These movement magnitudes must not be compared as identical denominators.

## Result attribution

The independent summary check confirms discrete → union oracle improvements of 69.942709% → 74.690546% at −3 mm and 63.266963% → 64.985326% at +3 mm. These are increased attainable gains under evaluator-only selection, not deployment performance. Grid photo selection gives 47.210548% / 38.548220%; the frozen full-step route gives 46.994378% / 39.022162%. Therefore the actual step rule is not an all-condition improvement.

Reverse MAE is approximately 15 mm even for identity (native 15.181720 mm, native recall@1mm 0.283411). Its target is the whole cropped reference versus only supported output rows. A reference area outside the output footprint contributes to this distance; it is not purely local surface displacement. We did not separately measure footprint area in this audit. Do not call these supplementary diagnostics official full-scene completeness or comprehensive structural recovery. Likewise the source-optimal oracle does not optimize reverse distance or preserve physical layer identity. Since A/B lie on the same reference ray, their segment union is the p/A/B envelope interval; this does not certify avoidance of thin-layer crossings.

Audit source and result hashes are recorded in `AUDIT.json`.
