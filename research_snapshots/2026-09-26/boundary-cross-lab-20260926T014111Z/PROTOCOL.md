# Frozen candidate × evidence crossing

Host liekkas, 2026-09-26. No outcome of the B/new-evidence combination has been read
when this protocol is written. Both historical candidates and all replay scenes are
already exposed; this is a mechanism/system comparison, not independent confirmation.

## One bounded question

Does replacing the single PCA-k24 full-patch A with the archived multi-normal,
half-patch B change the benefit of source-disjoint paired verification? Separate
candidate recoverability from deployable selection; do not hide ungated candidates.

B is the existing V28 B_all: 4 normals × 5 overlapping spatial footprints, a fixed
ray-depth search, top-three-of-four source aggregation and geometry-weighted anchor
interpolation. Half patches can avoid crossing boundaries; trimmed source aggregation
can tolerate a disagreeing source. Neither constitutes explicit occlusion/visibility
reasoning or guarantees physical layer preservation. Reuse exact sealed B coordinates.
This design tests a previously built capability under new evidence, not new B novelty.

## Paired 2×2

Candidate factor A/B; evidence factor F/R. F uses original four construction sources;
R uses the four source IDs frozen in reserved-evidence-lab-20260926T011717Z.
Reference image and input PCA-k24 normal are shared, with same full 7×7 patch scoring
and same32 paired features for both candidates. R excludes all five original IDs,
but is not statistically independent or held out from the initial reconstruction.
Do not change view selection, patch support or normal selection after inspecting B.

The cached64 base features are the corresponding A_all_post or B_all_post schema;
never feed B geometry with stale A proposal features. Candidate-specific models have
96 dimensions and identical HGB capacity, normalized-gain target, case weights and
scene-fold calibration. A_F/A_R are exactly the previous sealed fit_aug/reserved_aug
models, thresholds and decisions. B_F/B_R are trained on the SAME sampled rows, using
historically sealed B reference labels. Freeze B models before replay inference.

Predeclared candidate under test: B_R balanced. Comparator set includes B_F balanced,
A_R balanced and A_F balanced. Also report natural0 and native_priority for all four,
without replacing the primary arm by a post-hoc winner. Report raw A/B, identity,
previous normalized A, previous frozen A and four fixed paired-mean-margin>0 gates
(>=2 valid pairs) so training/calibration does not monopolize the evidence comparison.
No AB score-max router is deployed: normalized scores from different candidates are
not guaranteed commensurate with absolute MSE gains.

## Data and evaluation

Development24/37:24 cases,79594 old sampled rows;78598 fixed calibration-support rows.
Replay55/65/69:3×4ROI×5conditions=native,−1,+1,−3,+3mm,60 cases; no new confirmation.
All methods retain original point order/count. Recompute distances for B to the exact
same reference/support used for A; verify regenerated A/identity errors against the
preceding sealed arrays before accepting B metrics. References remain evaluator-only.

Primary: fixed-native-support source MSE. Secondary MAE/P95, >0.1mm benefit/harm rates,
actual edits and region wins; report 5 conditions separately, ROI then scene equal
weight. No claim of completeness, topology, physical-layer identity or missing-point
generation. Native is not synonymous with all-valid geometry.

Add evaluator-only A/KEEP, B/KEEP and A/B/KEEP oracles. Quantify B's incremental benefit
over the best of A/identity and how much of it deployable B selection captures.
These are fixed-candidate separable-loss bounds, not universal attainable performance.
For B_R balanced, ten count-matched and ten displacement-bin-matched random controls,
within support groups as in the previous experiment. No pointwise statistical claims.

Interaction per condition: (gain_BR−gain_BF)−(gain_AR−gain_AF), using the same identity
denominator; describe it, without treating 12 ROIs as 12 independent scenes.

## Queue and acceptance

Audit provenance -> read/reconstruct sealed B development labels -> score B on five
scenes -> train two B models -> freeze -> infer60 -> evaluate+oracles -> independent
audit+figure+report. Export B_F/B_R balanced and B_R native_priority (180PLY) plus all
decision masks. Preserve actual failed arms. End after this queue.

If B expands oracle recoverability but deployable routing fails, the gap is selection
or learning under the tested interface, not proof B is useless. If B adds no useful
candidate geometry here, do not claim more elaborate verification can invent it.
Do not promote an arm unless it supports the stated protection AND repair goal; a
tradeoff remains useful evidence without becoming a deployment upgrade.
