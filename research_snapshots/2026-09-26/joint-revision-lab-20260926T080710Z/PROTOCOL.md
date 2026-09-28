# Joint utility and support-aware revision — locked design

2026-09-26, liekkas. Question: can joint comparison of KEEP/A/B exploit complementary
geometry while preserving native input? This is exposed replay, not independent
confirmation. A and B geometry are historical; the experiment changes selection and
candidate-evidence matching. Historical post_AB routing exists: no first-router claim.

## Construct

For each input p, freeze three coordinates q0=p,q1=A,q2=B. Learn expected signed
source-MSE reductions in the SAME mm² scale, g_c=e0−e_c. KEEP has utility0. One shared
scorer consumes own features, competitor features, own/competitor displacement norms,
separation and displacement cosine. It is applied in both orders with tied parameters.
The resulting rule chooses the highest utility proposal above a calibrated movement
cost t; at natural t=0 it competes directly with KEEP. Exact/near-zero displacements
(<=1e−7 mm) are KEEP. Duplicate A/B coordinates use averaged utilities. Tie-breaking
uses displacement then coordinate lexicographic order, not learned candidate identity.
No spatial graph, smoothing or new geometry is introduced.

Candidate-specific evidence: A uses existing h5=PCA24/full paired reserved evidence.
At frozen B coordinates, choose a nuisance h from four input-derived normals × five
7x7 full/half footprints using ONLY four construction sources F, top3 aggregation with
at least2 finite sources. Evaluate p and B at this SAME h using four reserved sources R.
All invalid F hypotheses yield h=−1 and neutral/missing flags. The chosen h is newly
fitted at final B, not its historical anchor-interpolated construction hypothesis.
R is not used to optimize support. Its IDs, calibration and pictures stay frozen.
This adds genuine B evidence; A's added32 duplicates existing R32. It is not a symmetric
information-volume increase or explicit visibility estimation. Shared reference and
initial MVS mean source-disjoint does not imply statistical independence.

## Arms and fitting

Five fixed score producers, each balanced/native_priority/natural:

1. independent_absolute: two separate HGB models on own R96, raw mm² gains.
2. joint_common: one shared HGB on own+competitor R96 plus geometry4 (196D).
3. joint_support: one shared HGB on own+competitor augmented128 plus geometry4 (260D).
4. normalized_max: previous independent R normalized-score models, direct max. This
   intentionally tests the old unequal-denominator shortcut, not a valid utility identity.
5. support_margin: candidate-specific mean R margin, >=2 valid paired views. No training.

Primary: joint_support__balanced. Baselines include identity, A_R/B_R balanced,
B_R native_priority and historical post_AB when compatible. Evaluator-only A/B/AB
oracle remains separate. No winner substitution or extra variants after replay.

HGB: squared_error,learning_rate=.08,max_iter=80,max_leaf_nodes=7,max_depth=3,
min_samples_leaf=80,l2=1,seed20260926,early_stopping=False. Joint parameter sharing
reduces tree count relative to two independent models; it is not identical capacity.
Both consume the same two gain labels per eligible development row.

Development24/37 uses same79,594 sampled rows and78,598 fixed calibration-support rows.
New HGB fits only the fixed support rows, equally weighting cases after division by
case identity MSE: w=1/(n_case*max(mean_case(e0),1e−6)), normalized to mean1.
Development truth enters labels/weights/calibration only, never evidence features.
Two scene-held-out folds produce paired utility OOF scores. Calibrate one threshold
per arm using100 quantiles of best finite proposals plus0 and KEEP, minimizing mean
case-relative MSE. Native_priority requires mean native MSE<=identity and injected
relative MSE<=.95; otherwise KEEP. Natural uses0. Calibration is development tuning,
not confirmation. Freeze all models/thresholds before replay reference access.

## Evaluation and closure

Replay55/65/69,4 fixedROIs,5 conditions(native,−1,+1,−3,+3mm),60cases. All previously
exposed. Fixed native support and exact prior reference distances. Main MSE, secondary
MAE/P95, >.1mm harm/benefit, actual moves, A/B/KEEP fractions, ROI wins, oracle regret,
B complementary-positive-benefit capture (gross, not net). ROI then scene equal weight.
Export joint_common and joint_support balanced,120PLY; all15route masks and scores.
Ten randomized route-count controls match A/B/KEEP counts within support/non-support,
not necessarily actual move counts. Evaluator-only oracles never feed training.

Minimal verification: row/geometry identity, source split, units, shared-scoring swap
equivariance (coordinate output), missing evidence, identical geometries, exported
coordinates, independent metric reproduction and hashes. Reuse frozen datasets rather
than copying error arrays. CPU-only, no installs/downloads/GPU. End after one locked
queue: implementation/tests → evidence → fit → infer → evaluate → review/report.

Interpretation: if common utility helps, the old target/comparison matters; if support
helps beyond common, candidate-evidence matching matters. No change alone certifies
causal uniqueness or global architecture optimality. A successful recovery arm remains
an asset even if native protection fails; report both instead of only declaring failure.
