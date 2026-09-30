# 来源快照

原始路径：`/home/grf/Documents/Codex/2026-09-30/geometry-closeout-20260930/decision/PROTOCOL.md`

# Evaluation-aligned revision decision — frozen protocol

Date: 2026-09-30. Host: liekkas. Predecessor: texture-evidence-learning-20260929T094442Z.

## Question and controlled changes

On exactly the archived KEEP/A/B coordinates, can alignment of fitting and
calibration to the reported absolute-MSE macro objective, followed by a separately
calibrated photo-texture witness, select useful corrections better?

The historical sample weight is `1/(case_count * mean_KEEP_MSE)` (then normalized).
Historical calibration averages the per-case ratios of output to KEEP MSE.
Evaluation instead averages ROI absolute MSE within scenes, averages scenes, and
then computes its ratio to the similarly averaged KEEP MSE. These are different
objectives. The new model gives equal weight to scenes, conditions, and ROI cases
within each scene/condition, and equal weight to supported points within each case.
No scene, condition, case or row identifier enters model features.

The unchanged 202 features and four benefit/harm HGB heads retain 80 iterations,
depth 3, 7 leaves, and every other archived parameter. Fit scan24 and predict
scan37, and vice versa; fit the final model on the same supported rows in both.
No new geometric candidates or rendered features are constructed.

The witness is one scalar per candidate: mean camera-donor photo gain over four
cached views. An increasing isotonic regression maps it to probability of positive
geometric gain, fit with the same equal-case weights on development labels only.
Its OOF predictions use the same cross-scene folds. This is a separate calibration
channel, not statistically independent evidence. Final calibration uses both
development scenes; their small number is a generalization limitation.

## Arms and decisions

Arms: identity, raw_A, raw_B, archived_rank, archived_old_protected,
calibration_only, aligned_balanced, aligned_no_witness, aligned_witness (primary).
Calibration-only reuses frozen historical OOF/model scores but changes the risk
objective. Aligned-balanced is an unconstrained absolute-MSE diagnostic.

For protected arms, development native macro MSE must not exceed KEEP; each
shift must retain 90% of the frozen ranker's positive development macro gain.
Search 50 score quantiles plus zero. Witness gates are fixed before execution:
0, 0.5, 0.6, 0.7, 0.8, 0.9. Gate is `probability >= gate`, before candidate choice.
Select minimum equal-condition mean relative macro risk among jointly feasible
settings. If infeasible, choose the native-safe setting with greatest minimum
recovery margin, then minimum mean risk. KEEP is an explicit available setting.
No replay outcome can change this selection. Report an all-KEEP result as no repair.

Unchanged historical score canonicalization: same A/B physical coordinates have
the mean finite score; unchanged candidates are excluded at displacement <=1e-7
mm. Ties choose smaller displacement, then lexicographically smaller XYZ.

## Integrity and endpoints

Fit/calibration labels: the 79,594 archived development rows, 78,598 supported.
Replay: all 36 cached cases in scans83/97/105/106/110/114 (two ROIs, three inputs).
Seal all predictions, routes, actual PLY outputs and canonical API batches before
opening any replay reference. Compare the same uniform input cohort, original
reference support and fixed reverse KEEP footprints. Primary metric is absolute
ROI→scene macro source MSE, with MAE, precision/recall, movement and damage also
reported. Uniform and selected populations remain distinct.

Primary success: native is better than archived_old_protected and nonworse than
KEEP; both shifts retain at least 90% of positive archived_rank macro gains.
Also report improvements, harms and movements on supported inputs. These six
scenes are exposed development replay, not new confirmation. Do not promote any
best replay arm to deployment default. CPU run only; old artifacts remain read-only.

Stop after this locked comparison. No threshold retuning on replay, feature search,
new candidate generation, GPU run, or new-scene claim is part of this experiment.
