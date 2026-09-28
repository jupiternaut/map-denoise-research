# Independent review outcome

Status: PASS for the implemented, predeclared experiment and its numerical
accounting. This is not a claim of native-geometry safety, new-scene confirmation,
true visibility, or globally optimal architecture.

## Evidence construction

`AUDIT_EVIDENCE.json` verifies 84 cases and 20 ROIs. New sources exclude all five
original images, and input-only view rankings reproduce independently. Every case
retains the correct archived row order and exact incumbent/A pair. F/R summaries
recomputed independently from raw paired scores agree within `5.96e-08`; maximum
A ray residual is `2.84e-14 mm`. Forty-five fixed sampled rows across five scenes
were rescored from original pixels with identical missingness and zero numerical
score discrepancy. Full normal computation precedes row sampling. The two source
cohorts use the same full-point normal and reference patch.

The old columns named `heldout_*` are not held out for A_all; full provenance is
in `AUDIT_VIEW_PROVENANCE.md`. Additional views are disjoint from A's local search,
not certified disjoint from initial MVS or statistically independent. In-image
projection and finite photometry do not prove physical surface visibility.

## Training and decisions

`AUDIT_SELECTION.json` verifies 79,594 development rows, 78,598 calibration-support
rows, two scene folds, four matched-capacity HGB models and all operating-point
choices. The old X64 model's OOF predictions reproduce exactly. Threshold grid,
ALL/KEEP endpoints, tie breaks and native-priority feasibility were independently
recomputed. All 60 replay cases' predictions and decisions reproduce, including
old baselines, paired gates and intersections. All 240 exported PLYs exactly equal
their selected archived A/identity coordinates. No replay labels are used here.

## Metrics

`AUDIT_METRICS.json` independently recomputes all 2,460 method/case rows and 11
numeric metrics, with maximum absolute discrepancy `2.91e-11`. All 205 pooled
condition/method cells, equal-ROI/equal-scene aggregation and win/tie/loss counts
reproduce. The 1,200 matched random masks preserve target counts within their
declared support or support/displacement groups. The fixed-A oracle bounds every
arm's separable source MSE. Error arrays were accessed only in this evaluator
audit after model and inference seals existed; selectors were not changed.

## Native-priority interpretation

Development native conditions contain 26,664 sampled rows, all in calibration
support. At the selected OOF native-priority thresholds, `reserved_aug` and
`both_aug` accept zero native rows; their native-MSE equality with identity is
abstention, not evidence of successful native repair. `fit_aug` accepts 9 native
rows. `cached64` has no feasible operating point and explicitly returns KEEP.

The replay native queue contains 586,896 rows, with 516,926 fixed support rows.
Actual coordinate modifications are:

| Native-priority arm | Accepted rows | Actually moved rows | Moved support rows |
|---|---:|---:|---:|
| cached64 | 0 | 0 | 0 |
| fit_aug | 78 | 78 | 78 |
| reserved_aug | 61 | 61 | 61 |
| both_aug | 56 | 56 | 56 |

These are integer pooled counts, not equal-ROI averages. Do not equate the very
small edit fraction with native restoration. Reporting near-identity behavior
must retain both the accepted/moved counts and the measured geometric outcome.

## Scope of causal and architectural conclusions

The primary fit-direct versus reserved-direct comparison isolates the source
cohort under the same paired feature schema, model capacity, target, development
calibration and fixed candidate geometry. Comparisons with X64 also alter the
measurement location from interpolated anchor scores to final-point scores.
All arms remain choices between fixed A and identity. Their oracle is a ceiling
for this restricted candidate set and evaluation loss, not for geometric repair
or all architectures. The experiment does not test creation of missing coverage,
alternative directions, layer association, topological preservation or a globally
optimal reconstruction architecture.
