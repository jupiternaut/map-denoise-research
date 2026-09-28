# Relative gain versus confidence: first executable comparison

Date 2026-09-26; host liekkas. This is a new exploratory development/replay study,
not a new independent confirmation or an official baseline reproduction.

## Question

Can choosing by relative geometric improvement outperform choosing by absolute
candidate confidence, using the SAME A candidate and observable feature array?
Can alternative gain factorizations improve the native-input operating point
without discarding the already demonstrated large-error recovery?

## Fixed evidence

- Old V28 workspace: /home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z
- Frozen closeout: /home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z
- Training: 24 archived cases of scan24/37; archived row IDs, at most3333/case.
- Threshold validation: two leave-one-SCENE-out folds, never random point split.
- Replay: all60 cached cases, scenes55/65/69, native/-1/+1/-3/+3, four ROIs each.
- No replay scene/condition IDs enter a policy. No new views, candidate changes,
  candidate-dependent cropping, or evaluation-reference features.
- Old results are already known; replay can reject a construction, not establish
  independent generalization. A future untouched dataset remains necessary.

## Arms and model budgets

Keep identity, A_all, old frozen post_A_keep; add scalar photometric confidence,
source agreement and small-displacement selectors; same-feature learned absolute
candidate confidence/quality; gain-sign classification; direct gain regression;
normalized gain regression; separate benefit/harm regression; two-part probability
times conditional magnitude. Exact implementations must be documented before
scoring replay. Tree parameters reuse the old shallow HGB budget where possible.
Multi-head methods have larger compute budgets and must be labelled as such.

## Common operating-point selection

For each scored method, use out-of-scene development scores and a finite threshold
set: 101 score quantiles plus its default, accept-all, and keep-all endpoints.
Strict `score > threshold`. Case weights are equal; scenes and the three training
conditions are equally weighted. The balanced point minimizes mean relative MSE
(each case divided by its identity MSE, denominator floor1e-6 mm²). Tie: fewer accepted
candidates (including zero-displacement candidates); prefer an explicit endpoint
over an equivalent finite threshold. Quantiles use calibration-support rows only.
Accept-all and keep-all are explicit actions, not finite development extrema.
An additional native-priority point minimizes the same objective subject to
native case-equal ABSOLUTE development MSE not exceeding the case-equal identity
MSE AND mean +/-3mm relative MSE <=0.95.
If infeasible, report infeasible and return KEEP; do not invent a successful gate.
Report both predeclared operating points and natural thresholds where meaningful.
No threshold is recalibrated on replay. The normalized-score method is the
predeclared exploratory lead, not a winner selected after replay.
Archived training rows retain their historical condition-dependent valid support.
Threshold calibration uses only their intersection with each ROI's native fixed
support. This is a sampled calibration set, not full-ROI evaluation. OOF scores
used to choose thresholds are tuning evidence, not an unbiased performance test.
Training absolute squared errors are recomputed in float64 on the same reference;
their difference must reproduce the archived gain labels.

## Outcomes

Primary: fixed native-support MSE by each of five conditions, ROI then scene equal
weight. Report per-scene/per-ROI win counts. Native outcomes never averaged into
injected outcomes to claim general improvement. Secondary: MAE, P95, accepted and
moved fraction, displacement RMS, benefit/harm mass and point fractions (>0.1mm).
Evaluate identity/raw/frozen baselines numerically against archived CSV first.
Source support is read-only from the sealed closeout evaluator. Set completeness
and physical thin-layer identity are NOT established by this fixed-row experiment.

Random controls: count-matched and absolute-displacement-bin matched acceptance
for the old policy and exploratory lead (ten seeds); these are diagnostics, not
deployable methods. Retain counts/actual displacement; bins are fixed in mm.
The evaluator creates these random controls AFTER inference, stratified by native
evaluation-support membership so their accepted counts match inside and outside.
No support mask is provided to any learned/scalar policy. Bin boundaries in mm:
0, 1e-7, 0.25, 0.5, 1, 2, 3, 4, 6.000001, infinity. Bin matching is not exact RMS matching.
Oracle fixed-A choice is evaluation-only, used to split candidate and selection
headroom. Never choose a deployable winner by per-case oracle scores.

## Integrity and delivery

Fit and seal all models/thresholds before replay scoring. Save all decision masks
and predictions; export full PLY geometry for the old policy and predeclared lead
plus native-priority lead, not only metrics. Include a source hash manifest,
commands, tests, independently recomputed rows, and a report separating algorithm,
mechanism evidence, exposed replay, and still-missing external confirmation.
No rewriting the thesis or changing deployment default in this run.
