# Continuous step, not a fourth endpoint

Locked before new replay outcomes, 2026-09-26. Host liekkas.

## Questions and representation

With all historical KEEP/A/B coordinates fixed, replace the two endpoint choices
by the union of segments [p,A] and [p,B]. The exact oracle minimizes distance to
the same finite laser-reference cloud over these segments; this is not a claim
about an unknown continuous physical surface or a globally optimal architecture.
The union includes all three old candidates, so its squared source error cannot
exceed the discrete oracle pointwise. No extrapolation or A-to-B averaging.

Separate a second oracle restricted to the branch chosen by the previously locked
base_shallow__recovery selector. This separates wrong branch/KEEP decisions from
remaining step-length error. The restricted oracle need not beat discrete AB.

## Actual operator and matched baselines

Freeze prior recovery routes. KEEP stays KEEP. For each selected A/B branch,
score actual coordinates at lambda={0,.25,.5,.75,1} using the four previously
reserved source images, current-case input PCA-k24 normals and full 7x7 patches.
Only sources valid across all five positions are compared; require at least two.
Otherwise retain lambda=1 (the previous output). Use the common-source mean cost;
ties prefer lambda=0. This is a changed use of reserved views: they now select
step lengths, so they are NOT held-out validation of this new operator.

Primary actual arm: grid photometric minimum. Secondary: a bounded quadratic
vertex fitted to the selected grid minimum and its two neighbors when convex,
interior, and finite; otherwise grid output. The secondary is a continuous
approximation, NOT an exact continuous photometric minimum. Pure constant step
.25/.5/.75 on the same recovery branch provides the shrinkage control. No fitting
or threshold tuning in this round. Preserve all outcomes, including regressions.

## Data and timing

Frozen scans55/65/69, four photo-defined ROI each, conditions native/minus1/plus1/
minus3/plus3, total60 cases. Previously exposed data; no independent confirmation.
The observation operator reads input candidates, input-case normals, frozen route
and original images/cameras only. Its outputs and construction protocol are
sealed before the main evaluator loads laser references. Old file seals checked.
No GPU use. Record elapsed time and own process peak RSS (not total pipeline RAM).

## Metrics and oracle evaluation

Use identical historical fixed native-row support, ROI crop, official observation
mask and .8mm reference voxelization. Source point-to-reference MSE (mm^2) is
primary. Average within ROI, then four ROI equally within scene, then three scenes
equally; compute gain from the aggregated MSE, not mean per-point percentage.
Also source MAE, p95, >.1mm benefit/harm fractions, movement, reverse reference-to-
supported-output MAE, and precision/recall/F-score at1mm. These secondary metrics
are additional diagnostics, not claimed identical to an official full-scene score.

Evaluate identity, previous recovery, three constant shrinks, two observation
arms, discrete KEEP/A/B oracle, exact union-of-segments oracle, exact selected-
branch oracle (10 arms). Oracle source scores concern the frozen support only;
diagnostic oracle PLY retains original points outside support and is labelled.
Actual PLY outputs retain all rows and operate without the evaluator support mask.

## Tests, interpretation and completion

Exact kernel versus exhaustive finite-reference projections; zero-length segments;
endpoints and ties; union oracle <= discrete oracle; fixed-branch oracle <= prior
recovery; direct geometry/metric agreement; independent NN spot checks; photo
step rules tested on synthetic curves, invalid observations and known vertices.
Report extra oracle headroom and how much actual methods recover, without changing
the primary arm based on outcomes. A larger oracle gap is potential, not realized
benefit. A grid improvement alone does not prove the quadratic method works.
Finish with executable modules, tests,60-case metrics and actual point clouds,
input/output hashes, report and checkpoint. Do not overwrite deployment defaults.
