# Free construction: silhouette enclosures with nuisance elimination

Host: `liekkas`. Logical experiment root:
`/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z`,
which resolves through the existing runs symlink into `/mnt/data4t-usb/experiments/`.
First recorded construction checkpoint: 2026-10-13 08:08:27 UTC.

## Mathematical object and observation encoding

The representation is a finite union of depth intervals, each labeled by one
of the three rear-surface branches. Every retained cell denotes a relaxed set
of worlds that could explain the observations. The reported repair uses the
convex hull of that union. Depth subdivisions are uniform in reciprocal depth,
because projected rectangle edges are affine in reciprocal depth when the
camera center is fixed.

Camera measurements first constrain the common target error using the middle
camera, whose offset is exactly zero. Eliminating that error gives conservative
intervals for the side-camera centers, intersected with their physical bounds.
This elimination retains necessary camera constraints and discards their
remaining correlation. It uses no true camera parameters.

For every depth cell and camera interval, projected rectangle edge intervals
give an inner rectangle contained in every possible silhouette, and an outer
rectangle containing every silhouette. Pixel area is integrated analytically.
For front material k these give visible-area bounds Aminus <= A <= Aplus.
For rear R occluded by front F, set inclusion gives

    Rinner \\ Fouter subset Ractual \\ Factual subset Router \\ Finner.

Integrating these sets over each pixel bounds visible rear area while preserving
the common rear rectangle in the area-minus-intersection expression.

The linear nuisance variables are material darkness q[v,k]=1-c[v,k], three
photo biases b[v], and three independent appearance biases ba[v]: 15 variables.
q is bounded by [0,1]; photo biases by [-.01,.01]. Appearance references bound
ba, and each clipped coupon gives its applicable one- or two-sided inequality.
Transfer and appearance noise combine by Minkowski addition of their bounds.
For each material, all pairwise cross-view q differences are at most .1.

For a photographic pixel the three channels intersect into an interval [L,U]
for the pre-clipped grayscale plus photo bias. Saturation produces a one-sided
constraint, never a fictitious equality. Necessary linear constraints are

    sum_k Aminus[v,p,k] q[v,k] - b[v] <= 1-L[v,p]
    -sum_k Aplus[v,p,k] q[v,k] + b[v] <= U[v,p]-1.

Only finite bounds are added. Therefore infeasibility of this relaxation
excludes the entire geometric cell in exact arithmetic. Feasibility means
only that exclusion failed, not that there is a compatible physical world.

## Search, decoding, and preservation

Three initial cells cover all depths and branches. LP-infeasible cells are
removed; cells at the current minimum or maximum are bisected. Interior cells
need no refinement because decoding uses the hull. Every untested child is
retained, including at budget expiration. Boundary refinement stops at .025 mm.
The decoder clips the incumbent to [minimum retained lower endpoint, maximum
retained upper endpoint]. It never sees other incumbents or development truth.

If the true depth lies in this hull, interval projection cannot increase
absolute or squared coordinate error, and a correct incumbent is unchanged.
The physical information can be insufficient: e.g. if every material and the
background have identical gray value, geometry does not affect the photos.
Independent appearance and camera measurements then cannot identify depth.
This construction appropriately retains a broad interval in that case.

The guarantee above is conditional on enclosure soundness and reliable
infeasibility decisions. SciPy/HiGHS is floating point, not an exact certificate.
Small outward area padding and LP slack reduce boundary rejection risks; they
do not establish a machine-checked proof. Solver failure retains the affected
cell. Invalid encoding or an empty result returns the identity interval and
records the failure.

## Retained and relaxed information; expected behavior

Retained: pixel-area integration, three channels and saturation bounds, all
rear branches, exact center-camera offset, independent appearance/photo
biases, shared per-material cross-view range, and every stated uncertainty
bound. Relaxed: correlation among side-camera offsets after eliminating the
target error; correlation of geometric area fractions within one cell;
interior gaps in the final union when decoding through its hull. No new
measurement, color prior, depth prior, or noise distribution is introduced.

This is a conservative approximate representation and elimination of the
original feasible-world problem, followed by a decision rule for coordinate
loss. It is not a changed physical model or a claim of mathematical novelty.
The expected benefit is correction of incompatible incumbents with low damage
to already-correct ones. It deliberately forgoes shrinkage within ambiguous
depth intervals, so very small incumbent errors may remain unchanged.

## Frozen structure ablation

`prepare_ablation` removes only the cross-view material range constraints.
All other representation and search decisions are identical. This ablation
was specified before development data were read; it measures the utility of
that coupling, not a separate tuned method.

## Budget and information-exposure record

Only this root's `AGENTS.md`, public `task_bundle/TASK.md`, public
`task_bundle/forward.py`, and this arm's own files were read for construction.
An initial unnecessary `ssh liekkas` attempt failed host-key verification;
local `hostname` had already returned `liekkas`, so all actual work used that
same local host and the verified experiment root. No alternative workspace,
history, library, other arm, evaluator, generator, holdout, network API,
installation, or paid service was used.

Inference checks a wall-clock deadline before each LP. Solver time limits are
capped at .20 s, with reserved overhead. Atomic calls exceeding .25 s are
recorded in state. Single-thread environment variables are set in test commands
and by the enclosing runner. `prepare` reports wall and process CPU seconds,
LP calls, splits, numerical/solver failures, and fallback reasons.

### Completed development and freeze

Construction started at approximately 08:08 UTC, confirmed by the coordinator;
the first timestamped checkpoint was 08:08:27 UTC. Implementation froze at
**2026-10-13 08:17:50 UTC**, under ten minutes after assignment and inside the
20-minute cap. Frozen `method.py` SHA256:
`4058ee5464fe183629935d0982a00a3653c584e3c4004336717d3d7190cb054e`.

There was one initial construction and one implementation revision: explicitly
passing HiGHS `threads=1` after the first development run. The mathematical
representation, thresholds, search, decoding and ablation did not change.
No additional development-driven revision is reserved or proposed.

After initial implementation and synthetic testing, the allowed `dev.json`
packet was read: its schema, six public development observations, six truth
rows and 42 incumbent tasks. The initial inspection printed all six truths,
one observation's calibration/appearance values, and the first five tasks;
the development scorer subsequently used every development observation and
task. Only the scorer accesses truth or task identities. The inference module
imports neither the scorer nor the development packet. No other task or
experiment files were read.

Commands were run with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
and `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B`:

| Run, in execution order | Script under this arm | Measured process CPU s |
| --- | --- | ---: |
| Initial geometry/synthetic tests | `test_method.py` | 0.463655854 |
| Initial full+ablation, 6 s and 24 s | `test_dev.py` | 1.525287062 |
| Revised geometry/synthetic/edge tests | `test_method.py` | 0.536860113 |
| Revised full+ablation, 6 s and 24 s | `test_dev.py` | 1.462143689 |
| Total measured test regions | | **3.987946718** |

These process-time regions exclude module imports and the brief schema
inspection, whose total was not instrumented. Conservatively charging ten CPU
seconds for all development execution remains far below the 120-second cap.
No process ran for anything approaching the remaining budget.

Geometry tests sampled 150 random depth/camera boxes and all three branches
(450 comparisons against the public renderer); maximum numerical enclosure
violation was 2.22e-16. Synthetic inference covered each branch with deliberately
different photo and appearance biases. Zero-budget inference retained the
whole domain. An all-white observation retained [540,660], verifying the
explicit non-identifiability example.

Final development scores were identical at both time budgets and for the
frozen material-coupling ablation:

| Metric | Identity | Constructed representation |
| --- | ---: | ---: |
| MSE, 42 tasks | 20.9250315967 | 5.1604874170 |
| MAE, 42 tasks | 3.7307238095 | 1.7818905042 |
| Improved / unchanged / worsened | — | 25 / 17 / 0 |
| Correct-incumbent damage, 6 cases | — | 0 |

The six retained-hull widths were 3.305813, 2.186988, 4.152045, 4.677533,
4.535078, and 120 mm. Every development truth lay inside its returned hull.
For the final full method, total preparation CPU/wall times over six
observations were 0.365415/0.366085 s (6 s cap) and 0.369575/0.369580 s
(24 s cap). The largest individual wall time was 0.096626 s. There were no
solver failures, empty-enclosure fallbacks, or atomic steps above .25 s.
The unchanged ablation outcome supplies no evidence that cross-view material
coupling matters on these six development observations. Development scores
are not confirmation scores and do not establish a general discovery claim.
