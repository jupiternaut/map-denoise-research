Host verified as `liekkas`; requested source files read directly. I did not read private data or formal outputs, edit files, or run experiments. Design is fresh same-family/provisional.

1. Freeze the population and comparison.

Use all 36 public observations, all 324 public tasks and both methods (`calibrated`, `joint`), retaining η∈{0,1/10}, δ=1/10, original photos, branches, pose radius, calibration, noise/clipping rules, material/bias bounds, and existing incumbent depths.

For every original retained leaf containing a task’s exact incumbent `z`, compare:

- Interval: frozen necessary rows from `geometry_bounds(branch, radius, lo, hi, baseline)`, with that leaf’s saved bounds.
- Point: identical construction except depth arguments become `(z,z)`, with the same saved bounds.

No additional propagation, splitting, material-range optimization, replacement candidate, or decision-policy change. Reuse frozen interval proofs only after independent reconstruction and verification; never treat an interval witness’s failure at `z` as point infeasibility.

2. Denominators and aggregation.

| Unit | Fixed denominator |
|---|---:|
| Observation × method cases | 72 |
| Original interval leaves | 5,103, as registered in A5 |
| Task × method diagnoses | 648 |
| Task × method × branch records | 1,944 |
| Task diagnoses per η per method | 162 |

For same-leaf comparisons, precompute the exact public incidence set  
`S={(task,method,branch,leaf): leaf is originally retained and lo≤incumbent≤hi}`. Its denominator `K=|S|` is input-derived and frozen before solving; I cannot give its numeric value without opening the formal outers. Preserve every one of the 648 task-method and 1,944 branch records, including no-containing-leaf cases. Do not report only successful solves or only hull-contained tasks.

At shared endpoints include both closed leaves. Aggregate conservatively: any exact point witness gives `FEASIBLE_WITNESS`; all eligible alternatives certified infeasible gives `CERTIFIED_INFEASIBLE`; otherwise `UNKNOWN`. Apply this across leaves and then branches. Never combine nuisance coordinates from different witnesses. An absent retained leaf is a separate `EXCLUDED_BY_FROZEN_TREE` provenance, requiring the old exclusion chain to establish incompatibility; hull containment alone is insufficient.

Report the interval→point three-state transition table on all `K` incidences, and task-level tables over all 648. Separate new single-row contradictions from contradictions requiring combined rows. Report timing, LP calls, UNKNOWN reasons, and original retained-tolerance versus budget-leaf provenance.

3. Same-leaf versus global bounds.

The primary paired comparison should use identical same-leaf bounds. A useful secondary control solves point rows using the original global material box and frozen calibration intervals: nine material variables for calibrated, plus three bias variables for joint.

There is a stronger invariant here: because darkness is nonnegative and point area bounds nest inside interval area bounds, point rows imply the containing interval rows. Sound leaf propagation therefore only adds consequences of point rows plus the global box. Consequently,
`PointRows(z) ∩ GlobalBounds = PointRows(z) ∩ LeafBounds`
for each containing leaf, assuming verified nesting and propagation.

Thus global/same-leaf disagreement in exact feasibility is an implementation alarm, not scientific evidence. UNKNOWN differences may reflect numerical conditioning. Global point records can cover all 1,944 branch cases; the primary same-leaf calculation costs `K` point checks. Register the secondary control beforehand or omit it for budget reasons, rather than choosing after results.

4. Required adversarial fixtures.

- Interval relaxation feasible while a specified contained point is infeasible; distinguish this from merely invalidating the particular interval witness.
- Exact point feasible and contained in both endpoint-sharing leaves; reordered leaves/branches must not alter aggregation.
- One branch infeasible and another feasible; one infeasible and another UNKNOWN; all branches infeasible; empty retained union and a gap inside the convex hull.
- Contradiction needing multiple rows although every row intersects the box; signed material-difference constraints in both directions; zero coefficients and non-strict zero-margin certificates.
- Rational boundary witnesses, degenerate variable bounds, active-face reconstruction, malformed/tampered certificates, and wrong observation/method/branch/leaf associations.
- LP timeout, unsuccessful status, NaN, rationalization failure: all remain UNKNOWN.
- Point/interval area nesting and point-witness satisfaction of all containing interval rows/bounds.
- Calibrated versus joint dimensions, common per-view bias, unchanged 18 difference rows, clipped observations at 0/1 and noise limits.
- `unresolved_budget` leaves retain original initial bounds and introduce no hidden propagation.
- Positive synthetic complete forward witness, plus a relaxed witness incompatible with any tested common pose/bias.

5. Forward-image validation is useful only as a secondary sufficient-witness check.

`geometry_bounds(...,z,z,...)` still encloses camera-pose uncertainty. A joint LP witness contains nine darkness values and three `t` values, but no camera offsets. A calibrated witness additionally lacks a common per-view bias. Therefore neither can simply be sent through the renderer as a complete physical explanation.

A preregistered public completion—such as nominal allowed camera offsets—can test whether the witness extends to a full explanation. Use per-view colors `1−darkness`, bias `β−t`, bounded per-channel noise, and the original single final clip. For calibrated witnesses, derive/check one common bias per view across every pixel. Verify everything exactly. Avoid the generator wrapper that accepts private `fact` fields.

A successful complete witness establishes existence under the frozen forward model at that public depth. Failure establishes only failure of that attempted completion, never point infeasibility. Do not launch a new pose search or alter incumbent candidates in this stage. At zero pose radius, joint point rows may already permit direct completion, but that equivalence needs its own fixture.

6. What the diagnostic establishes.

New point exclusions would show that interval-depth relaxation concealed inconsistency at specific public incumbent depths. Exact point witnesses show only necessary feasibility unless independently completed in the forward model. No point exclusions with many UNKNOWNs is inconclusive; no exclusions with verified witnesses limits the benefit of this specific exact-depth tightening.

It cannot establish a replacement depth, recovery improvement, deployment readiness, identifiability, or unchanged true-parameter coverage. Keep hybrid actions frozen. The historical main-40 and other truth-based strata may be reported only by a separately authorized post-freeze evaluator; they must not select this diagnostic’s inputs or denominators.

