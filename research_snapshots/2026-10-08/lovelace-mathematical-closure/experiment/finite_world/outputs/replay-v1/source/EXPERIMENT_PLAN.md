# Finite-world certification · frozen protocol v1

Freeze before confirmation generation. Scope: stage 1 of ../PROBLEM_SPEC.zh.md.

## Contract

Use Python standard-library Fraction for geometry, piecewise-constant materials,
exact pixel-area integration, bounded l-infinity observation noise, feasibility
and squared reference-ray error gain. Reference camera origin is (0,0,0),
central unit ray is +Z, so first-hit camera-Z equals ray distance in this model.
Three known parallel pinhole cameras view the same opaque surface world.

The solver receives the complete declared finite family, its model predictions
and each candidate's target depth; it never receives the selected truth id or
evaluation result. All candidate predictions are compiled before selecting truth
and noise. Model target labels are derived for ALL hypotheses, not leaked from
this trial's actual world. Initial point and allowed action set are separately
fixed. KEEP is always admissible. No surface deletion changes the denominator.

Use exact enumerated feasible targets; move only for strictly positive worst
gain. Empty feasible set means INCOMPATIBLE + KEEP. No probabilistic assertion,
continuous-world proof, arbitrary-mesh guarantee, or real-map deployment claim.
Model errors not included in the budget are deliberate contract violations.

## Families and controls, fixed before seeing results

- Opaque rectangular foreground with fixed world checker texture and contrasting
  background. Finite 9-depth grid 540..660 mm in steps of 15 mm. Mixed pixels
  and parallax supply observation differences.
- Two opaque surfaces: unknown near foreground and fixed farther layer. Same
  first-hit target definition, exact occlusion, shared material across views.
- Identical foreground/background colors: multiple depths, same observations.
  This is an information-negative control, not an optimization failure.
- Finite family assembled from a local LOVELACE geometry patch if the actual
  mesh is readable without an environment install. State the patch abstraction,
  mesh identity/hash and all material/camera limitations; otherwise keep this
  as a documented subsequent stage rather than inventing mesh results.
- Out-of-family depth and underestimated/violated noise budget controls. Empty
  feasibility should abstain; nonempty but truth-excluding sets can miscertify.
  Never mix these controls into the in-contract zero-damage claim.

Confirmation uses deterministic independent noise seeds 1000..1011 after
implementation self-checks. Bounded per-component noise takes values from
{-epsilon,-epsilon/2,0,epsilon/2,epsilon}, epsilon=1/100. Incumbents are truth,
truth-60 mm, truth+60 mm. They are three states of one scene, not independent
scenes. Do not tune epsilon, material or cameras to erase confirmation failures.

Baselines: KEEP and same-family minimum l-infinity residual with KEEP ties.
The 9-depth grid is newly specified here; it is NOT a reproduction of the
existing map-denoise full9 model, scoring or published results.

## Acceptance

| Gate | Pass condition | Evidence |
|---|---|---|
| Exactness | No floats in prediction/feasibility/gain; hand-computable renderer and certificate cases pass | Independent unit tests |
| Coverage | Every in-contract trial contains its actual world in the feasible set | Exhaustive per-trial independent replay |
| Safety | Every approved action improves the actual target loss by at least its exact certificate | Replay every trial |
| Preservation | No exactly correct in-contract incumbent moves | Per-state counts |
| Nontrivial repair | Informative cases include offset repairs; all KEEP is insufficient | Recovery/KEEP counts |
| Ambiguity | Same-observation worlds force KEEP for exactly-correct compatible incumbents | Negative-control tests and trials |
| Incompatibility | Empty feasible set retains incumbent and reports cause | Dedicated control |
| Provenance | Plan, source hashes, inputs, predictions, outputs and checker together | JSON manifest and replay |

Failures stay in the report. Counterexamples to missing assumptions are expected
results, not scientific safety successes. A passing finite model does not show
that true real photographs fall inside that model. No Lean proof is promised
in this first executable stage.
