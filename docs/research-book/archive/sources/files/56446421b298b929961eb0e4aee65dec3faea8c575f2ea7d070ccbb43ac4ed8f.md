# Shared local fields with explicit competing surface hypotheses

Locked before new replay results, liekkas,2026-09-26. CPU only. The question is
whether shared local surface geometry and assignment can improve observation-only
reconstruction beyond pointwise photometric choice and the frozen recovery route.

## New mathematical object

Use reference-camera UV tiles32px with16px halo. Each tile has one or two shared
quadratic inverse-depth functions. A surface is a local zero set of
`f_k(x)=zc^2*(1/z(x)-1/zc)-B(u,v) beta_k`, with normalized local quadratic basis.
The tile domain is explicit. This is a local view-conditioned implicit field,
not a globally watertight SDF. Slopes/curvature are fitted jointly with soft
candidate-to-layer assignments. Layer numbers have meaning only within a tile.
Outputs stay on their reference rays; there is no tangential redistribution.

## Shared observation pool and fitting

At every current input row, directly score9 ray offsets:0,A,B,A/2,B/2,-6,-3,3,6mm.
Use input PCA-k24 normals,7x7 tangent patches and the4 frozen reserved views.
A/B offsets are their actual archived values. All9 positions share the same
valid-view set, with at least2 views; otherwise mark unknown and retain input.
These images are now construction inputs, not independent verification.
No method reads injection labels, evaluation laser or the unperturbed parent.

Photo cost is1-meanZNCC. K1/K2 fitting uses temperature.15, geometric scale.75mm
in centered/scaled inverse depth,12 EM iterations; weak ridge protects quadratic
coefficients. K2 uses3 deterministic starts, not balanced layer counts. Insufficient
support (<12 reliable points) falls back to the current input. Implementation's
exact ridge/init/tie details are recorded in MODEL.md before full inference.
This construction does not assert a new EM principle or global optimum.

Field predictions are converted back to ray offsets, capped at±6mm relative to
the current input. Re-score all3 new surface coordinates (K1,K2layer0,K2layer1)
and identity with identical input normals and common valid source views. Using
fixed normals for this photometric comparison isolates predicted positions; the
field derivatives are not yet used to warp patches. Missing evidence retains input.

## Actual arms (all frozen before reference access)

1. identity;
2. previous `base_shallow__recovery` output;
3. point_wta: choose the9 shared initial positions independently;
4. single_field: choose KEEP or the shared K1 prediction;
5. multi_field: choose KEEP or either shared K2 prediction;
6. multi_field_visibility: same K2 positions, reweight common-view evidence by
   candidate visibility against the rendered multi_field output;
7. multi_field_graph: same visibility unary and candidates, add local graph
   agreement within tiles,8 synchronous label updates.

Every observation-only choice uses photo cost + .005*ray_displacement_mm^2.
This is a fixed displacement preference, not a post-hoc safety gate. Ties KEEP.
Unknown-photo rows remain KEEP. Field K1/K2 may use original candidate scores for
fitting, but output positions are scored afresh.

Visibility uses source-camera Z-buffer and local3x3 inverse-depth support from
the CURRENT predicted cloud. Weight exp(-max(depth_gap,0)/.75mm); unknown support
has weight1 and is logged. It is a self-rendered occlusion proxy, not measured
depth or proof of surface ownership. The mean-photo comparison uses the identical
base valid source set. Graph edges are6-neighbor UV relations within each tile,
weighted by UV distance and input reference-image intensity difference (scale.1).
Potts coefficient.04, no cross-tile label coupling. Graph updates need not globally
minimize energy; record initial/final energy and don't silently call them exact.

## Evaluation

Same60 cases on scans55/65/69,4 frozen photoROI each,5 conditions. All known/exposed.
All actual outputs and protocol/source hashes sealed before evaluator opens GT.
Use unchanged fixed native-row support, ROI+ObsMask laser crop,0.8mm voxelization.
Primary sourceMSE; ROI equally then scene equally. Secondary sourceMAE,p95,
>.1mm harm/benefit, supported motion, reference-to-output MAE and1mmFscore.
Reverse metrics are local footprint diagnostics, not official complete-scene scores.

Oracle controls: old discretep/A/B; old continuousray interval; oracle_all_layers
over initial9 positions,new3field predictions,A/B and actual outputs; finally
oracle_expanded includes old continuous oracle as well. These use reference only
in evaluation. Expanded oracle must not worsen; sourceMSE alone does not establish
layer identity, topology or full reference coverage. Do not select a method from
these references. Primary new actual arm is multi_field; visibility/graph are
separate mechanism ablations, not a condition-wise best composite.

## Tests and completion

Mechanism tests: known perspective tilted plane (affine inverse depth), two layers,
curved patch, missing evidence, row permutation; photo common support and a
synthetic front/back visibility example; graph energy/label and boundary contracts.
Independent checks: oldmetric equality, new PLY row identity/finite coordinates,
no movement from unknown evidence, oracle set inclusion and NN spot checks.
Deliver actual PLY, curve/field/assignment diagnostics, all metric rows, focused
tests, cost accounting, scientific figure, report and checkpoint. No default update
based only on this replay, and no further tuning once replay is read.
# Integration details fixed before outcome access

The local graph uses an 8 px spatial Gaussian and 0.1 grayscale Gaussian.
Eight synchronous Potts sweeps can oscillate, so return the lowest-energy visited
state including the initial unary solution. This is not a global optimizer.
Every field arm reuses the same four-candidate KEEP/K1/K2a/K2b source-view
intersection. Visibility also computes this intersection before selecting K2.
