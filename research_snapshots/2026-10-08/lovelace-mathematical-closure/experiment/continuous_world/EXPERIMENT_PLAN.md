# Stage 2 · continuous depth and bounded camera positions

Protocol version 1. Freeze source hashes and this protocol before generating
confirmation inputs. Stage 1 sources and evidence remain historical inputs.

## Mathematical contract

Depth is EVERY real z in [540,660] mm, not a finite hypothesis grid. World family:
an isolated opaque, two-sided LOVELACE M4 seed triangle (same extracted rational
coordinates as stage 1), or an explicitly separate axis-aligned black rectangle
baseline. One fixed appearance and constant white background. Known intrinsics,
parallel +Z cameras, pixel footprint uniform normalized-image area, no shadows
or free per-candidate material fitting. The reference camera center is fixed at
0 to fix the gauge; left/right nominal x centers are -70/+70 mm, and each has
an independently allowed continuous position error in [-eta,+eta]. This does
not model camera rotation, unknown scale, deformation or arbitrary occlusion.

Prediction intervals must contain EVERY model output in the box, including all
allowed camera errors. For the triangle use rational projected-vertex interval
bounds, convex Hausdorff bounds and certified nominal polygon erosion/dilation
or a proved symmetric-difference area bound. Pixel bounds must cover the entire
continuous box; endpoint or midpoint samples alone are not a certificate.

Discard a box only if at least one prediction-channel enclosure is strictly
disjoint from that observed channel's closed +/-epsilon noise interval.
Otherwise split or retain it. Retained boxes form an outer approximation of
the feasible target set; unresolved boxes must be retained. A search budget is
a termination mechanism, not permission to silently drop unsearched worlds.

Take the hull [L,U] of retained target intervals. For incumbent a and chosen
action b, use min{(a-L)^2-(b-L)^2,(a-U)^2-(b-U)^2} as exact worst-gain lower bound.
Move only for a strictly positive bound. KEEP is always permitted. Empty outer
set reports INCOMPATIBLE + KEEP. Truth and actual camera offsets enter the
evaluator, not the solver. Convex-hull midpoint can be the proposed action;
there is no claim that moving the whole surface or preserving global topology
has been certified.

## Predeclared confirmation

Families: LOVELACE single triangle and opaque rectangle control. Image grid:
8x6, u in [-1/5,1/5], v in [-2/25,2/25], RGB, three cameras.
Camera radius eta: 0, 1/5, 1 mm. Photometric l-infinity epsilon: 1/100.
True depths: 3921/7, 4205/7, 4477/7 mm; all inside the domain and non-dyadic,
so none lie on the binary interval partition boundaries or stage-1 depth grid.
Seeds 2000..2003; actual left/right camera offsets independently drawn from
{-eta,-eta/2,0,eta/2,eta}. Photometric errors independently drawn from
{-epsilon,-epsilon/2,0,epsilon/2,epsilon}. The continuous contract covers MORE
than these finite confirmation samples. Measurements are additive/unclipped.
Incumbents: exactly correct, truth-60 mm, truth+60 mm. Three states share one
observation; count them as decisions, not independent scenes.

Total: 2 families x 3 radius conditions x 3 true depths x 4 seeds = 72 scene
configurations, 216 incumbent decisions. These are parameter/noise conditions,
not 72 independent physical objects.

Depth bisection tolerance 15/32 mm, at most 511 evaluated boxes per query.
Root [540,660] is included in every search. Exclusion order and split midpoint
are deterministic. Check budget exhaustion in dedicated tests and retain all
unprocessed boxes. Treat conservative interval width and KEEP as results,
without changing tolerance/radii based on confirmation recovery rates.

Baselines: KEEP and nominal-camera minimum residual on a 1mm depth grid plus
the incumbent (KEEP ties). Nominal residual scoring is not a certified global
continuous optimization or a reproduction of published map-denoise full9.

## Acceptance and evidence

1. Hand-computable interval and gain tests, including interior feasible truth
   when endpoint predictions fail, exact epsilon boundary, and unfinished work.
2. Per-box trace proves complete root partition and every exclusion witness;
   independent replay verifies enclosures and retained hulls.
3. Every in-contract actual depth belongs to the outer set; each MOVE has exact
   positive lower gain and actual gain no smaller. Correct points never move.
4. Report offsets repaired, MAE, retained interval widths, KEEP rates and costs
   at all three camera-error radii. Do not declare success from all KEEP.
5. Algorithm source locks, bundled geometry, generator parameters, inputs,
   traces and outputs are archived; rerun a source snapshot and compare data.
   The independently authored checker has its own hash receipt before it
   verifies evidence; it is excluded from the algorithm lock because it does
   not influence confirmation generation, search or decisions.
6. Model/photometric-budget violation controls remain separate. They must not
   be used to claim universal zero damage.

No real-map deployment, full-character reconstruction, arbitrary camera-error
coverage or Lean compilation claim is in this stage.
